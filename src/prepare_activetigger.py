# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Prepare Dollar Street data for an Active Tigger image tutorial.

Builds two ready-to-upload Active Tigger image projects in
usecase_data/:

  1_toilet_detection/   ~12k images: every toilet image + a random sample of other images
  2_toilet_income/      only the toilet images, for income prediction / sanitation coding

Zips contain resized JPEGs named <id>.jpg + metadata.csv (first column = id).
Each <name>.zip comes with <name>_labels.csv (Dollar Street labels) that can be
imported into a scheme or used to evaluate annotations.

  1_toilet_detection/all.zip    -> single dataset; set the test size in Active
                                   Tigger (random per-image split)
  2_toilet_income/train.zip     -> project data, split by HOUSEHOLD (`place`)
  2_toilet_income/test.zip      -> add as the image test set (no household in both)
  2_toilet_income/all.zip       -> same images in one zip, to let Active Tigger
                                   split randomly and compare: images of the same
                                   household on both sides inflate the test score

Run:  uv run src/prepare_activetigger.py       (uv installs Pillow automatically)
"""
import argparse
import csv
import random
import zipfile
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent  # project root (scripts live in src/)
SRC = ROOT / "dataset_dollarstreet"
OUT = ROOT / "usecase_data"

TOILET_TOPICS = {"toilet", "bathroom/toilet", "using toilet"}
# Images about toilets but not clearly showing one: left out of the detection
# project so the "other" class is not polluted.
AMBIGUOUS_TOPICS = {"toilet paper", "cleaning after toilet", "bathroom privacy"}

META_COLS = ["id", "place", "country", "region"]  # shown in Active Tigger; no label leakage


def topics_of(row):
    return {t.strip().lower() for t in row["topics"].split(",")}


def resize_one(job):
    src, dst, max_side, quality = job
    if dst.exists():
        return None
    try:
        with Image.open(src) as im:
            im.draft("RGB", (max_side, max_side))  # fast JPEG downscaling on decode
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return None
    except Exception as e:  # corrupt/unreadable file
        return f"{src}: {e}"


INCOME_LABELS = {
    2: ["low income", "high income"],
    4: ["low income", "lower-middle income", "upper-middle income", "high income"],
}


def income_thresholds(rows, k):
    """Cut points splitting households (one income per place) into k equal-size groups."""
    incomes = sorted({r["place"]: float(r["income"]) for r in rows}.values())
    n = len(incomes)
    return [incomes[i * n // k] for i in range(1, k)]


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_zip(path, rows, img_dir):
    meta = path.with_suffix(".metadata.csv")
    write_csv(meta, rows, META_COLS)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for r in rows:
            zf.write(img_dir / f"{r['id']}.jpg", arcname=f"{r['id']}.jpg")
        zf.write(meta, arcname="metadata.csv")
    meta.unlink()
    return path.stat().st_size / 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=12000, help="total images in the detection project (train+test)")
    p.add_argument("--test-frac", type=float, default=0.2, help="share of households put in the test set")
    p.add_argument("--max-side", type=int, default=512, help="resize so the longest side is this many px")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality")
    p.add_argument("--no-video", action="store_true", help="exclude video preview frames")
    p.add_argument("--income-classes", type=int, choices=sorted(INCOME_LABELS), default=2,
                   help="2 = low/high income (split at the household median); "
                        "4 = low / lower-middle / upper-middle / high income (quartiles)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    with open(SRC / "images_v2.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["income"]]
    if args.no_video:
        rows = [r for r in rows if r["type"] == "image"]

    # --- labels -----------------------------------------------------------
    labels = INCOME_LABELS[args.income_classes]
    q = income_thresholds(rows, args.income_classes)
    for r in rows:
        t = topics_of(r)
        r["toilet"] = "toilet" if t & TOILET_TOPICS else "other"
        r["ambiguous"] = bool(t & AMBIGUOUS_TOPICS) and r["toilet"] == "other"
        r["income_class"] = labels[bisect_right(q, float(r["income"]))]
        r["country"], r["region"] = r["country.name"], r["region.id"]

    # --- household split (shared by both projects) -----------------------
    places = sorted({r["place"] for r in rows})
    rng.shuffle(places)
    test_places = set(places[: round(len(places) * args.test_frac)])
    for r in rows:
        r["split"] = "test" if r["place"] in test_places else "train"

    # --- project selections ----------------------------------------------
    positives = [r for r in rows if r["toilet"] == "toilet"]
    negatives = [r for r in rows if r["toilet"] == "other" and not r["ambiguous"]]
    n_neg = max(0, min(len(negatives), args.n_images - len(positives)))
    detection = positives + rng.sample(negatives, n_neg)
    income = positives

    # --- resize (cached) ---------------------------------------------------
    img_dir = OUT / f"_resized_{args.max_side}"
    needed = {r["id"]: r for r in detection}.values()
    jobs = [(SRC / r["imageRelPath"], img_dir / f"{r['id']}.jpg", args.max_side, args.quality) for r in needed]
    print(f"Resizing {len(jobs)} images to {args.max_side}px (cached in {img_dir}) ...")
    errors = []
    with ProcessPoolExecutor() as ex:
        for i, err in enumerate(ex.map(resize_one, jobs, chunksize=32), 1):
            if err:
                errors.append(err)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}")
    if errors:
        print(f"{len(errors)} unreadable images skipped, e.g. {errors[0]}")
    ok = lambda r: (img_dir / f"{r['id']}.jpg").exists()

    # --- write projects ----------------------------------------------------
    projects = {
        "1_toilet_detection": (detection, ["id", "toilet", "topics", "income_class", "income", "country", "place"]),
        "2_toilet_income": (income, ["id", "income_class", "income", "country", "place", "topics"]),
    }
    # detection: one zip, let Active Tigger make the (random, per-image) split.
    # income: household split (train/test) + one zip, to compare and show leakage.
    splits = {"1_toilet_detection": ["all"], "2_toilet_income": ["train", "test", "all"]}
    for name, (sel, label_cols) in projects.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for old in [*d.glob("*.zip"), *d.glob("*_labels.csv")]:  # remove outputs of earlier runs
            old.unlink()
        print(f"\n{name}")
        for split in splits[name]:
            part = [r for r in sel if split in ("all", r["split"]) and ok(r)]
            rng.shuffle(part)
            size = write_zip(d / f"{split}.zip", part, img_dir)
            write_csv(d / f"{split}_labels.csv", part, label_cols)
            n_t = sum(r["toilet"] == "toilet" for r in part)
            qs = {k: sum(r["income_class"] == k for r in part) for k in labels}
            print(f"  {split}: {len(part)} images ({n_t} toilet) | {qs} | {size:.0f} MB")

    cuts = " | ".join(
        f"{lab}: {'>= $' + format(lo, '.0f') if lo is not None else ''}{' ' if lo is not None and hi is not None else ''}{'< $' + format(hi, '.0f') if hi is not None else ''}"
        for lab, lo, hi in zip(labels, [None] + q, q + [None])
    )
    print(f"\nIncome classes ($/month, equal-size groups of households): {cuts}")
    print(f"Test households: {len(test_places)}/{len(places)}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
