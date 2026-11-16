# Active Tigger: image use case data preparation

This repository prepares the data for the **image use case** of [Active Tigger](https://github.com/activetigger/activetigger). The use case is a computer vision tutorial for social scientists, built on photos of household toilets from the [Dollar Street dataset](https://openreview.net/forum?id=qnfYsave0U4). Dollar Street has photos of everyday objects in homes across 63 countries, along with each household's monthly income.

The tutorial has two Active Tigger projects:

1. **Toilet detection**: classify images as `toilet` or `other`.
2. **Income from toilets**: predict `low income` or `high income` from toilet images only. This project can also be used to hand-code the type of sanitation facility.

## Repository layout

```
dataset_dollarstreet/          original Dollar Street dataset (not modified)
src/prepare_activetigger.py    prepares the data for Active Tigger
usecase_data/activetigger/     output, ready to upload to Active Tigger
```

## Requirements

- The Dollar Street dataset in `dataset_dollarstreet/`. The script needs `images_v2.csv` and the `assets/` folder.
- [uv](https://docs.astral.sh/uv/). It installs the only dependency, Pillow, automatically.

## Usage

```bash
uv run src/prepare_activetigger.py
```

It takes about 45 seconds. Options:

| Option | Default | Meaning |
|---|---|---|
| `--n-images` | 12000 | number of images in the toilet detection project |
| `--max-side` | 512 | resize images so the longest side is this many pixels |
| `--quality` | 85 | JPEG quality |
| `--income-classes` | 2 | `2` = low/high income, split at the median household income; `4` = low / lower-middle / upper-middle / high income, split at the quartiles |
| `--test-frac` | 0.2 | share of households in the test set of the income project |
| `--no-video` | off | leave out still frames taken from Dollar Street videos |
| `--seed` | 42 | random seed for the image sample and the split |

Resized images are cached in `usecase_data/activetigger/_resized_<max-side>/`. You can delete this folder; the script rebuilds it.

## Output

Every `.zip` is in Active Tigger's image project format. It holds JPEGs named `<id>.jpg` plus a `metadata.csv` whose first column is `id`. The other metadata columns are `place` (the household), `country` and `region`. Labels are left out of the zips so that annotators aren't primed by the answer. Each zip comes with a matching `*_labels.csv` that holds the Dollar Street labels (`toilet`, `income_class`, `income`, `topics`, ...).

### `1_toilet_detection/`

| File | Content |
|---|---|
| `all.zip` | 12,000 images: all 747 toilet images plus 11,253 randomly chosen other images (about 440 MB) |
| `all_labels.csv` | labels `toilet` and `other`, taken from the Dollar Street topics |

- **What counts as a toilet:** an image is a toilet if its topics include `Toilet`, `Bathroom/Toilet` or `Using toilet`.
- **What's left out:** images tagged only `Toilet paper`, `Cleaning after toilet` or `Bathroom privacy` are in neither class, because many of them don't clearly show a toilet.
- **Test set:** set the test size in Active Tigger when you create the project. Its split is random, one image at a time.

### `2_toilet_income/`

| File | Content |
|---|---|
| `train.zip` / `train_labels.csv` | 584 toilet images |
| `test.zip` / `test_labels.csv` | 163 toilet images from households that aren't in the train set |
| `all.zip` / `all_labels.csv` | the same 747 images in one archive |

- **Income classes:** households are split at the median household income, **$494 a month**. `low income` is below it and `high income` is at or above it. The median is computed over the 404 households, not over images.
- **Why the split is by household:** most households have several toilet photos, often of the same toilet. If Active Tigger splits these images at random, photos of the same toilet end up in both train and test. The model can then recognise the toilet rather than learn what toilets reveal about income, which inflates the test score.

## Loading into Active Tigger

1. **Create a project:** choose the image project type and upload the zip. Set the test size to **0** when you provide your own test set.
2. **Add the test set (optional):** upload `test.zip` together with `test_labels.csv`.
3. **Import labels (optional):** import a `*_labels.csv` into a scheme, choosing `id` as the ID column and `toilet` or `income_class` as the label column. Use this to simulate a project that is already annotated, or to compare students' annotations with the Dollar Street labels.

**Leakage exercise:**
1. Train the same model in a project built from `train.zip` + `test.zip`.
2. Train it again in a project built from `all.zip`, with a random test set of about 160 images.
3. Compare the test scores. The gap measures how much household leakage inflates performance.

## Caveat: what images can and cannot show

In Dollar Street, "toilet" means *the place where the household goes to the toilet*. Some households have no toilet, and their "toilet" photo shows a field, a bush or a yard. Nothing in such an image shows what it is used for: that information comes from the household, not from the pixels, so no image classifier can reliably detect open defecation. The tutorial treats this as a lesson in measurement and construct validity. One way to do that is to hand-code a sanitation scheme in project 2, following the WHO/UNICEF JMP sanitation ladder: flush toilet, pit latrine with slab, unimproved facility, open defecation.
