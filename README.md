# Land Type Classification using Sentinel-2 Satellite Images

Digital Egypt Pioneers Initiative, AI & Data Science Track, Round 2, Project 6.

## Structure

```
├── app/                  Flask inference API (Milestone 4)
├── data/
│   ├── raw/               EuroSAT_RGB/ after download, or Sentinel-2 AOI GeoTIFFs
│   ├── interim/            tiles before labeling
│   ├── labels/              tile labels (csv/json)
│   └── processed/            train/val/test arrays
├── docs/                  milestone reports, final presentation
├── models/                checkpoints + classes.json (git-ignored)
├── notebooks/             EDA, feature engineering
├── reports/figures/       exported plots, confusion matrices, training curves
├── src/
│   ├── download_eurosat.py    pulls EuroSAT RGB/MS from Zenodo
│   ├── download_sentinel2.py  pulls raw Sentinel-2 AOIs via Copernicus Data Space
│   ├── aoi_config.py           AOI bounding boxes over Egypt
│   ├── preprocessing.py        NDVI/NDWI, tiling, normalization (Sentinel-2 path)
│   ├── dataset.py               EuroSAT loaders, splits, augmentation
│   ├── model.py                  small CNN + ResNet18 transfer learning
│   └── train.py                   training loop, eval, confusion matrix, curves
└── tests/
```

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` is only needed for the raw Sentinel-2 path (Copernicus OAuth client). Never commit it.

## Two data paths

**EuroSAT (recommended to start)** — ready-labeled, 27,000 images, 10 classes:

```powershell
python src/download_eurosat.py
python src/train.py --model cnn --epochs 30
python src/train.py --model resnet18 --epochs 15 --lr 0.0003
```

**Raw Sentinel-2 over Egypt** — unlabeled, needs manual tile labeling, matches the booklet's
6 classes (agriculture, water, urban, desert, road, trees) exactly:

```powershell
python src/download_sentinel2.py --aoi nile_delta_mansoura
```
AOIs: `nile_delta_mansoura`, `cairo_metro`, `aswan_nile`, `fayoum_oasis`.

## Deployment

```powershell
python app/app.py
# POST an image to http://localhost:5000/predict
```

Set `MODEL_NAME` at the top of `app/app.py` to match whichever checkpoint you trained
(`cnn` or `resnet18`).

## Milestones

| # | Milestone | Where |
|---|-----------|-------|
| 1 | Collection, EDA, preprocessing | `src/download_*.py`, `notebooks/` |
| 2 | Analysis and feature engineering | `notebooks/`, `src/preprocessing.py` |
| 3 | Model development | `src/dataset.py`, `src/model.py`, `src/train.py` |
| 4 | Deployment and monitoring | `app/` |
| 5 | Final report and presentation | `docs/` |

## Team

| Member | Responsibility |
|--------|----------------|
|        |                |
