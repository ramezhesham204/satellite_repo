# Land Type Classification — EuroSAT

Digital Egypt Pioneers Initiative, AI & Data Science Track, Round 2, Project 6.

Uses the EuroSAT RGB dataset (Helber et al.): 27,000 labeled Sentinel-2 patches across
10 land use/land cover classes. No satellite account or raw imagery download needed.

## Structure

```
├── app/app.py              Flask inference API (Milestone 4)
├── data/raw/                EuroSAT_RGB/<ClassName>/*.jpg after download
├── docs/                    milestone reports, final presentation
├── models/                  checkpoints + classes.json (git-ignored)
├── notebooks/01_land_classification.ipynb   EDA, feature engineering, training (Milestones 1-3)
├── reports/figures/         confusion matrices, training curves
├── src/
│   ├── download_eurosat.py   downloads + extracts EuroSAT from Zenodo
│   ├── dataset.py             loaders, stratified split, augmentation
│   ├── model.py                small CNN + ResNet18 transfer learning
│   └── train.py                 training loop, eval, confusion matrix, curves
└── tests/
```

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python src/download_eurosat.py
python src/train.py --model cnn --epochs 30
python src/train.py --model resnet18 --epochs 15 --lr 0.0003
```

Or open `notebooks/01_land_classification.ipynb` and run it top to bottom (same steps,
with EDA and plots inline).

## Deployment

```powershell
python app/app.py
# POST an image to http://localhost:5000/predict
```

Set `MODEL_NAME` at the top of `app/app.py` to match the checkpoint you trained
(`cnn` or `resnet18`).

## Milestones

| # | Milestone | Where |
|---|-----------|-------|
| 1 | Collection, EDA, preprocessing | `src/download_eurosat.py`, `notebooks/` |
| 2 | Analysis and feature engineering | `notebooks/`, `src/dataset.py` |
| 3 | Model development | `src/model.py`, `src/train.py` |
| 4 | Deployment and monitoring | `app/` |
| 5 | Final report and presentation | `docs/` |

## Team

| Member | Responsibility |
|--------|----------------|
|        |                |
