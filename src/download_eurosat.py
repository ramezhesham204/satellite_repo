"""
Download the EuroSAT RGB dataset (Helber et al., 2018/2019) from Zenodo.

Usage:
    python src/download_eurosat.py            # RGB version (~95 MB)
    python src/download_eurosat.py --ms        # multispectral, all 13 bands (~2.1 GB)
"""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

URLS = {
    "rgb": "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "ms": "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1",
}


def download(kind: str):
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / f"EuroSAT_{kind.upper()}.zip"
    print(f"Downloading {URLS[kind]} -> {zip_path}")
    urlretrieve(URLS[kind], zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    print(f"Done. Data at {RAW / ('EuroSAT_' + kind.upper())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ms", action="store_true", help="download multispectral version instead of RGB")
    args = p.parse_args()
    download("ms" if args.ms else "rgb")
