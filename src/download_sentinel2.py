"""
Download Sentinel-2 L2A imagery for the project's AOIs from the Copernicus Data Space Ecosystem.

WHY YOU RUN THIS YOURSELF:
This project's sandbox cannot reach Copernicus/ESA servers (network is locked down), so
this script has to run on your own machine, in Google Colab, or in Kaggle notebooks —
anywhere with open internet access.

SETUP (one-time):
1. Create a free account: https://dataspace.copernicus.eu
2. pip install sentinelhub  (or use the OData API directly, shown below)
3. Get your OAuth client credentials from the Copernicus dashboard
   (User Settings -> OAuth clients)
4. Save them as environment variables:
       export CDSE_CLIENT_ID="your_client_id"
       export CDSE_CLIENT_SECRET="your_client_secret"

USAGE:
    python src/download_sentinel2.py --aoi nile_delta_mansoura

This uses the sentinelhub-py library's Process API to fetch a true-color + NIR
composite clipped exactly to your AOI bbox, which is much simpler than downloading
full ~800MB SAFE tiles and cropping them yourself.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from aoi_config import AOIS, DATE_RANGE, MAX_CLOUD_COVER

try:
    from sentinelhub import (
        SHConfig,
        BBox,
        CRS,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        bbox_to_dimensions,
    )
except ImportError:
    raise SystemExit(
        "Install the client first:\n  pip install sentinelhub\n"
        "Docs: https://sentinelhub-py.readthedocs.io/"
    )

# Evalscript requesting all bands we need (10m + 20m, resampled to a common 10m grid)
EVALSCRIPT_ALL_BANDS = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","dataMask"],
      units: "REFLECTANCE"
    }],
    output: { bands: 11, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
          sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12, sample.dataMask];
}
"""


def get_config():
    config = SHConfig()
    config.sh_client_id = os.environ.get("CDSE_CLIENT_ID")
    config.sh_client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    if not config.sh_client_id or not config.sh_client_secret:
        raise SystemExit(
            "Missing credentials. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET env vars."
        )
    print(f"Using base URL: {config.sh_base_url}")
    print(f"Using token URL: {config.sh_token_url}")
    print(f"Client ID starts with: {config.sh_client_id[:8]}...")

    # Sanity check: fetch a token directly against CDSE before building the request,
    # so an auth problem shows up here with a clear message instead of during get_data().
    import requests
    resp = requests.post(
        config.sh_token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": config.sh_client_id,
            "client_secret": config.sh_client_secret,
        },
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"Token request to CDSE failed ({resp.status_code}): {resp.text}\n"
            "Double check the OAuth client was created in the Copernicus Data Space "
            "'Sentinel Hub' dashboard (shapps.dataspace.copernicus.eu), not a different "
            "commercial Sentinel Hub account, and that grant type is 'Client Credentials'."
        )
    print("Token fetch OK — credentials are valid for Copernicus Data Space.")
    return config


def download_aoi(aoi_name: str, out_dir: str, resolution: int = 10):
    if aoi_name not in AOIS:
        raise ValueError(f"Unknown AOI '{aoi_name}'. Choices: {list(AOIS)}")

    config = get_config()
    bbox_coords = AOIS[aoi_name]["bbox"]
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"AOI '{aoi_name}' -> image size {size} px at {resolution}m/px")

    os.makedirs(out_dir, exist_ok=True)

    request = SentinelHubRequest(
        data_folder=out_dir,
        evalscript=EVALSCRIPT_ALL_BANDS,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=DATE_RANGE,
                maxcc=MAX_CLOUD_COVER / 100,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    data = request.get_data(save_data=True)
    print(f"Saved {aoi_name} to {out_dir} (folder contains a request hash subdir with response.tiff)")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", required=True, choices=list(AOIS.keys()))
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--resolution", type=int, default=10)
    args = parser.parse_args()
    download_aoi(args.aoi, args.out, args.resolution)
