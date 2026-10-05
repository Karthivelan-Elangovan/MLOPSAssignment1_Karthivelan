from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Official UCI Heart Disease dataset
DATASET_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def download_dataset():
    """Download and extract the UCI Heart Disease dataset."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Heart Disease dataset...")

    with urlopen(DATASET_URL) as response:
        zip_data = response.read()

    print("Download completed.")

    with ZipFile(BytesIO(zip_data)) as zip_file:
        files_to_extract = [
            "processed.cleveland.data",
            "heart-disease.names"
        ]

        for file_name in files_to_extract:
            matching_files = [
                name for name in zip_file.namelist()
                if name.endswith(file_name)
            ]

            if not matching_files:
                print(f"Warning: {file_name} not found in archive.")
                continue

            source_file = matching_files[0]

            with zip_file.open(source_file) as source:
                output_path = RAW_DATA_DIR / file_name

                with open(output_path, "wb") as destination:
                    destination.write(source.read())

            print(f"Saved: {output_path}")


if __name__ == "__main__":
    download_dataset()