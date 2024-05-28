"""
Taken from https://github.com/shchur/automlbenchmark/blob/autogluon-timeseries-automl23/autogluon_timeseries_automl23/download_datasets.py
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Iterable

from joblib import delayed, Parallel
from gluonts.dataset.repository.datasets import get_dataset


amlb_dir = Path(__file__).parent.parent
default_dataset_dir = str(amlb_dir / "datasets")


original_name_to_gluonts_name = {
    "car_parts": "car_parts_without_missing",
    "cif_2016": "cif_2016",
    "covid_deaths": "covid_deaths",
    "electricity_hourly": "electricity_hourly",
    "electricity_weekly": "electricity_weekly",
    "fred_md": "fred_md",
    "hospital": "hospital",
    "kdd_cup_2018": "kdd_cup_2018_without_missing",
    "m1_monthly": "m1_monthly",
    "m1_quarterly": "m1_quarterly",
    "m1_yearly": "m1_yearly",
    "m3_monthly": "m3_monthly",
    "m3_other": "m3_other",
    "m3_quarterly": "m3_quarterly",
    "m3_yearly": "m3_yearly",
    "m4_daily": "m4_daily",
    "m4_hourly": "m4_hourly",
    "m4_weekly": "m4_weekly",
    "m4_yearly": "m4_yearly",
    "m4_monthly": "m4_monthly",
    "m4_quarterly": "m4_quarterly",
    "nn5_daily": "nn5_daily_without_missing",
    "nn5_weekly": "nn5_weekly",
    "pedestrian_counts": "pedestrian_counts",
    "tourism_monthly": "tourism_monthly",
    "tourism_quarterly": "tourism_quarterly",
    "tourism_yearly": "tourism_yearly",
    "vehicle_trips": "vehicle_trips_without_missing",
    "web_traffic_weekly": "kaggle_web_traffic_weekly",
}


def download_dataset(name: str, base_dir: Path) -> None:
    dataset_dir = base_dir / name
    dataset_dir.mkdir(exist_ok=True)
    csv_path = dataset_dir / "data.csv"
    if csv_path.exists():
        print(f"Skipping {name} as it already exists")
    else:
        print(f"Downloading {name}")
        gluonts_name = original_name_to_gluonts_name[name]
        gluonts_dataset = get_dataset(gluonts_name)
        freq = gluonts_dataset.metadata.freq
        if name == "m3_other":
            freq = "A"  # fix incorrect frequency of the M3 other dataset
        dataset_df = gluonts_dataset_to_data_frame(gluonts_dataset.test, freq=freq)
        dataset_df.to_csv(csv_path, index=False)


def gluonts_dataset_to_data_frame(dataset: Iterable, freq: str) -> pd.DataFrame:
    def convert_entry(item_id, entry):
        start = entry["start"].to_timestamp(how="S")
        target = entry["target"]
        timestamps = pd.date_range(start=start, freq=freq, periods=len(target))
        return pd.DataFrame(
            {
                "item_id": [item_id for _ in range(len(target))],
                "timestamp": timestamps,
                "target": target,
            }
        )

    dfs = Parallel(n_jobs=-1)(
        delayed(convert_entry)(item_id, entry) for item_id, entry in enumerate(dataset)
    )
    return pd.concat(dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, default=default_dataset_dir)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving datasets to {dataset_dir}")

    for name in original_name_to_gluonts_name:
        download_dataset(name, dataset_dir)
