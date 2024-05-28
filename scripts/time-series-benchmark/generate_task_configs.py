"""
Taken from https://github.com/shchur/automlbenchmark/blob/autogluon-timeseries-automl23/autogluon_timeseries_automl23/generate_task_configs.py
"""

import argparse
import pandas as pd
import yaml
from typing import List
from pathlib import Path


amlb_dir = Path(__file__).parent.parent
default_dataset_dir = str(amlb_dir / "datasets")
default_benchmark_dir = str(Path.home() / ".config" / "automlbenchmark" / "benchmarks")


name_to_prediction_length = {
    "car_parts": 12,
    "cif_2016": 12,
    "covid_deaths": 30,
    "electricity_hourly": 48,
    "electricity_weekly": 8,
    "fred_md": 12,
    "hospital": 12,
    "kdd_cup_2018": 48,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    "m1_yearly": 6,
    "m3_monthly": 18,
    "m3_other": 8,
    "m3_quarterly": 8,
    "m3_yearly": 6,
    "m4_daily": 14,
    "m4_hourly": 48,
    "m4_weekly": 13,
    "m4_yearly": 6,
    "m4_monthly": 12,
    "m4_quarterly": 8,
    "nn5_daily": 56,
    "nn5_weekly": 8,
    "pedestrian_counts": 48,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
    "tourism_yearly": 4,
    "vehicle_trips": 7,
    "web_traffic_weekly": 8,
}

freq_to_seasonality = {"H": 24, "D": 7, "W": 1, "M": 12, "Q": 4, "A": 1}


def generate_task(name: str, base_dir: str, metrics: List[str]) -> dict:
    if base_dir.startswith("s3://"):
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"
        path_to_data = f"{base_dir}{name}/data.csv"
    else:
        path_to_data = Path(base_dir) / name / "data.csv"
    try:
        header = pd.read_csv(path_to_data, nrows=4, parse_dates=["timestamp"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {name} not found in {path_to_data}")

    freq = pd.infer_freq(header["timestamp"]).split("-")[0]
    task = {
        "name": name,
        "dataset": {
            "path": str(path_to_data),
            "type": "timeseries",
            "target": "target",
            "forecast_horizon_in_steps": name_to_prediction_length[name],
            "freq": freq,
            "seasonality": freq_to_seasonality[freq],
        },
        "metric": metrics,
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "folds": 1,
    }
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default=default_dataset_dir,
        help="Local or S3 path to folder where the datasets are stored.",
    )
    parser.add_argument(
        "-b",
        "--benchmark_dir",
        type=str,
        default=default_benchmark_dir,
        help="Local path to a folder where to save the YAML config files.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    benchmark_dir = Path(args.benchmark_dir)
    benchmark_dir.mkdir(exist_ok=True, parents=True)

    print(f"Saving task definitions to {benchmark_dir}")
    point_forecast_tasks = [
        generate_task(name, dataset_dir, metrics=["mase", "wql"])
        for name in name_to_prediction_length
    ]
    quantile_forecast_tasks = [
        generate_task(name, dataset_dir, metrics=["wql", "mase"])
        for name in name_to_prediction_length
    ]

    if dataset_dir.startswith("s3://"):
        suffix = "_aws"
    else:
        suffix = ""

    with open(benchmark_dir / f"point_forecast{suffix}.yaml", "w") as out_file:
        yaml.safe_dump(
            point_forecast_tasks,
            out_file,
            width=10000,
            sort_keys=False,
            default_flow_style=False,
        )
        out_file.writelines("\n")
    print(f"Point forecast tasks saved to {out_file.name}")

    with open(benchmark_dir / f"quantile_forecast{suffix}.yaml", "w") as out_file:
        yaml.safe_dump(
            quantile_forecast_tasks,
            out_file,
            width=10000,
            sort_keys=False,
            default_flow_style=False,
        )
        out_file.writelines("\n")
    print(f"Quantile forecast tasks saved to {out_file.name}")
