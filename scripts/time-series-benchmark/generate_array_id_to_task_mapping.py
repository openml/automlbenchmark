from pathlib import Path
import yaml
import argparse

from generate_task_configs import name_to_prediction_length


ID_2_TASK_FILENAME = "id_to_task_mapping.yaml"


def main(output_path: str):
    id_2_task = {}
    for id, task in enumerate(name_to_prediction_length.keys()):
        id_2_task[id] = task

    # Dump dict into a yaml file
    output_file = Path(output_path) / ID_2_TASK_FILENAME
    print(f"Writing ID_2_TASK mapping into {output_file}")
    with open(output_file, "w") as out_file:
        yaml.safe_dump(
            id_2_task,
            out_file
        )

    print(f"Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    main(output_path=args.output_path)
