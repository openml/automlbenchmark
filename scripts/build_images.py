"""
This script can be used to build all/selected docker/singularity images.
0. Load your virtual environment like you would when using AMLB.
1. Run the script with the required mode:
    python build_images.py -m docker
    python build_images.py -m singularity

2. To delete files forcefully, add the --force flag. This is optional but recommended in case you change something.
    python build_images.py -m singularity --force

3. To specify frameworks, use the -f flag with a comma-separated list:
    python build_images.py -m docker -f autosklearn,flaml,gama --force

4. If no frameworks are specified, the script will act on all available frameworks.

"""

import os
import argparse
import pathlib
from pathlib import Path
from typing import List, Optional


def delete_singularity_files(
    directory: pathlib.Path, frameworks: Optional[List[str]] = None
) -> None:
    """Deletes Singularity-related files from the specified directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "Singularityfile" or file.endswith(".sif"):
                file_path = Path(root) / file

                if frameworks and not any(
                    framework in file_path.name.lower() for framework in frameworks
                ):
                    continue

                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Script to manage Docker and Singularity frameworks."
    )
    parser.add_argument(
        "-m", "--mode", help="Docker or singularity", type=str, required=True
    )
    parser.add_argument(
        "--force", help="Delete singularity/docker files.", action="store_true"
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        help="Comma-separated list of frameworks to act on.",
        type=str,
    )
    return parser.parse_args()


def get_frameworks(framework_dir: Path, user_input: Optional[str]) -> List[str]:
    """Gets the list of frameworks based on user input or directory listing."""
    if user_input:
        frameworks = [framework.lower().strip() for framework in user_input.split(",")]
        print(f"Running for given frameworks - {frameworks}")
    else:
        frameworks = [
            framework.lower()
            for framework in os.listdir(framework_dir)
            if "__" not in framework
        ]
        print(f"Running for all frameworks - {frameworks}")
    return frameworks


def main() -> None:
    args = parse_arguments()
    framework_dir = Path("../frameworks/")
    mode = args.mode.lower().strip()
    frameworks = get_frameworks(framework_dir, args.frameworks)

    if mode == "docker" and args.force:
        print("Deleting frameworks from Docker")
        for framework in frameworks:
            os.system(f"docker rmi $(docker images 'automlbenchmark/{framework}')")

    elif mode == "singularity" and args.force:
        print("Deleting frameworks from Singularity")
        delete_singularity_files(directory=framework_dir, frameworks=frameworks)

    for framework in frameworks:
        print(f"Setting up {framework}")
        os.system(
            f"yes | python ../runbenchmark.py {framework} openml/t/3812 --mode {mode} --setup only"
        )


if __name__ == "__main__":
    main()
