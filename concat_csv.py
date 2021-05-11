#!/usr/bin/python

import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description="Small utility to safely concatenate multiple csv files, "
                "especially useful here when concatenating multiple `results.csv` files.",
    epilog="""Usage example:
find ./results -name 'results.csv' -exec python concat_csv.py results.csv {} +
""",
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('target', type=str, help='The target file that will contain the concatenated csv files.')
parser.add_argument('sources',  nargs='*', help='the csv files to be concatenated.')
args = parser.parse_args()

print(f"concatenating following files to `{args.target}`:\n{args.sources}")
(pd.concat([pd.read_csv(file) for file in args.sources], ignore_index=True)
   .drop_duplicates()
   .to_csv(args.target, index=False))

