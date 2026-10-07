"""Create the fixed train/val/test split of a model zoo.

Run this once per zoo, before anything else:

    uv run data/create_split.py /path/to/zoo

It writes /path/to/zoo/split.json. Every dataset class reads its models from
that file, and the script refuses to overwrite an existing split.
"""

import argparse
import logging

from SANE.datasets.zoo_split import create_split

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("zoo", help="zoo directory, one subdirectory per model")
parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
parser.add_argument("--seed", type=int, default=42, help="seed of the one-time shuffle of the models")
args = parser.parse_args()

create_split(args.zoo, ratios=args.ratios, seed=args.seed)
