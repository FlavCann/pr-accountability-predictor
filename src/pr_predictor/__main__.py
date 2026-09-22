"""Enables `python -m pr_predictor`. The `prp` console script is equivalent."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
