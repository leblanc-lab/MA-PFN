"""Command-line entry point for training and benchmarking MA-PFN and PFN."""

from utils import parse_args, run_workflow


if __name__ == "__main__":
    run_workflow(parse_args())
