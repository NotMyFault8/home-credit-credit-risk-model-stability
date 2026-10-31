"""Generate MODEL_VALIDATION_REPORT.md from a completed run directory."""

import argparse

from home_credit.validation_report import generate_validation_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(generate_validation_report(args.run_root, args.output))


if __name__ == "__main__":
    main()
