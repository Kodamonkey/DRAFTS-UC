from pathlib import Path
from MJD import get_file_metadata  # reuse helper from scripts/MJD.py


def main():
    import argparse
    p = argparse.ArgumentParser(description="Read precise header metadata for .fil/.fits files")
    p.add_argument("--file", required=True, help="Path to the data file (.fil/.fits)")
    args = p.parse_args()
    meta = get_file_metadata(args.file)
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


