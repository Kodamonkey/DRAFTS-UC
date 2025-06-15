"""Entry point for running the Effelsberg FRB detection pipeline."""
from .pipeline import run_pipeline


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
