import argparse
from pathlib import Path
import pandas as pd


def preview_csv(path: Path, n_rows: int = 30) -> None:
    """Print the first *n_rows* lines of a potentially large CSV file.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    n_rows : int, optional
        How many rows to print, by default 30.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        # Read only the first *n_rows* rows to avoid loading the whole file
        df = pd.read_csv(path, nrows=n_rows)
    except pd.errors.EmptyDataError:
        print("The file is empty.")
        return
    except Exception as exc:
        print(f"Failed to read CSV: {exc}")
        return

    # Display without the DataFrame index for cleaner output
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the first N rows of a large CSV file without loading it entirely."
    )
    parser.add_argument(
        "--file",
        default=Path("data") / "test_categorical.csv",
        type=Path,
        help="Path to the CSV file (default: data/test_categorical.csv)",
    )
    parser.add_argument(
        "-n",
        "--rows",
        default=30,
        type=int,
        help="Number of rows to preview (default: 30)",
    )
    args = parser.parse_args()

    preview_csv(args.file, args.rows)


if __name__ == "__main__":
    main()



