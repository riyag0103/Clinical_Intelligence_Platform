from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Clinical_Project")

CSV_FILES = [
    PROJECT_ROOT / "datasets/metadata/full_metadata.csv",
    PROJECT_ROOT / "datasets/metadata/clean_ct_metadata.csv",
    PROJECT_ROOT / "datasets/metadata/ct_metadata.csv",
    PROJECT_ROOT / "datasets/metadata/balanced/balanced_metadata.csv",
]


def resolve_path(path_string):
    """
    Resolve metadata paths relative to the project root.
    """
    normalized = str(path_string).replace("\\", "/").strip()

    return (PROJECT_ROOT / normalized).resolve()


def validate_csv(csv_path):
    print("\n" + "=" * 70)
    print(f"FILE: {csv_path.name}")
    print("=" * 70)

    if not csv_path.exists():
        print("CSV not found.")
        return

    df = pd.read_csv(csv_path)

    for column in ["image_path", "mask_path"]:
        if column not in df.columns:
            print(f"Missing column: {column}")
            continue

        paths = df[column].dropna().astype(str)

        exists = paths.map(
            lambda value: resolve_path(value).is_file()
        )

        print(f"\n{column}:")
        print(f"Total paths: {len(paths)}")
        print(f"Existing files: {exists.sum()}")
        print(f"Missing files: {(~exists).sum()}")

        if (~exists).any():
            print("Examples of missing paths:")

            for path in paths[~exists].head(5):
                print(f"  {path}")


def main():
    for csv_path in CSV_FILES:
        validate_csv(csv_path)


if __name__ == "__main__":
    main()