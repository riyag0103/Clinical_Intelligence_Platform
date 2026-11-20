from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Clinical_Project")

CSV_FILES = [
    PROJECT_ROOT / "datasets" / "metadata" / "full_metadata.csv",
    PROJECT_ROOT / "datasets" / "metadata" / "clean_ct_metadata.csv",
    PROJECT_ROOT / "datasets" / "metadata" / "ct_metadata.csv",
    PROJECT_ROOT / "datasets" / "metadata" / "balanced" / "balanced_metadata.csv",
]

def inspect_csv(csv_path):
    print("\n" + "=" * 80)
    print(f"FILE: {csv_path.name}")
    print("=" * 80)

    if not csv_path.exists():
        print("File not found.")
        return

    try:
        df = pd.read_csv(csv_path)

        print(f"\nRows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumn names:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        # Inspect columns that might contain labels or identifiers.
        keywords = [
            "patient", "subject", "nodule", "malignancy",
            "cancer", "diagnosis", "label", "target",
            "image", "mask", "slice", "path", "risk"
        ]

        relevant_columns = [
            col for col in df.columns
            if any(word in col.lower() for word in keywords)
        ]

        if relevant_columns:
            print("\nPotentially relevant columns:")
            print(relevant_columns)

            for col in relevant_columns:
                if any(
                    word in col.lower()
                    for word in [
                        "malignancy", "diagnosis",
                        "label", "target", "cancer"
                    ]
                ):
                    print(f"\nValue counts for '{col}':")
                    print(df[col].value_counts(dropna=False).head(20))

    except Exception as error:
        print(f"Could not inspect file: {error}")


def main():
    for csv_path in CSV_FILES:
        inspect_csv(csv_path)


if __name__ == "__main__":
    main()
    