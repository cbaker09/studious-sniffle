#!/usr/bin/env python3
"""
Data Loader for Loan Default Prediction System

This script loads the Lending Club CSV data into a DuckDB database.
It handles large files with chunked reading and provides data validation.
"""

import os
import sys
from pathlib import Path
import duckdb
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


class LoanDataLoader:
    """Handles loading loan data from CSV to DuckDB"""

    def __init__(self, db_path: str = "data/loan_default.duckdb"):
        """
        Initialize the data loader

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Establish connection to DuckDB"""
        print(f"Connecting to DuckDB at: {self.db_path}")
        self.conn = duckdb.connect(self.db_path)
        print("✓ Connected to database")

    def load_csv_to_table(
        self,
        csv_path: str,
        table_name: str = "loans",
        chunk_size: int = 50000
    ):
        """
        Load CSV file into DuckDB table with chunked reading

        Args:
            csv_path: Path to CSV file
            table_name: Name of target table
            chunk_size: Number of rows per chunk
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"\nLoading CSV from: {csv_path}")
        print(f"Target table: {table_name}")

        # Get total rows for progress bar
        total_rows = sum(1 for _ in open(csv_path)) - 1  # Subtract header
        print(f"Total rows to load: {total_rows:,}")

        # Drop table if exists
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Read and load in chunks
        chunk_count = 0
        total_loaded = 0

        # Use pandas to read CSV in chunks
        for chunk in tqdm(
            pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False),
            total=(total_rows // chunk_size) + 1,
            desc="Loading chunks"
        ):
            chunk_count += 1
            rows_in_chunk = len(chunk)
            total_loaded += rows_in_chunk

            # Clean column names (replace spaces and special chars)
            chunk.columns = [
                col.strip().lower().replace(' ', '_').replace('-', '_')
                for col in chunk.columns
            ]

            # Create or append to table
            if chunk_count == 1:
                # First chunk - create table
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM chunk"
                )
            else:
                # Subsequent chunks - append
                self.conn.execute(
                    f"INSERT INTO {table_name} SELECT * FROM chunk"
                )

        print(f"\n✓ Successfully loaded {total_loaded:,} total rows into {table_name}")

    def validate_data(self, table_name: str = "loans"):
        """
        Run basic validation checks on loaded data

        Args:
            table_name: Name of table to validate
        """
        print("\n" + "=" * 50)
        print("Data Validation Results")
        print("=" * 50)

        # Total rows
        result = self.conn.execute(
            f"SELECT COUNT(*) as total FROM {table_name}"
        ).fetchone()
        total_rows = result[0]
        print(f"Total rows: {total_rows:,}")

        # Check for key columns
        key_columns = ['id', 'loan_amnt', 'loan_status', 'int_rate']
        for col in key_columns:
            result = self.conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NOT NULL"
            ).fetchone()
            non_null = result[0]
            pct = (non_null / total_rows * 100) if total_rows > 0 else 0
            print(f"{col}: {non_null:,} non-null ({pct:.1f}%)")

        # Loan status distribution
        print("\nLoan Status Distribution:")
        results = self.conn.execute(
            f"""
            SELECT loan_status, COUNT(*) as count
            FROM {table_name}
            GROUP BY loan_status
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

        for status, count in results:
            pct = (count / total_rows * 100) if total_rows > 0 else 0
            print(f"  {status}: {count:,} ({pct:.1f}%)")

        # Calculate default rate
        default_statuses = [
            'Charged Off',
            'Default',
            'Does not meet the credit policy. Status:Charged Off'
        ]

        paid_statuses = [
            'Fully Paid',
            'Does not meet the credit policy. Status:Fully Paid'
        ]

        # Count defaults
        default_count = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            WHERE loan_status IN ({','.join([f"'{s}'" for s in default_statuses])})
            """
        ).fetchone()[0]

        # Count paid
        paid_count = self.conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            WHERE loan_status IN ({','.join([f"'{s}'" for s in paid_statuses])})
            """
        ).fetchone()[0]

        completed_total = default_count + paid_count
        if completed_total > 0:
            default_rate = (default_count / completed_total) * 100
            print(f"\nCompleted Loans (Paid or Defaulted): {completed_total:,}")
            print(f"Defaults: {default_count:,}")
            print(f"Paid: {paid_count:,}")
            print(f"Default Rate: {default_rate:.2f}%")

        print("=" * 50)

    def create_sample_dataset(
        self,
        source_table: str = "loans",
        sample_size: int = 10000,
        target_table: str = "loans_sample"
    ):
        """
        Create a smaller sample dataset for testing

        Args:
            source_table: Source table name
            sample_size: Number of rows to sample
            target_table: Target table name for sample
        """
        print(f"\nCreating sample dataset: {target_table}")

        self.conn.execute(f"DROP TABLE IF EXISTS {target_table}")

        self.conn.execute(
            f"""
            CREATE TABLE {target_table} AS
            SELECT *
            FROM {source_table}
            USING SAMPLE {sample_size} ROWS
            """
        )

        count = self.conn.execute(
            f"SELECT COUNT(*) FROM {target_table}"
        ).fetchone()[0]

        print(f"✓ Created sample table with {count:,} rows")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    """Main execution function"""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "loan_default.duckdb"
    csv_path = project_root / "data" / "raw" / "loan.csv"

    # Alternative common names for the CSV file
    csv_alternatives = [
        "loan.csv",
        "accepted_2007_to_2018.csv",
        "accepted_2007_to_2018Q4.csv",
        "loans.csv"
    ]

    # Find the CSV file
    found_csv = None
    raw_data_dir = project_root / "data" / "raw"

    if raw_data_dir.exists():
        for alt_name in csv_alternatives:
            alt_path = raw_data_dir / alt_name
            if alt_path.exists():
                found_csv = alt_path
                break

        # If still not found, list available files
        if not found_csv:
            available_files = list(raw_data_dir.glob("*.csv"))
            if available_files:
                print(f"Found CSV files in {raw_data_dir}:")
                for f in available_files:
                    print(f"  - {f.name}")
                # Use the first one found
                found_csv = available_files[0]
                print(f"\nUsing: {found_csv.name}")

    if not found_csv:
        print("\n" + "=" * 70)
        print("CSV FILE NOT FOUND")
        print("=" * 70)
        print(f"\nPlease download the Lending Club dataset and place it in:")
        print(f"  {raw_data_dir}/")
        print(f"\nExpected filename (one of):")
        for name in csv_alternatives:
            print(f"  - {name}")
        print("\nDownload from: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("=" * 70)
        sys.exit(1)

    # Initialize loader
    loader = LoanDataLoader(db_path=str(db_path))

    try:
        # Connect to database
        loader.connect()

        # Load data
        loader.load_csv_to_table(
            csv_path=str(found_csv),
            table_name="loans",
            chunk_size=50000
        )

        # Validate data
        loader.validate_data(table_name="loans")

        # Create sample dataset (optional, for testing)
        print("\nWould you like to create a sample dataset for testing? (Y/n): ", end="")
        response = input().strip().lower()

        if response in ['', 'y', 'yes']:
            loader.create_sample_dataset(
                source_table="loans",
                sample_size=10000,
                target_table="loans_sample"
            )

    finally:
        # Always close connection
        loader.close()

    print("\n✓ Data loading complete!")
    print(f"\nNext steps:")
    print(f"1. cd dbt_project")
    print(f"2. dbt run")
    print(f"3. dbt test")


if __name__ == "__main__":
    main()
