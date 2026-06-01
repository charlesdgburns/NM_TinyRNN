#!/usr/bin/env python3
"""
Export Representative Models Based on Performance Metrics.
Usage:
    python export_models.py --model_id "2_unit_monoGRU_relu_unipolar" --metric median
"""

import argparse
import os
import sys
from pathlib import Path

# Define the parent directory of NM_TinyRNN
package_parent_dir = "/ceph/behrens/wsilver/reversal/code"

if package_parent_dir not in sys.path:
    sys.path.insert(0, package_parent_dir)

# Now your imports will work smoothly
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.models import submit_jobs
import shutil
from pathlib import Path
import pandas as pd

# Import your custom modules
from NM_TinyRNN.code.measures import analysis
from NM_TinyRNN.code.models import submit_jobs 


def closest_to_median(subdf):
    """Finds the row closest to the median value of 'eval_CE'."""
    if subdf.empty:
        return None
    med = subdf["eval_CE"].median()
    idx = (subdf["eval_CE"] - med).abs().idxmin()
    return subdf.loc[idx]


def idx_min(subdf):
    """Finds the row with the minimum value of 'eval_CE'."""
    if subdf.empty:
        return None
    min_val = subdf["eval_CE"].min()
    idx = (subdf["eval_CE"] - min_val).abs().idxmin()
    return subdf.loc[idx]


def copy_representative_models(df, example_path):
    """
    Copy model files into EXAMPLE_PATH/<subject_id>/<model_type>/.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['info_path', 'model_type', 'model_id', 'subject_id'].
    example_path : str or Path
        Parent directory where copies will be stored.
    """
    example_path = Path(example_path)
    example_path.mkdir(parents=True, exist_ok=True)

    # Files to copy for each model
    file_suffixes = [
        "_info.json",
        "_model.pickle",
        "_trials_data.htsv",
    ]
    
    copied_count = 0
    
    for _, row in df.iterrows():
        # Safeguard against unexpected missing values
        if pd.isna(row["info_path"]) or pd.isna(row["model_id"]) or pd.isna(row["subject_id"]):
            continue
            
        save_path = Path(row["info_path"]).parent
        model_type = row["model_type"]
        model_id = row["model_id"]
        subject_id = row["subject_id"]

        # Destination: parent_dir / subject_id / model_type
        dest_dir = example_path / str(subject_id) / str(model_type)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for suffix in file_suffixes:
            src = save_path / f"{model_id}{suffix}"
            dst = dest_dir / f"{model_id}{suffix}"
            if src.exists():
                shutil.copy(src, dst)
                print(f"Copied {src.name} → {dst.relative_to(example_path.parent)}")
                copied_count += 1
            else:
                print(f"Warning: {src} not found!")
                
    print(f"\nSuccessfully processed and copied {copied_count} files.")


def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Filter and copy model data based on model_id.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        required=True, 
        help="The specific model_id to filter by (e.g., '2_unit_monoGRU_relu_unipolar')"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        choices=["median", "min"], 
        default="median",
        help="Select whether to find the model closest to the 'median' or the absolute 'min' performance."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./NM_TinyRNN/data/rnns/subject_examples",
        help="Target parent directory to save model assets."
    )
    
    args = parser.parse_args()

    # 2. Fetch Dataframes
    print("Fetching analysis dataframes... (this may take a moment)")
    info_df = submit_jobs.get_job_info_df()
    all_models_df = analysis.get_analysis_df(info_df, mode='all')

    # 3. Filter DataFrame by the provided model_id
    print(f"Filtering dataset for model_id: '{args.model_id}'")
    filtered_df = all_models_df.query(f'model_id == "{args.model_id}"')

    if filtered_df.empty:
        print(f"Error: No records found matching model_id '{args.model_id}'. Exiting.")
        return

    # 4. Group by Subject and Apply Selection Metric
    print(f"Grouping by 'subject_id' and finding the representative '{args.metric}' models...")
    
    # Selection mapping based on CLI choice
    metric_func = closest_to_median if args.metric == "median" else idx_min

    # Group by subject_id (and model_id since it's a fixed constant here)
    # as_index=False keeps it clean for the subsequent iteration
    representative_df = (
        filtered_df.groupby(["model_id", "subject_id"], as_index=False)
        .apply(metric_func)
        .reset_index(drop=True)
    )

    # 5. Copy the chosen models over to target directory
    print(f"Starting file copy routine to: {args.output_dir}")
    copy_representative_models(representative_df, args.output_dir)


if __name__ == "__main__":
    main()