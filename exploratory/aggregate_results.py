import os
import json
import pandas as pd

base_dir = "results"
records = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("results_split") and file.endswith(".json"):
            filepath = os.path.join(root, file)

            # Example: results/ddr1/Morgan_RF_random
            rel_path = os.path.relpath(root, base_dir)
            parts = rel_path.split(os.sep)

            if len(parts) != 2:
                continue  # safety

            target = parts[0]  # e.g. "ddr1" or "mapk14"
            featurizer, model_type, split_type = parts[1].split("_")

            with open(filepath, "r") as f:
                results = json.load(f)

            row = {
                "target": target,
                "featurizer": featurizer,
                "model_type": model_type,
                "split_type": split_type,
            }

            # Test results
            if "test" in results:
                for metric, val in results["test"].items():
                    row[f"{metric}_test"] = val

            # Held-out sets
            if "all" in results:
                for condition, metrics in results["all"].items():
                    for metric, val in metrics.items():
                        row[f"{metric}_all_{condition}"] = val

            # In-library sets
            if "lib" in results:
                for condition, metrics in results["lib"].items():
                    for metric, val in metrics.items():
                        row[f"{metric}_lib_{condition}"] = val

            records.append(row)

# Build DataFrame
df_results = pd.DataFrame(records)

# Save to CSV
df_results.to_csv("results/all_results.csv", index=False)
# Preview
print(df_results.head())
print(df_results.shape)
