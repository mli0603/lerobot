import logging

logging.basicConfig(level=logging.INFO)

from lerobot.datasets.dataset_tools import update_info_with_data_files

dataset_path = "/home/max/git_ws/lerobot/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot_aligned"

# Update info.json with data_files list
data_files = update_info_with_data_files(dataset_path)

print(f"Updated {dataset_path}/meta/info.json")
print(f"  Found {len(data_files)} parquet files:")
for f in data_files:
    print(f"    - {f}")
