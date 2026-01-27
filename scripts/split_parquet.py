import logging

logging.basicConfig(level=logging.INFO)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.dataset_tools import realign_parquets_to_videos

repo_id = "libero_10_no_noops_1.0.0_lerobot"

# Load source dataset
dataset = LeRobotDataset(
    repo_id=repo_id,
    root=f"/mnt/sdb/IPEC-COMMUNITY/{repo_id}"
)

# Realign parquets to match video file structure
# This copies videos and rewrites parquets to have 1:1 correspondence
aligned = realign_parquets_to_videos(
    dataset,
    output_dir=f"/mnt/sdb/IPEC-COMMUNITY/{repo_id}_aligned",
    copy_videos=True,  # Set to False if videos are already in place
)

print(f"Created aligned dataset at {aligned.root}")
print(f"  Episodes: {aligned.meta.total_episodes}")
print(f"  Frames: {aligned.meta.total_frames}")

# Validation
print("\n--- Validation ---")

# 1. Basic metadata counts
assert aligned.meta.total_episodes == dataset.meta.total_episodes, "Episode count mismatch"
assert aligned.meta.total_frames == dataset.meta.total_frames, "Frame count mismatch"
assert aligned.meta.total_tasks == dataset.meta.total_tasks, "Task count mismatch"
assert aligned.meta.fps == dataset.meta.fps, "FPS mismatch"
print("✓ Metadata counts match")

# 2. Episode lengths match in same order (no shuffle)
for ep_idx in range(min(10, dataset.meta.total_episodes)):
    orig_len = dataset.meta.episodes[ep_idx]["length"]
    aligned_len = aligned.meta.episodes[ep_idx]["length"]
    assert orig_len == aligned_len, f"Episode {ep_idx} length mismatch: {orig_len} vs {aligned_len}"
print("✓ Episode lengths match")

# 3. Tasks preserved in same order
for ep_idx in range(min(10, dataset.meta.total_episodes)):
    orig_tasks = dataset.meta.episodes[ep_idx]["tasks"]
    aligned_tasks = aligned.meta.episodes[ep_idx]["tasks"]
    assert orig_tasks == aligned_tasks, f"Episode {ep_idx} tasks mismatch"
print("✓ Episode tasks match")

# 4. Sample frame data integrity
import torch
num_samples = 5
sample_indices = [0, dataset.meta.total_frames // 4, dataset.meta.total_frames // 2, 
                  3 * dataset.meta.total_frames // 4, dataset.meta.total_frames - 1]
for idx in sample_indices[:num_samples]:
    orig_frame = dataset[idx]
    aligned_frame = aligned[idx]
    
    # Check action matches
    assert torch.allclose(orig_frame["action"], aligned_frame["action"], atol=1e-5), \
        f"Frame {idx} action mismatch"
    
    # Check episode_index matches
    assert orig_frame["episode_index"] == aligned_frame["episode_index"], \
        f"Frame {idx} episode_index mismatch"
print("✓ Sample frame data matches")

# 5. Video frames load correctly
for video_key in aligned.meta.video_keys:
    frame = aligned[0]
    img = frame[video_key]
    assert img.dim() == 3 and img.shape[0] == 3, f"Video {video_key} invalid shape"
    assert img.min() >= 0 and img.max() <= 1, f"Video {video_key} invalid pixel range"
print("✓ Video frames decode correctly")

# 6. File alignment check - verify data and video files have same structure
first_video_key = aligned.meta.video_keys[0]
for ep_idx in range(min(20, aligned.meta.total_episodes)):
    data_chunk = aligned.meta.episodes[ep_idx]["data/chunk_index"]
    data_file = aligned.meta.episodes[ep_idx]["data/file_index"]
    video_chunk = aligned.meta.episodes[ep_idx][f"videos/{first_video_key}/chunk_index"]
    video_file = aligned.meta.episodes[ep_idx][f"videos/{first_video_key}/file_index"]
    meta_chunk = aligned.meta.episodes[ep_idx]["meta/episodes/chunk_index"]
    meta_file = aligned.meta.episodes[ep_idx]["meta/episodes/file_index"]
    
    assert (data_chunk, data_file) == (video_chunk, video_file), \
        f"Episode {ep_idx}: data ({data_chunk},{data_file}) != video ({video_chunk},{video_file})"
    assert (data_chunk, data_file) == (meta_chunk, meta_file), \
        f"Episode {ep_idx}: data ({data_chunk},{data_file}) != meta ({meta_chunk},{meta_file})"
print("✓ Files are aligned (data, video, meta have same chunk/file indices)")

print("\n✓ All validation checks passed!")

# 7. Load ALL frames to verify dataset integrity
print("\n--- Loading all frames to verify integrity ---")
from tqdm import tqdm

aligned = StreamingLeRobotDataset(
    repo_id=f"{repo_id}_aligned",
    root=f"/mnt/sdb/IPEC-COMMUNITY/{repo_id}_aligned"
)

for idx, frame in enumerate(tqdm(aligned, desc="Loading all frames")):
    pass

print("\n✓ Full dataset validation complete!")
