import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import shuffle_episodes

repo_id = "libero_90_no_noops_lerobot"

# Load source dataset
dataset = LeRobotDataset(
    repo_id=repo_id,
    root=f"/mnt/sdb/IPEC-COMMUNITY/{repo_id}"
)

# Shuffle episodes with reproducible seed
# This re-encodes videos with episodes in shuffled order
shuffled = shuffle_episodes(
    dataset,
    output_dir=f"/mnt/sdb/IPEC-COMMUNITY/{repo_id}_shuffled",
    seed=42,
)

print(f"Created shuffled dataset at {shuffled.root}")
print(f"  Episodes: {shuffled.meta.total_episodes}")
print(f"  Frames: {shuffled.meta.total_frames}")

# Validation
print("\n--- Validation ---")
import torch

# 1. Basic metadata counts match
assert shuffled.meta.total_episodes == dataset.meta.total_episodes, "Episode count mismatch"
assert shuffled.meta.total_frames == dataset.meta.total_frames, "Frame count mismatch"
assert shuffled.meta.total_tasks == dataset.meta.total_tasks, "Task count mismatch"
assert shuffled.meta.fps == dataset.meta.fps, "FPS mismatch"
print("✓ Metadata counts match")

# 2. Task distribution preserved (same tasks with same counts, order may differ)
original_tasks: Counter = Counter()
shuffled_tasks: Counter = Counter()
for ep_idx in range(dataset.meta.total_episodes):
    original_tasks[tuple(dataset.meta.episodes[ep_idx]["tasks"])] += 1
    shuffled_tasks[tuple(shuffled.meta.episodes[ep_idx]["tasks"])] += 1
assert original_tasks == shuffled_tasks, "Task distribution changed!"
print("✓ Task distribution preserved")

# 3. Episode lengths match (same lengths, just reordered)
original_lengths = sorted([ep["length"] for ep in dataset.meta.episodes])
shuffled_lengths = sorted([ep["length"] for ep in shuffled.meta.episodes])
assert original_lengths == shuffled_lengths, "Episode lengths don't match after sorting"
print("✓ Episode lengths match (sorted)")

# 4. Frame data integrity - verify shuffled episodes exist in original
def get_episode_actions(ds: LeRobotDataset, ep_idx: int) -> torch.Tensor:
    """Get action sequence for an episode (first 10 frames)."""
    ep = ds.meta.episodes[ep_idx]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    actions = []
    for i in range(start, min(start + 10, end)):
        frame = ds[i]
        actions.append(frame["action"])
    return torch.stack(actions)

# Check a few episodes from shuffled dataset
check_indices = [0, shuffled.meta.total_episodes // 2, shuffled.meta.total_episodes - 1]
for new_idx in check_indices[:3]:
    shuffled_actions = get_episode_actions(shuffled, new_idx)
    
    # Find matching episode in original by comparing action sequences
    found_match = False
    for old_idx in range(dataset.meta.total_episodes):
        try:
            original_actions = get_episode_actions(dataset, old_idx)
            if shuffled_actions.shape == original_actions.shape:
                if torch.allclose(shuffled_actions, original_actions, atol=1e-5):
                    found_match = True
                    break
        except Exception:
            continue
    
    assert found_match, f"Episode {new_idx} in shuffled dataset has no matching data in original!"
print("✓ Frame data integrity verified (sample episodes)")

# 5. Video frames load correctly
for video_key in shuffled.meta.video_keys:
    frame = shuffled[0]
    img = frame[video_key]
    assert img.dim() == 3 and img.shape[0] == 3, f"Video {video_key} invalid shape"
    assert img.min() >= 0 and img.max() <= 1, f"Video {video_key} invalid pixel range"
print("✓ Video frames decode correctly")

# 6. File alignment check - verify data and video files have same structure
first_video_key = shuffled.meta.video_keys[0]
for ep_idx in range(min(20, shuffled.meta.total_episodes)):
    data_chunk = shuffled.meta.episodes[ep_idx]["data/chunk_index"]
    data_file = shuffled.meta.episodes[ep_idx]["data/file_index"]
    video_chunk = shuffled.meta.episodes[ep_idx][f"videos/{first_video_key}/chunk_index"]
    video_file = shuffled.meta.episodes[ep_idx][f"videos/{first_video_key}/file_index"]
    meta_chunk = shuffled.meta.episodes[ep_idx]["meta/episodes/chunk_index"]
    meta_file = shuffled.meta.episodes[ep_idx]["meta/episodes/file_index"]
    
    assert (data_chunk, data_file) == (video_chunk, video_file), \
        f"Episode {ep_idx}: data ({data_chunk},{data_file}) != video ({video_chunk},{video_file})"
    assert (data_chunk, data_file) == (meta_chunk, meta_file), \
        f"Episode {ep_idx}: data ({data_chunk},{data_file}) != meta ({meta_chunk},{meta_file})"
print("✓ Files are aligned (data, video, meta have same chunk/file indices)")

# 7. Actually shuffled - verify episode order changed
if dataset.meta.total_episodes >= 5:
    n_compare = min(10, dataset.meta.total_episodes)
    original_first_tasks = [tuple(dataset.meta.episodes[i]["tasks"]) for i in range(n_compare)]
    shuffled_first_tasks = [tuple(shuffled.meta.episodes[i]["tasks"]) for i in range(n_compare)]
    
    if len(set(original_first_tasks)) > 1:  # Only check if there are different tasks
        assert original_first_tasks != shuffled_first_tasks, \
            "Episodes don't appear to be shuffled - first N episode tasks are identical!"
        print("✓ Episode order is different from original (shuffled)")
    else:
        print("⚠ All episodes have same task, cannot verify shuffle by task")
else:
    print("⚠ Too few episodes to verify shuffle")

print("\n✓ All validation checks passed!")
