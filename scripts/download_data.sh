#!/bin/bash
# =============================================================================
# Download Datasets
# =============================================================================
# Downloads LIBERO datasets from HuggingFace Hub via LeRobot

set -e

echo "📥 Downloading Datasets"
echo "======================="

# Create data directory
mkdir -p data/

# Download via Python (uses LeRobot / HuggingFace datasets)
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

datasets = {
    'LIBERO-Object': 'lerobot/libero_object_no_noops',
    'LIBERO-Spatial': 'lerobot/libero_spatial_no_noops',
    'LIBERO-Goal': 'lerobot/libero_goal_no_noops',
}

for name, repo_id in datasets.items():
    print(f'Downloading {name} ({repo_id})...')
    try:
        ds = LeRobotDataset(repo_id)
        print(f'  ✅ {name}: {len(ds)} samples')
    except Exception as e:
        print(f'  ⚠️ {name}: {e}')
        print(f'  Trying HuggingFace datasets fallback...')
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id)
            print(f'  ✅ {name}: loaded via HF datasets')
        except Exception as e2:
            print(f'  ❌ {name}: {e2}')

print()
print('✅ Dataset download complete!')
"

echo ""
echo "✅ All datasets downloaded!"
