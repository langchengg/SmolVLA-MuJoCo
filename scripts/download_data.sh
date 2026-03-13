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
import os
from huggingface_hub import snapshot_download

datasets = {
    'LIBERO-Object': 'lerobot/libero_object_image',
    'LIBERO-Spatial': 'lerobot/libero_spatial_image',
    'LIBERO-Goal': 'lerobot/libero_goal_image',
}

for name, repo_id in datasets.items():
    print(f'Downloading {name} ({repo_id}) to HuggingFace Cache...')
    try:
        # snapshot_download automatically caches it exactly where LeRobot expects it
        snapshot_download(repo_id=repo_id, repo_type='dataset', max_workers=8)
        print(f'  ✅ {name}: Downloaded successfully')
    except Exception as e:
        print(f'  ❌ {name} failed: {e}')

print()
print('✅ Dataset download complete!')
"

echo ""
echo "✅ All datasets downloaded!"
