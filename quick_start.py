"""
Quick Start Script - Chạy nhanh để test
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import VulnerabilityDataset


def main():
    print("=" * 70)
    print("VULNERABILITY DETECTION - QUICK START")
    print("=" * 70)
    
    # Check dataset
    root_dir = Path(__file__).parent
    
    print("\n1. Checking dataset...")
    dataset = VulnerabilityDataset(str(root_dir))
    
    if len(dataset) == 0:
        print("❌ No data found!")
        print("Please make sure you have JSON files in output/ and output_safe/")
        return
    
    print(f"✓ Found {len(dataset)} samples")
    
    # Check first sample
    print("\n2. Loading first sample...")
    sample = dataset[0]
    print(f"✓ Sample loaded:")
    print(f"   - Nodes: {sample.num_nodes}")
    print(f"   - Edges: {sample.edge_index.shape[1]}")
    print(f"   - Label: {sample.y.item()} ({dataset.LABEL_NAMES[sample.y.item()]})")
    
    print("\n3. Dataset distribution:")
    from collections import Counter
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    for label_id, count in sorted(label_counts.items()):
        label_name = list(dataset.LABEL_MAP.keys())[list(dataset.LABEL_MAP.values()).index(label_id)]
        print(f"   {label_name:20s}: {count:4d} samples")
    
    print("\n" + "=" * 70)
    print("✓ Everything looks good!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train model: cd src && python train.py")
    print("3. Predict: python predict.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
