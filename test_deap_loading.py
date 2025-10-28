"""
Quick test script to verify DEAP dataset loading
"""

from src.data_loaders import DEAPLoader
from pathlib import Path

# Initialize loader
data_dir = "data/DEAP/"
loader = DEAPLoader(data_dir=data_dir, preprocessed=True)

print("=" * 60)
print("DEAP Dataset Loading Test")
print("=" * 60)

# Try loading subject 1
try:
    print("\n1. Loading subject 1...")
    dataset = loader.load_subject(subject_id=1, eeg_only=True)
    
    print("\n2. Dataset Info:")
    info = dataset.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n3. Sample Data:")
    trial_data, labels = dataset[0]
    print(f"   First trial shape: {trial_data.shape}")
    print(f"   First trial labels: {labels}")
    print(f"   Data range: [{trial_data.min():.2f}, {trial_data.max():.2f}]")
    
    print("\n4. Channel Names:")
    print(f"   {', '.join(dataset.channel_names[:10])}... (showing first 10)")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS! DEAP dataset loaded correctly")
    print("=" * 60)
    
except FileNotFoundError as e:
    print(f"\n✗ ERROR: {e}")
    print("\nMake sure your data is organized as:")
    print("  data/DEAP/data_preprocessed_python/s01.dat")
    print("  data/DEAP/data_preprocessed_python/s02.dat")
    print("  ...")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
