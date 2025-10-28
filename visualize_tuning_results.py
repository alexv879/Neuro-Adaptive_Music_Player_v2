"""
Visualize Hyperparameter Tuning Results
========================================

This script creates comprehensive visualizations of your hyperparameter
optimization results to help understand what worked best.

Usage:
    python visualize_tuning_results.py --method keras-tuner
    python visualize_tuning_results.py --method optuna --show-importance

Features:
    - Optimization history plots
    - Parameter importance analysis  
    - Hyperparameter distribution heatmaps
    - Best trial comparison

Author: CMP9780M Assessment
"""

import json
import argparse
from pathlib import Path
import sys

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib/Seaborn not installed. Install with: pip install matplotlib seaborn")
    MATPLOTLIB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️  Pandas not installed. Install with: pip install pandas")
    PANDAS_AVAILABLE = False

sys.path.append(str(Path(__file__).parent / "src"))
from config import MODEL_DIR


def visualize_keras_tuner_results(tuner_dir: Path):
    """Visualize Keras Tuner results."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib required for visualization")
        return
    
    # Load trial results
    oracle_path = tuner_dir / "oracle.json"
    if not oracle_path.exists():
        print(f"❌ No tuning results found in {tuner_dir}")
        return
    
    with open(oracle_path, 'r') as f:
        oracle_data = json.load(f)
    
    trials = oracle_data.get('hyperparameters', {}).get('values', [])
    
    if not trials:
        print("❌ No trials found in oracle data")
        return
    
    print(f"✅ Found {len(trials)} trials")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Keras Tuner Optimization Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Optimization History
    ax = axes[0, 0]
    # TODO: Extract trial scores and plot
    ax.set_title('Optimization History')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Validation Accuracy')
    
    # Plot 2: Hyperparameter Distribution
    ax = axes[0, 1]
    ax.set_title('Learning Rate Distribution')
    
    # Plot 3: Top Configurations
    ax = axes[1, 0]
    ax.set_title('Top 10 Configurations')
    
    # Plot 4: Parameter Importance
    ax = axes[1, 1]
    ax.set_title('Hyperparameter Importance')
    
    plt.tight_layout()
    output_path = MODEL_DIR / "tuning_results_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    plt.show()


def visualize_optuna_results(study_name: str = "emotion_recognition_study"):
    """Visualize Optuna results with built-in plots."""
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna required for visualization")
        return
    
    # Load study from database
    storage_path = MODEL_DIR / "tuning" / "optuna.db"
    
    if not storage_path.exists():
        print(f"❌ No Optuna study found at {storage_path}")
        return
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}"
        )
    except KeyError:
        print(f"❌ Study '{study_name}' not found in database")
        return
    
    print(f"✅ Loaded study with {len(study.trials)} trials")
    print(f"   Best accuracy: {study.best_value:.4f}")
    
    # Create visualization directory
    viz_dir = MODEL_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Plot 1: Optimization History
    print("\n📊 Generating optimization history plot...")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(str(viz_dir / "optimization_history.png"))
    print(f"   ✅ Saved to {viz_dir}/optimization_history.png")
    
    # Plot 2: Parameter Importances
    print("📊 Generating parameter importance plot...")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(str(viz_dir / "parameter_importances.png"))
    print(f"   ✅ Saved to {viz_dir}/parameter_importances.png")
    
    # Plot 3: Parallel Coordinate Plot
    print("📊 Generating parallel coordinate plot...")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(str(viz_dir / "parallel_coordinate.png"))
    print(f"   ✅ Saved to {viz_dir}/parallel_coordinate.png")
    
    # Plot 4: Slice Plot
    print("📊 Generating slice plot...")
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(str(viz_dir / "slice_plot.png"))
    print(f"   ✅ Saved to {viz_dir}/slice_plot.png")
    
    # Plot 5: Contour Plot (top parameters)
    print("📊 Generating contour plot...")
    fig = optuna.visualization.plot_contour(study, params=['learning_rate', 'lstm_units'])
    fig.write_image(str(viz_dir / "contour_plot.png"))
    print(f"   ✅ Saved to {viz_dir}/contour_plot.png")
    
    print(f"\n✅ All visualizations saved to {viz_dir}/")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📈 OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best trial number: {study.best_trial.number}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:30s}: {value}")
    print("=" * 80)


def compare_configurations(config_path: Path):
    """Compare before/after configurations."""
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\n" + "=" * 80)
    print("📊 HYPERPARAMETER COMPARISON")
    print("=" * 80)
    
    # Default values (from config.py)
    defaults = {
        'learning_rate': 0.001,
        'cnn_filters_0': 64,
        'cnn_filters_1': 128,
        'cnn_filters_2': 256,
        'lstm_units': 128,
        'dense_units_0': 256,
        'dense_units_1': 128,
        'batch_size': 32,
        'optimizer': 'adam'
    }
    
    optimized = config.get('hyperparameters', {})
    
    print(f"\n{'Parameter':<25} {'Default':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 80)
    
    for key in defaults.keys():
        if key in optimized:
            default_val = defaults[key]
            opt_val = optimized[key]
            
            if isinstance(default_val, float):
                change = f"{((opt_val - default_val) / default_val * 100):+.1f}%"
            elif isinstance(default_val, int):
                change = f"{opt_val - default_val:+d}"
            else:
                change = "Changed" if opt_val != default_val else "Same"
            
            print(f"{key:<25} {str(default_val):>15} {str(opt_val):>15} {change:>15}")
    
    print("-" * 80)
    print(f"\nBest Accuracy: {config.get('metadata', {}).get('best_score', 'N/A')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Visualize hyperparameter tuning results')
    parser.add_argument('--method', type=str, choices=['keras-tuner', 'optuna'],
                       default='optuna', help='Tuning method to visualize')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with default configuration')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 HYPERPARAMETER TUNING VISUALIZATION")
    print("=" * 80)
    
    if args.method == 'keras-tuner':
        tuner_dir = MODEL_DIR / "tuning" / "emotion_tuning_bayesian"
        visualize_keras_tuner_results(tuner_dir)
        
        if args.compare:
            config_path = MODEL_DIR / "best_hyperparameters.json"
            compare_configurations(config_path)
    
    elif args.method == 'optuna':
        visualize_optuna_results()
        
        if args.compare:
            config_path = MODEL_DIR / "best_hyperparameters_optuna.json"
            compare_configurations(config_path)


if __name__ == "__main__":
    main()
