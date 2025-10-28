# Hyperparameter Optimization Guide

## Overview

Guide for automatically optimizing emotion recognition model hyperparameters to improve accuracy by 5-15%.

## Why Optimize?

Current `config.py` uses manual values. Automatic tuning finds optimal configuration for your specific data.

## Methods

**Keras Tuner (Recommended):**
- Easy to use, Bayesian optimization
- Install: `pip install keras-tuner`
- Best for: Most users

**Optuna (Advanced):**
- Fastest convergence, TPE algorithm
- Install: `pip install optuna`
- Best for: Maximum accuracy

## Quick Start

```bash
# Quick test (5 trials, 10 min)
python run_hyperparameter_tuning.py --method keras-tuner --max-trials 5

# Full optimization (30 trials, 1-2 hours)
python run_hyperparameter_tuning.py --method keras-tuner --max-trials 30
```

## Apply Results

1. Check `models/best_hyperparameters.json`
2. Update values in `src/config.py`
3. Retrain model: `python src/train_emotion_model.py`

## Expected Improvements

| Starting Accuracy | After Optimization |
|-------------------|-------------------|
| 70% | 76-82% (+6-12%) |
| 80% | 85-91% (+5-11%) |
| 85% | 88-93% (+3-8%) |

## Parameters Optimized

- Learning rate
- CNN filters & kernel sizes
- LSTM units & dropout
- Dense layer sizes
- Batch size
- Optimizer type

---
*For detailed usage, see Keras Tuner/Optuna documentation*
