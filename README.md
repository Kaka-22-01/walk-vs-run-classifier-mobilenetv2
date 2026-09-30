# Walk vs Run Image Classifier

A binary image classifier built with **TensorFlow / Keras** that distinguishes photos of people **running** from people **walking**, using transfer learning on top of MobileNetV2.

## Overview

- **Task:** Binary image classification (`run` vs `walk`)
- **Approach:** Transfer learning — a frozen MobileNetV2 (pretrained on ImageNet) as a feature extractor, plus a small custom classification head
- **Framework:** TensorFlow / Keras
- **Final test accuracy:** ~0.91

## Dataset

This project uses the [**Walk or Run**](https://www.kaggle.com/datasets/huan9huan/walk-or-run) dataset from Kaggle.

- 600 training/validation images (80/20 split), 2 classes
- 141 held-out test images

**The dataset is not included in this repository** due to its size. To reproduce the results:

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/huan9huan/walk-or-run
2. Extract it so the folder structure looks like:
   ```
   walk_or_run_train/
       run/
       walk/
   walk_or_run_test/
       run/
       walk/
   ```
3. Place both folders in the project root (next to the notebook).

## Model Architecture

```
Input(255, 255, 3)
  → Data Augmentation (RandomFlip, RandomBrightness, RandomZoom)
  → Rescaling (-1 to 1)
  → MobileNetV2 (frozen, ImageNet weights, no top)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(1, activation='sigmoid')
```

The MobileNetV2 base is frozen (`trainable = False`), so only the small classification head is trained — this works well even with a fairly small dataset.

## Training

- Optimizer: Adam
- Loss: Binary Crossentropy
- Callback: EarlyStopping (`monitor='val_loss'`, `patience=4`, `restore_best_weights=True`)

## Results

| Metric | Value |
|---|---|
| Test accuracy | ~0.82 |
| Test loss | ~0.35 |

### Training curves
Training and validation accuracy over epochs — see `results/accuracy_plot.png`.

### Confusion Matrix
See `results/confusion_matrix.png`. The model is stronger at recognizing `walk` than `run`; most errors are `run` images misclassified as `walk`.

## How to Run

1. Install dependencies:
   ```bash
   pip install tensorflow matplotlib numpy scikit-learn
   ```
2. Download and place the dataset as described above.
3. Open `walk_vs_run.ipynb` and run all cells in order.

## Project Structure

```
.
├── walk_vs_run.ipynb      # Main notebook (data loading, model, training, evaluation)
├── README.md
└── results/               # Saved plots (accuracy curve, confusion matrix)
```

## Possible Improvements

- Fine-tune the last few layers of MobileNetV2 with a low learning rate
- Collect more training data, especially borderline "jogging" images
- Try other pretrained backbones (EfficientNet, ResNet) for comparison
