# Audio Event Tagging Using SVM and 2D-CNN-LSTM Models

## About The Project

This project aims to classify environmental sounds into predefined categories using both traditional machine learning methods (SVM) and deep learning techniques (2D-CNN-LSTM). The dataset used, ESC-50, comprises 2000 audio recordings across 50 sound classes.

### Objectives
- Develop and evaluate SVM and 2D-CNN-LSTM models for audio event tagging.
- Explore the effects of feature extraction techniques on model performance.
- Investigate the impact of data augmentation on classification accuracy.

### Key Highlights
- ESC-50 dataset: Balanced with 40 recordings per class across five categories (e.g., animal sounds, human non-speech sounds).
- SVM models trained with various feature extraction methods, such as MFCCs and ZCR.
- Deep learning models incorporating log-mel spectrograms and data augmentation.

## Methodology

### Dataset
- **ESC-50**: 2000 audio recordings, each 5 seconds long, categorized into 50 classes.
- **ESC-10**: A subset of ESC-50 with 10 easily distinguishable classes.

### Preprocessing
- **Feature Extraction for SVM**: MFCCs, ZCR, energy statistics, and their derivatives.
- **Log-Mel Spectrograms**: Used as input for 2D-CNN-LSTM models.
- **Data Augmentation**: Noise addition, pitch shifting, and time stretching.

### Models
1. **SVM**:
   - Feature extraction techniques include PCA and random frame selection.
   - Approaches: One-vs-One (OVO) and One-vs-Rest (OVR).

2. **2D-CNN-LSTM**:
   - Combines convolutional layers for spatial feature learning with LSTM layers for temporal dependencies.
   - Trained with augmented data using AdamW optimizer and early stopping.

## Contributors

- Wong Wei Kang
- Anusha Porwal

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Dataset: [ESC-50 by Karol J. Piczak](https://github.com/karolpiczak/ESC-50)
- Libraries: Librosa, Audiomentations, TensorFlow, Scikit-learn
