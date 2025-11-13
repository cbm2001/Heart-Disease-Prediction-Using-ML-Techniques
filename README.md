# Heart Disease Prediction Using Advanced ML Techniques

> Final year research project achieving 98.5% accuracy in heart disease prediction through mathematics-driven feature engineering with neural extractors, genetic algorithms, and self-attention transformers.

## Overview

Cardiovascular diseases account for 32% of all deaths worldwide, making early detection critical. This research project develops an advanced machine learning system that combines deep learning-based feature extraction, genetic algorithm optimization, and ensemble classifiers to achieve state-of-the-art prediction accuracy.

**Key Achievements:**
- **98.5% Accuracy** (Training: 98.46%, Testing: 98.55%)
- **Novel Feature Engineering**: Neural extractors + Genetic algorithms
- **Transformer Architecture**: 20% accuracy improvement over baseline
- **Synthetic Data Generation**: Addressed small dataset challenges
- **Robust Validation**: 10-fold cross-validation with 0.985 F1-score

## Dataset

**Cleveland Heart Disease Dataset** (UCI Repository)
- **Original Size**: 303 patient records
- **Augmented Size**: 50,000+ synthetic samples using SMOTE and GAN techniques
- **Features**: 13 clinical and demographic attributes
- **Target**: Binary classification (presence/absence of heart disease)

**Key Features:**
- Age, sex, chest pain type, blood pressure
- Cholesterol levels, blood sugar, ECG results
- Maximum heart rate, exercise-induced angina
- ST depression, slope, vessels colored by fluoroscopy

## Methodology

### 1. Feature Engineering Pipeline

**Deep Learning Feature Extraction:**
- **Neural Extractors**: Autoencoder-based dimensionality reduction
- Learned complex, non-linear feature representations
- Captured latent patterns invisible to traditional methods

**Genetic Algorithm Optimization:**
- Identified most relevant extracted features
- Evolved feature combinations for optimal performance
- Reduced feature space while maintaining predictive power

### 2. Model Architecture

**Ensemble Classifiers:**
| Model | Accuracy | F1-Score | Specificity |
|-------|----------|----------|-------------|
| **XGBoost** | **98.5%** | **0.985** | **0.984** |
| Random Forest | 97.8% | 0.978 | 0.976 |
| Gradient Boosting | 97.2% | 0.972 | 0.970 |

**Self-Attention Transformer Model:**
- Built with TensorFlow and Scikit-learn
- Multi-head attention mechanism for feature interactions
- **20% accuracy improvement** over traditional ML baseline
- Enhanced feature importance visualization

### 3. Validation Strategy

- **10-fold Cross-Validation** for robust performance estimates
- Stratified sampling to handle class imbalance
- Synthetic data generation to overcome small dataset limitations
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Specificity, AUC-ROC)

## Results & Comparison

**Performance Metrics:**
- Training Accuracy: **98.46%**
- Testing Accuracy: **98.55%**
- F1-Score: **0.985**
- Specificity: **0.984**
- AUC-ROC: **0.992**

**Improvement Over Baseline:**
- Traditional ML (without feature engineering): ~78% accuracy
- With engineered features: **98.5% accuracy**
- **Net improvement: +20%**

**Comparison with Literature:**
The proposed approach outperformed existing research methods across most evaluation metrics, demonstrating the effectiveness of combining DL-based feature extraction with genetic algorithm optimization.

## Key Innovations

1. **Mathematics-Driven Feature Engineering**: Applied neural extractors to learn complex feature representations automatically

2. **Genetic Algorithm Integration**: Optimized feature selection post-extraction, identifying most predictive combinations

3. **Synthetic Data Generation**: Addressed small dataset challenges through SMOTE and GAN-based augmentation (303 → 50K+ samples)

4. **Transformer Architecture**: First application of self-attention mechanisms to Cleveland dataset, achieving significant performance gains

5. **Comprehensive Evaluation**: Tested on both original and augmented datasets with consistent superior performance


## Research Contribution

This work demonstrates that:
1. Deep learning feature extraction significantly enhances traditional medical datasets
2. Genetic algorithms effectively refine extracted feature sets
3. Synthetic data generation enables robust training on small medical datasets
4. Transformer architectures can improve medical diagnosis predictions
5. Combined approach achieves near-perfect classification accuracy


---


