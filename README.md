# Estimating Solvency Ratios Using Machine Learning

A machine learning approach to forecast solvency ratios for life insurance companies, enabling continuous monitoring between quarterly reporting cycles.

---

## Overview

Insurance companies must maintain solvency ratios above 100% to meet regulatory requirements. Traditional actuarial projections are computationally expensive, making continuous monitoring impractical. This project demonstrates that machine learning models can accurately predict solvency ratios from market parameters, enabling real-time risk management.

**Key Result**: XGBoost achieved 94.5% accuracy (R²) on test data and successfully tracked a dramatic decline from 290% to 84% in out-of-time backtesting.

---

## Problem

Calculating solvency ratios requires:
- Generating 1,000-5,000 stochastic economic scenarios
- Running asset-liability projections over 50-80 years  
- Aggregating multiple risk capital components

This process takes days, creating monitoring gaps between quarterly reports.

---

## Solution

Train ML models on synthetic data covering diverse market conditions, then use current market parameters for instant solvency estimates.

```
Solvency Ratio = Available Capital (EM) / Required Capital (SCR)
```

**Input**: 20 features (8 market parameters + 12 management rules)  
**Output**: Available capital, required capital, solvency ratio

---

## Results

### Model Performance

| Model | Test R² | RMSE | Notes |
|-------|---------|------|-------|
| **XGBoost** | 0.945 | 0.080 | Best overall |
| Neural Network | 0.941 | 0.083 | Comparable accuracy |
| Polynomial (deg 2) | ~0.850 | 0.110 | Struggles with extremes |
| Linear Models | 0.690 | 0.160-0.200 | Insufficient |

### Backtesting on Live Data (Q1-Q4)

| Quarter | Actual SR | XGBoost | Neural Net |
|---------|-----------|---------|------------|
| Q1 | 290% | 256% | 268% |
| Q2 | 226% | 218% | 253% |
| Q3 | 153% | 165% | 222% |
| Q4 | 84% | 106% | 143% |

**XGBoost RMSE: 0.21** | Neural Net RMSE: 0.49

XGBoost correctly predicted the approach to the critical 100% threshold, while neural networks underestimated the severity of the decline.

---

## Key Findings

### Most Important Features (SHAP Analysis)

1. **ZSK1** - Interest rate level (dominant driver)
2. **Vola4** - Interest rate volatility
3. **Vola6** - Real estate volatility  
4. **Verlust7/8** - Market value losses (stocks/real estate)

Management rules had secondary impact, confirming that short-term movements are primarily driven by external market conditions.

### Methodological Insights

- **Non-linearity matters**: Quadratic models improved RMSE by 40% vs linear
- **Direct vs indirect**: Direct SR prediction more stable than predicting EM/SCR separately
- **Multi-output preferred**: Simultaneous EM + SCR prediction provides better interpretability
- **Temporal stability**: XGBoost generalizes better to out-of-time data than neural networks

---

## Dataset

- **Size**: 10,230 synthetic scenarios
- **Features**: 20 inputs (normalized to [0,1])
  - 8 continuous market parameters
  - 12 management rule flags
- **Targets**: Available capital, required capital, solvency ratio
- **Split**: 60% train, 20% validation, 20% test

---

## Models Explored

### Regression
Linear, Ridge, Lasso, Elastic Net, Polynomial (2nd/3rd degree)

### Tree-Based  
Random Forest, XGBoost (with hyperparameter tuning)

### Neural Networks
scikit-learn MLPRegressor, custom PyTorch architectures

---

## Use Case

### Risk Management Dashboard

**Real-time monitoring of**:
- Current solvency ratio estimate
- Available and required capital breakdown  
- Input parameter drift from baseline
- Prediction confidence assessment

**Benefits**:
- Daily solvency updates without full actuarial runs
- Early warning for regulatory threshold approaches
- What-if scenario analysis
- Supports regulatory "Use Test" requirement

---

## Limitations

- Models trained on single time point (requires multi-period validation)
- Assumes stable business structure (asset allocation, liability mix)
- Large parameter deviations may reduce accuracy
- Indicator only—not a replacement for regulatory calculations

---


## Future Work

- Validate temporal stability across multiple reporting periods
- Implement consistency penalty: L² norm (SR - EM/SCR)²
- Develop automated drift detection and retraining triggers
- Test ensemble methods combining XGBoost and neural networks

