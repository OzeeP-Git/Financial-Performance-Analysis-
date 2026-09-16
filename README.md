# Financial-Performance-Analysis-```markdown
# Financial Performance Index Prediction with Regularized Linear Models

## Project Overview
This project explores the challenges of predicting a Financial Performance Index (FPI) using a dataset characterized by multicollinearity, scale mismatches, and significant economic noise. It demonstrates how standard Ordinary Least Squares (OLS) regression fails under such conditions and how regularization techniques (Ridge, Lasso, ElasticNet) can stabilize the model and improve its robustness for real-world financial applications.

## Dataset
The dataset `financial_performance_multicollinear.csv` contains 14 columns, including various economic signals and a target variable, `financial_performance_index`. Key characteristics include:
- **Multicollinearity:** Highly correlated features (e.g., `rev_signal_1`, `rev_redundant_1`).
- **Scale Mismatch:** Features with vastly different scales (e.g., `scale_mismatch_1`, `scale_mismatch_2`).
- **Noise:** Presence of `weak_noise` and `pure_noise` features impacting model stability.

## Project Tasks & Analysis

### Task 1: Initial Data Analysis & Multicollinearity Assessment
- Loaded and inspected the dataset.
- Calculated means and variances of features.
- Computed the correlation matrix to identify relationships between features.
- Determined the condition number of `(X^T X)` to assess multicollinearity, revealing severe instability.

### Task 2: Target Variable Distribution Analysis
- Visualized the distribution of `financial_performance_index` using a histogram and Q-Q plot.
- Confirmed a nearly perfect Gaussian (Normal) distribution, indicating additive white noise and no extreme outliers that would necessitate non-MSE loss functions.

### Task 3: Baseline OLS Model Performance & Interpretation
- Split data into training and testing sets.
- Trained a standard Linear Regression (OLS) model.
- Evaluated performance using RMSE, MAE, and R².
- Analyzed model coefficients, revealing:
    - **Scale Mismatch Instability:** Extremely large coefficients for minor features (e.g., `scale_mismatch_2`).
    - **Overfitting to Noise:** Higher importance assigned to `pure_noise_2` than to true revenue signals.
    - **Error Distribution:** Large gap between RMSE and MAE, indicating severe outliers.
    - **Multicollinearity Impact:** Unstable and unreliable coefficient estimates due to highly correlated features.

### Task 4: Bias-Variance Trade-off Analysis
- Compared training and test RMSE to diagnose overfitting/underfitting.
- Identified that the baseline OLS model exhibits **high variance** and is **structurally overfitting** to statistical noise rather than memorizing training data.
- Concluded the need to introduce bias through regularization to reduce variance and improve stability.

### Task 5: Regularized Model Training & Comparison
- Standardized features, a mandatory step for regularization.
- Trained Ridge, Lasso, and ElasticNet regression models with default alpha values.
- Compared their RMSE and the number of non-zero coefficients.
- Observed that while RMSE values were similar across models, Lasso performed automatic feature selection by forcing one coefficient to zero.

### Task 6: Comprehensive Model Comparison & Final Selection
- Consolidated performance metrics (RMSE, MAE, R², Non-Zero Coeffs) for all models (OLS, Ridge, Lasso, ElasticNet).
- **Metric Selection Rationale:**
    - **RMSE:** Regularization improved stability by reducing extreme errors.
    - **MAE:** Confirmed similar average prediction errors across regularized models, implying improved stability rather than fundamental signal change.
    - **R²:** Revealed a hard upper bound of ~0.55 explainable variance, indicating significant inherent noise.
- **Final Model Recommendation:** **Lasso Regression** was selected due to its optimal balance of accuracy, robustness, interpretability, and operational efficiency, primarily because it achieves similar predictive accuracy with fewer features by suppressing noisy/redundant predictors.

### Task 7: Final Decision Report
- A detailed report summarizing the findings, including:
    - Reasons for baseline OLS model failure (redundant features, exploding coefficients, noise obsession).
    - How regularization helped (standardization, weight penalization, automatic feature selection).
    - Bias-Variance trade-off analysis, highlighting the shift from high variance to a balanced model.
    - Justification for the final recommendation of Lasso Regression.

## How to Run the Notebook

### Prerequisites
- Python 3.x
- Google Colab environment (recommended) or a local Jupyter environment.

### Dependencies
Install the necessary Python libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Steps
1.  **Clone the repository** (if using locally):
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Upload Data:** Ensure `financial_performance_multicollinear.csv` is in the same directory as the notebook or upload it directly in Colab (as shown in the first cell).
3.  **Run Cells:** Execute all cells in the notebook sequentially. Each cell builds upon the previous one, performing data loading, analysis, model training, and evaluation.

## Conclusion
This project provides a robust framework for handling noisy and multicollinear financial data, emphasizing the importance of regularization techniques for building stable and reliable predictive models in complex economic environments.

## Contact
For questions or collaboration, please reach out via [Your GitHub Profile/LinkedIn/Email].

## License
This project is licensed under the [e.g., MIT License] - see the LICENSE.md file for details.
```
