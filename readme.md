# AIAP Batch 20 Technical Assessment

Full Name:
Email:

## Introduction

This submission is part of the AIAPⓇ Batch 20 Technical Assessment and addresses the challenge of predicting whether a client will subscribe to a term deposit, using historical marketing campaign data provided by AI-Vive-Banking. The project is designed to support more targeted and effective direct marketing efforts through data-driven insights and predictive modeling.

The assessment is divided into two key components:

- **Exploratory Data Analysis (EDA):** Conducted in a Jupyter notebook to explore the data, uncover patterns, and guide the feature engineering process.
- **Machine Learning Pipeline:** Implemented using modular Python scripts to ingest, preprocess, and model the data from a SQLite database, enabling evaluation across multiple classification algorithms.

The approach combines clarity, modularity, and explainability — with particular focus on aligning model design with business goals, ensuring reproducibility, and maintaining flexibility for experimentation. Key analytical decisions are informed by the EDA findings, with emphasis on interpretability, model performance, and potential deployment considerations.


## Exploratory Data Analysis (EDA) Summary

The Exploratory Data Analysis was conducted using a Jupyter Notebook (`eda.ipynb`) to identify key patterns and insights from the dataset provided by AI-Vive-Banking. The dataset includes client demographic information and marketing campaign outcomes, with the objective of predicting term deposit subscription.

### Key Findings

- **Data Quality & Cleaning:**
  - The `Age` column contained textual suffixes (“ years”) which were cleaned and converted to numeric.
  - The `Housing Loan` column had ~60% missing values, which were replaced with a new `"Unknown"` category.
  - The `Personal Loan` column had ~10% missing values, which were imputed using the mode.

- **Distribution Analysis:**
  - `Age` was left-skewed, concentrated between 20–60 years.
  - `Campaign Calls` had a long right tail, indicating that most clients received fewer than five contacts.
  - `Previous Contact Days` had a high concentration at 999, which likely indicates "no previous contact".

- **Correlation Analysis:**
  - No strong linear relationships among numerical features.
  - `Previous Contact Days` had the highest correlation with subscription (ρ ≈ -0.325), suggesting **recency of contact** is a strong indicator of conversion likelihood.

- **Categorical Variables:**
  - Chi-square tests showed statistically significant associations between subscription and variables like **Occupation**, **Marital Status**, **Education Level**, **Credit Default**, and **Contact Method**.
  - **Housing Loan** and **Personal Loan** did not show significant relationships (p > 0.05), though interactions may still be explored in modeling.

- **Class Imbalance:**
  - The target variable (`Subscription Status`) is heavily imbalanced: approximately 89% of clients did not subscribe.
  - This will require special handling in the modeling pipeline (e.g., stratified sampling, class weighting, or resampling).

These findings directly informed the design of the machine learning pipeline, particularly in feature selection, preprocessing strategy, and the choice of evaluation metrics suitable for imbalanced classification tasks.


## Folder Structure

The repository follows a modular structure to support reusability and clarity across the machine learning pipeline:


├── bmarket.db # SQLite database (excluded from GitHub as instructed)
├── eda.ipynb # Jupyter notebook containing Task 1: EDA
├── readme.md # Project documentation
├── requirements.txt # Python dependencies for pipeline execution
├── run.sh # Bash script to execute the machine learning pipeline
└── src/ # Python scripts implementing the pipeline
├── data_loader.py # Loads and queries data from the SQLite database
├── ensemble.py # Logic for combining model predictions (if applicable)
├── evaluate.py # Performance evaluation of trained models
├── main.py # Entry point that ties all pipeline components together
├── models.py # Defines and configures machine learning models
└── preprocessing.py # Preprocessing logic and feature engineering steps

### Execution Instructions

1. **Running the Pipeline**
   - The pipeline can be executed using the provided `run.sh` script:
   ```bash
   ./run.sh
   ```
   - Alternatively, you can run it directly with Python:
   ```bash
   python src/main.py
   ```

2. **Pipeline Flow**
   The pipeline follows these steps:
   - Downloads the dataset from the specified URL
   - Loads and preprocesses the data
   - Trains multiple base models
   - Trains ensemble models
   - Evaluates all models and compares their performance

### Parameter Modification

You can modify several parameters in the codebase:

1. **Model Parameters** (in `src/models.py`):
   - Class weights: Currently set to `{0: 1, 1: 9}` for handling class imbalance
   - Individual model parameters:
     - Random Forest: `n_estimators=200`, `min_samples_leaf=10`, `max_depth=10`
     - Gradient Boosting: `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`
     - XGBoost: `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`
     - CatBoost: `iterations=200`, `learning_rate=0.1`, `depth=6`
     - Neural Network: `hidden_layer_sizes=(100, 50)`, `max_iter=1000`

2. **Ensemble Configuration**:
   - Voting Classifier: Currently uses Random Forest, XGBoost, and CatBoost
   - Stacking Classifier: Uses the same base models with Logistic Regression as the final estimator

3. **Evaluation Metrics**:
   - The pipeline evaluates models using ROC AUC score and other metrics defined in the evaluation module

### Important Notes

1. The pipeline automatically downloads the dataset from the specified URL
2. The data is automatically preprocessed before model training
3. Results are printed to the console, showing:
   - Individual model performance
   - Ensemble model performance
   - Final comparison of all models ranked by ROC AUC



## Model Selection Rationale

### Base Models

1. **Logistic Regression**
   - **Why**: Despite its simplicity, logistic regression provides a strong baseline and interpretable results
   - **Advantages**:
     - Clear probability estimates
     - Feature importance through coefficients
     - Fast training and prediction
   - **Configuration**: Calibrated with 5-fold CV to improve probability estimates

2. **Random Forest**
   - **Why**: Excellent for handling non-linear relationships and feature interactions
   - **Advantages**:
     - Robust to outliers and noise
     - Handles mixed feature types well
     - Provides feature importance
     - Less prone to overfitting than single trees
   - **Configuration**: Balanced class weights and controlled depth to prevent overfitting

3. **Gradient Boosting**
   - **Why**: Strong performance on imbalanced datasets through sequential learning
   - **Advantages**:
     - High predictive power
     - Handles complex patterns
     - Good with imbalanced data
   - **Configuration**: Moderate learning rate and depth for stable learning

4. **XGBoost**
   - **Why**: State-of-the-art performance on structured/tabular data
   - **Advantages**:
     - Built-in handling of missing values
     - Regularization to prevent overfitting
     - Efficient implementation
   - **Configuration**: Scale positive weight to handle class imbalance

5. **CatBoost**
   - **Why**: Excellent handling of categorical features without manual encoding
   - **Advantages**:
     - Native handling of categorical variables
     - Robust to overfitting
     - Good with imbalanced data
   - **Configuration**: Class weights and moderate depth for balanced performance

6. **SVM**
   - **Why**: Effective in high-dimensional spaces with clear margin of separation
   - **Advantages**:
     - Works well with limited training data
     - Robust to overfitting
     - Flexible kernel options
   - **Configuration**: RBF kernel for non-linear relationships

7. **Neural Network**
   - **Why**: Capture complex non-linear patterns in the data
   - **Advantages**:
     - Can learn complex patterns
     - Works well with large datasets
     - Flexible architecture
   - **Configuration**: Two hidden layers for balanced complexity

### Ensemble Methods

1. **Voting Classifier**
   - **Why**: Combine predictions from multiple models to reduce variance
   - **Advantages**:
     - Reduces overfitting
     - More robust predictions
     - Can capture different aspects of the data
   - **Configuration**: Soft voting for probability estimates

2. **Stacking Classifier**
   - **Why**: Learn optimal combination of base model predictions
   - **Advantages**:
     - Can learn complex interactions between models
     - Often outperforms individual models
     - More sophisticated than simple voting
   - **Configuration**: Logistic Regression as final estimator for interpretability

### Model Selection Strategy

The model selection strategy follows these principles:

1. **Diversity**: Models with different learning approaches to capture various aspects of the data
2. **Interpretability**: Mix of interpretable (Logistic Regression) and powerful (XGBoost) models
3. **Robustness**: Ensemble methods to reduce variance and improve stability
4. **Class Imbalance**: All models configured to handle the 89:11 class imbalance
5. **Computational Efficiency**: Balance between model complexity and training time

### Performance Considerations

- **Training Time**: Models are configured for reasonable training times while maintaining performance
- **Memory Usage**: Parameters tuned to prevent excessive memory consumption
- **Prediction Speed**: Important for potential deployment scenarios
- **Model Complexity**: Balanced to prevent overfitting while capturing important patterns

## Model Evaluation

To evaluate model performance, I used a combination of metrics suited for **imbalanced binary classification**, where the positive class ("subscription = yes") represents only ~11% of the data. The following metrics were prioritized:

- **Precision:** How many predicted positive cases were correct.
- **Recall:** How many actual positive cases were correctly identified.
- **F1-Score:** Harmonic mean of precision and recall, balancing both concerns.
- **ROC AUC:** Captures the model's ability to discriminate across all thresholds, regardless of imbalance.
- **Accuracy** was tracked but not prioritized due to its misleading nature in imbalanced settings.

### Observations

- While most models achieved high accuracy (above 85%), this was driven by the majority class. 
- **CatBoost** achieved the highest ROC AUC (0.6342), suggesting it had the best discrimination power for identifying likely subscribers.
- Models like **SVM**, **Random Forest**, and **Bagging** showed slightly lower ROC AUC but generally more balanced precision and recall.
- **Recall values across models were low**, indicating that many potential subscribers were still being missed — a common challenge in imbalanced classification.

Overall, models were able to detect a portion of true subscribers, but further tuning (e.g. threshold adjustment, oversampling, cost-sensitive learning) may be necessary to improve recall and overall effectiveness. For the business use case — identifying clients likely to subscribe — optimizing **recall** while maintaining acceptable precision is crucial for targeting without overextending campaign efforts.
