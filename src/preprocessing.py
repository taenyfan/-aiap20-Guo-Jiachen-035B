import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def clean_data(df):
    df['Age'] = df['Age'].astype(str).str.replace(' years', '', regex=False).astype(float)
    df['Housing Loan'] = df['Housing Loan'].fillna('Unknown')
    df['Personal Loan'] = df['Personal Loan'].fillna(df['Personal Loan'].mode()[0])

    # 🔧 Convert categorical columns to 'category' dtype
    categorical_cols = ['Client ID', 'Occupation', 'Marital Status', 'Education Level',
                        'Credit Default', 'Housing Loan', 'Personal Loan',
                        'Contact Method', 'Subscription Status']

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

def feature_engineering(df):
    # Add interaction features
    if 'Age' in df.columns and 'Education Level' in df.columns:
        df['Age_Education'] = df['Age'] * df['Education Level'].cat.codes
    
    # Add polynomial features for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[f'{col}_squared'] = df[col] ** 2
    
    return df

def select_features(X, y):
    # Use Random Forest for feature selection
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        max_features=20
    )
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

def preprocess_data(df):
    df = clean_data(df)
    
    # Separate features and target
    X = df.drop(['Client ID', 'Subscription Status'], axis=1)
    y = df['Subscription Status'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Get categorical and numeric columns
    categorical_cols = ['Occupation', 'Marital Status', 'Education Level',
                       'Credit Default', 'Housing Loan', 'Personal Loan',
                       'Contact Method']
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Advanced feature engineering
    # 1. Create interaction features
    if 'Age' in numeric_cols:
        for col in categorical_cols:
            if col in X.columns:
                X[f'Age_{col}'] = X['Age'] * X[col].cat.codes
                X[f'Age_{col}_squared'] = (X['Age'] * X[col].cat.codes) ** 2
    
    # 2. Create polynomial features for numeric columns
    for col in numeric_cols:
        X[f'{col}_squared'] = X[col] ** 2
        X[f'{col}_cubed'] = X[col] ** 3
        X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
    
    # 3. Create ratio features
    if 'Age' in numeric_cols and 'Education Level' in X.columns:
        X['Age_Education_Ratio'] = X['Age'] / (X['Education Level'].cat.codes + 1)
        X['Age_Education_Product'] = X['Age'] * (X['Education Level'].cat.codes + 1)
    
    # 4. Create binned features
    if 'Age' in numeric_cols:
        X['Age_Bin'] = pd.qcut(X['Age'], q=5, labels=False)
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Handle any remaining categorical columns
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = X_encoded[col].astype('category')
    
    # Apply power transformation to numeric features
    numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        power_transformer = PowerTransformer(method='yeo-johnson')
        X_encoded[numeric_cols] = power_transformer.fit_transform(X_encoded[numeric_cols])
    
    # Handle missing and infinite values
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
    X_encoded = X_encoded.fillna(X_encoded.mean())
    
    # Feature selection using mutual information
    mi_scores = mutual_info_classif(X_encoded, y)
    mi_scores = pd.Series(mi_scores, index=X_encoded.columns)
    selected_features = mi_scores.nlargest(30).index
    X_selected = X_encoded[selected_features]
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_selected)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test
