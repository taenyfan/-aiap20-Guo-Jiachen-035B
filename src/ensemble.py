from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV

class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        weighted_probas = np.average(probas, weights=self.weights, axis=0)
        return weighted_probas
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def get_ensemble_models():
    # Base models with optimized parameters
    base_models = {
        'rf': RandomForestClassifier(
            n_estimators=500,  # Increased number of trees
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all available cores
        ),
        'xgb': XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,  # Reduced learning rate
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            scale_pos_weight=9,
            random_state=42,
            n_jobs=-1
        ),
        'gbm': GradientBoostingClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42
        ),
        'cat': CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            class_weights=[1, 9],
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
    }
    
    # 1. Simple Voting Ensemble with optimized weights
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', base_models['rf']),
            ('xgb', base_models['xgb']),
            ('gbm', base_models['gbm']),
            ('cat', base_models['cat'])
        ],
        voting='soft',
        weights=[1.2, 1.0, 0.8, 1.0]
    )
    
    # 2. Stacking Ensemble with cross-validation
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', base_models['rf']),
            ('xgb', base_models['xgb']),
            ('gbm', base_models['gbm']),
            ('cat', base_models['cat'])
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000,
            n_jobs=-1
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # 3. Bagging Ensemble with optimized parameters
    bagging_rf = BaggingClassifier(
        estimator=RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        ),
        n_estimators=20,  # Increased number of estimators
        max_samples=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Weighted Ensemble with optimized weights
    weighted_ensemble = WeightedEnsemble(
        models=[
            base_models['rf'],
            base_models['xgb'],
            base_models['gbm'],
            base_models['cat']
        ],
        weights=[1.2, 1.0, 0.8, 1.0]
    )
    
    # 5. Two-level Stacking with optimized parameters
    level1_models = [
        ('rf', base_models['rf']),
        ('xgb', base_models['xgb']),
        ('gbm', base_models['gbm']),
        ('cat', base_models['cat'])
    ]
    
    level2_models = [
        ('rf2', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=4,
            random_state=42
        )),
        ('xgb2', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            random_state=42
        ))
    ]
    
    # First level stacking with cross-validation
    first_level = StackingClassifier(
        estimators=level1_models,
        final_estimator=LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Second level stacking
    two_level_stacking = StackingClassifier(
        estimators=[
            ('first_level', first_level),
            *level2_models
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    return {
        'voting': voting_clf,
        'stacking': stacking_clf,
        'bagging': bagging_rf,
        'weighted': weighted_ensemble,
        'two_level_stacking': two_level_stacking
    } 