from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

def get_models():
    # Calculate class weights
    class_weights = {0: 1, 1: 9}  # Adjust based on your actual class ratio
    
    # Base models
    base_models = {
        'logistic_regression': CalibratedClassifierCV(
            LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0),
            cv=5
        ),
        'random_forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            min_samples_leaf=10,
            max_depth=10,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'xgboost': XGBClassifier(
            scale_pos_weight=9,
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42
        ),
        'catboost': CatBoostClassifier(
            class_weights=[1, 9],
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        ),
        'svm': SVC(
            class_weight='balanced',
            probability=True,
            kernel='rbf',
            random_state=42
        ),
        'neural_net': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
    }
    
    # Create a voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', base_models['random_forest']),
            ('xgb', base_models['xgboost']),
            ('cat', base_models['catboost'])
        ],
        voting='soft'
    )
    
    # Create a stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', base_models['random_forest']),
            ('xgb', base_models['xgboost']),
            ('cat', base_models['catboost'])
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Add ensemble models to the dictionary
    base_models['voting'] = voting_clf
    base_models['stacking'] = stacking_clf
    
    return base_models
