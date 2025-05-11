import os
from data_loader import load_data
from preprocessing import preprocess_data
from models import get_models
from ensemble import get_ensemble_models
from evaluate import evaluate_model
import requests
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    url = 'https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db'
    db_filename = 'bmarket.db'

    response = requests.get(url)
    with open(db_filename, 'wb') as f:
        f.write(response.content)

    db_path = 'bmarket.db'
    df = load_data(db_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Get base models
    base_models = get_models()
    
    # Get ensemble models
    ensemble_models = get_ensemble_models()
    
    # Train and evaluate base models
    print("\n=== Base Models Performance ===")
    base_scores = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        base_scores[name] = metrics['roc_auc']
        print(f"\n{name.upper()} Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    # Train and evaluate ensemble models
    print("\n=== Ensemble Models Performance ===")
    ensemble_scores = {}
    for name, model in ensemble_models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        ensemble_scores[name] = metrics['roc_auc']
        print(f"\n{name.upper()} Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    # Compare all models
    print("\n=== Model Comparison ===")
    all_scores = {**base_scores, **ensemble_scores}
    sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nModels ranked by ROC AUC:")
    for name, score in sorted_models:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()
