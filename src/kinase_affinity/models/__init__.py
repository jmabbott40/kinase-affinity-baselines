"""Baseline and advanced model implementations.

All models implement a common interface:
    - fit(X_train, y_train) — train the model
    - predict(X) — point predictions
    - predict_with_uncertainty(X) — (mean, std) tuple
    - save(path) / load(path) — model persistence
"""
