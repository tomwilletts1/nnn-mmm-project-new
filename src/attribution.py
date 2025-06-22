import numpy as np


def simulate_zero_out(model, X_full, channel_indices):
    base_pred = model.predict(X_full)
    attributions = {}

    for name, idx in channel_indices.items():
        X_copy = X_full.copy()
        X_copy[:, idx] = 0
        pred_wo = model.predict(X_copy)
        delta = base_pred - pred_wo
        attributions[name] = delta.mean()

    return attributions
