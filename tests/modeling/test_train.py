from __future__ import annotations

from sklearn.metrics import mean_squared_error

from biodiv.features import get_features
from biodiv.modeling.train import XGBoostTrainer


def test_xgboosttrainer_random_state():
    data = get_features(compute_variables=False)
    trainer = XGBoostTrainer(data, scoring=mean_squared_error)
    best_hyperparameters_1 = trainer.train(
        y='D', max_evals=5, show_plots=False, method='cv'
    ).get_params(deep=False)
    best_hyperparameters_2 = trainer.train(
        y='D', max_evals=5, show_plots=False, method='cv'
    ).get_params(deep=False)
    for k in best_hyperparameters_1:
        if k == 'steps':
            continue

        assert best_hyperparameters_1[k] == best_hyperparameters_2[k]
