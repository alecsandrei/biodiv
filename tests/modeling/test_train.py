from __future__ import annotations

from sklearn.metrics import mean_squared_error

from biodiv.features import get_features
from biodiv.modeling.train import XGBoostTrainer


def test_xgboosttrainer_random_state():
    data = get_features()
    trainer = XGBoostTrainer(data, scoring=mean_squared_error)
    best_hyperparameters_1 = trainer.train(
        y='D', max_evals=50, show_plots=False, method='cv'
    )
    best_hyperparameters_2 = trainer.train(
        y='D', max_evals=50, show_plots=False, method='cv'
    )
    assert best_hyperparameters_1 == best_hyperparameters_2
