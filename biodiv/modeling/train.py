from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biodiv.config import RANDOM_SEED
from biodiv.dataset import GeomorphometricVariable
from biodiv.features import get_features
from biodiv.plots import Diagnostic

if t.TYPE_CHECKING:
    import pandas as pd


@dataclass
class Trainer:
    data: pd.DataFrame
    scoring: c.Callable

    @cached_property
    def predictors(self) -> list[str]:
        return list(
            set(GeomorphometricVariable._value2member_map_).intersection(
                self.data.columns
            )
        )

    @property
    def cv(self) -> KFold:
        return KFold(10, shuffle=True, random_state=RANDOM_SEED)

    def log_results(self, true, pred, name: str) -> None:
        metrics = (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )
        for metric in metrics:
            result = metric(true, pred)
            logger.info(
                '%s of the model is %f for %s' % (metric.__name__, result, name)
            )


@dataclass
class RandomForestTrainer(Trainer):
    @cached_property
    def predictors(self) -> list[str]:
        return list(
            set(GeomorphometricVariable._value2member_map_).intersection(
                self.data.columns
            )
        )

    @cached_property
    def param_grid(self) -> dict[str, t.Any]:
        return {
            #'model__n_estimators': np.arange(100, 501, 100),
            'model__max_features': ['log2', 'sqrt'],
            'model__min_samples_leaf': np.arange(1, 5),
            'pca__n_components': np.arange(1, len(self.predictors)),
        }

    @cached_property
    def model(self) -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED)

    @cached_property
    def pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('model', self.model),
            ]
        )

    def train(self, y: str):
        train, test = train_test_split(self.data, test_size=0.2, train_size=0.8)
        model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            scoring=self.scoring,
            verbose=3,
            cv=self.cv,
        ).fit(train[self.predictors], train[y])
        best = model.best_estimator_
        test_preds = best.predict(test[self.predictors])
        train_preds = best.predict(train[self.predictors])
        self.log_results(test[y], test_preds, 'testing')
        self.log_results(train[y], train_preds, 'training')
        return (model, train_preds, test_preds)


@dataclass
class XGBoostTrainer(Trainer):
    @cached_property
    def param_grid_cv(self):
        return {
            'model__eta': hp.loguniform(
                'model__eta', np.log(1e-5), np.log(1.0)
            ),
            'model__reg_alpha': hp.loguniform(
                'model__reg_alpha', np.log(1e-6), np.log(2.0)
            ),
            'model__reg_lambda': hp.loguniform(
                'model__reg_lambda', np.log(1e-6), np.log(2.0)
            ),
            'model__gamma': hp.loguniform(
                'model__gamma', np.log(1e-6), np.log(64.0)
            ),
            'model__subsample': hp.quniform('model__subsample', 0.5, 1.0, 0.05),
            'model__colsample_bytree': hp.quniform(
                'model__colsample_bytree', 0.3, 1.0, 0.05
            ),
            'model__max_depth': hp.qloguniform(
                'model__max_depth', np.log(2), np.log(8), 1
            ),
            'model__n_estimators': hp.qloguniform(
                'model__n_estimators', np.log(10), np.log(1000), 10
            ),
            'pca__n_components': hp.quniform(
                'pca__n_components', 1, len(self.predictors), 1
            ),
        }

    @cached_property
    def param_grid_rfecv(self):
        return {
            'eta': hp.loguniform('eta', np.log(1e-5), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(2.0)),
            'reg_lambda': hp.loguniform(
                'reg_lambda', np.log(1e-6), np.log(2.0)
            ),
            'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(64.0)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'max_depth': hp.loguniform('max_depth', np.log(2), np.log(8)),
            'n_estimators': hp.qloguniform(
                'n_estimators', np.log(10), np.log(100), 1
            ),
        }

    @property
    def model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(device='cpu', seed=RANDOM_SEED)

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA(random_state=RANDOM_SEED)),
                ('model', self.model),
            ]
        )

    def get_pipeline(self, params: dict[str, t.Any]) -> Pipeline:
        params['pca__n_components'] = int(params['pca__n_components'])
        params['model__max_depth'] = int(params['model__max_depth'])
        params['model__n_estimators'] = int(params['model__n_estimators'])
        return self.pipeline.set_params(**params)

    def fit_rfecv(
        self, params: dict[str, t.Any], X: pd.DataFrame, y: pd.Series
    ) -> dict[str, t.Any]:
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        fit = RFECV(
            estimator=self.model.set_params(**params),
            cv=self.cv,
            scoring=make_scorer(self.scoring),
            step=1,
        ).fit(X, y)
        score = fit.cv_results_['mean_test_score'].max()
        return {
            'loss': -score,
            'status': STATUS_OK,
            'predictors': X.columns[fit.support_],
        }

    def fit_cv(
        self, params: dict[str, t.Any], X: pd.DataFrame, y: pd.Series
    ) -> dict[str, t.Any]:
        score = cross_val_score(
            self.get_pipeline(params),
            X,
            y,
            cv=self.cv,
            scoring=make_scorer(self.scoring),
            n_jobs=-1,
        ).mean()
        return {
            'loss': -score,
            'status': STATUS_OK,
        }

    def train_cv(
        self, y: str, max_evals: int | None = 100, show_plots: bool = False
    ):
        train, test = train_test_split(
            self.data, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
        )
        trials = Trials()
        best_hyperparameters = fmin(
            fn=partial(
                self.fit_cv,
                X=train[self.predictors],
                y=train[y],
            ),
            space=self.param_grid_cv,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED),
        )
        model = self.get_pipeline(best_hyperparameters).fit(
            train[self.predictors], train[y]
        )
        test_preds = model.predict(test[self.predictors])
        train_preds = model.predict(train[self.predictors])
        self.log_results(test[y], test_preds, 'testing')
        self.log_results(train[y], train_preds, 'training')
        Diagnostic(test[y].values, test_preds).plot(show=show_plots)
        return best_hyperparameters

    def train_rfecv(
        self, y: str, max_evals: int | None = 100, show_plots: bool = False
    ):
        train, test = train_test_split(
            self.data, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
        )
        trials = Trials()
        best_hyperparameters = fmin(
            fn=partial(
                self.fit_rfecv,
                X=train[self.predictors],
                y=train[y],
            ),
            space=self.param_grid_rfecv,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED),
        )
        predictors = trials.best_trial['result']['predictors']
        logger.info('Selected predictors: %s' % predictors)
        best_hyperparameters['max_depth'] = int(
            best_hyperparameters['max_depth']
        )
        best_hyperparameters['n_estimators'] = int(
            best_hyperparameters['n_estimators']
        )
        model = self.model.set_params(**best_hyperparameters).fit(
            train[predictors], train[y]
        )
        test_preds = model.predict(test[predictors])
        train_preds = model.predict(train[predictors])
        self.log_results(test[y], test_preds, 'testing')
        self.log_results(train[y], train_preds, 'training')
        Diagnostic(test[y].values, test_preds).plot(show=show_plots)
        return best_hyperparameters

    def train(
        self,
        y: str,
        max_evals: int,
        show_plots: bool = False,
        method: t.Literal['cv', 'rfecv'] = 'cv',
    ):
        if method == 'cv':
            result = self.train_cv(y, max_evals, show_plots)
        elif method == 'rfecv':
            result = self.train_rfecv(y, max_evals, show_plots)
        return result


def neg_mean_squared_error(*args, **kwargs):
    return -mean_squared_error(*args, **kwargs)


if __name__ == '__main__':
    data = get_features()
    trainer = XGBoostTrainer(data, scoring=neg_mean_squared_error)
    trained = trainer.train(y='D', max_evals=500, show_plots=True, method='cv')
