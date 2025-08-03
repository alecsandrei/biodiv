from __future__ import annotations

import collections.abc as c
import functools
import typing as t
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    make_scorer,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneGroupOut,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spatialkfold.clusters import spatial_kfold_clusters

from biodiv.config import FIGURES_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from biodiv.dataset import GeomorphometricVariable
from biodiv.features import get_features
from biodiv.modeling.predict import predict
from biodiv.plots import EDA, RegressorDiagnostic

if t.TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd


@dataclass
class Trainer:
    data: gpd.GeoDataFrame
    scoring: c.Callable | str

    @cached_property
    def predictors(self) -> list[str]:
        return self.data.columns.intersection(
            GeomorphometricVariable._value2member_map_
        )

    @property
    def cv(self) -> KFold:
        return KFold(5, shuffle=True, random_state=RANDOM_SEED)

    def spatial_cv(self, subset: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        subset['id'] = range(subset.shape[0])
        return spatial_kfold_clusters(
            subset,
            name='id',
            nfolds=5,
            algorithm='kmeans',
            random_state=RANDOM_SEED,
        )

    def diagnose(
        self, true: np.ndarray, preds: np.ndarray, name: str, show: bool = False
    ) -> None:
        diagnostic = RegressorDiagnostic(true, preds, name)
        diagnostic.log_results()
        diagnostic.plot(
            show=show,
            out_file=FIGURES_DIR / f'{diagnostic.name}_predictions.png',
        )


@dataclass
class RandomForestTrainer(Trainer):
    @cached_property
    def param_grid(self) -> dict[str, t.Any]:
        return {
            'model__n_estimators': np.arange(100, 501, 100),
            'pca__n_components': np.arange(1, len(self.predictors)),
        }

    @property
    def model(self) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=RANDOM_SEED)

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA(random_state=RANDOM_SEED)),
                ('model', self.model),
            ]
        )

    def train(self, y: str):
        train, test = train_test_split(self.data, test_size=0.2, train_size=0.8)
        spatial_folds = self.spatial_cv(train).folds.values.ravel()
        group_cvs = LeaveOneGroupOut()
        model = GridSearchCV(
            self.pipeline,
            self.param_grid,
            scoring=self.scoring,
            verbose=3,
            cv=group_cvs,
            n_jobs=-1,
        ).fit(train[self.predictors], train[y], groups=spatial_folds)
        best = model.best_estimator_
        test_preds = best.predict(test[self.predictors])
        train_preds = best.predict(train[self.predictors])
        self.diagnose(train[y].values, train_preds, 'rf_training')
        self.diagnose(test[y].values, test_preds, 'rf_testing')
        return (model, train_preds, test_preds)


class XGBoostHyperoptCV(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_evals: int,
        cv,
        scoring: c.Callable,
        random_seed: int | None = None,
    ):
        super().__init__()
        self.max_evals = max_evals
        self.cv = cv
        self.scoring = scoring
        self.random_seed = random_seed

    @property
    def model_(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            device='cpu', seed=self.random_seed, importance_type='weight'
        )

    @staticmethod
    def importance_getter(estimator: XGBoostHyperoptCV):
        return estimator.best_model_.feature_importances_

    @cached_property
    def param_grid_(self):
        return {
            'eta': hp.loguniform('eta', np.log(1e-5), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(2.0)),
            'reg_lambda': hp.loguniform(
                'reg_lambda', np.log(1e-6), np.log(2.0)
            ),
            'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(64.0)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'max_depth': scope.int(
                hp.qloguniform('max_depth', np.log(2), np.log(8), 1)
            ),
            'n_estimators': scope.int(
                hp.qloguniform('n_estimators', np.log(10), np.log(100), 1)
            ),
        }

    def fit_cv_(
        self, params: dict[str, t.Any], X: pd.DataFrame, y: pd.Series
    ) -> dict[str, t.Any]:
        score = cross_val_score(
            self.model_.set_params(**params),
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

    def fit(self, X, y):
        trials = Trials()
        best_hyperparameters = fmin(
            fn=partial(
                self.fit_cv_,
                X=X,
                y=y,
            ),
            space=self.param_grid_,
            verbose=False,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_seed),
        )
        self.trials_ = trials
        self.best_hyperparameters_ = best_hyperparameters
        to_convert_to_int = ('max_depth', 'n_estimators')
        for convertable in to_convert_to_int:
            self.best_hyperparameters_[convertable] = int(
                self.best_hyperparameters_[convertable]
            )

        self.best_model_ = self.model_.set_params(
            **self.best_hyperparameters_
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.best_model_.predict(X)


@dataclass
class XGBoostTrainer(Trainer):
    @cached_property
    def param_grid_cv(self):
        return {
            'model__eta': hp.loguniform(
                'model__eta', np.log(1e-6), np.log(1.0)
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
                'model__n_estimators', np.log(10), np.log(1000), 1
            ),
            'pca__n_components': hp.quniform(
                'pca__n_components', 1, len(self.predictors), 1
            ),
        }

    @cached_property
    def param_grid_rfecv(self):
        return {
            'eta': hp.loguniform('eta', np.log(1e-3), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(2.0)),
            'reg_lambda': hp.loguniform(
                'reg_lambda', np.log(1e-6), np.log(2.0)
            ),
            'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(64.0)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'max_depth': hp.loguniform('max_depth', np.log(2), np.log(8)),
            'n_estimators': hp.qloguniform(
                'n_estimators', np.log(10), np.log(10000), 100
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
            step=3,
        ).fit(X, y)
        score = fit.cv_results_['mean_test_score'].max()
        return {
            'loss': -score,
            'status': STATUS_OK,
            'predictors': X.columns[fit.support_],
        }

    def fit_cv(
        self,
        params: dict[str, t.Any],
        X: gpd.GeoDataFrame,
        y: pd.Series,
        predictors: list[str],
    ) -> dict[str, t.Any]:
        spatial_folds = self.spatial_cv(X).folds.values.ravel()
        group_cvs = LeaveOneGroupOut()
        score = cross_val_score(
            self.get_pipeline(params),
            X[predictors],
            y,
            groups=spatial_folds,
            cv=group_cvs,
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
                self.fit_cv, X=train, y=train[y], predictors=self.predictors
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
        self.diagnose(train[y].values, train_preds, 'xgboost_cv_training')
        self.diagnose(test[y].values, test_preds, 'xgboost_cv_testing')
        return model

    def train_rfecv(
        self, y: str, max_evals: int = 100, show_plots: bool = False
    ):
        train, test = train_test_split(
            self.data, test_size=0.2, train_size=0.8, random_state=RANDOM_SEED
        )
        spatial_folds = self.spatial_cv(train).folds.values.ravel()
        group_cvs = LeaveOneGroupOut()
        assert callable(self.scoring)
        model = XGBoostHyperoptCV(
            max_evals=max_evals,
            cv=self.cv,
            scoring=self.scoring,
            random_seed=RANDOM_SEED,
        )

        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=group_cvs,
            scoring=make_scorer(self.scoring),
            importance_getter=model.importance_getter,
            verbose=3,
        ).fit(train[self.predictors], train[y], groups=spatial_folds)
        predictors = [
            predictor
            for predictor, mask in zip(self.predictors, rfecv.support_)
            if mask
        ]
        model = rfecv.estimator_
        test_preds = model.predict(test[predictors])
        train_preds = model.predict(train[predictors])
        self.diagnose(train[y].values, train_preds, 'xgboost_rfecv_training')
        self.diagnose(test[y].values, test_preds, 'xgboost_rfecv_testing')
        return model

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


def negate(metric_func: c.Callable):
    @functools.wraps(metric_func)
    def wrapper(*args, **kwargs):
        return -metric_func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    data = get_features().rename(columns={'D': "Margalef's richness index"})
    EDA(data["Margalef's richness index"]).plot(
        show=False, out_file=FIGURES_DIR / 'margalef_eda.png'
    )

    trainer = XGBoostTrainer(data, scoring=negate(root_mean_squared_error))
    model = trainer.train(
        y="Margalef's richness index",
        max_evals=100,
        show_plots=False,
        method='cv',
    )

    out_file = PROCESSED_DATA_DIR / 'margalef.tif'
    predict(model, out_file)

    # For RF use:
    # RandomForestTrainer(data, scoring='neg_root_mean_squared_error').train('D')
