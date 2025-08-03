from __future__ import annotations

from sklearn.metrics import root_mean_squared_error

from biodiv.config import FIGURES_DIR, PROCESSED_DATA_DIR
from biodiv.features import get_features
from biodiv.modeling.predict import predict
from biodiv.modeling.train import XGBoostTrainer, negate
from biodiv.plots import EDA

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
