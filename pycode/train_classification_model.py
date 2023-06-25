import os
import numpy as np
import pandas as pd
from datetime import datetime

from pycaret.classification import *

today = datetime.now()
year, month, day = today.year, today.month, today.day
rawdata = pd.read_csv('./datasets/insurance-reshape.csv')

print(rawdata.dtypes)
num_x = ['age', 'bmi', 'charges']
ctg_x = ['sex', 'children', 'southwest', 'southeast', 'northwest', 'northeast']

exp = setup(data=rawdata, target='smoker',
            transformation=True, normalize=True, train_size=0.8,
            categorical_features=ctg_x,
            numeric_features=num_x,
            use_gpu=False)

best = compare_models(sort='F1', round=2)
internal_results = pull(best)
selected_models = [i for i in internal_results.index][:1]
print(f"Selected {len(selected_models)} models: ", selected_models)

for i in selected_models:
    if i == 'dummy':
        next
    print(f"Training {i}...")
    mod = create_model(i)
    mod_bagging = ensemble_model(mod)

    save_model(mod_bagging, f"models/clf_{i}_{year}-{month}-{day}")

    holdout_pred = predict_model(mod_bagging)
    # prob_df = pd.DataFrame({
    #     "charges": holdout_pred["charges"],
    #     "label": holdout_pred["prediction_label"],
    #     'prob': holdout_pred["prediction_score"]
    # })
    holdout_score = pull()
    f1 = np.float64(holdout_score['F1'])

    with open(f"models/F1_{i}_{year}-{month}-{day}.txt", "w") as f:
        f.write(f"{f1}")

