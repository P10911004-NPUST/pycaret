import os
import numpy as np
import pandas as pd
from datetime import datetime

from pycaret.regression import *

today = datetime.now()
year, month, day = today.year, today.month, today.day
rawdata = pd.read_csv('./datasets/insurance.csv')

for i in rawdata.region.unique():
    rawdata = rawdata.assign(**{i: np.where(rawdata.region == i, 'yes', 'no')})

rawdata = rawdata.drop(columns='region')
print(rawdata.dtypes)

y = 'charges'
num_x = ['age', 'bmi']
ctg_x = ['sex', 'children', 'smoker', 'southwest', 'southeast', 'northwest', 'northeast']

rawdata.to_csv("./datasets/insurance-reshape.csv", index=False)

exp = setup(data=rawdata, target=y,
            transformation=True, normalize=True, train_size=0.8,
            categorical_features=ctg_x,
            numeric_features=num_x,
            use_gpu=False)

best = compare_models()
internal_results = pull(best)
selected_models = [i for i in internal_results.index][:1]
print(f"Selected {len(selected_models)} models: ", selected_models)

for i in selected_models:
    if i == 'dummy':
        next
    print(f"Training {i}...")
    mod = create_model(i)
    mod_bagging = ensemble_model(mod)

    save_model(mod_bagging, f"models/reg_{i}_{year}-{month}-{day}")

    holdout_pred = predict_model(mod_bagging)
    holdout_score = pull()
    mape = np.float64(holdout_score['MAPE'])

    with open(f"models/mape_{i}_{year}-{month}-{day}.txt", "w") as f:
        f.write(f"{mape}")

