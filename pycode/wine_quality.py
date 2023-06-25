import os
import numpy as np
import pandas as pd

from pycaret.classification import *

wine_df = pd.read_csv('./datasets/winequality-red.csv')
row, col = wine_df.shape

wine_df.quality = np.where(wine_df.quality >= 6, 'Good', 'Bad')
wine_df.dtypes

exp_clf01 = setup(data=wine_df, target='quality', transformation=True, normalize=True)
best = compare_models(sort='F1')
internal_results = pull(best)
selected_models = [i for i in internal_results.index][:10]

print(f"Selected {len(selected_models)} models: ", selected_models)

for i in selected_models:
    if i == 'dummy':
        next
    mod = create_model(i)
    mod_bagging = ensemble_model(mod, method='Bagging')
    save_model(mod_bagging, f"models/{i}")

for i in models().index:
    print(i)