from pycaret.datasets import get_data
from pycaret.classification import *

diabetes = get_data('diabetes')
diabetes.head()

exp1 = setup(diabetes, target='Class variable', session_id=711)
# exp1.compare_models(sort='F1')

dt = create_model('dt')
dt_bagging = ensemble_model(dt, method='Bagging')
dt_boosting = ensemble_model(dt, method="Boosting")
dt_holdout_test = predict_model(dt_bagging)

save_model(dt, model_name='models/dt')
save_model(dt_bagging, model_name='models/dt_bagging')