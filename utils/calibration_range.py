from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


model_params_train = pd.read_csv('../data/calibration_params/Heston/Heston_params.csv').to_numpy()
scaler = StandardScaler()
model_params_train = scaler.fit_transform(model_params_train)

for i in range(model_params_train.shape[1]):
    print('max', np.max(model_params_train[:, i]), ' min', np.min(model_params_train[:, i]))