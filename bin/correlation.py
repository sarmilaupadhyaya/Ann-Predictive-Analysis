import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

load_data = pd.read_csv("/var/www/labor_productivity_prediction/data/productivity_data.csv")


def normailize(self, value, mean, std):
    train = (value - mean) / std
    return train

load_data["Labor percent"] = load_data["Labor percent"].apply(lambda x: float(x) / 100)
load_data = load_data.dropna()
labels = np.expand_dims(load_data["Actual Productivity (m3/hr)"].as_matrix().astype(np.float32), axis=1)
del load_data["Actual Productivity (m3/hr)"]
(load_data.corr()).to_csv("/var/www/labor_productivity_prediction/data/correlation.csv")
