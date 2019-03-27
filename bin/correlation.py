import pandas as pd

df = pd.read_csv("../data/productivity_data.csv")
y_train = df[["Actual Productivity (m3/hr)"]]


def normailize(value, mean, std):
    train = (value - mean) / std
    return train


# for column in list(df.columns):
#     numpy_format = df[column].as_matrix()
#     mean, std = numpy_format.mean(), numpy_format.std()
#     df[column] = df[column].apply(lambda x: normailize(x, mean, std))
dd = df.corr()
dd.to_csv("../data/correlation.csv")

import pdb
pdb.set_trace()
