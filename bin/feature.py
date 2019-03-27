import pandas as pd
# from sklearn.metrics import r2_score
# from rfpimp import permutation_importances
from matplotlib import pyplot as plt
ax = plt.gca()
#
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
#
# df = pd.read_csv("../data/productivity_data.csv")
# y_train = df[["Actual Productivity (m3/hr)"]]
#
#
#
# def normailize(value, mean, std):
#     train = (value - mean) / std
#     return train
#
#
# # for column in list(df.columns):
# #     numpy_format = df[column].as_matrix()
# #     mean, std = numpy_format.mean(), numpy_format.std()
# #     df[column] = df[column].apply(lambda x: normailize(x, mean, std))
# del df["Actual Productivity (m3/hr)"]
#
#
# train_data, validation_data, train_label, validation_label = train_test_split(df, y_train, \
#                                                                               train_size=0.9, test_size=0.1)
#
# import pdb
#
# pdb.set_trace()
#
# rf = RandomForestRegressor(n_estimators=100,
#                            n_jobs=-1,
#                            oob_score=True,
#                            bootstrap=True,
#                            random_state=42)
# rf.fit(train_data, train_label)
#
# print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(
#     rf.score(train_data, train_label),
#     rf.oob_score_,
#     rf.score(validation_data, validation_label)))
#
#
# def r2(rf, X_train, y_train):
#     return r2_score(y_train, rf.predict(X_train))
#
#
#
# perm_imp_rfpimp = permutation_importances(rf, train_data, train_label, r2)
# print(sum(perm_imp_rfpimp['Importance'].tolist()))

perm_imp_rfpimp = pd.read_csv("../data/feature.csv")
# perm_imp_rfpimp["feature"] = perm_imp_rfpimp.index
perm_imp_rfpimp.plot(kind='bar',x='Feature',y='Importance',ax=ax)
plt.show()
