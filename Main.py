import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pdb
import matplotlib.pyplot as plt
from numpy import argmax
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)

data = pd.read_csv('deepsolar_tract_withcounty.csv', encoding = "ISO-8859-1")
#data = data.drop('county', axis=1)
#data = data.drop('voting_2016_dem_win', axis=1)
#data = data.drop('voting_2012_dem_win', axis=1)
#data = data.drop('state', axis=1)
data = data.dropna()
data = data.drop(data[data.solar_panel_area_divided_by_area >= 600].index)
data["county"] = data["county"].map(str) + data["state"]
data = data.drop('state', axis=1)
county = data['county']
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(county)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
county = onehot_encoder.fit_transform(integer_encoded)

corr = data.corr(method='pearson')
print(corr)
pdb.set_trace()

x = data.drop('solar_panel_area_divided_by_area', axis=1)
y = data['solar_panel_area_divided_by_area']
x = x.drop('county', axis=1)
x = np.concatenate((x, county), axis=1)
x = np.array(x)
y = np.array(y)

#sns.boxplot(x=y)
#plt.show()
#pdb.set_trace()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

t0 = time.time()
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("decision tree")
print(time.time() - t0)
print(mean_absolute_error(y_test, y_predicted))


t0 = time.time()
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("random forest")
print(time.time() - t0)
print(mean_absolute_error(y_test, y_predicted))
'''
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
t0 = time.time()
model = LinearRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("linear regression")
print(time.time() - t0)
print(mean_absolute_error(y_test, y_predicted))
print(model.coef_)


t0 = time.time()
model = SGDRegressor(alpha=0.01)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("sgd regression")
print(time.time() - t0)
print(mean_absolute_error(y_test, y_predicted))
print(model.coef_)
'''
