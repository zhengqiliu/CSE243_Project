import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

data = pd.read_csv('deepsolar_tract_original.csv', encoding = "ISO-8859-1")
data = data.loc[:, ['solar_panel_area_divided_by_area', 'relative_humidity', 'lon', 'frost_days', 'electricity_consume_residential', 'heating_degree_days', 'voting_2016_gop_percentage', 'education_high_school_graduate_rate', 'voting_2012_gop_percentage', 'occupancy_vacant_rate', 'travel_time_less_than_10_rate', 'lat', 'heating_fuel_coal_coke_rate', 'heating_fuel_coal_coke', 'occupation_manufacturing_rate', 'race_white_rate', 'atmospheric_pressure', 'heating_fuel_other_rate', 'feedin_tariff', 'daily_solar_radiation', 'electricity_price_industrial', 'rebate', 'electricity_price_commercial', 'electricity_price_overall', 'housing_unit_median_gross_rent', 'incentive_residential_state_level', 'avg_electricity_retail_rate', 'incentive_nonresidential_state_level', 'heating_design_temperature', 'electricity_price_residential', 'race_asian_rate', 'earth_temperature', 'property_tax', 'voting_2016_dem_percentage', 'mortgage_with_rate', 'diversity', 'voting_2016_dem_win', 'air_temperature', 'race_two_more']]
data = data.dropna()
#data = data.drop(data[data.solar_panel_area_divided_by_area >= 65].index)
'''
data["county"] = data["county"].map(str) + data["state"]
data = data.drop('state', axis=1)
county = data['county']
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(county)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
county = onehot_encoder.fit_transform(integer_encoded)
'''

x = data.drop('solar_panel_area_divided_by_area', axis=1)
y = data['solar_panel_area_divided_by_area']
#x = x.drop('county', axis=1)
#x = np.concatenate((x, county), axis=1)
x = np.array(x)
y = list(y)
for i in range(len(y)):
    if y[i] <= 35:
        y[i] = 'A'
    elif 35 < y[i] <= 65:
        y[i] = 'B'
    elif y[i] <= 150:
        y[i] = "C"
    elif 150 < y[i] < 600:
        y[i] = 'D'
    else:
        y[i] = 'E'

y = np.array(y)
#print(y.mean())
#sns.boxplot(x=y)
#plt.show()
#pdb.set_trace()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

t0 = time.time()
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("decision tree")
print(time.time() - t0)
print(accuracy_score(y_test, y_predicted))


t0 = time.time()
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("random forest")
print(time.time() - t0)
print(accuracy_score(y_test, y_predicted))


#scaler = MinMaxScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
t0 = time.time()
model = LogisticRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("linear regression")
print(time.time() - t0)
print(accuracy_score(y_test, y_predicted))