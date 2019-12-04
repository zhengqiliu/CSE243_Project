import pandas as pd
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 400)
pd.set_option('display.max_rows', 400)
data = pd.read_csv('deepsolar_tract_original.csv', encoding = "ISO-8859-1")
corr = data.corr(method='pearson')['solar_panel_area_divided_by_area'].sort_values(ascending=False)
print(corr)