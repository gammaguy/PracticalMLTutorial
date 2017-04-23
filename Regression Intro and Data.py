import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low'] * 100.0
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change',"Adj. Volume"]]

print(df.head())

forcast_col = 'Adj. Close'

df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
print(df.tail())