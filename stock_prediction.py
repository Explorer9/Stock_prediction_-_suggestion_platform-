import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import preprocessing
import numpy as np; np.random.seed(0)
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
style.use('ggplot')
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math
import yfinance as yf
import streamlit as st

st.write("Stock market App")
def get_input():
  company =st.sidebar.text_input("Explore a company","APPL ")
  days=st.sidebar.text_input("Predict for days ","30")
  return company,days


import yfinance as yf
com,days=get_input()

msft = yf.Ticker(com)


# get historical market data
#inp= msft.history(interval="2m")
inp= msft.history(period="1000d")
#inp=inp.drop(columns=['Dividends','Stock Splits'])
inp.reset_index(inplace=True)
inp=inp.rename(columns={'Datetime':'Date'})


inp["D"]=pd.Series([])
inp["K"]=pd.Series([])

out=inp

def computeRSI (data, time_window):
    diff = data.diff(1).dropna()       
    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

inp['RSI'] = computeRSI(inp['Close'], 5)
plt.plot(inp['Date'].values, inp['Close'].values,color='green')


fig=plt.figure(figsize=(20,5))
plt.plot(inp['Date'].values, inp['Close'].values,color='green')
#plt.xticks(('2019-10-28','2020-01-28','2020-02-28','2020-03-28','2020-04-28','2020-05-28'))
plt.title('Close')
st.pyplot(fig)

fig1=plt.figure(figsize=(20,5))
plt.title('RSI chart')
plt.plot(inp['RSI'].values,color='orange')

#plt.xticks(('2020-10-09','2020-10-13','2020-10-17','2020-10-21','2020-10-25','2020-11-01'))

plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.text(100,80,r'SELL',fontsize=25,color='red')
plt.text(100,20,r'BUY',fontsize=25,color='green')
plt.axhline(30, linestyle='--')

plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
st.pyplot(fig1)

inp['Suggest']=pd.Series([])
for i in range(5,len(inp)):
  if(inp['RSI'][i]>=80):
    inp['Suggest'][i]="SELL"
  elif(inp['RSI'][i]<20):
    inp['Suggest'][i]="BUY"
  else:
    inp['Suggest'][i]="KEEP"

inp=inp.set_axis(inp["Date"]).drop(["Date"],axis=1)
closing= inp['Close']
moving_average = closing.rolling(window=10).mean()

fig2=plt.figure(figsize=(10,8))
closing.plot(label=com)
moving_average.plot(label='moving average over 100 days')
plt.legend()
st.pyplot(fig2)
Aclose=inp[['Close']]

Days_forecast=int(days)
Aclose['Prediction'] = Aclose[['Close']].shift(-Days_forecast)
x = np.array(Aclose.drop(['Prediction'],1))
x = x[:-Days_forecast]
y = np.array(Aclose['Prediction'])
y = y[:-Days_forecast]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LinearRegression()
model=lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
x_forecast = np.array(Aclose.drop(['Prediction'],1))[-Days_forecast:]
lr_prediction = lr.predict(x_forecast)

fig4=plt.figure(figsize=(20,8))
plt.plot(y)
plt.title('Forecasted for'+days,fontsize=25)
plt.plot(range(len(y),len(y)+len(lr_prediction)),lr_prediction[::-1],color='orange')
plt.text(1220,85,r'FORECAST',color='orange',fontsize=20)
st.pyplot(fig4)
st.sidebar.text(inp['Suggest'][len(inp)-1])
st.sidebar.text(lr_prediction)

df['L14'] = df['Low'].rolling(window=14).min()
#Create the "H14" column in the DataFrame
df['H14'] = df['High'].rolling(window=14).max()
#Create the "%K" column in the DataFrame
df['%K'] = 100*((df['Close'] - df['L14']) / (df['H14'] - df['L14']) )
#Create the "%D" column in the DataFrame
df['%D'] = df['%K'].rolling(window=3).mean()

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10))
df['Close'].plot(ax=axes[0]); axes[0].set_title('Close')
df[['%K','%D']].plot(ax=axes[1]); axes[1].set_title('Oscillator')