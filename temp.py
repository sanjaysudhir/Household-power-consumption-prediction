import streamlit as st

#import plotly as py 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime
from sklearn import preprocessing
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



#import plotly.graph_objs as go
#from scipy import stats

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""

# Power Consumption Prediction

""")
st.text("")

#Import data file
#LSTM
df1 = pd.read_csv("household_power_consumption.txt",
                  sep = ";" ,na_values = ["nan","?"],
                  parse_dates = {"dt" : ["Date","Time"]},
                  infer_datetime_format = True,
                  index_col = "dt")
st.write(df1.head())
df1 = df1.fillna(df1.mean())
#st.write(df1.isna().sum())









df = pd.read_csv('file1.csv',parse_dates=['Date_Time'])
df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
df.drop(df.columns[[0]], axis = 1, inplace = True)

#st.write(df.head())




#Preview data
#st.write(df.head())
#st.write(df.tail())

energy = df
st.sidebar.header('Input Values')

classifier_name = st.sidebar.selectbox("Select Classifier",("SARIMA","LSTM"))

st.markdown("***")
       
st.write("""
         
## Results


""")

    



def get_classifier(classifier_name):
    if classifier_name == "SARIMA":
        np.round(energy['Global_active_power'].describe(), 2).apply(lambda x: format(x, 'f'))
        energy_hourly_summary = energy.groupby(by=['Year','Month','Day',"Hour"], as_index=False)[['Global_active_power']].mean()
        #summarize by mean hourly energy
        #create the Label Encoder object
        le = preprocessing.LabelEncoder()
        #encode categorical data
        energy_hourly_summary['Year'] = le.fit_transform(energy_hourly_summary['Year'])
        #st.write(energy_hourly_summary.head())
        indexed_engery = energy[['Date_Time','Global_active_power', 'Year', 'Month']].set_index('Date_Time');
        train = indexed_engery[indexed_engery['Year'] < 2010] #train set is years 2006 - 2009
        test = indexed_engery[indexed_engery['Year'] == 2010] #test set is year 2010
        #st.write(train.head(),test.head())
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        train_monthly=train[['Global_active_power']].resample('W').mean()
        #st.write(train_monthly.head())
        mod = SARIMAX(train_monthly, order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 50), #50 = number of weeks that we are forcasting
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        results = mod.fit()
        #st.write(results.forecast())
        predictions = results.predict(start='2010-01-03', end='2010-12-19')
        #st.write(predictions.head())
        #merge on Date_Time
        test[['Global_active_power']].resample('W').mean().reset_index()
        prediction = pd.DataFrame(predictions).reset_index()
        prediction.columns = ['Date_Time','Global_active_power']
        res = pd.merge(test[['Global_active_power']].resample('W').mean(), 
               prediction, 
               how='left', 
               on='Date_Time')
        res.columns = ['Date_Time','actual','predictions']
        res.insert(3, 'residuals', res['actual'] - res['predictions']) #residuals
        st.write(res.head())
        st.text("")
        f, axes = plt.subplots(2, figsize=(15, 10), sharex=True)
        #plot of actual vs predictions
        axes[0].plot(res['Date_Time'],res['actual'], color='black', label='actual')
        axes[0].plot(res['Date_Time'],res['predictions'], color='blue', label='prediction')
        axes[0].set_title('Actual vs Predicted Energy')
        axes[0].set_ylabel('Global_active_power')
        axes[0].legend()
        #plot of actual - predictions
        axes[1].scatter(res['Date_Time'],(res['actual'] - res['predictions']))
        axes[1].set_title('Residual Plot')
        axes[1].set_xlabel('Date (By Week)')
        axes[1].set_ylabel('Actuals - Predictions')
        axes[1].axhline(y=0, color='r', linestyle=':')
        plt.show()
        st.pyplot()
        st.text("")
        MSE = np.mean(res['residuals']**2) 
        st.write("MSE =",MSE)
        st.markdown("***")
    elif classifier_name == "LSTM":
        
    
        df_resampled=df1.resample('h').mean()
        #st.write(df_resampled)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df_resampled)
        df_scaled =pd.DataFrame(scaled)
        df_resampled["target"] = df_resampled.Global_active_power.shift(-1)
        df_resampled = df_resampled.iloc[:-1,:] # remove last value as it is shifted upward
        #st.write(df_resampled.tail())
        values = df_resampled.values
        num_train = 365*24
        train = values[:num_train, :]
        test = values[num_train:, :]
        X_train, y_train = train[:,1:], train[:,0]
        X_test, y_test = test[:,1:], test[:,0]
        # Reshaping train and test sets
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        #st.write(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        model = tf.keras.Sequential()
        initializer = tf.keras.initializers.HeNormal()
        model.add(tf.keras.layers.LSTM(100,
                               activation='relu',
                               kernel_initializer=initializer,
                               input_shape=(X_train.shape[1],
                                            X_train.shape[2])))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), shuffle=False)
        history.history.keys()
        st.text("")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        st.pyplot()
        y_pred = model.predict(X_test)
        # lets scale the X_test too
        X_test = X_test.reshape(X_test.shape[0], 7)
        # Invert scaling for pred
        inv_x = np.concatenate((y_pred, X_test[:, -6:]), axis=1)
        inv_x = scaler.inverse_transform(inv_x)
        inv_y_pred = inv_x[:,0]
        # invert scaling for actual
        y_test = y_test.reshape((len(y_test), 1))
        inv_y = np.concatenate((y_test, X_test[:, -6:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_y_pred))
        st.text("")
        st.write('RMSE value : {}'.format(rmse))
        # calculate R2 Score
        r2 = r2_score(inv_y, inv_y_pred)
        st.text("")
        st.write("R2 Score : {}".format(r2))
        sns.set_style("darkgrid")
        #original-blue
        plt.plot(inv_y)
        #predicted-red
        st.text("")
        plt.plot(inv_y_pred,color="red")
        plt.show()
        st.pyplot()
        p = st.sidebar.slider('PREDICTION PLOT RANGE', 0, 25000, 1000)
        st.text("")
        aa=[x for x in range(p)]
        plt.figure(figsize=(25,10))
        plt.plot(aa, inv_y[:p], marker='.', label="actual")
        plt.plot(aa, inv_y_pred[:p], 'r', label="prediction")
        plt.ylabel(df.columns[0], size=15)
        plt.xlabel('Time step ', size=15)
        plt.legend(fontsize=15)
        plt.show()
        st.pyplot()
        st.markdown("***")
        
        
        
        

        

get_classifier(classifier_name)

