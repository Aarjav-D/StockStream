import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import streamlit as st
import time

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# %matplotlib inline
import pandas_datareader as data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

def app():

    user_input = st.text_input("Enter stock ticker",'AAPL')

    dataset_train = data.DataReader(user_input,data_source='yahoo',start='2012-01-01')
    dataset_train.to_csv('StockData.csv')
    dataset_train = pd.read_csv('StockData.csv')

    st.subheader('Raw data from 10th November 2016 to Yesterday')
    st.write(dataset_train.describe())

    cols = list(dataset_train)[1:6]

    # Extract dates (will be used in visualization)
    datelist_train = list(dataset_train['Date'])
    datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]

    dataset_train = dataset_train[cols].astype(str)
    for i in cols:
        for j in range(0, len(dataset_train)):
            dataset_train[i][j] = dataset_train[i][j].replace(',', '')

    dataset_train = dataset_train.astype(float)

    # Using multiple features (predictors)
    training_set = dataset_train.values

    print('Shape of training set == {}.'.format(training_set.shape))
    training_set

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    training_set_scaled = sc.fit_transform(training_set)

    sc_predict = StandardScaler()
    sc_predict.fit_transform(training_set[:, 0:1])

    X_train = []
    y_train = []

    n_future = 60   # Number of days we want to predict into the future
    n_past = 90     # Number of past days we want to use to predict the future

    for i in range(n_past, len(training_set_scaled) - n_future +1):
        X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # print('X_train shape == {}.'.format(X_train.shape))
    # print('y_train shape == {}.'.format(y_train.shape))

    warning = st.warning('Loading Model...')
    my_bar = st.progress(0)
    
    for percent_complete in range(100) :
        time.sleep(0.05)
        my_bar.progress(percent_complete+1)
    my_bar.empty()

    warning.empty()

    success = st.success('Model Successfully Trained')

    model = load_model('keras_LSTM_stock_model.h5')

    # Generate list of sequence of days for predictions
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

    '''
    Remeber, we have datelist_train from begining.
    '''

    # Convert Timestamp to Datetime object 
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())
    
    predictions_future = model.predict(X_train[-n_future:])

    predictions_train = model.predict(X_train[n_past:])

    # Inverse the predictions to original measurements

    # convert <datetime.date> to <Timestamp>
    def datetime_to_timestamp(x):
        '''
            x : a given datetime value (datetime.date)
        '''
        return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

    y_pred_future = sc_predict.inverse_transform(predictions_future)
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

    # Convert <datetime.date> to <Timestamp> for PREDICTION_TRAIN
    PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

    st.subheader('Prediction')

    dataset_train = pd.DataFrame(dataset_train, columns=cols)
    dataset_train.index = datelist_train
    dataset_train.index = pd.to_datetime(dataset_train.index)
    # Set plot size 
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 20, 8
    fig = plt.figure(figsize = (14,5))
    # Plot parameters
    START_DATE_FOR_PLOTTING = '2015-06-01'

    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted Stock Price')
    plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
    plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')

    plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predictions and Actual Stock Prices', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    st.pyplot(fig) 
        