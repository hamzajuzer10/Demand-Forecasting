from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model, Sequential
from keras import backend as K
import pandas as pd
import data_loader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import os 
from matplotlib import pyplot
import numpy as np
from keras.callbacks import ReduceLROnPlateau

def split_sequences(sequences, n_steps_in, n_steps_out):

    """
    Input: 
    This function is responsible for buliding a shifting window 
    sequences: data with row  = time, column = product 
    n_steps_in: training_X window 
    n_steps_out: trainin_Y window

    Output: 
    np.array(X): 3-D arry (number of windows, fixed n steps in, number of features/products)
    np.array(y): 3-D arry (number of windows, fixed n steps out, number of features/products)
    """

    X, y = list(), list()
    for i in range(len(sequences)):

        # find the end of this the data set
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break

        # gather input windown and output windows
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

        # store them
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_model(X,n_steps_in,n_steps_out): 

    """
    This function is responsible for buliding a the LSTM netowrk 

    This represents a encoder and decoder model of LSTM deisgn, I learned from the following link: 
    Intro1: https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
    Intro2 (more detailed): https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    """ 
    n_features = X.shape[2]
    model = Sequential()
    model.add(LSTM(250, activation='relu', input_shape=(n_steps_in, n_features))) # Encoder Layer
    ## fixed-length output of the encoder is repeated
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(250, activation='relu', return_sequences=True)) # Decoder Layer 
    # drop out layer to prevent overfitting 
    model.add(Dropout(0.2))
    model.add(LSTM(250, activation='relu', return_sequences=True)) # Decoder Layer 2
    # drop out layer to prevent overfitting 
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mae')

    """
    Adding this reduced LR will result in a normal distributed error, but the MAE will jump large 
    This ReduceLROnPlatea has intro in Keras, basically it monitor a metric and reduced learning rate 
    once the metric stooped improving significantly 

    But I finallhy decided to use it since if not, 
    every product will have the same sales quantity some time 

    If I increase my epoches, the model will benefit from this 
    """
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
    return model, reduce_lr

def forecaster(data,n_steps_in,n_steps_out,e_poches): 

    """
    trained model using a vlidation split of 0.25
    and output the result after mature 
    """

    # train and valid split
    #train_df = data.iloc[:data.shape[0]-n_steps_out,:]
    train = data
    train_df = data.values

    #Speicfy important variables, 
    X, y = split_sequences(train_df, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    model,reduce_lr  = get_model(X,n_steps_in,n_steps_out)
    
    # Trainning 
    # showing the variatoin of loss function 
    history = model.fit(X, y, epochs=e_poches, verbose=1, validation_split=0.20, callbacks = [reduce_lr])
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()  
    
    # Forecasting 12 weeks at one go
    x_input = train.tail(n_steps_in).values
    #print(x_input.shape)
    x_input = x_input.reshape((1, n_steps_in, n_features))
    #print(x_input.shape)
    yhat = model.predict(x_input, verbose=2)
    #print(yhat)

    yhat_change = yhat.reshape(yhat.shape[1],yhat.shape[2])

    #print('Mean absolute error of 12 weeks forecasting: {}'.format(mean_absolute_error(valid,yhat_change)))
    #print('Root mean squared error of 12 weeks forecasting: {}'.format(sqrt(mean_squared_error(valid,yhat_change))))

    return model, yhat_change 

def forecaster_train(data,n_steps_in,n_steps_out,e_poches): 
    # train and valid split
    valid_df = data.tail(12)
    train_df = data.iloc[:data.shape[0]-12,:]
    
    train = train_df.values
    valid = valid_df.values
    
    #Speicfy important variables, 
    #n_steps_in = 24 
    #n_steps_out = 12
    X, y = split_sequences(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    model, reduce_lr = get_model(X,n_steps_in,n_steps_out)
    
    # Trainning 
    # showing the variatoin of loss function 
    history = model.fit(X, y, epochs=e_poches, verbose=1, validation_split=0.20, callbacks = [reduce_lr])
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()  
     
    # Forecasting 12 weeks at one go
    x_input = train_df.tail(n_steps_in).values
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=2)

    yhat_change = yhat.reshape(yhat.shape[1],yhat.shape[2])

    print('Mean absolute error of 12 weeks forecasting: {}'.format(mean_absolute_error(valid,yhat_change)))
    print('Root mean squared error of 12 weeks forecasting: {}'.format(sqrt(mean_squared_error(valid,yhat_change))))

    return model, yhat_change, valid