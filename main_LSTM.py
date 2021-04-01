# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
"""
Package Loading
"""
import data_loader
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
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
import LSTM_func
from keras.utils.vis_utils import plot_model


# %%
"""
Section 1 Train Data Loading 
"""
# place holder for trainings, only for sales quantity information is necsessary
place_holder = os.getcwd()
files  = ['/DATASET3_sales.csv']
name = ['sales']
# load the data
df_dic = data_loader.loading(files,name,place_holder)

#  extract only the useful columns (no NA, no single unique values)
sales = df_dic['sales'] 
sales = data_loader.useful_info_extractor(sales,df_dic)


# %%
"""
Section 2 Hold out test loading and datetime conversion 
"""
test = pd.read_csv('DATASET3_sales_holdout_v0.2_MASKED_IDX.csv',delimiter='|')
test['reporting_date'] = pd.to_datetime(test['reporting_date'], dayfirst = True)
#test.tail()


# %%
"""
Section 3 Raise a potential data leakage issue
There are overlaps interms reporting date for test set and training set 
Overlapping data is trimed to make sure the test 
Unhash to visulise 
"""
#print(test.tail())
#pritn(train.tail())

#print(test['reporting_date'].nunique())
#print(test['reporting_date'].nunique())

# %%
"""
Section 4 Round down the number of products in the training size as most of them 
does not appear in the test set 
"""
print('There are {} products in the sales table to begin with'.format(sales['product_bkey1'].nunique()))
print('There are {} products in the test sales table to begin with  '.format(test['product_bkey1'].nunique()))
sales = sales[sales['reporting_date'] < test['reporting_date'].min()]
print('There are {} products in the training sales table after time chuncking'.format(sales['product_bkey1'].nunique()))

# roudn down the data set as we only need to forecast product that exists in the hold_out test
# this could potentially lead to error reduction 
product_in_test = list(test['product_bkey1'].unique())
sales = sales[sales['product_bkey1'].isin(product_in_test)]
print('There are {} products in the traiing sales table after round down'.format(sales['product_bkey1'].nunique()))


# %%
"""
Section 5 Create the pivot table and train the neural network 
"""

data = pd.pivot_table(sales, values='sales_quantity', 
                             index=['reporting_date'],
                             columns=['product_bkey1'], 
                             fill_value=0)
#data.head()
n_steps_in = 30
n_steps_out = 13 
e_poches = 100
model, yhat_change = LSTM_func.forecaster(data,n_steps_in,n_steps_out,e_poches)


# %%
"""
Section 6 Stack the pivot table and get the result in the required format 
"""
def result_generator(): 

     result_table = pd.DataFrame(yhat_change, index = test['reporting_date'].unique(), columns = data.columns)
     result_final = pd.DataFrame(result_table.stack()).reset_index(level=[0,1])
     result_final = result_final.rename(columns = {'level_0':'reporting_date', 0: 'sales_quantity'})
     result_final = pd.DataFrame(result_table.stack()).reset_index(level=[0,1])
     result_final = result_final.rename(columns = {'level_0':'reporting_date', 0: 'sales_quantity'})
     final = test.merge(result_final, on = ['product_bkey1','reporting_date'], how = 'left')
     final.to_csv('sales prediction(LSTM).csv')
     plot_model(model, show_shapes=True, show_layer_names=True)
     return final 

# %%
final  = result_generator()
print(final.max())
