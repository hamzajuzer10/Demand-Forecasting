# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% Section 1
"""
Load Relevant packages for running the main function 
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import math
import data_loader
import feature_engineer 
import data_generator
import sequential 
import pickle
import os 
import math 

# %% Section 2
"""
Load the pre-trained LGBM model, out best candidate in training session 
lag = maximum look back peroid ever existed, means that we only look back for 2 weeks max for each row
"""
RSEED = 42
lag = 2 
model = pickle.load(open('LGBM_Regressor', 'rb'))
print(model)
print('model loaded')

# %% Section 3
"""
Enter training and testing directory and file names
"""
# train 
place_holder = os.getcwd()
files  = ['/DATASET3_price.csv','/DATASET3_sales.csv','/DATASET3_stock.csv',
          '/Dataset3_Product.psv', 
          '/Dataset3_Product Hierarchy.psv']
name = ['price','sales','stock','product','hie']

#test
files_hold_out  = ['/DATASET3_price_holdout.csv','/DATASET3_sales_holdout_v0.2_MASKED_IDX.csv','/DATASET3_stock_holdout.csv']
name_holdout = ['price','sales','stock']

# %% Section 4 Data Loading & Cleaning 

"""
Module to check" data_loader
"""
# Load the training set
train_dic = data_loader.loading(files,name,place_holder)

# seperate data into its own dataframe 
price = train_dic['price']
sales = train_dic['sales']
stock = train_dic['stock']
product = train_dic['product']
hie = train_dic['hie']

# Delete useless information in each data set 
# Including column with all NAN, column with only one unique values, column 
price_train = data_loader.useful_info_extractor(price,train_dic)
sales_train = data_loader.useful_info_extractor(sales,train_dic)
stock_train = data_loader.useful_info_extractor(stock,train_dic)
product_train = data_loader.useful_info_extractor1(product,train_dic)
hie_train = data_loader.useful_info_extractor1(hie,train_dic)

#Load the holo-out/test set
df_dic_holdout = data_loader.loading(files_hold_out,name_holdout,place_holder)

# Delete useless information in each data set 
# Including column with all NAN, column with only one unique values, column 
price_test = data_loader.useful_info_extractor(df_dic_holdout['price'], df_dic_holdout)
sales_test_no_index = data_loader.useful_info_extractor(df_dic_holdout['sales'], df_dic_holdout).drop(columns = ['index'])
stock_test = data_loader.useful_info_extractor(df_dic_holdout['stock'],df_dic_holdout)

# %%
# Section 5
"""
Use the pre designed data generator to conduct feature engineering for data at one go
Module to check: data_generator
"""
# This section takes about 5-6 minutes to run, input will be shown on command line 
data_raw, date_matcher_df, product_matcher_df = \
data_generator.data_generator_forecast(price_test, 
                                       sales_test_no_index, 
                                       stock_test,
                                       price_train, 
                                       sales_train, 
                                       stock_train, 
                                       product_train, 
                                       hie_train)
# This function has 8 inputs, use 5 from train and 8 from test

# %% 
# Section 6 rolling forecasting 

"""
Use the pre designed data generator to conduct feature engineering for data at one go
Module to check: sequential
"""
train = data_raw[data_raw['reporting_date'] <= 92]
test = data_raw[data_raw['reporting_date']> 92]
if test['sales_quantity'].sum() == 0:
    print('Now test data has no sales quantity, all asumme to be zero')

result_table  = sequential.rolling_forecaster_expansion_test(lag,train,test,model)
print('Forecast done')
# %% 
# Section 7 Turn the output in to the required format 

def output_generator(): 
    
    # the date_matcher_df is broken
    # make a connector df connect the actual datetime with factoirsed week
    connector = pd.DataFrame(columns = ['datetime','factorised_week'], index = list(range(93,106)))

    list_of_datetime = sales_test_no_index['reporting_date'].unique()

    connector['datetime'] = list_of_datetime
    connector['factorised_week'] = list(range(93,106))

    rt_1 = result_table.merge(connector, left_on = 'reporting_date', right_on = 'factorised_week')
    rt_1 = rt_1.drop(columns = ['reporting_date','factorised_week'])

    # use pre-generated product_matcher_df to match back to product bkey1 
    rt_2 = rt_1.merge(product_matcher_df, left_on = 'product_bkey1', right_on = 'factorised_bkey') .drop_duplicates()
    
    rt_2 = rt_2.sort_values(by = 'datetime').drop(columns = ['product_bkey1_x','factorised_bkey'])
    rt_2 = rt_2.rename(columns = {'product_bkey1_y':'product_bkey1',"datetime":'reporting_date'})

    sales = df_dic_holdout['sales']
    sales['reporting_date'] = pd.to_datetime(sales['reporting_date'], dayfirst = True)

    sales = sales.merge(rt_2, on =['reporting_date','product_bkey1'], how = 'left')
    sales = sales.rename(columns = {'prediction':'sales_quantity'})

    sales.to_csv('sales prediction(traditioanl ML).csv')
    print('result table generated')
    return sales

# %%
sales = output_generator()
print('Output_generated')
# Couting time: the script takes about 9 minutes to run, result stored in 
###  "sales prediction(traditioanl ML).csv"