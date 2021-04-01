import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

## Please note that this one is used on vlaidation set
## NOT on hold out set 
def rolling_forecaster_expansion_valid(lag,train,test,model):
    MAE = []
    R2 =[]
    """
    input: Lag: the maximum looking back peroid, here is 2 
           train: the training data set 
           test: the test data set 
           model: model that we use 

    Output:   result_table: the result table, week, bkey, and predictions
              MAE: list of length 12, weekly MAE recorder
              
    """
    
    # OUTPUT 
    result_table = pd.DataFrame(columns = ['prediction','product_bkey1','reporting_date'])
    
    # train_X and train_y split 
    train_X, train_y = train.drop(columns = ['sales_quantity']), train['sales_quantity']
    test_X, test_y = test.drop(columns = ['sales_quantity']), test['sales_quantity'] 

    #Standardisation and put back to pandas form 
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_X_s = pd.DataFrame(X_scaler.fit_transform(train_X), columns = train_X.columns,index = train_X.index )
    test_X_s = pd.DataFrame(X_scaler.transform(test_X), columns = test_X.columns,index = test_X.index)

    train_y_s = pd.Series(y_scaler.fit_transform(np.array(train_y).reshape(-1, 1)).flatten(), index = train_y.index)
    test_y_s = pd.Series(y_scaler.transform(np.array(test_y).reshape(-1, 1)).flatten(), index = test_y.index)

    #Utility variables
    # the iterrables to guide the on step forecasting session 
    date_slider = list(test_X_s['reporting_date'].unique())
  
    #train_scaled/valid_sacled, df with scaled value

    # columns that I need forward update
    sq_col = ['sales_quantity(t-1)','sales_quantity(t-2)','sales_quantity(t-1)-(t-2)']

    # data transformation & fit 
    pca = PCA(n_components=19)
    model.fit(pca.fit_transform(train_X_s),train_y_s)
    

    for i in range(len(date_slider)): 

        print('forecasting for week {} begins'.format(i))
        
        temp_X = test_X_s[test_X_s['reporting_date'] == test_X_s['reporting_date'].unique()[0]]
        temp_y = test_y_s[temp_X.index]

        # generate the point estimation 
        prediction_s = model.predict(pca.transform(temp_X))
        
        prediction = y_scaler.inverse_transform(prediction_s)
        temp_s_sb = y_scaler.inverse_transform(temp_y)
        
        # mae visualisation 
        mae = mean_absolute_error(temp_s_sb,prediction)
        r2 = r2_score(temp_s_sb,prediction)
        print('forecasting mae for week {} is {}'.format(i,mae))
        print('forecasting R^2 for week {} is {}'.format(i,r2))
        MAE.append(mae)
        R2.append(r2)

        # generate the dictionary that used to replace value in the feature 
        temp = temp_X.copy()
        temp['prediction'] = prediction_s
        temp = temp[['prediction','product_bkey1','reporting_date']]
    
        # update valid_X in the future 
        for j in range(1,lag+1): 

            if i+j <= len(date_slider) - 1: 

                # look up the weeks after the exsiting test set, all the way up to predefined lag
                    X_test_next_week_sq = test_X_s[test_X_s['reporting_date'] == date_slider[i+j]][sq_col + ['product_bkey1']]

                    # update the test set that we are going to use in the future with the local prediction 
                    temp2 = X_test_next_week_sq.reset_index().merge(temp, on='product_bkey1', how = 'inner').set_index('index')

                    temp2 = temp2.drop(columns = ['sales_quantity(t-'+str(j)+')'])
                    temp2 = temp2.rename(columns = {'prediction':'sales_quantity(t-'+str(j)+')'})
                    
                    if j == 1: 
                        temp2['sales_quantity(t-1)-(t-2)'] = temp2['sales_quantity(t-'+str(j)+')'] - temp2['sales_quantity(t-2)']
                    
                    index_to_update = temp2.index
                    test_X_s.loc[index_to_update][temp2.columns] = temp2
            else: 
                pass
    
        # chop the size of the test_X
        test_X_s = test_X_s.drop(temp_X.index)
        test_y_s = test_y_s.drop(temp_X.index)
        
        #### UNHASH here than the strategy becomes sliding window ###
        # push the size of trian_X and trian_y down by one week, sliding window
        #train_X_s = train_X[train_X_s['reporting_date'] > train_X_s['reporting_date'].unique()[0]]
        #train_y_s = train_y[train_X_s.index]

        # increase the size of train_X and train_y
        train_X_s = pd.concat([train_X_s, temp_X])
        train_y_s = pd.concat([train_y_s, pd.Series(prediction_s,index = temp_X.index)])
     
        #store the result
        #refit the model 
        pca.fit(train_X_s)
        model.fit(pca.transform(train_X_s),train_y_s) 

        result_table = pd.concat([result_table,temp])
        print('forecasting for week {} completes'.format(i))
        
    return result_table, MAE, R2

# rolling forecasting on the hole out set
def rolling_forecaster_expansion_test(lag,train,test,model):

    """
    input: Lag: the maximum looking back peroid, here is 2 
           train: the training data set 
           test: the test data set 
           model: model that we use 

    Output:   only the result table 
              
    """
    
    # OUTPUT 
    result_table = pd.DataFrame(columns = ['prediction','product_bkey1','reporting_date'])
    
    # train_X and train_y split 
    train_X, train_y = train.drop(columns = ['sales_quantity']), train['sales_quantity']
    test_X = test.drop(columns = ['sales_quantity'])

    #Standardisation and put back to pandas form 
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaler.fit(train_X)
    y_scaler.fit(np.array(train_y).reshape(-1, 1))

    train_X_s = pd.DataFrame(X_scaler.fit_transform(train_X), columns = train_X.columns,index = train_X.index )
    train_y_s = pd.Series(y_scaler.fit_transform(np.array(train_y).reshape(-1, 1)).flatten(), index = train_y.index)
    
    #Utility variables
    # the iterrables to guide the on step forecasting session 
    date_slider = list(test_X['reporting_date'].unique())
  
    #train_scaled/valid_sacled, df with scaled value

    # columns that I need forward update
    sq_col = ['sales_quantity(t-1)','sales_quantity(t-2)','sales_quantity(t-1)-(t-2)']

    # data transformation & fit 
    pca = PCA(n_components=19)
    model.fit(pca.fit_transform(train_X_s),train_y_s)
    
    for i in range(len(date_slider)): 

        print('forecasting for week {} begins'.format(i))
        
        temp_X = test_X[test_X['reporting_date'] == test_X['reporting_date'].unique()[0]]
        
        # standardise the data frame 
        temp_X_s = pd.DataFrame(X_scaler.transform(temp_X), columns = temp_X.columns,index = temp_X.index)

        # generate the point estimation based on scaled test_X
        prediction_s = model.predict(pca.transform(temp_X_s))
        
        # scale back the predcition 
        prediction = y_scaler.inverse_transform(prediction_s)

        # store prediction using the unscaled dataset 
        temp = temp_X.copy()
        temp['prediction'] = prediction

        # generate the df that used to replace value in the feature 
        temp_replacer = temp[['prediction','product_bkey1','reporting_date']]
        
        ## Forward filling starts here 
        # update valid_X in the future 
        for j in range(1,lag+1): 

            if i+j <= len(date_slider) - 1: 

                # look up the weeks after the exsiting test set, all the way up to predefined lag
                X_test_next_week_sq = test_X[test_X['reporting_date'] == date_slider[i+j]][sq_col + ['product_bkey1']]

                # update the test set that we are going to use in the future with the local prediction 
                temp2 = X_test_next_week_sq.reset_index().merge(temp_replacer, on='product_bkey1', how = 'inner').set_index('index')

                temp2 = temp2.drop(columns = ['sales_quantity(t-'+str(j)+')'])
                temp2 = temp2.rename(columns = {'prediction':'sales_quantity(t-'+str(j)+')'})
                
                #fill t-1 - t-2 as well, only necesary when j = 1
                if j == 1: 
                    temp2['sales_quantity(t-1)-(t-2)'] = temp2['sales_quantity(t-'+str(j)+')'] - temp2['sales_quantity(t-2)']
                    
                index_to_update = temp2.index
                test_X.loc[index_to_update][temp2.columns] = temp2

            else: 
                pass
    
        # chop the size of the test_X
        test_X = test_X.drop(temp_X.index)
        
        #### UNHASH here than the strategy becomes sliding window ###
        # push the size of trian_X and trian_y down by one week, sliding window
        #train_X_s = train_X[train_X_s['reporting_date'] > train_X_s['reporting_date'].unique()[0]]
        #train_y_s = train_y[train_X_s.index]

        # now work with the sacled data set 
        # increase the size of train_X and train_y
        train_X_s = pd.concat([train_X_s, temp_X_s])
        train_y_s = pd.concat([train_y_s, pd.Series(prediction_s,index = temp_X.index)])
     
        #store the result
        #refit the model 
        pca.fit(train_X_s)
        model.fit(pca.transform(train_X_s),train_y_s) 

        result_table = pd.concat([result_table,temp],sort = True)
        print('forecasting for week {} completes'.format(i))
        
    return result_table[['prediction','reporting_date','product_bkey1']]