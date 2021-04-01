import data_loader
import pandas as pd 
import numpy as np 
import feature_engineer
from sklearn.preprocessing import StandardScaler

def data_generator(files,name,place_holder,n_weeks_out): 

    """
    Input: 
    files: file name of the data set, 
    place_holder of directory: place holder, 
    n_weeks_out: forecasting peroid 

    Output: 
    # train and test set ready to be fitted by ML models
    """

    ##### DATA CLEANING STARTS HERE 
    # load the dat from the file 
    df_dic = data_loader.loading(files,name,place_holder)
    
    # seperate data into its own dataframe 
    price = df_dic['price']
    sales = df_dic['sales']
    stock = df_dic['stock']
    product = df_dic['product']
    hie = df_dic['hie']
    
    # extract columns that are not fully null 
    # extract columns that does not have onely one unique value
    # convert reporting week into standrad python datetime data structure 
    # convert 'price status businsess' key into intger 
    # '3' -- > 3 
    price = data_loader.useful_info_extractor(price,df_dic)
    sales = data_loader.useful_info_extractor(sales,df_dic)
    stock = data_loader.useful_info_extractor(stock,df_dic)
    product = data_loader.useful_info_extractor1(product,df_dic)
    hie = data_loader.useful_info_extractor1(hie,df_dic)
    
    #drop because we will use our own value to encode 
    sales = sales.drop(columns = ['year_seasonality_bkey']) 
    
    # this feature should not appear in stock table, already appeared in hie
    stock = stock.drop(columns = ['node_name_level2'])

    # merge across differnet tables first conduct left merge and drop columns with missing values
    core1 = pd.merge(price, sales, on=['product_bkey1','reporting_date'],how='left')
    core2 = pd.merge(core1, stock, on=['product_bkey1','reporting_date'],how='left')
    core3 = pd.merge(product, hie, on=['product_bkey1'],how='left')
    final_core = pd.merge(core2, core3, on=['product_bkey1'],how = 'left')
    final_core = final_core.dropna(how = 'any')
    print('There are {} products in the our final data set'.format(final_core['product_bkey1'].nunique()))
    print('There are {} weeks in the our final data set'.format(final_core['reporting_date'].nunique()))
    
    # drop selling price becasue we selling price  == system price, contain no usefuel information 
    # drop node id since we already have names (no need to repeat inforamtion)
    # there is another year_seasonality_bkey in the price table 
    final_core = final_core.drop(columns = ['selling_price','node_id_level2','node_id_level3','cost_price',
                                            'node_id_level4','node_id_level5','year_seasonality_bkey'])
    ##### DATA CLEANING ENDS HERE 

    #### FEATURE ENGINEERING BEGINS HERE 
    # Generate a family of seasonal indx 
    final_core = feature_engineer.seasonal_holiday_week(final_core)

    # Factorisation 
    for cols in ['product_bkey1','product_name','product_description',
                 'node_name_level3','node_name_level4','node_name_level5','product_colour']:
        
        # add 1 because week normally starts from 1
        #if cols == 'reporting_date':
            #final_core['reporting_date'] = pd.factorize(final_core['reporting_date'])[0] + 1
        # as long as product_bkey1 is unique, it is fine 
        if cols == 'product_bkey1':
            final_core[cols] = pd.factorize(final_core[cols])[0] 
        
        # add modifier randomly to avoid mullticolinearlity 
        else: 
            modifier = np.random.randint(0, 10, 1)[0]
            final_core[cols] = pd.factorize(final_core[cols])[0] + modifier  

    # one hot encoding, only apply on node name level2 to highlight this column 
    level2_dummies = pd.get_dummies(final_core['node_name_level2'])
    final_core[level2_dummies.columns] = level2_dummies 
    final_core = final_core.drop(columns =['node_name_level2'])
    
    # return the stock quantity variables last week 
    final_core['stock_quantity(t-1)'] =  final_core.groupby('product_bkey1')['stock_quantity'].shift(1)
    final_core = final_core.fillna(0) 
    final_core['transit_stock_quantity(t-1)'] =  final_core.groupby('product_bkey1')['transit_stock_quantity'].shift(1)
    final_core = final_core.fillna(0) 
    final_core['intake_stock_quantity(t-1)'] =  final_core.groupby('product_bkey1')['intake_stock_quantity'].shift(1)
    final_core = final_core.fillna(0) 
    # calculate the price in the next 3 weeks to make preperation for price status bkey generation 
    final_core['system_price(t+3)'] = final_core.groupby('product_bkey1')['system_price'].shift(-3)
    final_core = final_core.fillna(0) 

    # orginal price and system price look back 
    list_of_cols = ['system_price','original_selling_price']
    lookback_peroid = 3
    final_core = feature_engineer.lookback_generator(list_of_cols,lookback_peroid, final_core)
    
    # correct data issues in the price status bkey 
    final_core['price_status_bkey']  = final_core['price_status_bkey'].astype(str)
    for i,r in final_core.iterrows(): 

        if r['system_price'] == r['original_selling_price']: 

            final_core.at[i,'price_status_bkey'] = 'full_pirce'

        if r['system_price'] > r['original_selling_price']: 
            
            final_core.at[i,'price_status_bkey'] = 'increased'

        if r['system_price'] < r['original_selling_price']: 

            if r['system_price'] < r['original_selling_price(t-3)']: 

                final_core.at[i,'price_status_bkey'] = 'markdown'

            if r['system_price'] > r['system_price(t+3)']: 

                final_core.at[i,'price_status_bkey'] = 'promotion'
            else: 
                final_core.at[i,'price_status_bkey'] = 'markdown'

    # one hot encoding price status bkey 
    price_status_dummies = pd.get_dummies(final_core['price_status_bkey'])
    final_core[price_status_dummies.columns] = price_status_dummies
    final_core = final_core.drop(columns =['price_status_bkey'])

    # look back the price 
    list_of_cols = ['sales_quantity']
    lookback_peroid = 2
    final_core = feature_engineer.lookback_generator(list_of_cols, lookback_peroid, final_core)
    
    # drop some columns to avoid curse of dimension and also multicolineality explored from trainning set of 80 weeks 
    # Also make sure information available at the same time
    final_core = final_core.drop(columns =['system_price(t-3)','original_selling_price',
                                           'system_price','system_price(t+3)',
                                           'original_selling_price(t-3)','stock_quantity',
                                           'transit_stock_quantity','intake_stock_quantity','increased',
                                           'system_price(t-2)','original_selling_price(t-2)',
                                           'product_description','node_name_level3'])
    
    final_core = feature_engineer.diff_lookback_generator(['sales_quantity(t-1)'], 2, final_core)
    

    # price status generation 
    final_core = feature_engineer.price_status_generator([final_core])[0]

    # train_test_split
    final_core = final_core.sort_values(by ='reporting_date')
    train,test = feature_engineer.train_test_splitter(n_weeks_out,final_core)

    return train, test
    


def data_generator_forecast(price_test, sales_test_no_index, stock_test, 
                         price_train, sales_train, stock_train, product_train, hie_train): 

    """
    Similar data generator function as for the train - validation process 
    Only differnce is that is will ouput two trackers to help get prediction back to the original table 
    """
    print('start the data genration process')
    print('merge the hold out test table') 
    test_merge1 = pd.merge(price_test, sales_test_no_index, on=['product_bkey1','reporting_date'],how='left').drop(columns = ['year_seasonality_bkey_x','year_seasonality_bkey_y'])

    test_raw = pd.merge(test_merge1, stock_test, on=['product_bkey1','reporting_date'],how='left').dropna()
    
    test_raw = test_raw.drop(columns = ['node_name_level2'])
    
    test_raw = pd.merge(test_raw, product_train[['product_name','product_colour','product_bkey1']], on = ['product_bkey1'],how = 'left')
    
    test_raw = pd.merge(test_raw, hie_train[['node_name_level2','node_name_level4','node_name_level5','product_bkey1']], on = ['product_bkey1'],how = 'left')

    test_raw = test_raw.dropna(how = 'any' )

    #drop because we will use our own value to encode 
    sales_train = sales_train.drop(columns = ['year_seasonality_bkey']) 

    # this feature should not appear in stock table, already appeared in hie
    stock_train = stock_train.drop(columns = ['node_name_level2'])
    
    # merge across differnet tables first conduct left merge and drop columns with missing values
    core1 = pd.merge(price_train, sales_train, on=['product_bkey1','reporting_date'],how='left')
    core2 = pd.merge(core1, stock_train, on=['product_bkey1','reporting_date'],how='left')
    core3 = pd.merge(product_train, hie_train, on=['product_bkey1'],how='left')
    train_raw = pd.merge(core2, core3, on=['product_bkey1'],how = 'left')
    train_raw = train_raw.dropna(how = 'any')
    
    #create target variable in hole out set as None
    prediction = [None]* test_raw.shape[0]
    test_raw['sales_quantity'] = prediction 
    print('drop columns that will never be used')
    data_list = []
    for data in [train_raw, test_raw]: 
        for col in data.columns: 
            if col in ['cost_price','node_id_level2','node_id_level3','node_id_level4','node_id_level5','year_seasonality_bkey','product_description','node_name_level3','selling_price']: 

               data = data.drop(columns = [col])

            # put sales quantity columns at the end 
            if 'sales_quantity' in col: 
            
               closed_at_end = ['sales_quantity']
               data = data[[c for c in data.columns if c not in closed_at_end] + [c for c in data.columns if c in closed_at_end]]
        data_list.append(data)
    
    print('{} unique products in train'.format(data_list[0]['product_bkey1'].nunique()))
    print('{} unique products in test'.format(data_list[1]['product_bkey1'].nunique()))

    print('{} weeks in train'.format(data_list[0]['reporting_date'].nunique()))
    print('{} weeks products in test'.format(data_list[1]['reporting_date'].nunique()))

    data = pd.concat([data_list[0],data_list[1]])

    print('start Feature Engineering')
    print('start generate seasonal featues: weeks, holidays, month etc')
    data = feature_engineer.seasonal_holiday_week(data)
    
    
    print('start factorisation')
    def fact(data): 
        
        for col in ['product_bkey1','product_name','node_name_level4','node_name_level5','product_colour','reporting_date']:
            # add 1 because week normally starts from 1
            if col == 'reporting_date':
               data['factorised_week'] = pd.factorize(data[col])[0] + 1 
               #date_matcher.append(pd.factorize(data[col]))
               
            # as long as product_bkey1 is unique, it is fine 
            if col == 'product_bkey1':
               data['factorised_bkey'] = pd.factorize(data[col])[0] 
               #product_matcher.append(pd.factorize(data[col]))

            # add modifier randomly to avoid mullticolinearlity 
            else: 
               modifier = np.random.randint(0, 10, 1)[0]
               data[col] = pd.factorize(data[col])[0] + modifier

        return data
        
    data = fact(data)
    
    print('start generate sales quantity in the past')
    data['sales_quantity(t-1)'] = data.groupby('factorised_bkey')['sales_quantity'].shift(1).fillna(0)
    data['sales_quantity(t-2)'] = data.groupby('factorised_bkey')['sales_quantity'].shift(2).fillna(0)
    data['sales_quantity(t-1)-(t-2)'] = data['sales_quantity(t-1)'] - data['sales_quantity(t-2)']

    date_matcher_df = data[['reporting_date','factorised_week']]
    product_matcher_df = data[['product_bkey1','factorised_bkey']]
    
    print('start prouct name dummy encoding')
    # one hot encoding, only apply on node name level2 to highlight this column 
    level2_dummies = pd.get_dummies(data['node_name_level2'])
    data[level2_dummies.columns] = level2_dummies 
    data = data.drop(columns =['node_name_level2']) 

    print('start past stock quantity generation')
    # define the stock quantity t-1 
    # define the transit stock quantity t-1 
    # define the intake stock quantity t-1 
    # generate future system price 
    data['stock_quantity(t-1)'] =  data.groupby('product_bkey1')['stock_quantity'].shift(1).fillna(0)

    data['transit_stock_quantity(t-1)'] =  \
                               data.groupby('product_bkey1')['transit_stock_quantity'].shift(1).fillna(0)

    data['intake_stock_quantity(t-1)'] =  \
                               data.groupby('product_bkey1')['intake_stock_quantity'].shift(1).fillna(0)
                               
    data['system_price(t+3)'] = data.groupby('product_bkey1')['system_price'].shift(-3).fillna(0)
    
    # generate system price and original selling price look back columns 
    print('start past price generation')
    list_of_cols = ['system_price','original_selling_price']
    lookback_peroid = 3
    data = feature_engineer.lookback_generator(list_of_cols,lookback_peroid,data)
    
    print('start markdown, promotion and full price generation')
    # modify the price status bkey 
    data['price_status_bkey']  = data['price_status_bkey'].astype(str)

    for i,r in data.iterrows(): 

       if r['system_price'] == r['original_selling_price']: 

           data.at[i,'price_status_bkey'] = 'full_pirce'

       if r['system_price'] > r['original_selling_price']: 
            
           data.at[i,'price_status_bkey'] = 'increased'

       if r['system_price'] < r['original_selling_price']: 

           if r['system_price'] < r['original_selling_price(t-3)']: 

               data.at[i,'price_status_bkey'] = 'markdown'

           if r['system_price'] > r['system_price(t+3)']: 

               data.at[i,'price_status_bkey'] = 'promotion'
           else: 
               data.at[i,'price_status_bkey'] = 'markdown'
    
    # one hot encoding price status bkey 
    price_status_dummies = pd.get_dummies(data['price_status_bkey'])
    data[price_status_dummies.columns] = price_status_dummies
    data = data.drop(columns =['price_status_bkey'])
    
    print('drop usless columns or columns identified as not usefuel during EDA')
    data = data.drop(columns =['system_price(t-3)',
                             'original_selling_price',
                            'system_price','system_price(t+3)',
                            'original_selling_price(t-3)','stock_quantity',
                            'transit_stock_quantity','intake_stock_quantity','increased',
                            'system_price(t-2)','original_selling_price(t-2)'])
    
    print('renmae columns and put target variable at the end')
    data = data.drop(columns = ['reporting_date','product_bkey1'])
    data = data.rename(columns = {'factorised_week':'reporting_date','factorised_bkey':'product_bkey1'})
    closed_at_end = ['sales_quantity']
    open_at_front = ['reporting_date','product_bkey1']
    data = data[[col for col in data.columns if col not in closed_at_end]+[col for col in data.columns if col in closed_at_end]]
    data = data[[col for col in data.columns if col in open_at_front]+[col for col in data.columns if col not in open_at_front]]
    
    print('end the data genration process')
    return data, date_matcher_df, product_matcher_df


