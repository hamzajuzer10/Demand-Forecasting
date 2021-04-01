# seasonal feature generator, must be used before train test split
# binners represent insight extracted from EDA and applied on test set based on pre-defined rules 

import pandas as pd 
import numpy as np
from pandas.tseries.offsets import Week

def seasonal_holiday_week(data): 

    #month index: 
    data['month'] = data['reporting_date'].apply(lambda x: month_index_generator(x))

    #week in a year index: 
    data['week_in_a_year'] = data['reporting_date'].apply(lambda x: x.week)

    # holiday index
    dic = get_holiday_dic()

    # for some reason this apply function not work
    #data['holiday'] = data['reporting_date'].apply(lambda x: feature_engineer.holiday_index_generator(x))
    # use foor loop instead
    for i,r in data.iterrows(): 
        if r['reporting_date'] in dic.keys(): 
            data.at[i,'holiday'] = dic[r['reporting_date']]
        else: 
            data.at[i,'holiday'] = 0
    # season
    data['season'] = data['reporting_date'].apply(lambda x: season_index_generator(x.week))
    return(data)

def price_status_bkey_modifier(system_price,
                               original_selling_price,
                               original_selling_price_3_before,
                               system_price_3_after): 
    price_status_bkey = str()
    if system_price == original_selling_price: 

        price_status_bkey = 'full_pirce'

    # we found that there are no increased price column 
    if system_price > original_selling_price: 
         
        price_status_bkey = 'increased' 

    if system_price <= original_selling_price and system_price <= original_selling_price_3_before: 

        price_status_bkey = 'markdown' 

    if system_price <= original_selling_price and system_price >=  system_price_3_after: 

        price_status_bkey = 'promotion' 

    return price_status_bkey

def price_status_generator(data_list): 

    def price_status_binner(t,t_2): 

        if t > t_2: 

            status = 1

        if t < t_2: 

            status = -1 
    
        if t == t_2: 

            status = 0

        return status

    for i in range(len(data_list)): 

        try: 
            data_list[i]['price_status'] =  data_list[i].apply(lambda x: price_status_binner(x['system_price'],x['system_price(t-2)']), axis =1) 

        except KeyError:
            pass 
    return(data_list)

def lookback_generator(list_of_cols, lookback_peroid, data):


     for col in list_of_cols:
         for i in range(1,lookback_peroid+1): 

            data[col+'(t-'+str(i)+')'] = data.groupby(['product_bkey1'])[col].shift(i)
            data = data.fillna(0)

     return data

def diff_lookback_generator(list_of_cols, lookback_peroid, data):
    for col in list_of_cols:
        
         for i in range(1,lookback_peroid): 
                data[col + '-' + '(t-'+str(i+1)+')'] = data.groupby(['product_bkey1'])[col].diff(i).fillna(0)
    return data

def get_holiday_dic(): 

    """
    get a dictionary of holidays, keys = weeks that has holidays, values = number of holidays in that week 
    """

    public_holiday2018 = pd.to_datetime(pd.Series(['2018-01-01','2018-01-15','2018-02-19',
                                    '2018-05-28','2018-07-04','2018-09-03','2018-10-08',
                                    '2018-11-12','2018-11-22',
                                    '2018-12-05','2018-12-25']))

    public_holiday2019 = pd.to_datetime(pd.Series(['2019-01-01','2019-01-21','2018-02-18',
                                    '2019-05-27','2019-07-04',
                                    '2019-09-02','2019-10-14','2019-11-11',
                                    '2019-11-28',
                                    '2019-12-25']))

    public_holiday2020 = pd.to_datetime(pd.Series(['2020-01-01','2020-01-20','2020-01-25','2020-02-17',
                                    '2020-05-25','2020-07-04','2020-09-07','2020-10-12',
                                    '2020-11-11','2020-11-26',
                                    '2020-12-25']))

    holiday_list = pd.concat([public_holiday2018,public_holiday2019,public_holiday2020],axis =0)

    holiday_list = holiday_list + Week(weekday=5)

    dic = {}
    lis = list(holiday_list)
    for i in range(len(lis)): 

        value = lis.count(lis[i])

        dic[lis[i]] = value
    return dic

def holiday_index_generator(reporting_date): 

    """
    create the holiday column 
    """
    dic = get_holiday_dic()

    if reporting_date in dic.keys(): 

        holiday = dic[reporting_date]
    return holiday
    

def season_index_generator(week): 

    """
    need to be used in apply function, cannot use directly 
    """

    if week // 13 == 0 or week // 13 == 4 or week // 13 == 8: 

        season = 1 

    if week // 13 == 1 or week // 13 == 5: 

        season = 2  
    
    if week // 13 == 2 or week // 13 == 6: 

        season = 3 

    if week // 13 == 3 or week // 13 == 7: 

        season = 4 

    return season

# monthly feature generator, must be used before train test split 
# must be used on the datetime object 

def month_index_generator(reporting_date): 
    """
    need to be used in apply function, cannot use directly 
    """
    month = reporting_date.month

    return month 

"""

########## Below this line all appliciable functions are used for Children data-set ###############

"""
# product name feature generator, must be used on train, test seperately to avoid bias
# merge on product name into train
def product_name_generator(data): 

    """
    return a seperate table 
    """

    def product_namer_binner(sales_quantity_median): 

        if sales_quantity_median >= 60: 

            product_name_cat = 2 

        if sales_quantity_median >= 50 and sales_quantity_median < 60: 

            product_name_cat = 3
    
        if sales_quantity_median >= 40 and sales_quantity_median < 50: 

            product_name_cat = 4

        if sales_quantity_median >= 30 and sales_quantity_median < 40: 

            product_name_cat = 1
        
        else: 

            product_name_cat = 5

        return(product_name_cat) 

    series = data.groupby('product_name')['sales_quantity'].mean()
    series_2 = series.apply(lambda x: product_namer_binner(x)).reset_index()
    series_2 = series_2.rename(columns = {'sales_quantity':'product_name_cat'})
    return series_2

#### Used for Children ####
# product description feature generator, must be used on train, test seperately to avoid bias
# merge on product description into train
def product_description_generator(data): 

    def product_description_binner(sales_quantity_median): 

        if sales_quantity_median >= 80: 

            product_description_cat = 2 

        if sales_quantity_median >= 40 and sales_quantity_median < 80: 

            product_description_cat = 3
    
        if sales_quantity_median >= 20 and sales_quantity_median < 40: 

            product_description_cat = 4

        if sales_quantity_median >= 10 and sales_quantity_median < 20: 

            product_description_cat = 5

        if sales_quantity_median < 10: 

            product_description_cat = 1
        
        else: 

            product_description_cat = 6

        return(product_description_cat) 
        
    series = data.groupby('product_description')['sales_quantity'].median()
    series_2 = series.apply(lambda x: product_description_binner(x)).reset_index()
    series_2 = series_2.rename(columns = {'sales_quantity':'product_description_cat'})
    return series_2


#### Used for Children ####
# supplier name feature generator, must be used on train, test seperately to avoid bias
# merge on supplierd name into train
def supplier_name_generator(data): 

    """
    return a seperate table 
    """

    def supplier_name_binner(sales_quantity_median): 

        if sales_quantity_median >= 120: 

            supplier_name_cat = 2 

        if sales_quantity_median >= 80 and sales_quantity_median < 120: 

            supplier_name_cat = 3

        if sales_quantity_median >= 60 and sales_quantity_median < 80: 

            supplier_name_cat = 4
        
        if sales_quantity_median >= 40 and sales_quantity_median < 60: 

            supplier_name_cat = 5
        
        if sales_quantity_median >= 20 and sales_quantity_median < 40: 

            supplier_name_cat = 6
        
        if sales_quantity_median >= 10 and sales_quantity_median < 20: 

            supplier_name_cat = 7

        if sales_quantity_median < 10: 

            supplier_name_cat = 1

        else: 

            supplier_name_cat = 8

        return(supplier_name_cat) 
        
    series = data.groupby('supplier_name')['sales_quantity'].median()
    series_2 = series.apply(lambda x: supplier_name_binner(x)).reset_index()
    series_2 = series_2.rename(columns = {'sales_quantity':'supplier_name_cat'})
    return series_2


#### Used for Children ####
# brand name feature generator, must be used on train, test seperately to avoid bias
# merge on brand name into train
def brand_name_generator(data): 

    """
    return a seperate table 
    """

    def brand_name_binner(sales_quantity_median): 

        if sales_quantity_median >= 10: 

            brand_name_cat = 2 
        
        if sales_quantity_median >= 5 and sales_quantity_median < 10: 

            brand_name_cat = 3

        if sales_quantity_median < 5: 

            brand_name_cat = 1

        else: 

            brand_name_cat = 4

        return(brand_name_cat) 
        
    series = data.groupby('brand_name')['sales_quantity'].median()
    series_2 = series.apply(lambda x: brand_name_binner(x)).reset_index()
    series_2 = series_2.rename(columns = {'sales_quantity':'brand_name_cat'})
    return series_2

#### Used for Children ####
def train_product_grouop_generator(train): 
    """
    return a seperate table , this fucntion only applied on TRAIN!!!!
    """
    # mean sales for each product bkey1
    table = train.groupby('product_bkey1')['sales_quantity'].mean().reset_index()
    table = table.rename(columns = {'sales_quantity':'mean_sales_quantity'})

    # sales count for each product bkey1 (how many weeks that this product is sold ? )
    table2 = train.groupby('product_bkey1')['sales_quantity'].count().reset_index()
    table['number_sales'] = table2['sales_quantity']

    # calculate the below 0 sales density for each sigle product  
    below_1_sales = train.groupby('product_bkey1')['sales_quantity'].agg(lambda x: (x<0).sum())
    table['below_1_sales'] = below_1_sales 
    table['sales_sparsity'] = table['below_1_sales']/table['number_sales']

    # generate the result table 
    table['mean_sales_ranking'] = pd.qcut(table['mean_sales_quantity'],
                                         q = 3, 
                                         labels=[1,3,2],duplicates='drop')

    table['below_1_sales_percentage'] = pd.qcut(table['sales_sparsity'],
                                               q = 3, 
                                               labels=[2,3,1],duplicates='drop')

    table = table[['product_bkey1','mean_sales_ranking','below_1_sales_percentage']]

    return table

def train_test_splitter(n_weeks_out,data): 
    
    total_week = list(data['reporting_date'].unique())
    
    split_week = total_week[-n_weeks_out]

    #split = data_list[i]['reporting_date'].unique()[split_week-1]

    train = data[data['reporting_date'] < split_week]
    valid = data[data['reporting_date'] >=  split_week]

    train = train.sort_values(by = ['reporting_date'])
    valid = valid.sort_values(by = ['reporting_date'])

    return train, valid






    



