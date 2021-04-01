import os 
import pandas as pd

def loading(files,name,place_holder): 
    
    place_holder = os.getcwd()
    dfs_p = dict()
    for i in range(len(files)):
        path = place_holder + files[i]
        dfs_p[name[i]] = pd.read_csv(path, sep='|')
    #for df in dfs_p.keys(): 
        #print('There are {} columns in {} dataset'.format(len(dfs_p[df].columns),str(df)))
    return(dfs_p)

# Two differnt useful info extractor created to apadt differnt format in terms 
# of datetime in different dataset

def useful_info_extractor(data,dfs_p): 
    data = data.fillna(0)
    data = data.drop(columns = data.columns[data.nunique().values ==1])
    data['reporting_date'] = pd.to_datetime(data['reporting_date'],dayfirst=True) 
    if 'price_status_bkey' in data.columns : 
        data['price_status_bkey'] = data['price_status_bkey'].astype(int)
    if 'Unnamed: 0' in data.columns : 
        data = data.drop(columns = ['Unnamed: 0'])
    return(data)

def useful_info_extractor1(data,dfs_p): 
    data = data.fillna(0)
    data = data.drop(columns = data.columns[data.nunique().values ==1])
    #data['reporting_date'] = pd.to_datetime(data['reporting_date'],dayfirst=True) 
    if 'price_status_bkey' in data.columns : 
        data['price_status_bkey'] = data['price_status_bkey'].astype(int)
    if 'Unnamed: 0' in data.columns : 
        data = data.drop(columns = ['Unnamed: 0'])
    return(data)