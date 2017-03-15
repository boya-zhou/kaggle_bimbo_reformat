
# coding: utf-8

# In[60]:

import pandas as pd
import numpy as np
import re
import sys

# In[61]:



# In[62]:

def readin_train(path) :
    """read the csv file in

    Args :
        path : project path
        nrows : number of rows to read in

    Returns:
        dataset

    """
    dtypes = {'Semana' : 'int32',
              'Agencia_ID' :'int32',
              'Canal_ID' : 'int32',
              'Ruta_SAK' : 'int32',
              'Cliente-ID' : 'int32',
              'Producto_ID':'int32',
              'Venta_hoy':'float32',
              'Venta_uni_hoy': 'int32',
              'Dev_uni_proxima':'int32',
              'Dev_proxima':'float32',
              'Demanda_uni_equil':'int32'}


    names_dict = {"Demanda_uni_equil" : "label"}
    train_dataset = pd.read_csv(path + "/data/interim/train_sample.csv",
                                usecols =['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Demanda_uni_equil'],
                                dtype  = dtypes)

    train_dataset.rename(columns = names_dict, inplace = True)
    return train_dataset


# In[63]:

def readin_test(path) :
    """read the csv file in

    Args :
        path : project path
        nrows : number of rows to read in

    Returns:
        dataset

    """

    dtypes = {'Semana' : 'int32',
              'Agencia_ID' :'int32',
              'Canal_ID' : 'int32',
              'Ruta_SAK' : 'int32',
              'Cliente-ID' : 'int32',
              'Producto_ID':'int32',
              'Venta_hoy':'float32',
              'Venta_uni_hoy': 'int32',
              'Dev_uni_proxima':'int32',
              'Dev_proxima':'float32',
              'Demanda_uni_equil':'int32'}

    test_dataset = pd.read_csv(path + "/data/interim/test_sample.csv",
                                usecols =['Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'],
                                dtype  = dtypes)
    return test_dataset


# In[64]:

def pivot_table(df):
    """pivot the table, for each (client, product) pair, it should show week3 to week 9 demand

    Week3 : product1, client1, demand
    Week4 : product1, client1, demand
    Week5 : product1, client1, demand
    Week6 : product1, client1, demand
    Week7 : product1, client1, demand
    Week8 : product1, client1, demand

    To

    product1, client1 : week3_demand, week4_demand, week5_demand, week6_demand, week7_demand, week8_demand

    Args :
        Origin training dataset

    Returns :
        Showed above
    """
    df["label"] = df["label"].apply(np.log1p)
    pivot_df = pd.pivot_table(data = df[["Semana","Cliente_ID", "Producto_ID", "label"]],
                              values = "label", index = ["Cliente_ID", "Producto_ID"],
                              columns = ["Semana"],  aggfunc= np.mean).reset_index()

    pivot_df.columns = ["label_" + str(ele) if str(ele).isdigit() else ele for ele in pivot_df.columns.values]
    return pivot_df


# In[65]:

def gen_se_dist(df):
    """generate statistical distribution value for Producto_ID(product_id) and Cliente_ID(client_id) pair

    The main issue of this dataset is most features are in nominal format, ex. ID
    So it's better to generate some numerical distribution value for reasonble nominal feature pair

    Args : train_dataset

    Returns : dist info provided by train_dataset
    """
    df_se_pro_cli_dist = df.groupby(["Cliente_ID","Producto_ID"],as_index = False).                        agg(['count','sum', 'min', 'max','median','mean']).reset_index()
    df_se_pro_cli_dist.columns = ["_".join(ele) + "_spc" if ele[1] != ""
                               else ele[0] for ele in df_se_pro_cli_dist.columns.values]

    df_se_pro_dist = df.drop("Cliente_ID",1).groupby(["Producto_ID"],as_index = False).                        agg(['count','sum', 'min', 'max','median','mean']).reset_index()
    df_se_pro_dist.columns = ["_".join(ele) + "_sp" if ele[1] != ""
                               else ele[0] for ele in df_se_pro_dist.columns.values]

    df_se_cli_dist = df.drop("Producto_ID",1).groupby(["Cliente_ID"],as_index = False).                        agg(['count','sum', 'min', 'max','median','mean']).reset_index()
    df_se_cli_dist.columns = ["_".join(ele) + "_sc" if ele[1] != ""
                           else ele[0] for ele in df_se_cli_dist.columns.values]

    return df_se_pro_cli_dist, df_se_pro_dist, df_se_cli_dist


# In[66]:

def gen_pro_cli_dist(df):
    df_cli_pro_dist = df[["Cliente_ID","Producto_ID",'label']].                 groupby(["Cliente_ID","Producto_ID"],as_index = False).                 agg(['count','sum', 'min', 'max','median','mean']).reset_index()
    df_cli_pro_dist.columns = ["_".join(ele) + "_cp" if ele[1] != ""
                       else ele[0] for ele in df_cli_pro_dist.columns.values]
    return df_cli_pro_dist


# In[67]:

def merge_df_dist(df, dist):
    """merge the dist info with train_dataset or test_dataset"""
    dist_join_key = [ele for ele in dist.columns if ele.split("_")[0] != "label"]
    df = df.merge(dist, how = 'left', on = dist_join_key)
    return df


# In[68]:

def create_lag(df):
    """create time lag between demand among each week

    The week10 is held out data and week11 is the data we need to predict
    So for week10 we create lag1 and lag2 between week3 ~ week8
    So for week11 we create lag1 and lag2 between week4 ~ week9

    It should be after pivot_table when training and test data have already been merged with dist infomation

    Args :
        training data or test data already merged with dist info
    """
    periods = [(3,8), (4,9)]
    lag_time = [1, 2]

    for period in periods:
        for lag in lag_time:
            for index in range(period[0], period[1] + 1 - lag):
                df['label_' + str(index + lag) + '_min_' + str(index )] = df["label_" + str(index + lag)].subtract(df["label_" + str(index)])
    return df


# In[24]:

def external_info(path):
    """add additional information from /external, this data were provided by Kaggle

    External Source name :
        cliente_tabla
        producto_tabla
        town_tabla

    """
    producto_tabla = pd.read_csv(path + "/data/external/producto_tabla.csv")
    town_state = pd.read_csv(path + "/data/external/town_state.csv")

    town_state['town_id'] = town_state['Town'].str.split()
    town_state['town_id'] = town_state['Town'].str.split(expand = True)
    town_state = pd.concat([town_state, pd.get_dummies(town_state[["town_id","State"]], prefix=['town_id', 'state_id'])],axis = 1)
    town_state.drop(["Town", "State", "town_id"], axis = 1, inplace = True)

    reg_weight = r" (\d{1,10})g "
    reg_piece = r" (\d{1,10})p "
    producto_tabla["weight"] = producto_tabla["NombreProducto"].apply(lambda x: re.findall(reg_weight, x)[0] if re.search(reg_weight, x) else np.nan)
    producto_tabla["piece"] = producto_tabla["NombreProducto"].apply(lambda x: re.findall(reg_piece, x)[0] if re.search(reg_piece, x) else np.nan)
    producto_tabla["wei_per_piece"] = producto_tabla["weight"].astype(float).divide(producto_tabla["piece"].astype(float))
    producto_tabla.drop("NombreProducto", axis = 1, inplace = True)

    return town_state, producto_tabla


# In[25]:

def gen_dist_train_test(train_df, test_df, pivot_table, gen_se_dist, gen_pro_cli_dist, external_info):
    """generate dist information on training data, merge the distribution with both training data and test data

    The Data flow should look like this:
        train_df ==> pivot_table ==> gen_se_dist ====> get_dist_info ==> merge info(train and test)
                |               |                   |
                |               |                   |
                |                ==> create_lag ====|
                |                                   |
                |                                   |
                 ==> gen_pro_cli_dist ==============|
                |                                   |
                |                                   |
                 ==> external_info =================|

    So after these process, the train_df.shape[0] will decrease, while test_df.shape[0] will not change
    """

    df_cli_pro_dist = gen_pro_cli_dist(train_df)

    train_pivot = pivot_table(train_df)
    df_se_pro_cli_dist, df_se_pro_dist, df_se_cli_dist = gen_se_dist(train_pivot)
    df_lag = create_lag(train_pivot)

    dist_list = [df_cli_pro_dist, df_lag, df_se_pro_cli_dist, df_se_pro_dist, df_se_cli_dist]

    for dist in dist_list:
        dist_join_key = [ele for ele in dist.columns if ele.split("_")[0] != "label"]
        train_df = train_df.merge(dist, on = dist_join_key, how = 'left')
        test_df = test_df.merge(dist, on = dist_join_key, how = 'left')


    town_state, producto_tabla = external_info(path)

    train_df = train_df.merge(town_state, on = "Agencia_ID", how = 'left')
    train_df = train_df.merge(producto_tabla, on = "Producto_ID", how = 'left')

    test_df = test_df.merge(town_state, on = "Agencia_ID", how = 'left')
    test_df = test_df.merge(producto_tabla, on = "Producto_ID", how = 'left')

    train_df.drop(["Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID", "label"], axis = 1, inplace = True)
    train_df.drop_duplicates(inplace = True)

    test_df.drop(["Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID"], axis = 1, inplace = True)

    return train_df, test_df


# In[26]:

def filter_name(column_name, type_name):
    """filter name for train and test data for model

    Args :
        column_name : colname for training_df and test_df
        type_name : can either be "train", "test", "hold"

    Returns:
        filtered column based on type
    """
    filtered_name = []
    if type_name == "train":
        # 8, 9 can be shown in featurem label is 9
        for name in column_name:
            if str(8) in name.split("_") or str(9) in name.split("_"):
                pass
            else:
                filtered_name.append(name)
    elif type_name == "test":
        # 3, 9 can be shown in featurem label is 10(label on Kaggle, so no local label)
        for name in column_name:
            if str(3) in name.split("_") or str(9) in name.split("_"):
                pass
            else:
                filtered_name.append(name)
    else:
        # 3, 4 can be shown in featurem label is 11(label on Kaggle, so no local label)
        for name in column_name:
            if str(3) in name.split("_") or str(4) in name.split("_"):
                pass
            else:
                filtered_name.append(name)

    return filtered_name


# In[56]:

def prepare_train_hold_test(train_df, test_df):
    """based on the specifity on this problem, we should split train test data based on time factor

    training_data : week3, week4, week5, week6, week7 ==> week9
    hold_out_data : week4, week5, week6, week7, week8 ==> week10 (get result from Kaggle public board)
    test_data : week5, week6, week7, week8, week9 ==> week11 (get result from Kaggle private board)

    generate (training_data, train_label), hold_out_data and test_data
    """
    train_data = train_df.loc[train_df["label_9"].isnull() == False]
    train_label = train_data[["label_9"]]
    train_data = train_data[filter_name(train_df.columns.values, "train")]


    hold_data = train_df[filter_name(train_df.columns.values, "hold")]
    test_data = test_df[filter_name(test_df.columns.values, "test")]

    return train_data, train_label, hold_data, test_data



# In[14]:

def prepare_whole():
    """prepare for the train_df and test_df"""
    train_dataset = readin_train(path)
    test_dataset = readin_test(path)
    train_dataset, test_dataset = gen_dist_train_test(train_dataset, test_dataset,
                                                      pivot_table,
                                                      gen_se_dist,
                                                      gen_pro_cli_dist,
                                                      external_info)
    train_data, train_label, hold_data, test_data = prepare_train_hold_test(train_dataset, test_dataset)
    train_data.to_csv(path + "/data/processed/train.csv", index = False)
    train_label.to_csv(path + "/data/processed/train_label.csv", index = False)
    hold_data.to_csv(path + "/data/processed/hold.csv", index = False)
    test_data.to_csv(path + "/data/processed/test.csv", index = False)

    print "Data preparation is done"
    print "%s of features is created" %train_data.shape[1]

    print "training data size : %s" %train_data.shape[0]
    print "holdout data size : %s" %hold_data.shape[0]
    print "test data size : %s" %test_data.shape[0]
    return train_data, train_label, hold_data, test_data


# In[20]:

if __name__ == "__main__":
    global path
    path = sys.argv[1]
    prepare_whole()


# In[ ]:
