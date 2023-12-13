import src.data_metrics as data_metrics
#import numpy as np
import pandas as pd
import logging
import src.binarize_queensland as bq

def read_csv_data(path, skip_errors=False, index_col=None, header=0, nrows=None, use_cols=None):
    """Read csv data using pandas

    Args:
        path (str): path to the csv file
        skip_errors (bool, optional): stop parsing on errors. Defaults to False.
        index_col (str/int, optional): index of dataframe in csv. Defaults to None.
        header (int, optional): headers of the dataframe in csv. Defaults to 0.
        nrows (int, optional): number of rows to read. Defaults to None.

    Returns:
        DataFrame: pandas DF with csv content
    """
    logging.debug('READ: %s', path)
    df = pd.read_csv(path, on_bad_lines='skip', index_col=index_col, header=header, nrows=nrows, usecols=use_cols) #skips bad lines with wrong number of cols

    logging.debug('\tSHAPE: %s', str(df.shape))
    logging.debug('\tCOLUMS: %s', str(df.columns))
    return df

def get_unique_counts(df, column):
    print("UNIQUE: ",column, '\t', len(df[column].unique()))

def encode_dataset(df_orig):
    #IPV4_SRC_ADDR  L4_SRC_PORT    IPV4_DST_ADDR  L4_DST_PORT  PROTOCOL  L7_PROTO   IN_BYTES  OUT_BYTES  IN_PKTS  OUT_PKTS  TCP_FLAGS  FLOW_DURATION_MILLISECONDS  Label Attack
    if ('L7_PROTO' in df_orig.columns) and ('Attack' in df_orig.columns) :
        df = df_orig.drop(columns=['L7_PROTO','Attack'])
    else: 
        df = df_orig

    #print("Encode")
    #print(df.columns)
    #IPV4_SRC_ADDR
    df_src_ip = bq.split_df_seperator(df, 'IPV4_SRC_ADDR', seperator='.')
    #L4_SRC_PORT
    df_src_pt = bq.binarize_and_split(df, 'L4_SRC_PORT', bits=16)
    #IPV4_DST_ADDR
    df_dst_ip = bq.split_df_seperator(df, 'IPV4_DST_ADDR', seperator='.')
    #L4_DST_PORT
    df_dst_pt = bq.binarize_and_split(df, 'L4_DST_PORT', bits=16)
    #PROTOCOL
    df_proto = bq.encode_protocol_one_hot(df, 'PROTOCOL')
    
    #IN_BYTES
    bits = 4*8
    bin_list = bq.create_bins_log_bits(bits)
    df_in_bytes = bq.quantisize_values_to_bins(df, 'IN_BYTES', bin_list=bin_list)
    #OUT_BYTES
    df_out_bytes = bq.quantisize_values_to_bins(df, 'OUT_BYTES', bin_list=bin_list)
    #IN_PKTS
    df_in_pckts = bq.quantisize_values_to_bins(df, 'IN_PKTS', bin_list=bin_list)
    #OUT_PKTS
    df_out_pckts = bq.quantisize_values_to_bins(df, 'OUT_PKTS', bin_list=bin_list)

    #TCP_FLAGS
    df_flags = bq.binarize_and_split(df, 'TCP_FLAGS', bits=8)

    #FLOW_DURATION_MILLISECONDS
    max_value = 4.294967e+06
    elements = 16
    bin_list = bq.create_bins_linear_max(max_value, elements)   
    df_duration = bq.quantisize_values_to_bins(df, 'FLOW_DURATION_MILLISECONDS', bin_list=bin_list)

    #only binary values, lower precision/memory required
    df_encoded = pd.concat([df_src_ip, df_src_pt, df_dst_ip, df_dst_pt, df_proto, df_in_bytes, df_out_bytes, df_in_pckts, df_out_pckts, df_flags, df_duration, df['Label']], axis=1)
    df_encoded = df_encoded.astype('int8')
    #print(df_encoded)
    #for col in df_encoded.columns:
    #    print(col)
    return df_encoded

def filter_and_sort_index_local(s1, unique_values):
    s1_c = s1.value_counts(normalize=True).to_frame()
    #print(unique_values)
    #quit()
    name = s1_c.columns[0]
    #idx_df = pd.read_csv('counts_order/'+name+'.csv', index_col=0, header=0)
    idx_df = pd.DataFrame(unique_values).set_index(0)
    #idx_df.index = idx_df.index.astype(str)
    #print(idx_df)

    #filter data from original
    s1_c = s1_c[s1_c.index.isin(idx_df.index)]
    #print(s1_c)
    #print("Before missing check",len(s1_c))
    #add missing values with 0 value?
    missing_values = idx_df.index[~idx_df.index.isin(s1_c.index)]
    #print('values to add:', len(missing_values))
    #s1_c.at['UAPRSF', name] = 10 #testing
    if len(missing_values) > 0:
        for val in missing_values:
            s1_c.at[val, name] = 0.0

    #sort values
    s1_c.index = s1_c.index.astype(str)
    #print(s1_c)
    s1_c.sort_index(inplace=True)
    #print("After missing check", len(s1_c))
    #quit()
    return s1_c

def test_metrics_attribute_raw(df_1, df_2):
    js_dict = {}
    wd_dict = {}

    for att in CATEGORICAL_ATTRIBUTES:
        #print(att)
        a_unique = list(set(list(df_1[att].unique()) + list(df_2[att].unique()))) #outer list to keep order
        #print(a_unique)
        s1_c = filter_and_sort_index_local(df_1[att].copy(), a_unique)
        #print('s1_c')
        #print(s1_c)
        
        s2_c = filter_and_sort_index_local(df_2[att].copy(), a_unique)
        #print('s2_c')
        #print(s2_c)
        #kde required?
        #js = data_metrics.calculate_jenson_shennon_divergence(s1_c[att].copy(), s2_c[att].copy())
        js = data_metrics.calculate_jenson_shennon_divergence(s1_c['proportion'], s2_c['proportion'])

        #print('js', js)
        js_dict.update({att:js[0][0]}) #returns array

        #wd = data_metrics.calculate_wasserstein_distance(s1_c[att].copy(), s2_c[att].copy())
        wd = data_metrics.calculate_wasserstein_distance(s1_c['proportion'], s2_c['proportion'])

        #print('wd', wd)
        wd_dict.update({att:wd}) 



        #quit()
    
    #quit()

    #normalize normal attributes?

    for att in NUMERICAL_ATTRIBUTES:
        a = list(df_1[att].copy())
        b = list(df_2[att].copy())
        #print(att)
        #print(df_1.shape, df_2.shape)
        #print(a)
        #print(b)

        a_pdf = data_metrics.kernel_density_estimation(a)
        b_pdf = data_metrics.kernel_density_estimation(b)

        #sample common points from both?
        a_pdf_points = pd.Series(a_pdf(a+b))
        #print(a_pdf_points)
        #print(len(a_pdf_points))
        #plt.plot(a_pdf_points))
        b_pdf_points = pd.Series(b_pdf(a+b))
        #print(b_pdf_points)
        #print(len(b_pdf_points))
        #plt.plot(b_pdf_points)
        
        #plt.show()
        #quit()
        js = data_metrics.calculate_jenson_shennon_divergence(a_pdf_points, b_pdf_points)
        #print("JS: ",js)
        js_dict.update({att:js[0][0]})

        wd = data_metrics.calculate_wasserstein_distance(a_pdf_points, b_pdf_points)
        #print("WD: ",wd)
        wd_dict.update({att:wd})

    #print(js_dict)
    #print(wd_dict)
    #return {'avg_js': np.array(js_list).mean(), 'avg_wd':np.array(wd_list).mean()}
    return {'JS': pd.Series(js_dict), 'WD':pd.Series(wd_dict)}

def test_metrics_correlation(df_1,df_2):
    pearson_1, pearson_2 = data_metrics.calculate_pearson(df_1.copy(),df_2.copy())
    #print("pearson",pearson)

    corr_ratio_1, corr_ratio_2 = data_metrics.calculate_correlation_ratio(df_1.copy(),df_2.copy())
    #print("corr_ratio",corr_ratio)

    theil_u_1, theil_u_2 = data_metrics.calculate_theil_u(df_1.copy(),df_2.copy())
    #print("Theil",theil_u)

    return {'pearson_1': pearson_1, 'corr_ratio_1': corr_ratio_1, 'theils_u_1':theil_u_1,
            'pearson_2': pearson_2, 'corr_ratio_2': corr_ratio_2, 'theils_u_2':theil_u_2}

def test_anomaly_detection_discriminator_raw(df_1, df_2):
    train = df_1
    test = df_2
    #print('TRAIN')
    #print(train)
    #print('TEST')
    #print(test)

    
    test = encode_dataset(test)
    train = encode_dataset(train)

    y_train = train['Label']
    x_train = train.drop(columns=['Label']) 

    y_test = test['Label']
    x_test = test.drop(columns=['Label']) 

    pred_dict_if = data_metrics.test_isolation_forest_dsicriminator_raw(x_train.copy(), x_test.copy())
    pred_dict_ocsvm = data_metrics.test_ocsvm_dsicriminator_raw(x_train.copy(),x_test.copy())

    pred_dict = pred_dict_if | pred_dict_ocsvm
    return pred_dict

def test_anomaly_detection_task_raw(df_1, df_2):
    train = df_1
    test = df_2

    test = encode_dataset(test)
    train = encode_dataset(train)

    #print(x_train)
    #print(x_test)

    task_dict_if = data_metrics.test_isolation_forest_task_raw(train.copy(), test.copy())
    task_dict_ocsvm = data_metrics.test_ocsvm_task_raw(train.copy(),test.copy())
    task_dict_xgb = data_metrics.test_xgboost_task_raw(train.copy(), test.copy())

    task_dict = task_dict_if | task_dict_ocsvm | task_dict_xgb
    return task_dict


NUMERICAL_ATTRIBUTES = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS'] #leave out timestamps since they are not equal anayway --> always a high distance
CATEGORICAL_ATTRIBUTES = [ 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
                            'PROTOCOL', 
                            #'L7_PROTO', #not used here
                            #'IN_BYTES', 'OUT_BYTES', 
                            #'IN_PKTS', 'OUT_PKTS',
                            'TCP_FLAGS', 
                            #'FLOW_DURATION_MILLISECONDS', 
                            'Label', 
                            #'Attack' #unequal amount of labels, to prevent errors for pcd dont use them
                            ]
COLUMNS_ORIGINAL = [ 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
                            'PROTOCOL', 
                            'L7_PROTO', #not used here
                            'IN_BYTES', 'OUT_BYTES', 
                            'IN_PKTS', 'OUT_PKTS',
                            'TCP_FLAGS', 
                            'FLOW_DURATION_MILLISECONDS', 
                            'Label', #'Attack' #unequal amount of labels, to prevent errors for pcd dont use them
                            ]

#### READ and STORE DATAFRAMES ####

        

    
    










        
