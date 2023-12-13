import numpy as np
import pandas as pd


# binarize Ports, since they are categories, but one-hot would be too many vectors (0 - 65535) 
def decimalToBinary(x, num_bits):
    try:
        bin_str = bin(int(x)).replace("0b", "")[:num_bits] #limit to 8 bits
    except:
        #print('CANNOT CONVERT TO INT:', x)
        bin_str='0'
    if bin_str == None:
        bin_str = '0'
    zeroes_to_add = num_bits-len(bin_str)
    bin_str = '0'*zeroes_to_add + bin_str
    return bin_str

def split_str_into_columns(df, col_name):
    #split str per char
    #remove first and last dummy columns
    df_tmp = pd.DataFrame([list(x) for x in df[col_name]])#slightly faster
    #rename columns with name prefix and number suffix
    df_tmp.columns = [ col_name+'_'+str(col) for col in df_tmp.columns ]
    #combine split cols with df
    #df = pd.concat( [df, df_tmp], axis=1)
    #remove orig col from df
    #df.drop(columns=[col_name], inplace=True)
    return df_tmp

def binarize_and_split(df_normal, column_name, bits=16):
    #print(df_normal[column_name])
    df_normal[column_name] = df_normal[column_name].apply(lambda x: decimalToBinary(x, bits))
    df_tmp = split_str_into_columns(df_normal, column_name)
    return df_tmp


def split_df_seperator(df, col_name, seperator):
    #split str per char
    df_tmp = pd.DataFrame(df[col_name].str.split(pat=seperator, expand=True, n=4))
    #rename columns with name prefix and number suffix
    df_tmp.columns = [ col_name+'_oct_'+str(col) for col in df_tmp.columns ]
    #print(df_tmp)
    #combine split cols with df
    #df = pd.concat( [df, df_tmp], axis=1)
    #print(df)
    #print(df)
    #remove orig col from df
    #df.drop(columns=[col_name], inplace=True)
    oct_list = []
    for oct_col in df_tmp.columns:
        #print(oct_col)
        df_oct = binarize_and_split(df_tmp, oct_col, bits=8)
        oct_list.append(df_oct)
    df_oct_tmp = pd.concat(oct_list, axis=1)
    return df_oct_tmp


def encode_protocol_one_hot(df_normal, col_name):
    UDP = 17
    TCP = 6
    ICMP = 1
    IGMP = 2
    #special encoding for protocol TCP, UDP, other, 
    df_proto = pd.DataFrame()
    df_proto[col_name+'_UDP'] = np.where( (df_normal[col_name] == UDP), 1, 0)
    df_proto[col_name+'_TCP'] = np.where( (df_normal[col_name] == TCP), 1, 0)
    df_proto[col_name+'_ICMP'] = np.where( (df_normal[col_name] == ICMP), 1, 0)
    df_proto[col_name+'_IGMP'] = np.where( (df_normal[col_name] == IGMP), 1, 0)

    df_proto['Protocol_OTHER'] = np.where( (df_normal[col_name] != UDP)&(df_normal[col_name] != TCP)&(df_normal[col_name] != ICMP)&(df_normal[col_name] != IGMP), 1, 0)
    #df_normal.drop(columns=[col_name], inplace=True)
    return df_proto

def create_bins_linear_max(max_value, elements):
    max_value = 4.294967e+06
    elements = 16
    return [i/elements*max_value for i in range(1,elements)]

def create_bins_log_bits(bits):
    #bits = 4*8
    return [2**i for i in range(0 ,bits-1,2)]

def quantisize_values_to_bins(df_normal, col_name, bin_list):
    df_bin_col = pd.DataFrame(np.digitize(df_normal[col_name], bins=bin_list))
    #print(col_name)
    #print("EXAMPLE",df_bin_col[:20])
    #print("VALUE COUNTS", df_bin_col.value_counts())
    #print("SORT IN COLS:")
    #turn bins to columns
    col_list = []
    for n, bin_n in enumerate(bin_list):
         col_list.append(col_name+'_'+str(n))
         df_normal[col_name+'_'+str(n)] = np.where( (df_bin_col == n), 1, 0)
         #print(bin_n, df_normal[col_name+'_'+str(bin_n)].sum())
    #drop old_column
    return df_normal[col_list]
    
