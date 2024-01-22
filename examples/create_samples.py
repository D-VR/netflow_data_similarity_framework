import logging
import pandas as pd
from os import listdir

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


root_prefix = '../../Datasets/Queensland_NetFlow/NetFlow_Benchmark/'
datasets  = [
                'NF-CSE-CIC-IDS2018',
                'NF-ToN-IoT',
                'NF-UNSW-NB15']

postfix = '.csv'

CHUNKS = 30
'''

### create dataset samples ###
for ds in datasets:
    df_1 = read_csv_data(root_prefix+ds+postfix)
    for i in range(3,CHUNKS):
        #print(i)
        #quit()
        file_name = ds+'_'+str(i)+'.csv'
        print(file_name)
        sample_df = df_1.sample(n=10000, axis=0)
        sample_df.to_csv('./dataset_samples/'+file_name, header=True, index=False)

'''

file_path = "./dataset_samples/"

for f in listdir(file_path):
        df = read_csv_data(file_path+f, header=0)
        counts = df['Label'].value_counts()
        print(f, counts[1])
        if len(counts)<2:
             #print(f)
             print("missing labels!")
