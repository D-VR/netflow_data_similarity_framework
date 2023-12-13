from os import makedirs
import argparse
import datetime

from src.data_metrics_framework import read_csv_data, CATEGORICAL_ATTRIBUTES, test_metrics_attribute_raw, test_metrics_correlation, test_anomaly_detection_discriminator_raw, test_anomaly_detection_task_raw

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--ds1", required=True)
    parser.add_argument("-b", "--ds2", required=True)
    parser.add_argument("-e", "--expid", required=True)
    parser.add_argument("-s", '--store_path', required=True)

    args = parser.parse_args()
    ds_1 = args.ds1
    ds_2 = args.ds2
    experiment_id = args.expid
    save_path = args.store_path

    df1_sample = read_csv_data(ds_1, header=0)
    df2_sample = read_csv_data(ds_2, header=0)

    df1_sample[CATEGORICAL_ATTRIBUTES].select_dtypes('category')
    df2_sample[CATEGORICAL_ATTRIBUTES].select_dtypes('category')

    print(df1_sample.shape)
    print(df2_sample.shape)

    dict_setup ={'ds_1':ds_1, 'ds_2':ds_2, 'ds_1_chunk_len':df1_sample.shape[0], 'ds_2_chunk_len':df2_sample.shape[0]}
    ds_1 = ds_1.split('/')[-1].split('.')[0]
    ds_2 = ds_2.split('/')[-1].split('.')[0]

    #save_path = '../../Datasets/Queensland_NetFlow/results_raw/exp_'+experiment_id+'/'+ds_1+'_'+ds_2+'/' 
    save_path = save_path+'/results_raw/exp_'+experiment_id+'/'+ds_1+'_'+ds_2+'/' 
    makedirs(save_path, exist_ok=True)
    
    print('START...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    def run_task():
        print("TASK")
        task_dict = test_anomaly_detection_task_raw(df1_sample, df2_sample)
        #print(task_dict)
        #quit()
        for key in task_dict.keys():
            task_dict[key].to_csv(save_path+key+'.csv')
        #quit()
        #print('TASK END...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        

    def run_discriminator():
        print("DISCRIMINATOR")
        disc_dict = test_anomaly_detection_discriminator_raw(df1_sample, df2_sample)
        #print(disc_dict)
        #quit()
        for key in disc_dict.keys():
            disc_dict[key].to_csv(save_path+key+'.csv')
        #quit()
        #print('DISCRIMINATOR END...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        

    def run_attribute():
        print('ATTRIBUTE')
        att_dict = test_metrics_attribute_raw(df1_sample, df2_sample)
        #print(corr_dict)
        for key in att_dict.keys():
            att_dict[key].to_csv(save_path+key+'.csv')
        #print('ATTRIBUTE END...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        

    def run_correlation():
        print('CORRELATION')
        corr_dict = test_metrics_correlation(df1_sample, df2_sample)
        #print(corr_dict)
        for key in corr_dict.keys():
            corr_dict[key].to_csv(save_path+key+'.csv')
        #print('CORRELATION END...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        

    run_attribute()
    run_correlation()
    run_discriminator()
    run_task()

   
    print('END...', datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))