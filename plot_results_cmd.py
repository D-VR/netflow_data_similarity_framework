from os import listdir, makedirs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

TARGET_ATTRIBUTES = [ 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
                            'PROTOCOL', 
                            #'L7_PROTO', #not used here
                            'IN_BYTES', 'OUT_BYTES', 
                            'IN_PKTS', 'OUT_PKTS',
                            'TCP_FLAGS', 
                            'FLOW_DURATION_MILLISECONDS', 
                            'Label', 
                            #'Attack' #not used here
                            ]

def remove_error_rows(df):
    print(df)
    new_df_list = []

    for row in df.itertuples():
        error_row = False
        for val in row[5:-1]:
            if val < 0:
                error_row=True
                #print(row)
        if error_row==False:
            new_df_list.append(row[1:]) #do not add old index (row[0])
        error_row=False

    new_df = pd.DataFrame(new_df_list, columns=df.columns)
    #print(new_df)
    return new_df

#TODO can this be paralllized?
def aggregate_raw_results(dir_path, save_path, model):
    
    files_attributes = ['JS.csv', 'WD.csv']
    files_correlation = [
        ('pearson_1.csv', 'pearson_2.csv'),
        ('corr_ratio_1.csv', 'corr_ratio_2.csv'),
        ('theils_u_1.csv', 'theils_u_2.csv'), 
    ]
    files_disc = [
     #['disc_tr-ds1_tst-ds2.csv', 'disc_tr-ds2_tst-ds1.csv', 'disc_y_ds1.csv', 'disc_y_ds2.csv'], 
        ['if_disc_tr-ds1_tst-ds2.csv', 'if_disc_tr-ds2_tst-ds1.csv', 'if_disc_y_ds1.csv', 'if_disc_y_ds2.csv'],
        ['ocsvm_disc_tr-ds1_tst-ds2.csv', 'ocsvm_disc_tr-ds2_tst-ds1.csv', 'ocsvm_disc_y_ds1.csv', 'ocsvm_disc_y_ds2.csv'], 

    ]
    files_task = [
     #['task_tr-ds1_tst-ds2.csv', 'task_tr-ds2_tst-ds1.csv', 'task_y_ds1.csv', 'task_y_ds2.csv'], 
     ['if_task_tr-ds1_tst-ds2.csv', 'if_task_tr-ds2_tst-ds1.csv', 'if_task_y_ds1.csv', 'if_task_y_ds2.csv'],
     ['ocsvm_task_tr-ds1_tst-ds2.csv', 'ocsvm_task_tr-ds2_tst-ds1.csv', 'ocsvm_task_y_ds1.csv', 'ocsvm_task_y_ds2.csv'],
     ['xgb_task_tr-ds1_tst-ds2.csv', 'xgb_task_tr-ds2_tst-ds1.csv', 'xgb_task_y_ds1.csv', 'xgb_task_y_ds2.csv'], 

    ]
    print("AGGREGATE RAW DATA:", dir_path)
    list_all_results = []
    ds_dirs = listdir(dir_path)
    #print(dirs)
    for dir in tqdm(ds_dirs[:]):
        dict_results = {}
        #print(dir)
        try:
            name_1, s_1, name_2, s_2 = dir.split('_')
        except:
            name_1, seed, seednum, step_word, s_1, name_2, s_2 = dir.split('_')

        dict_results.update({'ds_1' : name_1, 'sample_1':s_1, 'ds_2' : name_2, 'sample_2':s_2})
        #attributes
        for file in files_attributes:
             name = file.split('.')[0]
             try:
                f = pd.read_csv(dir_path+'/'+dir+'/'+file, index_col=0, header=0)
                #TODO filter for relevant attributes: 
                f = f[f.index.isin(TARGET_ATTRIBUTES)]
                mean =  f['0'].values.mean()
             except:
                 mean = -1
             dict_results.update({name : mean})

        for file1, file2 in files_correlation:
            #print(file1, file2)
            name = file1.split('.')[0][:-2]
            #print(name)
            try:
                f1 = pd.read_csv(dir_path+'/'+dir+'/'+file1, index_col=0, header=0)
                f2 = pd.read_csv(dir_path+'/'+dir+'/'+file2, index_col=0, header=0)

                mae = mean_absolute_error(f1.values.flatten(), f2.values.flatten())
            except:
                mae = -1

            dict_results.update({name : mae})

        for ds1ds2, ds2ds1, ds1y, ds2y in files_disc:
            name_ds1ds2 = ds1ds2.split('.')[0] + '_fpr'
            name_ds2ds1 = ds2ds1.split('.')[0] + '_fpr'

            try:
     
                f_ds1ds2 = pd.read_csv(dir_path+'/'+dir+'/'+ds1ds2, index_col=0, header=0)
                f_ds2ds1 = pd.read_csv(dir_path+'/'+dir+'/'+ds2ds1, index_col=0, header=0)
                f_ds1y = pd.read_csv(dir_path+'/'+dir+'/'+ds1y, index_col=0, header=0)
                f_ds2y = pd.read_csv(dir_path+'/'+dir+'/'+ds2y, index_col=0, header=0)

                f_ds1ds2 = [1.0 if val > 0 else -1.0 for val in f_ds1ds2.values ]
                f_ds2ds1 = [1.0 if val > 0 else -1.0 for val in f_ds2ds1.values ]


                #print(f_ds2y, f_ds1ds2)
                cm = confusion_matrix(f_ds2y, f_ds1ds2,  labels=[-1,1])
                tn, fp, fn, tp  = cm.ravel() #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
                fpr = fp / (fp+tn)
                dict_results.update({name_ds1ds2 : fpr})

                cm = confusion_matrix(f_ds1y, f_ds2ds1,  labels=[-1,1])
                tn, fp, fn, tp  = cm.ravel() #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
                fpr = fp / (fp+tn)
                dict_results.update({name_ds2ds1 : fpr})
            except:
                    dict_results.update({name_ds1ds2 : -1})
                    dict_results.update({name_ds2ds1 : -1})



        for ds1ds2, ds2ds1, ds1y, ds2y in files_task:
            name_ds1ds2 = ds1ds2.split('.')[0] + '_f1'
            name_ds2ds1 = ds2ds1.split('.')[0] + '_f1'

            try:
     
                f_ds1ds2 = pd.read_csv(dir_path+'/'+dir+'/'+ds1ds2, index_col=0, header=0)
                f_ds2ds1 = pd.read_csv(dir_path+'/'+dir+'/'+ds2ds1, index_col=0, header=0)
                f_ds1y = pd.read_csv(dir_path+'/'+dir+'/'+ds1y, index_col=0, header=0)
                f_ds2y = pd.read_csv(dir_path+'/'+dir+'/'+ds2y, index_col=0, header=0)

                #print(f_ds2ds1.shape, f_ds1ds2.shape, f_ds1y.shape, f_ds2y.shape)

                f_ds1ds2 = [1.0 if val > 0 else -1.0 for val in f_ds1ds2.values ]
                f_ds2ds1 = [1.0 if val > 0 else -1.0 for val in f_ds2ds1.values ]


                #print(f_ds2y, f_ds1ds2)
                average= 'macro'
                f1_sc =  f1_score(f_ds2y, f_ds1ds2,  labels=[-1,1], average=average),
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                
                f1_sc =  f1_score(f_ds1y, f_ds2ds1,  labels=[-1,1], average=average),
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})

                average= 'micro'
                f1_sc =  f1_score(f_ds2y, f_ds1ds2,  labels=[-1,1], average=average),
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                
                f1_sc =  f1_score(f_ds1y, f_ds2ds1,  labels=[-1,1], average=average),
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})

                average= 'weighted'
                f1_sc =  f1_score(f_ds2y, f_ds1ds2,  labels=[-1,1], average=average),
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                
                f1_sc =  f1_score(f_ds1y, f_ds2ds1,  labels=[-1,1], average=average),
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})
            except:
                f1_sc = -1
                average= 'macro'
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})
                average= 'micro'
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})
                average= 'weighted'
                dict_results.update({name_ds1ds2+'-'+average : f1_sc})
                dict_results.update({name_ds2ds1+'-'+average : f1_sc})
             

        #print(dict_results)
        list_all_results.append(pd.DataFrame(dict_results, index=[0]))
    df_results = pd.concat(list_all_results, ignore_index=True)
    df_results['model']=model
    df_results.to_csv(save_path, header=True, index=False)

    #quit()

def split_ds_file(df):
        df['ds'] = df['file'].str.split('_').str[0]
        df['step'] = df['file'].str.split('_').str[1].str.split('-').str[1].str.split('.').str[0].astype(int)
        df = df.set_index('step')
        return df.sort_index()

if __name__ == '__main__':
    
    ### aggregate raw results into single files ###
    prefix = '../../Datasets/Queensland_NetFlow/results_raw/'
    experiment_id = "exp_04-real"
    load_path_real = prefix+experiment_id+'/'
    save_path_real=  prefix+experiment_id+'_results_raw.csv'
    #aggregate_raw_results(load_path_real, save_path_real, model='real')
     
    prefix = '../../Datasets/Queensland_NetFlow/results_raw/'
    #experiment_id = "exp_01-gpt2"
    experiment_id = "exp_05-gpt2_history"

    load_path_gpt = prefix+experiment_id+'/'
    save_path_gpt=  prefix+experiment_id+'_results_raw.csv'
    #aggregate_raw_results(load_path_gpt, save_path_gpt, model='gpt2')   
    
    '''
    prefix = '../../Datasets/Queensland_NetFlow/results_raw/'
    #experiment_id = "exp_01-wgan"
    experiment_id = "exp_01-wgan_history"
    load_path_wgan = prefix+experiment_id+'/'
    save_path_wgan =  prefix+experiment_id+'_results_raw.csv'
    #aggregate_raw_results(load_path_wgan, save_path_wgan, model='wgan')
    '''

    prefix = '../../Datasets/Queensland_NetFlow/results_raw/'
    #experiment_id = "exp_01-wgan"
    experiment_id = "exp_05-wganbin_history"
    load_path_wgan = prefix+experiment_id+'/'
    save_path_wgan =  prefix+experiment_id+'_results_raw.csv'
    aggregate_raw_results(load_path_wgan, save_path_wgan, model='wgan-binary')

    ### plot data ###

    #load data
    df_results_real = pd.read_csv(save_path_real, header=0)
    prefix_plot = prefix + 'plots_exp_05/'
    makedirs(prefix_plot, exist_ok=True)
    prefix_plot_detail = prefix + 'plots_detail/'
    makedirs(prefix_plot_detail, exist_ok=True)
    prefix_plot_classifier = prefix + 'plots_classifier/'
    makedirs(prefix_plot_classifier, exist_ok=True)
    
    #create_detailed_plots(load_path_gpt, prefix_plot_detail, model='gpt2')
    #create_detailed_plots(load_path_wgan, prefix_plot_detail, model='wgan')

    #TODO combine classfication results with orignal data and analyze/plot
    #crate list of ds_paths + classfication results
    
    #real_path = '../../Datasets/Queensland_NetFlow/samples/real/'
    #gen_path = '../../Datasets/Queensland_NetFlow/synthetic/gpt2_data_checked/'
    #gen_path = '../../Datasets/Queensland_NetFlow/synthetic/wgan_data_checked/'
    #create_classfication_report(load_path_gpt, prefix_plot_classifier, 'gpt2', gen_path, real_path)

    
    save_plot_real = prefix_plot+'results_real_'
    df_results_gpt = pd.read_csv(save_path_gpt, header=0)
    df_results_wgan = pd.read_csv(save_path_wgan, header=0)
    
    df_results_gpt = remove_error_rows(df_results_gpt) #necessary? corr measures can be negative
    df_results_wgan = remove_error_rows(df_results_wgan)

    #print(df_results_wgan)
    #print(df_results_wgan[df_results_wgan['sample_1']=='step-0500'])
    #quit()


    #TODO remove values of -1 (error code values, when there was no file)

    df_results_real['WD'] *= 10
    df_results_gpt['WD'] *= 10
    df_results_wgan['WD'] *= 10

    rename_dict = {
                   'JS':'Jensen-Shannon Div. Mean', 
                   'WD': 'Wasserstein Dist. Mean x10', 
                   'pearson': 'Pearson MAE',
                    'corr_ratio': 'Correlation Ratio MAE', 
                    'theils_u': 'Theils U MAE', 
                    'if_disc_tr-ds1_tst-ds2_fpr': 'IF Discriminator TSTR FPR',
                    'if_disc_tr-ds2_tst-ds1_fpr': 'IF Discriminator TRTS FPR',

                    'ocsvm_disc_tr-ds1_tst-ds2_fpr': 'OCSVM Discriminator TSTR FPR',
                    'ocsvm_disc_tr-ds2_tst-ds1_fpr': 'OCSVM Discriminator TRTS FPR',  
                    
                    
                    'if_task_tr-ds1_tst-ds2_f1-macro' : 'IF Task TSTR F1-Score-Macro',
                    'if_task_tr-ds2_tst-ds1_f1-macro': 'IF Task TRTS F1-Score-Macro',
                    'if_task_tr-ds1_tst-ds2_f1-micro': 'IF Task TSTR F1-Score-Micro',
                    'if_task_tr-ds2_tst-ds1_f1-micro': 'IF Task TRTS F1-Score-Micro', 
                    'if_task_tr-ds1_tst-ds2_f1-weighted': 'IF Task TSTR F1-Score-Weighted',
                    'if_task_tr-ds2_tst-ds1_f1-weighted': 'IF Task TRTS F1-Score-Weighted',

                    'ocsvm_task_tr-ds1_tst-ds2_f1-macro' : 'OCSVM Task TSTR F1-Score-Macro',
                    'ocsvm_task_tr-ds2_tst-ds1_f1-macro': 'OCSVM Task TRTS F1-Score-Macro',
                    'ocsvm_task_tr-ds1_tst-ds2_f1-micro': 'OCSVM Task TSTR F1-Score-Micro',
                    'ocsvm_task_tr-ds2_tst-ds1_f1-micro': 'OCSVM Task TRTS F1-Score-Micro', 
                    'ocsvm_task_tr-ds1_tst-ds2_f1-weighted': 'OCSVM Task TSTR F1-Score-Weighted',
                    'ocsvm_task_tr-ds2_tst-ds1_f1-weighted': 'OCSVM Task TRTS F1-Score-Weighted',

                    'xgb_task_tr-ds1_tst-ds2_f1-macro' : 'XGB Task TSTR F1-Score-Macro',
                    'xgb_task_tr-ds2_tst-ds1_f1-macro': 'XGB Task TRTS F1-Score-Macro',
                    'xgb_task_tr-ds1_tst-ds2_f1-micro': 'XGB Task TSTR F1-Score-Micro',
                    'xgb_task_tr-ds2_tst-ds1_f1-micro': 'XGB Task TRTS F1-Score-Micro', 
                    'xgb_task_tr-ds1_tst-ds2_f1-weighted': 'XGB Task TSTR F1-Score-Weighted',
                    'xgb_task_tr-ds2_tst-ds1_f1-weighted': 'XGB Task TRTS F1-Score-Weighted',
    }

    df_results_synthtic = pd.concat([df_results_gpt, df_results_wgan], axis=0, ignore_index=True)
    df_results_real.rename(columns=rename_dict, inplace=True)
    df_results_synthtic.rename(columns=rename_dict, inplace=True)
    store_path = prefix_plot
    #benchmark_line_plot_raw_all(df_results_real, df_results_synthtic, store_path)
    #benchmark_line_plot_raw_hist(df_results_real, df_results_synthtic, store_path)
    #benchmark_line_plot_raw_all_history(df_results_real, df_results_synthtic, store_path)

    #TODO combine error counts to aggregation of results
    #get error counts
    path_gpt = './test_data/gpt2_error_counts.csv'
    df_gpt_error = pd.read_csv(path_gpt, header=0)
    df_gpt_error = split_ds_file(df_gpt_error)
    print(df_gpt_error)
    
    path_wgan = './test_data/wgan_error_counts.csv'
    df_wgan_error = pd.read_csv(path_wgan, header=0)
    df_wgan_error = split_ds_file(df_wgan_error)
    print(df_wgan_error)

    df_gpt_error['model'] = 'gpt2'
    df_wgan_error['model'] = 'wgan-binary'

    df_combi_error = pd.concat([df_gpt_error, df_wgan_error], axis=0)

    print(df_results_real)
    print(df_results_synthtic)
    print(df_combi_error)

    '''
    plot_history_mean_metric(df_results_real.copy(), df_results_synthtic.copy(), store_path+'mean_', df_combi_error)



    #quit()

    #ds_dict_name = { 'NF-CSE-CIC-IDS2018':'A',
    #             'NF-ToN-IoT': 'B',
    #             'NF-UNSW-NB15': 'C'}
    ds_dict_name = { 'NF-CSE-CIC-IDS2018':'CC',
                 'NF-ToN-IoT': 'TI',
                 'NF-UNSW-NB15': 'UN'}
    for key in ds_dict_name.keys():
        df_results_real['ds_1'] = df_results_real['ds_1'].str.replace(key, ds_dict_name[key])
        df_results_real['ds_2'] = df_results_real['ds_2'].str.replace(key, ds_dict_name[key])
    
    print(df_results_real)
    plot_boxplot_real(df_results_real.copy(), save_plot_real)

    plot_boxplot_real_mean_metrics(df_results_real.copy(), save_plot_real+'scores_')
    metric_correlation_all(df_results_real.copy(), save_plot_real+'metrics_')
    metric_correlation(df_results_real.copy(), save_plot_real+'metrics_')
'''