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