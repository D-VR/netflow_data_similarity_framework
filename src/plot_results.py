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

        
def plot_history_mean_metric(df_real, df_synthetic, store_path):
    def calc_mean(df_results):
        mean_columns = ['Jensen-Shannon Div. Mean', 
                        #'Wasserstein Dist. Mean x10', 
                        'Pearson MAE',
                        'Correlation Ratio MAE', 
                        'Theils U MAE', 
                        
                        'IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR', 
                        #'Task TSTR F1-Score-Macro',
                        #'Task TRTS F1-Score-Macro',
                        #'Task TSTR F1-Score-Micro',
                        #'Task TRTS F1-Score-Micro', 
                        'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
        ]

        invert_cols = ['IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR', 
                        #'Task TSTR F1-Score-Macro',
                        #'Task TRTS F1-Score-Macro',
                        #'Task TSTR F1-Score-Micro',
                        #'Task TRTS F1-Score-Micro', 
                        'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
        ]
        
        data_metrics = ['Jensen-Shannon Div. Mean', 
                        #'Wasserstein Dist. Mean x10', 
                        'Pearson MAE',
                        'Correlation Ratio MAE', 
                        'Theils U MAE', 
                        
                        'IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR',  ]
                    
        domain_metrics = [ 'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
                        'Error_Flows']
                    
        #invert values
        df_results[invert_cols] = 1- df_results[invert_cols]
        df_results['Data Dissimilarity Score'] = df_results[data_metrics].mean(axis=1)
        df_results['Domain Dissimilarity Score'] = df_results[domain_metrics].mean(axis=1)

        df_results['Data Dissimilarity Score_std'] = df_results[data_metrics].std(axis=1)
        df_results['Domain Dissimilarity Score_std'] = df_results[domain_metrics].std(axis=1)
        print(df_results)
        df_results = df_results[['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model', 'Data Dissimilarity Score', 'Domain Dissimilarity Score', 'Data Dissimilarity Score_std', 'Domain Dissimilarity Score_std']]
        #create mean for data and task(with syntax errors) metrics, seperate plots
        return df_results
    #print("START")
    #print(df_synthetic)
    #quit()
    #print(df_synthetic[df_synthetic['sample_1']=='step-0500'])
    #TODO combine similarity with error flows rate..

    #df_error['ds_1'] = df_error['ds']
    #df_error=df_error.reset_index()
    #df_error['sample_1'] = 0#df_error['step']
    #df_synthetic['Error_Flows'] = df_synthetic['Error_Flows']/10000
    #print(df_error)
    #print(df_error.columns)
    #df_error = df_error[['ds_1', 'sample_1', 'model', 'Error_Flows']]
    #print('Error')
    #print(df_error)
    #print(df_error.columns)
    #df_error=df_error.drop(index=list(range(500,10000, 1000))) #drop intermediate steps
    #print(df_error)

    #df_synthetic['sample_1'] = df_synthetic['sample_1'].str.split('-').str[1].astype(int) #remove step- prefix
    #still necessary
    #print("Synthetic")
    #print(df_synthetic[['ds_1', 'model', 'sample_1']])
    #df_synthetic = df_synthetic.merge(df_error, how='inner', left_on=['ds_1', 'model', 'sample_1'], right_on=['ds_1', 'model', 'sample_1'])

    #df_real['Error_Flows'] = 0 #real data have no syntax errors
    #print("Synthetic merge")
    #print(df_synthetic[['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model', 'Error_Flows']])
    #print(df_synthetic.columns)
    #quit()


    df_real = calc_mean(df_real)
    #print(df_synthetic)
    #print(df_synthetic[df_synthetic['sample_1']==1000])
    #quit()
    df_synthetic = calc_mean(df_synthetic)
    #print("DF_SYNTETIC")
    #print(df_synthetic)
    #print(df_synthetic[df_synthetic['sample_1']==1000])

    df_real.to_csv(store_path+'metrics_real.csv')
    df_synthetic.to_csv(store_path+'metrics_synthetic.csv')

    #quit()
    #df_synthetic['error_flows'] = np.where( ( (df_synthetic['ds_1']==df_error['ds']) & (df_synthetic['model']==df_error['model']) & (df_synthetic['sample_1']==df_error.index)), df_error['Flows'] )
    #df_synthetic['error_flows'] = np.where( ( (df_synthetic['ds_1']==df_error['ds']) & (df_synthetic['model']==df_error['model']) & (df_synthetic['sample_1']==df_error.index)), df_error['Flows'] )

    #df_synthetic['error_flows'] = np.where( ( (df_synthetic['ds_1']==df_error['ds']) & (df_synthetic['model']==df_error['model']) & (df_synthetic['sample_1']==df_error.index)), df_error['Flows'] )

    

    # plotting
    def benchmark_line_plot_history(df_real, synthetic_data_points, store_path):
        name_list = df_real.columns.drop(['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model'])
        name_list = [n for n in name_list if not ('std' in n)] #remove std values, will be used later
        #print(name_list)
        #quit()

        for ds_group in synthetic_data_points['ds_1'].unique():
            df_synthetic_filter = synthetic_data_points[synthetic_data_points['ds_1'].str.contains(ds_group, case=False) & synthetic_data_points['ds_2'].str.contains(ds_group, case=False)]
            df_real_filter = df_real[df_real['ds_1'].str.contains(ds_group, case=False) & df_real['ds_2'].str.contains(ds_group, case=False)]
            
            #df_real_filter_other = df_real[(~ df_real['ds_1'].str.contains(ds_group, case=False)) &  (~ df_real['ds_2'].str.contains(ds_group, case=False))]
            df_real_filter_other = df_real[df_real['ds_1'].str.contains(ds_group, case=False) ^ df_real['ds_2'].str.contains(ds_group, case=False)]
            #df_real_filter_other = df_real

            print("TARGET:")
            print(df_real_filter['ds_1'].unique())
            print(df_real_filter['ds_2'].unique())
            print('OTHER')
            print(df_real_filter_other['ds_1'].unique())
            print(df_real_filter_other['ds_2'].unique())

            df_real_stats = df_real_filter.describe()
            df_real_stats_other = df_real_filter_other.describe()
            #print(df_real_stats)
            #print(df_real_stats_other)

            #TODO filter important data firest (thresholds + datapoints)
            #plot afterwareds

            #fig, axs = plt.subplots(len(name_list))
            fig = plt.figure(figsize = (3,7))
            #plt.yticks([0, 0.25,0.5,0.75,1.0])

            gs = fig.add_gridspec(len(name_list), hspace=0.4)
            #axs = gs.subplots(sharex=True, sharey=True)
            axs = gs.subplots(sharex=True, sharey=False)
            #print(axs)
            #quit()
            #axs = [axs]


            for axs_num, col in enumerate(name_list):
                print(col)
                val_25 = df_real_stats.loc['25%', col]
                val_75 = df_real_stats.loc['75%', col]
                val_mean = df_real_stats.loc['mean', col]
                val_min = df_real_stats.loc['min', col]
                val_max = df_real_stats.loc['max', col]

                val_25_other = df_real_stats_other.loc['25%', col]
                val_75_other = df_real_stats_other.loc['75%', col]
                val_mean_other = df_real_stats_other.loc['mean', col]
                val_min_other = df_real_stats_other.loc['min', col]
                val_max_other = df_real_stats_other.loc['max', col]
                #print(val_25)
                #fig = plt.figure(figsize=(8,8))
                tmp = df_synthetic_filter[['sample_1', col, col+'_std', 'model']]
                print(tmp)
                tmp.set_index('sample_1', inplace=True)
                print(tmp)
                
                axs[axs_num].tick_params(left = True, right = False , labelleft = True ,
                    labelbottom = True, bottom = False)
                
                #remove spine lines
                axs[axs_num].spines['right'].set_visible(False)
                axs[axs_num].spines['left'].set_visible(True)
                axs[axs_num].spines['top'].set_visible(False)
                axs[axs_num].spines['bottom'].set_visible(True)
                #plot_height = -0.1

                pmin = 0
                pmax=10000
                #pmax = plot_height*1.1

                #TODO annotate lines

                dates = list(tmp[col].values) + [val_25_other, val_75_other, val_mean_other, val_25, val_75, val_mean]
                names = ['WGAN', 'GPT', '25%', '75%', 'Mean', '25%', '75%', 'Mean']

                # annotate lines
                zip_list = zip(dates, names)
                zip_list_sorted = sorted(zip_list, key=lambda dat: dat[0]) #sort by dates
                print(zip_list_sorted)
                #quit()

                #axs[axs_num].plot(dates, np.zeros_like(dates),
                #    color="k")  # Baseline and markers on it.

                axs[axs_num].fill_between([0, 10000], val_min_other, val_max_other, facecolor='y', alpha=0.1)
                axs[axs_num].fill_between([0, 10000], val_25_other, val_75_other, facecolor='y', alpha=0.25)
                axs[axs_num].fill_between([0, 10000], val_min, val_max, facecolor='g', alpha=0.1)
                axs[axs_num].fill_between([0, 10000], val_25, val_75, facecolor='g',alpha=0.25)

                axs[axs_num].hlines(val_25_other, pmin, pmax, linestyles='dashed', colors='y')
                axs[axs_num].hlines(val_75_other, pmin, pmax, linestyles='dashed', colors='y')
                axs[axs_num].hlines(val_mean_other, pmin, pmax, colors='y')
                axs[axs_num].hlines(val_min_other, pmin, pmax, linestyles='dotted', colors='y')
                axs[axs_num].hlines(val_max_other, pmin, pmax, linestyles='dotted', colors='y')

                axs[axs_num].hlines(val_25, pmin, pmax, linestyles='dashed', colors='g')
                axs[axs_num].hlines(val_75, pmin, pmax, linestyles='dashed', colors='g')
                axs[axs_num].hlines(val_mean, pmin, pmax, colors='g')
                axs[axs_num].hlines(val_min, pmin, pmax, linestyles='dotted', colors='g')
                axs[axs_num].hlines(val_max, pmin, pmax, linestyles='dotted', colors='g')

            

                for model in tmp['model'].unique():
                    tmp_tmp = tmp[tmp['model']==model]
                    #tmp_tmp.index = tmp_tmp.index.str.split('-').str[1].astype(int)
                    tmp_tmp = tmp_tmp.sort_index()
                    print(tmp_tmp[col].values)
                    #axs[axs_num].plot(tmp_tmp.index/-100000, tmp_tmp[col].values,  marker='x', linewidth=1, label=model, markersize=3)
                    axs[axs_num].plot(tmp_tmp.index, tmp_tmp[col].values,  marker='x', linewidth=1, label=model, markersize=5)
                    #axs[axs_num].fill_between(tmp_tmp.index, tmp_tmp[col].values - tmp_tmp[col+'_std'].values, tmp_tmp[col].values + tmp_tmp[col+'_std'].values, alpha=0.2)
                    #axs[axs_num].yticks([0, 0.25,0.5,0.75,1.0])
                    axs[axs_num].set_ylim([0, 1])
                    axs[axs_num].grid(True)
                    axs[axs_num].set_xlabel('training step')
                    axs[axs_num].set_ylabel('score value')

                
                axs[axs_num].set_title(col, fontsize='medium')#, pad=20)
                save_title = ds_group+'__'+col

            #plt.yticks([0, 0.25,0.5,0.75,1.0])
            plt.suptitle(ds_group, fontsize='large')
            plt.tight_layout()
            plt.subplots_adjust(left=0.2, right=0.95, top=0.88, bottom=0.15)
            plt.legend(ncol=2, loc=(0, -0.4))
            #plt.legend()
            #plt.show()
            save_title = ds_group+'__all'
            plt.savefig(store_path+'mean_history_threshold_'+save_title+'-'+str(10000)+'.pdf')
    #add errors? normalize values to 0-1 shall we add weights?
    benchmark_line_plot_history(df_real, df_synthetic, store_path)


def combine_raw_error(df, df_error):
    df_error.drop(columns=['file'], inplace=True)
    df_error.rename(columns={'ds':'ds_1', 'step':'sample_1'}, inplace=True)
    if df['sample_1'].dtype != np.int64:
        #remove step-
        df['sample_1'] = df['sample_1'].astype(str).str[5:].astype(np.int64)
    #merge raw results + syntax checks for each (real, wgan, gpt) individually
    df_error = df_error.rename(columns={'Flows':'Error_Flows'})
    df_error=df_error.reset_index()
    df_error = df_error[['ds_1', 'sample_1', 'model', 'Error_Flows']]
    df_error['Error_Flows'] = df_error['Error_Flows']/10000
    print(df.columns)
    print(df[['ds_1', 'model', 'sample_1']])
    print(df_error.columns)
    print(df_error[['ds_1', 'model', 'sample_1']])
    df = df.merge(df_error, how='inner', left_on=['ds_1', 'model', 'sample_1'], right_on=['ds_1', 'model', 'sample_1'])
    return df


def create_heatmap(df, title, figsize=(20,5), save_path=''):
    #plt.pcolor(df)
    #plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    #plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    #fig = plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)

    sns.set(font_scale=0.7)
    #axs = sns.heatmap(df, annot=True, fmt=".2f", cmap='Greens', linewidth=1, vmin=-1, vmax=1)
    axs = sns.heatmap(df, annot=True, fmt=".2f", cmap='RdBu', linewidth=1, vmin=-1, vmax=1)

    fig.add_axes(axs)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, horizontalalignment='right')
    #axs.set_yticklabels(axs.get_yticklabels(), rotation=45, horizontalalignment='right')
    #plt.xticks(rotation=45, horizontalalignment='right')
    #plt.yticks(rotation=45, horizontalalignment='right')

    #plt.title(title,fontsize=20)
    plt.tight_layout()
    #plt.show()
    fig.savefig(save_path+'correlation'+title+'.pdf')

def metric_correlation_all(df_results, save_path):
    invert_cols = ['IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR', 
                        #'Task TSTR F1-Score-Macro',
                        #'Task TRTS F1-Score-Macro',
                        #'Task TSTR F1-Score-Micro',
                        #'Task TRTS F1-Score-Micro', 
                        'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
        ]
        
    data_metrics = ['Jensen-Shannon Div. Mean', 
                        #'Wasserstein Dist. Mean x10', 
                        'Pearson MAE',
                        'Correlation Ratio MAE', 
                        'Theils U MAE', 
                        
                        'IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR',  ]
                    
    domain_metrics = [ 'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
                        #'Error_Flows'
                        ]
                
    #invert values
    df_results[invert_cols] = 1- df_results[invert_cols]
    #df_results['Error_Flows'] = 0

    #df_results['Data Similarity Score'] = df_results[data_metrics].mean(axis=1)
    #df_results['Domain Similarity Score'] = df_results[domain_metrics].mean(axis=1)

    #df_results = df_results[['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model', 'Data Similarity Score', 'Domain Similarity Score']]
    
    #filter ds
    df_filter = df_results


    ### plotting ###

    name_list = df_filter.columns.drop(['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model'])
    
    df_corr_data = df_filter[data_metrics].corr()
    df_corr_domain = df_filter[domain_metrics].corr()
    df_corr = df_filter[data_metrics+domain_metrics].corr()

    print(df_corr_data)
    print(df_corr_domain)

    #create_heatmap(df_corr_data, 'Correlation Data Metrics', figsize=(5,5), save_path=save_path)
    #create_heatmap(df_corr_domain, 'Correlation Domain Metrics', figsize=(5,5), save_path=save_path)
    create_heatmap(df_corr, 'Metrics Correlation - all DS', figsize=(7,6), save_path=save_path+'all_')


def plot_boxplot_real_mean_metrics(df_results, save_path):
    '''
    invert_cols = ['Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                    'Discriminator TRTS FPR', 
                    #'Task TSTR F1-Score-Macro',
                    #'Task TRTS F1-Score-Macro',
                    #'Task TSTR F1-Score-Micro',
                    #'Task TRTS F1-Score-Micro', 
                    'Task TSTR F1-Score-Weighted',
                    'Task TRTS F1-Score-Weighted',]
        
    data_metrics = ['Jensen-Shannon Div. Mean', 
                    #'Wasserstein Dist. Mean x10', 
                    'Pearson MAE',
                    'Correlation Ratio MAE', 
                    'Theils U MAE', 
                    
                    'Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                    'Discriminator TRTS FPR', ]
                
    domain_metrics = [ 'Task TSTR F1-Score-Weighted',
                    'Task TRTS F1-Score-Weighted',
                    'Error_Flows'
                    ]
    
    '''

    invert_cols = ['IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR', 
                        #'Task TSTR F1-Score-Macro',
                        #'Task TRTS F1-Score-Macro',
                        #'Task TSTR F1-Score-Micro',
                        #'Task TRTS F1-Score-Micro', 
                        'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
        ]
        
    data_metrics = ['Jensen-Shannon Div. Mean', 
                        #'Wasserstein Dist. Mean x10', 
                        'Pearson MAE',
                        'Correlation Ratio MAE', 
                        'Theils U MAE', 
                        
                        'IF Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'IF Discriminator TRTS FPR', 
                        'OCSVM Discriminator TSTR FPR', #calc 1-metric for value where 0-1, 0 is better
                        'OCSVM Discriminator TRTS FPR',  ]
                    
    domain_metrics = [ 'IF Task TSTR F1-Score-Weighted',
                        'IF Task TRTS F1-Score-Weighted',
                        'OCSVM Task TSTR F1-Score-Weighted',
                        'OCSVM Task TRTS F1-Score-Weighted',
                        'XGB Task TSTR F1-Score-Weighted',
                        'XGB Task TRTS F1-Score-Weighted',
                        'Error_Flows']
                
    #invert values
    df_results[invert_cols] = 1- df_results[invert_cols]
    df_results['Error_Flows'] = 0

    df_results['Data Dissimilarity Score'] = df_results[data_metrics].mean(axis=1)
    df_results['Domain Dissimilarity Score'] = df_results[domain_metrics].mean(axis=1)

    df_results = df_results[['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model', 'Data Dissimilarity Score', 'Domain Dissimilarity Score']]

    ### plotting ###

    name_list = df_results.columns.drop(['ds_1', 'sample_1', 'ds_2', 'sample_2', 'model'])
    
    for value in name_list:
        df_results.boxplot(column=[value], by=['ds_1', 'ds_2'], rot=0, figsize=(3,3), vert=False)#, sharey=False) figsize=(8,3)
        #df_results.boxplot(column=[value], by=['ds_2', 'ds_1'], rot=0, figsize=(8,5), vert=False)#, sharey=False) figsize=(8,3)

        plt.subplots_adjust(left=0.35)
        #fig = plt.figure()
        
        plt.title('')
        plt.suptitle('')
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()
        #plt.show()
        plt.savefig(save_path+'_boxplot_'+value+'.pdf')
    #plt.tight_layout()
    #plt.show()

#if __name__ == '__main__':
def plot_results(prefix):
    
    ### load data
    #prefix='test_data/' #set via cmd
    save_path_real = prefix+'04-real_aggregate.csv'
    save_path_gpt = prefix+'04-gpt2_aggregate.csv'
    save_path_wgan = prefix+'04-wganbin_aggregate.csv'

    ### plot data ###

    #load data
    df_results_real = pd.read_csv(save_path_real, header=0)
    prefix_plot = prefix + 'plots/'
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

                    'Flows' : 'Error_Flows'
    }

    
    store_path = prefix_plot
    #benchmark_line_plot_raw_all(df_results_real, df_results_synthtic, store_path)
    #benchmark_line_plot_raw_hist(df_results_real, df_results_synthtic, store_path)
    #benchmark_line_plot_raw_all_history(df_results_real, df_results_synthtic, store_path)

    #TODO combine error counts to aggregation of results
    #get error counts
    #path_real = prefix+'04-real_aggregate_syntax.csv'
    #df_real_error = pd.read_csv(path_real, header=0)
    #df_gpt_error = split_ds_file(df_gpt_error)
    #print(df_real_error)

    path_gpt = prefix+'04-gpt2_aggregate_syntax.csv'
    df_gpt_error = pd.read_csv(path_gpt, header=0)
    #df_gpt_error = split_ds_file(df_gpt_error)
    print(df_gpt_error)
    
    path_wgan = prefix+'/04-wganbin_aggregate_syntax.csv'
    df_wgan_error = pd.read_csv(path_wgan, header=0)
    #df_wgan_error = split_ds_file(df_wgan_error)
    print(df_wgan_error)

    #combine raw+errors
    #df_results_real = combine_raw_error(df_results_real, df_real_error)
    df_results_real['Error_Flows'] = 0
    print(df_results_real)

    df_results_gpt = combine_raw_error(df_results_gpt, df_gpt_error)
    print(df_results_gpt)
    
    df_results_wgan = combine_raw_error(df_results_wgan, df_wgan_error)
    print(df_results_wgan)
    #quit()
    df_results_gpt['model'] = 'gpt2'
    df_results_wgan['model'] = 'wgan-binary'

    #df_combi_error = pd.concat([df_gpt_error, df_wgan_error], axis=0).reset_index()

    df_results_synthtic = pd.concat([df_results_gpt, df_results_wgan], axis=0, ignore_index=True)
    df_results_real.rename(columns=rename_dict, inplace=True)
    df_results_synthtic.rename(columns=rename_dict, inplace=True)

    print('real')
    print(df_results_real)
    print('synthetic')
    print(df_results_synthtic)
    #print('error')
    #df_combi_error.drop(columns=['file'], inplace=True)
    #df_combi_error.rename(columns={'ds':'ds_2', 'step':'sample_2'}, inplace=True)
    #df_combi_error['sample_2'] = 'step-' + df_combi_error['sample_2'].astype(str)
    #print(df_combi_error)


    #merge raw results + syntax checks for each (real, wgan, gpt) individually
    #df_results_synthtic = df_results_synthtic.merge(df_combi_error, how='inner', left_on=['ds_2', 'model', 'sample_2'], right_on=['ds_2', 'model', 'sample_2'])
    #print(df_results_synthtic)
    
    plot_history_mean_metric(df_results_real.copy(), df_results_synthtic.copy(), store_path+'mean_')
 


    
    ds_dict_name = { 'NF-CSE-CIC-IDS2018':'CC',
                 'NF-ToN-IoT': 'TI',
                 'NF-UNSW-NB15': 'UN'}
    for key in ds_dict_name.keys():
        df_results_real['ds_1'] = df_results_real['ds_1'].str.replace(key, ds_dict_name[key])
        df_results_real['ds_2'] = df_results_real['ds_2'].str.replace(key, ds_dict_name[key])
    
    print(df_results_real)
    #plot_boxplot_real(df_results_real.copy(), save_plot_real)

    plot_boxplot_real_mean_metrics(df_results_real.copy(), save_plot_real+'scores_')
    metric_correlation_all(df_results_real.copy(), save_plot_real+'metrics_')
    #metric_correlation(df_results_real.copy(), save_plot_real+'metrics_')


if __name__ == '__main__':
    prefix='test_data/' #set via cmd
    plot_results(prefix)