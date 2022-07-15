#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function synchronizes and processes all data from the local directory 
it is run. This data can include files from the Summit RC+S system, Apple watches, 
PKG watches, and patient reports. 

All data must be CSV formatted and include a 'timestamp' column header

@author: mariaolaru
"""

import sync_funcs as sync

feature_key_list = ['spectra', 'fooof_flat', 'fooof_peak_rm']

feature_key = feature_key_list[0]

target_list = ['pkg_dk', 'apple_dk', 'pkg_bk', 'pkg_tremor', 'apple_tremor']
data_dir = '/Volumes/GoogleDrive-110482430750397643411/My Drive/UCSF_Neuroscience/starr_lab_MO/studies/3day_sprint/data/RCS02L' #ask me for access
  

sync.preproc_files(data_dir)

df_times = sync.get_meta_data(data_dir, feature_key) #should fix to calculate better, I think there's a bug here?
targets = sync.get_psd_overlaps(df_times, target_list)

#Check to ensure data is still feeding in properly
df = sync.merge_dfs(data_dir, feature_key, targets) 
#da = sync.reshape_data(df, feature_key, targets)

#Plot spectral data 
da_psd = sync.reshape_data(df, feature_key, targets, psd_only = True)

sync.plot_spectra(da_psd, 'psd', 'overlaid', data_dir, feature_key)

#Plot spectral data summary stats
da_psd_stats = sync.compute_spectra_stats(df, feature_key)
sync.plot_spectra(da_psd_stats, 'psd', 'ci', data_dir, feature_key)

#Normalize all data
std_thresh = 3
df_norm = sync.scale_data(df, targets, std_thresh, 'minmax')

#Convert data to categorical percentile values
quantiles = sync.convert_category(df.loc[:, targets], [0.2, 0.4, 0.6, 0.8, 1])
sync.plot_categories(df_norm.loc[:, targets], quantiles) #sanity check
df_norm.loc[:, targets] = sync.continuous2ordinal(df_norm.loc[:, targets], quantiles)

df_norm.pkg_dk.value_counts().sort_index()
df_norm.pkg_bk.value_counts().sort_index()
df_norm.pkg_tremor.value_counts().sort_index()

#Organize data into 4D (data arrays)    
df_norm_psd = sync.get_psd(df_norm, 'min1_' + feature_key)[0]
da_norm = sync.reshape_data(df_norm, feature_key, targets)
#da_norm_psd = sync.reshape_data(df_norm, feature_key, targets, True)

#Normalize data by channel, instead of by frequency
da_normch_psd = sync.scale_data_ch(da_psd, 'minmax')
df_normch_psd = sync.reshape_xr2df(da_normch_psd, df_norm_psd.columns, 'min1')

#Concatenate dataframe by frequency features
[df_normch_psd_concat, channels] = sync.concat_feature2sample(df_normch_psd, 4)

#Get top PCs for each timestamp
#[da_pcaf_t, df_pcaf_ratios_t] = sync.compute_pca(df_norm_psd, 20, 'samples')
#[da_pcaf_ch_t, df_pcaf_ch_ratios_t] = sync.compute_pca(df_normch_psd, 20, 'samples')

#Plot PCA loadings using sklearn    
#[da_pcaf, df_pcaf_ratios] = sync.compute_pca(da_norm_psd, 5, 'features')
#sync.plot_spectra(da_pcaf, 'pca', 'frequency', data_dir, 'feature_key')

#Plot top PCs - frequency features
[da_pcaf_ch, df_pcaf_ch_ratios] = sync.compute_pca(da_normch_psd, 5, 'features')

sync.plot_spectra(da_pcaf_ch, 'pca', 'frequency', data_dir, 'feature_key')
sync.plot_variance(df_pcaf_ch_ratios)
sync.plot_pcs(da_pcaf_ch)

#Compute top singular values/PCs
[singular_values, vec_num_pcs] = sync.get_pc_components(da_normch_psd)

sync.plot_pc_components(singular_values, vec_num_pcs)

#Plot PCA energy
#temp_vecs = vec_num_pcs.copy()
#temp_vecs.iloc[:, :] = 5
#pca_energy = sync.get_pca_energy(da_normch_psd)
#sync.plot_pca_energy(pca_energy)

#Get top PCs - timepoint samples
#[df_pcat, df_pcat_ratios] = sync.compute_pca(df_norm_psd.iloc[:, 0:252], 20, 'time') #only min1 spectra data
#[df_pcat, df_pcat_ratios] = sync.compute_pca(df_norm_psd, 20, 'samples') #only min1 spectra data

#top PCs across frequency
[df_pcat_ch, df_pcat_ratios_ch] = sync.compute_pca(df_normch_psd, 20, 'samples') #only min1 spectra data

#top PCs across time
[df_pcaf, df_pcaf_ratios] = sync.compute_pca(df_normch_psd.T, 20, 'samples') #only min1 spectra data

#Get top frequency-concatenated PCs across frequency
[df_pcat_ch_concat, df_pcat_ratios_ch_concat] = sync.compute_pca(df_normch_psd_concat, 20, 'samples') #only min1 spectra data

#Get top time-concatenated PCs across time
[df_pcaf_ch_concat, df_pcaf_ratios_ch_concat] = sync.compute_pca(df_normch_psd_concat.T, 20, 'samples')

#Get channel labels for each time-concatenated timeseries timestamp
df_channels = sync.get_channels4samples(df_pcat_ch_concat.T, channels)

#Plot top PC frequency components
npcs = 5
da_pcaf_toppcs = sync.get_toppcs(da_pcaf_ch, df_pcaf, npcs)

sync.plot_spectra(da_pcaf_toppcs, 'pca', 'frequency', data_dir, 'feature_key')
sync.plot_pcs(da_pcaf_toppcs)
#sync.plot_variance(df_pcaf_ratios) #need to debug func first


#Plot top concat PC frequency components
npcs = 5

sync.plot_spectra_df(df_pcaf_ch_concat, npcs)
sync.plot_variance(df_pcaf_ratios_ch_concat)


# Plot PCs in relation to symptom scores
sync.plot_pcs_symptoms(df_pcat_ch, df_norm, npcs, feature_key, data_dir)

#Cluster PCs
sync.plot_pcs_clustered(df_pcat_ch.T.iloc[:, 0:5], df_norm.loc[:, targets])


#Cluster time-concatenated PCs
## by category
targets = ['pkg_dk', 'pkg_bk', 'pkg_tremor']
sync.plot_pcs_clustered_severity(df_pcat_ch, df_channels, df_norm, targets)

## complete set
sync.plot_pcs_clustered(df_pcat_ch_concat.T.iloc[:, 0:5], df_channels)

#Plot Pearson's r of spectral PCs with wearable targets
df_merged = sync.merge_dfs_time(df_pcat_ch.T, df_norm.loc[:, targets])
da_pearsonr_pcs = sync.compute_correlations(df_merged, 'pc', target_list)

sync.plot_spectra(da_pearsonr_pcs, 'corr', 'r_pval', data_dir, 'pc')
sync.plot_spectra(da_pearsonr_pcs, 'corr', 'r', data_dir, 'pc')

#Plot Pearson's r of individual features with symptom scores
da_pearsonr = sync.compute_correlations(df_norm, 'min2', target_list)
sync.plot_spectra(da_pearsonr, 'corr', 'r_pval', data_dir, 'spectra')
sync.plot_spectra(da_pearsonr, 'corr', 'r', data_dir, 'spectra')

#TODO create heatmap output for each correlation plot
#TODO Should also run the correlation with coherence, not just power scores

#Get highest r-val with a significant pvalue for sub-cort & cort channels
da_top_sigcorrs = sync.get_top_sigcorrs(da_pearsonr, abs_val = True) 
df_top_ch = sync.get_top_channels(da_top_sigcorrs) 
#TODO should save out these tables 

dfl_top_ts = sync.get_top_timeseries(da_norm, da_top_sigcorrs, df_top_ch)
sync.plot_timeseries(dfl_top_ts, data_dir, feature_key, 'datetime')
sync.plot_timeseries(dfl_top_ts, data_dir, feature_key, 'samples')

#compute cross-correlation
dfl_top_cc = sync.compute_top_ccfs(dfl_top_ts)
sync.plot_crf(dfl_top_cc, feature_key, data_dir)


#run linear regressions
[df_lr_preds, df_lr_metrics] = sync.run_baseline_model(df_norm_psd, 
                                                       df_norm.loc[:, targets], 
                                                       'lr', 
                                                       '504 freq features', 
                                                       True)

#[df_lr_preds_pcs, df_lr_metrics_pcs] = sync.run_baseline_model(df_pcat.T, 
#                                                         df_norm.loc[:, targets], 
#                                                         'lr', 
#                                                         'top 20 PCs', 
#                                                         True)

[df_lr_preds_pcs_ch, df_lr_metrics_pcs_ch] = sync.run_baseline_model(df_pcat_ch.T, 
                                                         df_norm.loc[:, targets], 
                                                         'lr', 
                                                         'top 20 PCs ch-norm', 
                                                         True)


[df_lr_preds_f, df_lr_metrics_f] = sync.run_baseline_model(df_norm_psd, 
                                                       df_norm.loc[:, targets], 
                                                       'lr', 
                                                       '504 freq features', 
                                                       False)

#[df_lr_preds_pcs_f, df_lr_metrics_pcs_f] = sync.run_baseline_model(df_pcat.T, 
#                                                         df_norm.loc[:, targets], 
#                                                         'lr', 
#                                                         'top 20 PCs', 
#                                                         False)

df_lr_metrics_cv = sync.run_baseline_model_cv(df_norm_psd, 
                                              df_norm.loc[:, targets], 
                                              'lr')        

#df_lr_metrics_pcs_cv = sync.run_baseline_model_cv(df_pcat.T, 
#                                              df_norm.loc[:, targets], 
#                                              'lr')        

df_lr_metrics_pcs_ch_cv = sync.run_baseline_model_cv(df_pcat_ch.T, 
                                              df_norm.loc[:, targets], 
                                              'lr')        

#run lm lasso
[df_lasso_preds, df_lasso_metrics] = sync.run_baseline_model(df_norm_psd, 
                                                             df_norm.loc[:, targets], 
                                                             'lasso', 
                                                             '504 freq features',
                                                             True)

df_lasso_metrics_cv = sync.run_baseline_model_cv(df_norm_psd, 
                                                       df_norm.loc[:, targets], 
                                                       'lasso',
                                                       alpha=0.0001)    

#[df_lasso_preds_pcs, df_lasso_metrics_pcs] = sync.run_baseline_model(df_pcat.T, 
#                                                               df_norm.loc[:, targets], 
#                                                               'lasso', 
#                                                               'top 20 PCs',
#                                                               True)

#df_lasso_metrics_pcs_cv = sync.run_baseline_model_cv(df_pcat.T, 
#                                                       df_norm.loc[:, targets], 
#                                                       'lasso',
#                                                       alpha=0.0001)    

nalpha = 100
upper_alpha_thresh = 0.1
alpha_curve_norm = sync.compute_alpha_curve(df_norm_psd, 
                                           df_norm.loc[:, targets],
                                           upper_alpha_thresh,
                                           nalpha)

alpha_curve_pcs = sync.compute_alpha_curve(df_pcat_ch.T, 
                                           df_norm.loc[:, targets],
                                           upper_alpha_thresh,
                                           nalpha)

sync.plot_alpha_curve(alpha_curve_norm)
sync.plot_alpha_curve(alpha_curve_pcs)


#~~~~~~~~ INITIAL MODEL PARAMS ~~~~~~
#input_dim = len(df_norm_psd.columns)
#hidden_dim = 64
#layer_dim = 3
#output_dim = 1
#dropout = 0.2   
#n_epochs = 100    
#learning_rate = 1e-3   
#weight_decay = 1e-6                                                      

model_inputs = {'hidden_dim' : 32,
                'layer_dim' : 3,
                'output_dim' : 1,
                'dropout_prob' : 0.6,
                'n_epochs' : 60,
                'learning_rate' : 1e-3,
                'weight_decay' : 1e-6}

#Then, run the blstm, and show learning curve
#dys_metrics = sync.run_blstm(df_norm_psd, 
#                             df_norm.loc[:, targets], 
#                            'pkg_dk', 
#                            'lr', 
#                            '504 freq features', 
#                            False,
#                            model_inputs)

#dys_metrics = sync.run_blstm(df_norm_psd, 
#                             df_norm.loc[:, targets], 
#                            'pkg_dk', 
#                            'lr', 
#                            '504 freq features', 
#                            True,
#                            model_inputs)

#dys_pc_metrics = sync.run_blstm(df_pcat.T, 
#                            df_norm.loc[:, targets], 
#                            'pkg_dk', 
#                            'lr', 
#                            'top 20 PCs', 
#                            False,
#                            model_inputs)

#dys_pc_metrics = sync.run_blstm(df_pcat.T, 
#                            df_norm.loc[:, targets], 
#                            'pkg_dk', 
#                            'lr', 
#                            'top 20 PCs', 
#                            True,
#                            model_inputs)

#bradyk_metrics = sync.run_blstm(df_norm_psd, 
#                            df_norm.loc[:, targets], 
#                            'pkg_bk', 
#                            'lr', 
#                            '504 freq features', 
#                            True,
#                            model_inputs)

#bradyk_pc_metrics = sync.run_blstm(df_pcat.T, 
#                            df_norm.loc[:, targets], 
#                            'pkg_bk', 
#                            'lr', 
#                            '504 freq features', 
#                            True,
#                            model_inputs)
