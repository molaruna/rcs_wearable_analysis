#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function synchronizes and processes all data from the local directory 
it is run. This data can include files from the Summit RC+S system, Apple watches, 
PKG watches, and patient reports. 

All data must be CSV formatted and include a 'timestamp' column header

python3 sync_data.py <feature_key> <target_key>
python3 sync_data.py 'spectra' 'DK'

@author: mariaolaru
"""

import sync_funcs as sync

def main(data_dir, feature_key):
    #Assumes 250Hz sr
    #Processes all data in the current directory

    #feature_key = int(sys.argv[1])
    #target_key = float(sys.argv[2])    

    #For debugging
    #TODO: should include option for which types of files to analyze
    #feature_key = 'fooof_peak_rm'
    #feature_key = 'spectra'
    
    target_list = ['pkg_dk', 'apple_dk', 'pkg_bk', 'pkg_tremor', 'apple_tremor']
    
    #data_dir = os.getcwd()
    #For debugging
# feature_key = 'spectra'
# data_dir = '/Users/mariaolaru/Documents/temp/rcs_wearable_analysis/data/RCS14L/RCS14L_m2'
#    data_dir = '/Users/mariaolaru/Documents/temp/rcs_wearable_analysis/data/RCS12R/RCS12R_mp'
    #data_dir = '/Users/mariaolaru/Documents/temp/rcs_wearable_analysis/data/RCS12L/RCS12L_ma'
    
    sync.preproc_files(data_dir)

    df_times = sync.get_meta_data(data_dir, feature_key) #should fix to calculat 2min scores as well
    targets = sync.get_psd_overlaps(df_times, target_list)
    #targets = ['apple_dk', 'apple_tremor']
    
    #Check to ensure data is still feeding in properly
    df = sync.merge_dfs(data_dir, feature_key, targets) 

    #for i in range(df.shape[0]):
    #   plt.plot(df.iloc[i, 126:126+125], color= 'b', alpha = 0.01)
    #for i in range(df.shape[0]):
    #   plt.plot(da.sel(time_interval = 'min1', contact = '+2-0')[i, :], color = 'b', alpha = 0.01)

    da = sync.reshape_data(df, feature_key, targets)
    #da.sel(time_interval = 'min1', contact = '+2-0').values.shape
    #867, 131
    
    #Plot spectral data 
    da_psd = sync.reshape_data(df, feature_key, targets, psd_only = True)
    sync.plot_spectra(da_psd, 'psd', 'overlaid', data_dir, feature_key)

    #Plot spectral data summary stats
    da_psd_stats = sync.compute_spectra_stats(df, feature_key)
    sync.plot_spectra(da_psd_stats, 'psd', 'ci', data_dir, feature_key)

    #Normalize all data
    df_norm = sync.scale_data(df, 'minmax')
    da_norm = sync.reshape_data(df_norm, feature_key, targets)
    
    #Plot top PCs for each frequency    
    da_psd_norm = sync.reshape_data(df_norm, feature_key, targets, psd_only = True)
    [da_pcaf, df_pcaf_ratios] = sync.compute_pca(da_psd_norm, 5, 'frequency')
    sync.plot_spectra(da_pcaf, 'pca', 'frequency', data_dir, 'feature_key')

    #Get top PCs for each timestamp
    df_norm_psd = sync.get_psd(df_norm, 'min1_spectra')[0]
    [da_pcaf_t, df_pcaf_ratios_t] = sync.compute_pca(df_norm_psd, 20, 'time')        
    
    #Plot Pearson's r of spectral features with wearable targets
    da_pearsonr = sync.compute_correlations(df_norm, feature_key, target_list)

    sync.plot_spectra(da_pearsonr, 'corr', 'r_pval', data_dir, feature_key)
    sync.plot_spectra(da_pearsonr, 'corr', 'r', data_dir, feature_key)

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

    #time-domain PCs
    [df_pcat, df_pcat_ratios] = sync.compute_pca(df_norm_psd.iloc[:, 0:252], 20, 'time') #only min1 spectra data

    #run linear regressions
    [df_lr_preds, df_lr_metrics] = sync.run_lr_model(df_pcat.T, df_norm.loc[:, targets], 'top 20 PCs')
    [df_lr_preds2, df_lr_metrics2] = sync.run_lr_model(df_norm_psd.iloc[:, 0:252], df_norm.loc[:, targets], '504 freq features')

    #run blstm
    #sync.run_blstm(df_norm_psd.iloc[:, 0:252], df_norm.loc[:, 'apple_tremor'].to_numpy(), 'apple_tremor')    

    #Then, run the blstm, and show learning curve
    
    #Lastly, should plot comparison with predicted values of linear regression and BLSTM
        