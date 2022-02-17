#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:47 2021

@author: mariaolaru

Individual freq correlations
"""

import rcs_pkg_sync_funcs as sync
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

subj_ID = 'RCS14'
#subj_ID = 'GRCS02'
#subj_ID = 'GRCS01'
subj_side_brain = 'L'
subj_side_arm = 'R'
montage_num = '2'

study_dir = '/Users/mariaolaru/Documents/temp/rcs_wearable_analysis'
data_dir = study_dir + '/data'
table_dir = study_dir + '/tables'
plot_dir = study_dir + '/plots'
#GRCS01R_pre-stim/3day_sprint/montage_1
pkg_dir = data_dir + '/wearable/pkg/'
apple_dir = data_dir + '/wearable/apple/'
#fp_phs = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_phs.csv'

#fp_psd = subj_dir + '/' + subj_ID + subj_side + '_pre-stim' + '/' + '3day_sprint/montage' + '/' + 'montage_psd_total.csv'
fp_psd = data_dir + '/neural/psd/' + str(subj_ID) + str(subj_side_brain) + '_montage_' + str(montage_num) + '_psd_total.csv'
#fp_psd = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_psd_aperiodic.csv'
#fp_psd = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_psd_periodic.csv'
#fp_coh = '/Users/mariaolaru/Documents/temp/' 'RCS07/RCS07L/RCS07L_pre-stim/RCS07L' '_pre-stim_coh.csv'
#fp_notes = '/Users/mariaolaru/Documents/temp/RCS07/RCS07L/RCS07L_pre-stim/RCS07L_pre-stim_meta_session_notes.csv'

sr = 250
### NOTE – NEED TO ADD FUNCTION THAT PULLS APPROPRIATE SUBJ FILES FROM DIRS
[df_apple, start_time, stop_time] = sync.preproc_apple(apple_dir, subj_ID, subj_side_arm)
#df_apple = pd.DataFrame([])

[df_pkg, start_time, stop_time] = sync.preproc_pkg(pkg_dir, subj_ID, subj_side_arm)
#df_pkg = pd.DataFrame([])

[df_wearable, start_time, stop_time] = sync.preproc_wearables(df_apple, df_pkg)
#df_wearable = pd.DataFrame([])

#df_phs = sync.preproc_phs(fp_phs, start_time, stop_time)
df_phs = pd.DataFrame([])

df_psd = sync.preproc_psd(fp_psd, start_time, stop_time)

#df_coh = sync.preproc_coh(fp_coh, start_time, stop_time, sr)
df_coh = pd.DataFrame([])

#df_notes = sync.preproc_notes(fp_notes, start_time, stop_time)
#df_dys = sync.find_dyskinesia(df_notes)
#df_meds = sync.get_med_times()
df_notes = pd.DataFrame([])
df_dys = pd.DataFrame([])
df_meds = pd.DataFrame([])

#Processing
df_merged = sync.process_dfs(df_pkg, df_apple, df_wearable, df_phs, df_psd, df_coh, df_meds, df_dys)

#hardcoded sleep times of 10PM to 8AM
if (df_notes.empty == False):
    df_merged = sync.add_sleep_col(df_merged)


#remove BK scores reflecting periods of inactivity
if (df_pkg.empty == False):
    df_merged = df_merged[df_merged['inactive'] == 0]

#correlate all scores
keyword = 'spectra'
#keyword = 'fooof_flat'
#keyword = 'fooof_peak_rm'

corr_vals = ['Tremor_Score', 'apple_tremor']
#corr_vals = ['DK', 'BK', 'Tremor_Score']
#corr_vals = ['apple_dk', 'apple_tremor']

[df_spectra_corr, contacts] = sync.compute_correlation(df_merged, keyword, corr_vals)
out_fp = table_dir + '/' + subj_ID + subj_side_brain + '_corrs' + '.csv'
df_spectra_corr.to_csv(out_fp)

#plot correlations for each frequency
sync.plot_corrs(df_spectra_corr, corr_vals, plot_dir)
tt = len(df_merged)/60
tt_h = int(np.floor(tt))
tt_m = int(np.round(tt % tt_h * 60))
print("Time streamed: ", str(tt_h), "H ", str(tt_m), "M")

#correlate coherence
#df_coh_corr = sync.compute_correlation(df_merged, 'Cxy')
#sync.plot_corrs(df_coh_corr, 'DK')
#sync.plot_corrs(df_coh_corr, 'BK')

####### Plotting timeseries data ####################
df = df_merged
freq = 23
plt.close()
#contacts = np.array(['+2-0', '+3-1', '+10-8', '+11-9'])
#breaks = sync.find_noncontinuous_seg(df_merged['timestamp'])
#title =  subj_ID + subj_side_brain + ' wearable-RCS pre-stim time-series sync'
#title = ("freq_band: " + str(freq_band) + "Hz")
#plt.title(title)
plt.figure(figsize = (15, 5))
plt.rcParams.update({'font.size': 16}) 

#plt.plot(np.arange(1, len(df)+1, 1), df['phs_gamma'], alpha = 0.7, label = 'phs-gamma', markersize = 1, color = 'slategray')
#plt.plot(np.arange(1, len(df)+1, 1), df['phs_beta'], alpha = 0.7, label = 'phs-beta', markersize = 1, color = 'olivedrab')

plt.plot(np.arange(1, len(df)+1, 1), df[corr_vals[0]], alpha = 0.9, label = corr_vals[0], markersize = 1, color = 'steelblue')
plt.plot(np.arange(1, len(df)+1, 1), df[corr_vals[1]], alpha = 0.7, label = corr_vals[1], markersize = 1, color = 'indianred')
#plt.plot(np.arange(1, len(df)+1, 1), df[corr_vals[2]], alpha = 0.7, label = corr_vals[2], markersize = 1, color = 'forestgreen')

#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[0] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[0], markersize = 1, color = 'darkkhaki')
plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[1] + "')"], alpha = 0.9, label = str(freq)+ "Hz " + contacts[1], markersize = 1, color = 'darkorange')

#freq = 13
#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[2] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[2], markersize = 1, color = 'orchid')
#plt.plot(np.arange(1, len(df)+1, 1), df["('" + keyword + "'," + str(freq) + ".0,'" + contacts[3] + "')"], alpha = 0.7, label = str(freq)+ "Hz " + contacts[3], markersize = 1, color = 'mediumpurple')

#plt.vlines(df_merged[df['dyskinesia'] == 1].index, 0, 1, color = 'black', label = 'dyskinesia')
#plt.vlines(df_merged[df['med_time'] == 1].index, 0, 1, color = 'green', label = 'meds taken')
#plt.vlines(np.where(df['asleep'] == 1)[0], 0, 1, alpha = 0.1, label = 'asleep', color = 'grey')
#plt.vlines(breaks, 0, 1, alpha = 0.7, label = 'break', color = 'red')

#plt.hlines(class_thresh[0], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[1], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[2], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[3], len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')
#plt.hlines(class_thresh[4], 0, len(df), alpha = 0.7, label = 'LDA thresh', color = 'red')

plt.legend(ncol = 6, loc = 'upper right')
plt.ylabel('scores (normalized)')
plt.xlabel('time (samples)')
#####################################################

#create dataset with combo channels of spectra and coherence features
i_rm = [x for x, s in enumerate(list(df_merged.columns)) if '+2-0_+' in s]
i_rm2 = [x for x, s in enumerate(list(df_merged.columns)) if "'+2-0')" in s]
i_rm3 = [x for x, s in enumerate(list(df_merged.columns)) if "'+11-9')" in s]
i_rmt = np.concatenate([i_rm, i_rm2, i_rm3])
df_merged_combo = df_merged.drop(df_merged.columns[i_rmt], axis = 1)

i_rm4 = [x for x, s in enumerate(list(df_merged_combo.columns)) if '+3-1_+' in s]
df_merged_spectra_2ch = df_merged_combo.drop(df_merged_combo.columns[i_rm4], axis = 1)

irm = i_rm = [x for x, s in enumerate(list(df_merged.columns)) if 'Cxy' in s]
df_merged_spectra = df_merged.drop(df_merged.columns[i_rm], axis = 1)

#run PCA analysis
df = df_merged_spectra_2ch
[df_pcs, pcs_vr] = sync.run_pca(df, 'spectra', 10, 0)

#re-add DK data into pc dataframe
df_svm = df_pcs.copy()
df_svm['DK'] = df.dropna().reset_index(drop=True)['DK']

"""
keys = ['+2-0', '+3-1', '+10-8', '+11-9']
for i in range(len(keys)):
    [pcs, test_vr] = sync.run_pca(df, keys[i], 5, 1)
    sync.plot_pcs(pcs.iloc[:, 0:pcs.shape[1]-1], keys[i], pkg_dir)

keys = ['+3-1', '+10-8']
df_pcs = sync.run_pca_wrapper(df, keys, 5, 0, pkg_dir)

#run SVM with PCA feature selection
sync.run_svm_wrapper(df_pcs, 'PC', 'DK', 0.03)
"""

#run LDA with PCA features
"""
#get top features
[df_top_pos, df_top_neg] = sync.get_top_features(coefs, x_names)
import pandas as pd
df_top_coefs = pd.concat([df_top_pos, df_top_neg])
"""
## split DK data in 5 equal dyskinesia classes above SVM threshold for dyskinesia
df = df_merged_spectra_2ch
min_thresh = 0.03
[df_lda, class_thresh] = sync.add_classes_wrapper(df, min_thresh)

label = 'log(normalized DK scores)'
sync.plot_classes(df_lda['DK_log'], label, class_thresh)
[accuracy, sem] = sync.run_lda(df, 'spectra', 'DK_class')

#run LR with PCA feature selection
sync.run_lm_wrapper(df_svm, 'PC', 'DK', 2) #all spectra channels

#run SVM
sync.run_svm_wrapper(df_merged, 'spectra', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+2-0', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+3-1', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+10-8', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_spectra, '+11-9', 'DK', 0.03)
sync.run_svm_wrapper(df_merged, 'Cxy', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, '+', 'DK', 0.03)
[coefs, x_names] = sync.run_svm_wrapper(df_merged_combo, 'spectra', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, 'Cxy', 'DK', 0.03)
sync.run_svm_wrapper(df_merged_combo, '+3-1_+10-8', 'DK', 0.03)

#run lm on one channel:
i_rm = [x for x, s in enumerate(list(df_merged.columns)) if "'+3-1')" in s]
i_rm2 = [x for x, s in enumerate(list(df_merged.columns)) if "'+10-8')" in s]
i_rm3 = [x for x, s in enumerate(list(df_merged.columns)) if "'+11-9')" in s]
#i_rm4 = [x for x, s in enumerate(list(df_merged.columns)) if "'+2-0')" in s]
i_rmt = np.concatenate([i_rm, i_rm2, i_rm3])
df_merged_singch = df_merged.drop(df_merged.columns[i_rmt], axis = 1)
df_coefs = sync.run_lm_wrapper(df_merged_singch, 'spectra', 'DK', 'base', 2) #all spectra channels

#run linear regressions
df_coefs = sync.run_lm_wrapper(df_merged, 'spectra', 'apple_tremor', 'base', 2) #all spectra channels
df_coefs = sync.run_lm_wrapper(df_merged, 'spectra', 'apple_tremor', 'lasso', 2) #all spectra channels

sync.run_lm_wrapper(df_merged, 'Cxy', 'DK', 2) #all coherence combos
sync.run_lm_wrapper(df_merged_combo, '+', 'DK', 2) #2 spectra, 2 coh channels
sync.run_lm_wrapper(df_merged_combo, 'spectra', 'DK', 2) #2 spectra channels
sync.run_lm_wrapper(df_merged_combo, 'Cxy', 'DK', 2) #2 coh channels
sync.run_lm_wrapper(df_merged_combo, "'+3-1'", 'DK', 2) #1 spectra channels

#run lstm
[result_metrics, baseline_metrics, fig] = sync.run_lstm_wrapper(df_merged, 'spectra', 'apple_tremor')

#fig.colorbar(img)
#temp = df_lm.sort_values('coefs').head(10)
#temp = df_lm.sort_values('coefs').tail(10)
#temp = temp.iloc[::-1]

#linear regression w/ top features
#df_top = df_merged.loc[:, np.append(features, 'DK')]
#sync.run_lm_wrapper(df_top, '+', 'DK', 2) #2 spectra, 2 coh channels

###Try PCA analysis
#from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
#import pandas as pd

# Make an instance of the Model
#pca = PCA(n_components=5) #minimum components to explain 95% of variance
#pca.fit(x_train)
#pcs = pca.fit_transform(x_train)
#pcs = pd.DataFrame(data = pcs)
#pca.n_components_

###Try NMF analysis
#from sklearn.decomposition import NMF
#nmf = NMF(n_components=5, init = 'random', random_state = 0, max_iter = 2000)
#W = nmf.fit_transform(x_train)
#H = nmf.components_

