#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:22:34 2022

@author: mariaolaru
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stat
import xarray as xr
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from datetime import date
import datetime


def get_files(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    if '.DS_Store' in files:
        i = files.index('.DS_Store')
        del files[i]
    
    return files

def make_dir(fp, dirname):
    path = os.path.join(fp, dirname)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def preproc_data(df, label, file=None):
    if label == 'pkg':
        df_out = df[df['Off_Wrist'] == 0]
        col_keep = ['Date_Time', 'BK', 'DK', 'Tremor_Score']
        df_out = df_out.loc[:, col_keep]
        df_out = df_out.rename(columns={'Date_Time': 'pkg_dt', 
                                'BK': 'pkg_bk', 
                                'DK': 'pkg_dk', 
                                'Tremor_Score': 'pkg_tremor'})
        df_out['pkg_bk'] = df_out['pkg_bk']*-1
    elif label == 'apple':
        df_out = df.rename(columns={'time': 'timestamp'})
        if 'tremor' in file:
            df_out = df_out.rename(columns={'probability': 'apple_tremor'})
        if 'dyskinesia' in file:
            df_out = df_out.rename(columns={'probability': 'apple_dk'})
            
    return df_out
    

def merge_targets(dfp, df_out):
    if df_out.empty:
        return dfp
    
    df_merged = df_out.merge(dfp, how='outer')
    
    return df_merged

def print_loop(i, num, message, file):
        print('(' + str(i+1) + '/' + str(num) + ') ' + message + ': ', file)
    
def preproc_files(data_dir):
    files = get_files(data_dir)
    nfiles = len(files)

    df_pkg = pd.DataFrame([])
    df_apple = pd.DataFrame([])
    for i in range(nfiles):
        file = files[i]
        print_loop(i, nfiles, 'preprocessing file', file)
        file_fp = os.path.join(data_dir, file)
        df = pd.read_csv(file_fp)
        if 'BK' in df.columns: #pkg data
            dfp = preproc_data(df, 'pkg')
            df_pkg = merge_targets(dfp, df_pkg)
            pkg_dir = make_dir(data_dir, 'orig_pkg')
            os.replace(file_fp, os.path.join(pkg_dir, file))
        if 'time' in df.columns: #apple data
            dfp = preproc_data(df, 'apple', file)
            df_apple = merge_targets(dfp, df_apple)
            apple_dir = make_dir(data_dir, 'orig_apple')
            os.replace(file_fp, os.path.join(apple_dir, file))

    if not df_pkg.empty:
        out_pkg_file = 'pkg_2min_scores.csv'
        df_pkg.to_csv(os.path.join(data_dir, out_pkg_file), index=False)
        
    if not df_apple.empty:
        out_apple_file = 'apple_1min_scores.csv'
        df_apple.to_csv(os.path.join(data_dir, out_apple_file), index=False)        

    
def pivot_df(df): 
    #assumes values of pivot table are in column #1 and columns are column #0 & #2
    dfp = df.pivot_table(index = df.index, 
                         values = [df.columns[1]], 
                         columns = [df.columns[0], df.columns[2]])    
    dfp.columns = ['_'.join(map(str, col)) for col in dfp.columns]

    return dfp

def average_2min_scores(df_psd):
    #find indices of timestamp on even minutes
    s = pd.Series(df_psd.index.minute % 2 == 1)
    odd_i = s[s].index.values
    odd_prev_i = odd_i-1
    diff = (df_psd.index[odd_i] - df_psd.index[odd_prev_i]).astype('timedelta64[m]')
    
    s = pd.Series(diff == 1)
    s_i = s[s].index.values
    keep_i = odd_i[s_i]
    
    if keep_i[0] == 0:
        keep_i = np.delete(keep_i, 0)
        
    ts_2min = df_psd.index[keep_i]
    
    colnames = df_psd.columns
    colnames_2min = [sub.replace('min1', 'min2') for sub in colnames]
    
    df_psd_avg1 = df_psd.iloc[keep_i, :].reset_index(drop=True).to_numpy()
    df_psd_avg2 = df_psd.iloc[keep_i-1, :].reset_index(drop=True).to_numpy() 
    df_avg = np.mean([df_psd_avg1, df_psd_avg2], axis=0)        

    df_psd_2min = pd.DataFrame(data = df_avg,
                          columns = colnames_2min,
                          index = ts_2min)
    
    return df_psd_2min
    
def add_2min_scores(df_psd):
    colnames = df_psd.columns
    addl = 'min1_'
    addl_colnames =     [addl + s for s in colnames]

    df_psd.columns = addl_colnames
    df_psd_2min = average_2min_scores(df_psd)

    df_merged = df_psd.merge(df_psd_2min, how = 'outer', left_index = True, right_index = True)   

    return df_merged
    
    
def merge_df(df_psd, df_target, get_2min_scores=False):    
    if len(df_psd.index.unique()) < len(df_psd.index): #assumes this is long-form power spectra data
        df_psd = pivot_df(df_psd)
        if (get_2min_scores==True):
            df_psd = add_2min_scores(df_psd) 

    if df_target.empty:
        return df_psd
    
    df_out = df_target.merge(df_psd, left_index=True, right_index=True, sort = True)
    return df_out
        

def add_timestamps(df, file):  

    colname = 'timestamp' 
    unit = 'ms'
    ts_pkg = 'pkg_dt'
    ts_out = 'timestamp_dt'
    
    if ts_pkg in df.columns:
        colname = ts_pkg
        unit = 'ns'    
    elif colname in df.columns:
        if len(str(int(df['timestamp'].head(1).values))) == 10:
            unit = 's'    
    elif colname not in df.columns:
        raise ValueError('This file does not contain a timestamp column header: ' 
                         + file)

    df[ts_out] = pd.to_datetime(df[colname], unit = unit)

    if colname == ts_pkg:
        df[ts_out] = df[ts_out]
        df[ts_out] = df[ts_out] + pd.DateOffset(minutes=1)
        df = df.drop(ts_pkg, axis = 1)

    else:
        df[ts_out] = df[ts_out] -  pd.Timedelta(7, unit = 'hours') #Assumes local time is America/Los_Angeles
        
    df = df.set_index([ts_out])

    return df

def check_overlap(colnames, feature_key, target_list):
    merge_data = False
    if colnames[1] in target_list:
        merge_data = True
    if feature_key in colnames:
       merge_data = True 
       
    return merge_data
    
def merge_dfs(data_dir, feature_key, targets):
    files = get_files(data_dir)

    nfiles = len(files)
    
    df_out = pd.DataFrame([])
    for i in range(nfiles):
        file = files[i]
        file_fp = os.path.join(data_dir, file)
        df = pd.read_csv(file_fp)      
        df = add_timestamps(df, file)
        merge_data = check_overlap(df.columns, feature_key, targets)
        if merge_data:
            print_loop(i, nfiles, 'merging file', file)
            df_out = merge_df(df, df_out, get_2min_scores=True)    
    
    df_out = order_features(df_out, feature_key) #Assumes feature_key has spectra that is not ordered
    #df_out = df_out.dropna()
    if feature_key == 'spectra':
        feature_i = [j for j, s in enumerate(df_out.columns) if feature_key in s]    
        df_out.iloc[:, feature_i] = np.log10(df_out.iloc[:, feature_i])
    
    return df_out

def scale_data(df, scaler_type):
    scaler = get_scaler(scaler_type)
    df_scale = pd.DataFrame(data = scaler.fit_transform(df),
                            columns = df.columns,
                            index = df.index)
    return df_scale

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_targets(colnames, target_list):
    targets = list(set(target_list) & set(colnames))
    return targets

def get_frequencies(colnames):
    frequencies = np.array([])
    for i in range(len(colnames)):
        if '+' in colnames[i]: #assumes frequency column headers include contact channel w/ '+'
            x = colnames[i].split('_')[-2]
            x = float(x)
            frequencies = np.append(frequencies, x)    
    return np.unique(frequencies)

def sort_contacts(contacts):
    num_contacts = len(contacts)
    contacts_sorted = np.empty(num_contacts, dtype='<U32')
    contact_start = np.zeros(num_contacts)
    for i in range(num_contacts):
        contact_end_i = contacts[i].find('-')
        contact_start[i] =  int(contacts[i][1:contact_end_i])
    
    contacts_sorted_i = np.argsort(contact_start)
    for i in range(num_contacts):
        contacts_sorted[i] = contacts[contacts_sorted_i[i]]

    return contacts_sorted

def get_contacts(colnames):
    contacts = np.array([])
    for i in range(len(colnames)):
        if '+' in colnames[i]: #assumes channel column headers include '+'
            x = '+' + colnames[i].split('+')[1]
            contacts = np.append(contacts, x)
        
    contacts = sort_contacts(np.unique(contacts))
    return contacts

def order_features(df, feature_key):
    feature_i = [j for j, s in enumerate(df.columns) if feature_key in s]    
    target_i = list(set(np.arange(0, df.shape[1])).difference(feature_i))

    colnames = df.columns[feature_i]       
    contacts = get_contacts(colnames)

    df_ordered = pd.DataFrame([])
    for contact in contacts:
        contact_i = [i for i, s in enumerate(df.columns) if contact in s]        
        df_ordered = pd.concat([df_ordered, df.iloc[:, contact_i]], axis = 1)
       
    df_out = pd.merge(df_ordered, df.iloc[:, target_i], left_index = True, right_index = True)
    
    return df_out

def get_psd(df, feature_key):
    feature_i = [j for j, s in enumerate(df.columns) if feature_key in s]    
    df_in = df.iloc[:, feature_i]
    contacts = get_contacts(df_in.columns)
    frequencies = get_frequencies(df_in.columns)
    return [df_in, contacts, frequencies]

def reshape_data(df, feature_key, targets, psd_only = False):
    [df_psd, contacts, frequencies] = get_psd(df, feature_key)
    df_psd.index.name = 'measure'    

    description = 'spectral overlaid increments'
    da = c2dto4d(df_psd, contacts, frequencies, description)

    if (psd_only == True):
        return da
    
    cols_keep = list(set(targets) & set(df.columns))
    data_keep = df.loc[:, cols_keep]
    da_redund = xr.DataArray(
                             dims = ['time_interval', 'contact', 'measure', 'feature'],
                             coords = dict(
                                 time_interval = da['time_interval'].values,
                                 contact = da['contact'].values,
                                 measure = data_keep.index,
                                 feature = cols_keep))
    for i in range(len(da_redund['contact'].values)):
        contact = da_redund['contact'].values[i]
        da_redund.loc[dict(contact = contact)] = data_keep.values
    da = xr.concat([da, da_redund], dim='feature')
    return da

def get_single_meta_data(index, interval):
    df_variable = pd.DataFrame(columns = ['start_date', 
                                          'stop_date',
                                          'hours', 
                                          'minutes'])

    start_date = np.array([0, 0, 0])
    start_date[0] = pd.DatetimeIndex(index).year[0]
    start_date[1] = pd.DatetimeIndex(index).month[0]
    start_date[2] = pd.DatetimeIndex(index).day[0]

    stop_date = np.array([0,0,0])
    stop_date[0] = pd.DatetimeIndex(index).year[len(index)-1]
    stop_date[1] = pd.DatetimeIndex(index).month[len(index)-1]
    stop_date[2] = pd.DatetimeIndex(index).day[len(index)-1]

    df_variable.loc[0, 'start_date'] = str(start_date[0]) + '-' + str(start_date[1]) + '-' + str(start_date[2])
    df_variable.loc[0, 'stop_date'] = str(stop_date[0]) + '-' + str(stop_date[1]) + '-' + str(stop_date[2])

    h = math.floor(len(index)/interval/60)
    m = math.floor(((len(index)/interval/60)-h)*60)
    df_variable.loc[0, 'hours'] = h
    df_variable.loc[0, 'minutes'] = m        
    return df_variable
      
def get_meta_data(data_dir, feature_key):
    files = get_files(data_dir)
    nfiles = len(files)

    df_psd = pd.DataFrame([])
    df_pkg = pd.DataFrame([])
    df_apple = pd.DataFrame([])
    
    df_times = pd.DataFrame(columns = ['start_date', 
                                          'stop_date',
                                          'hours', 
                                          'minutes'],
                               index = ['psd', 'pkg', 'apple',
                                        'psd-pkg', 'psd-apple', 'pkg-apple',
                                        'psd-pkg-apple'])

    for i in range(nfiles):
        file = files[i]
        print_loop(i, nfiles, 'processing data', file)
        file_fp = os.path.join(data_dir, file)
        df = pd.read_csv(file_fp)
        if 'pkg' in file: 
            df_pkg = add_timestamps(df, file)
            interval = 2
            df_times.loc['pkg', :] = get_single_meta_data(df_pkg.index, interval).values
        elif 'apple' in file: 
            df_apple = add_timestamps(df, file)
            df_apple = df_apple.drop(['timestamp'], axis = 1)
            interval = 1
            df_times.loc['apple', :] = get_single_meta_data(df_apple.index, interval).values
        elif 'psd' in file:
            df_psd = add_timestamps(df, file)
            df_psd = df_psd.drop(['timestamp'], axis = 1)
            interval = 1
            df_times.loc['psd', :] = get_single_meta_data(df_psd.index.unique(), interval).values
    
    if (not df_pkg.empty) & (not df_psd.empty):
        df_pkg_psd = merge_df(df_psd, df_pkg)
        if not df_pkg_psd.empty:
            interval = 2
            df_times.loc['psd-pkg', :] = get_single_meta_data(df_pkg_psd.index, interval).values
    
    if (not df_apple.empty) & (not df_psd.empty):
        df_apple_psd = merge_df(df_psd, df_apple)
        if not df_apple_psd.empty:
            interval = 1
            df_times.loc['psd-apple', :] = get_single_meta_data(df_apple_psd.index, interval).values

    if (not df_pkg.empty) & (not df_apple.empty):
        df_pkg_apple = merge_df(df_pkg, df_apple)
        if not df_pkg_apple.empty:
            interval = 1
            df_times.loc['pkg-apple', :] = get_single_meta_data(df_pkg_apple.index, interval).values

    if (not df_pkg.empty) & (not df_apple.empty) & (not df_psd.empty):
        df_psd_pkg_apple = merge_df(df_psd, df_pkg_apple)
        if not df_psd_pkg_apple.empty:           
            interval = 2
            df_times.loc['psd-pkg-apple', :] = get_single_meta_data(df_psd_pkg_apple.index, interval).values

    out_dir = make_dir(data_dir, 'tables')
    filename = 'meta_timetable'
    df_times.to_csv(os.path.join(out_dir, filename+'.csv'))
    
    
    plot_table(df_times, data_dir, filename)

    return df_times

def plot_table(df, data_dir, filename):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', rowLabels=df.index)
    
    fig.tight_layout()
    
    plt.show()
    out_dir = make_dir(data_dir, 'tables')
    fig.savefig(os.path.join(out_dir, filename+'.pdf')) 

def get_psd_overlaps(df_times, target_list):
    if df_times.loc['psd-apple', :].isnull().any():
        indx_rm = [i for i, s in enumerate(target_list) if 'apple' in s]
        target_list = np.delete(target_list, indx_rm)

    if df_times.loc['psd-pkg', :].isnull().any():
        indx_rm = [i for i, s in enumerate(target_list) if 'pkg' in s]
        target_list = np.delete(target_list, indx_rm)

    return target_list


def compute_pca_2d(df, pca, pc_labels, domain):
    pcs = pca.fit_transform(df.values).T
    columns = df.index
    if domain == 'frequency':
        pcs = pca.fit_transform(df.T.values).T   
        columns = df.columns

    df_pcs = pd.DataFrame(data = pcs,
                          columns = columns,
                          index = pc_labels)
    return df_pcs
            
def compute_pca(da_psd, ncomponents, domain):
    pc_nums = np.linspace(1, ncomponents, ncomponents).astype(int).astype(str).tolist()
    pc_labels = ['pc' + sub for sub in pc_nums]
    pca = PCA(n_components=ncomponents)

    if domain == 'frequency':  
        da_psd = da_psd.sel(time_interval='min1')
        da = xr.DataArray(
                          dims = ['contact', 'pc', 'feature'], 
                          coords=dict(
                              contact=da_psd['contact'].values,
                              pc=pc_labels,
                              feature=da_psd['feature'].values,
                              ),
                          attrs=dict(description='PCs in ' + domain + ' domain'),
                          )
        contacts = da['contact'].values    
        df_pc_ratio = pd.DataFrame([], columns = contacts, index = pc_labels)
    
        for i in range(len(contacts)):
            contact = contacts[i]   
            df = pd.DataFrame(da_psd.sel(contact = contact).values,
                              columns = da_psd.feature.values, 
                              index = da_psd.measure.values)
    
            df = df.dropna()
            df_pcs = compute_pca_2d(df, pca, pc_labels, domain)
            da.loc[dict(contact=contact)] = df_pcs            
            df_pc_ratio.iloc[:, i] = pca.explained_variance_ratio_

    elif domain == 'time':
            df_pcs = compute_pca_2d(da_psd, pca, pc_labels, domain)
            da = df_pcs
            df_pc_ratio = pca.explained_variance_ratio_

    return [da, df_pc_ratio]
    
def compute_spectra_stats(df, feature_key):
    [df_psd, contacts, frequencies] = get_psd(df, feature_key)
    
    feature_1min = [j for j, col in enumerate(df_psd.columns) if 'min1' in col]
    df_psd_1min = df_psd.iloc[:, feature_1min]

    #get statistics      
    stats = pd.DataFrame([], columns = df_psd_1min.columns, 
                         index = ['mean', 'sem', 'ci_lower', 'ci_upper'])

    confidence_level = 0.95
    degrees_freedom = df_psd_1min.shape[0] - 1   

    stats.loc['mean'] = df_psd_1min.mean(axis=0)
    stats.loc['sem'] = stat.sem(df_psd_1min, axis=0)
    [stats.loc['ci_lower'], stats.loc['ci_upper']] = stat.t.interval(confidence_level, 
                                                             degrees_freedom, 
                                                             stats.loc['mean'].astype(float), 
                                                             stats.loc['sem'].astype(float))   
    #convert into 3d array
    description = 'Summary statistics for spectra'
    da = c2dto3d(stats, contacts, frequencies, description)
    return da

def train_val_test_split(X, y, test_ratio, shuffle=True):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state = 42, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state = 42, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_correlation(df, feature_key, target):
    if 'apple' in target:
        addl = 'min1_'
    if 'pkg' in target:
        addl = 'min2_'
        
    feature_key_addl = addl + feature_key 
    feature_i = [j for j, s in enumerate(df.columns) if feature_key_addl in s]    

    df = df.dropna()
    X = df.iloc[:, feature_i]
    y = df.loc[:, target]
        
    df_corrs = pd.DataFrame(index = ['r', 'r_pval'], columns = X.columns)
    for i in range(X.shape[1]):
        df_corrs.iloc[:, i] = stat.pearsonr(X.iloc[:, i], y)
        
    return df_corrs

def c2dto3d(df, contacts, frequencies, description):
    dim = len(contacts)
    df_3d = np.array(np.hsplit(df, dim))
               
    da = xr.DataArray(data = df_3d, 
                      dims = ['contact', 'measure', 'feature'], 
                      coords = dict(
                          contact=contacts,
                          measure=df.index.values,
                          feature=frequencies
                          ),
                      attrs=dict(description=description),
                      )
    return da        

def c2dto4d(df, contacts, frequencies, description):
    dim = len(contacts)
    df_list = []
    time_intervals = ['min1', 'min2']
    for i in range(len(time_intervals)):
        time_interval = time_intervals[i]
        feature_i = [j for j, s in enumerate(df.columns) if time_interval in s]    
        dft = df.iloc[:, feature_i]        
        df_3d = np.array(np.hsplit(dft, dim))
        df_list.append(df_3d)

    #debug sanity check
    #for i in range(df.shape[0]):
    #   plt.plot(df_3d[0,i, :], color= 'b', alpha = 0.01)
       
    df_4d = np.stack(df_list)  
        
    da = xr.DataArray(data = df_4d, 
                      dims = ['time_interval', 'contact', 'measure', 'feature'], 
                      coords = dict(
                          time_interval = time_intervals,
                          contact=contacts,
                          measure=df.index.values,
                          feature=frequencies
                          ),
                      attrs=dict(description=description),
                      )
    return da        
    
def compute_correlations(df, feature_key, target_list):
    contacts = get_contacts(df.columns)
    frequencies = get_frequencies(df.columns)
    targets = list(set(target_list) & set(df.columns))
    
    da = xr.DataArray(
                      dims = ['target', 'contact', 'measure', 'feature'], 
                      coords=dict(
                          target=targets,
                          contact=contacts,
                          measure=['r', 'r_pval'],
                          feature=frequencies
                          ),
                      attrs=dict(description='Pearson r test'),
                      )

    for i in range(len(targets)):
        target = targets[i]
        print_loop(i, len(targets), 'correlating', target)
        df_corrs = compute_correlation(df, feature_key, target)
        description = 'Pearsonr for ' + target
        df_corrs_3d = c2dto3d(df_corrs, contacts, frequencies, description)
        da.loc[dict(target=target)] = df_corrs_3d
        
    return da

def plot_psds(da, measure, out_gp, filename, feature_key):
    if measure == 'overlaid':
        suptitle_text = 'spectral overlay'
    elif measure == 'ci':
        suptitle_text = 'spectral summary'
        
    dirname = os.path.basename(out_gp)

    contacts = da['contact'].values
    num_plots = len(contacts)
    
    ncols = math.ceil(num_plots/2)

    fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1, :], xlabel = 'Frequencies (Hz)')
    plt.setp(axs[:, 0], ylabel = 'log10(Power)')
    
    fig.suptitle(dirname + ' ' + suptitle_text)
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      

    ylim = [-10, -2]
    if feature_key == 'fooof_flat':
        ylim = [-2, 2]
    for i in range(len(contacts)):  
        contact = contacts[i]
        ax = fig.get_axes()[i]
        ax.set_ylim(ylim)
        ax.set_xlim([0, 100])

        ax.axvspan(4, 8, color = 'grey', alpha = 0.1)
        ax.axvspan(13, 30, color = 'grey', alpha = 0.1)
        ax.axvspan(60, 90, color = 'grey', alpha = 0.1)
        ax.set_title(contact)

        if (measure == 'overlaid'):
            df = pd.DataFrame(da.sel(contact = contact).values,
                              columns = da.feature.values, 
                              index = da.measure.values)
            alpha = 0.1
            if df.shape[0] > 3000:
                alpha = 0.01
            for j in range(df.shape[0]):
                df_singrow = df.iloc[j, :]
                ax.plot(df_singrow.index, 
                        df_singrow.values,
                        color = colors[0],
                        alpha = alpha)
        
        elif (measure == 'ci'):
            df = pd.DataFrame(da.sel(contact = contact).values, 
                              columns = da.feature, index = da.measure)
            
            ax.plot(df.columns, df.loc['mean'].astype(float), color = colors[0])
            ax.fill_between(df.columns, 
                            df.loc['ci_lower'].astype(float), 
                            df.loc['ci_upper'].astype(float),
                            alpha = 0.5,
                            color = colors[0])

    make_dir(out_gp, 'plots')
    out_dir= make_dir(os.path.join(out_gp, 'plots'), feature_key)
    fig.savefig(os.path.join(out_dir, filename))
    
def plot_pca(da, measure, out_gp, filename, feature_key):       
    dirname = os.path.basename(out_gp)
    contacts = da['contact'].values
    pcs = da['pc'].values
    num_plots = len(contacts)
    
    ncols = math.ceil(num_plots/2)

    fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1, :], xlabel = 'Frequencies (Hz)')
    plt.setp(axs[:, 0], ylabel = 'component values')
    
    fig.suptitle(dirname + ' ' + 'top PCs ' + measure + ' domain')
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      

    for i in range(len(contacts)):  
        contact = contacts[i]
        ax = fig.get_axes()[i]
        ax.set_xlim([0, 100])

        ax.axhline(y=0, color = 'grey')
        ax.axvspan(4, 8, color = 'grey', alpha = 0.1)
        ax.axvspan(13, 30, color = 'grey', alpha = 0.1)
        ax.axvspan(60, 90, color = 'grey', alpha = 0.1)
        ax.set_title(contact)

        if (measure == 'frequency'):
            df = pd.DataFrame(da.sel(contact = contact).values,
                              index = da.pc.values,
                              columns = da.feature.values) 

            for j in range(df.shape[0]):
                pc = pcs[j]
                df_sngl = df.iloc[j, :]
                ax.plot(df_sngl.index, 
                        df_sngl.values,
                        color = colors[j],
                        label = pc)
        
        elif (measure == 'time'):
            print('inp, need to write code')
 
        if i == 0:
            fig.legend(ncol = 2, loc = 'upper right', prop={"size":10}, title = 'components')

    make_dir(out_gp, 'plots')
    out_dir= make_dir(os.path.join(out_gp, 'plots'), feature_key)
    fig.savefig(os.path.join(out_dir, filename))
    
def plot_corrs(da, measure, out_gp, filename, feature_key):
    #dims = da.coords.dims
    dirname = os.path.basename(out_gp)
    target_vals = da['target'].values

    num_plots = len(target_vals)
    ncols = math.ceil(num_plots/2)
    nrows = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})  
    xlabel = 'Frequencies (Hz)'
    ylabel = 'Pearson ' + measure
    if num_plots <= nrows:
        plt.setp(axs[-1], xlabel = xlabel)
        plt.setp(axs[:], ylabel = ylabel)
    else:
        plt.setp(axs[-1, :], xlabel = xlabel)
        plt.setp(axs[:, 0], ylabel = ylabel)
    
    fig.suptitle(dirname)
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      
    for i in range(len(target_vals)):  
        target = target_vals[i]
        ax = fig.get_axes()[i]

        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 100])
        ax.axhline(y=0, color = 'grey')
        ax.axvspan(4, 8, color = 'grey', alpha = 0.1)
        ax.axvspan(13, 30, color = 'grey', alpha = 0.1)
        ax.axvspan(60, 90, color = 'grey', alpha = 0.1)
        ax.set_title(target)

        df = pd.DataFrame(da.sel(target = target, 
                                 measure = measure).values.T, 
                          columns = da.contact, index = da.feature)
        
        for j in range(len(df.columns)):
            channel = df.columns[j]
            ax.plot(df.index, df[channel], label = channel, color = colors[j])

        if i == 0:
            fig.legend(ncol = 2, loc = 'upper right', prop={"size":10}, title = 'channels')
        
    fig.tight_layout()   
       
    make_dir(out_gp, 'plots')
    out_dir= make_dir(os.path.join(out_gp, 'plots'), feature_key)
    fig.savefig(os.path.join(out_dir, filename))

    return df

def plot_spectra(da, da_type, measure, out_gp, feature_key):
    if 'time_interval' in da.dims:
        da = da.sel(time_interval = 'min1')

    dirname = os.path.basename(out_gp)
    if (da_type == 'corr'):
        filename = dirname + '_corr_' + measure + '_linegraphs.pdf'
        plot_corrs(da, measure, out_gp, filename, feature_key)
    elif (da_type == 'psd'):
        filename = dirname + '_spectra_' + measure + '_linegraphs.pdf'
        plot_psds(da, measure, out_gp, filename, feature_key)  
    elif (da_type == 'pca'):
        filename = dirname + '_pca_' + measure + '_linegraphs.pdf'
        plot_pca(da, measure, out_gp, filename, feature_key)  
        
def plot_timeseries(dfl_top_ts, out_gp, feature_key, time='datetime'):
    dirname = os.path.basename(out_gp)
    target_vals = list(dfl_top_ts.keys())
    nplots = len(target_vals)
    xlabel = 'Time (datetime)'
    if time == 'samples':
        xlabel = 'Time (samples)'

    fig, axs = plt.subplots(nrows = nplots, ncols = 1, figsize=(15, 5*nplots))
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1], xlabel = xlabel)
    fig.text(0.007, 0.5, 'scores (normalized)', ha="center", va="center", rotation=90)
    
    fig.suptitle(dirname + ' timeseries')
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      
    locs = ['upper right', 'center right', 'lower right']
    for i in range(len(target_vals)):  
        target = target_vals[i]
        ax = fig.get_axes()[i]

        ax.set_ylim([0, 1])
        ax.set_title(target)

        df = pd.DataFrame(dfl_top_ts[target])
        x = df.index
        if time == 'samples':
            x = range(len(df.index))
        
        for j in range(len(df.columns)):
            feature = df.columns[j]
            ax.plot(x, df[feature], label = feature, color = colors[j], alpha = 0.8)
        ax.legend(ncol = 1, loc = 'upper right', prop={"size":10}, title = 'features')
                
    fig.tight_layout()   
       
    out_dir = make_dir(out_gp, 'plots')
    out_dir = make_dir(os.path.join(out_gp, 'plots'), feature_key)
    filename = 'top_pearsonr_timeseries_' + time
    fig.savefig(os.path.join(out_dir, filename))    
        
def plot_crf(dfl_top_cc, feature_key, out_gp):
    #dims = da.coords.dims
    dirname = os.path.basename(out_gp)
    target_vals = list(dfl_top_cc.keys())

    num_plots = len(target_vals)
    ncols = math.ceil(num_plots/2)
    nrows = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})  
    xlabel = 'Lag (# of samples)'
    ylabel = 'Pearsons r'
    if num_plots <= nrows:
        plt.setp(axs[-1], xlabel = xlabel)
        plt.setp(axs[:], ylabel = ylabel)
    else:
        plt.setp(axs[-1, :], xlabel = xlabel)
        plt.setp(axs[:, 0], ylabel = ylabel)
    
    fig.suptitle(dirname)
    
    colors = ['b', 'o']
    for i in range(len(target_vals)):  
        target = target_vals[i]
        ax = fig.get_axes()[i]

        ax.set_ylim([-1, 1])
        ax.set_title(target)
        ax.axvline(x=0, color = 'grey', alpha = 0.5)

        df = dfl_top_cc[target]
        
        for j in range(len(df.columns)):
            channel = df.columns[j]
            ax.stem(range(-len(df.iloc[:, j])//2, 
                          len(df.iloc[:,j])//2),
                    df.iloc[:,j],
                    linefmt = 'C' + str(j) + '-',
                    markerfmt = 'C' + str(j) + 'o',
                    label = channel)

        ax.legend(ncol = 2, loc = 'upper right', prop={"size":10}, title = 'features')
        
    fig.tight_layout()   
       
    filename = 'corr_r_cross'
    make_dir(out_gp, 'plots')
    out_dir= make_dir(os.path.join(out_gp, 'plots'), feature_key)
    fig.savefig(os.path.join(out_dir, filename))
    
def get_top_sigcorrs(da_pearsonr, abs_val=False):
    contacts = da_pearsonr['contact'].values
    frequencies = da_pearsonr['feature'].values
    target_list = da_pearsonr['target'].values
    measures = ['r', 'r_pval', 'frequency']

    ds_out = xr.DataArray(
                        dims = ['target', 'contact', 'measure'],
                        coords = dict(
                            target = target_list,
                            measure = measures,
                            contact = contacts,
                            )
                        )    
    
    for i in range(len(target_list)):
        df_out = pd.DataFrame([],
                          columns = measures, 
                          index = contacts)
        target = target_list[i]
        df_corr = pd.DataFrame(data = da_pearsonr.sel(target = target, measure = 'r').values,
                                                      index = contacts,
                                                      columns = frequencies)
        df_pval = pd.DataFrame(data = da_pearsonr.sel(target = target, measure = 'r_pval').values,
                                                      index = contacts,
                                                      columns = frequencies)
        if df_corr.isnull().values.any():
            continue
        if abs_val == True:
            df_sigcorr = abs(df_corr.copy())  
        else: 
            df_sigcorr = df_corr.copy()
            
        df_sigcorr[df_pval.values > 0.05] = 0
        
        #remove low frequencies with artifact
        freq_thresh = 4
        if 'tremor' in target:
            freq_thresh = 2
        cols_keep = df_sigcorr.columns[df_sigcorr.columns.astype(float) >= freq_thresh]
        df_sigcorr = df_sigcorr.iloc[:, cols_keep]

        df_out['r'] = df_sigcorr.T.max()
        df_out['r_pval'] = np.diag(df_pval[df_sigcorr.T.idxmax()])
        df_out['frequency'] = df_sigcorr.T.idxmax().round(1)
        
        ds_out.loc[dict(target=target)] = df_out

    return ds_out

def get_top_channels(da_top_sigcorrs):
    targets = da_top_sigcorrs['target'].values
    df_ch = pd.DataFrame([], 
                         columns = targets)

    for i in range(len(targets)):
        target = targets[i]
        df_target = pd.DataFrame(da_top_sigcorrs.sel(target=target).values, 
                                 columns = da_top_sigcorrs['measure'].values,
                                 index = da_top_sigcorrs['contact'].values)
        ch_comps = [[0, 1], [2, 3]]
        target_r = df_target['r']    

        if (target_r.isnull().values.any()): #default to channels +3-1 & 9-8 or 10-8
            df_ch.loc[:, target] = target_r.index[1:3].values
        else:
            for j in range(len(ch_comps)):
                top_j = np.argsort(target_r[ch_comps[j]])   
                ch = top_j[top_j==1].index[0]  
                df_ch.loc[j, target] = ch
    return df_ch

def get_top_corr_chanfreqs(da_top_sigcorrs, df_top_ch, target):
        df_corrs = pd.DataFrame(data = da_top_sigcorrs.sel(target = target).values,
                                columns = da_top_sigcorrs['measure'].values,
                                index = da_top_sigcorrs['contact'].values
                                )
        channels = df_top_ch[target]
        channel_freqs = df_corrs.loc[channels, 'frequency']
        if channel_freqs.isnull().any():
            if 'dk' in target:
                channel_freqs[0:2] = [65.0, 65.0]
            elif 'tremor' in target:
                channel_freqs[0:2] = [3.0, 3.0]
            elif 'bk' in target:
                channel_freqs[0:2] = [25.0, 25.0]
                
        return channel_freqs

def get_top_timeseries(da_norm, da_top_sigcorrs, df_top_ch):
    df_collection = {}
    targets = da_top_sigcorrs['target'].values
    
    for i in range(len(targets)):
        target = targets[i]
        channels = df_top_ch[target]

        channel_freqs = get_top_corr_chanfreqs(da_top_sigcorrs, df_top_ch, target)

        spacers = ['_', '_']
        
        features = np.char.add(channel_freqs.index.values.astype(str), spacers)
        features = np.char.add(features, channel_freqs.values.astype(str))
        features = np.append(features, target)
        
        df = pd.DataFrame(index = da_norm['measure'].values,
                          columns = features)

        if 'apple' in target:
            time_interval = 'min1'
        if 'pkg' in target:
            time_interval = 'min2'

        for j in range(len(channels)):
            df.loc[:, features[j]] = da_norm.sel(time_interval = time_interval,
                                                 contact = channels[j], 
                                                 feature = channel_freqs[j]).values
        
        df.loc[:, target] = da_norm.sel(time_interval = time_interval, 
                                        contact = channels[0],
                                        feature = target).values
                
        df_collection[target] = df
        
    return df_collection

def compute_top_ccfs(dfl_top_ts):
    df_collection = {}
    targets = list(dfl_top_ts.keys())
    for i in range(len(targets)):
        target = targets[i]
        df = dfl_top_ts[target]
        df = df.dropna()
        max_lag = 50
        df_out = pd.DataFrame(index = range(0, max_lag), 
                              columns = df.columns[0:len(df.columns)-1])
        for j in range(len(df.columns)-1):
            backwards = sm.tsa.stattools.ccf(df.loc[:, target].values, df.iloc[:, j].values, adjusted = False)[0:int(max_lag/2)][::-1]
            forwards = sm.tsa.stattools.ccf(df.iloc[:, j].values, df.loc[:, target].values, adjusted = False)[0:int(max_lag/2)+1]
            df_out.iloc[:, j] = np.r_[backwards, forwards[1:]]
            
        df_collection[target] = df_out

    return df_collection

def compute_lr(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result = result.rename(columns = {0: 'value'})
    result["prediction"] = prediction
    result = result.sort_index()
  
    return result

def run_lr_model(X, y_all, description):
    if (y_all.ndim == 1):
        targets = ['apple_tremor']
    elif (isinstance(y_all, pd.Series)):
        targets = [y_all.name]
    else:
        targets = y_all.columns
    dfl_predictions = []
    df_metrics = pd.DataFrame(columns = ['mae', 'rmse', 'r2'], index = targets)
    for i in range(len(targets)):
        target = targets[i]
        if y_all.ndim == 1:
            y = y_all
        elif isinstance(y_all, pd.Series):
            y = y_all
        else:
            y = y_all.loc[:, target].to_numpy()
        [X_train, X_val, X_test, y_train, y_val, y_test] = train_val_test_split(X, y, 0.2, shuffle=False)
        df_prediction = compute_lr(X_train, y_train, X_test, y_test)
        df_prediction.columns = ['value', 'prediction']
        df_metrics.loc[target, :] = calculate_metrics(df_prediction)
        dfl_predictions.append(df_prediction)
    
    df_multidim = np.stack(dfl_predictions)  
        
    da = xr.DataArray(data = df_multidim, 
                      dims = ['target', 'measure', 'value'], 
                      coords = dict(
                          target = targets,
                          #measure=range(0, df_multidim.shape[1]),
                          value=df_prediction.columns
                          ),
                      attrs=dict(description=description),
                      )
    return [da, df_metrics]

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction)}   

def run_blstm(X, y, target):    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.2, shuffle=False)

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    
    y_train_arr = scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_arr = scaler.transform(y_val.reshape(-1, 1))
    y_test_arr = scaler.transform(y_test.reshape(-1, 1))
    
    batch_size = 64

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)
    
    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    input_dim = len(X_train.columns)
    output_dim = 1
    hidden_dim = 64
    layer_dim = 3
    batch_size = 64
    dropout = 0.2
    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-6
    
    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}
    
    model = get_model('lstm', model_params)
    
    loss_fn = torch.nn.MSELoss(reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    opt.plot_losses()
    
    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
    
    df_result = format_predictions(predictions, values, X_test, scaler)
    result_metrics = calculate_metrics(df_result)
    
    df_baseline = run_lr_model(X, y, target)
    #df_baseline['value'] = df_baseline[target]
    baseline_metrics = calculate_metrics(df_baseline)

    fig = plot_predictions_MO(df_result, df_baseline)
    
    return [result_metrics, baseline_metrics, fig]

def plot_predictions_MO(df_result, df_baseline):
    plt.close()
    plt.rcParams["figure.figsize"] = (15,5)

    plt.plot(df_result.index, df_result.value, color = 'gray', label = 'apple_tremor')
    plt.plot(df_baseline.index, df_baseline.prediction, alpha  = 0.5, label = 'linear regression')
    plt.plot(df_result.index, df_result.prediction, alpha = 0.5, label = 'ML model')
    plt.title('Model comparison')


    plt.legend(ncol = 1, loc = 'upper right')
    plt.ylabel('scores (normalized)')
    plt.xlabel('Time')

class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = torch.nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
    
class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = torch.nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

def get_model(model, model_params):
    models = {
            "rnn": RNNModel,
            "lstm": LSTMModel,
            "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)
    
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        device = torch.device("cpu")
        model_path = f'models/{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        device = torch.device("cpu")
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values
    
    def plot_losses(self):
        plt.close()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result








        