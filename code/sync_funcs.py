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
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from datetime import date
import datetime
from itertools import combinations


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

def convert_category(df, quantile_list):
    targets = df.columns
    df_quantiles = pd.DataFrame(index = quantile_list, columns = targets)
    #log scale
    df_cat = np.log10(df)
    df_cat = df_cat.replace([-np.inf],0)
    for i in range(len(targets)):
        target = targets[i]
        symps = df_cat[target][df_cat[target] != 0]
        if target == 'pkg_bk':
            symps = df_cat[target][df_cat[target] > 0.1] #remove negative bk scores
        symps_norm = (symps-min(symps))/(max(symps)-min(symps))
        df_quantiles[target] = symps_norm.quantile(quantile_list)

    return df_quantiles
    
def plot_categories(df_norm, quantiles):
    targets = df_norm.columns
        
    nplots = len(targets)
    xlabel = 'Time (datetime)'

    fig, axs = plt.subplots(nrows = nplots, ncols = 1, figsize=(15, 5*nplots))
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1], xlabel = xlabel)
    fig.text(0.007, 0.5, 'scores (normalized)', ha="center", va="center", rotation=90)
    
    fig.suptitle('20th percentile categories of symptoms')
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      
    locs = ['upper right', 'center right', 'lower right']
    for i in range(len(targets)):  
        target = targets[i]
        ax = fig.get_axes()[i]

        ax.set_ylim([0, 1])
        ax.set_title(target)

        df = pd.DataFrame(df_norm[target])
        x = df.index
        
        for j in range(len(df.columns)):
            feature = df.columns[j]
            ax.plot(x, df[feature], label = feature, color = colors[0], alpha = 0.8)
            for k in range(quantiles.shape[0]):
                ax.hlines(y = quantiles[target].iloc[k], 
                           xmin = df.index[0], xmax = df.index[len(df)-1],
                           color = colors[1])
        ax.legend(ncol = 1, loc = 'upper right', prop={"size":10}, title = 'features')

def continuous2ordinal(df_norm, quantiles):
    targets = df_norm.columns
    df_norm_ord = df_norm.copy()
    for i in range(len(targets)):
        target = targets[i]
        df_norm_ord[target][df_norm[target] < quantiles[target].iloc[0]] = 0
        df_norm_ord[target][(df_norm[target] > quantiles[target].iloc[0]) & 
                            (df_norm[target] < quantiles[target].iloc[1])] = 1
        df_norm_ord[target][(df_norm[target] > quantiles[target].iloc[1]) & 
                            (df_norm[target] < quantiles[target].iloc[2])] = 2
        df_norm_ord[target][(df_norm[target] > quantiles[target].iloc[2]) & 
                            (df_norm[target] < quantiles[target].iloc[3])] = 3
        df_norm_ord[target][(df_norm[target] > quantiles[target].iloc[3]) & 
                            (df_norm[target] <= quantiles[target].iloc[4])] = 4

    return df_norm_ord
        
def merge_dfs_time(x, y):
    x.index.name = 'samples'
    y.index.name = 'samples'

    df_merged = x.merge(y, left_index = True, right_index = True)
    df_merged = df_merged.dropna()
    
    return df_merged

def scale_data(df, targets, std_thresh, scaler_type):
    
    
    df.loc[:, targets] = df[((df[targets] - df[targets].median(axis=0)) / df[targets].std()).abs() < std_thresh]
    
    scaler = get_scaler(scaler_type)
    df_scale = pd.DataFrame(data = scaler.fit_transform(df),
                            columns = df.columns,
                            index = df.index)
    return df_scale 

def scale_data_ch(da_psd, scaler_type):
    da_psd_norm_ch = da_psd.copy()
    contacts = da_psd.contact.to_numpy()
    time_intervals = da_psd.time_interval.to_numpy()
    for i in range(len(contacts)):
        for j in range(len(time_intervals)):
            df = da_psd.sel(time_interval = time_intervals[j]).sel(contact = contacts[i]).data
            arr = df.flatten().reshape(-1, 1)
            scaler = get_scaler(scaler_type)
            arr_scale = scaler.fit_transform(arr)
            df_scale = arr_scale.reshape(df.shape)
            da_psd_norm_ch.loc[dict(time_interval = time_intervals[j])].loc[dict(contact = contacts[i])] = df_scale
    return da_psd_norm_ch

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

def reshape_xr2df(da, colnames, time_interval):
    data = da.sel(time_interval = time_interval).to_numpy().transpose(1,0,2).reshape(-1, 126*4)
    df = pd.DataFrame(data = data, 
                     columns = colnames,
                     index = da.measure.to_numpy())
    return df

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

def concat_feature2sample(df, nchannels):
    ncols = int(df.shape[1]/nchannels)
    index_labels = np.tile(df.index.to_numpy(), nchannels)

    df_out = pd.DataFrame(columns = range(0, 126),
                          index = index_labels)
    
    df_out.index.name = 'samples'
    df_out.columns.name = 'features'
    
    i_start = 0
    i_stop = df.shape[0]
    
    j_start = 0
    j_stop = ncols

    for i in range(nchannels):
        df_out.iloc[i_start:i_stop, :] = df.iloc[:, j_start:j_stop]
        i_start += df.shape[0]
        i_stop += df.shape[0]
        
        j_start += ncols
        j_stop += ncols
    
    return [df_out, get_contacts(df.columns)]

def get_channels4samples(df_samples, channels):
    nsamples = len(df_samples.index.unique())
    data = np.empty(df_samples.shape[0], dtype = np.dtype('U100'))
    
    i_start = 0
    i_stop = nsamples
    
    #ch_labels = {'+2-0' : 0,
    #             '+3-1' : 0.33,
    #             '+10-8' : 0.66,
    #             '+11-9' : 1}
    
    for channel in channels:
        data[i_start:i_stop] = channel
        i_start += nsamples
        i_stop += nsamples
        
    df = pd.DataFrame(data = data,
                      index = df_samples.index,
                      columns = ['channel'])
    
    return df

def get_toppcs(da_pcaf_ch, df_pcaf, npcs):
    da_pcaf = da_pcaf_ch.copy()

    contacts = da_pcaf.contact.values

    i_start = 0
    i_stop = 126
    
    for i_ch in range(len(contacts)):
        channel = contacts[i_ch]
        da_pcaf.loc[dict(contact = channel)] = df_pcaf.iloc[0:npcs, i_start:i_stop].values
        i_start += 126
        i_stop += 126

    return da_pcaf

def compute_pca_2d(df, pca, pc_labels, domain):
    pcs = pca.fit_transform(df.values).T
    columns = df.index
    if domain == 'features':
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

    if domain == 'features':  
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

    elif domain == 'samples':
            df_pcs = compute_pca_2d(da_psd, pca, pc_labels, domain)
            da = df_pcs
            df_pc_ratio = pca.explained_variance_ratio_

    return [da, df_pc_ratio]
    
def get_pc_components(da_psd):
    da_psd_sub = da_psd.sel(time_interval = 'min1')
    contacts = da_psd.contact.to_numpy()
    
    X_sv = pd.DataFrame([], columns = contacts, index = np.arange(0, da_psd_sub.shape[2]))
    vec_num_pcs = pd.DataFrame([], columns = contacts, index = [0])
    for i in range(len(contacts)):
        df_psd = da_psd_sub.sel(contact = contacts[i]).data
        X_sv.iloc[:, i] = np.linalg.svd(df_psd, full_matrices = False)[1]
        vec_num_pcs[contacts[i]] = 1/(np.sum(X_sv.iloc[:, i]**2)/np.sum(X_sv.iloc[:, i])**2)

    X_sv.index.name = 'singular_values'
    return [X_sv, vec_num_pcs]
        
def plot_pc_components(singular_values, vec_num_pcs):
    plt.figure()
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      
    for i in range(singular_values.shape[1]):
        plt.plot(np.log(singular_values.iloc[:, i]), 
                 color = colors[i],
                 label = singular_values.columns[i],
                 alpha = 0.8)
        plt.axvline(x=vec_num_pcs.iloc[0, i], color = colors[i], alpha = 0.5)
    plt.legend()
    plt.ylabel('log10(singular value)')
    plt.xlabel('Vector #')
    plt.title('Top singular values')
 
def get_pca_energy(da_psd):
    da_psd_sub = da_psd.sel(time_interval = 'min1')
    contacts = da_psd.contact.to_numpy()
    
    X_uavar = pd.DataFrame([], columns = contacts, index = np.arange(0, da_psd_sub.shape[2]))
    for i in range(len(contacts)):
        df_psd = da_psd_sub.sel(contact = contacts[i]).data
        uavar = np.linalg.svd(df_psd, full_matrices = False)[0]
        X_uavar.iloc[:, i] = np.sum(np.square(uavar), axis = 0)
        
    X_uavar.index.name = 'unitary_array_toppcvar'
    return X_uavar
    
def plot_pca_loading(pca_loading):
    plt.figure()
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      
    for i in range(pca_loading.shape[1]):
        plt.plot(pca_loading.iloc[:, i], 
                 color = colors[i],
                 label = pca_loading.columns[i],
                 alpha = 0.8)
        
        plt.axvspan(4, 8, color = 'grey', alpha = 0.01)
        plt.axvspan(13, 30, color = 'grey', alpha = 0.01)
        plt.axvspan(60, 90, color = 'grey', alpha = 0.01)

    plt.legend()
    plt.ylabel('loadings')
    plt.xlabel('Frequency (Hz)')
    plt.title('top PC loadings')

def plot_variance(df_ratios):
    df_cum = np.cumsum(df_ratios)
    title = 'PC cumulative variance explained'
    xlabel = 'principal components'
    ylabel = '% explained variance'
    
    if df_ratios.ndim == 1:
        plt.figure()
        plt.plot(df_cum)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
    else:
        ncols = math.ceil(df_ratios.shape[1]/2) 
        
        fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize=(5*ncols, 10))
        plt.rcParams.update({'font.size': 16})    
        plt.setp(axs[-1, :], xlabel = xlabel)
        plt.setp(axs[:, 0], ylabel = ylabel)
        
        fig.suptitle(title)
        
        ymin = df_cum.min().min() - 0.01
        ymax = math.ceil(df_cum.max().max())
        
        
        for i in range(df_ratios.shape[1]):  
          ax = fig.get_axes()[i]
          ax.set_ylim([ymin, ymax])
          ax.set_title(df_ratios.columns[i])
          ax.plot(df_cum.index.values,
                  df_cum.iloc[:, i],
                  marker = 'o')
    pass    
    
def plot_pcs_symptoms(df_pcat_ch, df_norm, npcs, feature_key, data_dir):
    targets = df_pcat_ch.T.columns
    std_thresh = 10
    scaler_type = 'minmax'
    
    df_pcat_norm = scale_data(df_pcat_ch.T, targets, std_thresh, scaler_type)
    
    for i in range(npcs):    
        pc = i+1

        dfl_top_pcs = {}

        dfl_top_pcs['pkg_dk'] = pd.concat([df_pcat_norm.iloc[:, pc-1:pc], df_norm.loc[:, 'pkg_dk']], axis = 1)
        dfl_top_pcs['pkg_bk'] = pd.concat([df_pcat_norm.iloc[:, pc-1:pc], df_norm.loc[:, 'pkg_bk']], axis = 1)
        dfl_top_pcs['pkg_tremor'] = pd.concat([df_pcat_norm.iloc[:, pc-1:pc], df_norm.loc[:, 'pkg_tremor']], axis = 1)    
        
        plot_timeseries(dfl_top_pcs, data_dir, feature_key, 'samples')
    pass
    
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

    if 'min' in df.columns:
        if 'apple' in target:
            feature_key = 'min1'
        elif 'pkg' in target:
            feature_key = 'min2'
    
    feature_i = [j for j, s in enumerate(df.columns) if feature_key in s]    
    

    df = df.dropna()
    X = df.iloc[:, feature_i]
    y = df.loc[:, target]
        
    df_corrs = pd.DataFrame(index = ['r', 'r_pval'], columns = X.columns)
    for i in range(X.shape[1]):
        df_corrs.iloc[:, i] = stat.pearsonr(X.iloc[:, i], y)
        
    return df_corrs

def c2dto3d(df, contacts, features, description):
    dim = len(contacts)
    df_3d = np.array(np.hsplit(df, dim))
               
    da = xr.DataArray(data = df_3d, 
                      dims = ['contact', 'measure', 'feature'], 
                      coords = dict(
                          contact=contacts,
                          measure=df.index.values,
                          feature=features
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
    if '+' in df.columns.values[0]:
        contacts = get_contacts(df.columns)
        features = get_frequencies(df.columns)

    elif 'pc' in df.columns.values[0]:
        contacts = ['combined_channels']       
        feature_i = [j for j, s in enumerate(df.columns) if feature_key in s]    
        features_init = df.columns[feature_i].values
        features_list = [sub.replace('pc', '') for sub in list(features_init)]
        features = np.array(features_list).astype(int)
        
    targets = list(set(target_list) & set(df.columns))
    
    da = xr.DataArray(
                      dims = ['target', 'contact', 'measure', 'feature'], 
                      coords=dict(
                          target=targets,
                          contact=contacts,
                          measure=['r', 'r_pval'],
                          feature=features
                          ),
                      attrs=dict(description='Pearson r test'),
                      )

    for i in range(len(targets)):
        target = targets[i]
        print_loop(i, len(targets), 'correlating', target)
        df_corrs = compute_correlation(df, feature_key, target)
        description = 'Pearsonr for ' + target
        df_corrs_3d = c2dto3d(df_corrs, contacts, features, description)
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
    da = da.where(da.feature <= 100, drop = True)

    contacts = da['contact'].values
    pcs = da['pc'].values
    num_plots = len(contacts)
    
    ncols = math.ceil(num_plots/2)

    fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1, :], xlabel = 'Frequencies (Hz)')
    plt.setp(axs[:, 0], ylabel = 'PC loadings')
    
    fig.suptitle(dirname + ' ' + 'PCA loadings')
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']      

    ymin = da.values.min()
    ymax = da.values.max()

    for i in range(len(contacts)):  
        contact = contacts[i]
        ax = fig.get_axes()[i]
        ax.set_xlim([0, 100])
        ax.set_ylim([ymin, ymax])

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

def plot_pcs(da):
    da = da.where(da.feature <= 100, drop = True)
    
    channels = da['contact'].values
    pcs = da['pc'].values
    
    npcs = len(pcs)
    nchannels = len(channels)
    
    colors = ['#3976AF', '#F08536', '#519D3E', '#C63A32', '#8D6BB8', 
              '#84584E', '#D57FBE', '#BDBC45', '#56BBCC']  
    
    fig, axs = plt.subplots(nrows = npcs, ncols = nchannels, 
                            figsize=(5*nchannels, 15),
                            constrained_layout = True)
    
    plt.rcParams.update({'font.size': 16})    
    plt.setp(axs[-1, :], xlabel = 'Frequencies (Hz)')
    plt.setp(axs[:, 0], ylabel = 'PC loadings')

    fig.suptitle('Individual PC loadings')

    ymin = da.values.min()
    ymax = da.values.max()
    
    for i_ch in range(nchannels):
        channel = channels[i_ch]
        for i_pc in range(npcs):
            pc = pcs[i_pc]

            ax = axs[i_pc, i_ch]
            ax.set_xlim([0, 100])
            ax.set_ylim([ymin, 10])
            ax.axhline(y=0, color = 'grey')
            ax.axvspan(4, 8, color = 'grey', alpha = 0.1)
            ax.axvspan(13, 30, color = 'grey', alpha = 0.1)
            ax.axvspan(60, 90, color = 'grey', alpha = 0.1)

            ax.set_title(channel)         

            df = pd.DataFrame(da.sel(contact = channel, 
                                     pc = pc).values.T, 
                              columns = [channel],
                              index = da.feature)
            
            ax.plot(df, color = colors[i_pc], label = pc)
            ax.legend()

    pass    

def plot_spectra_df(df, npcs):

    pc_list = df.index.values[0:npcs]
    
    for i in range(len(pc_list)):
        pc = pc_list[i]
        plt.plot(df.loc[pc, :], label = pc)

    plt.rcParams.update({'font.size': 16})  

    plt.axvspan(4, 8, color = 'grey', alpha = 0.1)
    plt.axvspan(13, 30, color = 'grey', alpha = 0.1)
    plt.axvspan(60, 90, color = 'grey', alpha = 0.1)
    
    plt.xlim(0, 100)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PC loadings')
    plt.legend()

    pass

def plot_pcs_clustered(df_pca, df_targets):
    targets = df_targets.columns.values
    pcs = df_pca.columns.values
    combos_list = list(combinations(pcs, 2))

    if 'channel' in df_targets.columns:
        channel_labels = df_targets.channel.unique()
        colors = {channel_labels[0]: 'tab:blue',
                  channel_labels[1]: 'tab:orange',
                  channel_labels[2]: 'tab:red',
                  channel_labels[3]: 'tab:green'}

    for i_t in range(len(targets)):
        target = targets[i_t]
        
        df = pd.concat([df_pca, df_targets.loc[:, target]], axis = 1).dropna()
        
        nplots = len(combos_list)
        ncols = math.ceil(nplots/2)
        nrows = 2
        
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, 
                                figsize=(25, 7), constrained_layout = True)
        
        fig.suptitle('clustered time PCs: ' + target)
        plt.rcParams.update({'font.size': 16})  

        for i_c in range(nplots):
            ax = fig.get_axes()[i_c]
            
            xlabel = combos_list[i_c][0]
            ylabel = combos_list[i_c][1]
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            
            alpha = 0.5
            cmap = 'viridis'
            if type(df[target].values[0]) == str:                                 
                ax.scatter(df[xlabel].values, df[ylabel].values, edgecolors = 'none',
                           c = df[target].map(colors), alpha = alpha)
                     
            else:
                sc = ax.scatter(df[xlabel].values, df[ylabel].values, edgecolors = 'none',
                       c = df[target], cmap = cmap , alpha = alpha)
    
        if 'channel' in df_targets.columns:
        # add a legend                
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in colors.items()]
            ax.legend(title='color', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        else:
            cbar = fig.colorbar(sc, ax = ax, orientation = "vertical", shrink = 0.5)
            cbar.ax.get_yaxis().labelpad = 15

            cbar.ax.set_ylabel(target + ' scores', rotation = 270)
    pass

def plot_pcs_clustered_severity(df_pcat_ch_concat, df_channels, df_norm, targets):
    for i in range(len(targets)):
        target = targets[i]
        categories = df_norm[target].dropna().unique()
        for j in range(len(categories)):
            category = categories[j]
            timestamp_list = df_norm[target][df_norm[target] == category].index
            plot_pcs_clustered(df_pcat_ch_concat.T.loc[timestamp_list, :].iloc[:, 0:5], 
                            df_channels.loc[timestamp_list, :])
            plt.title(target + " == " + str(category))
    

def plot_corrs(da, measure, out_gp, filename, feature_key):
    #dims = da.coords.dims
    dirname = os.path.basename(out_gp)
    target_vals = da['target'].values

    num_plots = len(target_vals)
    ncols = math.ceil(num_plots/2)
    nrows = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(5*ncols, 15))
    plt.rcParams.update({'font.size': 16})  
    if 'pc' in feature_key:
        xlabel = 'PC'
    else:
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
        ax.axhline(y=0, color = 'grey')

        if 'pc' not in feature_key:
            ax.set_xlim([0, 100])
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
        df_sigcorr = df_sigcorr.loc[:, cols_keep] #MO edit: iloc to loc 04/15

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

def compute_model(X_train, y_train, X_test, y_test, model):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    if model == 'lr':
        model = LinearRegression()
    elif model == 'lasso':
        model = LassoCV(alphas = np.arange(0, 1, 0.01), cv = cv, n_jobs = -1)
        
        #print('alpha: %f' % model.alphas_)
    else:
        raise ValueError('Incorrect model input: model options are `lr` or `lasso`')

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result = result.rename(columns = {0: 'value'})
    result["prediction"] = prediction
    result = result.sort_index()
  
    return result

def compute_metrics_cv(X, y, target, model, alpha = 0.6):
    if model == 'lr':
        model = LinearRegression()
    elif model == 'lasso':
        model = Lasso(alpha = alpha)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    metrics = pd.DataFrame(columns = ['mae', 'rmse', 'r2'], index = [target])
        
    metrics.mae = abs(np.mean(cross_val_score(model, 
                                  X.to_numpy(), 
                                  y, 
                                  scoring = 'neg_mean_absolute_error', 
                                  cv=cv, 
                                  n_jobs=-1)))
    
    metrics.rmse = abs(np.mean(cross_val_score(model, 
                                       X.to_numpy(), 
                                       y, 
                                       scoring = 'neg_root_mean_squared_error', 
                                       cv=cv, 
                                       n_jobs=-1)))
    
    metrics.r2 = np.mean(cross_val_score(model, 
                                 X.to_numpy(), 
                                 y, 
                                 scoring = 'r2', 
                                 cv=cv, n_jobs=-1))

    return metrics

def match_dims(X, y_all):
    indx_nan = y_all[y_all.isnull().any(axis=1)].index
    if len(indx_nan) > 0:
        X = X.drop(indx_nan, axis = 0)
        y_all = y_all.drop(indx_nan, axis = 0)
    return [X, y_all]

def run_baseline_model(X, y_all, model, description, shuffle):
    [X, y_all] = match_dims(X, y_all)
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
        [X_train, X_val, X_test, y_train, y_val, y_test] = train_val_test_split(X, y, 0.2, shuffle)
        df_prediction = compute_model(X_train, y_train, X_test, y_test, model)
        df_prediction.columns = ['value', 'prediction']
        df_prediction.index = X_test.index
        df_prediction.index.name = 'measure'
        df_metrics.loc[target, :] = calculate_metrics(df_prediction)
        dfl_predictions.append(df_prediction)
    
    df_multidim = np.stack(dfl_predictions)  
        
    da = xr.DataArray(data = df_multidim, 
                      dims = ['target', 'measure', 'value'], 
                      coords = dict(
                          target = targets,
                          measure=df_prediction.index,
                          value=df_prediction.columns
                          ),
                      attrs=dict(description=description),
                      )
    return [da, df_metrics]

def run_baseline_model_cv(X, y_all, model, alpha = 0.6):
    [X, y_all] = match_dims(X, y_all)
    if (y_all.ndim == 1):
        targets = ['apple_tremor']
    elif (isinstance(y_all, pd.Series)):
        targets = [y_all.name]
    else:
        targets = y_all.columns

    df_metrics = pd.DataFrame(columns = ['mae', 'rmse', 'r2'], index = targets)
    for i in range(len(targets)):
        target = targets[i]
        if y_all.ndim == 1:
            y = y_all
        elif isinstance(y_all, pd.Series):
            y = y_all
        else:
            y = y_all.loc[:, target].to_numpy()
        df_metrics.loc[target, :] = compute_metrics_cv(X, y, target, model, alpha).to_numpy()
    
    return df_metrics

def compute_alpha_curve(X, y_all, upper_alpha_thresh, nalpha):
    alpha_vals = np.linspace(0.001, upper_alpha_thresh, num=nalpha)
    
    alpha_curve = pd.DataFrame(columns = y_all.columns.values,
                      index = alpha_vals)
    
    for i in range(nalpha):
        print(i)
        alpha_val = alpha_vals[i]
        stats = run_baseline_model_cv(X, y_all, 'lasso', alpha_val)
        alpha_curve.loc[alpha_val, :] = stats.loc[:, 'r2'].T
    
    alpha_curve.index.name  = 'alpha_values'
    
    return alpha_curve
    
def plot_alpha_curve(alpha_curve):
    
    measures = alpha_curve.columns.values
    for i in range(len(measures)):
        measure = measures[i]
        plt.plot(alpha_curve.loc[:, measure], label = measure)
        plt.ylabel('r2')
        plt.xlabel('alpha values')
        plt.title('alpha-accuracy curve')

    plt.legend()
    
def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction)}   

def compare_model_metrics(ml_metrics, baseline_metrics, target):
    ml_arr = np.array(list(ml_metrics.items())).T
    bl_arr = baseline_metrics.loc[target, :].to_numpy().reshape(-1, 1)
    table = pd.DataFrame(data = np.concatenate((bl_arr, ml_arr[1].reshape(-1, 1)), axis = 1),
                         columns = ['baseline_model', 'ml_model'],
                         index = [ml_arr[0]])
    return table

def run_blstm(X, y_all, target, baseline_model, description, shuffle, model_inputs): 
    [X, y_all] = match_dims(X, y_all)
    y = y_all.loc[:, target].to_numpy()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, 
                                                                          y, 
                                                                          0.2, 
                                                                          shuffle=shuffle)

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
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
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=shuffle, drop_last=True)

    batch_size = 64
    model_params = {'input_dim': len(X_train.columns),
                    'hidden_dim' : model_inputs['hidden_dim'],
                    'layer_dim' : model_inputs['layer_dim'],
                    'output_dim' : model_inputs['output_dim'],
                    'dropout_prob' : model_inputs['dropout_prob']}
    
    model = get_model('lstm', model_params)
    
    loss_fn = torch.nn.MSELoss(reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=model_inputs['learning_rate'], 
                           weight_decay=model_inputs['weight_decay'])
    
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size, 
              n_epochs=model_inputs['n_epochs'], n_features=model_params['input_dim'])
    fig_losses = opt.plot_losses()
    
    predictions, values = opt.evaluate(test_loader_one, batch_size=1, 
                                       n_features=model_params['input_dim'])
    
    ml_pred = format_predictions(predictions, values, X_test, scaler)
    ml_metrics = calculate_metrics(ml_pred)
    
    [baseline_preds, baseline_metrics] = run_baseline_model(X, y_all, 
                                                            baseline_model, 
                                                            description, 
                                                            shuffle)
    
    baseline_pred = pd.DataFrame(baseline_preds.sel(target=target).data,
                                 columns = baseline_preds.value.to_numpy(),
                                 index = baseline_preds['measure'].data).sort_index()

    fig_preds = plot_predictions_MO(ml_pred, baseline_pred, target)
    
    metrics = compare_model_metrics(ml_metrics, baseline_metrics, target)
    print(metrics)
    return metrics

def plot_predictions_MO(ml_pred, baseline_pred, target):
    #plt.close()
    plt.figure()
    plt.rcParams["figure.figsize"] = (15,5)

    plt.plot(ml_pred.index, ml_pred.value, color = 'gray', label = target)
    plt.plot(baseline_pred.index, baseline_pred.prediction, alpha  = 0.5, label = 'linear regression')
    plt.plot(ml_pred.index, ml_pred.prediction, alpha = 0.5, label = 'ML model')
    plt.title('Model comparison')


    plt.legend(ncol = 1, loc = 'upper right')
    plt.ylabel('scores (normalized)')
    plt.xlabel('Time')
    plt.show()

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
