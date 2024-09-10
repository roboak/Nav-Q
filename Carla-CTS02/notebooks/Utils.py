from packaging import version

#import pandas as pd
from matplotlib import pyplot as plt
#import seaborn as sns
#from scipy import stats
import tensorboard as tb
import statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import random
import time
import math
import pickle





#Important Functions to process event files of tensorboard

weight = 0.995
def smooth(scalars: list, weight: float=0.995) -> list:  # Weight between 0 and 1
    '''
    The function used to smoothen the return vs episodes curve
    '''
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    std = 0
    var = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
        delta = -1 if point - smoothed_val < -1 else point - smoothed_val
        var.append(delta)
    return smoothed, var


def import_data_from_multiple_runs(batch_path):
    '''The function is used to import tensorboard log files from multiple runs. 
    batch_path contains multiple run_dirs and each run_dir contains the summary folder where the tensorboard event is stored'''
    sub_dirs = [x for x in os.walk(batch_path)]
    summary_paths = [os.path.join(os.path.join(batch_path, folder), "summary") for folder in sub_dirs[0][1]]
    out = {"r": [], "e": [], "tt":[], "step": []}
    for path in summary_paths:
        returns_list, entropy_list, actual_training_time = get_data_from_single_run(path)
        out["r"].append(returns_list)
        out["e"].append(entropy_list)
        out["tt"].append(actual_training_time)
    return out

def get_data_from_single_run(path):
    '''This function is used to load data from a single run_dir. 
    path is the path of the summary_dir inside the run_dir where the tensorboard event is stored'''
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    w_times, step, value = zip(*event_acc.Scalars('Return/Reward'))
    returns_list = list(value)
    w_times, step, value = zip(*event_acc.Scalars('Stats/Entropy'))
    entropy_list = list(value)
    actual_training_time = 0
    for i in range(len(w_times)-1):
        diff = w_times[i+1] - w_times[i] # in seconds
        if diff < 7200:
            actual_training_time += diff
    return returns_list, entropy_list, actual_training_time

# Saving data to csv
def save_data_to_csv():
    r = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in c_out["r"]])
    e = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in c_out["e"]])
    column_name = ["c_1", "c_2","c_3","c_4","c_5","c_6"]
    c_df_r = pd.DataFrame(r).T
    c_df_r.columns = column_name
    c_df_e = pd.DataFrame(e).T
    c_df_e.columns = column_name

    r = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in q_out["r"]])
    e = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in q_out["e"]])
    column_name = ["q_1", "q_2","q_3","q_4","q_5","q_6"]
    q_df_r = pd.DataFrame(r).T
    q_df_r.columns = column_name
    q_df_e = pd.DataFrame(e).T
    q_df_e.columns = column_name

    df_merged = pd.concat([c_df_r, q_df_r], axis=1)
    # df_merged.shape
    df_merged_e = pd.concat([c_df_e, q_df_e], axis=1)
    # df_merged_e.shape
    df_merged.to_csv("data_test/return.csv", encoding='utf-8', index=False)
    df_merged_e.to_csv("data_test/entropy.csv", encoding='utf-8', index=False)

def get_inference(dict_out, smoothed=True):
    if smoothed:
        r = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in dict_out["r"]])
        e = np.array([smooth(ele[:5000], weight=0.99)[0] for ele in dict_out["e"]])
    else:
        r = np.array([ele[:5000] for ele in dict_out["r"]])
        e = np.array([ele[:5000] for ele in dict_out["e"]])
    std_r = r.std(axis=0)
    std_e = e.std(axis=0)
    range_r = (r.max(axis=0), r.min(axis=0))
    range_e = (e.max(axis=0), e.min(axis=0))
    return r.mean(axis=0), std_r, range_r, e.mean(axis=0), std_e, range_e

def plot_batch_return(data, xlabel="Episodes", ylabel="Return", loc="lower right"):
    
    #data should be a dict {exp_name: data}
    plt.grid()
    for key, value in data.items():
        plt.plot(value, label=key)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=loc)
    plt.show()

def format_key_names(key):
    # change key names from "q_4_l_1_ls_32" to "q=4, l=1"
    ls = key.split('_')
    new_key = ls[0] + "=" + ls[1] + ","+ls[2] + "="+ls[3]
    return new_key





# batch_path = r"D:\Cluster_out_Q-NavSACp\Experiments\A2C_Exp\02.06\vel_fix\Deter\LS_6"
# out = import_data_from_multiple_runs(batch_path)
# sub_dirs = [x for x in os.walk(batch_path)][0][1]
# sub_dirs =['_'.join(s.split('_')[2:]) for s in sub_dirs]
# # Data Structure creation for plots
# data_len=2500
# data = {}
# data_e ={}
# for k,v, e in zip(sub_dirs, out['r'], out['e']):
#     # k = format_key_names(k)
#     if 'back_prop' in k:
#         k = "Nav-Q (backpropagation)"
#     elif 'param_shift_wo_noise' in k:
#         k = "Nav-Q (param shift)"
#     elif 'param_shift' in k:
#         k = "Nav-Q (param shift + noise)"
#     else:
#         k = "NavA2C"
#     r, _ = smooth(v[:data_len])
#     e, _ = smooth(e[:data_len])
#     # r = v[:data_len]
#     data[k] = r
#     data_e[k] = e
# plot_batch_return(data)
# plot_batch_return(data_e, ylabel="Entropy", loc="upper right")