import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import string
import random

import spikeinterface.full as si


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str


def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def append_to_csv(df_file, new_data, subset_columns=None):
    new_df = None
    if df_file.is_file():
        df_benchmark = pd.read_csv(df_file, index_col=False)
        if not is_entry(df_file, new_data, subset_columns):
            print("Adding new row to csv")
            new_data_arr = {k: [v] for k, v in new_data.items()}
            new_df = pd.concat([df_benchmark, pd.DataFrame(new_data_arr)])
    else:
        print("Adding new row to csv")
        new_data_arr = {k: [v] for k, v in new_data.items()}
        new_df = pd.DataFrame(new_data_arr)
    if new_df is not None:
        new_df.to_csv(df_file, index=False)
        
def is_entry(df_file, data, subset_columns=None):
    if df_file.is_file():
        df = pd.read_csv(df_file)
        if subset_columns is None:
            subset_columns = list(data.keys())
        query = ""
        data_keys = list(data.keys())
        query_idx = 0
        for i, k in enumerate(data_keys):
            if k in subset_columns:
                v = data[k]
                if isinstance(v, str):
                    query += f"{k} == '{v}'"
                else:
                    query += f"{k} == {v}"
                if query_idx < len(subset_columns) - 1:
                    query += " and "
                query_idx += 1
        
        if len(df.query(query)) == 0:
            return False
        else:
            return True
    else:
        return False
    
    
def get_median_and_lsb(recording, num_random_chunks=10):
    # compute lsb and median
    # gather chunks
    chunks = None
    for i in tqdm(range(num_random_chunks), desc="Extracting chunks"):
        chunks_i2 = si.get_random_data_chunks(recording, chunk_size=30000, seed=i**2)
        if chunks is None:
            chunks = chunks_i2
        else:
            chunks = np.vstack((chunks, chunks_i2))
    
    lsb_value = 0 
    num_channels = recording.get_num_channels()
    gain = recording.get_channel_gains()[0]
    dtype = recording.get_dtype()

    channel_idxs = np.arange(num_channels)
    min_values = np.zeros(num_channels, dtype=dtype)
    median_values = np.zeros(num_channels, dtype=dtype)
    offsets = np.zeros(num_channels, dtype=dtype)

    for ch in tqdm(channel_idxs, desc="Estimating channel stats"):
        unique_vals = np.unique(chunks[:, ch])
        unique_vals_abs = np.abs(unique_vals)
        lsb_val = np.min(np.diff(unique_vals))
        
        min_values[ch] = np.min(unique_vals_abs)
        median_values[ch] = np.median(chunks[:, ch]).astype(dtype)
        
        unique_vals_m = np.unique(chunks[:, ch] - median_values[ch])
        unique_vals_abs_m = np.abs(unique_vals_m)
        offsets[ch] = np.min(unique_vals_abs_m)
        
        if lsb_val > lsb_value:
            lsb_value = lsb_val

    print(f"LSB int16 {lsb_value} --> {lsb_value * gain} uV")
    
    return lsb_value, median_values