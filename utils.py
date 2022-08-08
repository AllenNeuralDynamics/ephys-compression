import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import string
import random
import numcodecs
import time

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
            
        if np.any([k not in df.columns for k in list(data.keys())]):
            return False
        
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
    """
    Compute lsb and median from a regording

    Parameters
    ----------
    recording : BaseRecording

    num_random_chunks : int

    Returns
    -------
    int
        lsb value
    np.arrahy
        median values for each channel
    """
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


def trunc_filter(bits, recording):
    scale = 1.0 / (2 ** bits)
    dtype = recording.get_dtype()
    if bits == 0:
        return []
    else:
        return [numcodecs.FixedScaleOffset(offset=0, scale=scale, dtype=dtype)]


def benchmark_compression(rec_to_compress, compressor, zarr_path, filters=None,
                          time_range=[10, 20], channel_chunk_size=-1, **job_kwargs):
    fs = rec_to_compress.get_sampling_frequency()
    print("compressing")
    t_start = time.perf_counter()
    rec_compressed = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                          compressor=compressor, filters=filters, 
                                          channel_chunk_size=channel_chunk_size,
                                          **job_kwargs)
    t_stop = time.perf_counter()
    elapsed_time = np.round(t_stop - t_start, 2)
    dur = rec_to_compress.get_num_samples() / fs
    xRT = dur / elapsed_time
    cr = np.round(rec_compressed.get_annotation("compression_ratio"), 2)

    # rmse
    print("computing rmse")
    rec_gt_f = si.bandpass_filter(rec_to_compress)
    rec_compressed_f = si.bandpass_filter(rec_compressed)
    frames = np.array(time_range) * fs
    frames = frames.astype(int)
    
    traces_gt = rec_gt_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)
    traces_zarr_f = rec_compressed_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)

    rmse = np.round(np.sqrt(((traces_zarr_f.ravel() - traces_gt.ravel()) ** 2).mean()), 3)

    return rec_compressed, cr, xRT, elapsed_time, rmse

def prettify_axes(axs, label_fs=15):
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]
    
    axs = np.array(axs).flatten()
    
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)        
        
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fs)     

