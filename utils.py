import os
import pandas as pd
import numpy as np
import string
import random
import time

from pathlib import Path
from tqdm import tqdm

import numcodecs

import spikeinterface.full as si


def is_notebook() -> bool:
    """Checks if Python is running in a Jupyter notebook

    Returns
    -------
    bool
        True if notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


### DATARAME UTILS ###
def is_entry(csv_file, entry, subset_columns=None):
    """Checks if a dictionary is already present in a CSV file.

    Parameters
    ----------
    csv_file : str ot path
        The CSV file
    entry : dict
        The entry dictionary to test
    subset_columns : list, optional
        List of str to only check a subset of columns, by default None

    Returns
    -------
    bool
        True if entry is already in the dataframe, False otherwise
    """
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
        if subset_columns is None:
            subset_columns = list(entry.keys())
            
        if np.any([k not in df.columns for k in list(entry.keys())]):
            return False
        
        query = ""
        data_keys = list(entry.keys())
        query_idx = 0
        for i, k in enumerate(data_keys):
            if k in subset_columns:
                v = entry[k]
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


def append_to_csv(csv_file, new_entry, subset_columns=None):
    """Appends a new entry to a CSV file.

    Parameters
    ----------
    csv_file : str ot path
        The CSV file
    entry : dict
        The new entry dictionary to add
    subset_columns : list, optional
        List of str to only check a subset of columns, by default None
    """
    new_df = None
    if csv_file.is_file():
        df_benchmark = pd.read_csv(csv_file, index_col=False)
        if not is_entry(csv_file, new_entry, subset_columns):
            print("Adding new row to csv")
            new_data_arr = {k: [v] for k, v in new_entry.items()}
            new_df = pd.concat([df_benchmark, pd.DataFrame(new_data_arr)])
    else:
        print("Adding new row to csv")
        new_data_arr = {k: [v] for k, v in new_entry.items()}
        new_df = pd.DataFrame(new_data_arr)
    if new_df is not None:
        new_df.to_csv(csv_file, index=False)


### COMPRESSION UTILS ###
def trunc_filter(bits, dtype):
    """Bit truncation filter in numcodecs.

    Parameters
    ----------
    bits : int
        Number of bits to truncate
    dtype : numpy.dtype
        The dtype of the truncation filter

    Returns
    -------
    _type_
        _description_
    """
    scale = 1.0 / (2 ** bits)
    if bits == 0:
        return []
    else:
        return [numcodecs.FixedScaleOffset(offset=0, scale=scale, dtype=dtype)]


def benchmark_compression(rec_to_compress, compressor, zarr_path, filters=None,
                          time_range=[10, 20], channel_chunk_size=-1, **job_kwargs):
    """_summary_

    Parameters
    ----------
    rec_to_compress : _type_
        _description_
    compressor : _type_
        _description_
    zarr_path : _type_
        _description_
    filters : _type_, optional
        _description_, by default None
    time_range : list, optional
        _description_, by default [10, 20]
    channel_chunk_size : int, optional
        _description_, by default -1

    Returns
    -------
    _type_
        _description_
    """
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


### PLOTTING UTILS ###
def prettify_axes(axs, label_fs=15):
    """_summary_

    Parameters
    ----------
    axs : _type_
        _description_
    label_fs : int, optional
        _description_, by default 15
    """
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]
    
    axs = np.array(axs).flatten()
    
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)        
        
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fs)     


#### CLOUD UTILS ###
def get_s3_client(region_name):
    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED
    bc = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name=region_name)
    return bc


def s3_download_public_file(object, destination, bucket, region_name):
    """
    downloads file from public bucket
    :param object: relative path of file" 'atlas/dorsal_cortex_50.nrrd'
    :param destination: full file path on local machine '/usr/ibl/dorsal_cortex_50.nrrd'
    :param bucket: if not specified, 'ibl-brain-wide-map-public'
    :return:
    """
    destination = Path(destination)
    boto_client = get_s3_client(region_name)
    destination.mkdir(parents=True, exist_ok=True)
    object_name = object.split("/")[-1]
    boto_client.download_file(bucket, object, str(destination / object_name))


def s3_download_public_folder(remote_folder, destination, bucket, region_name, skip_patterns=None,
                              overwrite=False, verbose=True):
    """
    downloads a public folder content to a local folder
    :param prefix: relative path within the bucket, for example: 'spikesorting/benchmark'
    :param destination: local folder path
    :param bucket: if not specified, 'ibl-brain-wide-map-public'
    :param boto_client: if not specified, will instantiate one in anonymous mode
    :return:
    """
    boto_client = get_s3_client(region_name)
    response = boto_client.list_objects_v2(Prefix=remote_folder, Bucket=bucket)

    if skip_patterns is not None:
        if isinstance(skip_patterns, str):
            skip_patterns = [skip_patterns]

    for item in response.get('Contents', []):
        object = item['Key']
        if object.endswith('/') and item['Size'] == 0:  # skips  folder
            continue
        local_file_path = Path(destination).joinpath(Path(object).relative_to(remote_folder))
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        skip = False
        if any(sp in object for sp in skip_patterns):
            skip = True

        if not overwrite and local_file_path.exists() and local_file_path.stat().st_size == item['Size'] or skip:
            if verbose:
                print(f"skipping {local_file_path}")
        else:
            if verbose:
                print(f"downloading {local_file_path}")
            boto_client.download_file(bucket, object, str(local_file_path))


def gs_download_folder(bucket, remote_folder, destination):
    dst = Path(destination)
    if not dst.is_dir():
        dst.mkdir()

    if not bucket.endswith("/"):
        bucket += "/"
    src = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m cp -r {src} {dst}")


def gs_upload_folder(bucket, remote_folder, local_folder):
    if not bucket.endswith("/"):
        bucket += "/"
    dst = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m rsync -r {local_folder} {dst}")


### STATS UTILS ###
def cohen_d(x, y):
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def stat_test(df, param, metrics, sig=0.01, verbose=False):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    param : _type_
        _description_
    metrics : _type_
        _description_
    sig : float, optional
        _description_, by default 0.01
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from scipy.stats import kruskal, f_oneway, shapiro, levene
    import scikit_posthocs as sp
    df_gb = df.groupby(param)
    results = {}
    parametric = False
    for metric in metrics:
        if verbose:
            print(f"\nTesting metric {metric}\n")
        results[metric] = {}
        samples = ()
        for val in np.unique(df[param]):
            df_val = df_gb.get_group(val)
            samples += (df_val[metric].values,)
        # shapiro test for normality
        for sample in samples:
            _, pval_n = shapiro(sample)
            if pval_n < sig:
                parametric = True
                if verbose:
                    print("Non normal samples: using non parametric tests")
                break
        # levene test for equal variances
        _, pval_var = levene(*samples)
        if pval_var < sig:
            if verbose:
                print("Non equal variances: using non parametric tests")
            parametric = True
        if parametric:
            pop_test = kruskal
            ph_test = sp.posthoc_conover
        else:
            pop_test = f_oneway
            ph_test = sp.posthoc_ttest
        # run test:
        _, pval = pop_test(*samples)
        if pval < sig:
            # compute posthoc and cohen's d
            posthoc = ph_test(df, val_col=metric, group_col=param, p_adjust='holm')
            if verbose:
                print("Post-hoc")
                if is_notebook():
                    display(posthoc)
            
            pvals = np.tril(posthoc.to_numpy(), -1)
            pvals[pvals == 0] = np.nan
            pvals[pvals >= sig] = np.nan
            ph_c = pd.DataFrame(pvals, columns=ph.columns, index=ph.index)
            cols = ph_c.columns.values
            cohens = ph_c.copy()
            for index, row in ph_c.iterrows():
                val = row.values
                ind_non_nan, = np.nonzero(~np.isnan(val))
                for col_ind in ind_non_nan:
                    x = df_gb.get_group(index)[metric].values
                    y = df_gb.get_group(cols[col_ind])[metric].values
                    cohen = cohen_d(x, y)
                    cohens.loc[index, cols[col_ind]] = cohen
            if verbose:
                if is_notebook():
                    display(cohens)
        else:
            posthoc = None
            cohens = None 
        results[metric]["pvalue"] = pval
        results[metric]["posthoc"] = posthoc
        results[metric]["cohens"] = cohens
        results[metric]["parametric"] = parametric
        results[metric]["samples"] = samples
        
    return results
