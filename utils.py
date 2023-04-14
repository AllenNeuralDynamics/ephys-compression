import os
import time
from pathlib import Path

import pandas as pd
import numpy as np


import numcodecs
import spikeinterface.preprocessing as spre


def is_notebook() -> bool:
    """Checks if Python is running in a Jupyter notebook

    Returns
    -------
    bool
        True if notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
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
        for k in data_keys:
            if k in subset_columns:
                v = entry[k]
                if isinstance(v, str):
                    query += f"{k} == '{v}'"
                else:
                    query += f"{k} == {v}"
                query += " and "
        # remove final 'and'
        if query.endswith("and "):
            query = query[:-4]

        if len(df.query(query)) == 0:
            return False
        else:
            return True
    else:
        return False


def append_to_csv(csv_file, new_entry, subset_columns=None, verbose=False):
    """Appends a new entry to a CSV file.

    Parameters
    ----------
    csv_file : str ot path
        The CSV file
    entry : dict
        The new entry dictionary to add
    subset_columns : list, optional
        List of str to only check a subset of columns, by default None
    verbose : bool
        If True, it prints whether the new entry was successfull, by default False
    """
    new_df = None
    if csv_file.is_file():
        df_benchmark = pd.read_csv(csv_file, index_col=False)
        if not is_entry(csv_file, new_entry, subset_columns):
            new_data_arr = {k: [v] for k, v in new_entry.items()}
            new_df = pd.concat([df_benchmark, pd.DataFrame(new_data_arr)])
    else:
        new_data_arr = {k: [v] for k, v in new_entry.items()}
        new_df = pd.DataFrame(new_data_arr)
    if new_df is not None:
        if verbose:
            print("Adding new row to csv")
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
    list
        List of numcodecs filters
    """
    scale = 1.0 / (2**bits)
    if bits == 0:
        return []
    else:
        return [numcodecs.FixedScaleOffset(offset=0, scale=scale, dtype=dtype)]


def benchmark_lossy_compression(
    rec_to_compress, compressor, zarr_path, filters=None, time_range_rmse=[10, 20], channel_chunk_size=-1, **job_kwargs
):
    """Benchmarks lossy compression for one recording, including:

    - compression ratio (CR)
    - compression speed
    - root mean squared error (RMSE)

    Parameters
    ----------
    rec_to_compress : spikeinterface.BaseRecording
        The recording to compress
    compressor : numcodecs.Codec
        The compressor to use
    zarr_path : str or path
        The output zarr path
    filters : list, optional
        List of numcodecs filters, by default None
    time_range_rmse : list, optional
        Time range to compute RMSE, by default [10, 20]
    channel_chunk_size : int, optional
        The chunk size in the channel dimensions, by default -1

    Returns
    -------
    rec_compressed : ZarrRecordingExtractor
        The compressed recording
    cr : float
        The compression ratio
    cspeed_xrt : float
        The compression speed in x real-time
    cspeed : float
        Compression speed in seconds
    rmse : float
        The RMSE value
    """
    fs = rec_to_compress.get_sampling_frequency()
    t_start = time.perf_counter()
    rec_compressed = rec_to_compress.save(
        format="zarr",
        folder=zarr_path,
        compressor=compressor,
        filters=filters,
        channel_chunk_size=channel_chunk_size,
        **job_kwargs,
    )
    t_stop = time.perf_counter()
    cspeed = np.round(t_stop - t_start, 2)
    dur = rec_to_compress.get_num_samples() / fs
    cspeed_xrt = dur / cspeed
    cr = np.round(rec_compressed.get_annotation("compression_ratio"), 2)

    # rmse
    rec_gt_f = spre.bandpass_filter(rec_to_compress)
    rec_compressed_f = spre.bandpass_filter(rec_compressed)
    frames = np.array(time_range_rmse) * fs
    frames = frames.astype(int)

    traces_gt = rec_gt_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)
    traces_zarr_f = rec_compressed_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)

    rmse = np.round(np.sqrt(((traces_zarr_f.ravel() - traces_gt.ravel()) ** 2).mean()), 3)

    return rec_compressed, cr, cspeed_xrt, cspeed, rmse


### PLOTTING UTILS ###
def prettify_axes(axs, label_fs=15):
    """Makes axes prettier by removing top and right spines and fixing label fontsizes.

    Parameters
    ----------
    axs : list
        List of matplotlib axes
    label_fs : int, optional
        Label font size, by default 15
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
    """Set up s3 public client

    Parameters
    ----------
    region_name : str
        The region name

    Returns
    -------
    boto3.client
        The boto3 Client
    """
    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED

    bc = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=region_name)
    return bc


def s3_download_public_file(object, destination, bucket, region_name):
    """Downloads a file from a S3 public bucket.

    Parameters
    ----------
    object : str
        The object to download
    destination : str
        The destination path
    bucket : str
        The public bucket name
    region_name : str
        The region of the public bucket
    """
    destination = Path(destination)
    boto_client = get_s3_client(region_name)
    destination.mkdir(parents=True, exist_ok=True)
    object_name = object.split("/")[-1]
    boto_client.download_file(bucket, object, str(destination / object_name))


def s3_download_public_folder(
    remote_folder, destination, bucket, region_name, skip_patterns=None, overwrite=False, verbose=True
):
    """Downloads a folder from a S3 public bucket.

    Parameters
    ----------
    remote_folder : str
        The remote folder in the bucket
    destination : str
        The local destination folder
    bucket : str
        The public bucket name
    region_name : str
        The region of the public bucket
    skip_patterns : list, optional
        List of string patterns to skip, by default None
    overwrite : bool, optional
        If True, it overwrites to local destination, by default False
    verbose : bool, optional
        If True output is verbose, by default True
    """
    boto_client = get_s3_client(region_name)
    response = boto_client.list_objects_v2(Prefix=remote_folder, Bucket=bucket)

    if skip_patterns is not None:
        if isinstance(skip_patterns, str):
            skip_patterns = [skip_patterns]

    for item in response.get("Contents", []):
        object = item["Key"]
        if object.endswith("/") and item["Size"] == 0:  # skips  folder
            continue
        local_file_path = Path(destination).joinpath(Path(object).relative_to(remote_folder))
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        skip = False
        if any(sp in object for sp in skip_patterns):
            skip = True

        if not overwrite and local_file_path.exists() and local_file_path.stat().st_size == item["Size"] or skip:
            if verbose:
                print(f"skipping {local_file_path}")
        else:
            if verbose:
                print(f"downloading {local_file_path}")
            boto_client.download_file(bucket, object, str(local_file_path))


def s3_download_folder(bucket, remote_folder, destination):
    """Downloads a folder from an S3 bucket using aws s3 CLI.
    It assumes credentials are correctly set to access the bucket.

    Parameters
    ----------
    bucket : str
        The bucket name
    remote_folder : str
        The remote folder in the bucket
    destination : str
        The local destination folder
    """
    dst = Path(destination)
    if not dst.is_dir():
        dst.mkdir(parents=True)

    if not bucket.endswith("/"):
        bucket += "/"
    src = f"{bucket}{remote_folder}"

    os.system(f"aws s3 sync {src} {dst}")


def s3_upload_folder(bucket, remote_folder, local_folder):
    """Uploads a folder to a s3 bucket using aws s3 CLI.
    It assumes credentials are correctly set to access the bucket.

    Parameters
    ----------
    bucket : str
        The bucket name
    remote_folder : str
        The remote folder in the bucket
    local_folder : str
        The local folder to upload
    """
    if not bucket.endswith("/"):
        bucket += "/"
    dst = f"{bucket}{remote_folder}"

    os.system(f"aws s3 sync {local_folder} {dst}")


def gs_download_folder(bucket, remote_folder, destination):
    """Downloads a folder from a GCS bucket using gsutil.
    It assumes credentials are correctly set to access the bucket.

    Parameters
    ----------
    bucket : str
        The bucket name
    remote_folder : str
        The remote folder in the bucket
    destination : str
        The local destination folder
    """
    dst = Path(destination)
    if not dst.is_dir():
        dst.mkdir(parents=True)

    if not bucket.endswith("/"):
        bucket += "/"
    src = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m cp -r {src} {dst}")


def gs_upload_folder(bucket, remote_folder, local_folder):
    """Uploads a folder to a GCS bucket using gsutil.
    It assumes credentials are correctly set to access the bucket.

    Parameters
    ----------
    bucket : str
        The bucket name
    remote_folder : str
        The remote folder in the bucket
    local_folder : str
        The local folder to upload
    """
    if not bucket.endswith("/"):
        bucket += "/"
    dst = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m rsync -r {local_folder} {dst}")


### STATS UTILS ###
def cohen_d(x, y):
    """Computes the Cohen's d coefficient between samples x and y

    Parameters
    ----------
    x : np.array
        Sample x
    y : np.array
        Sample y

    Returns
    -------
    float
        the Cohen's d coefficient
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def stat_test(df, column_group_by, test_columns, sig=0.01, verbose=False):
    """Performs statistical tests and posthoc analysis (in case of multiple groups).

    If the distributions are normal with equal variance, it performs the ANOVA test and
    posthoc T-tests (parametric).
    Otherwise, the non-parametric Kruskal-Wallis and posthoc Conover's tests are used.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe
    column_group_by : str
        The categorical column used for grouping
    test_columns : list
        The columns containing real values to test for differences.
    sig : float, optional
        Significance level, by default 0.01
    verbose : bool, optional
        If True output is verbose, by default False

    Returns
    -------
    dict
        The results dictionary containing, for each metric:

        - "pvalue" :  the p-value for the multiple-sample test
        - "posthoc" : DataFrame with posthoc p-values
        - "cohens": DataFrame with Cohen's d coefficients for significant posthoc results
        - "parametric": True if parametric, False if non-parametric
    """
    from scipy.stats import kruskal, f_oneway, shapiro, levene, ttest_ind, mannwhitneyu
    import scikit_posthocs as sp

    df_gb = df.groupby(column_group_by)
    results = {}
    parametric = False
    for metric in test_columns:
        if verbose:
            print(f"\nTesting metric {metric}\n")
        results[metric] = {}
        samples = ()
        for i, val in enumerate(np.unique(df[column_group_by])):
            df_val = df_gb.get_group(val)
            if verbose:
                print(f"Sample {i+1}: {val} - n. {len(df_val)}")
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
        if not parametric:
            _, pval_var = levene(*samples)
            if pval_var < sig:
                if verbose:
                    print("Non equal variances: using non parametric tests")
                parametric = True
        if len(samples) > 2:
            if verbose:
                print("Population test")
            parametric = True
            if parametric:
                test_fun = kruskal
                ph_test = sp.posthoc_conover
            else:
                test_fun = f_oneway
                ph_test = sp.posthoc_ttest
            # run test:
            _, pval = test_fun(*samples)
            pval_round = pval
            if pval < sig:
                # compute posthoc and cohen's d
                posthoc = ph_test(df, val_col=metric, group_col=column_group_by, p_adjust="holm", sort=False)

                # here we just consider the bottom triangular matrix and just keep significant values
                pvals = np.tril(posthoc.to_numpy(), -1)
                pvals[pvals == 0] = np.nan
                pvals[pvals >= sig] = np.nan

                # cohen's d are computed only on significantly different distributions
                ph_c = pd.DataFrame(pvals, columns=posthoc.columns, index=posthoc.index)
                pval_round = ph_c.copy()
                cols = ph_c.columns.values
                cohens = ph_c.copy()
                for index, row in ph_c.iterrows():
                    val = row.values
                    (ind_non_nan,) = np.nonzero(~np.isnan(val))
                    for col_ind in ind_non_nan:
                        x = df_gb.get_group(index)[metric].values
                        y = df_gb.get_group(cols[col_ind])[metric].values
                        cohen = cohen_d(x, y)
                        cohens.loc[index, cols[col_ind]] = cohen
                        pval = ph_c.loc[index, cols[col_ind]]
                        if pval < 1e-10:
                            exp = -10
                        else:
                            exp = int(np.ceil(np.log10(pval)))
                        pval_round.loc[index, cols[col_ind]] = f"<1e{exp}"
                if verbose and is_notebook():
                    print("Post-hoc")
                    display(ph_c)
                    print("Post-hoc")
                    display(pval_round)
                    print("Cohen's d")
                    display(cohens)
            else:
                if verbose:
                    print("Non significant")
                posthoc = None
                cohens = None
                pval_round = None
        else:
            if verbose:
                print("2-sample test")
            posthoc = None
            if parametric:
                test_fun = ttest_ind
            else:
                test_fun = mannwhitneyu
            _, pval = test_fun(*samples)
            if pval < sig:
                cohens = cohen_d(*samples)
                if verbose:
                    if pval < 1e-10:
                        pval_round = "<1e-10"
                    else:
                        exp = int(np.ceil(np.log10(pval)))
                        pval_round = f"<1e{exp}"
                    print(f"P-value {pval_round} ({pval}) - effect size: {np.round(cohens, 3)}")
            else:
                if verbose:
                    print("Non significant")
                posthoc = None
                pval_round = None
                cohens = None

        results[metric]["pvalue"] = pval
        results[metric]["pvalue-round"] = pval_round
        results[metric]["posthoc"] = posthoc
        results[metric]["cohens"] = cohens
        results[metric]["parametric"] = parametric

    return results
