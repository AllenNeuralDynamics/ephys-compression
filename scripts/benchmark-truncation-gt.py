# Benchmark compression strategies
import spikeinterface.full as si
import probeinterface as pi

import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

from numcodecs import Blosc, FixedScaleOffset

sys.path.append("..")

from utils import append_to_csv, get_median_and_lsb


print(f"spikeinterface: {si.__version__}")

remove_folders = False
overwrite_compression = False
compare_gt_wfs = True
overwrite_study = False
spike_sort = True

# sorter_list = ["tridesclous"]
# sorter_params = {"tridesclous": {"n_jobs_bin": 10, "total_memory": "4G"}}
    
sorter_list = ["kilosort2_5"]
sorter_params = {"kilosort2_5": {"n_jobs_bin": 10, "total_memory": "4G"}}
    
print(f"Remove folders: {remove_folders} -- Overwrite compression: {overwrite_compression} -- "
      f"Compute GT wfs: {compare_gt_wfs} -- Spike sort: {spike_sort} -- Overwrite study: {overwrite_study}")
if spike_sort:
    print(sorter_list)
    print(sorter_params)
    
def trunc_filter(bits, recording):
    scale = 1.0 / (2 ** bits)
    dtype = recording.get_dtype()
    if bits == 0:
        return []
    else:
        return [FixedScaleOffset(offset=0, scale=scale, dtype=dtype)]
    
def trunc_filter_si(bits, recording):
    scale = 1.0 / (2 ** bits)
    dtype = recording.get_dtype()
    
    if bits == 0:
        return recording
    else:
        rec_scaled = si.scale(recording, gain=scale, dtype=dtype)
        original_gains = rec.get_channel_gains()
        rec_scaled.set_channel_gains(original_gains / scale)
        return rec_scaled


data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression" / "lossy"
tmp_folder.mkdir(exist_ok=True)

# NP1.0 and NP2.0
rec_files = ["/home/alessio/Documents/codes/allen/compression/ephys-compression/notebooks/mearec/np1_mearec_dist-corr.h5", 
             "/home/alessio/Documents/codes/allen/compression/ephys-compression/notebooks/mearec/np2_mearec_dist-corr.h5"]

n_jobs = 10
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="10s", progress_bar=True)

clevel = 9
compressor = Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
trunc_bits = [0, 1, 2, 3, 4, 5, 6]

print(f"Benchmarking bit truncation: {trunc_bits}")

trunc_folder = tmp_folder / "gt"
benchmark_file = data_folder / "results" / "benchmark-truncation-gt.csv"

if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
else:
    df = None

trunc_with = "zarr" # zarr | si

gt_dict = {}

for rec_file in rec_files:
    rec_file = Path(rec_file)
    try:
        if trunc_folder.is_dir():
            if overwrite_compression:
                shutil.rmtree(trunc_folder)
        trunc_folder.mkdir(parents=True, exist_ok=True)
    except:
        print("Couldn't remove tmp folder")

    print(f"\n\n\nBenchmarking {rec_file.name}\n\n\n")
    t_start_all = time.perf_counter()

    rec, sort_gt= si.read_mearec(rec_file)
    print(rec)    
    
    if "np1" in rec_file.name:
        probe_name = "Neuropixels1.0"
        lsb_value = 12
    else:
        probe_name = "Neuropixels2.0"
        lsb_value = 3

    gt_dict[probe_name] = {}        
    gt_dict[probe_name]["rec_gt"] = rec
    gt_dict[probe_name]["sort_gt"] = sort_gt
     
    dur = rec.get_num_samples() / rec.get_sampling_frequency()
    dtype = rec.get_dtype()
    gain = rec.get_channel_gains()[0]
    
    num_channels = rec.get_num_channels()
    fs = rec.get_sampling_frequency()
    
    start_frame_1s = int(20 * fs)
    end_frame_1s = int(21 * fs)
    start_frame_10s = int(30 * fs)
    end_frame_10s = int(40 * fs)
    
    full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

    rec_zarr_df = None
    zarr_root = f"{rec_file.name}"

    for trunc_bit in trunc_bits:
        zarr_path = trunc_folder / f"{zarr_root}_trunc{trunc_bit}.zarr"

        if overwrite_compression:
            if zarr_path.is_dir():
                shutil.rmtree(zarr_path)

        if zarr_path.is_dir():
            print(f"{trunc_bit} already computed")
            rec_zarr = si.read_zarr(zarr_path)
            elapsed_time = 0
        else:
            t_start = time.perf_counter()

            # median correction not needed
            rec_to_compress = si.scale(rec, gain=1. / lsb_value, dtype=dtype)
            
            if trunc_with == "zarr":
                filters = trunc_filter(trunc_bit, rec)
            else:
                filters = []
                rec_to_compress = trunc_filter_si(trunc_bit, rec)

            rec_zarr = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                            compressor=compressor, filters=filters, 
                                            **job_kwargs)
            t_stop = time.perf_counter()
            elapsed_time = np.round(t_stop - t_start, 2)

        cr = np.round(rec_zarr._root['traces_seg0'].nbytes / rec_zarr._root['traces_seg0'].nbytes_stored, 2)

        new_data = {"trunc_bit": trunc_bit, "CR": cr, "Cspeed": elapsed_time, 
                    "rec_zarr_path": str(zarr_path.absolute()), "lsb_value": lsb_value,
                    "probe": probe_name}
        append_to_csv(benchmark_file, new_data, subset_columns=["trunc_bit", "probe"])

        print(f"Elapsed time truncation: {trunc_bit}: {elapsed_time}s - CR: {cr}")    
    
    if df is None:
        df = pd.read_csv(benchmark_file, index_col=False)
    
    
    # if "rmse" not in df.columns:
    time_range = [15, 20]
    frames = np.array(time_range) * rec.get_sampling_frequency()
    frames = frames.astype(int)
    
    print(probe_name)
    df_probe = df.query(f"probe == '{probe_name}'")
    rec_0_row = df_probe.query(f"trunc_bit == 0")
    lsb_value = rec_0_row["lsb_value"].values[0]
    rec_zarr_path = rec_0_row["rec_zarr_path"].values[0]
    
    rec_0 = si.read_zarr(rec_zarr_path)
    rec_0 = si.scale(rec_0, gain=lsb_value, dtype=dtype)
    rec_0_f = si.bandpass_filter(rec_0)
    traces_0 = rec_0_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)
        
    # compute rms errors
    compute_rmse = False
    if "rmse" not in df.columns:
        compute_rmse = True
    elif np.any(df["rmse"].isna()):
        compute_rmse = True
    if compute_rmse:
        print("Computing RMSE")
        for index, row in df_probe.iterrows():
            trunc_bit = row["trunc_bit"]
            zarr_path = row["rec_zarr_path"]
            lsb_value = row["lsb_value"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr = si.scale(rec_zarr, gain=lsb_value, dtype=dtype)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            
            traces_trunc_f = rec_zarr_f.get_traces(start_frame=frames[0], end_frame=frames[1], return_scaled=True)

            error_rms = np.sqrt(((traces_trunc_f.ravel() - traces_0.ravel()) ** 2).mean())
            df.at[index, "rmse"] = error_rms
            print(f"RMS for truncation {trunc_bit}: {np.round(error_rms, 4)} uV")
        df.to_csv(benchmark_file, index=False)


if compare_gt_wfs:
    for index, row in df.iterrows():
        trunc_bit = row["trunc_bit"]
        zarr_path = row["rec_zarr_path"]
        lsb_value = row["lsb_value"]
        probe_name = row["probe"]
        
        sort_gt = gt_dict[probe_name]["sort_gt"]
        
        # compare waveform degradation
        waveform_folder = trunc_folder / f"waveforms_{probe_name}"
        waveform_folder.mkdir(exist_ok=True)
        
        rec = si.read_zarr(zarr_path)
        rec = si.scale(rec, gain=lsb_value, dtype=rec.get_dtype())
        
        print(f"Extacting waveforms for trunc bit: {trunc_bit} - probe {probe_name}")
        wf_trunc_path = waveform_folder / str(trunc_bit)
        we = si.extract_waveforms(rec, sort_gt, folder=wf_trunc_path,
                                  load_if_exists=True, **job_kwargs)
        
        df.at[index, "we_path"] = str(wf_trunc_path.absolute())

    # update csv
    df.to_csv(benchmark_file, index=False)
    

if spike_sort:
    for probe_name in np.unique(df.probe):
        print(f"Running study for {probe_name}")
        df_probe = df.query(f"probe == '{probe_name}'")
        gt_study_dict = {}
        study_folder = trunc_folder / f"study_{probe_name}_{sorter_list[0]}"
        sort_gt = gt_dict[probe_name]["sort_gt"]

        for index, row in df_probe.iterrows():
            trunc_bit = row["trunc_bit"]
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr = si.scale(rec_zarr, gain=row["lsb_value"], dtype=dtype)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            
            gt_study_dict[f"bit{trunc_bit}"] = (rec_zarr_f, sort_gt)
        
        if overwrite_study:
            if study_folder.is_dir():
                shutil.rmtree(study_folder)
                
        if not study_folder.is_dir():
            study = si.GroundTruthStudy.create(study_folder, gt_study_dict, **job_kwargs)
        else:
            study = si.GroundTruthStudy(study_folder)
        
        
        # sorter_list = st.sorters.available_sorters() # this get all sorters.
        study.run_sorters(sorter_list, mode_if_folder_exists="keep", verbose=True, 
                          sorter_params=sorter_params)   
        study.copy_sortings()
        # study.run_comparisons(exhaustive_gt=True, verbose=True)
        # dataframes = study.aggregate_dataframes()

     

    # sorting_outputs = si.run_sorters(sorter_list, rec_dict, 
    #                                  working_folder=trunc_folder / "sorting_outputs" / zarr_root,
    #                                  mode_if_folder_exists="keep", engine="loop", 
    #                                  sorter_params=sorter_params, verbose=True)
            
#             # auto curation
#             isi_viol_threshold = 0.5
#             amp_cutoff_threshold = 0.1
#             presence_ratio_threshold = 0.95
            
#             auto_curation_query = (f"isi_violations_ratio < {isi_viol_threshold} and "
#                                     f"amplitude_cutoff < {amp_cutoff_threshold} and "
#                                     f"presence_ratio > {presence_ratio_threshold}")
            
#             # compute waveforms
#             waveform_folder = trunc_folder / "waveforms"
#             waveform_folder.mkdir(exist_ok=True)
#             for trunc_bit, sort in sorting_outputs:
#                 print(f"Extacting waveforms for trunc bit: {trunc_bit}")
#                 wf_trunc_path = waveform_folder / trunc_bit
#                 we = si.extract_waveforms(rec_dict[trunc_bit], sort, folder=wf_trunc_path,
#                                             load_if_exists=True, **job_kwargs)
#                 qm = si.compute_quality_metrics(we)
                
#                 df.at[index, "we_path"] = str(wf_trunc_path.absolute())
