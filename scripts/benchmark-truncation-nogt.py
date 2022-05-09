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
overwrite = False
spike_sort = True
autocuration = True

sorter_list = ["kilosort2_5"]
sorter_params = {"kilosort2_5": {"n_jobs_bin": 10, "total_memory": "4G"}}
    
    
print(f"Remove folders: {remove_folders} -- Overwrite: {overwrite} -- Spike sort: {spike_sort}")
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
if tmp_folder.is_dir():
    if overwrite:
        shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True)

# NP1.0 and NP2.0
rec_files = ["/home/alessio/Documents/data/allen/npix-open-ephys/595262_2022-02-21_15-18-07/Record Node 102", 
             "/home/alessio/Documents/data/allen/npix-open-ephys/618382_2022-03-31_14-27-03/Record Node 102"]

n_jobs = 10
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="10s", progress_bar=True)

clevel = 9
compressor = Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE,)
trunc_bits = [0, 1, 2, 3, 4, 5, 6]

print(f"Benchmarking bit truncation: {trunc_bits}")

trunc_folder = tmp_folder / "nogt"
benchmark_file = data_folder / "results" / "benchmark-truncation-nogt.csv"

if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
    print(f"Number of entries in dataframe: {len(df)}")
else:
    df = None
    

trunc_with = "zarr" # zarr | si

for rec_file in rec_files:
    rec_file = Path(rec_file)
    try:
        if trunc_folder.is_dir():
            if overwrite:
                shutil.rmtree(trunc_folder)
        trunc_folder.mkdir(parents=True, exist_ok=True)
    except:
        print("Couldn't remove tmp folder")

    print(f"\n\n\nBenchmarking {rec_file.name}\n\n\n")
    t_start_all = time.perf_counter()
    
    probe = pi.read_openephys(rec_file)
    probe_name = probe.annotations["probe_name"]
    print(probe_name)
    
    if "1.0" in probe_name:
        stream_id = "0"
        probe_name = "Neuropixels1.0"
    else:
        stream_id = None
        probe_name = "Neuropixels2.0"
    rec = si.read_openephys(rec_file, stream_id=stream_id)
    rec = si.split_recording(rec)[0]   
    print(rec)    
     
    dur = rec.get_num_samples() / rec.get_sampling_frequency()
    dtype = rec.get_dtype()
    gain = rec.get_channel_gains()[0]
    
    lsb_value, median_values = get_median_and_lsb(rec)

    num_channels = rec.get_num_channels()
    fs = rec.get_sampling_frequency()
    
    start_frame_1s = int(20 * fs)
    end_frame_1s = int(21 * fs)
    start_frame_10s = int(30 * fs)
    end_frame_10s = int(40 * fs)
    
    full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

    rec_zarr_df = None
    zarr_root = f"{rec_file.parent.name}"

    for trunc_bit in trunc_bits:
        zarr_path = trunc_folder / f"{zarr_root}_trunc{trunc_bit}.zarr"

        if overwrite:
            if zarr_path.is_dir():
                shutil.rmtree(zarr_path)

        if zarr_path.is_dir():
            print(f"{trunc_bit} already computed")
            rec_zarr = si.read_zarr(zarr_path)
            elapsed_time = 0
        else:
            t_start = time.perf_counter()

            # median correction
            rec_to_compress = si.scale(rec, gain=1., offset=-median_values, dtype=dtype)
            rec_to_compress = si.scale(rec_to_compress, gain=1. / lsb_value, dtype=dtype)
            
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
        
    # # compute rms errors
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
    
if spike_sort:
    rec_dict = {}
    
    probe_names = np.unique(df.probe)

    for probe_name in probe_names:
        print(f"Spike sorting: {probe_name}")
        df_probe = df.query(f"probe == '{probe_name}'")
        
        for index, row in df_probe.iterrows():
            trunc_bit = row["trunc_bit"]
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr = si.scale(rec_zarr, gain=row["lsb_value"], dtype=dtype)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            rec_zarr_cmr = si.common_reference(rec_zarr_f, reference="local")
            
            rec_dict[f"bit{trunc_bit}"] = rec_zarr_cmr

        compute_spikesorting = False
        if "sort_path" not in df.columns:
            compute_spikesorting = True
        elif np.any(df["sort_path"].isna()):
            compute_spikesorting = True
                
        sorting_output_folder_npz = trunc_folder / f"sorting_npz_{probe_name}"
        sorting_output_folder_npz.mkdir(exist_ok=True)
        
        if compute_spikesorting:
            sorting_output_folder = trunc_folder / f"sorting_outputs_{probe_name}"
            sorting_outputs = si.run_sorters(sorter_list, rec_dict, 
                                            working_folder=sorting_output_folder,
                                            mode_if_folder_exists="keep", engine="loop", 
                                            sorter_params=sorter_params, verbose=True)
            
            

            for index, row in df_probe.iterrows():
                rec_name = f"bit{row.trunc_bit}"
                sort_name = sorter_list[0]
                sort = sorting_outputs[(rec_name, sort_name)]
                
                print(f"Saving sorting output for trunc bit: {rec_name}")
                # save sorting
                
                sort_path = sorting_output_folder_npz / f"{rec_name}_{sort_name}"
                sort_saved = sort.save(folder=sort_path)
                df.at[index, "sort_path"] = str(sort_path.absolute())
            df.to_csv(benchmark_file, index=False)
            shutil.rmtree(sorting_output_folder)
        else:
            print(f"Spike sorting for {probe_name} already run")

            
if autocuration:
    assert "sort_path" in df.columns, "Run spike sorting before!!!"
    
    # auto curation
    isi_viol_threshold = 0.5
    amp_cutoff_threshold = 0.1
    presence_ratio_threshold = 0.95
    
    auto_curation_query = (f"isi_violations_ratio < {isi_viol_threshold} and "
                            f"amplitude_cutoff < {amp_cutoff_threshold} and "
                            f"presence_ratio > {presence_ratio_threshold}")
    
    # compute waveforms
    probe_names = np.unique(df.probe)
    for probe_name in probe_names:
        sorting_output_folder_npz = trunc_folder / f"sorting_npz_{probe_name}"
        sorting_output_folder_npz.mkdir(exist_ok=True)

        waveform_folder = trunc_folder / f"waveforms_{probe_name}"
        waveform_folder.mkdir(exist_ok=True)
        
        print(f"Extracting waveforms: {probe_name}")
        df_probe = df.query(f"probe == '{probe_name}'")
        
        for index, row in df_probe.iterrows():
            rec_name = f"bit{row.trunc_bit}"
            sort_name = sorter_list[0]
            
            trunc_bit = row["trunc_bit"]
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr = si.scale(rec_zarr, gain=row["lsb_value"], dtype=dtype)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            rec_zarr_cmr = si.common_reference(rec_zarr_f, reference="local")
            rec = rec_zarr_cmr
            
            sort = si.load_extractor(row.sort_path)
            n_units_original = len(sort.unit_ids)
            sort = sort.remove_empty_units()
            n_units_removed = len(sort.unit_ids)
            print(f"Removed {n_units_original - n_units_removed} empty units")

            sort_curated_path = sorting_output_folder_npz / f"{rec_name}_{sort_name}_curated"
            
            if sort_curated_path.is_dir():
                print(f"Sorting output for {rec_name} already curated")
                sort_curated = si.load_extractor(sort_curated_path)
            else:
                wf_path = waveform_folder / f"{rec_name}_{sort_name}"
                we = si.extract_waveforms(rec, sort, folder=wf_path,
                                        load_if_exists=True, **job_kwargs)
                qm = si.compute_quality_metrics(we)
                units_to_keeps = qm.query(auto_curation_query).index.values
                del we
                sort_curated = sort.select_units(units_to_keeps)
                sort_saved = sort_curated.save(folder=sort_curated_path)
                shutil.rmtree(wf_path)

            df.at[index, "sort_curated_path"] = str(sort_curated_path.absolute())
            
            print(f"{rec_name}: \n\tbefore curation - {sort}\n\tafter curation - {sort_curated}")
            # delete waveforms folder
            
        shutil.rmtree(waveform_folder)

    df.to_csv(benchmark_file, index=False)
