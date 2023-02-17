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

from wavpack_numcodecs import WavPack
from utils import append_to_csv, is_entry, trunc_filter, benchmark_compression


print(f"spikeinterface: {si.__version__}")

remove_folders = False
overwrite = False
spike_sort = False
autocuration = True

sorter_list = ["kilosort2_5"]
sorter_params = {"kilosort2_5": {"n_jobs_bin": 10, "total_memory": "2G"}}
    
    
print(f"Remove folders: {remove_folders} -- Overwrite: {overwrite} -- Spike sort: {spike_sort}")
if spike_sort:
    print(sorter_list)
    print(sorter_params)


data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression" / "lossy" / "nogt"
if tmp_folder.is_dir():
    if overwrite:
        shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True, parents=True)

# NP1.0 and NP2.0
rec_files = ["/home/alessio/Documents/data/allen/npix-open-ephys/595262_2022-02-21_15-18-07/", 
             "/home/alessio/Documents/data/allen/npix-open-ephys/618382_2022-03-31_14-27-03/"]

n_jobs = 10
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

strategies = ["bit_truncation", "wavpack"]

# define options for bit truncation
clevel = 9
zarr_compressor = Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)

# define wavpack options
compression_level = 3

factors = {"bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
           "wavpack": [6, 5, 4, 3.5, 3, 2.5, 2, 0]}

benchmark_file = data_folder / "results" / "benchmark-lossy-nogt.csv"

if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
    print(f"Entries in results: {len(df)}")
else:
    df = None

subset_columns=["strategy", "factor", "probe"]

for rec_file in rec_files:
    rec_file = Path(rec_file)
    try:
        if tmp_folder.is_dir():
            if overwrite:
                shutil.rmtree(tmp_folder)
        tmp_folder.mkdir(parents=True, exist_ok=True)
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
    
    rec_to_compress = None

    num_channels = rec.get_num_channels()
    fs = rec.get_sampling_frequency()
    
    time_range_rmse = [15, 20]
    full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

    zarr_root = f"{rec_file.stem}"

    for strategy in strategies:
        print(f"Benchmarking {strategy}: {factors[strategy]}")
        for factor in factors[strategy]:
            print(f"Compression factor {factor}")
            entry_data = {"probe": probe_name, "strategy": strategy, "factor": factor}

            if not is_entry(benchmark_file, entry_data):
                
                if rec_to_compress is None:
                    # compute rec to compress if needed
                    rec_to_compress = si.correct_lsb(rec)

                zarr_path = tmp_folder / "zarr" / f"{zarr_root}_{probe_name}_{strategy}_{factor}.zarr"

                if zarr_path.is_dir():
                    shutil.rmtree(zarr_path)
                
                if strategy == "bit_truncation":
                    filters = trunc_filter(factor, rec.get_dtype())
                    compressor = zarr_compressor
                else:
                    filters = None
                    compressor = WavPack(level=compression_level, bps=factor)

                cr, xRT, elapsed_time, rmse = benchmark_compression(rec_to_compress, compressor, zarr_path, 
                                                                    filters=filters, time_range=time_range_rmse, 
                                                                    **job_kwargs)

                new_data = {"probe": probe_name, "rec_exp": str(rec_file.absolute()), "strategy": strategy, 
                            "factor": factor, "CR": cr, "Cspeed": elapsed_time, 
                            "xRT": xRT, "rmse": rmse, "rec_zarr_path": str(zarr_path.absolute())}
                append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)

                print(f"Elapsed time {strategy}-{factor}: {elapsed_time}s - CR: {cr} - rmse: {rmse}")
            else:
                print(f"{strategy} factor {factor} already computed")
    
df = pd.read_csv(benchmark_file, index_col=False)
print(f"Entries in results: {len(df)}")
    
if spike_sort:
    rec_dict = {}
    probe_names = np.unique(df.probe)

    for probe_name in probe_names:
        print(f"\n\n\nSpike sorting: {probe_name}\n\n\n")
        df_probe = df.query(f"probe == '{probe_name}'")
        
        raw_sorting_output_folder = tmp_folder / f"sorting_outputs_{probe_name}"
        sorting_output_folder_npz = tmp_folder / f"sorting_npz_{probe_name}"
        raw_sorting_output_folder.mkdir(exist_ok=True)
        sorting_output_folder_npz.mkdir(exist_ok=True)
        
        for index, row in df_probe.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            rec_zarr_cmr = si.common_reference(rec_zarr_f, reference="local")
            rec_name = f"{strategy}_{factor}"

            sorting_output_folder = raw_sorting_output_folder / rec_name
            sort_path = sorting_output_folder_npz / rec_name
            
            if not sort_path.is_dir():
                print(f"\n\nSorting {rec_name}\n\n")
                sorting = si.run_sorter(sorter_list[0], rec_zarr_cmr, output_folder=sorting_output_folder,
                                        **sorter_params["kilosort2_5"], verbose=True)
                sort_path = sorting_output_folder_npz / rec_name
                sort_saved = sorting.save(folder=sort_path)
                # cleanup
                shutil.rmtree(sorting_output_folder)
            else:
                print(f"\n\n{rec_name} already sorted")
                sort_saved = si.load_extractor(sort_path)
            
            selected_units = sort_saved.unit_ids[sort_saved.get_property('KSLabel')=="good"]
            sort_good = sort_saved.select_units(unit_ids=selected_units)
            print(f"{rec_name}: \ntn units - {len(sort_saved.unit_ids)}\n\tn ks good - {len(sort_good.unit_ids)}")
            
            df.at[index, "sort_path"] = str(sort_path.absolute())
            df.at[index, "n_raw_units"] = len(sort_saved.unit_ids)
            df.at[index, "n_ks_good_units"] = len(sort_good.unit_ids)

        df.to_csv(benchmark_file, index=False)
        # cleanup
        shutil.rmtree(raw_sorting_output_folder)

            
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
        print(f"\n\n\nCurating: {probe_name}\n\n\n")
        df_probe = df.query(f"probe == '{probe_name}'")
        
        waveform_folder = tmp_folder / f"waveforms_{probe_name}"
        sorting_output_folder_npz = tmp_folder / f"sorting_npz_{probe_name}"
        waveform_folder.mkdir(exist_ok=True)
        sorting_output_folder_npz.mkdir(exist_ok=True)
        
        df_probe = df.query(f"probe == '{probe_name}'")
        for index, row in df_probe.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            rec_name = f"{strategy}_{factor}"
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            rec_zarr_cmr = si.common_reference(rec_zarr_f, reference="local")
            
            sort = si.load_extractor(row.sort_path)
            n_units_original = len(sort.unit_ids)
            sort = sort.remove_empty_units()
            n_units_removed = len(sort.unit_ids)
            print(f"Removed {n_units_original - n_units_removed} empty units")

            sort_curated_path = sorting_output_folder_npz / f"{rec_name}_curated"
            
            if sort_curated_path.is_dir():
                print(f"Sorting output for {rec_name} already curated")
                sort_curated = si.load_extractor(sort_curated_path)
            else:
                wf_path = waveform_folder / rec_name
                we = si.extract_waveforms(rec_zarr_cmr, sort, folder=wf_path,
                                          load_if_exists=True, **job_kwargs)
                qm = si.compute_quality_metrics(we)
                units_to_keeps = qm.query(auto_curation_query).index.values
                del we
                sort_curated = sort.select_units(units_to_keeps)
                sort_saved = sort_curated.save(folder=sort_curated_path)
                shutil.rmtree(wf_path)

            df.at[index, "sort_curated_path"] = str(sort_curated_path.absolute())
            df.at[index, "n_curated_good_units"] = len(sort_curated.unit_ids)
            df.at[index, "n_curated_bad_units"] = len(sort.unit_ids) - len(sort_curated.unit_ids)
            
            print(f"{rec_name}: \n\tbefore curation - {len(sort.unit_ids)}\n\tafter curation - {len(sort_curated.unit_ids)}")

        # delete waveforms folder
        shutil.rmtree(waveform_folder)

    df.to_csv(benchmark_file, index=False)
