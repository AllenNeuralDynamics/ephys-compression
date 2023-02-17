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

from numcodecs import Blosc
from wavpack_numcodecs import WavPack

sys.path.append("..")

from utils import append_to_csv, is_entry, trunc_filter, benchmark_lossy_compression


print(f"spikeinterface: {si.__version__}")

remove_folders = False
overwrite_compression = False
compute_wfs = True
spike_sort = False
overwrite_study = False
    
sorter_list = ["kilosort2_5"]
sorter_params = {"kilosort2_5": {"n_jobs_bin": 10, "total_memory": "4G"}}
    
print(f"Remove folders: {remove_folders} -- Overwrite compression: {overwrite_compression} -- "
      f"Compute GT wfs: {compute_wfs} -- Spike sort: {spike_sort} -- Overwrite study: {overwrite_study}")

if spike_sort:
    print(sorter_list)
    print(sorter_params)

data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression" / "lossy" / "gt"
tmp_folder.mkdir(exist_ok=True, parents=True)

# NP1.0 and NP2.0
rec_files = ["/home/alessio/Documents/codes/allen/compression/ephys-compression/data/mearec/mearec_NP1.h5", 
             "/home/alessio/Documents/codes/allen/compression/ephys-compression/data/mearec/mearec_NP2.h5"]

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

benchmark_file = data_folder / "results" / "benchmark-lossy-gt.csv"

if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
    print(f"Entries in results: {len(df)}")
else:
    df = None

subset_columns=["strategy", "factor", "probe"]

gt_dict = {}
for rec_file in rec_files:
    rec_file = Path(rec_file)
    try:
        if tmp_folder.is_dir():
            if overwrite_compression:
                shutil.rmtree(tmp_folder)
        tmp_folder.mkdir(parents=True, exist_ok=True)
    except:
        print("Couldn't remove tmp folder")

    print(f"\n\n\nBenchmarking {rec_file.name}\n\n\n")
    t_start_all = time.perf_counter()

    rec, sort_gt= si.read_mearec(rec_file)
    print(rec)    
    
    if "np1" in rec_file.name:
        probe_name = "Neuropixels1.0"
    else:
        probe_name = "Neuropixels2.0"

    gt_dict[probe_name] = {}        
    gt_dict[probe_name]["rec_gt"] = rec
    gt_dict[probe_name]["sort_gt"] = sort_gt
     
    dur = rec.get_num_samples() / rec.get_sampling_frequency()
    dtype = rec.get_dtype()
    gain = rec.get_channel_gains()[0]
    
    num_channels = rec.get_num_channels()
    fs = rec.get_sampling_frequency()
    
    time_range_rmse = [15, 20]

    rec_to_compress = si.correct_lsb(rec)
    
    full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

    zarr_root = f"{rec_file.stem}"

    for strategy in strategies:
        print(f"Benchmarking {strategy}: {factors[strategy]}")
        for factor in factors[strategy]:
            print(f"Compression factor {factor}")
            entry_data = {"probe": probe_name, "strategy": strategy, "factor": factor}

            if not is_entry(benchmark_file, entry_data):

                zarr_path = tmp_folder / "zarr" / f"{zarr_root}_{strategy}_{factor}.zarr"

                if zarr_path.is_dir():
                    shutil.rmtree(zarr_path)
                
                if strategy == "bit_truncation":
                    filters = trunc_filter(factor, rec.get_dtype())
                    compressor = zarr_compressor
                else:
                    filters = None
                    compressor = WavPack(level=compression_level, bps=factor)


                cr, cspeed_xrt, elapsed_time, rmse = benchmark_lossy_compression(rec_to_compress, compressor, zarr_path, 
                                                                          filters=filters, time_range=time_range_rmse, 
                                                                          **job_kwargs)

                new_data = {"probe": probe_name, "rec_gt": str(rec_file.absolute()), "strategy": strategy, 
                            "factor": factor, "CR": cr, "Cspeed": elapsed_time, 
                            "cspeed_xrt": cspeed_xrt, "rmse": rmse, "rec_zarr_path": str(zarr_path.absolute())}
                append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)

                print(f"Elapsed time {strategy}-{factor}: {elapsed_time}s - CR: {cr} - rmse: {rmse}")
            else:
                print(f"{strategy} factor {factor} already computed")
    
df = pd.read_csv(benchmark_file, index_col=False)
print(f"Entries in results: {len(df)}")

if compute_wfs:
    template_metrics = si.get_template_metric_names()
    benchmark_wfs_file = data_folder / "results" / "benchmark-lossy-gt-wfs.csv"
    dist_interval = 30
    ndists = 4
    target_distances = [i*dist_interval for i in range(ndists)]
    seed = 2308
    ms_after = 5
    # first compute GT waveforms
    probes = np.unique(df.probe)
    df_templates = None
    if benchmark_wfs_file.is_file():
        df_templates = pd.read_csv(benchmark_wfs_file, index_col=False)
        print(f"Entries in waveform results: {len(df_templates)}")
        
    for probe in probes:
        print(f"GT waveforms for {probe}")
        df_probe = df.query(f"probe == '{probe}'")
        rec_gt = gt_dict[probe]["rec_gt"]
        sort_gt = gt_dict[probe]["sort_gt"]
        rec_gt_f = si.bandpass_filter(rec_gt)
        
        we_gt_path = tmp_folder / f"wf_gt_{probe}"
        we_gt = si.extract_waveforms(rec_gt_f, sort_gt, folder=we_gt_path, 
                                     load_if_exists=True, ms_after=ms_after,
                                     seed=seed, **job_kwargs)
        # find channels for each "GT" unit
        extremum_channels = si.get_template_extremum_channel(we_gt)
        rec_locs = rec_gt.get_channel_locations()

        sparsity = {}
        for unit, main_ch in extremum_channels.items():
            main_ch_idx = rec_gt.id_to_index(main_ch)
            
            # compute distances
            main_loc = rec_locs[main_ch_idx]
            distances = np.array([np.linalg.norm(loc - main_loc) for loc in rec_locs])
            distances_sort_idxs = np.argsort(distances)
            distances_sorted = distances[distances_sort_idxs]
            dist_idxs = np.searchsorted(distances_sorted, target_distances)
            selected_channel_idxs = distances_sort_idxs[dist_idxs]
            sparsity[unit] = rec_gt.channel_ids[selected_channel_idxs]

        print(f"Calculating GT template metrics")
        df_tm = si.calculate_template_metrics(we_gt, upsample=10,
                                                sparsity=sparsity)
        df_tm["probe"] = [probe] * len(df_tm)
        if sparsity is None:
            df_tm["unit_id"] = df_tm.index
            df_tm["distance"] = [0] * len(df_tm)
        else:
            df_tm["unit_id"] = df_tm.index.to_frame()["unit_id"].values
            df_tm["channel_id"] = df_tm.index.to_frame()["channel_id"].values

            # add channel distance
            for unit_id in np.unique(df_tm.unit_id):
                if isinstance(unit_id, str):
                    tm_unit = df_tm.query(f"unit_id == '{unit_id}'")
                else:
                    tm_unit = df_tm.query(f"unit_id == {unit_id}")
                    
                loc_main = rec_gt.get_channel_locations(channel_ids=[extremum_channels[unit_id]])[0]
                for index, row in tm_unit.iterrows():
                    loc = rec_gt.get_channel_locations(channel_ids=[row["channel_id"]])[0]
                    distance = np.linalg.norm(loc - loc_main)
                    # round distance to dist interval
                    df_tm.at[index, "distance"] = int(dist_interval * np.round(distance / dist_interval))
                    
        for metric in template_metrics:
            df_tm[f"{metric}_gt"] = df_tm[metric]
            del df_tm[metric]
            
        print("Distances", np.unique(df_tm["distance"]))
        
        # now with lossy
        for index, row in df_probe.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            zarr_path = row["rec_zarr_path"]
            rec_name = f"{strategy}_{factor}"
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            
            compute_metrics = False
            if df_templates is None:
                compute_metrics = True
            else:
                df_templates_probe = df_templates.query(f"probe == '{probe}'")
                test_metric = template_metrics[0]
                if np.all(np.isnan(df_templates_probe[f"{test_metric}_{rec_name}"].values)):
                    compute_metrics = True
            
            we_lossy_path = tmp_folder / f"wf_lossy_{probe}" / rec_name
            if compute_metrics:
                # compute waveforms
                we_lossy = si.extract_waveforms(rec_zarr_f, sort_gt, folder=we_lossy_path, 
                                                load_if_exists=True, ms_after=ms_after,
                                                seed=seed, **job_kwargs)
                # compute features
                print(f"Calculating template metrics for {strategy}-{factor}")
                df_tm_lossy = si.calculate_template_metrics(we_lossy, upsample=10,
                                                            sparsity=sparsity)
                df_tm_lossy["probe"] = [probe] * len(df_tm_lossy)
                
                if sparsity is None:
                    df_tm_lossy["unit_id"] = df_tm_lossy.index
                    df_tm_lossy["distance"] = [0] * len(df_tm_lossy)
                else:
                    df_tm_lossy["unit_id"] = df_tm_lossy.index.to_frame()["unit_id"].values
                    df_tm_lossy["channel_id"] = df_tm_lossy.index.to_frame()["channel_id"].values

                    # add channel distance
                    for unit_id in np.unique(df_tm_lossy.unit_id):
                        if isinstance(unit_id, str):
                            tm_unit = df_tm_lossy.query(f"unit_id == '{unit_id}'")
                        else:
                            tm_unit = df_tm_lossy.query(f"unit_id == {unit_id}")
                            
                        loc_main = rec_gt.get_channel_locations(channel_ids=[extremum_channels[unit_id]])[0]
                        for index, row in tm_unit.iterrows():
                            loc = rec_gt.get_channel_locations(channel_ids=[row["channel_id"]])[0]
                            distance = np.linalg.norm(loc - loc_main)
                            df_tm_lossy.at[index, "distance"] = int(dist_interval * np.round(distance / dist_interval))
                
                for metric in template_metrics:
                    df_tm[f"{metric}_{rec_name}"] = df_tm_lossy[metric]
                
                # cleanup
                del we_lossy
                shutil.rmtree(we_lossy_path)
                
                if we_lossy_path.parent.is_dir():
                    shutil.rmtree(we_lossy_path.parent)

        if df_templates is None:
            df_templates = df_tm
        else:
            df_templates = pd.concat([df_templates, df_tm])
            
        # update csv
        df_templates.to_csv(benchmark_wfs_file, index=False)
    

if spike_sort:
    for probe_name in np.unique(df.probe):
        print(f"Running study for {probe_name}")
        df_probe = df.query(f"probe == '{probe_name}'")
        gt_study_dict = {}
        study_folder = tmp_folder / f"study_{probe_name}_{sorter_list[0]}"
        sort_gt = gt_dict[probe_name]["sort_gt"]

        for index, row in df_probe.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            zarr_path = row["rec_zarr_path"]
            
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = si.bandpass_filter(rec_zarr)
            
            gt_study_dict[f"{strategy}_{factor}"] = (rec_zarr_f, sort_gt)

        print(f"GT study recordings: {list(gt_study_dict.keys())}")
        
        if overwrite_study:
            if study_folder.is_dir():
                shutil.rmtree(study_folder)
                
        if not study_folder.is_dir():
            print("Creating study")
            study = si.GroundTruthStudy.create(study_folder, gt_study_dict, **job_kwargs)
        else:
            study = si.GroundTruthStudy(study_folder)
        
        # count npz
        sortings_dir = study.study_folder / "sortings"
        if sortings_dir.is_dir():
            computed_sortings = len([f for f in sortings_dir.iterdir() if f.suffix == ".npz"])
        else:
            computed_sortings = 0
        
        if computed_sortings < len(study.rec_names) * len(sorter_list):
            study.run_sorters(sorter_list, mode_if_folder_exists="keep", verbose=True, 
                              sorter_params=sorter_params, remove_sorter_folders=True)   
        compute_comparisons = False
        if "avg_accuracy" not in df_probe.columns:
            compute_comparisons = True
        elif np.any(np.isnan(df_probe["avg_accuracy"])):
            compute_comparisons = True
            
        if compute_comparisons:
            print("Running comparisons")
            study.run_comparisons(exhaustive_gt=True, verbose=False)
            dataframes = study.aggregate_dataframes()
            perf_by_unit = dataframes["perf_by_unit"]
            perf_columns = ['accuracy', 'recall', 'precision', 'false_discovery_rate', 'miss_rate']
            counts = dataframes["count_units"]
            count_columns = ['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant', 'num_overmerged', 
                            'num_false_positive', 'num_bad']

            for rec_name in study.rec_names:
                rec_split = rec_name.split("_")
                factor = rec_split[-1]
                strategy = "_".join(rec_split[:-1])
                index = df.query(f"probe == '{probe_name}' and factor == {factor} and strategy == '{strategy}'").index[0]
                perf = perf_by_unit.query(f"rec_name == '{rec_name}'")
                for metric in perf_columns:
                    avg = perf[metric].values.mean()
                    df.at[index, f"avg_{metric}"] = avg
                cnt = counts.query(f"rec_name == '{rec_name}'")
                for count_col in count_columns:
                    df.at[index, count_col] = int(cnt[count_col])
        else:
            print(f"Comparisons already computed for {probe_name}")

        # update csv
        df.to_csv(benchmark_file, index=False)
        
