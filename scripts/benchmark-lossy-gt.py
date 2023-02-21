"""
Benchmark lossy compression strategies on simulated data.

The script expects a CodeOcean file organization

- code
- data
- results

The script is run from the "code" folder and expect the "aind-ephys-compression-benchmark-data" bucket to be attached 
to the data folder.
"""

import time
import shutil
import sys

from pathlib import Path
import numpy as np
import pandas as pd

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.comparison as sc



from numcodecs import Blosc
from wavpack_numcodecs import WavPack

# add utils to path
this_folder = Path(__file__).parent
sys.path.append(str(this_folder.parent))
from utils import append_to_csv, is_entry, trunc_filter, benchmark_lossy_compression

data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")

job_kwargs = dict(n_jobs=16, chunk_duration="1s", progress_bar=True)
ks25_sorter_params = job_kwargs
time_range_rmse = [15, 20]

tmp_folder = scratch_folder / "tmp"
tmp_folder.mkdir(exist_ok=True, parents=True)

# COMPRESSION PARAMS #
strategies = ["bit_truncation", "wavpack"]

# define options for bit truncation
zarr_clevel = 9
zarr_compressor = Blosc(cname='zstd', clevel=zarr_clevel, shuffle=Blosc.BITSHUFFLE)

# define wavpack options
wv_level = 3
factors = {"bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
           "wavpack": [0, 6, 5, 4, 3.5, 3, 2.25]}

# TEMPLATE METRICS PARAMS #
dist_interval = 30
ndists = 4
target_distances = [i*dist_interval for i in range(ndists)]
seed = 2308
ms_after = 5

subset_columns=["strategy", "factor", "probe"]

all_dsets = ["NP1", "NP2"]

if __name__ == "__main__":

    if len(sys.argv) == 2:
        if sys.argv[1] == "all":
            dsets = all_dsets
        else:
            dsets = [sys.argv[1]]
    else:
        dsets = all_dsets

    ephys_benchmark_folders = [p for p in data_folder.iterdir() if p.is_dir() and "compression-benchmark" in p.name]
    if len(ephys_benchmark_folders) != 1:
        raise Exception("Can't find attached compression benchamrk data bucket")
    ephys_benchmark_folder = ephys_benchmark_folders[0]
    print(f"Benchmark data folder: {ephys_benchmark_folder}")

    print(f"spikeinterface version: {si.__version__}")

    df = None

    gt_dict = {}
    for dset in all_dsets:
        benchmark_file = results_folder / f"benchmark-lossy-gt-{dset}.csv"

        rec_file = [p for p in (ephys_benchmark_folder / "mearec").iterdir() if p.suffix == ".h5" and dset in p.name][0]

        print(f"\n\n\nBenchmarking {rec_file.name}\n\n\n")
        t_start_all = time.perf_counter()

        rec, sort_gt= se.read_mearec(rec_file)
        print(rec)
        
        if dset == "NP1":
            probe_name = "Neuropixels1.0"
        else:
            probe_name = "Neuropixels2.0"

        gt_dict[dset] = {}
        gt_dict[dset]["rec_gt"] = rec
        gt_dict[dset]["sort_gt"] = sort_gt

        fs = rec.sampling_frequency
        dur = rec.get_total_duration()
        dtype = rec.get_dtype()
        gain = rec.get_channel_gains()[0]
        num_channels = rec.get_num_channels()

        rec_to_compress = spre.correct_lsb(rec, verbose=True)
        zarr_root = f"{rec_file.stem}"

        print("\n\nCOMPRESSION\n\n")
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
                        compressor = WavPack(level=wv_level, bps=factor)

                    rec_compressed, cr, cspeed_xrt, cspeed, rmse = \
                        benchmark_lossy_compression(rec_to_compress, compressor, zarr_path,
                                                    filters=filters, time_range_rmse=time_range_rmse,
                                                    **job_kwargs)

                    new_data = {"probe": probe_name, "rec_gt": str(rec_file.absolute()), "strategy": strategy,
                                "factor": factor, "CR": cr, "Cspeed": cspeed,
                                "cspeed_xrt": cspeed_xrt, "rmse": rmse, "rec_zarr_path": str(zarr_path.absolute())}
                    append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)

                    print(f"Elapsed time {strategy}-{factor}: cspeed xrt - {cspeed_xrt} - CR: {cr} - rmse: {rmse}")
                else:
                    print(f"{strategy} factor {factor} already computed")

        df = pd.read_csv(benchmark_file, index_col=False)
        print(f"Entries in results: {len(df)}")

        print("\nTEMPLATE METRICS\n\n")
        template_metrics = spost.get_template_metric_names()
        benchmark_wfs_file = results_folder / f"benchmark-lossy-gt-wfs-{dset}.csv"
        waveforms_folder = results_folder / f"waveforms-{dset}"
        waveforms_folder.mkdir(exist_ok=True, parents=True)

        print(f"GT waveforms for {dset}")
        rec_gt = gt_dict[dset]["rec_gt"]
        sort_gt = gt_dict[dset]["sort_gt"]
        rec_gt_f = spre.bandpass_filter(rec_gt)
        
        we_gt_path = waveforms_folder / "wf_gt"
        # cache sorting for disk persistence
        sort_gt_path = waveforms_folder / "sort_gt"
        sort_gt = sort_gt.save(folder=sort_gt_path)
        we_gt = si.extract_waveforms(rec_gt_f, sort_gt, folder=we_gt_path, 
                                     ms_after=ms_after, precompute_template=('average', 'std'),
                                     seed=seed, use_relative_path=True, **job_kwargs)
        # find channels for each "GT" unit
        extremum_channels = si.get_template_extremum_channel(we_gt)
        rec_locs = rec_gt.get_channel_locations()

        unit_id_to_channel_ids = {}
        for unit, main_ch in extremum_channels.items():
            main_ch_idx = rec_gt.id_to_index(main_ch)
            
            # compute distances
            main_loc = rec_locs[main_ch_idx]
            distances = np.array([np.linalg.norm(loc - main_loc) for loc in rec_locs])
            distances_sort_idxs = np.argsort(distances)
            distances_sorted = distances[distances_sort_idxs]
            dist_idxs = np.searchsorted(distances_sorted, target_distances)
            selected_channel_idxs = distances_sort_idxs[dist_idxs]
            unit_id_to_channel_ids[unit] = rec_gt.channel_ids[selected_channel_idxs]
        sparsity = si.ChannelSparsity.from_dict(dict(unit_ids=we_gt.unit_ids,
                                                     channel_ids=we_gt.channel_ids,
                                                     unit_id_to_channel_ids=unit_id_to_channel_ids))

        print(f"Calculating GT template metrics")
        df_tm = spost.compute_template_metrics(we_gt, upsampling_factor=10,
                                               sparsity=sparsity)
        df_tm["probe"] = [probe_name] * len(df_tm)
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
        
        # now with lossy
        for index, row in df.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            zarr_path = row["rec_zarr_path"]
            rec_name = f"{strategy}_{factor}"
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = spre.bandpass_filter(rec_zarr)

            print(f"Lossy waveforms for {strategy}-{factor}")
            we_lossy_path = waveforms_folder / f"wf_lossy_{strategy}_{factor}"
            # compute waveforms
            we_lossy = si.extract_waveforms(rec_zarr_f, sort_gt, folder=we_lossy_path,
                                            ms_after=ms_after, precompute_template=('average', 'std'),
                                            seed=seed, use_relative_path=True, **job_kwargs)
            # compute features
            print(f"Calculating template metrics for {strategy}-{factor}")
            df_tm_lossy = spost.compute_template_metrics(we_lossy, upsampling_factor=10,
                                                         sparsity=sparsity)
            df_tm_lossy["probe"] = [probe_name] * len(df_tm_lossy)
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
                df_tm[f"{metric}_{strategy}_{factor}"] = df_tm_lossy[metric]

            # cleanup
            we_lossy.delete_waveforms()
            del we_lossy

        we_gt.delete_waveforms()
        del we_gt

        # update csv
        df_tm.to_csv(benchmark_wfs_file, index=False)

        print(f"\nSPIKE SORTING\n\n")
        gt_study_dict = {}
        study_folder = tmp_folder / f"study_{dset}"
        sort_gt = gt_dict[dset]["sort_gt"]

        for index, row in df.iterrows():
            strategy = row["strategy"]
            factor = row["factor"]
            zarr_path = row["rec_zarr_path"]
            rec_zarr = si.read_zarr(zarr_path)
            rec_zarr_f = spre.bandpass_filter(rec_zarr)
            gt_study_dict[f"{strategy}_{factor}"] = (rec_zarr_f, sort_gt)

        print(f"GT study recordings: {list(gt_study_dict.keys())}")
        study = sc.GroundTruthStudy.create(study_folder, gt_study_dict, **job_kwargs)

        sorter_list = ["kilosort2_5"]
        sorter_params = dict(kilosort_2_5=ks25_sorter_params)
        study.run_sorters(sorter_list, mode_if_folder_exists="keep", verbose=True,
                          sorter_params=sorter_params, remove_sorter_folders=True)

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

        # update csv
        df.to_csv(benchmark_file, index=False)
