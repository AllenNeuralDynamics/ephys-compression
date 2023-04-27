"""
Benchmark lossy compression strategies on simulated data.

The script expects a CodeOcean file organization

- code
- data
- results

The script is run from the "code" folder and expect the "aind-ephys-compression-benchmark-data" bucket to be attached 
to the data folder.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from numcodecs import Blosc
from wavpack_numcodecs import WavPack

# add utils to path
this_folder = Path(__file__).parent
sys.path.append(str(this_folder.parent))
from utils import append_to_csv, benchmark_lossy_compression, is_entry, trunc_filter

data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")

n_jobs = None
job_kwargs = dict(
    n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
    chunk_duration="1s",
    progress_bar=False,
    verbose=False,
)
sorter_name = "kilosort2_5"
sorter_params = job_kwargs
time_range_rmse = [15, 20]

tmp_folder = scratch_folder / "tmp"
tmp_folder.mkdir(exist_ok=True, parents=True)

# COMPRESSION PARAMS #
all_dsets = ["NP1", "NP2"]
all_strategies = ["bit_truncation", "wavpack"]
all_factors = {
    "bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
    "wavpack": [0, 6, 5, 4, 3.5, 3, 2.25],
}

# define options for bit truncation
zarr_clevel = 9
zarr_compressor = Blosc(cname="zstd", clevel=zarr_clevel, shuffle=Blosc.BITSHUFFLE)

# define wavpack options
wv_level = 3


# TEMPLATE METRICS PARAMS #
dist_interval = 30
ndists = 4
target_distances = [i * dist_interval for i in range(ndists)]
seed = 2308
ms_after = 5

subset_columns = ["strategy", "factor", "probe"]


if __name__ == "__main__":
    # check if json files in data
    json_files = [p for p in data_folder.iterdir() if p.suffix == ".json"]

    if len(sys.argv) == 4:
        if sys.argv[1] == "all":
            dsets = all_dsets
        else:
            dset = str(sys.argv[1])
            assert dset in all_dsets, "Invalid dataset!"
            dsets = [dset]
        if sys.argv[2] == "all":
            strategies = all_strategies
        else:
            strategy = str(sys.argv[2])
            assert strategy in all_strategies, "Invalid strategy!"
            strategies = [strategy]
        if float(sys.argv[3]) < 0:
            factors = None
        else:
            factors = [float(sys.argv[3])]
    elif len(json_files) == 1:
        config_file = json_files[0]
        config = json.load(open(config_file, "r"))
        dsets = [config["dset"]]
        strategies = [config["strategy"]]
        factors = [config["factor"]]
    else:
        dsets = all_dsets
        strategies = all_strategies
        factors = None

    ephys_benchmark_folders = [p for p in data_folder.iterdir() if p.is_dir() and "compression-benchmark" in p.name]
    if len(ephys_benchmark_folders) != 1:
        raise Exception("Can't find attached compression benchamrk data bucket")
    ephys_benchmark_folder = ephys_benchmark_folders[0]
    print(f"Benchmark data folder: {ephys_benchmark_folder}")

    print(f"spikeinterface version: {si.__version__}")

    df = None
    print(f"\nUsing datasets: {dsets}")

    gt_dict = {}
    for dset in dsets:
        rec_file = [p for p in (ephys_benchmark_folder / "mearec").iterdir() if p.suffix == ".h5" and dset in p.name][0]

        print(f"\n\nBenchmarking {rec_file.name}\n")
        t_start_all = time.perf_counter()

        rec, sort_gt = se.read_mearec(rec_file)

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

        print(f"\tExtracting GT waveforms and metrics")
        rec_gt = gt_dict[dset]["rec_gt"]
        sort_gt = gt_dict[dset]["sort_gt"]
        rec_gt_f = spre.bandpass_filter(rec_gt)

        waveforms_gt_path = results_folder / f"gt-{dset}" / "waveforms"
        # cache sorting for disk persistence
        sorting_gt_path = results_folder / f"gt-{dset}" / "sorting"
        if sorting_gt_path.is_dir():
            sort_gt = si.load_extractor(sorting_gt_path)
        else:
            sort_gt = sort_gt.save(folder=sorting_gt_path)

        if waveforms_gt_path.is_dir():
            we_gt = si.load_waveforms(waveforms_gt_path)
        else:
            we_gt = si.extract_waveforms(
                rec_gt_f,
                sort_gt,
                folder=waveforms_gt_path,
                ms_after=ms_after,
                precompute_template=("average", "std"),
                seed=seed,
                use_relative_path=True,
                **job_kwargs,
            )
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
        sparsity = si.ChannelSparsity.from_dict(
            dict(
                unit_ids=we_gt.unit_ids,
                channel_ids=we_gt.channel_ids,
                unit_id_to_channel_ids=unit_id_to_channel_ids,
            )
        )

        print(f"\tComputing GT template metrics")
        template_metrics = spost.get_template_metric_names()
        df_tm = spost.compute_template_metrics(we_gt, upsampling_factor=10, sparsity=sparsity)
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
        we_gt.delete_waveforms()

        for strategy in strategies:
            waveforms_folder = results_folder / f"waveforms-{dset}-{strategy}"
            waveforms_folder.mkdir(exist_ok=True, parents=True)
            sortings_folder = results_folder / f"sortings-{dset}-{strategy}"
            sortings_folder.mkdir(exist_ok=True, parents=True)

            if factors is None:
                factors_to_run = all_factors[strategy]
            else:
                factors_to_run = factors
            print(f"\nBenchmarking {strategy}: {factors_to_run}\n")

            for factor in factors_to_run:
                if strategy == "bit_truncation":
                    factor = int(factor)
                else:
                    factor = float(factor)
                print(f"\n\tFactor {factor}\n")
                # assert factor in all_factors[strategy], f"Factor {factor} is invalid for startegy {strategy}"

                benchmark_file = results_folder / f"benchmark-lossy-sim-{dset}-{strategy}-{factor}.csv"
                entry_data = {
                    "probe": probe_name,
                    "strategy": strategy,
                    "factor": factor,
                }

                print("\n\tCOMPRESSION")
                # if not is_entry(benchmark_file, entry_data):
                zarr_path = tmp_folder / "zarr" / f"{zarr_root}_{strategy}_{factor}.zarr"

                if zarr_path.is_dir():
                    shutil.rmtree(zarr_path)

                if strategy == "bit_truncation":
                    filters = trunc_filter(factor, rec.get_dtype())
                    compressor = zarr_compressor
                else:
                    filters = None
                    compressor = WavPack(level=wv_level, bps=factor)

                (rec_compressed, cr, cspeed_xrt, cspeed, rmse,) = benchmark_lossy_compression(
                    rec_to_compress,
                    compressor,
                    zarr_path,
                    filters=filters,
                    time_range_rmse=time_range_rmse,
                    **job_kwargs,
                )

                new_data = {
                    "probe": probe_name,
                    "rec_gt": str(rec_file.absolute()),
                    "strategy": strategy,
                    "factor": factor,
                    "CR": cr,
                    "Cspeed": cspeed,
                    "cspeed_xrt": cspeed_xrt,
                    "rmse": rmse,
                    "rec_zarr_path": str(zarr_path.absolute()),
                }

                print(
                    f"\tCompression factor {factor}: elapsed time {cspeed}s: "
                    f"CR: {cr} - cspeed xrt - {cspeed_xrt} - rmse: {rmse}"
                )

                print(f"\n\tSPIKE SORTING")
                # TODO run one sorter at a time!
                sorting_output_folder = tmp_folder / f"sorting_{dset}-{strategy}-{factor}"

                rec_zarr = si.read_zarr(zarr_path)
                rec_zarr_f = spre.bandpass_filter(rec_zarr)

                sorter = "kilosort2_5"
                sort_ks = ss.run_sorter(
                    sorter_name,
                    recording=rec_zarr_f,
                    output_folder=sorting_output_folder,
                    delete_output_folder=True,
                    **sorter_params,
                )
                sort_ks = sort_ks.save(folder=sortings_folder / f"sorting_{strategy}_{factor}")

                print("\tRunning comparison")
                cmp = sc.compare_sorter_to_ground_truth(sort_gt, sort_ks, exhaustive_gt=True)

                perf_avg = cmp.get_performance(method="pooled_with_average", output="dict")
                counts = cmp.count_units_categories()
                new_data.update(perf_avg)
                new_data.update(counts.to_dict())

                append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)
                shutil.rmtree(sorting_output_folder)

                print("\n\tTEMPLATE METRICS")
                benchmark_waveforms_file = (
                    results_folder / f"benchmark-lossy-sim-waveforms-{dset}-{strategy}-{factor}.csv"
                )
                rec_name = f"{strategy}_{factor}"
                rec_zarr = si.read_zarr(zarr_path)
                rec_zarr_f = spre.bandpass_filter(rec_zarr)

                print(f"\tLossy waveforms for {strategy}-{factor}")
                we_lossy_path = waveforms_folder / f"wf_lossy_{strategy}_{factor}"
                # compute waveforms
                we_lossy = si.extract_waveforms(
                    rec_zarr_f,
                    sort_gt,
                    folder=we_lossy_path,
                    ms_after=ms_after,
                    precompute_template=("average", "std"),
                    seed=seed,
                    use_relative_path=True,
                    **job_kwargs,
                )
                # compute features
                print(f"\tComputing lossy template metrics")
                df_tm_lossy = spost.compute_template_metrics(we_lossy, upsampling_factor=10, sparsity=sparsity)
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

                df_tm_local = df_tm.copy()
                for metric in template_metrics:
                    df_tm_local[f"{metric}_{strategy}_{factor}"] = df_tm_lossy[metric]

                # cleanup
                we_lossy.delete_waveforms()
                # update csv
                print(f"Done with strategy: {strategy} - factor {factor}")
                # write waveforms csv for strategy
                df_tm_local.to_csv(benchmark_waveforms_file, index=False)

        print(f"Done with dataset: {dset}")

    # Aggregate results
    csv_sorting_files = [p for p in results_folder.iterdir() if p.suffix == ".csv" and "waveforms" not in p.name]
    # only aggregate if more than 1
    if len(csv_sorting_files) > 1:
        benchmark_file = results_folder / f"benchmark-lossy-sim.csv"
        print(f"Found {len(csv_sorting_files)} sorting results CSV files: aggregating results")
        df = None
        for sorting_csv_file in csv_sorting_files:
            print(f"Aggregating {sorting_csv_file.name}")
            df_single = pd.read_csv(sorting_csv_file, index_col=False)
            df = df_single if df is None else pd.concat((df, df_single))
            sorting_csv_file.unlink()
        df.to_csv(benchmark_file, index=False)

    # aggregate waveform results (do by probe then concat)
    on = ["probe", "unit_id", "channel_id", "distance"]
    for metric in template_metrics:
        on += [f"{metric}_gt"]

    df_probes = []
    for dset in dsets:
        csv_wfs_probe_files = [
            p for p in results_folder.iterdir() if p.suffix == ".csv" and "waveforms" in p.name and dset in p.name
        ]
        # only aggregate if more than 1
        if len(csv_wfs_probe_files) > 1:
            print(f"Found {len(csv_wfs_probe_files)} waveforms results CSV files for dset {dset}")
            df_wfs = None
            for i, wf_csv_file in enumerate(csv_wfs_probe_files):
                print(f"Aggregating {wf_csv_file.name}")
                df_single = pd.read_csv(wf_csv_file, index_col=False)
                df_wfs = df_single if df_wfs is None else df_wfs.merge(df_single, on=on)
                wf_csv_file.unlink()
            df_probes.append(df_wfs)

    if len(df_probes) > 0:
        benchmark_waveforms_file = results_folder / f"benchmark-lossy-sim-waveforms.csv"
        df_wfs_all = pd.concat(df_probes)
        df_wfs_all.to_csv(benchmark_waveforms_file, index=False)
