"""
Benchmark lossy compression strategies on experimental data.

The script expects a CodeOcean file organization

- code
- data
- results

The script is run from the "code" folder and expect the "aind-ephys-compression-benchmark-data" bucket to be attached 
to the data folder.

Different datasets (aind1, aind2, ibl, mindscope) can be run in parallel by passing them as an argument (or using the 
"App Panel").
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
import spikeinterface.curation as scur
import spikeinterface.comparison as sc
import spikeinterface.postprocessing as spost
import spikeinterface.preprocessing as spre
import spikeinterface.qualitymetrics as sqm
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
ks25_sorter_params = job_kwargs
time_range_rmse = [15, 20]


sorter = "kilosort2_5"
sorter_params = ks25_sorter_params

# we split AIND datasets in two sessions to parallelize computations
sessions = {
    "aind-np2-1": [
        "595262_2022-02-21_15-18-07_ProbeA",
        "602454_2022-03-22_16-30-03_ProbeB",
    ],
    "aind-np2-2": [
        "612962_2022-04-13_19-18-04_ProbeB",
        "618384_2022-04-14_15-11-00_ProbeB",
    ],
    "aind-np1": [
        "625749_2022-08-03_15-15-06_ProbeA",
        "634568_2022-08-05_15-59-46_ProbeA",
    ],
    "ibl-np1": [
        "CSHZAD026_2020-09-04_probe00",
        "SWC054_2020-10-05_probe00"
    ],
}

# auto curation
isi_viol_threshold = 0.5
amp_cutoff_threshold = 0.1
presence_ratio_threshold = 0.95

metric_names = ["isi_violation", "presence_ratio", "amplitude_cutoff"]
qm_params = {
    "presence_ratio": {"bin_duration_s": 60},
    "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0},
    "amplitude_cutoff": {
        "peak_sign": "neg",
        "num_histogram_bins": 100,
        "histogram_smoothing_value": 3,
        "amplitudes_bins_min_ratio": 5,
    },
}

auto_curation_query = (
    f"isi_violations_ratio < {isi_viol_threshold} and "
    f"amplitude_cutoff < {amp_cutoff_threshold} and "
    f"presence_ratio > {presence_ratio_threshold}"
)

all_dsets = ["aind-np2-1", "aind-np2-2", "ibl-np1", "aind-np1"]
all_strategies = ["bit_truncation", "wavpack"]
all_factors = {
    "bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
    "wavpack": [0, 6, 5, 4, 3.5, 3, 2.25],
}

# define options for bit truncation
zarr_clevel = 9
zarr_compressor = Blosc(cname="zstd", clevel=zarr_clevel, shuffle=Blosc.BITSHUFFLE)

# define wavpack options
level = 3

# define compress time range for short snippet saved in the "compressed_recordings" folder
compressed_recordings_folder = results_folder / "compressed_recordings"
compress_range = [28, 33]

# define match_score and comparison folder
match_score = 0.9
comparisons_folder = results_folder / "comparisons"
accuracies_folder = results_folder / "accuracies"
comparisons_folder.mkdir()
accuracies_folder.mkdir()


subset_columns = ["dset", "session", "strategy", "factor", "probe"]

if __name__ == "__main__":
    # check if json files in data
    json_files = [p for p in data_folder.iterdir() if p.suffix == ".json"]

    if len(sys.argv) == 5:
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
            factors = [0, float(sys.argv[3])]
        if int(sys.argv[4]) < 0:
            num_runs = 1
        else:
            num_runs = int(sys.argv[4])
    elif len(json_files) == 1:
        config_file = json_files[0]
        config = json.load(open(config_file, "r"))
        dsets = [config["dset"]]
        strategies = [config["strategy"]]
        factors = [0, config["factor"]]
        num_runs = config.get("num_runs", 1)
    else:
        dsets = all_dsets
        strategies = all_strategies
        factors = None
        num_runs = 2

    ephys_benchmark_folders = [p for p in data_folder.iterdir() if p.is_dir() and "compression-benchmark" in p.name]
    if len(ephys_benchmark_folders) != 1:
        raise Exception("Can't find attached compression benchamrk data bucket")
    ephys_benchmark_folder = ephys_benchmark_folders[0]
    print(f"Benchmark data folder: {ephys_benchmark_folder}")

    print(f"spikeinterface version: {si.__version__}")

    print(f"Running lossy benchmarks on:")
    print(f"\tDatasets: {dsets}")
    print(f"\tStrategies: {strategies}")
    print(f"\tFactors: {factors if factors is not None else 'all'}")
    print(f"\tNum runs: {num_runs}")

    tmp_folder = scratch_folder / "tmp"
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir()
    recordings_folder = tmp_folder

    sorting_outputs_folder = results_folder / "sortings"
    raw_sorting_outputs_folder = sorting_outputs_folder / "raw"
    curated_sorting_outputs_folder = sorting_outputs_folder / "curated"

    raw_sorting_outputs_folder.mkdir(parents=True)
    curated_sorting_outputs_folder.mkdir(parents=True)

    for dset in dsets:
        print(f"\nProcessing dataset {dset}")
        t_start_dset = time.perf_counter()

        if "aind-np2" in dset:
            probe_name = "Neuropixels2.0"
            dset_name = "aind-np2"
        elif "aind-np1" in dset:
            probe_name = "Neuropixels1.0"
            dset_name = "aind-np1"
        else:
            probe_name = "Neuropixels1.0"
            dset_name = dset

        for strategy in strategies:
            t_start_strategy = time.perf_counter()
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

                benchmark_file = results_folder / f"benchmark-lossy-exp-{dset}-{strategy}-{factor}.csv"

                if benchmark_file.is_file():
                    df = pd.read_csv(benchmark_file, index_col=False)
                else:
                    df = None

                for session in sessions[dset]:
                    t_start_session = time.perf_counter()
                    print(f"\tBenchmarking {session}")
                    rec = si.load_extractor(ephys_benchmark_folder / dset_name / session)
                    dur = rec.get_total_duration()
                    print(f"\tDuration {dur}s\n")
                    dtype = rec.get_dtype()
                    gain = rec.get_channel_gains()[0]

                    rec_to_compress = None
                    num_channels = rec.get_num_channels()
                    fs = rec.get_sampling_frequency()

                    entry_data = {
                        "probe": probe_name,
                        "strategy": strategy,
                        "factor": factor,
                        "dataset": dset_name,
                        "session": session,
                    }
                    rec_name = f"{dset_name}-{session}_{strategy}-{factor}"

                    if not is_entry(benchmark_file, entry_data):
                        print(f"\t\t{rec_name}")
                        if "ibl" not in dset_name:
                            if rec_to_compress is None:
                                rec_to_compress = spre.correct_lsb(rec)
                        else:
                            rec_to_compress = rec

                        zarr_path = recordings_folder / f"{rec_name}.zarr"

                        if zarr_path.is_dir():
                            shutil.rmtree(zarr_path)

                        if strategy == "bit_truncation":
                            filters = trunc_filter(factor, rec.get_dtype())
                            compressor = zarr_compressor
                        else:
                            filters = None
                            compressor = WavPack(level=3, bps=factor)

                        (rec_compressed, cr, cspeed_xrt, elapsed_time, rmse,) = benchmark_lossy_compression(
                            rec_to_compress,
                            compressor,
                            zarr_path,
                            filters=filters,
                            time_range_rmse=time_range_rmse,
                            **job_kwargs,
                        )
                        print(f"\t\t\tCompression: cspeed xrt - {cspeed_xrt} - CR: {cr} - rmse: {rmse}\n")

                        # save snippet for visualization
                        rec_slice = rec.frame_slice(
                            start_frame=int(compress_range[0] * fs), end_frame=int(compress_range[1] * fs)
                        )
                        zarr_path_slice = compressed_recordings_folder / f"{session}_{strategy}-{factor}.zarr"
                        if zarr_path_slice.is_dir():
                            shutil.rmtree(zarr_path_slice)
                        rec_compressed = rec_slice.save(
                            folder=zarr_path_slice,
                            format="zarr",
                            n_jobs=n_jobs,
                            chunk_duration="1s",
                            progress_bar=False,
                            compressor=compressor,
                            filters=filters,
                            verbose=False,
                        )

                        new_data = {
                            "dataset": dset_name,
                            "session": session,
                            "probe": probe_name,
                            "strategy": strategy,
                            "factor": factor,
                            "CR": cr,
                            "cspeed_xrt": cspeed_xrt,
                            "rmse": rmse,
                            "rec_zarr_path_slice": str(zarr_path_slice.relative_to(results_folder)),
                        }

                        sortings_raw = []
                        sortings_curated = []
                        for i in range(num_runs):
                            print(f"\t\tRunning spike sorting run {i + 1} / {num_runs}")
                            # run spike sorting
                            sorting_name = f"sorting_{rec_name}"
                            curated_sorting_name = f"{sorting_name}_curated"
                            if i > 0:
                                sorting_name += f"_{i}"
                                curated_sorting_name += f"_{i}"
                            tmp_sorting_output_folder = tmp_folder / sorting_name
                            raw_sorting_path = raw_sorting_outputs_folder / sorting_name
                            curated_sorting_path = curated_sorting_outputs_folder / curated_sorting_name

                            # basic pre-processing
                            rec_zarr = si.read_zarr(zarr_path)
                            rec_zarr_f = spre.bandpass_filter(rec_zarr)
                            rec_zarr_cmr = spre.common_reference(rec_zarr_f)

                            sorting = ss.run_sorter(
                                sorter,
                                rec_zarr_cmr,
                                output_folder=tmp_sorting_output_folder,
                                delete_output_folder=True,
                                **sorter_params,
                            )
                            sorting = sorting.remove_empty_units()
                            # remove duplicated spikes
                            sorting = scur.remove_redundant_units(
                                sorting,
                                duplicate_threshold=0.9,
                                align=False,
                                remove_strategy="max_spikes",
                            )
                            sorting = scur.remove_excess_spikes(sorting, recording=rec_zarr_cmr)
                            ks_good_unit_ids = sorting.unit_ids[sorting.get_property("KSLabel") == "good"]
                            sorting_good = sorting.select_units(unit_ids=ks_good_unit_ids)
                            sorting_saved = sorting.save(folder=raw_sorting_path)
                            # cleanup
                            sorting = sorting_saved

                            new_data["sorting_path"] = str(raw_sorting_path.relative_to(results_folder))
                            new_data["n_raw_units"] = len(sorting_saved.unit_ids)
                            new_data["n_ks_good_units"] = len(sorting_good.unit_ids)

                            print(
                                f"\t\t\tSpike sorting run {i}: num units - {len(sorting.unit_ids)} num KS good units "
                                f"- {len(sorting_good.unit_ids)}\n"
                            )

                            # run auto-curation
                            wf_path = tmp_folder / f"waveforms_raw_{dset_name}_{session}_{i}"
                            we = si.extract_waveforms(rec_zarr_cmr, sorting, folder=wf_path, **job_kwargs)
                            _ = spost.compute_spike_amplitudes(we, **job_kwargs)
                            qm = sqm.compute_quality_metrics(we, metric_names=metric_names)
                            units_to_keeps = qm.query(auto_curation_query).index.values

                            sorting_curated = sorting.select_units(units_to_keeps)
                            sorting_curated_saved = sorting_curated.save(folder=curated_sorting_path)

                            new_data["sorting_curated_path"] = str(curated_sorting_path.relative_to(results_folder))
                            new_data["n_curated_good_units"] = len(sorting_curated.unit_ids)
                            new_data["n_curated_bad_units"] = len(sorting.unit_ids) - len(sorting_curated.unit_ids)

                            print(
                                f"\t\t\tCuration: num units - {len(sorting.unit_ids)} num auto-curated units "
                                f"{len(sorting_curated.unit_ids)}\n"
                            )

                            append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)

                            print(f"\n\t\tSummary {rec_name}:\n")
                            print(f"\t\tCompression: cspeed xrt - {cspeed_xrt} - CR: {cr} - rmse: {rmse}\n")
                            print(
                                f"\t\tSpike sorting: num units - {len(sorting.unit_ids)} num KS good units - "
                                f"{len(sorting_good.unit_ids)}\n"
                            )
                            print(f"\t\tCuration: num auto-curated units {len(sorting_curated.unit_ids)}\n")
                            # clean up
                            shutil.rmtree(wf_path)
                            del we

                            if not save_recordings:
                                shutil.rmtree(zarr_path)

                t_stop_session = time.perf_counter()
                elapsed_session = np.round(t_stop_session - t_start_session)
                print(f"\n\t\tElapsed time session: {elapsed_session}s")

            df = pd.read_csv(benchmark_file, index_col=False)
            print(f"\n\tFinal # entries in results for {strategy}: {len(df)}")

            t_stop_strategy = time.perf_counter()
            elapsed_strategy = np.round(t_stop_strategy - t_start_strategy)
            print(f"\tElapsed time strategy: {elapsed_strategy}s")
        t_stop_dset = time.perf_counter()
        elapsed_dset = np.round(t_stop_dset - t_start_dset)
        print(f"\nElapsed time dataset: {elapsed_dset}s\n")

    # aggregate pandas dataframes into one
    benchmark_file = results_folder / f"benchmark-lossy-exp.csv"
    csv_files = [p for p in results_folder.iterdir() if p.suffix == ".csv"]
    print(f"Found {len(csv_files)} CSV files")
    df = None

    if len(csv_files) > 1:
        for csv_file in csv_files:
            print(f"Aggregating {csv_file.name}")
            df_single = pd.read_csv(csv_file, index_col=False)
            df = df_single if df is None else pd.concat((df, df_single))

        if df is not None:
            df.to_csv(benchmark_file, index=False)

        # remove single CSV files
        for csv_file in csv_files:
            csv_file.unlink()

    # compute spike sorting comparisons against lossless
    res_lossy = pd.read_csv(results_folder / "benchmark-lossy-exp.csv", index_col=False)
    sessions = np.unique(res_lossy.session)
    sortings_folder = raw_sorting_outputs_folder

    print("\n\nComputing and saving pairwise comparisons\n\n")
    for session in sessions:
        t_start_session = time.perf_counter()
        probe = np.unique(res_lossy.query(f"session == '{session}'").probe)[0]
        print(f"{session} - {probe}\n")

        # Load lossless sortings
        lossless_sorting_folders_wv = [
            p for p in sortings_folder.iterdir() if f"wavpack-0" in p.name and session in p.name
        ]
        if len(lossless_sorting_folders_wv) == 1:
            lossless_sorting_wv = si.load_extractor(lossless_sorting_folders_wv[0])
            print(f"Lossless WavPack: {lossless_sorting_wv}")
        else:
            lossless_sorting_wv = None
        lossless_sorting_folders_bt = [
            p for p in sortings_folder.iterdir() if f"bit_truncation-0" in p.name and session in p.name
        ]
        if len(lossless_sorting_folders_bt) == 1:
            lossless_sorting_bt = si.load_extractor(lossless_sorting_folders_bt[0])
            print(f"Lossless Bit Truncation: {lossless_sorting_wv}")
        else:
            lossless_sorting_bt = None

        for strategy in strategies:
            if factors is None:
                factors_to_run = all_factors[strategy]
            else:
                factors_to_run = factors
            if strategy == "wavpack":
                lossless_sorting = lossless_sorting_wv
                lossless_other = lossless_sorting_bt
            else:
                lossless_sorting = lossless_sorting_bt
                lossless_other = lossless_sorting_wv

            print(f"Strategy {strategy}\n")
            tested_sortings = []
            for test_factor in factors:
                print(f"\tComparing factor {test_factor}")
                if test_factor == 0:
                    if lossless_other is not None:
                        tested_sorting = lossless_other
                    else:
                        print(f"Cannot compute comparison for factor 0: other strategy is not available.")
                        continue
                else:
                    tested_sorting_folder = [
                        p
                        for p in sortings_folder.iterdir()
                        if f"{strategy}-{test_factor}" in p.name and session in p.name
                    ][0]
                    tested_sorting = si.load_extractor(tested_sorting_folder)
                print(f"\tTested: {tested_sorting}")
                tested_sortings.append(tested_sorting)
                mcmp = sc.compare_multiple_sorters(
                    [lossless_sorting, tested_sorting],
                    name_list=["original", f"{strategy}-{test_factor}"],
                    match_score=match_score,
                )
                mcmp.save_to_folder(comparisons_folder / f"{session}-{strategy}-{str(test_factor)}")
                sorting_agr = mcmp.get_agreement_sorting(minimum_agreement_count=2)
                cmp_single = mcmp.comparisons[list(mcmp.comparisons.keys())[0]]
                acc = np.diag(cmp_single.get_ordered_agreement_scores())
                np.save(accuracies_folder / f"{session}-{strategy}-{str(test_factor)}.npy", acc)
                print(f"\tUnits in agreement: {len(sorting_agr.unit_ids)}")
        t_stop_session = time.perf_counter()
        print(f"Elapsed time {session}: {np.round(t_stop_session - t_start_session, 2)} s")

    # final
    shutil.rmtree(tmp_folder)
