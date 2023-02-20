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

import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sys


import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as scur

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


sorter = "kilosort2_5"
sorter_params = ks25_sorter_params

# we split AIND datasets in two sessions to parallelize computations
sessions = {
    "aind-np2-1": ["595262_2022-02-21_15-18-07_ProbeA",
                   "602454_2022-03-22_16-30-03_ProbeB"],
    "aind-np2-2": ["612962_2022-04-13_19-18-04_ProbeB",
                   "618384_2022-04-14_15-11-00_ProbeB"],
    "aind-np1": ['613482_2022-06-16_17-49-19_ProbeA',
                 '625749_2022-08-03_15-15-63_ProbeA'],
    "ibl-np1": ["CSHZAD026_2020-09-04_probe00",
                "SWC054_2020-10-05_probe00"],
    # "mindscope-np1": ["754312389_probe756781559",
    #                   "829720705_probe832129157"]

}
all_dsets = ["aind-np2-1", "aind-np2-2", "ibl-np1", "aind-np1"] #"mindscope-np1"]


# auto curation
isi_viol_threshold = 0.5
amp_cutoff_threshold = 0.1
presence_ratio_threshold = 0.95

metric_names = ['isi_violation', 'presence_ratio', 'amplitude_cutoff']
qm_params = {
    'presence_ratio': {'bin_duration_s': 60},
    'isi_violation': {
        'isi_threshold_ms': 1.5, 'min_isi_ms': 0
    },
    'amplitude_cutoff': {
        'peak_sign': 'neg',
        'num_histogram_bins': 100,
        'histogram_smoothing_value': 3,
        'amplitudes_bins_min_ratio': 5
    }
}

auto_curation_query = (f"isi_violations_ratio < {isi_viol_threshold} and "
                       f"amplitude_cutoff < {amp_cutoff_threshold} and "
                       f"presence_ratio > {presence_ratio_threshold}")

sorting_outputs_folder = results_folder / "sortings"
sorting_outputs_folder.mkdir()

strategies = ["bit_truncation", "wavpack"]

# define options for bit truncation
zarr_clevel = 9
zarr_compressor = Blosc(cname='zstd', clevel=zarr_clevel, shuffle=Blosc.BITSHUFFLE)

# define wavpack options
level = 3
factors = {"bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
           "wavpack": [0, 6, 5, 4, 3.5, 3, 2.25]}

subset_columns = ["dset", "session", "strategy", "factor", "probe"]

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

    tmp_folder = scratch_folder / "tmp"
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(exist_ok=True, parents=True)

    for dset in dsets:
        print(f"\n\nProcessing dataset {dset}\n\n") 

        benchmark_file = results_folder / f"benchmark-lossy-nogt-{dset}.csv"

        if benchmark_file.is_file():
            df = pd.read_csv(benchmark_file, index_col=False)
            print(f"Entries in results: {len(df)}")
        else:
            df = None

        t_start_dset = time.perf_counter()

        for session in sessions[dset]:
            print(f"\nBenchmarking {session}\n\n")

            t_start_session = time.perf_counter()
            
            if "aind-np2" in dset:
                probe_name = "Neuropixels2.0"
                dset_name = "aind-np2"
            elif "aind-np1" in dset:
                probe_name = "Neuropixels2.0"
                dset_name = "aind-np1"
            else:
                probe_name = "Neuropixels1.0"
                dset_name = dset

            rec = si.load_extractor(ephys_benchmark_folder / dset_name / session)
            print(rec)    
            
            dur = rec.get_num_samples() / rec.get_sampling_frequency()
            dtype = rec.get_dtype()
            gain = rec.get_channel_gains()[0]
            
            rec_to_compress = None

            num_channels = rec.get_num_channels()
            fs = rec.get_sampling_frequency()
            
            full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

            zarr_root = session

            for strategy in strategies:
                print(f"Benchmarking {strategy}: {factors[strategy]}")
                for factor in factors[strategy]:
                    t_start_factor = time.perf_counter()

                    print(f"Compression factor {factor}")
                    entry_data = {"probe": probe_name, "strategy": strategy, "factor": factor, 
                                  "dataset": dset_name, "session": session}
                    rec_name = f"{dset_name}-{session}-{strategy}-{factor}"

                    if not is_entry(benchmark_file, entry_data):
                        print(f"Compression: {rec_name}")
                        if dset_name != "ibl":
                            if rec_to_compress is None:
                                rec_to_compress = spre.correct_lsb(rec)
                        else:
                            rec_to_compress = rec

                        zarr_path = tmp_folder / "zarr" / f"{zarr_root}_{strategy}_{factor}.zarr"

                        if zarr_path.is_dir():
                            shutil.rmtree(zarr_path)
                        
                        if strategy == "bit_truncation":
                            filters = trunc_filter(factor, rec.get_dtype())
                            compressor = zarr_compressor
                        else:
                            filters = None
                            compressor = WavPack(level=3, bps=factor)

                        rec_compressed, cr, cspeed_xrt, elapsed_time, rmse = \
                            benchmark_lossy_compression(rec_to_compress, compressor, zarr_path, 
                                                        filters=filters, time_range_rmse=time_range_rmse,
                                                        **job_kwargs)
                        print(f"Compression {rec_name}: cspeed xrt - {cspeed_xrt} - CR: {cr} - rmse: {rmse}\n")


                        new_data = {"dataset": dset_name, "session": session , "probe": probe_name, "strategy": strategy, 
                                    "factor": factor, "CR": cr, "cspeed_xrt": cspeed_xrt, 
                                    "rmse": rmse, "rec_zarr_path": str(zarr_path.absolute())}
                        
                        # run spike sorting
                        raw_sorting_output_folder = tmp_folder / f"sorting_{rec_name}"
                        sorting_path = sorting_outputs_folder / dset_name / f"sorting_{rec_name}"

                        # basic pre-processing
                        rec_zarr = si.read_zarr(zarr_path)
                        rec_zarr_f = spre.bandpass_filter(rec_zarr)
                        rec_zarr_cmr = spre.common_reference(rec_zarr_f)
                        
                        print(f"Spike sorting: {rec_name}")
                        sorting = ss.run_sorter(sorter, rec_zarr_cmr, output_folder=raw_sorting_output_folder,
                                                verbose=True, **sorter_params)
                        sorting = sorting.remove_empty_units()
                        # remove duplicated spikes
                        sorting = scur.remove_redundant_units(sorting, duplicate_threshold=0.9, align=False,
                                                              remove_strategy="max_spikes")
                        ks_good_unit_ids = sorting.unit_ids[sorting.get_property('KSLabel')=="good"]
                        sorting_good = sorting.select_units(unit_ids=ks_good_unit_ids)
                        sorting_saved = sorting.save(folder=sorting_path)
                        # cleanup
                        shutil.rmtree(raw_sorting_output_folder)
                        sorting = sorting_saved
                        
                        new_data["sorting_path"] = str(sorting_path.absolute())
                        new_data["n_raw_units"] = len(sorting_saved.unit_ids)
                        new_data["n_ks_good_units"] = len(sorting_good.unit_ids)

                        print(f"Spike sorting {rec_name}: num units - {len(sorting.unit_ids)} num KS good units "
                              f"- {len(sorting_good.unit_ids)}\n")

                    
                        # run auto-curation
                        print(f"Curation: {rec_name}")

                        sorting_curated_path = sorting_outputs_folder / dset_name / f"sorting_{rec_name}_curated"
                    
                        wf_path = tmp_folder / f"waveforms_raw_{dset_name}_{session}"
                        we = si.extract_waveforms(rec_zarr_cmr, sorting, folder=wf_path, **job_kwargs)
                        _ = spost.compute_spike_amplitudes(we, **job_kwargs)
                        qm = sqm.compute_quality_metrics(we, metric_names=metric_names)
                        units_to_keeps = qm.query(auto_curation_query).index.values

                        sorting_curated = sorting.select_units(units_to_keeps)
                        sorting_curated_saved = sorting_curated.save(folder=sorting_curated_path)

                        new_data["sorting_curated_path"] = str(sorting_curated_path.absolute())
                        new_data["n_curated_good_units"] = len(sorting_curated.unit_ids)
                        new_data["n_curated_bad_units"] = len(sorting.unit_ids) - len(sorting_curated.unit_ids)

                        print(f"Curation {rec_name}: num units - {len(sorting.unit_ids)} num auto-curated units "
                              f"{len(sorting_curated.unit_ids)}\n")

                        append_to_csv(benchmark_file, new_data, subset_columns=subset_columns)

                        print(f"\nSummary {rec_name}:\n")
                        print(f"Compression: cspeed xrt - {cspeed_xrt} - CR: {cr} - rmse: {rmse}\n")
                        print(f"Spike sorting: num units - {len(sorting.unit_ids)} num KS good units - "
                              f"{len(sorting_good.unit_ids)}\n")
                        print(f"Curation: num auto-curated units {len(sorting_curated.unit_ids)}\n")
                        # clean up
                        shutil.rmtree(zarr_path)
                        shutil.rmtree(wf_path)
                        del we

                    else:
                        print(f"{rec_name} already computed")
                    t_stop_factor = time.perf_counter()
                    elapsed_factor = np.round(t_stop_factor - t_start_factor)
                    print(f"Elapsed time {strategy}-{factor}: {elapsed_factor}s")
            t_stop_session = time.perf_counter()
            elapsed_session = np.round(t_stop_session - t_start_session)
            print(f"Elapsed time session: {elapsed_session}s")
        t_stop_dset = time.perf_counter()
        elapsed_dset = np.round(t_stop_dset - t_start_dset)
        print(f"Elapsed time dataset: {elapsed_dset}s")

    # final cleanup
    shutil.rmtree(tmp_folder)
            
    df = pd.read_csv(benchmark_file, index_col=False)
    print(f"Final # entries in results: {len(df)}")
