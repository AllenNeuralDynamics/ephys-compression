"""
Benchmark lossless compression strategies on experimental data.

The script expects a CodeOcean file organization

- code
- data
- results

The script is run from the "code" folder and expect the "aind-ephys-compression-benchmark-data" bucket to be attached 
to the data folder.

Different datasets (aind1, aind2, ibl, mindscope) can be run in parallel by passing them as an argument (or using the 
"App Panel").
"""
import os
import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sys
import json

from numcodecs import Blosc, Shuffle
import numcodecs

import spikeinterface as si
import spikeinterface.preprocessing as spre

# add utils to path
this_folder = Path(__file__).parent
sys.path.append(str(this_folder.parent))

from utils import append_to_csv, is_entry, gs_download_folder

from flac_numcodecs import Flac
from wavpack_numcodecs import WavPack

overwrite = False

data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")

tmp_folder = scratch_folder / "tmp_compression" / "lossless"
if tmp_folder.is_dir():
    shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True, parents=True)

# gather data
sessions = {
    "aind-np2-1": ['595262_2022-02-21_15-18-07_ProbeA',
                   '602454_2022-03-22_16-30-03_ProbeB',
                   '612962_2022-04-13_19-18-04_ProbeB',
                   '612962_2022-04-14_17-17-10_ProbeC',],
    "aind-np2-2": ['618197_2022-06-21_14-08-06_ProbeC',
                   '618318_2022-04-13_14-59-07_ProbeB',
                   '618384_2022-04-14_15-11-00_ProbeB',
                   '621362_2022-07-14_11-19-36_ProbeA'],
    "aind-np1": ['605642_2022-03-11_16-03-34_ProbeA',
                 '613482_2022-06-16_17-49-19_ProbeA',
                 '625749_2022-08-03_15-15-06_ProbeA',
                 '634568_2022-08-05_15-59-46_ProbeA'],
    "ibl-np1": ['CSHZAD026_2020-09-04_probe00',
                'CSHZAD029_2020-09-09_probe00',
                'SWC054_2020-10-05_probe00',
                'SWC054_2020-10-05_probe01'],
    # "mindscope-np1": ['754312389_probe756781559',
    #                   '766640955_probe773592324',
    #                   '829720705_probe832129157',
    #                   '839557629_probe846401838']

}
all_dsets = ["aind-np2-1", "aind-np2-2", "ibl-np1", "aind-np1"] # "mindscope-np1"]

# Define compressions
blosc_compressors = ['blosc-lz4', 'blosc-lz4hc', 'blosc-zlib', 'blosc-zstd']
numcodecs_compressors = ['zstd', 'zlib', 'lz4', 'gzip', 'lzma']
audio_compressors = ['flac', 'wavpack']
all_compressors = blosc_compressors + numcodecs_compressors + audio_compressors

# define levels
levels = {
    "blosc": {"low": 1, "medium": 5, "high": 9},
    "zstd": {"low": 1, "medium": 11, "high": 22},
    "zlib": {"low": 1, "medium": 5, "high": 9},
    "lz4": {"low": 9, "medium": 5, "high": 1},
    "gzip": {"low": 1, "medium": 5, "high": 9},
    "lzma": {"low": 1, "medium": 5, "high": 9},
    "flac": {"low": 1, "medium": 5, "high": 8},
    "wavpack": {"low": 1, "medium": 2, "high": 3}
}

# define filters and shuffles
shuffles = {
    "blosc" : {"no": Blosc.NOSHUFFLE,
               "bit": Blosc.BITSHUFFLE,
               "byte": Blosc.SHUFFLE},
    "numcodecs" : {"no": [],
                   "byte": [Shuffle(2)]},  # int16 --> 2 bytes
    "audio" : {"no": []}
}

# define chunk sizes
channel_chunk_sizes = {"blosc": [-1], "numcodecs": [-1], "flac": [-1, 2], "wavpack": [-1]}
all_chunk_durations = ["0.1s", "1s", "10s"]
skip_durations = []

# define job kwargs
n_jobs = None
job_kwargs = {'n_jobs': n_jobs if n_jobs is not None else os.cpu_count(), "verbose": False, "progress_bar": False}

# define LSB correction options
lsb_corrections = {"ibl-np1": {'none': False},  # spikeGLX is already "LSB-corrected"
                   "aind-np2": {'false': False, 'true': True},
                   "aind-np1": {'false': False, 'true': True},
                   "mindscope-np1": {'false': False, 'true': True}}
subset_columns = ["session", "dataset", "compressor", "compressor_type", "chunk_duration",
                  "level", "shuffle", "lsb", "probe", "channel_chunk_size"]

if __name__ == "__main__":
    # check if json files in data
    json_files = [p for p in data_folder.iterdir() if p.suffix == ".json"]

    if len(sys.argv) == 4:
        if sys.argv[1] == "all":
            dsets = all_dsets
        else:
            dsets = [sys.argv[1]]
        if sys.argv[2] == "all":
            chunk_durations = all_chunk_durations
        else:
            chunk_durations = [sys.argv[2]]
        if sys.argv[3] == "all":
            compressors = all_compressors
        else:
            compressors = [sys.argv[2]]
    elif len(json_files) == 1:
        config_file = json_files[0]
        config = json.load(open(config_file, 'r'))
        dsets = [config["dset"]]
        chunk_durations = [config["chunk_duration"]]
        compressors = [config["compressor"]]
    else:
        dsets = all_dsets
        chunk_durations = all_chunk_durations
        compressors = all_compressors

    print(f"Benchmarking:\n\tDatasets: {dsets}\n\tChunk durations: {chunk_durations}\n\tCompressors: {compressors}")

    ephys_benchmark_folders = [p for p in data_folder.iterdir() if p.is_dir() and "compression-benchmark" in p.name]
    if len(ephys_benchmark_folders) != 1:
        raise Exception("Couldn't find attached benchmark data")    
    ephys_benchmark_folder = ephys_benchmark_folders[0]
    print(f"Benchmark data folder: {ephys_benchmark_folder}")

    print(f"spikeinterface version: {si.__version__}")

    # check if the ephys data is available
    for dset in dsets:
        if "aind-np2" in dset:
            probe_name = "Neuropixels2.0"
            dset_name = "aind-np2"
        elif "aind-np1" in dset:
            probe_name = "Neuropixels1.0"
            dset_name = "aind-np1"
        else:
            probe_name = "Neuropixels1.0"
            dset_name = dset
        lsb_correction = lsb_corrections[dset_name]

        for chunk_dur in chunk_durations:
            job_kwargs["chunk_duration"] = chunk_dur
            
            for cname in compressors:
                print(f"\n\n\nBenchmarking dset {dset} - duration: {chunk_dur} - compressor {cname}\n\n")

                # create results file
                benchmark_file = results_folder / f"benchmark-lossless-{dset}-{chunk_dur}-{cname}.csv"
                benchmark_file.parent.mkdir(exist_ok=True, parents=True)
                if overwrite:
                    if benchmark_file.is_file():
                        benchmark_file.unlink()
                else:
                    if benchmark_file.is_file():
                        df = pd.read_csv(benchmark_file, index_col=False)
                        print(f"Number of existing entries: {len(df)}")

                # loop over sessions in dataset
                for session in sessions[dset]:
                    print(f"\nBenchmarking session: {session}\n")
                    t_start_all = time.perf_counter()

                    rec = None
                    rec_lsb = None
                    rec_folder = None

                    num_channels = None
                    fs = None
                    gain = None
                    dtype = None

                    if cname in blosc_compressors:
                        compressor_type = "blosc"
                        levels_compressor = levels[compressor_type]
                        channel_chunk_size_comp = channel_chunk_sizes[compressor_type]
                    elif cname in numcodecs_compressors:
                        compressor_type = "numcodecs"
                        levels_compressor = levels[cname]
                        channel_chunk_size_comp = channel_chunk_sizes[compressor_type]
                    elif cname in audio_compressors:
                        compressor_type = "audio"
                        levels_compressor = levels[cname]
                        channel_chunk_size_comp = channel_chunk_sizes[cname]

                    for channel_chunk_size in channel_chunk_size_comp:
                        for level_name, level in levels_compressor.items():
                            for shuffle_name, shuffle in shuffles[compressor_type].items():
                                for lsb_str, lsb in lsb_correction.items():
                                    entry_data = {"session": session, "dataset": dset_name,
                                                    "compressor": cname, "compressor_type": compressor_type,
                                                    "level": level_name, "chunk_duration": chunk_dur,
                                                    "shuffle": shuffle_name, "lsb": lsb_str, "probe": probe_name,
                                                    "channel_chunk_size": channel_chunk_size}

                                    if not is_entry(benchmark_file, entry_data):
                                        print(f"\n\tCompressor {cname}: level {level_name} "
                                                f"chunk duration - {chunk_dur} shuffle {shuffle_name} - "
                                                f"lsb {lsb_str} - channel_chunk_size {channel_chunk_size}")
                                        # download only if needed
                                        if rec is None:
                                            rec_folder = ephys_benchmark_folder / dset_name / session
                                            rec = si.load_extractor(rec_folder)

                                            # rec_info
                                            num_channels = rec.get_num_channels()
                                            fs = rec.get_sampling_frequency()
                                            gain = rec.get_channel_gains()[0]
                                            dtype = rec.get_dtype()

                                            # define intervals for decompression
                                            fs = 30000
                                            start_frame_1s = int(20 * fs)
                                            end_frame_1s = int(21 * fs)
                                            start_frame_10s = int(30 * fs)
                                            end_frame_10s = int(40 * fs)
                                            dur = rec.get_total_duration()

                                        # setup filters and compressors
                                        if compressor_type == "blosc":
                                            filters = []
                                            blosc_cname = cname.split("-")[1]
                                            compressor = Blosc(cname=blosc_cname,
                                                            clevel=level, shuffle=shuffle)
                                        elif compressor_type == "numcodecs":
                                            if cname != "lzma":
                                                compressor = numcodecs.registry.codec_registry[cname](level)
                                            else:
                                                compressor = numcodecs.registry.codec_registry[cname](preset=level)
                                            filters = shuffle
                                        elif compressor_type == "audio":
                                            filters = shuffle
                                            compressor = numcodecs.registry.codec_registry[cname](level)

                                        if lsb:
                                            if rec_lsb is None:
                                                rec_lsb = spre.correct_lsb(rec, verbose=True)
                                            rec_to_compress = rec_lsb
                                        else:
                                            rec_to_compress = rec

                                        zarr_path = tmp_folder / f'{dset_name}_{session}.zarr'
                                        if zarr_path.is_dir():
                                            shutil.rmtree(zarr_path)

                                        if channel_chunk_size == -1:
                                            chan_size = None
                                        else:
                                            chan_size = channel_chunk_size

                                        t_start = time.perf_counter()
                                        rec_compressed = rec_to_compress.save(folder=zarr_path, format="zarr",
                                                                                compressor=compressor,
                                                                                filters=filters,
                                                                                channel_chunk_size=chan_size,
                                                                                **job_kwargs)
                                        t_stop = time.perf_counter()
                                        compression_elapsed_time = np.round(t_stop - t_start, 2)

                                        cspeed_xrt = dur / compression_elapsed_time

                                        # cr
                                        cr = np.round(rec_compressed.get_annotation("compression_ratio"), 3)

                                        # get traces 1s
                                        t_start = time.perf_counter()
                                        traces = rec_compressed.get_traces(start_frame=start_frame_1s,
                                                                        end_frame=end_frame_1s)
                                        t_stop = time.perf_counter()
                                        decompression_1s_elapsed_time = np.round(t_stop - t_start, 2)

                                        # get traces 10s
                                        t_start = time.perf_counter()
                                        traces = rec_compressed.get_traces(start_frame=start_frame_10s,
                                                                                end_frame=end_frame_10s)
                                        t_stop = time.perf_counter()
                                        decompression_10s_elapsed_time = np.round(t_stop - t_start, 2)

                                        decompression_10s_rt = 10. / decompression_10s_elapsed_time
                                        decompression_1s_rt = 1. / decompression_1s_elapsed_time

                                        # record entry
                                        data = {"session": session, "dataset": dset_name,
                                                "probe": probe_name, "num_channels": num_channels,
                                                "duration": dur, "dtype": dtype, "compressor": cname,
                                                "level": level_name, "shuffle": shuffle_name, "lsb": lsb_str,
                                                "chunk_duration": chunk_dur, "CR": cr,
                                                "C-speed": compression_elapsed_time,
                                                "compressor_type": compressor_type,
                                                "D-1s": decompression_1s_elapsed_time,
                                                "D-10s": decompression_10s_elapsed_time, "cspeed_xrt": cspeed_xrt,
                                                "dspeed10s_xrt": decompression_10s_rt,
                                                "dspeed1s_xrt": decompression_1s_rt,
                                                "channel_chunk_size": channel_chunk_size}
                                        append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                        print(f"\t--> elapsed time {compression_elapsed_time}s - CR={cr} - "
                                                f"cspeed_xrt={cspeed_xrt} - dspeed10s_xrt={decompression_10s_rt}")
                                        # remove tmp path
                                        shutil.rmtree(zarr_path)
