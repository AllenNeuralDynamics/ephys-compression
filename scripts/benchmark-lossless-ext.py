# Benchmark compression strategies
import gcsfs
import spikeinterface as si
import spikeinterface.preprocessing as spre
import probeinterface as pi

from pprint import pprint

import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sys

from numcodecs import Blosc, Shuffle
import numcodecs

sys.path.append("..")

from utils import append_to_csv, is_entry, gs_download_folder

from flac_numcodecs import Flac
from wavpack_numcodecs import WavPack

overwrite = False

print(f"spikeinterface: {si.__version__}")

gfs = gcsfs.GCSFileSystem()

data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression_ext" / "lossless"
# if tmp_folder.is_dir():
#     shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True, parents=True)
data_folder.mkdir(exist_ok=True)

compression_bucket_path = "aind-ephys-compression-benchmark-data"
compression_bucket = f"gs://{compression_bucket_path}"

# gather data
rec_dsets_keys = ["aind", "ibl", "mindscope"]
skip_sessions = ['797828357_probe805579745', '618382_2022-03-31_14-27-03_ProbeC', '613373_2022-04-26_15-36-12_ProbeC']

rec_dsets = {}
for rec_key in rec_dsets_keys:
    rec_dsets[rec_key] = []
    sessions = gfs.ls(f"{compression_bucket_path}/{rec_key}")
    for sess in sessions:
        session_name = sess.split("/")[-1]
        if session_name not in skip_sessions:
            rec_dsets[rec_key].append(session_name)

print("DATASETS\n")
pprint(rec_dsets)


# Define compressions
blosc_compressors = ['blosc-lz4', 'blosc-lz4hc', 'blosc-zlib', 'blosc-zstd']
numcodecs_compressors = ['zstd', 'zlib', 'lz4', 'gzip', 'lzma']
audio_compressors = ['flac', 'wavpack']
compressors = blosc_compressors + numcodecs_compressors + audio_compressors

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
chunk_durations = ["0.1s", "1s", "10s"]
skip_durations = []

# define job kwargs
n_jobs = 12
job_kwargs = {'n_jobs': n_jobs, "verbose": False, "progress_bar": True}

# define LSB correction options
lsb_corrections = {"ibl": {'none': False},  # spikeGLX is already "LSB-corrected"
                   "aind": {'false': False, 'true': True},
                   "mindscope": {'false': False, 'true': True}}

print(f"Benchmarking compressors: {compressors}")

benchmark_file = data_folder / "results" / "benchmark-lossless-extended-no-lsb-true-correct.csv"
benchmark_file.parent.mkdir(exist_ok=True, parents=True)

print(benchmark_file)
if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
    print(f"Number of existing entries: {len(df)}")
    print(f"LSB in dataset: {np.unique(df.lsb)}")

subset_columns = ["session", "dataset", "compressor", "compressor_type", "chunk_duration",
                  "level", "shuffle", "lsb", "probe", "channel_chunk_size"]

if overwrite:
    if benchmark_file.is_file():
        benchmark_file.unlink()

for rec_dset, sessions in rec_dsets.items():
    lsb_correction = lsb_corrections[rec_dset]
    for session in sessions:
        print(f"\n\n\nBenchmarking {rec_dset}: {session}\n\n\n")
        t_start_all = time.perf_counter()

        if rec_dset == "aind":
            probe_name = "Neuropixels2.0"
        else:
            probe_name = "Neuropixels1.0"

        rec = None
        rec_lsb = None
        rec_folder = None

        num_channels = None
        fs = None
        gain = None
        dtype = None

        # SIMPLIFY LOOP
        for cname in compressors:
            print(f"COMPRESSOR: {cname}")
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
                    for chunk_dur in chunk_durations:
                        job_kwargs["chunk_duration"] = chunk_dur
                        for shuffle_name, shuffle in shuffles[compressor_type].items():
                            for lsb_str, lsb in lsb_correction.items():
                                print(f"\ncompressor {cname} - level {level_name} chunk dur - {chunk_dur} "
                                      f"shuffle {shuffle_name} - lsb {lsb} - channel_chunk_size {channel_chunk_size}")
                                entry_data = {"session": session, "dataset": rec_dset,
                                              "compressor": cname, "compressor_type": compressor_type, 
                                              "level": level_name, "chunk_duration": chunk_dur, "shuffle": shuffle_name,
                                              "lsb": lsb_str, "probe": probe_name,
                                              "channel_chunk_size": channel_chunk_size}

                                if not is_entry(benchmark_file, entry_data):
                                    # download only if needed
                                    if rec is None:
                                        rec_folder = tmp_folder / session
                                        if not rec_folder.is_dir():
                                            print(f"Downloading {rec_dset}-{session}")
                                            gs_download_folder(compression_bucket,
                                                                f"{rec_dset}/{session}", rec_folder.parent)
                                        else:
                                            print(f"{rec_folder} found locally")
                                        rec = si.load_extractor(rec_folder)
                                        print(rec)

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
                                            rec_lsb = spre.correct_lsb(rec)
                                        rec_to_compress = rec_lsb
                                    else:
                                        rec_to_compress = rec

                                    zarr_path = tmp_folder / f'{rec_dset}_{session}.zarr'
                                    if zarr_path.is_dir():
                                        shutil.rmtree(zarr_path)

                                    if channel_chunk_size == -1:
                                        chan_size = None
                                    else:
                                        chan_size = channel_chunk_size

                                    t_start = time.perf_counter()
                                    rec_compressed = rec_to_compress.save(folder=zarr_path, format="zarr",
                                                                          compressor=compressor, filters=filters,
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
                                    data = {"session": session, "dataset": rec_dset,
                                            "probe": probe_name, "num_channels": num_channels,
                                            "duration": dur, "dtype": dtype, "compressor": cname, "level": level_name,
                                            "shuffle": shuffle_name, "lsb": lsb_str, "chunk_duration": chunk_dur,
                                            "CR": cr, "C-speed": compression_elapsed_time, "compressor_type": compressor_type,
                                            "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                            "cspeed_xrt": cspeed_xrt, "dspeed10s_xrt": decompression_10s_rt,
                                            "dspeed1s_xrt": decompression_1s_rt, "channel_chunk_size": channel_chunk_size}
                                    append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                    print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                          f"DC10s={decompression_10s_elapsed_time}s")
                                    # remove tmp path
                                    shutil.rmtree(zarr_path)
                                else:
                                    print(f"Entry for {rec_dset}/{session} with compressor {cname} - level {level} "
                                          f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb_str} "
                                          f"channel_chunk_size {channel_chunk_size} already present\n")

        # remove tmp path
        if rec_folder is not None:
            if rec_folder.is_dir():
                shutil.rmtree(rec_folder)
