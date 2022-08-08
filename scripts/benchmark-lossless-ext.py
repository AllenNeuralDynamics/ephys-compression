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

from numcodecs import Blosc, Shuffle, blosc
import numcodecs

sys.path.append("..")

from audio_numcodecs import FlacCodec
from utils import append_to_csv, is_entry, get_median_and_lsb, benchmark_compression, \
    get_oe_stream, gs_download_folder, gs_upload_folder, s3_download_public_file, s3_download_public_folder

overwrite = False

print(f"spikeinterface: {si.__version__}")


data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression_ext" / "lossless"
if tmp_folder.is_dir():
    shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True, parents=True)

data_folder = Path("data") / "compression"

# gather data
rec_dsets_keys = ["aind", "ibl", "mindscope"]
rec_dsets = {}
for rec_key in rec_dsets_keys:
    rec_dsets[rec_key] = [p for p in data_folder / rec_key if p.is_dir()]

## Define compressions
blosc_compressors = ['blosc-lz4', 'blosc-lz4hc', 'blosc-zlib', 'blosc-zstd'] 
zarr_compressors = ['zstd', 'zlib', 'lz4', 'gzip', 'lzma']
audio_compressors = ['flac', 'wavpack'] 
compressors = blosc_compressors + zarr_compressors + audio_compressors

# define levels
levels = {
    "blosc": {"low": 1, "medium": 5, "high": 9},
    "zstd": {"low": 1, "medium": 11, "high": 22},
    "zlib": {"low": 1, "medium": 5, "high": 9},
    "lz4": {"low": 1, "medium": 5, "high": 9},
    "gzip": {"low": 1, "medium": 5, "high": 9},
    "lzma": {"low": 1, "medium": 5, "high": 9},
    "flac": {"low": 1, "medium": 5, "high": 8},
    "wavpack": {"low": 1, "medium": 2, "high": 3}
}

# define filters and shuffles
blosc_shuffles_dict = {"no": Blosc.NOSHUFFLE, 
                       "bit": Blosc.BITSHUFFLE, 
                       "shuffle": Blosc.SHUFFLE}
zarr_shuffles_dict = {"no": [], 
                      "shuffle": [Shuffle(2)]} # int16 --> 2 bytes
audio_shuffles_dict = {"no": []}

# define chunk sizes
channel_chunk_sizes = {"blosc": [-1], "zarr": [-1], "flac": [-1, 2], "wavpack": [-1]}
chunk_durations = ["0.1s", "1s", "10s"]
skip_durations = []

# define job kwargs
n_jobs = 20
job_kwargs = {'n_jobs': n_jobs, "verbose": False, "progress_bar": True}

# define LSB correction options
lsb_corrections = {"ibl": [False], # spikeGLX is already "LSB-corrected"
                   "aind": [False, True],
                   "mindscope": [False, True]}

print(f"Benchmarking compressors: {compressors}")

blosc_folder = tmp_folder / "blosc"
zarr_folder = tmp_folder / "zarr"
audio_folder = tmp_folder / "audio"

benchmark_file = data_folder / "results" / "benchmark-lossless-extended.csv"

print(benchmark_file)
if benchmark_file.is_file():
    df = pd.read_csv(benchmark_file, index_col=False)
    print(len(df))

subset_columns = ["session", "dataset", "compressor", "compressor_type", "chunk_duration", 
                  "level", "shuffle", "lsb", "probe", "channel_chunk_size"]

if overwrite:
    if benchmark_file.is_file():
        benchmark_file.unlink()

for rec_dset, rec_folders in rec_dsets.items():
    lsb_correction = lsb_corrections[rec_dset]
    for rec_folder in rec_folders:
        session = rec_folder.name
        try:
            if blosc_folder.is_dir():
                shutil.rmtree(blosc_folder)
            blosc_folder.mkdir(parents=True)
            if zarr_folder.is_dir():
                shutil.rmtree(zarr_folder)
            zarr_folder.mkdir(parents=True)
            if audio_folder.is_dir():
                shutil.rmtree(audio_folder)
            audio_folder.mkdir(parents=True)
        except:
            print("Couldn't remove tmp folder")

        print(f"\n\n\nBenchmarking {rec_dset}: {session}\n\n\n")
        t_start_all = time.perf_counter()
        
        if rec_dset == "aind":
            probe_name = "Neuropixels2.0"
        else:
            probe_name = "Neuropixels1.0"

        rec = si.load_extractor(rec_folder)
        print(rec)    

        num_channels = rec.get_num_channels()
        fs = rec.get_sampling_frequency()
        
        start_frame_1s = int(20 * fs)
        end_frame_1s = int(21 * fs)
        start_frame_10s = int(30 * fs)
        end_frame_10s = int(40 * fs)
        
        # compute lsb and median
        gain = rec.get_channel_gains()[0]
        dtype = rec.get_dtype()
        rec_lsb = None
        
        for cname in compressors:
            print(f"COMPRESSOR: {cname}")
            if cname in blosc_compressors:
                levels_blosc = levels["blosc"]
                channel_chunk_sizes_blosc = channel_chunk_sizes["blosc"]
                for channel_chunk_size in channel_chunk_sizes_blosc:
                    for level in levels_blosc:
                        for chunk_dur in chunk_durations:
                            job_kwargs["chunk_duration"] = chunk_dur
                            for shuffle_name, shuffle in blosc_shuffles_dict.items():
                                for lsb in lsb_correction:
                                    print(f"\ncompressor {cname} - level {level} chunk dur - {chunk_dur} "
                                        f"shuffle {shuffle_name} - lsb {lsb} - channel_chunk_size {channel_chunk_size}")
                                    entry_data = {"session": session, "dataset": rec_dset,
                                                 "compressor": cname, "compressor_type": "blosc", "level": level, 
                                                 "chunk_duration": chunk_dur, "shuffle": shuffle_name, "lsb": lsb, 
                                                 "probe": probe_name, "channel_chunk_size": channel_chunk_size}

                                    if not is_entry(benchmark_file, entry_data):
                                        filters = []
                                        blosc_cname = cname.split("-")[1]
                                        compressor = Blosc(cname=blosc_cname, clevel=levels_blosc[level], shuffle=shuffle)
                                        if lsb:
                                            if rec_lsb is None:
                                                # compute rec to compress if needed
                                                lsb_value, median_values = get_median_and_lsb(rec)

                                                # median and LSB correction 
                                                rec_lsb = si.scale(rec, gain=1., offset=-median_values, dtype=dtype)
                                                rec_lsb = si.scale(rec_lsb, gain=1. / lsb_value, dtype=dtype)
                                                rec_lsb.set_channel_gains(rec_lsb.get_channel_gains()*lsb_value)
                                            rec_to_compress = rec_lsb
                                        else:
                                            rec_to_compress = rec

                                        zarr_path = zarr_folder / f'{rec_file.name}_{cname}_{shuffle_name}_{chunk_dur}_{level}_lsb{lsb}.zarr'

                                        if zarr_path.is_dir():
                                            shutil.rmtree(zarr_path)

                                        if channel_chunk_size == -1:
                                            chan_size = None
                                        else:
                                            chan_size = channel_chunk_size

                                        t_start = time.perf_counter()
                                        rec_compressed = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                                                              compressor=compressor, filters=filters, 
                                                                              channels_per_chunk=chan_size,
                                                                              **job_kwargs)
                                        t_stop = time.perf_counter()
                                        compression_elapsed_time = np.round(t_stop - t_start, 2)
                                        
                                        cspeed_xrt = dur / compression_elapsed_time
                                        
                                        # cr
                                        cr = np.round(rec_compressed.get_annotation("compression_ratio"), 3)
                                        
                                        if lsb:
                                            rec_to_decompress = si.scale(rec_compressed, gain=lsb_value, dtype="int16")
                                            rec_to_decompress = si.scale(rec_to_decompress, offset=median_values, 
                                                                         dtype="int16")
                                        else:
                                            rec_to_decompress = rec_compressed

                                        # get traces 1s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_1s, 
                                                                              end_frame=end_frame_1s)
                                        t_stop = time.perf_counter()
                                        decompression_1s_elapsed_time = np.round(t_stop - t_start, 2)

                                        # get traces 10s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_10s, 
                                                                            end_frame=end_frame_10s)
                                        t_stop = time.perf_counter()
                                        decompression_10s_elapsed_time = np.round(t_stop - t_start, 2)

                                        decompression_10s_rt = 10. / decompression_10s_elapsed_time
                                        decompression_1s_rt = 1. / decompression_1s_elapsed_time

                                        # record entry
                                        data = {"session": session, "dataset": rec_dset,
                                                "probe": probe_name, "num_channels": num_channels,
                                                "duration": dur, "dtype": dtype, "compressor": cname, "level": level,
                                                "shuffle": shuffle_name, "lsb": lsb, "chunk_duration": chunk_dur,
                                                "CR": cr, "C-speed": compression_elapsed_time, "compressor_type": "audio",
                                                "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                                "cspeed_xrt": cspeed_xrt, "dspeed10s_xrt": decompression_10s_rt,
                                                "dspeed1s_xrt": decompression_1s_rt,
                                                "channel_chunk_size": channel_chunk_size}
                                        append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                        print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                              f"DC10s={decompression_10s_elapsed_time}s")
                                        # remove tmp path
                                        shutil.rmtree(zarr_path)
                                    else:
                                        print(f"Entry for {rec_file.name} with compressor {cname} - level {level} "
                                            f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb} "
                                            f"channel_chunk_size {channel_chunk_size} already present\n")
            elif cname in zarr_compressors:
                levels_zarr = levels[cname]
                channel_chunk_sizes_zarr = channel_chunk_sizes["zarr"]
                for channel_chunk_size in channel_chunk_sizes_zarr:
                    for level in levels_zarr:
                        for chunk_dur in chunk_durations:
                            job_kwargs["chunk_duration"] = chunk_dur
                            for shuffle_name, filters in zarr_shuffles_dict.items():
                                for lsb in lsb_correction:
                                    print(f"\ncompressor {cname} - level {level} chunk dur - {chunk_dur} "
                                          f"shuffle {shuffle_name} - lsb {lsb} - channel_chunk_size {channel_chunk_size}")
                                    entry_data = {"session": session, "dataset": rec_dset,
                                                  "compressor": cname, "compressor_type": "numcodecs", "level": level, 
                                                  "chunk_duration": chunk_dur, "shuffle": shuffle_name, "lsb": lsb, 
                                                  "probe": probe_name, "channel_chunk_size": channel_chunk_size}

                                    if not is_entry(benchmark_file, entry_data):
                                        if cname != "lzma":
                                            compressor = numcodecs.registry.codec_registry[cname](levels_zarr[level])
                                        else:
                                            compressor = numcodecs.registry.codec_registry[cname](preset=levels_zarr[level])
                                        if lsb:
                                            if rec_lsb is None:
                                                # compute rec to compress if needed
                                                lsb_value, median_values = get_median_and_lsb(rec)

                                                # median and LSB correction 
                                                rec_lsb = si.scale(rec, gain=1., offset=-median_values, dtype=dtype)
                                                rec_lsb = si.scale(rec_lsb, gain=1. / lsb_value, dtype=dtype)
                                                rec_lsb.set_channel_gains(rec_lsb.get_channel_gains()*lsb_value)
                                            rec_to_compress = rec_lsb
                                        else:
                                            rec_to_compress = rec

                                        zarr_path = zarr_folder / f'{rec_file.name}_{cname}_{shuffle_name}_{chunk_dur}_{level}_lsb{lsb}.zarr'

                                        if zarr_path.is_dir():
                                            shutil.rmtree(zarr_path)

                                        if channel_chunk_size == -1:
                                            chan_size = None
                                        else:
                                            chan_size = channel_chunk_size

                                        t_start = time.perf_counter()
                                        rec_compressed = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                                                            compressor=compressor, filters=filters, 
                                                                            channels_per_chunk=chan_size,
                                                                            **job_kwargs)
                                        t_stop = time.perf_counter()
                                        compression_elapsed_time = np.round(t_stop - t_start, 2)
                                        
                                        cspeed_xrt = dur / compression_elapsed_time
                                        
                                        # cr
                                        cr = np.round(rec_compressed.get_annotation("compression_ratio"), 3)

                                        
                                        if lsb:
                                            rec_to_decompress = si.scale(rec_compressed, gain=lsb_value, dtype="int16")
                                            rec_to_decompress = si.scale(rec_to_decompress, offset=median_values, 
                                                                         dtype="int16")
                                        else:
                                            rec_to_decompress = rec_compressed

                                        # get traces 1s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_1s, 
                                                                            end_frame=end_frame_1s)
                                        t_stop = time.perf_counter()
                                        decompression_1s_elapsed_time = np.round(t_stop - t_start, 2)

                                        # get traces 10s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_10s, 
                                                                            end_frame=end_frame_10s)
                                        t_stop = time.perf_counter()
                                        decompression_10s_elapsed_time = np.round(t_stop - t_start, 2)

                                        decompression_10s_rt = 10. / decompression_10s_elapsed_time
                                        decompression_1s_rt = 1. / decompression_1s_elapsed_time

                                        # record entry
                                        data = {"session": session, "dataset": rec_dset,
                                                "probe": probe_name, "num_channels": num_channels,
                                                "duration": dur, "dtype": dtype, "compressor": cname, "level": level,
                                                "shuffle": shuffle_name, "lsb": lsb, "chunk_duration": chunk_dur,
                                                "CR": cr, "C-speed": compression_elapsed_time, "compressor_type": "audio",
                                                "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                                "cspeed_xrt": cspeed_xrt, "dspeed10s_xrt": decompression_10s_rt,
                                                "dspeed1s_xrt": decompression_1s_rt,
                                                "channel_chunk_size": channel_chunk_size}
                                        append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                        print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                              f"DC10s={decompression_10s_elapsed_time}s")
                                        # remove tmp path
                                        shutil.rmtree(zarr_path)
                                    else:
                                        print(f"Entry for {rec_dset}/{session} with compressor {cname} - level {level} "
                                              f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb} "
                                              f"channel_chunk_size {channel_chunk_size} already present\n")
            else:
                # audio  
                levels_audio = levels[cname]
                channel_chunk_sizes_audio = channel_chunk_sizes[cname]
                for channel_chunk_size in channel_chunk_sizes_audio:
                    for level in levels_audio:
                        for chunk_dur in chunk_durations:        
                            if cname in skip_durations:
                                if chunk_dur in skip_durations[cname]:
                                    print(f"Skipping {chunk_dur} for {cname}")
                                    continue
                            job_kwargs["chunk_duration"] = chunk_dur
                            for shuffle_name, filters in audio_shuffles_dict.items():
                                for lsb in lsb_correction:
                                    print(f"compressor {cname} - level {level} chunk dur - {chunk_dur} "
                                          f"shuffle {shuffle_name} - lsb {lsb} - channel_chunk_size {channel_chunk_size}\n")
                                    entry_data = {"session": session, "dataset": rec_dset,
                                                  "compressor": cname, "compressor_type": "audio", "level": level, 
                                                  "chunk_duration": chunk_dur, "shuffle": shuffle_name, "lsb": lsb, 
                                                  "probe": probe_name, "channel_chunk_size": channel_chunk_size}

                                    if not is_entry(benchmark_file, entry_data):
                                        level_compressor = levels_audio[level]
                                        compressor = numcodecs.registry.codec_registry[cname](level_compressor)
                                        if lsb:
                                            if rec_lsb is None:
                                                # compute rec to compress if needed
                                                lsb_value, median_values = get_median_and_lsb(rec)

                                                # median and LSB correction 
                                                rec_lsb = si.scale(rec, gain=1., offset=-median_values, dtype=dtype)
                                                rec_lsb = si.scale(rec_lsb, gain=1. / lsb_value, dtype=dtype)
                                                rec_lsb.set_channel_gains(rec_lsb.get_channel_gains()*lsb_value)
                                            rec_to_compress = rec_lsb
                                        else:
                                            rec_to_compress = rec

                                        zarr_path = zarr_folder / f'{rec_file.name}_{cname}_{shuffle_name}_{chunk_dur}_{level}_lsb{lsb}_chans{channel_chunk_size}.zarr'
                                        if zarr_path.is_dir():
                                            shutil.rmtree(zarr_path)

                                        if channel_chunk_size == -1:
                                            chan_size = None
                                        else:
                                            chan_size = channel_chunk_size

                                        t_start = time.perf_counter()
                                        rec_compressed = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                                                              compressor=compressor, filters=filters, 
                                                                              channel_chunk_size=chan_size,
                                                                              **job_kwargs)
                                        t_stop = time.perf_counter()
                                        compression_elapsed_time = np.round(t_stop - t_start, 2)
                                        
                                        cspeed_xrt = dur / compression_elapsed_time
                                        
                                        # cr
                                        cr = np.round(rec_compressed.get_annotation("compression_ratio"), 3)

                                        
                                        if lsb:
                                            rec_to_decompress = si.scale(rec_compressed, gain=lsb_value, dtype="int16")
                                            rec_to_decompress = si.scale(rec_to_decompress, offset=median_values, 
                                                                         dtype="int16")
                                        else:
                                            rec_to_decompress = rec_compressed

                                        # get traces 1s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_1s, 
                                                                                end_frame=end_frame_1s)
                                        t_stop = time.perf_counter()
                                        decompression_1s_elapsed_time = np.round(t_stop - t_start, 2)

                                        # get traces 10s
                                        t_start = time.perf_counter()
                                        traces = rec_to_decompress.get_traces(start_frame=start_frame_10s, 
                                                                                end_frame=end_frame_10s)
                                        t_stop = time.perf_counter()
                                        decompression_10s_elapsed_time = np.round(t_stop - t_start, 2)
                                        
                                        decompression_10s_rt = 10. / decompression_10s_elapsed_time
                                        decompression_1s_rt = 1. / decompression_1s_elapsed_time

                                        # record entry
                                        data = {"session": session, "dataset": rec_dset,
                                                "probe": probe_name, "num_channels": num_channels,
                                                "duration": dur, "dtype": dtype, "compressor": cname, "level": level,
                                                "shuffle": shuffle_name, "lsb": lsb, "chunk_duration": chunk_dur,
                                                "CR": cr, "C-speed": compression_elapsed_time, "compressor_type": "audio",
                                                "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                                "cspeed_xrt": cspeed_xrt, "dspeed10s_xrt": decompression_10s_rt,
                                                "dspeed1s_xrt": decompression_1s_rt,
                                                "channel_chunk_size": channel_chunk_size}
                                        append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                        print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                              f"DC10s={decompression_10s_elapsed_time}s")
                                        # remove tmp path
                                        shutil.rmtree(zarr_path)
                                    else:
                                        print(f"Entry for {rec_dset}/{session} with compressor {cname} - level {level} "
                                              f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb} "
                                              f"channel_chunk_size {channel_chunk_size} already present\n")
