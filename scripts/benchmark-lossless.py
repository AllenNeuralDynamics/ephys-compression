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

from numcodecs import Blosc, Delta, blosc

sys.path.append("..")

from audiocompression import write_recording_audio, AudioRecordingExtractor
from utils import get_dir_size, append_to_csv, is_entry

overwrite = False

print(f"spikeinterface: {si.__version__}")


data_folder = Path("../data")
tmp_folder = data_folder / "tmp_compression" / "lossless"
if tmp_folder.is_dir():
    shutil.rmtree(tmp_folder)
tmp_folder.mkdir(exist_ok=True, parents=True)

# NP1.0 and NP2.0
rec_files = ["/home/alessio/Documents/data/allen/npix-open-ephys/595262_2022-02-21_15-18-07/Record Node 102", 
             "/home/alessio/Documents/data/allen/npix-open-ephys/618382_2022-03-31_14-27-03/Record Node 102"]

shuffles_dict = {"no": Blosc.NOSHUFFLE, 
                 "bit": Blosc.BITSHUFFLE, 
                 "shuffle": Blosc.SHUFFLE}

## Define compressions
zarr_compressors = ['lz4', 'lz4hc', 'zlib', 'zstd'] 
audio_compressors = ['flac', 'wavpack'] # blosc.list_compressors()
compressors = zarr_compressors + audio_compressors
levels = [1, 5, 9]

levels_audio = {"flac": levels, "wavpack": [1]}

chunk_durations = ["0.1s", "1s", "10s"]
n_jobs = 10

job_kwargs = {'n_jobs': n_jobs, "verbose": False, "progress_bar": True}

num_compressors = len(compressors)
num_shuffles = len(shuffles_dict)
num_chunk_durations = len(chunk_durations)
num_levels = len(levels)
lsb_correction = [False, True]
num_entries_zarr = num_compressors * num_levels * num_shuffles * num_chunk_durations

print(f"Benchmarking compressors: {compressors}")

zarr_folder = tmp_folder / "zarr"
audio_folder = tmp_folder / "audio"

benchmark_file = data_folder / "results" / "benchmark-lossless-lsb.csv"
print(benchmark_file)

subset_columns = ["compressor", "chunk_dur", "level", "shuffle", "lsb", "probe"]

if overwrite:
    if benchmark_file.is_file():
        benchmark_file.unlink()

for rec_file in rec_files:
    rec_file = Path(rec_file)
    try:
        if zarr_folder.is_dir():
            shutil.rmtree(zarr_folder)
        zarr_folder.mkdir(parents=True)
        if audio_folder.is_dir():
            shutil.rmtree(audio_folder)
        audio_folder.mkdir(parents=True)
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
    dur = rec.get_num_samples() / rec.get_sampling_frequency()
    dtype = rec.get_dtype()
    gain = rec.get_channel_gains()[0]
    print(rec)    

    num_channels = rec.get_num_channels()
    fs = rec.get_sampling_frequency()
    
    start_frame_1s = int(20 * fs)
    end_frame_1s = int(21 * fs)
    start_frame_10s = int(30 * fs)
    end_frame_10s = int(40 * fs)
    
    full_size = rec.get_dtype().itemsize * rec.get_num_samples() * rec.get_num_channels()

    # compute lsb and median
    # gather chunks
    num_random_chunks = 10
    chunks = None
    for i in tqdm(range(num_random_chunks), desc="Extracting chunks"):
        chunks_i2 = si.get_random_data_chunks(rec, chunk_size=30000, seed=i**2)
        if chunks is None:
            chunks = chunks_i2
        else:
            chunks = np.vstack((chunks, chunks_i2))
    
    lsb_value = 0 
    num_channels = rec.get_num_channels()
    gain = rec.get_channel_gains()[0]
    dtype = rec.get_dtype()

    channel_idxs = np.arange(num_channels)
    min_values = np.zeros(num_channels, dtype=dtype)
    median_values = np.zeros(num_channels, dtype=dtype)
    offsets = np.zeros(num_channels, dtype=dtype)

    for ch in tqdm(channel_idxs, desc="Estimating channel stats"):
        unique_vals = np.unique(chunks[:, ch])
        unique_vals_abs = np.abs(unique_vals)
        lsb_val = np.min(np.diff(unique_vals))
        
        min_values[ch] = np.min(unique_vals_abs)
        median_values[ch] = np.median(chunks[:, ch]).astype(dtype)
        
        unique_vals_m = np.unique(chunks[:, ch] - median_values[ch])
        unique_vals_abs_m = np.abs(unique_vals_m)
        offsets[ch] = np.min(unique_vals_abs_m)
        
        if lsb_val > lsb_value:
            lsb_value = lsb_val

    print(f"LSB int16 {lsb_value} --> {lsb_value * gain} uV")

    # No ZARR filters
    filters = []

    for cname in compressors:
        print(f"COMPRESSOR: {cname}")
        if cname not in audio_compressors:
            for level in levels:
                for chunk_dur in chunk_durations:
                    job_kwargs["chunk_duration"] = chunk_dur
                    for shuffle_name, shuffle in shuffles_dict.items():
                        for lsb in lsb_correction:
                            print(f"compressor {cname} - level {level} chunk dur - {chunk_dur} "
                                  f"shuffle {shuffle_name} - lsb {lsb}\n")
                            entry_data = {"compressor": cname, "level": level, "chunk_dur": chunk_dur,
                                          "shuffle": shuffle_name, "lsb": lsb, "probe": probe_name}


                            if not is_entry(benchmark_file, entry_data):

                                compressor = Blosc(cname=cname, clevel=level, shuffle=shuffle)
                                if lsb:
                                    rec_to_compress = si.scale(rec, gain=1., offset=-median_values, dtype="int16")
                                    rec_to_compress = si.scale(rec_to_compress, gain=1. / lsb_value, dtype="int16")
                                else:
                                    rec_to_compress = rec

                                zarr_path = zarr_folder / f'{rec_file.name}_{cname}_{shuffle_name}_{chunk_dur}_{level}_lsb{lsb}.zarr'

                                if zarr_path.is_dir():
                                    shutil.rmtree(zarr_path)

                                t_start = time.perf_counter()
                                rec_compressed = rec_to_compress.save(format="zarr", zarr_path=zarr_path, 
                                                                      compressor=compressor, filters=filters, 
                                                                      **job_kwargs)
                                t_stop = time.perf_counter()
                                compression_elapsed_time = np.round(t_stop - t_start, 2)
                                
                                xRT = dur / compression_elapsed_time
                                
                                # cr
                                cr = np.round(rec_compressed._root['traces_seg0'].nbytes / 
                                              rec_compressed._root['traces_seg0'].nbytes_stored, 3)
                                
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

                                # record entry
                                data = {"probe": probe_name, "num_channels": num_channels, 
                                        "duration": dur, "dtype": dtype, "compressor": cname, "level": level,
                                        "shuffle": shuffle_name, "lsb": lsb, "chunk_dur": chunk_dur,
                                        "CR": cr, "C-speed": compression_elapsed_time,
                                        "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                        "xRT": xRT}
                                append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                                print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                      f"DC10s={decompression_10s_elapsed_time}s")
                                # remove tmp path
                                shutil.rmtree(zarr_path)
                            else:
                                print(f"Entry for {rec_file.name} with compressor {cname} - level {level} "
                                      f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb} "
                                      f"already present\n")
        else:
            for level in levels_audio[cname]:
                for chunk_dur in chunk_durations:
                    job_kwargs["chunk_duration"] = chunk_dur
                    for lsb in lsb_correction:
                        shuffle_name = "no"
                        print(f"compressor {cname} - level {level} chunk dur - {chunk_dur} "
                              f"shuffle {shuffle_name} - lsb {lsb}\n")
                        entry_data = {"compressor": cname, "level": level, "chunk_dur": chunk_dur,
                                      "shuffle": shuffle_name, "lsb": lsb, "probe": probe_name}

                        if not is_entry(benchmark_file, entry_data):
                            if lsb:
                                rec_to_compress = si.scale(rec, gain=1., offset=-median_values, dtype="int16")
                                rec_to_compress = si.scale(rec_to_compress, gain=1. / lsb_value, dtype="int16")
                            else:
                                rec_to_compress = rec

                            output_folder = audio_folder / f'{rec_file.name}_{cname}_{shuffle_name}_{chunk_dur}_{level}_lsb-{lsb}'

                            if output_folder.is_dir():
                                shutil.rmtree(output_folder)

                            t_start = time.perf_counter()
                            rec_compressed = write_recording_audio(rec_to_compress, cformat=cname, 
                                                                   output_folder=output_folder,
                                                                   compression_level=np.min([level, 8]), 
                                                                   **job_kwargs)
                            t_stop = time.perf_counter()
                            compression_elapsed_time = np.round(t_stop - t_start, 2)
                            
                            xRT = dur / compression_elapsed_time
                            
                            # cr
                            cr = np.round(full_size / get_dir_size(output_folder), 3)

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

                            # record entry
                            data = {"probe": probe_name, "num_channels": num_channels,
                                    "duration": dur, "dtype": dtype, "compressor": cname, "level": level,
                                    "shuffle": shuffle_name, "lsb": lsb, "chunk_dur": chunk_dur,
                                    "CR": cr, "C-speed": compression_elapsed_time,
                                    "D-1s": decompression_1s_elapsed_time, "D-10s": decompression_10s_elapsed_time,
                                    "xRT": xRT}
                            append_to_csv(benchmark_file, data, subset_columns=subset_columns)
                            print(f"Compression took {compression_elapsed_time}s - CR={cr} - "
                                    f"DC10s={decompression_10s_elapsed_time}s")
                            # remove tmp path
                            shutil.rmtree(output_folder)
                        else:
                            print(f"Entry for {rec_file.name} with compressor {cname} - level {level} "
                                  f"chunk duration - {chunk_dur} shuffle {shuffle_name} - lsb {lsb} "
                                  f"already present\n")
