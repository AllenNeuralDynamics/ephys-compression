"""
This script generates a folder with a series of json files that can be used to parallelize 
computations in CodeOcean pipelines.

It generates a "job_configs" output folder with three additional sub-folders:

- lossless: config files for lossless compression benchmark
- lossy-exp: config files for lossy compression on experimental data
- lossy-gt: config files for lossy compression on simulated data
"""

import json
from pathlib import Path

DEBUG = False

output_folder = Path("../data")

exp_datasets = ["aind-np2-1", "aind-np2-2", "ibl-np1", "aind-np1"]
sim_datasets = ["NP1", "NP2"]

lossy_strategies = ["bit_truncation", "wavpack"]
lossy_factors = {
    "bit_truncation": [0, 1, 2, 3, 4, 5, 6, 7],
    "wavpack": [0, 6, 5, 4, 3.5, 3, 2.25],
}

# lossless: parallelize over datasets, chunk durations, compressors
chunk_durations = ["0.1s", "1s", "10s"]
compressors = [
    "blosc-lz4",
    "blosc-lz4hc",
    "blosc-zlib",
    "blosc-zstd",
    "zstd",
    "zlib",
    "lz4",
    "gzip",
    "lzma",
    "flac",
    "wavpack",
]
compressors_delta_pre = ["blosc-zstd", "lzma", "flac", "wavpack"]

if DEBUG:
    job_config_folder_name = "job_configs_example"
    max_configs = 2
else:
    job_config_folder_name = "job_configs"
    max_configs = None

job_folder = output_folder / job_config_folder_name
job_folder.mkdir(exist_ok=True)
lossless_folder = job_folder / "lossless"
lossless_folder.mkdir(exist_ok=True)

i = 1
for dset in exp_datasets:
    for chunk_duration in chunk_durations:
        for compressor in compressors:
            config_dict = dict(dset=dset, chunk_duration=chunk_duration, compressor=compressor)
            with (lossless_folder / f"job_config_{i}.json").open("w") as f:
                json.dump(config_dict, f)
            i += 1
            if max_configs is not None and i >= max_configs:
                break

# lossy-exp: parallelize over datasets, strategy, factors
lossy_exp_folder = job_folder / "lossy-exp"
lossy_exp_folder.mkdir(exist_ok=True)

i = 1
for dset in exp_datasets:
    for strategy in lossy_strategies:
        for factor in lossy_factors[strategy]:
            config_dict = dict(dset=dset, strategy=strategy, factor=factor)
            with (lossy_exp_folder / f"job_config_{i}.json").open("w") as f:
                json.dump(config_dict, f)
            i += 1
            if max_configs is not None and i >= max_configs:
                break

# lossy-sim: parallelize over datasets, strategy, factors
lossy_sim_folder = job_folder / "lossy-sim"
lossy_sim_folder.mkdir(exist_ok=True)

i = 1
for dset in sim_datasets:
    for strategy in lossy_strategies:
        for factor in lossy_factors[strategy]:
            config_dict = dict(dset=dset, strategy=strategy, factor=factor)
            with (lossy_sim_folder / f"job_config_{i}.json").open("w") as f:
                json.dump(config_dict, f)
            i += 1
            if max_configs is not None and i >= max_configs:
                break

lossless_delta_pre_folder = job_folder / "lossless_delta_pre"
lossless_delta_pre_folder.mkdir(exist_ok=True)

i = 1
for dset in exp_datasets:
    for compressor in compressors_delta_pre:
        config_dict = dict(dset=dset, compressor=compressor)
        with (lossless_delta_pre_folder / f"job_config_{i}.json").open("w") as f:
            json.dump(config_dict, f)
        i += 1
        if max_configs is not None and i >= max_configs:
            break
