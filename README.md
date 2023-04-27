# Compression strategies for large-scale electrophysiology data 

Collection of scripts to generate compression data for the "Compression strategies for large-scale electrophysiology data" 
manuscript.

The `scripts` folder contains the following Python scripts:

# Data preparation:

The `scripts/prepare_data_for_compression.py` script downloads 16 Neuropixels datasets (8 NP1, 8 NP2) from the cloud*, 
saves them as raw binary with SpikeInterface, and uploads them to the open `arn:aws:s3:::aind-ephys-compression-benchmark-data` bucket.

The datasets divided into four sources:
- `aind-np1`: 4 NP1 datasets from the Allen Institute for Neural Dynamics (AIND)
- `ibl-np1`: 4 NP1 datasets from the International Brain Laboratory (IBL)
- `aind-np2`: 8 NP2 datasets from the AIND (further divided in `aind-np2-1` and `aind-np2-2` in the benchmark scripts)

**NOTE:** the `arn:aws:s3:::aind-ephys-compression-benchmark-data` hase already been generated and publicly shared, so there is no need to re-rn this script.

*it requires access to the Allen Institute for Neural Dynamics (AIND) AWS account


# Data simulation:

The `scripts/generate-gt-neuropixels-data.py` uses the [MEArec](https://mearec.readthedocs.io/en/latest/) simulator to generate ground-truth NP1 and NP2 datasets.
The two simulated datasets can be found in the the `aind-ephys-compression-benchmark-data/mearec` folder.

# Benchmarks:

## Lossless

The scripts compute three performance metrics: compression ratio (CR), compression speed, and decompression speeds.

### `scripts/benchmark-lossless.py`

Benchmark of 11 lossless compressors (`blosc-lz4`, `blosc-lz4hc`, `blosc-zlib`, `blosc-zstd`,
`gzip`, `lz4`, `lzma`, `zlib`, `zstd`, `flac`, `wavpack`) on the 16 experimental datasets.
For each codec, three compression levels (*low*, *medium*, *high*) and three chunk durations (0.1, 1, and 10s) are benchmarked.
For AIND datasets, the script also tests the effect of *LSB correction*.
  
The script accepts 3 arguments: 
```
>>> python scripts/benchmark-lossless.py "dataset" "chunk_duration" "compressor"
```
For example:
```
>>> python scripts/benchmark-lossless.py ibl-np1 1s flac
```
will run the `flac` codec with a chunk duration of `1s` on the `ibl-np1` dataset. 
Alternatively, arguments can be specified with a JSON file containing the same keys in the `../data` folder.

The script produces the `results/benchmark-lossless.csv` file.



### `scripts/benchmark-lossless-delta.py`
 
Benchmark of different types of *delta* filters (1d, 2d in time, 2d in space, 2d in time and space) for the four best performing codecs (`blosc-zstd`, `lzma`, `flac`, `wavpack`).

The script accepts 2 arguments: 
```
>>> python scripts/benchmark-lossless-delta.py "dataset" "compressor"
```
For example:
```
>>> python scripts/benchmark-lossless.py aind-np2-1 wavpack
```
will run the delta benchmark for the `wavpack` codec on the `aind-np2-1` dataset.
Alternatively, arguments can be specified with a JSON file containing the same keys in the `../data` folder.

The script produces the `results/benchmark-lossless-delta.csv` file.


### `scripts/benchmark-lossless-delta.py`
 
Benchmark of different types of preprocessing (highpass_300, bandpass_300-6000, bandpass_300-15000, anti-aliasing) for the four best performing codecs (`blosc-zstd`, `lzma`, `flac`, `wavpack`).

The script accepts 2 arguments: 
```
>>> python scripts/benchmark-lossless-preprocessing.py "dataset" "compressor"
```
For example:
```
>>> python scripts/benchmark-preprocessing.py aind-np1 blosc-zstd
```
will run the preprocessing benchmark for the `blosc-zstd` codec on the `aind-np1` dataset.
Alternatively, arguments can be specified with a JSON file containing the same keys in the `../data` folder.

The script produces the `results/benchmark-lossless-preprocessing.csv` file.


## Lossy

For lossy compression, the **Bit truncation** and **WavPack Hybrid** strategies are tested.
Both strategies accept a lossy *factor*, which is the number of bits truncated for **Bit truncation** and the target number
of bits per samples (*bps*) for **WavPack Hybrid**. By default, the following factors are tested:

- Bit truncation: [0, 1, 2, 3, 4, 5, 6, 7]
- WavPack Hybrid: [0, 6, 5, 4, 3.5, 3, 2.5, 2.25]


### `scripts/benchmark-lossy-sim.py`
 
Benchmark lossy compression on simulated data. Additional metrics are computed, including the RMS error (RMSE), spike sorting performance 
(using [Kilosort2.5](https://github.com/MouseLand/Kilosort/tree/c31df11de9a4235c22a20909884f467c3813a2e4)), and errors on waveform features (peak-to-valley duration, half-width duration, peak-to-trough ratio).

The script accepts 3 arguments: 
```
>>> python scripts/benchmark-lossy-sim.py "dataset" "strategy" "factor"
```
For example:
```
>>> python scripts/benchmark-lossy-sim.py NP2 wavpack 3
```
will run the lossy benchmark for the **WavPack Hybrid** strategy with *bps*=3 on the `NP1` simulated dataset.

The script produces the `results/benchmark-lossy-sim.csv` and `results/benchmark-lossy-sim-waveforms.csv` files and several folders 
containing sorting and waveforms for the different strategies and datasets.



### `scripts/benchmark-lossy-exp.py`
 
Benchmark lossy compression on experimental data (only half of the datasets are used by default). 
For each dataset, in addition to RMSE, spike sorting is run and the number of units passing or failing quality control (isi violation ratio < 0.5, presence ratio > 0.9, amplitude cutoff < 0,1) is computed.
The spike sorting results are also compared against the lossless result, and the agreements (or *accuracies*) are computed.

The script accepts 4 arguments: 
```
>>> python scripts/benchmark-lossy-exp.py "dataset" "strategy" "factor" "num_runs"
```
For example:
```
>>> python scripts/benchmark-lossy-exp.py aind-np1 bit_truncation 4 1
```
will run the lossy benchmark for the **Bit truncation** with 4 bits truncated on the `aind-np1` datasets. The `num_runs` allows 
to run multiple spike sorting runs on the same condition.

The script produces the `results/benchmark-lossy-exp.csv` file and several folders containing sorting results (`results/sortings`), 
5-s decompressed recordings saved in binary format and useful for visualization (`results/compressed_recordings`), and comparison results (`results/comparisons` and `results/accuracies`).


# Installation

The `requirements.txt` contains all the packages needed to run the benchmarks. It is recommended to install them in a virtual or conda enviroment with:

```
pip install -r requirements.txt
```

The `scripts/prepare_data_for_compression.py` additionally requires:
- boto3
- s3fs
