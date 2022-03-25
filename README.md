# ephys-compression

Sandbox to investigate compression strategies for ephys datasets.

The goal of this effort is to be able to compress efficiently Neuropixels ephys data after acquisition.

The compressed dataset should:
    - have a CR > 3 (something around 8-10 would be desirable)
    - be fast to write and to read
    - allow lazy access of the data (ideally interfacing directly with SpikeInterface)
    - not affect spike-sorting results
    - not affect waveform shapes and features

Several strategies methods to investigate:
    - [zarr + blosc](https://zarr.readthedocs.io/en/v2.11.1/index.html): lossless + lossy (e.g. with bit truncation)
    - [MED](https://medformat.org/)
    - audio compression (e.g. via [pyFLAC](https://github.com/sonos/pyFLAC))
