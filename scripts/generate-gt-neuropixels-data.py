"""
Generate MEArec synthetic for compression benchmark.

This script generates MEArec templates and recordings for NP1 and NP2 as recorded by the Open Ephys GUI.

In particular, the simulated data mimic the experimental data in terms of:

- Dtype: int16 (both NP1 and NP2)
- Least Significant Bit (LSB): 
   * 12 for NP1
   * 3 for NP2 
- Gain to uV: 0.195 (both NP1 and NP2)

**Note** that the LSB > 1 is only for Open Ephys data and not for SpikeGLX.
"""

from pathlib import Path

import MEArec as mr

NP_VERSION = 2


mearec_folder = Path("mearec/")
mearec_data_folder = Path("../data/mearec")
mearec_data_folder.mkdir(exist_ok=True)

if NP_VERSION == 1:
    template_file = mearec_folder / "templates_drift_Neuropixels-384.h5"
else:
    template_file = mearec_folder / "templates_drift_Neuropixels2-384.h5"
tempgen = mr.load_templates(template_file)

recordings_params = mr.get_default_recordings_params()
recordings_params["recordings"]
# NP1.0 settings
recordings_params["recordings"]["dtype"] = "int16"
recordings_params["recordings"]["filter"] = False

recordings_params["recordings"]["adc_bit_depth"] = 10
recordings_params["recordings"]["gain"] = 0.195
recordings_params["recordings"]["dtype"] = "int16"
recordings_params["recordings"]["chunk_duration"] = 10
recordings_params["recordings"]["noise_mode"] = "distance-correlated"


recordings_params["spiketrains"]["n_exc"] = 80
recordings_params["spiketrains"]["n_inh"] = 20
recordings_params["spiketrains"]["duration"] = 600
if NP_VERSION == 1:
    recordings_params["recordings"]["lsb"] = 12
else:
    recordings_params["recordings"]["lsb"] = 3
recordings_params["recordings"]["lsb"]
recgen = mr.gen_recordings(
    params=recordings_params,
    tempgen=tempgen,
    n_jobs=10,
    verbose=True,
)


rec_name = f"mearec_NP{NP_VERSION}.h5"

mr.save_recording_generator(recgen, mearec_folder / rec_name)
