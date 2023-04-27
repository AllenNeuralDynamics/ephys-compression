"""
Download and prepare data for extended compression.

Datasets are from 3 different sources:

IBL:
- 4 NP1 datasets (from Olivier Winter)

AIND:
- 8 NP2 datasets
- 4 NP1 datasets


**MINDSCOPE:
- 4 NP1 datasets (publicly available)

To ensure a fair comparison in terms of compression time (not affected by how the data is stored),
all data are loaded and saved to binary by SI.

The final datasets are uploaded in the "s3://aind-ephys-compression-benchmark-data".
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import s3fs
import spikeinterface as si
import spikeinterface.extractors as se

sys.path.append("..")

from utils import s3_download_folder, s3_download_public_folder, s3_upload_folder

ephys_compression_folder_path = Path(__file__).parent.parent

"""
SET YOUR LOCAL CACHE DIRECTORY HERE
"""
ROOT_PATH = Path("/home/alessio/Documents/data/compression")
N_JOBS = 20

job_kwargs = dict(n_jobs=N_JOBS, chunk_duration="1s", progress_bar=True)

compression_bucket_path = "aind-ephys-compression-benchmark-data"
compression_bucket = f"s3://{compression_bucket_path}"
delete_tmp_files_as_created = True

print(job_kwargs)


fs = s3fs.S3FileSystem()


# temporary folder to download temporary data
tmp_folder = ROOT_PATH / "tmp"
tmp_folder.mkdir(exist_ok=True)
output_folder = ROOT_PATH / "compression_benchmark"
output_folder.mkdir(exist_ok=True)

### AIND NP2
print("\n\n\nAIND-NP2\n\n\n")
aind_ephys_bucket = f"s3://aind-ephys-data/"
dataset = "aind-np2"

aind_np2_sessions = {
    "595262_2022-02-21_15-18-07": {"probe": "ProbeA"},
    "602454_2022-03-22_16-30-03": {"probe": "ProbeB"},
    "612962_2022-04-13_19-18-04": {"probe": "ProbeB"},
    "612962_2022-04-14_17-17-10": {"probe": "ProbeC"},
    "618197_2022-06-21_14-08-06": {"probe": "ProbeC"},
    "618318_2022-04-13_14-59-07": {"probe": "ProbeB"},
    "618384_2022-04-14_15-11-00": {"probe": "ProbeB"},
    "621362_2022-07-14_11-19-36": {"probe": "ProbeA"},
}


for session, session_data in aind_np2_sessions.items():
    print(session)
    session_folder = output_folder / dataset / f"{session}_{session_data['probe']}"
    remote_location = f"{compression_bucket_path}/{dataset}/{session}_{session_data['probe']}"

    process_and_upload_session = False
    if f"{compression_bucket_path}/{dataset}" not in fs.ls(f"{compression_bucket_path}"):
        process_and_upload_session = True
    elif remote_location not in fs.ls(f"{compression_bucket_path}/{dataset}"):
        process_and_upload_session = True

    if process_and_upload_session:
        # save output to binary
        if not session_folder.is_dir():
            dest = tmp_folder / dataset
            oe_folder = dest / session
            print(f"Syncing {session}")
            s3_download_folder(aind_ephys_bucket, f"ecephys_{session}/ecephys", oe_folder)

            # clean (keep experiment1 only)
            record_node_folder = [p for p in oe_folder.iterdir() if "Record" in p.name][0]
            # check if multiple experiments
            experiments = [p for p in record_node_folder.iterdir() if "experiment" in p.name]
            settings = [p for p in record_node_folder.iterdir() if "settings" in p.name]
            if len(experiments) > 1:
                for exp in experiments:
                    if exp.name != "experiment1":
                        print(f"Removing {exp.name}")
                        shutil.rmtree(exp)
                for sett in settings:
                    if sett.name != "settings.xml":
                        print(f"Removing {sett.name}")
                        sett.unlink()

            # streams
            stream_names, stream_ids = se.get_neo_streams("openephys", oe_folder)
            stream_name = [stream_name for stream_name in stream_names if session_data["probe"] in stream_name][0]
            stream_id = stream_ids[stream_names.index(stream_name)]

            # load recording
            recording = se.read_openephys(oe_folder, stream_id=stream_id)

            # find longest segment
            if recording.get_num_segments() > 1:
                segment_lengths = []
                for segment_index in range(recording.get_num_segments()):
                    segment_lengths.append(recording.get_num_samples(segment_index=segment_index))
                longest_segment = np.argmax(segment_lengths)
                recording = recording.select_segments([longest_segment])
            print(recording)
            print(recording.get_channel_locations()[:4])
            recording.save(folder=session_folder, **job_kwargs)

            s3_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

            if delete_tmp_files_as_created:
                print(f"Deleting tmp folder {oe_folder}")
                shutil.rmtree(oe_folder)
                print(f"Deleting SI folder {session_folder}")
                shutil.rmtree(session_folder)

        else:
            print(f"Loading binary")
            recording = si.load_extractor(session_folder)
            print(recording)
    else:
        print(f"{dataset}/{session} already in remote bucket")


### AIND NP1
print("\n\n\nAIND-NP1\n\n\n")
aind_ephys_bucket = f"s3://aind-ephys-data/"
dataset = "aind-np1"

aind_np1_sessions = {
    "625749_2022-08-03_15-15-06": {"probe": "ProbeA"},
    "634568_2022-08-05_15-59-46": {"probe": "ProbeA"},
    "634569_2022-08-09_16-14-38": {"probe": "ProbeA"},
    "634571_2022-08-04_14-27-05": {"probe": "ProbeA"},
}


for session, session_data in aind_np1_sessions.items():
    print(session)
    session_folder = output_folder / dataset / f"{session}_{session_data['probe']}"
    remote_location = f"{compression_bucket_path}/{dataset}/{session}_{session_data['probe']}"

    process_and_upload_session = False
    if f"{compression_bucket_path}/{dataset}" not in fs.ls(f"{compression_bucket_path}"):
        process_and_upload_session = True
    elif remote_location not in fs.ls(f"{compression_bucket_path}/{dataset}"):
        process_and_upload_session = True

    if process_and_upload_session:
        # save output to binary
        if not session_folder.is_dir():
            dest = tmp_folder / dataset
            oe_folder = dest / session
            print(f"Syncing {session}")
            s3_download_folder(aind_ephys_bucket, f"ecephys_{session}/ecephys", oe_folder)

            # clean (keep experiment1 only)
            record_node_folder = [p for p in oe_folder.iterdir() if "Record" in p.name][0]
            # check if multiple experiments
            experiments = [p for p in record_node_folder.iterdir() if "experiment" in p.name]
            settings = [p for p in record_node_folder.iterdir() if "settings" in p.name]
            if len(experiments) > 1:
                for exp in experiments:
                    if exp.name != "experiment1":
                        print(f"Removing {exp.name}")
                        shutil.rmtree(exp)
                for sett in settings:
                    if sett.name != "settings.xml":
                        print(f"Removing {sett.name}")
                        sett.unlink()

            # streams
            stream_names, stream_ids = se.get_neo_streams("openephys", oe_folder)
            stream_name = [
                stream_name
                for stream_name in stream_names
                if session_data["probe"] in stream_name and "AP" in stream_name
            ][0]
            stream_id = stream_ids[stream_names.index(stream_name)]

            # load recording
            recording = se.read_openephys(oe_folder, stream_id=stream_id)

            # find longest segment
            if recording.get_num_segments() > 1:
                segment_lengths = []
                for segment_index in range(recording.get_num_segments()):
                    segment_lengths.append(recording.get_num_samples(segment_index=segment_index))
                longest_segment = np.argmax(segment_lengths)
                recording = recording.select_segments([longest_segment])
            print(recording)
            print(recording.get_channel_locations()[:4])
            recording.save(folder=session_folder, **job_kwargs)

            s3_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

            if delete_tmp_files_as_created:
                print(f"Deleting tmp folder {oe_folder}")
                shutil.rmtree(oe_folder)
                print(f"Deleting SI folder {session_folder}")
                shutil.rmtree(session_folder)

        else:
            print(f"Loading binary")
            recording = si.load_extractor(session_folder)
            print(recording)
    else:
        print(f"{dataset}/{session} already in remote bucket")


# ### IBL (AWS)
print("\n\n\nIBL\n\n\n")
s3_bucket_ibl = "ibl-brain-wide-map-public"
region_name_ibl = "us-east-1"
prefix = "spikesorting/benchmark"
skip_patterns = ".lf."
dataset = "ibl-np1"

ibl_sessions = {
    "CSHZAD026_2020-09-04_probe00": "CSH_ZAD_026/2020-09-04/001/raw_ephys_data/probe00",
    "CSHZAD029_2020-09-09_probe00": "CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00",
    "SWC054_2020-10-05_probe00": "SWC_054/2020-10-05/001/raw_ephys_data/probe00",
    "SWC054_2020-10-05_probe01": "SWC_054/2020-10-05/001/raw_ephys_data/probe01",
}

for session, session_path in ibl_sessions.items():
    print(session)

    session_folder = output_folder / dataset / session
    remote_location = f"{compression_bucket_path}/{dataset}/{session}"

    process_and_upload_session = False

    if f"{compression_bucket_path}/{dataset}" not in fs.ls(f"{compression_bucket_path}"):
        process_and_upload_session = True
    elif remote_location not in fs.ls(f"{compression_bucket_path}/{dataset}"):
        process_and_upload_session = True

    if process_and_upload_session:
        if not session_folder.is_dir():
            dest = tmp_folder / dataset

            s3_download_public_folder(
                f"{prefix}/{session_path}",
                dest / session,
                s3_bucket_ibl,
                region_name_ibl,
                skip_patterns=skip_patterns,
            )

            cbin_folder = dest / session

            recording = se.read_cbin_ibl(cbin_folder)
            print(recording)

            # save output to binary
            recording.save(folder=session_folder, **job_kwargs)

            s3_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

            if delete_tmp_files_as_created:
                print(f"Deleting tmp folder {cbin_folder}")
                shutil.rmtree(cbin_folder)
                print(f"Deleting SI folder {session_folder}")
                shutil.rmtree(session_folder)
        else:
            print(f"Loading binary")
            recording = si.load_extractor(session_folder)
            print(recording)
    else:
        print(f"{dataset}/{session} already in remote bucket")
