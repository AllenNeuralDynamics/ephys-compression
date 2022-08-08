"""
Download and prepare data for extended compression.

Datasets are from 3 different sources:

IBL: 4 NP1 datasets (from Olivier Winter)

MINDSCOPE: 4 NP1 datasets (publicly available)

AIND: 8 NP2 datasets

To ensure a fair comparison in terms of compression time (not affected by how the data is stored),
all data are loaded and saved to binary by SI.

The final datasets are uploaded in the "gs://aind-ephys-compression-benchmark-data" bucket.
"""

from pathlib import Path

import boto3
from botocore.config import Config
from botocore import UNSIGNED
import os
import shutil
import numpy as np

import spikeinterface.full as si
import probeinterface as pi
import neo

import gcsfs


ephys_compression_folder_path = Path(__file__).parent

"""
SET YOUR LOCAL DIRECTORY HERE
"""
ROOT_PATH = Path("/home/alessio/Documents/data/compression")

job_kwargs = dict(n_jobs=20, chunk_duration="1s", progress_bar=True)

compression_bucket_path = "aind-ephys-compression-benchmark-data"
compression_bucket = f"gs://{compression_bucket_path}"
delete_tmp_files_as_created = True


fs = gcsfs.GCSFileSystem()

### AWS ###


def get_s3_client(region_name):
    bc = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name=region_name)
    return bc


def s3_download_public_file(object, destination, bucket, region_name):
    """
    downloads file from public bucket
    :param object: relative path of file" 'atlas/dorsal_cortex_50.nrrd'
    :param destination: full file path on local machine '/usr/ibl/dorsal_cortex_50.nrrd'
    :param bucket: if not specified, 'ibl-brain-wide-map-public'
    :return:
    """
    destination = Path(destination)
    boto_client = get_s3_client(region_name)
    destination.mkdir(parents=True, exist_ok=True)
    object_name = object.split("/")[-1]
    boto_client.download_file(bucket, object, str(destination / object_name))


def s3_download_public_folder(remote_folder, destination, bucket, region_name, skip_patterns=None,
                              overwrite=False, verbose=True):
    """
    downloads a public folder content to a local folder
    :param prefix: relative path within the bucket, for example: 'spikesorting/benchmark'
    :param destination: local folder path
    :param bucket: if not specified, 'ibl-brain-wide-map-public'
    :param boto_client: if not specified, will instantiate one in anonymous mode
    :return:
    """
    boto_client = get_s3_client(region_name)
    response = boto_client.list_objects_v2(Prefix=remote_folder, Bucket=bucket)

    if skip_patterns is not None:
        if isinstance(skip_patterns, str):
            skip_patterns = [skip_patterns]

    for item in response.get('Contents', []):
        object = item['Key']
        if object.endswith('/') and item['Size'] == 0:  # skips  folder
            continue
        local_file_path = Path(destination).joinpath(Path(object).relative_to(remote_folder))
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        skip = False
        if any(sp in object for sp in skip_patterns):
            skip = True

        if not overwrite and local_file_path.exists() and local_file_path.stat().st_size == item['Size'] or skip:
            if verbose:
                print(f"skipping {local_file_path}")
        else:
            if verbose:
                print(f"downloading {local_file_path}")
            boto_client.download_file(bucket, object, str(local_file_path))

# GCS
def gs_download_folder(bucket, remote_folder, destination):
    dst = Path(destination)
    if not dst.is_dir():
        dst.mkdir()

    if not bucket.endswith("/"):
        bucket += "/"
    src = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m cp -r {src} {dst}")


def gs_upload_folder(bucket, remote_folder, local_folder):
    if not bucket.endswith("/"):
        bucket += "/"
    dst = f"{bucket}{remote_folder}"

    os.system(f"gsutil -m rsync -r {local_folder} {dst}")


def get_oe_stream(oe_folder):
    # we have to first access the different streams (i.e., different probes)
    io = neo.rawio.OpenEphysBinaryRawIO(oe_folder)
    io._parse_header()
    streams = io.header['signal_streams']

    return streams


# temporary folder to download temporary data
tmp_folder = ROOT_PATH / "tmp"
tmp_folder.mkdir(exist_ok=True)
output_folder = ROOT_PATH / "compression_benchmark"
output_folder.mkdir(exist_ok=True)

### AIND (GCS)
print("\n\n\nAIND\n\n\n")
aind_ephys_bucket = "gs://aind-ephys-data/"
dataset = "aind"

aind_sessions = {"595262_2022-02-21_15-18-07": {"probe": "ProbeA"}, 
                 "618197_2022-06-21_14-08-06": {"probe": "ProbeC"},
                 "621362_2022-07-14_11-19-36": {"probe": "ProbeA"},
                 "602454_2022-03-22_16-30-03": {"probe": "ProbeB"},
                 "618382_2022-03-31_14-27-03": {"probe": "ProbeC"}, 
                 "618318_2022-04-13_14-59-07": {"probe": "ProbeB"},
                 "612962_2022-04-13_19-18-04": {"probe": "ProbeB"},
                 "613373_2022-04-26_15-36-12": {"probe": "ProbeC"},
                 "618384_2022-04-14_15-11-00": {"probe": "ProbeB"},
                 "612962_2022-04-14_17-17-10": {"probe": "ProbeC"},}

for session, session_data in aind_sessions.items():
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
            if not oe_folder.is_dir():
                print(f"Downloading {session}")
                gs_download_folder(aind_ephys_bucket, session, dest)

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
            streams = get_oe_stream(oe_folder)
            stream_name = [stream_name for stream_name in streams["name"] if session_data["probe"] in stream_name][0]
            stream_id = streams["id"][list(streams["name"]).index(stream_name)]

            # load recording
            recording = si.read_openephys(oe_folder, stream_id=stream_id)

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

            gs_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

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
s3_bucket_ibl = 'ibl-brain-wide-map-public'
region_name_ibl = 'us-east-1'
prefix = 'spikesorting/benchmark'
skip_patterns = ".lf."
dataset = "ibl"

ibl_sessions = {"CSHZAD026_2020-09-04_probe00": "CSH_ZAD_026/2020-09-04/001/raw_ephys_data/probe00",
                "CSHZAD029_2020-09-09_probe00": "CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00",
                "SWC054_2020-10-05_probe01": "SWC_054/2020-10-05/001/raw_ephys_data/probe00"}

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

            s3_download_public_folder(f"{prefix}/{session_path}", dest / session, s3_bucket_ibl, region_name_ibl,
                                    skip_patterns=skip_patterns)

            cbin_folder = dest / session

            recording = si.read_cbin_ibl(cbin_folder)
            print(recording)

            # save output to binary
            recording.save(folder=session_folder, **job_kwargs)

            gs_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

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


# ### MINDSCOPE (AWS)
print("\n\n\nMINDSCOPE\n\n\n")
s3_bucket_mindscope = 'allen-brain-observatory'
region_name_mindscope = 'us-west-2'
prefix = 'visual-coding-neuropixels/raw-data'
dataset = "mindscope"

mindscope_sessions = {
    "754312389_756781559": "754312389/756781559/spike_band.dat",
    "766640955_773592324": "766640955/773592324/spike_band.dat",
    "797828357_805579745": "797828357/805579745/spike_band.dat",
    "829720705_832129157": "829720705/832129157/spike_band.dat",}
#     "766640955/773592324/spike_band.dat",
#     "797828357/805579745/spike_band.dat",
#     "829720705/832129157/spike_band.dat",
# ]

binary_dict = dict(num_chan=384, sampling_frequency=30000, dtype="int16", gain_to_uV=0.195, offset_to_uV=0)

for session, session_path in mindscope_sessions.items():
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
            dat_file_name = session_path.split("/")[-1]
            dest_file = dest / session / dat_file_name
            if not dest_file.is_file():
                s3_download_public_file(f"{prefix}/{session_path}", dest, 
                                        s3_bucket_mindscope, region_name_mindscope)

            recording = si.read_binary(dest_file, **binary_dict)
            print(recording)

            # load NP1 probe config probe and save    
            probegroup = pi.read_probeinterface(ephys_compression_folder_path / "probes" / "NP1_tip_config.json")
            recording = recording.set_probegroup(probegroup)

            # save output to binary
            recording.save(folder=session_folder, **job_kwargs)

            gs_upload_folder(compression_bucket, f"{dataset}/{session_folder.name}", session_folder)

            if delete_tmp_files_as_created:
                print(f"Deleting tmp folder {dest_file.parent}")
                shutil.rmtree(dest_file.parent)
                print(f"Deleting SI folder {session_folder}")
                shutil.rmtree(session_folder)
        else:
            print(f"Loading binary")
            recording = si.load_extractor(session_folder)
            print(recording)
    else:
        print(f"{dataset}/{session} already in remote bucket")