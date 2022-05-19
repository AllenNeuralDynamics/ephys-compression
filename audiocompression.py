import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json

import spikeinterface as si

from spikeinterface.core.job_tools import ensure_n_jobs, ChunkRecordingExecutor
from spikeinterface.core.core_tools import check_json


__version__ = "0.0.1"

_max_channels_per_stream = {
    "flac": 2,
    "mp3": 2,
    "aac": 2,
    "wavpack": 1024
}

# use SI job tools for blocks!
def write_recording_audio(recording: si.BaseRecording, output_folder, cformat="flac",
                          compression_level=5, mode="stream", lossless=True,
                          overwrite=True, verbose=False,  **job_kwargs):
    """
    Write Recording with Audio compression (flac, mp3, aac)

    Parameters
    ----------
    recording : si.BaseRecording
        The input recording
    output_folder : str or Path
        The folder where the compressed data is saved
    cformat: str
        The compression format ("flac" | "mp3" | "aac")
    compression_level: int
        The compression level (0-8) for flac format
    lossless: bool
        Whether to use lossless or hybrid format for wavpack
    mode: str
        If "streams", the channels are split in streams of 2 channels and then compressed.
        If "concat", the channels are concatenated as a 2-channel signal for each block
    overwrite: bool
        If True, the output folder is overwritten (if existing)
    **job_kwargs: kwargs for parallel processing
    """
    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get('n_jobs', 1))
    assert recording.get_dtype().kind == "i"
    
    output_folder = Path(output_folder)
    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError("Folder already exists! Use overwrite True")
    output_folder.mkdir(exist_ok=True)
    
    # divide in streams
    if mode == "stream":
        max_channels_per_stream = _max_channels_per_stream[cformat]
        num_streams = int(np.ceil(recording.get_num_channels() / max_channels_per_stream))
        print(f"Num. streams: {num_streams}")
    
        # save streams to channel ids to json
        stream_dict = {}
        for i, ch in enumerate(recording.channel_ids):
            stream_idx = i // max_channels_per_stream
            if stream_idx not in stream_dict:
                stream_dict[stream_idx] = {}
                stream_dict[stream_idx]["ids"] = [ch]
                stream_dict[stream_idx]["idxs"] = [i]
            else:
                stream_dict[stream_idx]["ids"].append(ch)
                stream_dict[stream_idx]["idxs"].append(i)
    else:
        stream_dict = None
            
    info_dict = {}
    info_dict["channel_ids"] = recording.channel_ids
    info_dict["sampling_frequency"] = recording.get_sampling_frequency()
    info_dict["dtype"] = np.dtype(recording.get_dtype()).str
    info_dict["mode"] = mode
    
    if mode == "stream":
        info_dict["streams"] = stream_dict
        
    segment_samples = {}
    for segment_index in range(recording.get_num_segments()):
        segment_samples[segment_index] = recording.get_num_samples(segment_index=segment_index)
        (output_folder / f"traces_seg{segment_index}").mkdir()
    
    info_dict["segment_samples"] = segment_samples
    
    # use executor (loop or workers)
    func = _write_audio_chunk
    init_func = _init_audio_worker
    if n_jobs == 1:
        init_args = (recording, output_folder, stream_dict, cformat, compression_level, lossless, mode)
    else:
        init_args = (recording.to_dict(), output_folder, stream_dict, cformat, compression_level, lossless, mode)
    executor = ChunkRecordingExecutor(recording, func, init_func, init_args, verbose=verbose,
                                      job_name='write_audio_recording', **job_kwargs)
    executor.run()
    
    with open(output_folder / "info.json", "w") as f:
        json.dump(check_json(info_dict), f)

    rec_audio = AudioRecordingExtractor(output_folder)
    rec_audio.set_probegroup(recording.get_probegroup(), in_place=True)
    recording.copy_metadata(rec_audio)
    
    return rec_audio
    

# used by write_binary_recording + ChunkRecordingExecutor
def _init_audio_worker(recording, output_folder, stream_dict, cformat, compression_level, lossless, mode):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        worker_ctx['recording'] = load_extractor(recording)
    else:
        worker_ctx['recording'] = recording

    worker_ctx['stream_dict'] = stream_dict
    worker_ctx['mode'] = mode
    worker_ctx['output_folder'] = output_folder
    worker_ctx['compression_level'] = compression_level
    worker_ctx["lossless"] = lossless
    worker_ctx['cformat'] = cformat

    return worker_ctx


# used by write_binary_recording + ChunkRecordingExecutor
def _write_audio_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    output_folder = Path(worker_ctx['output_folder'])
    compression_level = worker_ctx['compression_level']
    mode = worker_ctx['mode']
    cformat = worker_ctx['cformat']
    lossless = worker_ctx["lossless"]
    

    if cformat == "wavpack":
        ext = "wv"
    else:
        ext = cformat

    stream_dict = worker_ctx['stream_dict']
    sample_rate = int(recording.get_sampling_frequency())
    output_chunk_folder = output_folder / f"traces_seg{segment_index}" / f"{start_frame}_{end_frame}"
    output_chunk_folder.mkdir(exist_ok=True)

    max_channels_per_stream = _max_channels_per_stream[cformat]
    
    if mode == "stream":
        for stream, sd in stream_dict.items():
            output_file = output_chunk_folder / f"{stream}.{ext}"
            # apply function
            traces = recording.get_traces(channel_ids=sd["ids"], start_frame=start_frame, 
                                          end_frame=end_frame, segment_index=segment_index)
            assert traces.shape[1] <= max_channels_per_stream
            if cformat == "flac":
                encoder = FlacFromDataEncoder(traces, output_file, sample_rate, compression_level=compression_level)
            elif cformat == "wavpack":
                # print("Wavpack", stream, output_file, traces.shape)
                encoder = WavPackFromDataEncoder(traces, output_file, sample_rate, lossless)
            else:
                encoder = AvFromDataEncoder(traces, output_file, sample_rate)
                # save mean and ptp of encoded traces
                output_json_file = output_chunk_folder / f"{stream}.json"
                gain_info = {"ptp": int(np.ptp(traces))}
                with open(output_json_file, "w") as f:
                    json.dump(gain_info, f)
            encoder.process()
            
    else:
        output_file = output_chunk_folder / f"block.{ext}"
        # apply function
        traces = recording.get_traces(start_frame=start_frame, 
                                      end_frame=end_frame, segment_index=segment_index)
        num_samples, num_channels = traces.shape
        traces_concat = traces.reshape((int(num_samples * 
                                       (num_channels // max_channels_per_stream + 
                                       np.mod(num_channels, max_channels_per_stream))), 
                                       max_channels_per_stream), order="F")
        if cformat == "flac":
            encoder = FlacFromDataEncoder(traces_concat, output_file, sample_rate, compression_level=compression_level)
        elif cformat == "wavpack":
            encoder = WavPackFromDataEncoder(traces_concat, output_file, sample_rate)
        else:
            encoder = AvFromDataEncoder(traces_concat, output_file, sample_rate)
            # save mean and ptp of encoded traces
            output_json_file = output_chunk_folder / f"block.json"
            gain_info = {"ptp": int(np.ptp(traces))}
            with open(output_json_file, "w") as f:
                json.dump(gain_info, f)
        encoder.process()
        
                
            

global _worker_ctx
global _func


class AudioRecordingExtractor(si.BaseRecording):
    def __init__(self, folder_path):
        folder_path = Path(folder_path)
        info_file =  folder_path / "info.json"
        assert info_file.is_file()
        
        info = json.load(open(info_file, "r"))
        sampling_frequency = info["sampling_frequency"]
        channel_ids = info["channel_ids"]
        dtype = info["dtype"]
        segment_samples = info["segment_samples"]
        mode = info["mode"]
        streams = info.get("streams", None)
        
        super().__init__(sampling_frequency, channel_ids, dtype)
        
        for segment_index, segment_samples in segment_samples.items():
            segment = AudioRecordingSegment(folder_path, segment_index=segment_index, num_samples=segment_samples,
                                            num_channels=len(channel_ids), streams=streams, dtype=dtype, mode=mode,
                                            times_kwargs={"sampling_frequency": sampling_frequency})
            self.add_recording_segment(segment)
            
        self._kwargs = {"folder_path": str(folder_path.absolute())}
        
        

class AudioRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, folder_path, segment_index, num_samples, num_channels, streams, dtype, mode, times_kwargs):
        si.BaseRecordingSegment.__init__(self, **times_kwargs)
        segment_index = int(segment_index)
        self._num_samples = num_samples
        self._num_channels = num_channels
        self._dtype = dtype
        self._mode = mode
        # search startframes
        start_frames = []
        end_frames = []
        folders = []
        self._folder_path = folder_path / f"traces_seg{segment_index}"
        for d in self._folder_path.iterdir():
            if d.is_dir():
                sf, ef = d.name.split("_")
                start_frames.append(int(sf))
                end_frames.append(int(ef))
                folders.append(d)
        start_frames_idxs = np.argsort(start_frames)
        self._start_frames = np.array(start_frames)[start_frames_idxs].astype(int)
        self._end_frames = np.array(end_frames)[start_frames_idxs].astype(int)
        self._folders = np.array(folders)[start_frames_idxs]
        self._streams = streams
        
        # get format
        for f in folders[0].iterdir():
            if f.suffix != ".json":
                self._ext = f.suffix
                self._cformat = self._ext[1:]
                break
        if "flac" in self._ext:
            self._decode_class = FlacFromDataDecoder
        elif "wv" in self._ext:
            self._decode_class = WavPackFromDataDecoder
        else:
            self._decode_class = AvFromDataDecoder
        

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        verbose = False
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
            
        start_folder_idx = np.searchsorted(self._start_frames, start_frame, side='right') - 1
        end_folder_idx = np.searchsorted(self._start_frames, end_frame, side='left') - 1
        
        if verbose:
            print(f"Start chunk: {start_folder_idx} - End chunk {end_folder_idx}")
            print(f"Start frame chunk: {self._start_frames[start_folder_idx]}\nEnd chunk frame chunk: {self._end_frames[end_folder_idx]}")
        
        if isinstance(channel_indices, slice):
            start = channel_indices.start
            if start is None:
                start = 0
            stop = channel_indices.stop
            if stop is None:
                stop = self._num_channels
            step = channel_indices.step
            if step is None:
                step = 1
            channel_indices = np.arange(start, stop, step)
        
        traces = np.zeros((end_frame - start_frame, len(channel_indices)), dtype=self._dtype)
        
        if self._mode == "stream":
            # map channel_indices
            channel_map = {}
            trace_idx = 0
            for ch in channel_indices:
                for stream_id, stream in self._streams.items():
                    if ch in stream["idxs"]:
                        if stream_id not in channel_map:
                            channel_map[stream_id] = {}
                        if "trace_idxs" not in channel_map[stream_id]:
                            channel_map[stream_id]["trace_idxs"] = [trace_idx]
                            channel_map[stream_id]["decoded_idxs"] = [list(stream["idxs"]).index(ch)]
                            trace_idx += 1
                        else:
                            channel_map[stream_id]["trace_idxs"].append(trace_idx)
                            channel_map[stream_id]["decoded_idxs"].append(list(stream["idxs"]).index(ch))
                            trace_idx += 1

        if start_folder_idx == end_folder_idx:
            i_chunk = start_folder_idx
            folder = self._folders[start_folder_idx]
            start_idx_decoded = start_frame - self._start_frames[start_folder_idx]
            end_idx_decoded = end_frame - self._start_frames[start_folder_idx]
            
            if self._mode == "stream":
                for stream_id, stream in channel_map.items():
                    encoded_file = folder / f"{stream_id}{self._ext}"
                    encoded_file_json = folder / f"{stream_id}.json"
                    # print(f"Decoding {stream_id} - {encoded_file} - channel idxs {stream['idxs']}")
                    decoder = self._decode_class(encoded_file)
                    decoder.process()
                    decoded_data = decoder.decoded_data
                    if encoded_file_json.is_file():
                        gain_info = json.load(open(encoded_file_json, "r"))
                        gain = gain_info["ptp"] / np.ptp(decoded_data)
                    else:
                        gain = 1
                        
                    traces[:, stream["trace_idxs"]] = gain * \
                        decoded_data[start_idx_decoded:end_idx_decoded, stream["decoded_idxs"]]
            else:
                encoded_file = folder / f"block{self._ext}"
                encoded_file_json = folder / f"block.json"
                if verbose:
                    print(f"Decoding block - {encoded_file}")
                decoder = self._decode_class(encoded_file)
                decoder.process()
                decoded_data = decoder.decoded_data
                if encoded_file_json.is_file():
                    gain_info = json.load(open(encoded_file_json, "r"))
                    gain = gain_info["ptp"] / np.ptp(decoded_data)
                else:
                    gain = 1
                decoded_data = decoded_data.reshape((self._end_frames[i_chunk] - self._start_frames[i_chunk], 
                                                    self._num_channels), order="F")
                traces = gain * \
                    decoded_data[start_idx_decoded:end_idx_decoded, channel_indices]
        else:
            for i_chunk in range(start_folder_idx, end_folder_idx + 1):
                if verbose:
                    print(f"Chunk {i_chunk}")
                folder = self._folders[i_chunk]
                if i_chunk == start_folder_idx:
                    # print("start")
                    start_idx_decoded = start_frame - self._start_frames[i_chunk]
                    end_idx_decoded = self._end_frames[i_chunk] - self._start_frames[i_chunk]
                    start_idx = 0
                    end_idx = end_idx_decoded - start_idx_decoded
                elif i_chunk == end_folder_idx:
                    # print("end")
                    start_idx_decoded = 0
                    end_idx_decoded = end_frame - self._start_frames[i_chunk]
                    start_idx = -end_idx_decoded
                    end_idx = None
                else:
                    # print("middle")
                    start_idx_decoded = 0
                    end_idx_decoded = None
                    start_idx = self._start_frames[i_chunk] - start_frame
                    end_idx = self._end_frames[i_chunk] - start_frame
                # print(start_idx_decoded, end_idx_decoded, start_idx, end_idx)
                if self._mode == "stream":
                    for stream_id, stream in channel_map.items():
                        encoded_file = folder / f"{stream_id}{self._ext}"
                        encoded_file_json = folder / f"{stream_id}.json"
                        # print(f"Decoding {stream_id} - {encoded_file} - channel idxs {stream['idxs']}")
                        decoder = self._decode_class(encoded_file)
                        decoder.process()
                        decoded_data = decoder.decoded_data
                        
                        if encoded_file_json.is_file():
                            gain_info = json.load(open(encoded_file_json, "r"))
                            gain = gain_info["ptp"] / np.ptp(decoded_data)
                        else:
                            gain = 1
                        
                        traces[start_idx:end_idx, stream["trace_idxs"]] = gain * \
                         decoded_data[start_idx_decoded:end_idx_decoded, stream["decoded_idxs"]]
                else:
                    encoded_file = folder / f"block{self._ext}"
                    encoded_file_json = folder / f"block.json"
                    if verbose:
                        print(f"Decoding block - {encoded_file}")
                    decoder = self._decode_class(encoded_file)
                    decoder.process()
                    decoded_data = decoder.decoded_data
                    if encoded_file_json.is_file():
                        gain_info = json.load(open(encoded_file_json, "r"))
                        gain = gain_info["ptp"] / np.ptp(decoded_data)
                    else:
                        gain = 1
                        
                    decoded_data = decoded_data.reshape((self._end_frames[i_chunk] - self._start_frames[i_chunk], 
                                                        self._num_channels), order="F")
                        
                    traces[start_idx:end_idx] = gain * \
                        decoded_data[start_idx_decoded:end_idx_decoded, channel_indices]
                
        return traces