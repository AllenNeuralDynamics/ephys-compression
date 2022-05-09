from matplotlib.pyplot import streamplot
import numpy as np
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import subprocess
import json
from utils import get_random_string

import spikeinterface as si

from spikeinterface.core.job_tools import ensure_n_jobs, ChunkRecordingExecutor
from spikeinterface.core.core_tools import check_json


__version__ = "0.0.1"

_max_channels_per_stream = {
    "flac": 2,
    "mp3": 2,
    "aac": 2,
    "wavpack": 192 #192 # splitin half! 256
}


try:
    import pyflac
    from pyflac.encoder import _Encoder, EncoderInitException
    from pyflac.decoder import _Decoder, DecoderInitException, DecoderProcessException, DecoderState

    from pyflac._encoder import ffi as e_ffi
    from pyflac._encoder import lib as e_lib

    from pyflac._decoder import ffi as d_ffi
    from pyflac._decoder import lib as d_lib
except ImportError:
    print("pyFLAC not available: some functions will not run!")
    
try:
    import av
except ImportError:
    print("pyav not available: some functions will not run!")


# for wavpack we need to go through a tmp wav file
import scipy.io.wavfile as wavfile
    

class FlacFromDataEncoder(_Encoder):
    """
    The pyFLAC file encoder reads the raw audio data from the WAV file and
    writes the encoded audio data to a FLAC file.

    Args:
        data (numpy.ndarray): the data to encode (n_samples x 2)
        sample_rate (int): the sample rate
        output_file (pathlib.Path): Path to the output FLAC file, a temporary
            file will be created if unspecified.
        compression_level (int): The compression level parameter that
            varies from 0 (fastest) to 8 (slowest). The default setting
            is 5, see https://en.wikipedia.org/wiki/FLAC for more details.
        blocksize (int): The size of the block to be returned in the
            callback. The default is 0 which allows libFLAC to determine
            the best block size.
        streamable_subset (bool): Whether to use the streamable subset for encoding.
            If true the encoder will check settings for compatibility. If false,
            the settings may take advantage of the full range that the format allows.
        verify (bool): If `True`, the encoder will verify it's own
            encoded output by feeding it through an internal decoder and
            comparing the original signal against the decoded signal.
            If a mismatch occurs, the `process` method will raise a
            `EncoderProcessException`.  Note that this will slow the
            encoding process by the extra time required for decoding and comparison.

    Raises:
        ValueError: If any invalid values are passed in to the constructor.
    """
    def __init__(self,
                 data,
                 output_file,
                 sample_rate=48000,
                 compression_level: int = 5,
                 blocksize: int = 0,
                 streamable_subset: bool = True,
                 verify: bool = False):
        cformat = "flac"
        assert data.shape[1] <= _max_channels_per_stream[cformat]
        super().__init__()

        self.__raw_audio = data
        self.__output_file = output_file

        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._compression_level = compression_level
        self._streamable_subset = streamable_subset
        self._verify = verify

    def _init(self):
        """
        Initialise the encoder to write to a file.

        Raises:
            EncoderInitException: if initialisation fails.
        """
        c_output_filename = e_ffi.new('char[]', str(self.__output_file).encode('utf-8'))
        rc = e_lib.FLAC__stream_encoder_init_file(
            self._encoder,
            c_output_filename,
            e_lib._progress_callback,
            self._encoder_handle,
        )
        e_ffi.release(c_output_filename)
        if rc != e_lib.FLAC__STREAM_ENCODER_INIT_STATUS_OK:
            raise EncoderInitException(rc)

        self._initialised = True

    def process(self) -> bytes:
        """
        Process the audio data from the WAV file.

        Returns:
            (bytes): The FLAC encoded bytes.

        Raises:
            EncoderProcessException: if an error occurs when processing the samples
        """
        super().process(self.__raw_audio)
        self.finish()
        # self.total_bytes = os.path.getsize(self.__output_file) 


class FlacFromDataDecoder(_Decoder):
    """
    The pyFLAC file decoder reads the encoded audio data directly from a FLAC
    file and writes to a WAV file.

    Args:
        input_file (pathlib.Path): Path to the input FLAC file
        output_file (pathlib.Path): Path to the output WAV file, a temporary
            file will be created if unspecified.

    Raises:
        DecoderInitException: If initialisation of the decoder fails
    """
    def __init__(self,
                 input_file):
        super().__init__()

        self.write_callback = self._write_callback
        self.total_samples = 0
        self.decoded_data = None
        
        c_input_filename = d_ffi.new('char[]', str(input_file).encode('utf-8'))
        rc = d_lib.FLAC__stream_decoder_init_file(
            self._decoder,
            c_input_filename,
            d_lib._write_callback,
            d_ffi.NULL,
            d_lib._error_callback,
            self._decoder_handle,
        )
        d_ffi.release(c_input_filename)
        if rc != d_lib.FLAC__STREAM_DECODER_INIT_STATUS_OK:
            raise DecoderInitException(rc)

    def process(self):
        """
        Process the audio data from the FLAC file.

        Returns:
            (tuple): A tuple of the decoded numpy audio array, and the sample rate of the audio data.

        Raises:
            DecoderProcessException: if any fatal read, write, or memory allocation
                error occurred (meaning decoding must stop)
        """
        result = d_lib.FLAC__stream_decoder_process_until_end_of_stream(self._decoder)
        if self.state != DecoderState.END_OF_STREAM and not result:
            raise DecoderProcessException(str(self.state))

        self.finish()


    def _write_callback(self, data: np.ndarray, sample_rate: int, num_channels: int, num_samples: int):
        """
        Internal callback to write the decoded data to a WAV file.
        """
        if self.decoded_data is None:
            self.decoded_data = data
        else:
            self.decoded_data = np.vstack((self.decoded_data, data))
        self.total_samples += num_samples


class AvFromDataEncoder:
    def __init__(self,
                 data,
                 output_file,
                 sample_rate,
                 rate=48000
                 ):
        cformat = Path(output_file).suffix[1:]
        assert data.shape[1] <= _max_channels_per_stream[cformat]
        self._data = data
        self._output_file = Path(output_file)
        self._sample_rate = sample_rate
        self._rate = rate
        self.total_bytes = 0
        
    def process(self):
        output_container = av.open(str(self._output_file), 'w')
        output_stream = output_container.add_stream(self._output_file.suffix[1:], rate=self._rate)
        
        data_stream = self._data.T
        frame = av.audio.AudioFrame.from_ndarray(np.ascontiguousarray(data_stream), 
                                                 format="s16p", layout="stereo")
        frame.sample_rate = self._rate
        frame.rate = self._rate
        
        frame.pts = 0
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

        for packet in output_stream.encode(None):
            output_container.mux(packet)

        output_container.close()
        # self.total_bytes = os.path.getsize(self._output_file)
        

class AvFromDataDecoder:
    def __init__(self,
                 input_file):
        self._input_file = Path(input_file)
        self.decoded_data = None

    def process(self):
        input_container = av.open(str(self._input_file), 'r')
        input_stream = input_container.decode(audio=0)        
        num_samples = 0
        output_data = None
        for frame in input_stream:
            data = frame.to_ndarray()

            if output_data is None:
                output_data = data
            else:
                output_data = np.hstack((output_data, data))
            num_samples += data.shape[1]
        self.decoded_data = output_data.T


class WavPackFromDataEncoder:
    def __init__(self,
                 data,
                 output_file,
                 sample_rate,
                 compression_mode="f",
                 ):
        cformat = "wavpack"
        assert data.shape[1] <= _max_channels_per_stream[cformat]
        self._data = data
        self._output_file = Path(output_file)
        self._sample_rate = sample_rate
        assert compression_mode in ["f", "h", "hh"]
        self._cmode = compression_mode
        self.total_bytes = 0
        
    def process(self):
        # approach: convert data to tmp wav file --> convert to wv --> rm wav
        tmp_wav_file = self._output_file.parent / f"tmp{get_random_string(10)}.wav"
            
        # print(f"Writing {tmp_wav_file}")
        wavfile.write(tmp_wav_file, int(self._sample_rate), self._data)    
        
        # print(f"Converting {self._output_file} - {self._lossless} - {self._hybrid_n}")
        cmd = ["wavpack", "-y", "-i", "-d", f"-{self._cmode}", str(tmp_wav_file), "-o", str(self._output_file)]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # rm tmp wav file
        if tmp_wav_file.is_file():
            tmp_wav_file.unlink()

        # self.total_bytes = os.path.getsize(self._output_file)
        

class WavPackFromDataDecoder:
    def __init__(self,
                 input_file):
        self._input_file = Path(input_file)
        self.decoded_data = None

    def process(self):
        # convert to tmp wav file
        tmp_wav_file = self._input_file.parent / f"tmp.wav"

        # print(f"Converting {self._input_file}")
        subprocess.run(["wvunpack", "-y", str(self._input_file), "-o", str(tmp_wav_file)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # load wav in memory
        rate, data = wavfile.read(tmp_wav_file)
        
        # rm tmp wav file
        tmp_wav_file.unlink()

        self.decoded_data = data



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