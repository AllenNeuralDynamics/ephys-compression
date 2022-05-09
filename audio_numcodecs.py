"""
Numcodecs Codec implementation for AUDIO codecs:

Lossless:
    * FLAC
    * WavPack
Lossy:
    * MP3
    * WavPack Hybrid
    
The (sub-optimal) approach is:
- for compression: to convert to the audio file and read it as the encoded bytes
- for decompression: dump the encoded data to a tmp file and decode it using the codec

Multi-channel data exceeding the number of channels that can be encoded by the codec are reshaped to fit the 
compression procedure.

"""
from pathlib import Path
from math import gcd
import numpy as np
import subprocess

import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy

from utils import get_random_string

# length of random string for tmp files
RND_LEN = 10

_max_channels_per_stream = {
    "flac": 2,
    "mp3": 2,
    "aac": 2,
    "wavpack": 256 
}

_format_to_ext = {
    "flac": ".flac",
    "mp3": ".mp3",
    "aac": ".aac",
    "wavpack": ".wv"
}


##### LOW-LEVEL Classes #####
try:
    import pyflac
    from pyflac.encoder import _Encoder, EncoderInitException
    from pyflac.decoder import _Decoder, DecoderInitException, DecoderProcessException, DecoderState

    from pyflac._encoder import ffi as e_ffi
    from pyflac._encoder import lib as e_lib

    from pyflac._decoder import ffi as d_ffi
    from pyflac._decoder import lib as d_lib
    HAVE_PYFLAC = True
except ImportError:
    HAVE_PYFLAC = False
    
try:
    import av
    HAVE_AV = True
    
except ImportError:
    HAVE_AV = False


# for wavpack we need to go through a tmp wav file
import scipy.io.wavfile as wavfile
    

class FlacNumpyEncoder(_Encoder):
    """
    The pyFLAC data encoder converts data from np.array to a FLAC file.

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


class FlacNumpyDecoder(_Decoder):
    """
    The pyFLAC file decoder reads the encoded audio data directly from a FLAC
    file and returns a numpy array.

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


class AvNumpyEncoder:
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
        

class AvNumpyDecoder:
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


class WavPackNumpyEncoder:
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
        

class WavPackNumpyDecoder:
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


##### NUMCODECS CODECS ######

_encoder_classes = {
    "flac": FlacNumpyEncoder,
    "mp3": AvNumpyEncoder,
    "aac": AvNumpyEncoder,
    "wavpack": WavPackNumpyEncoder
}

_decoder_classes = {
    "flac": FlacNumpyDecoder,
    "mp3": AvNumpyDecoder,
    "aac": AvNumpyDecoder,
    "wavpack": WavPackNumpyDecoder
}


class BaseAudioCodec(Codec):    
    codec_id = ""
    
    def __init__(self, cformat, tmp_folder=None, flatten=False, **kwargs):
        if tmp_folder is None:
            tmp_folder = Path("tmp")
        else:
            tmp_folder = Path(tmp_folder)
        tmp_folder.mkdir(exist_ok=True, parents=True)
        self._tmp_folder = tmp_folder
        self._format = cformat
        self._encoder_class = _encoder_classes[cformat]
        self._decoder_class = _decoder_classes[cformat]
        self._flatten = flatten
        self._kwargs = kwargs
        
    def encode(self, buf):
        # checks
        assert len(buf.shape) <= _max_channels_per_stream[self._format]
        assert buf.dtype.kind in ["i", "u"]
        if buf.ndim == 1:
            data = buf[:, None]
        else:
            nsamples, nchannels = buf.shape
            
            if nchannels > _max_channels_per_stream[self._format]:
                if self._flatten:
                    data = buf.flatten()[:, None]    
                else:
                    # reshape C order with the GCD bewteen num channels and max channels
                    nenc_channels = gcd(nchannels, _max_channels_per_stream[self._format])

                    data = buf.reshape((int(nsamples * (nchannels // nenc_channels + np.mod(nchannels, nenc_channels))), 
                                        nenc_channels), order="C")
            else:
                data = buf        
        
        tmp_file = self._tmp_folder / f"{get_random_string(RND_LEN)}{_format_to_ext[self._format]}"
        encoder = self._encoder_class(data, tmp_file, **self._kwargs)
        encoder.process()
        
        with tmp_file.open("rb") as f:
            enc = f.read()
        # delete tmp file
        tmp_file.unlink()
        
        return enc

    def decode(self, buf, out=None):        
        tmp_file = self._tmp_folder / f"{get_random_string(RND_LEN)}{_format_to_ext[self._format]}"
        
        with tmp_file.open("wb") as f:
            f.write(buf)
            
        decoder = self._decoder_class(tmp_file)
        decoder.process()
        
        dec = decoder.decoded_data
        
        # handle output
        out = ndarray_copy(dec, out)
        
        # delete tmp flac file
        tmp_file.unlink()
        
        return out


class FlacCodec(BaseAudioCodec):
    """Codec for FLAC (Free Lossless Audio Codec).
    
    The implementation uses [pyFlac]()
    

    Parameters
    ----------
    tmp_folder : _type_, optional
        _description_, by default None
    flatten : bool, optional
        _description_, by default False
    compression_level : int, optional
        _description_, by default 5
    """
    codec_id = "flac"
    
    def __init__(self, tmp_folder=None, flatten=False, compression_level=5):
        BaseAudioCodec.__init__(self, "flac", tmp_folder=tmp_folder, flatten=flatten,
                                compression_level=compression_level, sample_rate=48000)
        self.compression_level = compression_level
        
    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            tmp_folder=str(self._tmp_folder),
            flatten=self._flatten,
            compression_level=self.compression_level,
        )

numcodecs.register_codec(FlacCodec)


class WavPackCodec(BaseAudioCodec):
    
    codec_id = "wavpack"
    
    def __init__(self, tmp_folder=None, flatten=True, lossless=True):
        BaseAudioCodec.__init__(self, "wavpack", tmp_folder=tmp_folder, flatten=flatten,
                                compression_mode="f", sample_rate=48000)
        self.lossless = lossless    
        
    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            tmp_folder=str(self._tmp_folder),
            flatten=self._flatten,
            lossless=self.lossless
        )

numcodecs.register_codec(WavPackCodec)

