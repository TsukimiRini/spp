from pyloudnorm import Meter
import pyloudnorm as pyln
from .phase import ProcessPhase, InputFormat
import os
import librosa

class VolumeNormalizer(ProcessPhase):
    def __init__(self, desired_loudness=-18.0):
        super().__init__("volume_normalization", InputFormat.WAVEFORM)
        self.desired_loudness = desired_loudness
    
    def process_waveform(self, waveform):
        for i in range(len(waveform)):
            y, sr = waveform[i]
            meter = Meter(sr)
            loudness = meter.integrated_loudness(y)
            loudness_normalized_audio = pyln.normalize.loudness(y, loudness, self.desired_loudness)
            waveform[i] = (loudness_normalized_audio, sr)
        return waveform