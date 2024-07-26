import numpy as np

from .phase import ProcessPhase, InputFormat
from typing import List
from scipy import signal

class FrequencyFilter(ProcessPhase):
    def __init__(self):
        super().__init__("frequency_filter", InputFormat.WAVEFORM)
    
    def process_waveform(self, waveform:List):
        # filter out noise for speech audio with low pass filter
        for i in range(len(waveform)):
            y, sr = waveform[i]
            # create low pass filter
            nyquist = sr / 2
            cutoff = 4000
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
            waveform[i] = (signal.lfilter(b, a, y), sr)
            
        return waveform