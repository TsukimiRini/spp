import numpy as np
import noisereduce as nr

from .phase import ProcessPhase, InputFormat
from typing import List

class NoiseReducer(ProcessPhase):
    def __init__(self):
        super().__init__("noise_reducer", InputFormat.WAVEFORM)
    
    def process_waveform(self, waveform:List):
        for i in range(len(waveform)):
            y, sr = waveform[i]
            waveform[i] = (nr.reduce_noise(y=y, sr=sr, stationary=True), sr)
            
        return waveform