from .phase import ProcessPhase, InputFormat, OutputFormat
import numpy as np
import librosa

class AudioTrimmer(ProcessPhase):
    def __init__(self):
        super().__init__("audio_trimmer", InputFormat.WAVEFORM, OutputFormat.WAV_PATH)
    
    def process_waveform(self, waveform):
        for i in range(len(waveform)):
            y, sr = waveform[i]
            y, _ = librosa.effects.trim(y)
            
            waveform[i] = (y, sr)
        return waveform