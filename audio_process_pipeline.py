import librosa
import os
import sys
from tools import ProcessPhase, InputFormat, NoiseReducer, FrequencyFilter, AudioTrimmer, SilenceRemover, SpeechSplitter, VolumeNormalizer
from typing import List
import soundfile as sf

phase_map = {
    "noise_reducer": NoiseReducer,
    "frequency_filter": FrequencyFilter,
    "audio_trimmer": AudioTrimmer,
    "silence_remover": SilenceRemover,
    "speech_splitter": SpeechSplitter,
    "volume_normalizer": VolumeNormalizer
}

class AudioProcessPipeline:
    def __init__(self, phase_list:List[str]):
        self.phase_list = phase_list
        self.phase_funcs = []
        for phase in phase_list:
            self.phase_funcs.append(phase_map[phase]())
    
    def process(self, paths, output_dir):
        obj = []
        for path in paths:
            assert os.path.exists(path)
            y, sr = librosa.load(path)
            obj.append((y, sr))
        for phase in self.phase_funcs:
            if phase.input_format == InputFormat.PATH:
                temp_paths = []
                for i in range(len(obj)):
                    y, sr = obj[i]
                    output_path = os.path.join(output_dir, os.path.basename(paths[i]))
                    temp_paths.append(output_path)
                    sf.write(output_path, y, sr)
                obj = phase.process(temp_paths)
            elif phase.input_format == InputFormat.WAVEFORM:
                obj = phase.process(obj)
            
            if isinstance(obj, tuple):
                obj, paths = obj
            
        for i in range(len(obj)):
            y, sr = obj[i]
            output_path = os.path.join(output_dir, os.path.basename(paths[i]))
            sf.write(output_path, y, sr)
        return obj