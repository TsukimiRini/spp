from .phase import ProcessPhase, InputFormat, OutputFormat
from pydub import AudioSegment
from pydub.silence import split_on_silence

import librosa
import os

class SpeechSplitter(ProcessPhase):
    def __init__(self):
        super().__init__("speech_splitter", InputFormat.PATH, OutputFormat.WAV_PATH)
    
    def process_path(self, paths):
        obj = []
        override_paths = []
        for path in paths:
            audio = AudioSegment.from_file(path)
            chunks = split_on_silence(audio, min_silence_len=800, silence_thresh=-60)
            for i in range(len(chunks)):
                new_path = f"{os.path.splitext(path)[0]}_{i}.wav"
                chunks[i].export(new_path, format="wav")
                y, sr = librosa.load(new_path)
                obj.append((y, sr))
                override_paths.append(new_path)
        return obj, override_paths