from .phase import ProcessPhase, InputFormat
from pydub import AudioSegment
from pydub.silence import split_on_silence

import librosa

class SilenceRemover(ProcessPhase):
    def __init__(self):
        super().__init__("silence_remover", InputFormat.PATH)
    
    def process_path(self, paths):
        obj = []
        for path in paths:
            audio = AudioSegment.from_file(path)
            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
            audio = chunks[0]
            for i in range(1, len(chunks)):
                audio += chunks[i]
            audio.export(path, format="wav")
            y, sr = librosa.load(path)
            obj.append((y, sr))
        return obj