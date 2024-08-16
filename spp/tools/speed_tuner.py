from pyloudnorm import Meter
import pyloudnorm as pyln
from .phase import ProcessPhase, InputFormat
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from collections import Counter
import os
import librosa
import torch
import ffmpeg

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class SpeedTuner(ProcessPhase):
    def __init__(self):
        super().__init__("speed_tuner", InputFormat.PATH)
    
    def process_path(self, paths, speed_ratio):
        obj = []
        for path in paths:
            tmp = path.replace(".wav", "_to_tune.wav")
            os.rename(path, tmp)
            (
                ffmpeg
                .input(tmp)
                .output(path, **{'filter:a': f'atempo={speed_ratio}'})
                .run()
            )
            y, sr = librosa.load(path)
            obj.append((y, sr))
        return obj
            