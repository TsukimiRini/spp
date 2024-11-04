import os
import sys
import numpy as np
from enum import Enum
from typing import List

class InputFormat(Enum):
    PATH = 1
    WAVEFORM = 2
    
class OutputFormat(Enum):
    WAV_PATH = 1
    TEXT = 2

class ProcessPhase:
    def __init__(self, label:str, input_format:InputFormat, output_format:OutputFormat):
        self.label = label
        self.input_format = input_format
        self.output_format = output_format
    
    def process(self, obj, **kwargs):
        if self.input_format == InputFormat.PATH:
            return self.process_path(obj, **kwargs)
        elif self.input_format == InputFormat.WAVEFORM:
            return self.process_waveform(obj, **kwargs)
    
    def process_path(self, path:List[str]):
        raise NotImplementedError
    
    def process_waveform(self, waveform:List[np.ndarray]):
        raise NotImplementedError
    
