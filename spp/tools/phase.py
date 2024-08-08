import os
import sys
import numpy as np
from enum import Enum
from typing import List

class InputFormat(Enum):
    PATH = 1
    WAVEFORM = 2

class ProcessPhase:
    def __init__(self, label:str, input_format:InputFormat):
        self.label = label
        self.input_format = input_format
    
    def process(self, obj, **kwargs):
        if self.input_format == InputFormat.PATH:
            return self.process_path(obj)
        elif self.input_format == InputFormat.WAVEFORM:
            return self.process_waveform(obj, **kwargs)
    
    def process_path(self, path:List[str]):
        raise NotImplementedError
    
    def process_waveform(self, waveform:List[np.ndarray]):
        raise NotImplementedError
    
