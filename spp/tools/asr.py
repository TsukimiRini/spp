from pyloudnorm import Meter
import pyloudnorm as pyln
from .phase import ProcessPhase, InputFormat, OutputFormat
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from collections import Counter
import os
import librosa
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class SpeechRecognizer(ProcessPhase):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__("asr", InputFormat.WAVEFORM, OutputFormat.TEXT)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps="word",
            torch_dtype=torch_dtype,
            device=device,
        )
    
    def trim_word(self, text):
        text = text.strip()
        # trim the punctuation at the end of the text
        while len(text) > 0 and text[-1] in ".,!:;?":
            text = text[:-1].strip()
        return text.strip()
    
    def bag_of_words_sim(self, list1, list2):
        char_list1 = [char for word in list1 for char in word]
        char_list2 = [char for word in list2 for char in word]
        inter = list((Counter(char_list1) & Counter(char_list2)).elements())
        return len(inter) / max(len(char_list1), len(char_list2))

    
    def process_waveform(self, waveform, language=None):
        results = []
        to_asr = [{
            "array": y,
            "sampling_rate": sr
        } for y, sr in waveform]
        if language:
            result_list = self.pipe(to_asr, batch_size=16, generate_kwargs={"language": language})
        else:
            result_list = self.pipe(to_asr, batch_size=16)
        
        for result in result_list:
            generated_words = [chunk["text"].strip() for chunk in result["chunks"]]
            results.append(" ".join(generated_words))
        return results