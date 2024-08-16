from pyloudnorm import Meter
import pyloudnorm as pyln
from .phase import ProcessPhase, InputFormat
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from collections import Counter
import os
import librosa
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class TailTrimmer(ProcessPhase):
    def __init__(self, model_name="openai/whisper-large-v3", language=None):
        super().__init__("tail_trimmer", InputFormat.WAVEFORM)
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
        self.language = language
        
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

    
    def process_waveform(self, waveform, expected_texts):
        for i in range(len(waveform)):
            y, sr = waveform[i]
            expected_text = expected_texts[i]
            if self.language:
                result = self.pipe({
                    "array": y,
                    "sampling_rate": sr
                }, generate_kwargs={"language": self.language})
            else:
                result = self.pipe({
                    "array": y,
                    "sampling_rate": sr,
                })
            expected_words = expected_text.split()
            expected_words = [self.trim_word(word) for word in expected_words]
            expected_words = [word.lower() for word in expected_words if len(word) > 0]
            generated_words = [self.trim_word(chunk["text"]).lower() for chunk in result["chunks"]]
            
            tail_idx = len(generated_words) - 1
            print(f"gen:{generated_words}")
            print(f"exp:{expected_words}")
            cur_jaccard = self.bag_of_words_sim(expected_words, generated_words)
            while True:
                tailed_generated_words = generated_words[:-1]
                tailed_jaccard = self.bag_of_words_sim(expected_words, tailed_generated_words)
                if tailed_jaccard > cur_jaccard:
                    cur_jaccard = tailed_jaccard
                    generated_words = tailed_generated_words
                    tail_idx -= 1
                else:
                    break
            
            print(generated_words)
            last_timestamp = result["chunks"][tail_idx]["timestamp"][-1]
            print(len(y) / sr - last_timestamp)
            if tail_idx == len(generated_words) - 1 and len(y) / sr - last_timestamp < 0.5:
                continue
            last_timestamp += 0.1
            waveform[i] = (y[:int(last_timestamp * sr)], sr)
        return waveform