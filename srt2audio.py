import os
import csv
import random
import sys
import soundfile as sf
import torch
import tqdm
from cached_path import cached_path
from pathlib import Path
from omegaconf import OmegaConf
from importlib.resources import files
from hydra.utils import get_class
import librosa
# from transformers.utils.dummy_pt_objects import Wav2Vec2BertModel
# from transformers import Wav2Vec2BertModel
from wav2txt import wav2txt
import re
import srt2csv
import difflib

COUNTER_MAX = 10
REMOVE_SILENCE_TOP_DB = 35
# DEFAULT_SPEED = 1.3

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    # remove_silence_edges,
    # save_spectrogram,
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(self, model_type="F5-TTS", ckpt_file="", vocab_file="", ode_method="euler",
                 use_ema=True, vocoder_name="vocos", local_path=None, device=None):
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"device = {self.device}")

        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        # Use correct model name
        model = "F5TTS_v1_Base"
        
        # Load model configuration
        model_cfg = OmegaConf.load(
            str(files("f5_tts").joinpath(f"configs/{model}.yaml"))
        )
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        
        # Set correct checkpoint path
        if not ckpt_file:
            if mel_spec_type == "vocos":
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            elif mel_spec_type == "bigvgan":
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"))
        
        # Load the model with correct parameters
        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, 
            mel_spec_type=mel_spec_type, 
            vocab_file=vocab_file, 
            device=self.device
        )

        # For older models
        if model_type == "F5-TTS":
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=None):
        sf.write(file_wave, wav, self.target_sample_rate)

    def infer(self, ref_file, ref_text, gen_text, show_info=print, progress=tqdm, target_rms=0.1,
              cross_fade_duration=0.15, sway_sampling_coef=-1, cfg_strength=2, nfe_step=32, speed=1.0,
              fix_duration=None, 
              remove_silence=True, # to start from start
              file_wave=None, seed=-1):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)#, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if remove_silence:
            trimmed, index = librosa.effects.trim(wav, top_db=REMOVE_SILENCE_TOP_DB)
            print(f"Trimmed from {file_wave} from sample {index[0]} to {index[1]}")
            wav = trimmed
            

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        return wav, sr

    @staticmethod
    def all_segments_in_folder_check(csv_file:str, folder:str):
        """
        Checks if all fragments specified in the CSV file are present in the given folder.

        Args:
            csv_file (str): Path to the CSV file containing the fragment details.
            folder (str): Path to the folder where the fragments should be located.
        """
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            missing_files = []
            for i, _ in enumerate(reader):
                expected_file = f"segment_{i + 1}.wav"
                if not os.path.exists(os.path.join(folder, expected_file)):
                    missing_files.append(expected_file)

        if not missing_files:
            print("All fragments are present in the folder.")
            return True
        else:
            print(f"Missing fragments: {len(missing_files)} missing files.")
            for missing_file in missing_files:
                print(f"Missing file: {missing_file}")
            return False

    @staticmethod
    def linear_predict(speed_1, duration_1, speed_2, duration_2, limit_duration):
        """
        Performs linear extrapolation to predict the speed that would result in the desired duration.

        Args:
            speed_1 (float): The first speed value.
            duration_1 (float): The duration corresponding to the first speed value.
            speed_2 (float): The second speed value.
            duration_2 (float): The duration corresponding to the second speed value.
            limit_duration (float): The target duration for which we want to predict the speed.

        Returns:
            float: The predicted speed value that should result in the target duration.
        """
        if duration_1 == duration_2:
            return speed_1  # If durations are equal, return the first speed (arbitrary choice)

        # Linear extrapolation formula: speed = speed_1 + (limit_duration - duration_1) * (speed_2 - speed_1) / (duration_2 - duration_1)
        predicted_speed = speed_1 + (limit_duration - duration_1) * (speed_2 - speed_1) / (duration_2 - duration_1)
        return predicted_speed

    def infer_wav(self, gen_text, speed, ref_file, ref_text, file_wave=None):
        wav, sr = self.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            show_info=print,
            progress=tqdm,
            fix_duration=None,
            file_wave=file_wave
        )
        return wav, sr, len(wav) / sr 

    def generate_wav_if_longer(self, wav, sr, gen_text, duration, previous_duration, previous_speed, ref_file, ref_text, i):
        # if len(gen_text.split()) < MIN_NUMBER_OF_WORDS:
        #     # previous_speed = DEFAULT_SPEED
        #     wav, sr, previous_duration = self.infer_wav(gen_text, previous_speed, ref_file, ref_text,file_wave=f"segment_{i}_speed_{previous_speed}.wav")
        #     return wav,sr, previous_duration
        counter = 0
        start_speed = previous_speed + 0.1
        while duration < previous_duration:  
            print(f"duration < duration_seconds_tts = {duration} < {previous_duration}")
            next_speed = previous_speed + 0.1
            wav, sr,next_duration = self.infer_wav(gen_text, next_speed, ref_file, ref_text)

            predict_linear_speed = self.linear_predict(previous_speed, previous_duration, next_speed, next_duration, duration)
            if predict_linear_speed-previous_speed > 0.1:  # if jump is less then 0.1 speed make speed just +0.1speed
                next_speed = predict_linear_speed
                print(f"Let`s regenerate {i}-fragment with speed = {next_speed}")
                wav, sr = self.infer(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    speed=next_speed,
                    fix_duration=None,
                    # file_wave=f"segment_{i}_speed_{next_speed}.wav" # for debug
                )
            previous_duration = len(wav) / sr
            previous_speed = next_speed
            counter += 1
            if counter > COUNTER_MAX:
                wav, sr,previous_duration = self.infer_wav(gen_text, start_speed, ref_file, ref_text)
                break
        return wav, sr, previous_duration

    def clean_text(self,text,replacement = r"[.,!?\-:;’'\"]"):
        text = text.strip().lower().replace("\n", " ")
        text = re.sub(replacement, " ", text)
        text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
        return text.strip()   

    def similarity(self,gen_text,subtitles_text):
        return difflib.SequenceMatcher(None, gen_text, subtitles_text).ratio()


    def is_generated_text_equal_to_subtitles_text(self,wav,subtitles_text):
        gen_text = wav2txt(wav)
        gen_text = self.clean_text(gen_text)
        subtitles_text = self.clean_text(subtitles_text)
        similarity = self.similarity(gen_text,subtitles_text)
        if gen_text != subtitles_text:
            print(f"ALARM !!! Generated text: {gen_text} != Subtitles text: {subtitles_text} \n Similarity: {similarity}")
        return gen_text == subtitles_text,gen_text,subtitles_text,similarity 

    def generate_from_csv_with_speakers(self, csv_file, output_folder, speakers, default_speaker, rewrite=False):
        os.makedirs(output_folder, exist_ok=True)
        filename_errors_csv = f"{str(csv_file)[:-4]}_errors.csv"
        with open(csv_file, 'r', encoding='utf-8') as csvfile, \
             open(filename_errors_csv, 'w', newline='', encoding='utf-8') as csv_writer:
            reader = csv.DictReader(csvfile)
            writer_filednames = [*reader.fieldnames, "similarity","gen_error","whisper_text","subtitle_text"]
            writer = csv.DictWriter(csv_writer, fieldnames=writer_filednames, delimiter=';')
            writer.writeheader()
            generated_segments = []
            generated_texts = []
            for i, row in enumerate(reader):
                file_wave = os.path.join(output_folder, f"segment_{i + 1}.wav")
                if not rewrite and os.path.exists(file_wave):
                    continue
                duration = float(row['Duration'])
                gen_text = row['Text']
                generated_texts.append(gen_text)
                previous_speed = float(row.get('TTS Speed Closest', 1.0))  # Read the speed from `speed_tts_closest`, default to 1.0 if missing

                try:
                    speaker_name = row['Speaker']
                    ref_text = speakers[speaker_name]["ref_text"]
                    ref_file = speakers[speaker_name]["ref_file"]
                except:
                    print("Something is wrong. Let's take default speaker")
                    ref_text = default_speaker["ref_text"]
                    ref_file = default_speaker["ref_file"]

                file_wave_debug = None # f"segment_{i}_speed_{previous_speed}.wav" # for debug
                wav, sr, previous_duration = self.infer_wav(gen_text, previous_speed, ref_file, ref_text,file_wave=file_wave_debug)
                
                wav, sr, previous_duration = self.generate_wav_if_longer(wav, sr, gen_text, duration, previous_duration, previous_speed, ref_file, ref_text, i)

                print(f"Generated WAV-{i} with symbol duration {previous_duration}")        
                generated_segments.append((wav, file_wave, sr)) 
                is_equal,gen_text,subtitles_text, similarity = self.is_generated_text_equal_to_subtitles_text(wav,gen_text)
                writer.writerow({**row, "similarity": f"{similarity:.2f}", "gen_error": "1" if not is_equal else "0", "whisper_text": gen_text, "subtitle_text": subtitles_text})
            
            # Reset to the beginning of the file
            csvfile.seek(0)
            # Re-create the DictReader to re-parse the header row
            reader = csv.DictReader(csvfile)      
            
            for i, row in enumerate(reader):
                wav, file_wave, sr = generated_segments[i]
                generated_texts.append(row['Text'])
                sf.write(file_wave, wav, sr)
                print(f"Saved WAV as {file_wave}")
        excel_file_name = os.path.basename(filename_errors_csv)
        excel_file_name = excel_file_name.split("_3.0_")[0] + ".xlsx"
        parent_of_parent = os.path.dirname(os.path.dirname(filename_errors_csv))
        excel_file = os.path.join(parent_of_parent, excel_file_name)
        srt2csv.csv2excel(filename_errors_csv, excel_file)                    
        print(f"All audio segments generated and saved in {output_folder}")



    def generate_speeds_csv(self, output_csv, ref_text, ref_file):
        gen_text = "Some call me nature, others call me mother nature. Let's try some long text. We are just trying to get more fidelity. It's OK!"
        speeds = [0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        rows = []
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        for speed in speeds:

            file_name = Path(output_csv).parent/f"gen_out_{speed}.wav"


            wav, sr = self.infer(ref_file, ref_text, gen_text, speed=speed,fix_duration=None, remove_silence=True,  file_wave=file_name)
            duration = len(wav) / sr
            symbol_duration = duration / len(gen_text)  # Assuming each character is considered a symbol

            rows.append([speed, duration, symbol_duration, file_name])

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['speed', 'duration', 'symbol_duration', 'file_name'])
            writer.writerows(rows)
        print(f"CSV file generated and saved as {output_csv}")

