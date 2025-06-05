import argparse
import os
import csv
from pathlib import Path
from time import sleep

import srt2csv
import srt2audio
import correct_times
import wavs2wav
import ffmpeg_commands

import vocabular

import librosa
import soundfile as sf
import numpy as np
import shutil
import demucs.separate
from wavs2wav import convert_mono_to_stereo, normalize_stereo_audio






def get_speakers_from_folder(voice_folder):
    speakers = {}
    default_speaker_name = ""
    for snd_file in Path(voice_folder).glob("*.wav"):
        snd_file_name = snd_file.stem
        if default_speaker_name == "":
            default_speaker_name = snd_file_name
        speakers[snd_file_name] = {}
        speakers[snd_file_name]["ref_file"] = snd_file
        text_file_path = snd_file.with_suffix('.txt')
        if text_file_path.is_file():
            with open(text_file_path) as text_file:
                ref_text = text_file.read().strip()
                speakers[snd_file_name]["ref_text"] = ref_text
        
        speeds_file = Path(voice_folder) / Path(snd_file).stem / "speeds.csv"
        if speeds_file.is_file():
            with open(speeds_file) as sf:
                csv_reader = csv.DictReader(sf)
                speeds = []
                durations = []
                symbol_durations = []
                for row in csv_reader:
                    speeds.append(float(row["speed"]))
                    durations.append(float(row["duration"]))
                    symbol_durations.append(float(row["symbol_duration"]))
                speakers[snd_file_name]["speeds"] = speeds
                speakers[snd_file_name]["durations"] = durations
                speakers[snd_file_name]["symbol_durations"] = symbol_durations

    speakers["default_speaker_name"] = default_speaker_name
    speakers["speakers_names"] = list(speakers.keys())
    return speakers

def check_texts(voice_dir):
    for sound_file in Path(voice_dir).glob("*.wav"):
        text_file_path = sound_file.with_suffix(".txt")
        if not text_file_path.is_file():
            print(f"I need text file {text_file_path}")
            exit(1)
    print("All text files are OK!")

def check_speeds_csv(voice_dir):
    for sound_file in Path(voice_dir).glob("*.wav"):
        text_file_path = sound_file.with_suffix(".txt")
        with open(text_file_path) as text_file:
            text = text_file.read().strip()

        speeds_file = Path(voice_dir) / Path(sound_file).stem / "speeds.csv"
        if not speeds_file.is_file():
            srt2audio.F5TTS().generate_speeds_csv(speeds_file, text, sound_file)
    print("All speeds.csv are OK!")

def check_vocabular(voice_dir):
    vocabular_pth = Path(voice_dir) / "vocabular.txt"
    if vocabular_pth.is_file():
        return vocabular_pth
    else:
        print(f"I need vocabulary file {vocabular_pth}")
        exit(1)
    print(f"Vocabulary file is {vocabular_pth}.")

def make_video_from(video_path, subtitle, speakers, default_speaker, vocabular_pth, coef):
    directory = subtitle.with_suffix("")
    directory.mkdir(exist_ok=True)
    subtitle_name = subtitle.stem
    out_path = directory / f'{subtitle_name}_0_mod.srt'
    
    # Modify subtitles using the vocabulary
    if not out_path.exists():
        vocabular.modify_subtitles_with_vocabular_wholefile(subtitle, vocabular_pth, out_path)
    
    # Convert the modified subtitles to CSV
    srt_csv_file = directory / f'{subtitle_name}_1.0_srt.csv'
    if not srt_csv_file.exists():
        srt2csv.srt_to_csv(out_path, srt_csv_file)
    
       # Add column with speaker for audio
    output_csv_with_speakers = directory / f'{subtitle_name}_1.5_output_speakers.csv'
    if not output_csv_with_speakers.exists():
        srt2csv.add_speaker_columns(srt_csv_file, output_csv_with_speakers)
    

    output_with_preview_speeds_csv = directory / f'{subtitle_name}_3.0_output_speed.csv'
    if not output_with_preview_speeds_csv.exists():
        srt2csv.add_speed_columns_with_speakers(output_csv_with_speakers, speakers, output_with_preview_speeds_csv)
    
    if  not srt2audio.F5TTS.all_segments_in_folder_check(output_with_preview_speeds_csv,directory):
        f5tts = srt2audio.F5TTS()
        # Generate audio from CSV with varying speeds based on `speed_tts_closest`
        default_speaker_name = speakers["default_speaker_name"]
        default_speaker = speakers[default_speaker_name]
        f5tts.generate_from_csv_with_speakers(output_with_preview_speeds_csv, directory, speakers, default_speaker, rewrite=False)
    
        # Lets get correct time srt csv
    corrected_time_output_speed_csv = directory / f"{subtitle_name}_4_corrected_output_speed.csv"
    if not corrected_time_output_speed_csv.exists():
        correct_times.correct_end_times_in_csv(directory, output_with_preview_speeds_csv, corrected_time_output_speed_csv)

        # Make audiotrack
    output_audio_file = directory / f"{subtitle_name}_5.0_output_audiotrack_eng.wav"
    if not output_audio_file.exists():
        wavs2wav.collect_full_audiotrack(directory, corrected_time_output_speed_csv, output_audio_file)
    
    stereo_eng_file = directory / f"{subtitle_name}_5.3_stereo_eng.wav"
    if not stereo_eng_file.exists():
        convert_mono_to_stereo(output_audio_file, stereo_eng_file)
        

    if os.path.exists(video_path):
        out_ukr_wav = directory / f"{subtitle_name}_5.5_out_ukr.wav"
        if not out_ukr_wav.exists():
            # Save the audio output as a WAV file
            command = ffmpeg_commands.extract_audio(video_path, out_ukr_wav) 
            ffmpeg_commands.run(command)
        acomponiment = directory / f"{subtitle_name}_5.7_accompaniment_ukr.wav"
        model_demucs = "mdx_extra"
        model_folder = directory / model_demucs
        demucs_folder = model_folder / out_ukr_wav.stem
        acomponiment_temp = demucs_folder / "no_vocals.wav"

        if not acomponiment_temp.exists():
            demucs.separate.main(["--jobs", "4","-o", str(directory), "--two-stems", "vocals", "-n", model_demucs, str(out_ukr_wav)])
        if acomponiment_temp.exists():    
            shutil.move(demucs_folder /"no_vocals.wav", acomponiment)
            shutil.rmtree(model_folder)
            normalize_stereo_audio(acomponiment, acomponiment)
        # Verify the accompaniment exists and is valid
        if not acomponiment.exists():
            raise FileNotFoundError(f"Failed to extract accompaniment: {acomponiment}")
        output_audio = directory / f"{subtitle_name}_6_out_reduced_ukr.wav"
        if not output_audio.exists():
            volume_intervals = ffmpeg_commands.parse_volume_intervals(srt_csv_file)
            normalize_stereo_audio(out_ukr_wav, out_ukr_wav)
            ffmpeg_commands.adjust_stereo_volume_with_librosa(out_ukr_wav, output_audio, volume_intervals, coef,acomponiment)

        # Make mix
        # mix_video = os.path.join(directory, f"{subtitle_name}_7_out_mix.mp4")
        mix_video = directory.parent / f"{subtitle_name}_out_mix.mp4"
        if not mix_video.exists():
            command = ffmpeg_commands.create_ffmpeg_mix_video_file_command(video_path,output_audio, stereo_eng_file, mix_video)
            # command = " ".join(command)
            ffmpeg_commands.run(command)

def fast_rglob(root_dir, extension):
    ext = extension.lstrip('.')
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(dirpath,dirnames,filenames)
        for file in filenames:
            if file.endswith(f".{ext}"):
                matches.append(os.path.join(dirpath, file))
    return matches



def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Script that processes a subtitle file")

    # Add a folder argument
    parser.add_argument('--subtitle', type=str, help="Path to the subtitle folder to be processed",
                        default=r"records")
    # Add default tts speeds file
    parser.add_argument('--speeds', type=str, help="Path to the speeds of tts",
                        default="speeds.csv")
    # Add delay to think that must be only one sentences, default value very very low
    parser.add_argument('--delay', type=float, help="Delay to think that must be only one sentences",
                        default=0.00001) 
    # Add voice
    parser.add_argument('--voice', type=str, help="Path to voice", default="basic_ref_en.wav")
    # Add text
    parser.add_argument('--text', type=str, help="Path to text for voice", default="some call me nature, others call me mother nature.")
    # Add voice coeficient
    parser.add_argument('--coef', type=float, help="Voice coeficient", default=0.2)
    # Add video extension
    parser.add_argument('--videoext', type=str, help="Video extension of video files", default=".mp4")
    # Add subtitles extension
    parser.add_argument('--srtext', type=str, help="Subtitle extension of files", default=".srt")
    # Add out video ending
    parser.add_argument('--outfileending', type=str, help="Out video file ending", default="_out_mix.mp4")
    # Add vocabular
    parser.add_argument('--vocabular', type=str, help="Vocabular of transcriptions", default="vocabular.txt")
    # Add config
    parser.add_argument('--config', "-c", type=str, help="Config file", default="basic.toml")

    # Parse the arguments
    args = parser.parse_args()

    # Extract the folder argument
    subtitle = args.subtitle
    speeds = args.speeds
    delay = args.delay
    voice = args.voice
    text = args.text
    coef = args.coef
    videoext = args.videoext
    srtext = args.srtext
    outfileending = args.outfileending
    vocabular = args.vocabular

    print(f"Processing folder: {subtitle}")

    voice_dir = Path(subtitle)/"VOICE"

    vocabular_pth = check_vocabular(voice_dir)
    check_texts(voice_dir)
    check_speeds_csv(voice_dir)

    speakers = get_speakers_from_folder(voice_dir)
    if not speakers:
        print("I need at least one speaker.")
        exit(1)
    default_speaker = speakers.get(speakers["default_speaker_name"])

    sbt_files = list(Path(subtitle).rglob(f"*.{srtext.replace('.', '')}"))
    #sbt_files = fast_rglob(subtitle, srtext)

    # we need exclude srt modified files that we used for right pronunciation
    sbt_files = [sbt for sbt in sbt_files if not sbt.name.endswith("_0_mod.srt")]

    for subtitle in sbt_files:
        video_path = subtitle.with_suffix(videoext)
        ready_video_file_name = subtitle.stem + "_out_mix.mp4"
        ready_video_path = video_path.parent / ready_video_file_name
        if video_path.is_file() and not ready_video_path.is_file():
            make_video_from(video_path,subtitle,speakers,default_speaker,vocabular_pth,coef)

    sbt_files = fast_rglob(subtitle, srtext)
    # we need exclude srt modified files that we used for right pronunciation
    sbt_files = [sbt for sbt in sbt_files if not sbt.name.endswith("_0_mod.srt")]

    for subtitle in sbt_files:
        video_path = subtitle.with_suffix(videoext)
        ready_video_file_name = subtitle.stem + "_out_mix.mp4"
        ready_video_path = video_path.parent / ready_video_file_name
        if video_path.is_file() and not ready_video_path.is_file():
            make_video_from(video_path, subtitle, speakers, default_speaker, vocabular_pth, coef)



if __name__ == "__main__":
    main()


