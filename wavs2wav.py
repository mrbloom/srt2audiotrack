from datetime import datetime
import csv
import os
import soundfile as sf
import numpy as np
from correct_times import time_to_seconds   
import librosa
import demucs.separate
import shutil

def extract_acomponiment_or_vocals(directory, subtitle_name, out_ukr_wav,
        pipeline_suffix="_extracted.wav",
        model_demucs = "mdx_extra",
        sound_name="no_vocals.wav"
    ):
    acomponiment = directory / f"{subtitle_name}{pipeline_suffix}"    
    model_folder = directory / model_demucs
    demucs_folder = model_folder / out_ukr_wav.stem
    acomponiment_temp = demucs_folder / sound_name

    demucs.separate.main(["--jobs", "4","-o", str(directory), "--two-stems", "vocals", "-n", model_demucs, str(out_ukr_wav)])
    
    if acomponiment_temp.exists():
        shutil.move(demucs_folder / sound_name, acomponiment)
        shutil.rmtree(model_folder)

    # Verify the accompaniment exists and is valid
    if not acomponiment.exists():
        raise FileNotFoundError(f"Failed to extract accompaniment: {acomponiment}")
    return acomponiment
        
    

def collect_full_audiotrack(fragments_folder, csv_file, output_audio_file):
    """Concatenate all audio segments in the specified order from csv_file into a full audio track, using start times to add silence."""
    audio_segments = []
    sample_rate = None

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        previous_end_time = 0.0  # Start at 0 for initial silence

        for i, row in enumerate(reader):
            start_time_str = row.get('Start Time')

            if start_time_str is None:
                print(f"Warning: 'Start Time' missing in row {i + 1}. Skipping segment.")
                continue

            try:
                start_time = time_to_seconds(start_time_str)
            except ValueError as e:
                print(f"Error in row {i + 1}: {e}. Skipping segment.")
                continue

            segment_file = os.path.join(fragments_folder, f"segment_{i + 1}.wav")

            if os.path.exists(segment_file):
                wav, sr = sf.read(segment_file)

                # Ensure sample rate consistency
                if sample_rate is None:
                    sample_rate = sr
                elif sample_rate != sr:
                    raise ValueError(f"Sample rate mismatch in segment {segment_file}")

                # Calculate silence duration (in samples) needed before this segment
                if start_time > previous_end_time:
                    silence_duration = int((start_time - previous_end_time) * sample_rate)
                    silence = np.zeros(silence_duration)
                    audio_segments.append(silence)

                # Add the current segment
                audio_segments.append(wav)
                print(f"Processed start time: {start_time_str} - {previous_end_time/60}")
                # Update previous_end_time to reflect the end time of the current segment
                previous_end_time = start_time + len(wav) / sample_rate
            else:
                print(f"Warning: Expected segment {segment_file} not found.")

    # Concatenate all segments with silences
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        sf.write(output_audio_file, full_audio, sample_rate)
        print(f"Full audio track saved to {output_audio_file}")
    else:
        print("No audio segments to concatenate. Please check the input files.")

def convert_mono_to_stereo(input_path: str, output_path: str):
    # Load mono audio
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Duplicate mono channel to create stereo
    stereo_audio = np.vstack([audio, audio])

    # Save as stereo WAV
    sf.write(output_path, stereo_audio.T, sr)

    print(f"Converted {input_path} to stereo and saved as {output_path}")


def normalize_stereo_audio(input_path: str, output_path: str, target_db: float = -12.0):
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    if audio.ndim == 1:
        raise ValueError("The input file is mono. Use a mono-specific normalization function.")

    # Compute RMS loudness for each channel
    rms_left = np.sqrt(np.mean(audio[0]**2))
    rms_right = np.sqrt(np.mean(audio[1]**2))

    # Convert RMS to decibel scale
    rms_db_left = 20 * np.log10(rms_left)
    rms_db_right = 20 * np.log10(rms_right)

    # Compute gain needed for each channel
    gain_db_left = target_db - rms_db_left
    gain_db_right = target_db - rms_db_right

    gain_left = 10 ** (gain_db_left / 20)
    gain_right = 10 ** (gain_db_right / 20)

    # Apply gain to each channel separately
    normalized_audio = np.vstack([audio[0] * gain_left, audio[1] * gain_right])

    # Save the normalized stereo audio
    sf.write(output_path, normalized_audio.T, sr)

    print(f"Normalized {input_path} to {target_db} dB per channel and saved as {output_path}")
