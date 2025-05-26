from datetime import datetime
import csv
import os
import soundfile as sf
import numpy as np

def collect_full_audiotrack(fragments_folder, csv_file, output_audio_file):
    """Concatenate all audio segments in the specified order from csv_file into a full audio track, using start times to add silence."""
    audio_segments = []
    sample_rate = None

    def time_to_seconds(time_str):
        """Convert timestamp string to seconds, with enhanced error handling."""
        time_formats = ['%H:%M:%S,%f', '%H:%M:%S.%f']  # Try both comma and dot for milliseconds
        for time_format in time_formats:
            try:
                dt = datetime.strptime(time_str.strip(), time_format)
                return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            except ValueError:
                continue
        raise ValueError(f"Invalid time format: '{time_str}'. Expected format is 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm'.")

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


if __name__ == "__main__":
    # Example usage
    output_folder = "2_chkd"
    csv_file = "corrected_output_speed.csv"
    output_audio_file = "output_audiotrack.wav"

    collect_full_audiotrack(output_folder, csv_file, output_audio_file)
