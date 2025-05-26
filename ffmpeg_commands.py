import csv
import subprocess
import tempfile
import os
import librosa
import soundfile as sf
import shutil


# Read CSV file to get volume reduction time intervals
def parse_volume_intervals(csv_file):
    volume_intervals = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start_time = row['Start Time']
            end_time = row['End Time']
            volume_intervals.append((start_time, end_time))
    return volume_intervals

# Format time to seconds for ffmpeg
def time_to_seconds(time_str):
    """Convert timestamp string to seconds."""
    hours, minutes, seconds = time_str.replace(",", ".").split(":")
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

# Create the ffmpeg command to reduce volume in specified intervals
def create_ffmpeg_command(input_video, output_audio):  #, volume_intervals, k_volume):
    # volume_filters = []
    #
    # for start_time, end_time in volume_intervals:
    #     start_seconds = time_to_seconds(start_time)
    #     end_seconds = time_to_seconds(end_time)
    #     # Add volume filter for each segment
    #     volume_filters.append(f"volume={k_volume}:enable='between(t,{start_seconds},{end_seconds})'")
    #
    # # Combine all volume filters
    # filter_complex = ",".join(volume_filters) + "[aout]"

    # Build the full ffmpeg command to extract audio and adjust volume, output as WAV
    ffmpeg_command = (
        f'ffmpeg -y -i "{input_video}" '
        # f'-filter_complex "[0:a]{filter_complex}" '
        # f'-map "[aout]"'
        f'-c:a pcm_s16le "{output_audio}"'
    )

    return ffmpeg_command


def time_to_seconds(time_str):
    """Convert timestamp string (HH:MM:SS,mmm) to seconds (float)."""
    hours, minutes, seconds = time_str.replace(",", ".").split(":")
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

def adjust_volume_with_librosa(input_audio, output_audio, volume_intervals, k_volume):
    """
    Adjusts the volume of an audio file using librosa.

    :param input_audio: Path to input WAV file
    :param output_audio: Path to output WAV file
    :param volume_intervals: List of tuples (start_time, end_time) where volume needs adjustment
    :param k_volume: Volume adjustment factor (e.g., 0.5 for 50% reduction, 1.5 for increase)
    """
    # Load audio file
    y, sr = librosa.load(input_audio, sr=None)

    # Convert time to sample index
    for start_time, end_time in volume_intervals:
        start_time, end_time = time_to_seconds(start_time),time_to_seconds(end_time)
        start_sample = int(librosa.time_to_samples(float(start_time), sr=sr))
        end_sample = int(librosa.time_to_samples(float(end_time), sr=sr))

        # Apply volume adjustment in the given range
        y[start_sample:end_sample] *= k_volume

    # Save the modified audio
    sf.write(output_audio, y, sr)

    print(f"Volume adjusted and saved to {output_audio}")

def adjust_stereo_volume_with_librosa(input_audio, output_audio, volume_intervals, k_volume, acomponimemt):
    """
    Adjusts the volume of a stereo audio file using librosa.

    :param input_audio: Path to input WAV file
    :param output_audio: Path to output WAV file
    :param volume_intervals: List of tuples (start_time, end_time) where volume needs adjustment
    :param k_volume: Volume adjustment factor (e.g., 0.5 for 50% reduction, 1.5 for increase)
    """
    # Load audio file with stereo channels
    y, sr = librosa.load(input_audio, sr=None, mono=False)
    a, sr = librosa.load(acomponimemt, sr=None, mono=False)

    # Convert time to sample index
    for start_time, end_time in volume_intervals:
        start_time, end_time = time_to_seconds(start_time), time_to_seconds(end_time)
        start_sample = int(librosa.time_to_samples(float(start_time), sr=sr))
        end_sample = int(librosa.time_to_samples(float(end_time), sr=sr))

        # Apply volume adjustment in the given range for both channels
        y[:, start_sample:end_sample] =  y[:, start_sample:end_sample] * k_volume + a[:, start_sample:end_sample]*(1-k_volume)*0.5

    # Save the modified audio
    sf.write(output_audio, y.T, sr)  # Transpose y to match the expected shape for stereo

    print(f"Stereo volume adjusted and saved to {output_audio}")


# Create the ffmpeg command to mix two audio files
def create_ffmpeg_mix_video_file_command(video_file, audio_file_1, audio_file_2, output_video):
    # Build the full ffmpeg command to mix two audio files
    # The `-shortest` flag ensures that the length is taken from the shortest input, i.e., the first audio file.
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", f"\"{video_file}\"", "-i", f"\"{audio_file_1}\"", "-i", f"\"{audio_file_2}\"",
        " -filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f"\"{output_video}\""
    ]
    # ffmpeg_command = [
    #     "ffmpeg", "-y", "-i", f'"{video_file}"', "-i", f'"{audio_file_1}"', "-i", f'"{audio_file_2}"',
    #     " -filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
    #     "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f'"{output_video}"'
    # ]


    return " ".join(ffmpeg_command)

# Create the ffmpeg command to mix two audio files
def create_ffmpeg_mix_video_file_command_list(video_file, audio_file_1, audio_file_2, output_video):
    # Build the full ffmpeg command to mix two audio files
    # The `-shortest` flag ensures that the length is taken from the shortest input, i.e., the first audio file.
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", f"\"{video_file}\"", "-i", f"\"{audio_file_1}\"", "-i", f"\"{audio_file_2}\"",
        " -filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f"\"{output_video}\""
    ]
    # ffmpeg_command = [
    #     "ffmpeg", "-y", "-i", f'"{video_file}"', "-i", f'"{audio_file_1}"', "-i", f'"{audio_file_2}"',
    #     " -filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]",
    #     "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f'"{output_video}"'
    # ]


    return ffmpeg_command

def create_ffmpeg_mix_audio_file_command(audio_file_1, audio_file_2, output_audio):
    """
    Generate an ffmpeg command to mix two audio files into a single AAC file.

    :param audio_file_1: Path to the first audio file.
    :param audio_file_2: Path to the second audio file.
    :param output_audio: Path to the output mixed audio file.
    :return: ffmpeg command as a string.
    """
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", f"\"{audio_file_1}\"", "-i", f"\"{audio_file_2}\"",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
        "-map", "[aout]", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f"\"{output_audio}\""
    ]
    return " ".join(ffmpeg_command)

def create_ffmpeg_mix_audio_file_command_list(audio_file_1, audio_file_2, output_audio):
    """
    Generate an ffmpeg command to mix two audio files into a single AAC file.

    :param audio_file_1: Path to the first audio file.
    :param audio_file_2: Path to the second audio file.
    :param output_audio: Path to the output mixed audio file.
    :return: ffmpeg command as a string.
    """
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", f"\"{audio_file_1}\"", "-i", f"\"{audio_file_2}\"",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
        "-map", "[aout]", "-c:a", "aac", "-b:a", "320k", "-ar", "44100", f"\"{output_audio}\""
    ]
    return ffmpeg_command

# Create the ffmpeg command to add a new audio track to a video, making the new audio the first track
# def create_ffmpeg_add_audio_to_video_command(input_video, additional_audio, output_video):
#     # Build the ffmpeg command to make the additional audio track as the first audio track
#     ffmpeg_command = [
#         "ffmpeg", "-y", "-i", input_video, "-i", additional_audio,
#         "-map", "0:v",          # Map the original video stream
#         "-map", "1:a",          # Map the additional audio stream as the first audio track
#         "-map", "0:a",          # Map the original audio stream as the second audio track
#         "-c:v", "copy",         # Copy the video stream without re-encoding
#         "-c:a", "aac", "-b:a", "192k",  # Encode audio as AAC with a bitrate of 192kbps
#         "-disposition:a:0", "default",  # Mark the additional (new) audio track as the default
#         "-disposition:a:1", "0",        # Set the original audio track as not default
#         output_video
#     ]
#
#     return ffmpeg_command

# Function to execute the ffmpeg command
def run_ffmpeg_command(command):
    try:
        # Run the command using subprocess
        result = subprocess.run(command, check=True) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg command executed successfully.")
        # print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("An error occurred while running FFmpeg:")
        # print(e.stderr.decode())

def separate_and_extract_accompaniment(input_path, output_path,temp_dir: str,max_dur_sec=1800):
    """
    Runs spleeter to split `input_path` into vocals+accompaniment,
    then moves only the accompaniment wav up to `output_dir` and
    removes the temp folder.   
    """
    _, sr = librosa.load(input_path, sr=None)

    # 1. Run the CLI separation
    subprocess.run(
        [
            "spleeter", "separate",
            "-p", "spleeter:2stems",
            "-o", str(temp_dir),
            "-b", str(sr),
            "-d", str(max_dur_sec),
            "--verbose",
            input_path
        ], check=True)

    # 2. Compute where Spleeter put the files
    base = os.path.splitext(os.path.basename(input_path))[0]
    temp_folder = os.path.join(temp_dir, base)
    acc_path = os.path.join(temp_folder, "accompaniment.wav")

    # 3. Move accompaniment out and clean up
    shutil.move(acc_path, output_path)
    shutil.rmtree(temp_folder)


if __name__ == "__main__":
    # Example usage
    csv_file = "output_speed.csv"
    input_video = "2_chkd.mp4"
    output_audio = "output_with_reduced_volume.wav"  # Save the audio output as a WAV file

    volume_intervals = parse_volume_intervals(csv_file)
    ffmpeg_command = create_ffmpeg_command(input_video, output_audio, volume_intervals, 0.3)

    # Output the generated command
    print("Generated ffmpeg command:")
    print(ffmpeg_command)
