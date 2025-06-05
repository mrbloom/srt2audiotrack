import csv
import re
from datetime import datetime
import pandas as pd
import csv
import srt
from datetime import timedelta
from pathlib import Path

DELAY = 0.0005 # Lets connect everything with less than DELAY  seconds 

def format_timedelta(td: timedelta) -> str:
    """
    Convert a timedelta to an SRT‐style timestamp 'HH:MM:SS,mmm'.
    """
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    hours = total_s // 3600
    minutes = (total_s % 3600) // 60
    seconds = total_s % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"



def calculate_duration(start, end):
    time_format = '%H:%M:%S,%f'
    start_dt = datetime.strptime(start, time_format)
    end_dt = datetime.strptime(end, time_format)
    duration = (end_dt - start_dt).total_seconds()
    return duration

def srt_to_csv(srt_file: str, csv_file: str):
    # 1) Read & parse all subtitles
    with open(srt_file, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))

    # 2) Open CSV for writing
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow([
            'Number', 'Start Time', 'End Time',
            'Duration', 'Symbol Duration', 'Text'
        ])

        # 3) Iterate subtitles
        for sub in subs:
            # Start/end in SRT format
            start_str = format_timedelta(sub.start)
            end_str   = format_timedelta(sub.end)

            # Duration in seconds
            duration = (sub.end - sub.start).total_seconds()

            # Clean up the text: collapse any internal newlines
            text = sub.content.replace('\n', ' ').strip()

            # Per-character duration
            symbol_duration = duration / len(text) if text else 0

            # Write row
            writer.writerow([
                sub.index,
                start_str,
                end_str,
                f"{duration:.3f}",
                f"{symbol_duration:.4f}",
                text
            ])

            # Also print to console (tab-separated)
            print("\t".join(map(str, [
                sub.index,
                start_str,
                end_str,
                f"{duration:.3f}",
                f"{symbol_duration:.4f}",
                text
            ])))


def srt_to_csv_with_full_sentences(csv_file, full_csv_file, delay_min=DELAY):
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        
        current_text = ""
        current_start_time = None
        current_end_time = None

        for row in reader:
            start_time = row['Start Time']
            end_time = row['End Time']
            duration = float(row['Duration'])
            symbol_duration = float(row['Symbol Duration'])
            text = row['Text']

            if current_end_time:
                end_dt = datetime.strptime(current_end_time, '%H:%M:%S,%f')
                start_dt = datetime.strptime(start_time, '%H:%M:%S,%f')
                delay = (start_dt - end_dt).total_seconds()
                print(f"delay = (start_dt={start_dt} - end_dt={end_dt}).total_seconds() = {delay}")

                if delay <= delay_min:
                    print(f"delay = {delay} <= {delay_min} = delay_min")
                    current_text += " " + text
                    current_end_time = end_time
                    continue

            if current_text:
                combined_duration = calculate_duration(current_start_time, current_end_time)
                combined_symbol_duration = combined_duration / len(current_text) if len(current_text) > 0 else 0
                rows.append([current_start_time, current_end_time, combined_duration, combined_symbol_duration, current_text])

            current_text = text
            current_start_time = start_time
            current_end_time = end_time

        if current_text:
            combined_duration = calculate_duration(current_start_time, current_end_time)
            combined_symbol_duration = combined_duration / len(current_text) if len(current_text) > 0 else 0
            rows.append([current_start_time, current_end_time, combined_duration, combined_symbol_duration, current_text])

    with open(full_csv_file, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        writer.writerow(['Start Time', 'End Time', 'Duration', 'Symbol Duration', 'Text'])
        writer.writerows(rows)

# def find_closest_speed_symbol_duration(symbol_duration, speed_df):
#     closest_index = (np.abs(speed_df['symbol_duration'] - symbol_duration)).idxmin()
#     return speed_df.iloc[closest_index]

def find_closest_from_floor(symbol_duration, speed_df):
    # Filter for rows where `symbol_duration` is less than or equal to the target
    filtered_df = speed_df[speed_df['symbol_duration'] <= symbol_duration]
    
    if not filtered_df.empty:
        # Return the closest matching row from below or exact match
        closest_index = (symbol_duration - filtered_df['symbol_duration']).idxmin()
    else:
        # If no smaller or equal value exists, return the closest available (smallest higher)
        closest_index = (speed_df['symbol_duration'] - symbol_duration).abs().idxmin()
        
    return speed_df.loc[closest_index]

def add_speed_columns(output_csv, speed_csv, output_speed_csv):
    speed_df = pd.read_csv(speed_csv)
    
    with open(output_csv, 'r', encoding='utf-8') as input_file, open(output_speed_csv, 'w', newline='', encoding='utf-8') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames[:-1] + ['tts_symbol_duration', 'speed_tts_closest', 'Text']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            symbol_duration = float(row['Symbol Duration'])
            closest_match = find_closest_from_floor(symbol_duration, speed_df)

            row['tts_symbol_duration'] = closest_match['symbol_duration']
            row['speed_tts_closest'] = closest_match['speed']
            # row['file_name_tts'] = closest_match['file_name']
            
            writer.writerow(row)
            print("\t".join(map(str, row.values())))


def add_speaker_columns(input_csv, output_csv, speakers=[]):
    with open(input_csv, 'r', encoding='utf-8') as input_file, open(output_csv, 'w', newline='',
                                                                     encoding='utf-8') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames[:-1] + ['Speaker', 'Text']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for row in reader:
            text = row['Text']
            # Extract the first occurrence of [ ... ]
            match = re.search(r"\[(.*?)\]: ", text)
            first_bracket_content =  match.group(1) if match else None
            # Remove only the first occurrence
            cleaned_text = re.sub(r"\[.*?\]: ", "", text, count=1).strip()
            if first_bracket_content:
                row["Speaker"] = first_bracket_content
                row["Text"] = cleaned_text
            else:
                row["Speaker"] = ""
            writer.writerow(row)
            print("\t".join(map(str, row.values())))

def find_closest_from_floor_value_index(value, array):
    """
    Find the closest floor value (largest value ≤ input value) and its index in a sorted array.
    
    :param value: The target value to compare against.
    :param array: A list of numbers (assumed to be sorted in ascending order).
    :return: A tuple (floor_value, index) where:
             - floor_value is the closest value ≤ target value.
             - index is the position of floor_value in the array.
    """
    max_value, index = array[0], 0  # Initialize with the first element
    
    for i, arr in enumerate(array):
        if arr < value:
            max_value, index = arr, i 
            break  # Stop when we find the first value greater than the target
        max_value, index = arr, i  # Update max_value and index
    
    return max_value, index


def add_speed_columns_with_speakers(output_csv_with_speakers, speakers, output_with_preview_speeds_csv):
    with open(output_csv_with_speakers, 'r', encoding='utf-8') as input_file, open(output_with_preview_speeds_csv, 'w', newline='', encoding='utf-8') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames[:-2] + ['TTS Symbol Duration', 'TTS Speed Closest', 'Speaker', 'Text']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()    

        for row in reader:
            symbol_duration = float(row['Symbol Duration'])
            speaker_name = row['Speaker']
            try:
                speaker = speakers[speaker_name]
            except KeyError:
                print(f"Speaker {speaker_name} not found in speakers")
                speaker_name = speakers["default_speaker_name"]
                speaker = speakers[speaker_name]
                print(f"Speaker not found in speakers, using default speaker {speaker_name}")
            closest_duration, index = find_closest_from_floor_value_index(symbol_duration, speaker['symbol_durations'])
            closet_speed = speaker['speeds'][index]
            row['TTS Symbol Duration'] = closest_duration
            row['TTS Speed Closest'] = closet_speed
            row['Speaker'] = speaker_name
            writer.writerow(row)
            print("\t".join(map(str, row.values())))

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

