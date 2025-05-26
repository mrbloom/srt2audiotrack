from datetime import timedelta
import csv
import os
import soundfile as sf
from datetime import datetime


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

def correct_end_times_in_csv(fragments_folder, input_csv_file, output_csv_file):
    """
    Correct the end times in the CSV file using the actual duration of generated TTS fragments.
    The new CSV file will have updated end times only.
    """
    corrected_rows = []

    with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        for i, row in enumerate(reader):
            segment_file = os.path.join(fragments_folder, f"segment_{i + 1}.wav")

            if os.path.exists(segment_file):
                # Read the generated audio segment to get its duration
                wav, sr = sf.read(segment_file)
                duration_seconds = len(wav) / sr
                duration_timedelta = timedelta(seconds=duration_seconds)

                # Get the current start time from the CSV row
                start_time_str = row['Start Time']
                try:
                    start_time = timedelta(seconds=time_to_seconds(start_time_str))
                except ValueError as e:
                    print(f"Error in row {i + 1}: {e}. Skipping correction for this row.")
                    continue

                # Calculate the new end time
                end_time = start_time + duration_timedelta

                # Format the new end time to match SRT format (HH:MM:SS,mmm)
                row['End Time'] = format_timedelta(end_time)
                row['Duration'] = duration_seconds

                corrected_rows.append(row)
            else:
                print(f"Warning: Expected segment {segment_file} not found. Skipping correction for this row.")
                corrected_rows.append(row)  # Add the original row if no correction is made

    # Write the corrected times to a new CSV file
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrected_rows)

    print(f"Corrected CSV file saved to {output_csv_file}")


def format_timedelta(td):
    """Format a timedelta to match the SRT format (HH:MM:SS,mmm)."""
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

if __name__ == "__main__":
    # Example usage
    correct_end_times_in_csv("P017025-01-003-REALNAYA_ISTOR_chkd",
                             "output_speed.csv",
                             "corrected_output_speed.csv")
