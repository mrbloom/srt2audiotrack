# srt2audiotrack

Make directory with videos and srts to videos with english audiotracks.

For installation:

In Windows:
conda create -n f5-tts-demucs python=3.10
conda activate f5-tts-demucs
pip install f5-tts-demucs

then install demucs (https://github.com/adefossez/demucs)
python -m pip install -U demucs


Run:
conda activate f5-tts-demucs
python main.py --subtitle records\one_voice
