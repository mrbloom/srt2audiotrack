# srt2audiotrack

Make from directory with videos and srt files videos with english audiotrack.

For installation:

In Windows:

1. Create conda environment:
conda create -n f5-tts-demucs python=3.10
conda activate f5-tts-demucs

2. Install f5-tts (https://github.com/SWivid/F5-TTS/tree/main)
pip install f5-tts

then install demucs (https://github.com/adefossez/demucs)
python -m pip install -U demucs

3. Run:
python main.py --subtitle records\one_voice

In records\one_voice directory will be created videos with suffix "_mix_out.mp4"

Result must be:
https://fex.net/ru/s/fctovr0

ToDo:

1. Make something with short not-generated segments.
2. Test work with single file.
3. Refactor everything.