import demucs.separate

def separate_audio():
    demucs.separate.main(["--jobs", "4","-o", "records\\one_voice\\separated", "--two-stems", "vocals", "-n", "mdx_extra", "records\\one_voice\\tanks_en.mp4"])

if __name__ == "__main__":
    separate_audio()