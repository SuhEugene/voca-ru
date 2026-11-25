import glob
import numpy as np
import librosa
import soundfile as sf

path = sorted(glob.glob("/workspace/corpus/audio/f_neutral/*.flac"))[0]
y, sr = sf.read(path)
print("audio", len(y), sr)
f0, vf, vp = librosa.pyin(y, fmin=50, fmax=600, sr=sr)
print("pyin ok", float(np.nanmean(f0)))
