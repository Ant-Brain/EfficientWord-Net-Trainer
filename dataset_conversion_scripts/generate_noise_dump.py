
NOISE_PATH = "/home/captain-america/disk-4/.eff/ESC-50-noise_files"

import librosa
import tqdm
import glob
import os

from audiomentations import ( 
    Compose, 
    Trim, 
    AddBackgroundNoise, 
    AddGaussianNoise, 
    AirAbsorption, 
#    PitchShift,
    Normalize,
    TimeMask,
    TanhDistortion,
    TimeStretch,
    PeakingFilter
)

noise_transform = Compose(
    transforms = [
        PeakingFilter(p=1),
        Normalize(p=1)
    ]
)
noise_waves = []
for x in tqdm.tqdm(glob.glob(os.path.join(NOISE_PATH, "*"))):
    #noise_waves.append(librosa.load(x, sr=16000)[0])
    noise_waves.append(noise_transform(librosa.load(x,sr=16000)[0],sample_rate=16000))
import pickle
with open("noise_files.dump",'wb') as file :
    pickle.dump(noise_waves, file)