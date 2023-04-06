import webdataset as wds
import tqdm
import pickle
import python_speech_features
train_files_names = open("/home/captain-america/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/train_files_filtered.txt",'r').read().split("\n")

trackbar = tqdm.tqdm(total = len(train_files_names))
for sample in iter(wds.WebDataset("train_wavs.tar.gz")):
    waveform = pickle.loads(sample["wave.pyd"])
    #python_speech_features.mfcc(waveform,samplerate=16000,winlen=0.025,winstep=0.01,numcep=64,
    #             nfilt=64,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
    #ceplifter=64,appendEnergy=True)
    python_speech_features.logfbank(waveform,samplerate=16000,winlen=0.025,winstep=0.01,nfilt=64,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

    trackbar.update(1)