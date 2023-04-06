#from model import AttentiveMobileWordClassifier

NOISE_PATH = "/home/captain-america/disk-4/.eff/ESC-50-noise_files"
CLASSES_FILE = "/home/captain-america/external_disk/.eff/spoken_words_en_ml_commons_filtered_split/classes.txt"
CLASSES = open(CLASSES_FILE,'r').read().split("\n")
word2idx = { CLASS:i for i,CLASS in enumerate(CLASSES)}
idx2word = { i:CLASS for i,CLASS in enumerate(CLASSES)}
#print(len(CLASSES))

#att_mob_word_classifier = AttentiveMobileWordClassifier(len(CLASSES))

TRAIN_TAR = "/home/captain-america/external_disk/.eff/spoken_words_en_ml_commons_filtered_split/train_wavs.tar.gz"
TEST_TAR = "/home/captain-america/external_disk/.eff/spoken_words_en_ml_commons_filtered_split/test_wavs.tar.gz"
TRAIN_FILES_TXT = "/home/captain-america/external_disk/.eff/spoken_words_en_ml_commons_filtered_split/train_files_filtered.txt"
TEST_FILES_TXT = "/home/captain-america/external_disk/.eff/spoken_words_en_ml_commons_filtered_split/test_files_filtered.txt"

import webdataset as wds
import torch
import webdataset
import librosa
import io
import numpy as np
import random
import glob
from torch.nn import functional as F

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

from audiomentations import Trim, Normalize, PeakingFilter
from audiomentations import Compose as Compose_no_torch

from torch_audiomentations import (
    #Compose,
    #AirAbsorption, # no issues
    PitchShift, #buggy
    #TanhDistortion,
    #TimeStretch
)

import python_speech_features
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
    find_audio_files_in_paths,
)
import pickle
import os
bruh = None
bruh2 = None

class AudioClassifierDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, tar_file:str, files_txt:str,word2idx:dict,max_audio_length = 1.5, noise_waves=[], mode = True):
        assert mode in ("train", "test"), "invalid mode"

        self.tar_file = tar_file
        self.count_files = len(open(files_txt, 'r').read().split())
        self.mode = mode
        self.word2idx:dict = word2idx
        self.max_audio_length = max_audio_length
        self.sr = 16000
        self.silence_trimmer = Trim(
            top_db=15.0,
            p=1.0
        )
        self.noise_waves = noise_waves
        self.normalize = Normalize(p=1)
        self.train_augumentations = Compose(
            #transforms = []
            transforms = [
                AirAbsorption(p=0.5), # no issues
            #    PitchShift(min_semitones=0, max_semitones=3, p=1), #buggy
                TanhDistortion(min_distortion = 0.01,max_distortion = 0.05,p=1,),
            #    TimeStretch(min_rate=0.8, max_rate=1.,p=1)
            ]
        )
        self.pitch_shift = PitchShift(min_transpose_semitones=0, max_transpose_semitones=4.0,p=1,sample_rate=16000, target_rate=16000)
        self.min_snr = 5.0 # min signal to noise ratio
        self.max_snr = 15.0 # max signal to noise ratio
    
    def compute_mfcc_features(self, audio):
        global bruh
        return python_speech_features.logfbank(
            audio,samplerate=16000,winlen=0.025,winstep=0.01,nfilt=64,
            nfft=512,lowfreq=0,highfreq=None,preemph=0.97
        )
        #return python_speech_features.mfcc(audio,samplerate=16000,winlen=0.025,winstep=0.01,numcep=64,
        #         nfilt=64,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        #         ceplifter=64,appendEnergy=True)

    def get_noise(self, audio):
        global bruh
        #random.seed(len(audio))
        noise_wav = self.noise_waves[random.randint(0, len(self.noise_waves)-1)]
        max_length = int(self.sr*self.max_audio_length)

        selected_noise_wav = noise_wav + 0.0 #deepcopy
        while len(selected_noise_wav) < len(audio):
            selected_noise_wav = np.concatenate((selected_noise_wav,selected_noise_wav),axis=0)

        start_index = random.randint(0,len(selected_noise_wav)-max_length-1)
        noise_chunk = selected_noise_wav[start_index:start_index+max_length]

        #noise_chunk = self.noise_transform(noise_chunk, sample_rate = self.sr)

        target_snr = random.uniform(self.min_snr, self.max_snr)

        noise_std = noise_chunk.std()
        signal_std = audio.std()

        current_snr = signal_std/noise_std
        #print(np.isinf(noise_std))
        if noise_std==0.0 :
            return np.zeros(max_length) + 0.0

        #print("signal std", signal_std, noise_std, current_snr, target_snr)
        target_noise_std = (current_snr / target_snr) * noise_std

        if target_noise_std < 1e-12 :
            return np.zeros(max_length) + 0.0

        #print(noise_std, target_noise_std)
        noise_chunk = (noise_chunk / noise_std) * target_noise_std

        return noise_chunk

    def pitch_shift_torch(self, audio, min_ratio=0.85, max_ratio=1.1):
        #basically performs timestretch
        random_scale_factor = random.uniform(min_ratio, max_ratio)
        target_shape = int(audio.shape[0]*random_scale_factor)

        if target_shape > int(self.sr*self.max_audio_length) :
            target_shape = int(self.sr*self.max_audio_length) - 10

        modded_inp = audio[None, None, None , :]

        return F.interpolate(modded_inp, size = (1, target_shape), mode = "bicubic")[0][0][0]

    def correct_audio_length(self, audio):
        if self.mode=="test":
            random.seed(len(audio))
        if self.sr*self.max_audio_length >= len(audio) :
            remaining_vals = int(self.sr * self.max_audio_length - len(audio))
            index = int(random.randint(0,remaining_vals))
            #print(index,type(index))
            #print(audio, audio.dtype)
            return np.concatenate(
                (
                    np.zeros(index,dtype=np.float32), 
                    audio, 
                    np.zeros(remaining_vals-index,dtype=np.float32)
                ), 
                axis=0
            )
        else:
            remaining_vals = int(len(audio)-self.sr*self.max_audio_length)
            index = int(random.randint(0, remaining_vals-1))
            return audio[index:index+self.sr*self.max_audio_length]

    def __iter__(self): 
        global bruh, bruh2
        self.tar_iterator = iter(webdataset.WebDataset(self.tar_file).shuffle(100000))
        for sample in self.tar_iterator :
            word = sample["__key__"].split("/")[2]
            key = sample["__key__"]
            idx = word2idx[word]
            audio = pickle.loads(sample["wave.pyd"])
            original_length = len(audio)
            audio = np.array(self.silence_trimmer(audio, sample_rate=self.sr))
            
            orig_audio = audio + 0.0

            if self.mode == "train" :
                #audio = self.train_augumentations(audio, self.sr)
                audio = self.pitch_shift_torch(torch.Tensor(audio)).numpy()
                audio = self.correct_audio_length(audio)
                
                noise = torch.Tensor(np.expand_dims(self.get_noise(audio), axis=[0,1]))
                audio = torch.Tensor(np.expand_dims(audio, axis=[0,1]))
                
                addNoise = random.choice([True,False])
                if addNoise :
                    audio = (audio + noise)[0][0]
                else:
                    audio = audio[0][0]
                #print("after",audio.shape)
                pass
            else:
                audio = self.correct_audio_length(audio)
            #audio = self.normalize(audio, sample_rate=self.sr)
            out = {
                "y":torch.Tensor([idx,]),
                "audio":torch.Tensor(audio),
                #"orig":torch.Tensor(orig_audio),
                "key":key,
                "x":torch.Tensor(
                        np.expand_dims(
                            self.compute_mfcc_features(audio),
                            axis=0
                        )
                    )
            }

            yield out

    def __len__(self):
        return self.count_files

class AudioClassifierDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, tar_file:str, files_txt:str,word2idx:dict,max_audio_length = 1.5, noise_waves=[], mode = "train"):
        assert mode in ("train", "test"), "invalid mode"

        self.tar_file = tar_file
        self.count_files = len(open(files_txt, 'r').read().split())
        self.mode = mode
        self.word2idx:dict = word2idx
        self.max_audio_length = max_audio_length
        self.sr = 16000
        self.silence_trimmer = Trim(
            top_db=100.0,
            p=1.0
        )
        self.noise_waves = noise_waves
        self.normalize = Normalize(p=1)
        self.train_augumentations = Compose(
            #transforms = []
            transforms = [
                AirAbsorption(p=0.5), # no issues
            #    PitchShift(min_semitones=0, max_semitones=3, p=1), #buggy
                TanhDistortion(min_distortion = 0.01,max_distortion = 0.05,p=1,),
            #    TimeStretch(min_rate=0.8, max_rate=1.,p=1)
            ]
        )
        self.pitch_shift = PitchShift(min_transpose_semitones=0, max_transpose_semitones=4.0,p=1,sample_rate=16000, target_rate=16000)
        self.min_snr = 2.0 if mode=="train" else 3.0 # min signal to noise ratio
        self.max_snr = 5.1 # max signal to noise ratio
    
    def compute_mfcc_features(self, audio):
        global bruh
        
        return python_speech_features.logfbank(
            audio,
            samplerate=16000,
            winlen=0.025,
            winstep=0.01,
            nfilt=64,
            nfft=512,
            preemph=0.0
        )
        """
        return python_speech_features.mfcc(
            audio,
            samplerate=16000,
            winlen=0.025,
            winstep=0.01,
            #nfilt=64,
            numcep=32,
            nfilt=32,
            nfft=512
        )
        """

    def get_noise(self, audio):
        global bruh
        #random.seed(len(audio))
        noise_wav = self.noise_waves[random.randint(0, len(self.noise_waves)-1)]
        max_length = int(self.sr*self.max_audio_length)

        selected_noise_wav = noise_wav + 0.0 #deepcopy
        while len(selected_noise_wav) < len(audio):
            selected_noise_wav = np.concatenate((selected_noise_wav,selected_noise_wav),axis=0)

        start_index = random.randint(0,len(selected_noise_wav)-max_length-1)
        noise_chunk = selected_noise_wav[start_index:start_index+max_length]

        #noise_chunk = self.noise_transform(noise_chunk, sample_rate = self.sr)

        target_snr = random.uniform(self.min_snr, self.max_snr)

        noise_std = noise_chunk.std()
        signal_std = audio.std()

        current_snr = signal_std/noise_std
        #print(np.isinf(noise_std))
        if noise_std==0.0 :
            return np.zeros(max_length) + 0.0

        #print("signal std", signal_std, noise_std, current_snr, target_snr)
        target_noise_std = (current_snr / target_snr) * noise_std

        if target_noise_std < 1e-12 :
            return np.zeros(max_length) + 0.0

        #print(noise_std, target_noise_std)
        noise_chunk = (noise_chunk / noise_std) * target_noise_std

        return noise_chunk

    def pitch_shift_torch(self, audio, min_ratio=0.85, max_ratio=1.1):
        #basically performs timestretch
        random_scale_factor = random.uniform(min_ratio, max_ratio)
        target_shape = int(audio.shape[0]*random_scale_factor)

        if target_shape > int(self.sr*self.max_audio_length) :
            target_shape = int(self.sr*self.max_audio_length) - 10

        modded_inp = audio[None, None, None , :]

        return F.interpolate(modded_inp, size = (1, target_shape), mode = "bicubic")[0][0][0]

    def correct_audio_length(self, audio):
        if self.mode=="test":
            random.seed(len(audio))
        if self.sr*self.max_audio_length >= len(audio) :
            remaining_vals = int(self.sr * self.max_audio_length - len(audio))
            index = int(random.randint(0,remaining_vals))
            #print(index,type(index))
            #print(audio, audio.dtype)
            return np.concatenate(
                (
                    np.zeros(index,dtype=np.float32), 
                    audio, 
                    np.zeros(remaining_vals-index,dtype=np.float32)
                ), 
                axis=0
            )
        else:
            remaining_vals = int(len(audio)-self.sr*self.max_audio_length)
            index = int(random.randint(0, remaining_vals-1))
            return audio[index:index+self.sr*self.max_audio_length]

    def __iter__(self): 
        global bruh, bruh2
        self.tar_iterator = iter(webdataset.WebDataset(self.tar_file).shuffle(100000))
        for sample in self.tar_iterator :
            word = sample["__key__"].split("/")[2]
            key = sample["__key__"]
            idx = word2idx[word]
            audio = pickle.loads(sample["wave.pyd"])
            original_length = len(audio)
            #audio = np.array(self.silence_trimmer(audio, sample_rate=self.sr))
            audio = np.array(audio)
            orig_audio = audio + 0.0

            if self.mode == "train" :
                #audio = self.train_augumentations(audio, self.sr)
                audio = self.pitch_shift_torch(torch.Tensor(audio)).numpy()
            audio = self.correct_audio_length(audio)

            noise = torch.Tensor(np.expand_dims(self.get_noise(audio), axis=[0,1]))
            audio = torch.Tensor(np.expand_dims(audio, axis=[0,1]))

            audio = (audio + noise)[0][0]
            #audio = audio[0][0]
            out = {
                "y":torch.Tensor([idx,]),
                "audio":torch.Tensor(audio),
                #"orig":torch.Tensor(orig_audio),
                "key":key,
                "x":torch.Tensor(
                        np.expand_dims(
                            self.compute_mfcc_features(audio),
                            axis=0
                        )
                    )
            }

            yield out

    def __len__(self):
        return self.count_files

import pickle
with open("/home/captain-america/disk-4/.eff/AttentiveMobileWord-Trainer/noise_files.dump",'rb') as rfile:
    print(rfile)
    noise_waves_2 = pickle.load(rfile)

import tqdm

import os
os.makedirs("shards")

train_tar_file = wds.ShardWriter("shards/train_vectors-%06d.tar.gz",maxcount=5000)
for i in range(1):
    train_dataset = AudioClassifierDataset(TRAIN_TAR, TRAIN_FILES_TXT, word2idx=word2idx, noise_waves = noise_waves_2, mode = "train")
    train_iterator = iter(train_dataset)
    tracker = tqdm.tqdm(total = len(train_dataset))
    count = 0
    for sample in train_iterator :
        train_tar_file.write({
            "__key__":sample["key"]+"__"+str(i),
            #"audio.pyd":sample["audio"],
            "y.pyd":sample["y"],
            "x.pyd":sample["x"]
        })
        tracker.update(1)
        count += 1
    print(count)
train_tar_file.close()

test_tar_file = wds.ShardWriter("shards/test_vectors-%06d.tar.gz",maxcount=5000)
for i in range(1):
    test_dataset = AudioClassifierDataset(TEST_TAR, TEST_FILES_TXT, word2idx=word2idx, noise_waves = noise_waves_2, mode = "test")
    test_iterator = iter(test_dataset)
    tracker = tqdm.tqdm(total = len(test_dataset))
    count = 0
    for sample in test_iterator :
        count += 1
        test_tar_file.write({
            "__key__":sample["key"]+"__"+str(i),
            #"audio.pyd":sample["audio"],
            "y.pyd":sample["y"],
            "x.pyd":sample["x"]
        })
        tracker.update(1)
    print(count)
test_tar_file.close()