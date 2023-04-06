#import faulthandler
#faulthandler.enable()

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
import webdataset
from simple_model import *
import pickle

NOISE_PATH = "/home/captain-america/disk-4/.eff/ESC-50-noise_files"
CLASSES_FILE = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/classes.txt"
CLASSES = open(CLASSES_FILE,'r').read().split("\n")
word2idx = { CLASS:i for i,CLASS in enumerate(CLASSES)}
idx2word = { i:CLASS for i,CLASS in enumerate(CLASSES)}

TRAIN_TAR = "/home/captain-america/disk-4/.eff/shards/train_vectors-{000000..000031}.tar.gz"
TEST_TAR = "/home/captain-america/disk-4/.eff/shards/test_vectors-{000000..000019}.tar.gz"
TRAIN_FILES_TXT = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/train_files_filtered.txt"
TEST_FILES_TXT = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/test_files_filtered.txt"

class AudioClassifierVectorDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, tar_file:str, files_txt:str,word2idx:dict,max_audio_length = 1.5, mode = True):
        assert mode in ("train", "test"), "invalid mode"

        self.tar_file = tar_file
        self.count_files = len(open(files_txt, 'r').read().split())
        self.mode = mode
        self.word2idx:dict = word2idx
        self.max_audio_length = max_audio_length
        self.sr = 16000

    def __iter__(self): 
        global bruh, bruh2
        self.tar_iterator = iter(webdataset.WebDataset(self.tar_file, shardshuffle=True, resampled=True).shuffle(100000))
        for sample in self.tar_iterator :
            word = sample["__key__"].split("/")[2]
            idx = self.word2idx[word]

            out = {
                "y":pickle.loads(sample["y.pyd"]).to(torch.int64),
                #"y":nnf.one_hot(pickle.loads(sample["y.pyd"]).to(torch.int64), num_classes = len(CLASSES)),
                #"audio":pickle.loads(sample["audio.pyd"]),
                #"orig":torch.Tensor(orig_audio),
                "x":pickle.loads(sample["x.pyd"])
            }

            yield out

    def __len__(self):
        return self.count_files

BATCH_SIZE = 256
class AudioClassifierVectorDatasetPL(pl.LightningDataModule) :

    def setup(self, stage):
        self.train_dataset = AudioClassifierVectorDataset(
            TRAIN_TAR,
            TRAIN_FILES_TXT,
            word2idx,
            mode="train"
        )
        self.test_dataset = AudioClassifierVectorDataset(
            TEST_TAR,
            TEST_FILES_TXT,
            word2idx,
            mode="test"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class LightningClassifier(pl.LightningModule):

  def __init__(self):
    super().__init__()

    self.pytorch_model = ResNet50Arc_Classifier(len(CLASSES))
    #self.pytorch_model = ResNet50_ClassifierV2(len(CLASSES))
    #self.pytorch_model = ResNet50_Classifier(len(CLASSES))
    #self.pytorch_model = AttentiveResNet50Arc_Classifier(len(CLASSES))
    self.loss = FocalLoss()

  def forward(self, x):
      x = self.pytorch_model(x)
      return x

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch["x"], train_batch["y"]
      logits = self.forward(x)
      loss = self.loss(logits, y.squeeze())
      self.log('train_loss', loss)
      return loss

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch["x"] ,val_batch["y"]
      logits = self.forward(x)
      loss = self.loss(logits, y.squeeze())
      self.log('val_loss', loss)

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-4)
      return optimizer

pl_model = LightningClassifier()
pl_data_module = AudioClassifierVectorDatasetPL()

trainer = pl.Trainer(precision=16,accelerator="gpu", devices=1)

print("#########################################")
print("#########################################")
print("#########################################")
print("#########################################")
trainer.fit(pl_model, pl_data_module)