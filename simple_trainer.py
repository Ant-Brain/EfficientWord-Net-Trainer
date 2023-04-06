import math
NOISE_PATH = "/mnt/disks/disk-4/.eff/ESC-50-noise_files"
CLASSES_FILE = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/classes.txt"
CLASSES = open(CLASSES_FILE,'r').read().split("\n")
word2idx = { CLASS:i for i,CLASS in enumerate(CLASSES)}
idx2word = { i:CLASS for i,CLASS in enumerate(CLASSES)}

TRAIN_TAR = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/shards/train_vectors-{000000..000046}.tar.gz"
TEST_TAR = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/shards/test_vectors-{000000..000030}.tar.gz"
TRAIN_FILES_TXT = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/train_files_filtered.txt"
TEST_FILES_TXT = "/mnt/disks/disk-4/.eff/spoken_words_en_ml_commons_filtered_split/test_files_filtered.txt"

import webdataset as wds
import torch
import webdataset
import librosa
import io
import numpy as np
import random
import glob
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from simple_model import ResNet50_Classifier, ResNetArc_Classifier, AttentiveResNet50Arc_Classifier

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
        if self.mode=="train" :
            self.tar_iterator = iter(webdataset.WebDataset(self.tar_file, shardshuffle=True).shuffle(100000))
        else:
            self.tar_iterator = iter(webdataset.WebDataset(self.tar_file)) 
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

            yield ( out['x'], out['y'])

    def __len__(self):
        return self.count_files

BATCH_SIZE = 256
class AudioClassifierVectorDatasetPL(pl.LightningDataModule) :

    def __init__(self, batch_size = BATCH_SIZE):
        super(AudioClassifierVectorDatasetPL, self).__init__()
        self.batch_size = batch_size

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

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


class LightningWordClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pytorch_model = ResNetArc_Classifier(model_type="resnet50", class_count=len(CLASSES))
        self.loss = FocalLoss()

    def l2_norm_model(self) :
        for module in self.pytorch_model.modules():
            if hasattr(module, "weight") :
                module.weight = nn.Parameter(module.weight/torch.linalg.norm(module.weight))
    
    def forward(self, x,y = None) :
        #print(x)
        return self.pytorch_model(x, y)

    def topk_accuracy(self, y_preds,y,k, mode="train"):
        output = {}
        with torch.no_grad() :
            topk_y_preds = torch.topk(y_preds, k).indices
            prev_val = 0.0
            for i in range(k) :
                row = topk_y_preds[:, i]
                accuracy = (row==y).float().mean().item()
                #print(accuracy.shape)
                prev_val += accuracy
                if math.log(i+1,10)==int(math.log(i+1,10)) :
                    output[f"{mode}_top{i+1}"]=prev_val
                    #self.log(f"{mode}_top_{i+1}_accuracy", (prev_val), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return output

    def min_max_normalize(self, x):
        ele_dims = x.shape[1:]
        batch_dim = x.shape[0]

        x_flat = x.reshape(batch_dim, -1)

        x_min = x_flat.min(dim=1).values.reshape(batch_dim, 1)
        x_max = x_flat.max(dim=1).values.reshape(batch_dim, 1)

        t = (x_flat - x_min)/(x_max - x_min)*2 -1
        x_min_max = t.reshape(batch_dim ,*ele_dims)

        return x_min_max

    def training_step(self, train_batch, batch_idx):
        #self.l2_norm_model()

        x, y = train_batch
        x = self.min_max_normalize(x)

        logits = self.forward(x,y)
        loss = self.loss(logits, torch.squeeze(y))

        accuracies = self.topk_accuracy(logits, torch.squeeze(y),101, mode="train")
        self.log('train_loss', loss, logger=True, on_epoch=True)
        self.log_dict(accuracies, logger=True, on_epoch=True)

        return loss
  
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            x = self.min_max_normalize(x)

            logits = self.forward(x)
            loss = self.loss(logits, torch.squeeze(y)) 
            
            accuracies = self.topk_accuracy(logits, torch.squeeze(y),101, mode="val")
            self.log('val_loss', loss, logger=True, on_epoch=True)
            self.log_dict(accuracies, logger=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx) :
        with torch.no_grad():
            x, y = test_batch
            x = self.min_max_normalize(x)

            logits = self.forward(x)
            loss = self.loss(logits, torch.squeeze(y)) 
            
            accuracies = self.topk_accuracy(logits, torch.squeeze(y),101, mode="test")
            self.log('test_loss', loss, logger=True, on_epoch=True)
            self.log_dict(accuracies, logger=True, on_epoch=True)

    def l2_norm_model(self) :
        for module in self.pytorch_model.modules():
            if hasattr(module, "weight") and not(isinstance(module, nn.Linear)):
                module.weight = nn.Parameter(module.weight/torch.linalg.norm(module.weight))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3,
            eps=1e-4
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        return [optimizer], [{"scheduler":lr_scheduler, "interval":"epoch", "monitor":"train_loss_epoch"}]



if __name__=="__main__":

    pl_model = LightningWordClassifier()
    pl_model = pl_model.load_from_checkpoint("/home/captain-america/external_disk/.eff/AttentiveMobileWord-Trainer/resnet_50_noise/epoch=43-val_top1=0.7127.ckpt")
    
    pl_model = torch.load("/home/captain-america/external_disk/.eff/AttentiveMobileWord-Trainer/workspace/model_51_59.87%_67.8034%.pt")
    
    pl_data_module = AudioClassifierVectorDatasetPL()
    
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        deterministic=True,
        callbacks=[
            EarlyStopping(monitor="train_loss", mode="min", patience=10),
            ModelCheckpoint(
                dirpath  = "results_dir/",
                filename = "{epoch}-{val_top1:.4f}",
                monitor  = "val_top1",
                mode = "max",
                save_last = True,
                every_n_epochs=1,
            )
        ],
        logger=WandbLogger(project="atmob_cls_r50")
    )

    trainer.fit(pl_model, pl_data_module)