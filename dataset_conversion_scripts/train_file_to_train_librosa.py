import tqdm
import webdataset as wds
import librosa
import io

train_files_tar = wds.WebDataset("train.tar.gz")
test_files_tar = wds.WebDataset("test.tar.gz")
train_files_names = open("train_files_filtered.txt",'r').read().split("\n")
test_files_names = open("test_files_filtered.txt", "r").read().split("\n")

train_wavs_tar = wds.TarWriter("train_wavs.tar.gz")
test_wavs_tar = wds.TarWriter("test_wavs.tar.gz")

train_files_iterator = iter(train_files_tar)
test_files_iterator = iter(test_files_tar)

track_bar = tqdm.tqdm(total = len(train_files_names)+len(test_files_names))
track_bar = tqdm.tqdm(total = len(test_files_names))

for sample in train_files_iterator :
    train_wavs_tar.write({
        "__key__":sample["__key__"],
        "__url__":sample["__url__"],
        "wave.pyd":librosa.load(io.BytesIO(sample["opus"]), sr=16000)[0]
    })
    track_bar.update(1)

train_wavs_tar.close()

for sample in test_files_iterator :
    test_wavs_tar.write({
        "__key__":sample["__key__"],
        "__url__":sample["__url__"],
        "wave.pyd":librosa.load(io.BytesIO(sample["opus"]), sr=16000)[0]
    })
    track_bar.update(1)

test_wavs_tar.close()