import tqdm
import webdataset as wds

train_files = open("train_files_filtered.txt").read().split("\n")
test_files = open("test_files_filtered.txt").read().split("\n")

raw_dataset = wds.WebDataset("en.tar.gz")

train_tar_file = wds.TarWriter("train.tar.gz")
test_tar_file = wds.TarWriter("test.tar.gz")


#total_count = len(train_files)
total_count = len(train_files)+len(test_files)

raw_iterator = iter(raw_dataset)

trackbar = tqdm.tqdm(total = total_count)
raw_iterator = iter(raw_dataset)
for sample in raw_iterator :
    file_name = sample["__key__"] + ".opus"
    inner_file = False
    
    if file_name in train_files :
        tar_pointer = train_tar_file
        train_files.remove(file_name)
        inner_file = True
        #print(file_name, "train")
    
    if file_name in test_files :
        tar_pointer = test_tar_file
        test_files.remove(file_name)
        inner_file = True
        #print(file_name, "test")
    else:
        pass
        #print(file_name, "other")
    if inner_file :
        tar_pointer.write(
            {
                "__key__":sample["__key__"],
                "__url__":"test.tar",
                "opus":sample["opus"]
            }
        )
        trackbar.update(1)
        #trackbar.update(total_count-(len(train_files)+len(test_files)))
    #input()
#print(test_files)
train_tar_file.close()
test_tar_file.close()