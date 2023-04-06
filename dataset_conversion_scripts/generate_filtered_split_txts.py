TARCONTENTS_PATH="/mnt/disks/disk-4/.eff/contents.txt"
TARCONTENTS = [x for x in open(TARCONTENTS_PATH, encoding='utf8').read().split("\n") if x!=""]
from typing import OrderedDict, List, Dict

class Folder :

    def __init__(self, name, parent_path):
        self.name = name
        self.path = parent_path + "/" + name
        self.subfolders:Dict[str, Folder] = OrderedDict()
        self.subfiles:List[str] = []

    def __str__(self):
        return f"<Folder '{self.name}'>"

    def __repr__(self) -> str:
        return self.__str__()

def get_structure(tar_contents:List) -> List[Folder] :
    root:Folder = Folder("root", "")
    for ele in tar_contents :
        flow = ele.split("/")
        folder_pointer = root
        for i in range(len(flow)):
            found = False
            val = folder_pointer.subfolders.get(flow[i], None)
            if val!=None :
                folder_pointer = val
                found = True
            else:
                if i!=len(flow)-1 :#not last
                    #print(folder_pointer.path)
                    folder_pointer.subfolders[flow[i]] = Folder(flow[i], folder_pointer.path)
                    folder_pointer = folder_pointer.subfolders[flow[i]]
                elif flow[i] == "" :
                    pass
                else:
                    folder_pointer.subfiles.append(flow[i])
    return root

structure = get_structure(TARCONTENTS)
import random
import math

def create_split(inp_pointer:Folder, min_sample_count = 200, limit=220)->List[Folder]:
    
    train_folder = Folder(inp_pointer.name, "")
    train_folder.path = inp_pointer.path

    test_folder = Folder(inp_pointer.name, "")
    test_folder.path = inp_pointer.path

    if len(inp_pointer.subfolders)>0:
        for key in inp_pointer.subfolders.keys() :
            current_pointer = inp_pointer.subfolders[key]
            if len(current_pointer.subfolders)==0 and len(current_pointer.subfiles) < min_sample_count :
                continue
            sub_train_folder, sub_test_folder = create_split(inp_pointer.subfolders[key])
            train_folder.subfolders[key] = sub_train_folder
            test_folder.subfolders[key] = sub_test_folder
     
    train_files = []
    test_files = []
    
    if len(inp_pointer.subfiles)>0 :
        subfiles = inp_pointer.subfiles + [] #deepcopy
        
        random.seed(2)
        random.shuffle(subfiles)
        if len(subfiles) > limit :
            subfiles = subfiles[:limit]

        train_index = math.ceil(len(subfiles)*0.6)
        train_files = subfiles[:train_index]
        test_files = subfiles[train_index:]

    train_folder.subfiles = train_files
    test_folder.subfiles = test_files

    return train_folder, test_folder

train_folder, test_folder = create_split(structure)

from g2p_en import G2p
from tqdm import tqdm
g2p = G2p()

train_dir = train_folder.subfolders["en"].subfolders["clips"]
test_dir = test_folder.subfolders["en"].subfolders["clips"]

import os
import tqdm
train_files = []
test_files = []

print(len(train_dir.subfolders), len(test_dir.subfolders))

import random

#selected_classes = random.sample(train_dir.subfolders.keys(), 1750)

filtered_classes = []
for word in train_dir.subfolders.keys() :
    pronunciation = g2p(word)
    if len(pronunciation)>=5 :
        filtered_classes.append(word)

selected_classes = random.sample(filtered_classes, 1750)

for word in tqdm.tqdm(selected_classes) :
    for subfile in train_dir.subfolders[word].subfiles :
        path = train_dir.subfolders[word].path
        tmp = os.path.join(path, subfile)
        train_files.append(tmp.split("/root/")[1])

for word in tqdm.tqdm(selected_classes) :
    for subfile in test_dir.subfolders[word].subfiles :
        path = test_dir.subfolders[word].path
        tmp = os.path.join(path, subfile)
        test_files.append(tmp.split("/root/")[1])
   
        
open("train_files_filtered.txt", 'w').write("\n".join(train_files))
open("test_files_filtered.txt", 'w').write("\n".join(test_files))