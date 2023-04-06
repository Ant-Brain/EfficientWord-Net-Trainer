from g2p_en import G2p
from tqdm import tqdm
g2p = G2p()

train_files = open("train_files.txt",'r').read().split("\n")
test_files = open("test_files.txt",'r').read().split("\n")

filtered_train_files = []
prev_word = None
prev_pronunciation = []
for train_file in tqdm(train_files) :
    word_name = train_file.split("/")[2]
    if word_name == prev_word :
        pronunciation = prev_pronunciation
    else:
        pronunciation = g2p(word_name)
    prev_word = word_name
    prev_pronunciation = pronunciation

    if len(pronunciation) > 3 :
        filtered_train_files.append(train_file)

filtered_test_files = []
prev_word = None
prev_pronunciation = []
for test_file in tqdm(test_files) :
    word_name = test_file.split("/")[2]
    if word_name == prev_word :
        pronunciation = prev_pronunciation
    else:
        pronunciation = g2p(word_name)
    prev_word = word_name
    prev_pronunciation = pronunciation

    if len(pronunciation) > 3 :
        filtered_test_files.append(test_file)

open("train_files_filtered.txt",'w').write("\n".join(filtered_train_files))
open("test_files_filtered.txt",'w').write("\n".join(filtered_test_files))