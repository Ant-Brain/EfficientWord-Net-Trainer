train_files = open("test_files_filtered.txt","r").read().split("\n")

words = []

for train_file in train_files:
    word = train_file.split("/")[2]
    if word not in words :
        words.append(word)

open("classes.txt",'w').write("\n".join(words))