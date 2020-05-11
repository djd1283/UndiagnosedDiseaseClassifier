import os

def split_train_val(file_name):
    train_val_proportion = 3
    file_header = open(file_name)

    train_file_header = open("train_"+file_name,"w")
    val_file_header = open("val_"+file_name,"w")

    line = file_header.readline()
    count=1
    while line:
        if count%(train_val_proportion+1)==0:
            val_file_header.write(line)
        else:
            train_file_header.write(line)
        count+=1
        line = file_header.readline()

split_train_val("selected.txt")
split_train_val("texts_clean2_ranked.txt")
split_train_val("texts_negative_clean2.txt")