import os

sfile_header = open("texts.txt")
dfile_header = open("saved.txt","w")

line = sfile_header.readline()

text_list = []

while line:
    text_list.append(line)
    line = sfile_header.readline()

indexes = []
for i in range(len(text_list)):
    indexes.append(i)

import random

random.shuffle(indexes)

for index in indexes:
    print(text_list[index])
    str = input("please select:")
    if str=="y":
        dfile_header.write(line)
    elif str=="e":
        break

