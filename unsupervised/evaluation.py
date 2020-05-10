import os
from locale import str

sfile_header = open("texts_clean2_ranked.txt")
dfile_header = open("texts_clean2_ranked_eval.txt","a")

line = sfile_header.readline()

count=1
while line:
    print("----------"+str(count)+"------------")
    if count>203:
        print(line)
        opion = input("please select:")
        if opion=="y":
            dfile_header.write("True\n")
        elif opion=="n":
            dfile_header.write("False\n")
        elif opion=="s":
            dfile_header.write("Skip\n")
        elif opion=="e":
            break
    count+=1
    print("")
    line = sfile_header.readline()
