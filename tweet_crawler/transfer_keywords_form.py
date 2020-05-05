import os

def transfer(src_file,dst_file):
    src_header = open(src_file)
    dst_header = open(dst_file,"w")

    line = src_header.readline()

    while line:

        if " " in line:
            line = "\""+line
            line = line.replace('\n','\"\n')
        dst_header.write(line)

        line = src_header.readline()

if __name__ == "__main__":
    transfer("terms.txt","transfered_terms.txt")