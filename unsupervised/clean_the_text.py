import os

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def remove_https(inputString):
    elements = inputString.split(" ")
    outputString = ""
    for element in elements:
        if not "http" in element:
            outputString+=element
            outputString+=" "
    return outputString
def remove_atname(inputString):
    elements = inputString.split(" ")
    outputString = ""
    for element in elements:
        if not "@" in element:
            outputString+=element
            outputString+=" "
    return outputString
def remove_a(inputString):
    elements = inputString.split(" ")
    outputString = ""
    for element in elements:
        if not "&" in element:
            outputString+=element
        outputString+=" "
    return outputString
def clean_text(file_dir):
    src_file_header = open(file_dir)
    dst_file_header = open(file_dir.replace(".txt","_clean.txt"),"w")
    line = src_file_header.readline()
    
    while line:
        clean_string = deEmojify(line)
        clean_string = remove_https(clean_string)
        clean_string = remove_atname(clean_string)
        clean_string = remove_a(clean_string)
        if "   " in clean_string[-3:]:
            clean_string = clean_string[:-3]
        if not "\n" in clean_string:
            clean_string = clean_string+"\n"
        dst_file_header.write(clean_string)
        line = src_file_header.readline()
    
    
#clean_text("manually_selected.txt")

#clean_text("texts_negative.txt")
def containenglish(str0):
    import re
    return bool(re.search('[a-z]', str0))
def clean2(file_name):
    file_header = open(file_name)
    dst_header = open(file_name.replace(".txt","2.txt"),"w")
    line = file_header.readline()

    while line:
        #if not (line=="\n" or line=="  \n" or line==" \n"):
        if containenglish(line) and not ("COVID" in line or "Coronavirus" in line or "coronavirus" in line):
            dst_header.write(line)
        line = file_header.readline()

clean2("texts_negative_clean.txt")