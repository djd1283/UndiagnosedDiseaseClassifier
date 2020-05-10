import os
import json
import glob

def recursive_json_key(json_data,target_key):
    target_text=""
    got_it = False
    if isinstance(json_data,dict):
        for key in json_data:
            if isinstance(json_data[key],dict):
                target_text,got_it = recursive_json_key(json_data[key],target_key)
                if got_it==True:
                    return target_text,got_it
            else:
                if key==target_key:
                    target_text = json_data[key]
                    return target_text,True
    return target_text,got_it

def readLinebyLine(file_name,save_text):
    file_header = open(file_name)

    line = file_header.readline()

    while line:
        json_data = json.loads(line)
        #if "undiagnosed disease" in json_data["text"] and not("covid" in json_data["text"]) and not("coronavirus" in json_data["text"]):
        #    save_text.write(json_data["text"]+"\n")
        target_text,got_it = recursive_json_key(json_data,"full_text")
        if got_it:
            save_text.write(target_text+"\n")
        else:
            target_text,got_it = recursive_json_key(json_data,"text")
            if got_it:
                save_text.write(target_text+"\n")
        line = file_header.readline()

record_files = glob.glob("tweets_clawler/negatives/incoming*")
save_text = open("texts_negative.txt","w")

for record_file in record_files:
    readLinebyLine(record_file,save_text)