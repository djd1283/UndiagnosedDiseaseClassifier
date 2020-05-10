import os
import json

json_data = json.load(open("test.txt"))


def recursive_json_key(json_data,target_key):
    target_text=""
    got_it = False
    if isinstance(json_data,dict):
        for key in json_data:
            print(key)
            if isinstance(json_data[key],dict):
                target_text,got_it = recursive_json_key(json_data[key],target_key)
                if got_it==True:
                    return target_text,got_it
            else:
                if key==target_key:
                    target_text = json_data[key]
                    return target_text,True
    return target_text,got_it
    '''    
    else:

        if key==target_key:
            return json_data[key]        
        else:
            return ""
    '''        
                

print(recursive_json_key(json_data,"full_text"))