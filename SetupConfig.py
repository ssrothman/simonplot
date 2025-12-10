import json
import re

with open("simon_mpl_config.json", "r") as f:
    config = json.load(f)

def lookup_axis_label(axiskey: str) -> str:
    #use regex to match axiskey with config keys:
    for key_pattern, label in config['axis_labels'].items():
        key_pattern = key_pattern.replace('.', r'\.')  #escape dots
        key_pattern = key_pattern.replace("*", "[a-zA-Z0-9]*")  #convert wildcard * to regex for any alphanumeric characters [but NOT underscores]
        #need to match with no trailing characters:
        key_pattern = "^" + key_pattern + "$"
        if re.fullmatch(key_pattern, axiskey):
            return label
    return axiskey  #default to axiskey if no match found