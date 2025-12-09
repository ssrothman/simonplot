import json
import re

with open("simon_mpl_config.json", "r") as f:
    config = json.load(f)

def lookup_axis_label(axiskey: str) -> str:
    #use regex to match axiskey with config keys:
    for key_pattern, label in config['axis_labels'].items():
        key_pattern = key_pattern.replace("*", ".*")  #convert wildcard * to regex .*
        if re.fullmatch(key_pattern, axiskey):
            return label
    return axiskey  #default to axiskey if no match found