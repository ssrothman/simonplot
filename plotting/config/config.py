import json
import os

from simon_mpl_util.util.dictmerge import merge_dict

#first lookup path to the directory containing this file
current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, 'default.json'), 'r') as f:
    defaults = json.load(f)

#then look for local user config
if os.path.exists("simon_mpl_config.json"):
    with open("simon_mpl_config.json", "r") as f:
        user_config = json.load(f)
else:
    user_config = {}

config = merge_dict(defaults, user_config, allow_new_keys=False, replace_dict=['axis_labels'])