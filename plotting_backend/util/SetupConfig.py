import json
import re

with open("simon_mpl_config.json", "r") as f:
    config = json.load(f)

def strip_collection_names(axiskey:str) -> str:
    split_wrt_dots = axiskey.split('.')

    stripped_parts = []
    for part in split_wrt_dots[:-1]:
        splitted_part = re.split(r'[_(),]', part)

        #we want to remove the last element,
        #but put back in all the separators that were removed
        stripped_parts.append(part[:-len(splitted_part[-1])])

    stripped_axiskey = ''.join(stripped_parts)
    stripped_axiskey+=split_wrt_dots[-1]  #add last part back

    return stripped_axiskey

def attempt_regex_match(pattern: str, axiskey: str) -> bool:
    #escape special characters
    pattern_escaped = re.escape(pattern)
    #replace '*' wildcard with alphanumeric match
    pattern_escaped = pattern_escaped.replace(r'\*', r'[a-zA-Z0-9]*')
    #ensure no leading or trailing characters
    pattern_escaped = "^" + pattern_escaped + "$" 

    return re.fullmatch(pattern_escaped, axiskey) is not None

'''
Utility function to lookup axis labels from config

Lookup logic is to match the given axiskey against the keys in config['axis_labels'] using two different strategies (in order of precedence):
1. Exact match
2. Match ignoring the collection name (e.g. allowing "MergedSimClustersECALHCAL.track_pdgId" to match "track_pdgId" in the config)
    For complex/compound Variables (ie represented by anything other than BasicVariable objects) 
    there is some nontrivial logic needed to identify the collectionname part(s) of the axiskey.
    This logic depends on the strong assumption that:
       - Collection names do not contain underscores, parentheses, dots, or commas (I believe this is enforced by ROOT/CMSSW)
       - The key identifiers for complex/compound variables separate sub-variable keys with underscores, parentheses, or commas (but never dots)
       - Dots ONLY appear in axiskeys to separate collection names from variable names

    If there is no '.' in the axiskey then matchtype 2 is not applied, since there is no collection name to strip
    
    Strategy 2 only attempts matches agains config keys that do not themselves contain '.' characters, since those refer to specific collections

Both strategies support ONLY the '*' wildcard character, with all other regex special characters escaped (eg '.', '+', etc)
Furthermore, the behavior of the '*' wildcard is modified to only match alphanumeric characters [a-zA-Z0-9]


For example, consider the following config:
{
    "axis_labels": {
        "x": "x [cm]",
        "RecHit*.x" : "RecHit x [cm]",
        "energy_minus_simenergy_over_simenergy" : "(E - E_sim) / E_sim"
    }
}
A variable like BasicVariable("X") will match via strategy 1 to yield "x [cm]".
A variable like BasicVariable("x", 'GenVtx') will fail strategy 1, but match with the first key via strategy 2 to yield the label "x [cm]".
A variable like BasicVariable("x", 'RecHitECAL') will match via strategy 1 to yield "RecHit x [cm]".
A variable like RelativeResolutionVariable(
    BasicVariable("simenergy", "RecHitECAL"), 
    BasicVariable("energy", "RecHitECAL")
) will have axiskey "RecHitECAL.energy_minus_RecHitECAL.simenergy_over_RecHitECAL.simenergy"
which will fail strategy 1, but match with the third key via strategy 2 to yield "(E - E_sim) / E_sim".
'''
def lookup_axis_label(axiskey: str) -> str:
    #first pass with strategy 1
    for key_pattern, label in config['axis_labels'].items():
        if attempt_regex_match(key_pattern, axiskey):
            return label
        
    #second pass with strategy 2
    if '.' in axiskey: 
        stripped_axiskey = strip_collection_names(axiskey)
        for key_pattern, label in config['axis_labels'].items():
            if '.' in key_pattern:
                continue  #strategy 2 not applicable

            if attempt_regex_match(key_pattern, stripped_axiskey):
                return label

    print("WARNING: No axis label in config for key:", axiskey)
    return axiskey  #default to axiskey if no match found
        

def check_auto_logx(axiskey: str) -> bool:
    for pattern in config['auto_logx_patterns']:
        if attempt_regex_match(pattern, axiskey):
            return True
        
    if '.' in axiskey:
        stripped_axiskey = strip_collection_names(axiskey)
        for pattern in config['auto_logx_patterns']:
            if '.' in pattern:
                continue  #strategy 2 not applicable
            
            if attempt_regex_match(pattern, stripped_axiskey):
                return True
            
    return False