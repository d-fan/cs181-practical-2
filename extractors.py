#!/usr/bin/python

from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

ffs = []

# Add extractors automatically by putting "@extractor" before
def extractor(feature_extractor):
    ffs.append(feature_extractor)
    return feature_extractor

"""
DLL file & address location
Registry key access
web addresses

"""

@extractor
def syscall_count(tree):
    """
    Counts the number of each system call and returns the result as a Counter
    (dict) mapping 'sys_call': count
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            # Increment our count of this syscall
            c[el.tag] += 1
            
    return c

@extractor
def dll_loads(tree):
    """
    Counts how many times a dll gets loaded by each program (should be 1 or 0)
    """
    c = Counter()
    for el in tree.iter():
        if el.tag == "load_dll" and "filename" in el.attrib:
            file_path = el.attrib["filename"]
            # Get the last part which should be *.dll
            file_name = file_path.split("\\")[-1].lower()
            # Soft assertion
            # if (len(file_name) != 0 and "dll" not in file_name):
            #     print "Bad dll: %s" % file_path
            c[file_name] += 1
    return c

@extractor
def reg_key_final_name(tree):
    return Counter();

@extractor
def reg_values(tree):
    """
    Looks at syscalls to 'query_value' and counts how many times each value was accessed
    """
    c = Counter()
    for el in tree.iter():
        if el.tag == "query_value" and "value" in el.attrib:
            # Increment our count of this syscall
            c[el.attrib["value"]] += 1
            
    return c

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
@extractor
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

@extractor
def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c
