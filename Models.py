#This file has functions for handling the models, specifically for helping with verifying countermodels
#Most of these functions will not be generally used

import numpy as np
from TautologyChecking import *

#standard model ordering is:
#Length
#satisfaction
#C connective values in same order as: np.array([[1, 1], [0, 1]]) (standard model C)
#N connective values (if applicable)
#O constant value (if applicable)


def model_from_string(seq, type):
    """
    Generates the model defined by a string.
    
    Parameters:
    seq: the string
    type: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
    
    Returns:
    A usable model
    """
    domainsize = int(seq[0])
    if type=='N':
        model = {"connectives": {"N": np.empty(domainsize, dtype=int),
                                 "C": np.empty((domainsize, domainsize), dtype=int)},
                 "satisfaction": np.empty(domainsize, dtype=int),
                 "constants": {}}
    elif type in ['0', '1', 'O']:
        model = {"connectives": {"C": np.empty((domainsize, domainsize), dtype=int)},
                 "satisfaction": np.empty(domainsize, dtype=int),
                 "constants": {"O": None}}
    else:
        raise ValueError("Invalid Type: "+str(type))
        
        
    p = 1
    for i in range(domainsize):
        model["satisfaction"][i] = int(seq[p])
        p += 1
    for i in range(domainsize):
        for j in range(domainsize):
            model["connectives"]["C"][i][j] = int(seq[p])
            p += 1
    if type=='N':
        for i in range(domainsize):
            model["connectives"]["N"][i] = int(seq[p])
            p += 1
    elif type in ['0', '1', 'O']:
        model["constants"]["O"] = 0 #This probably deserves an explination... 
                                    #vampire will return constants in the equation
                                    #in the first position, aka ["satisfaction"][0]
                                    #the constant value just points to this so its 0
    
    return model
    
def verify_countermodel_line(line, type):
    """
    Verifies a line in a file containing a countermodels.
    
    Parameters:
    line: line in file
    type: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
    
    Returns
    True if verified, False if something is wrong (NOTE: Errors will be caught and 
    ignored and just return False)
    """
    if ":" in line:
        polish, modelstring = line.split(":")
        try:
            model = model_from_string(modelstring, type)
            return check_against_model(polish, model)
        except:
            return False
    return True
    
def verify_countermodel_file(filename, type):
    """
    Verifies a file containing countermodels.
    
    Parameters:
    filename: name of file
    type: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
    
    Returns
    True if file is verified, False if something is wrong (NOTE: Errors will be caught and 
    ignored and just return False), the first line with an issue will be printed.
    """
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line and line!="":
            if not verify_countermodel_line(line, type):
                print("Line: "+str(line)+" either causes an error or is not a valid countermodel.")
                return False
            
            line = f.readline().strip()
    return True
    
def get_all_countermodels(filename):
    """
    Extracts all the countermodels in a file as strings.
    
    Parameter:
    filename: name of file
    
    Returns:
    list of strings describing countermodels, sorted for convience
    """
    all_models = set()
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line and line!="":
            if ":" in line:
                 polish, modelstring = line.split(":")
                 all_models.add(modelstring)
            line = f.readline().strip()
    return sorted(list(all_models))

def get_all_countermodels_as_models(filename, type):
    """
    Extracts all the countermodels in a file as countermodels.
    
    Parameter:
    filename: name of file
    
    Returns:
    list of countermodels, sorted for convience
    """
    return [model_from_string(m, type) for m in get_all_countermodels(filename)]

def extract_remaining_candidates(inputfile, outputfile):
    """
    Extracts all the remaining candidates in a file to another file
    
    Parameters:
    inputfile: name of input file
    outputfile: name of file to place remaining candidates
    """
    with open(inputfile, 'r') as inf:
        with open(outputfile, 'w') as ouf:
            line = inf.readline().strip()
            while line and line!="":
                if ":" not in line:
                    ouf.write(line+"\n")
                line = inf.readline().strip()








