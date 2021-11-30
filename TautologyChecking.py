#models are of form (t, i, n)
import numpy as np
#import warnings
var_language = [str(i) for i in range(0, 26)]
from PolishFormUtilities import split_along_value

def np_var_broadcast(arr, idx, val, set_val):
    """
    Im so sorry. I know of no better way of handling this.
    Does a broadcast on a variable dimension of a numpy array

    Parameters:
    arr: array to be broadcasted
    idx: index of dimension to broadcast across
    val: value in dimension to broadcast to
    set_val: value to set broadcasted values to

    Returns:
    None
    """
    exec("arr["+":,"*idx+"val] = set_val")
    
def seperate_along_function_polish(polish):
    """
    Seperate values in polish notation along the top function

    Parameters:
    polish: polish of formula
    funcs: dictionary of the function names and their arrity

    Returns:
    Tuple of polish values for the splits in the function
    """
    raise ValueError("seperate_along_function_polish not properly implemented yet")

def check_against_model_helper(polish, model, fills, debug):
    """
    Recursive component of checking against a model.
    
    Parameters:
    polish: polish of formula
    model: model being checked against
    fills: number of independent variables
    debug: in debug mode or not
    
    Returns:
    Array representing truth of this formula given model.
    """
    if polish[0] in model["connectives"].keys():
        phrases = split_along_value(polish)
        inners = []
        phrasings = []
        for i in range(len(phrases)):
            inners.append(check_against_model_helper(phrases[i], model, fills, debug))
            phrasings.append("inners["+str(i)+"]")
            
        results = eval("model[\"connectives\"][polish[0]]["+",".join(phrasings)+"]")
        return results
    elif polish[0] in model["constants"].keys():
        results = np.full((len(model["satisfaction"]),)*fills, model["constants"][polish[0]], dtype=int)
        return results
    else: #literal
        results = np.zeros((len(model["satisfaction"]),)*fills, dtype=int)
        for i in range(len(model["satisfaction"])):
            np_var_broadcast(results, int(polish[0]), i, i)
        res = results
        if debug:
            print(polish+": "+str(res))
        return res
    
def check_against_model(polish, model, debug=False):
    """
    Checks a polish formula written with Cs, Ns, and whole numbers for variables against a given model.
    
    Parameters:
    polish: polish of the formula being checked against
    model: model to use
    debug: debug mode or not
    
    Returns:
    True if tautological, otherwise False    
    """
    for c in list(set(polish)):
        if not c.isdigit():
            if c not in model["connectives"] and c not in model["constants"]:
                raise ValueError(c+" is not defined in the model.")
    fills = 0
    fillvals = []
    for e in polish:
        if e!="C" and e!="N":
            if e not in fillvals:
                fillvals.append(e)
                fills += 1
    helper_out = check_against_model_helper(polish, model, fills, debug)
    sat_res = model["satisfaction"][helper_out]
    if np.count_nonzero(sat_res)==(len(model["satisfaction"])**fills):
        return True
    return False
        
#This is the model for C + N with C and N defined classically
STANDARD_MODEL = {"connectives": {"N": np.array([1, 0]), "C": np.array([[1, 1], [0, 1]])}, "satisfaction": np.array([0, 1]), "constants": {}}
C_AND_ONE_MODEL = {"connectives": {"N": np.array([1, 0]), "C": np.array([[1, 1], [0, 1]])}, "satisfaction": np.array([0, 1]), "constants": {"O": 1}}
C_AND_ZERO_MODEL = {"connectives": {"N": np.array([1, 0]), "C": np.array([[1, 1], [0, 1]])}, "satisfaction": np.array([0, 1]), "constants": {"O": 0}}












