#This file contains the functions used for generating all templates given specifications.

def valid_CN_inputs_given_length(length):
    """
    What are the valid inputs for \"yield_all_theorem_templates_full_CN\" at a given symbol length
    
    Parameters:
    length: objective symbol length, odd number
    
    Returns:
    List of tuples in (v, i, n) order for inputs into \"yield_all_theorem_templates_full_CN\"
    In general for all of these valid_input_generator functions, first value is variable count
    """
    valid_inputs = []
    for v in range(2, length):
        i = v - 1
        n = length - v - i
        if n <= v + i and n >= 0:
            valid_inputs.append((v, i, n))
    return valid_inputs
    

def yield_all_theorem_templates_full_CN(v=10, i=9, n=4, o=0, ln=False):
    """
    Recursive function that generates all templates for C and N within specifications in polish notation.
    
    Parameters:
    v: Number of variables remaning
    i: number of material implications remaining
    n: number of negations remaining
    o: number of literal placeable
    ln: Boolean indicating if a negation is placeable next (prevents double negations)
    
    Yields:
    All templates with blanks \"_\" representing variable locations    
    """
    if v==0:
        yield ""
    else:
        if n>0 and not ln: 
            for theorem in yield_all_theorem_templates_full_CN(v=v, i=i, n=n-1, o=o, ln=True):
                yield "N"+theorem
        if i>0 and i+1==v: #have to place this "C"
            for theorem in yield_all_theorem_templates_full_CN(v=v, i=i-1, n=n, o=o+2, ln=False):
                yield "C"+theorem
        else:
            if i>0:
                for theorem in yield_all_theorem_templates_full_CN(v=v, i=i-1, n=n, o=o+2, ln=False):
                    yield "C"+theorem
            if o>0:
                if n==0 or v>1: #have to place this "N_"
                    for theorem in yield_all_theorem_templates_full_CN(v=v-1, i=i, n=n, o=o-1, ln=False):
                        yield "_"+theorem

def valid_CO_inputs_given_length(length):
    """
    What are the valid inputs for \"yield_all_theorem_templates_full_CO\" at a given symbol length
    
    Parameters:
    length: objective symbol length, odd number
    
    Returns:
    List of tuples in (v, n, i) order for inputs into \"yield_all_theorem_templates_full_CO\"
    In general for all of these valid_input_generator functions, first value is variable count
    """
    valid_inputs = []
    i = int((length - 1) // 2) #I dont trust python much
    for v in range(1, int((length - 1) // 2) + 1):
        n = length - i - v
        if v >= 1 and n >= 0:
            valid_inputs.append((v, n, i))
    return valid_inputs

def yield_all_theorem_templates_full_CO(v=10, n=0, i=9, o=0):
    """
    Recursive function that generates all templates for C and O within specifications in polish notation.
    Works for both C + 1 and C + 0
    
    Parameters:
    v: Number of variables remaning
    n: number of constants remaining
    i: number of material implications remaining
    o: number of literal placeable
    
    Yields:
    All templates with blanks \"_\" representing variable locations    
    """
    if v==0 and n==0:
        yield ""
    else:
        if i>0 and i+1==v+n: #have to place this "C"
            for theorem in yield_all_theorem_templates_full_CO(v=v, n=n, i=i-1, o=o+2):
                yield "C"+theorem
        else:
            if i>0:
                for theorem in yield_all_theorem_templates_full_CO(v=v, n=n, i=i-1, o=o+2):
                    yield "C"+theorem
            if o>0:
                if n>0:
                    for theorem in yield_all_theorem_templates_full_CO(v=v, n=n-1, i=i, o=o-1):
                        yield "O"+theorem
                if v>0:
                    for theorem in yield_all_theorem_templates_full_CO(v=v-1, n=n, i=i, o=o-1):
                        yield "_"+theorem
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        