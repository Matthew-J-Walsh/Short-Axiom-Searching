import numpy as np
from MathUtilities import *

def generate_fills(l):
    """
    Generate all fills of a specified length. Size of list scales with bells numbers.
    
    Parameters:
    l: length of fills
    
    Returns:
    Integer numpy array (bells(l) x l) of all the potential fills in numerical form. Each fill starts 
    with 0, each value after that is at most the supremum of all previous \'variables\' in the fill.
    """
    values = np.zeros((bells(l), l), dtype=int)
    
    if l==1:
        values[0][0] = 0
        return values
    
    l_fill_tree = generate_fills(l=l-1)
    
    nempty = 0
    for i in range(l_fill_tree.shape[0]):
        for j in range(np.amax(l_fill_tree[i])+1+1):
            for k in range(l-1):
                values[nempty, k] = l_fill_tree[i, k]
            values[nempty, l-1] = j
            nempty += 1
    if (np.unique(values, axis=0)!=values).all() or nempty!=bells(l):
        raise ValueError("Something wrong, size mismatch no gnerate_fills")
    
    return values

def generate_specification_lists(fills):
    """
    Generates all the specifications for a set of fills. Specifications
    discribe which fills are 1 varible more specific than the given fill.
    
    For example, at length 3 there are 5 fills:
    000, 001, 010, 011, 012
    000 has nothing more specific than it
    001 only has 000 as a specification (by 1=0)
    010 only has 000 as a specification (by 1=0)
    011 only has 000 as a specification (by 1=0)
    012 has 001, 010, 011, and 012 as specifications, but only
            001, 010, and 011 are 1 variable more specific
    So the specification list is: [[], [0], [0], [0], [1, 2, 3]]
    
    Parameter:
    fills: fills to generate specification lists for
    
    Returns:
    Specification lists
    """
    spec_pointers = []
    for i in range(fills.shape[0]):
        if (i%100==0):
            #This step can take a while (multiple minuites) 
            #so we spam the user to tell them its working
            print("Spec lists done with: "+str(i))
            
        specs = []
        for varnum in range(1, 1+np.amax(fills[i])):
            for setnum in range(0, varnum):
                #The objective of these 3 lines of code is to
                #generate a specification of fills[i]
                #that specification will have varnum be a reduced value
                #then to satisfy the requirements of a fill
                #all vars>varnum will need to be reduced by 1
                spec = np.copy(fills[i])
                spec[np.where(spec==varnum)] = setnum
                spec[np.where(spec>varnum)] -= 1
                
                #Get the location of the specification in the fills array
                specs.append(np.where(np.all(fills == spec, axis=1))[0][0])
        spec_pointers.append(specs)
    return spec_pointers

def subsumptive_cleave(i, tautologies, spec_lists):
    """
    Helper function for \"remove_subsumptions\". Does a \"cleave\" removing
    any tautology that is a more specific version of a given tautology.
    Do not be confused, this has NOTHING to do with cleaves for bindings.
    
    Parameters:
    i: index to cleave from (the given tautology)
    tautologies: numpy array of which fills are tautologies
    spec_lists: list of specification pointers for fills, see
        \"generate_specification_lists\"
        
    Returns:
    None, changes tautologies array
    """
    for j in range(len(spec_lists[i])):
        if tautologies[spec_lists[i][j]]:
            tautologies[spec_lists[i][j]] = False
            subsumptive_cleave(spec_lists[i][j], tautologies, spec_lists)

def remove_subsumptions(tautologies, spec_lists):
    """
    Removes the all tautologies that aren't most general from 
    the given set of fills that are tautologies.
    
    Parameters:
    tautologies: numpy array of which fills are tautologies
        Note: running this function will \'destroy\' this array
    spec_lists: list of specification pointers for fills, see
        \"generate_specification_lists\"
        
    Returns:
    numpy array of which fills are the most general fills
    """
    for i in range(tautologies.shape[0]-1,-1,-1):
        if tautologies[i]:
            subsumptive_cleave(i, tautologies, spec_lists)
    return tautologies