import numpy as np

#Name of file lies a little, this doesn't define anything conjunctive normal form wise, nor does 
#it convert to conjunctive normal form, it just handles the conjunctive normal form

def cnf_to_bindings(cnf, Ovalue=None):
    """
    Converts conjunctive normal form of template (see template_to_cnf) to 
    bindings statements. Special case of None return means that entire 
    template is unsatifiable and no fills will result in a tautology.
    The bindings statements indicate which variables must be the same to satisfy
    the template.
    
    Parameters:
    cnf: conjunctive normal form of a template
    Ovalue: boolean value of \"O\" in the language, either True or False
    
    Returns:
    None if no fills would be able to be made from template
    Otherwise a list of lists of bindings. The other list is a conjunction of inner lists.
    The inner lists are disjunctions of bindings, one of which must be true 
    for the disjunctive block to be true.
    """
    bindings = []
    for block in cnf:
        negs = []
        pos = []
        blocksat = False
        for e in block:
            if type(e)==type([]):
                if e[0]=="O":
                    if not Ovalue:
                        blocksat = True
                else:
                    negs.append(e[0])
            else:
                if e=="O":
                    if Ovalue:
                        blocksat = True
                else:
                    pos.append(e)
        if blocksat:
            continue
        binds = []
        for neg in negs:
            for non in pos:
                if neg<non:
                    binds.append((neg, non))
                else:
                    binds.append((non, neg))
        bindings.append(binds)
    return bindings
    
def generate_cleaves(fills):
    """
    Generate all cleaves from the list of fills. A cleave is an indication of what bindings 
    exist within the each fill.
    
    Parameter:
    fills: fills to generate cleaves for
    
    Returns:
    Boolean numpy array (number of fills x fills length x fills length).
    [f, x, y] is true when the f fill has its x'th variable equal its y'th variable, otherwise its false.
    """
    cleaves = np.zeros((fills.shape[0], fills.shape[1], fills.shape[1]), dtype=bool)
    
    for i in range(cleaves.shape[0]):
        for x in range(fills.shape[1]):
            for y in range(fills.shape[1]):
                if fills[i, x]==fills[i, y]:
                    cleaves[i, x, y] = 1
    
    return cleaves

def bindings_to_tautologies(bindings, cleaves):
    """
    Converts the bindings statements (that are in conjunctive normal form) for
    a given template into an array indiciating which fills are tautologies.
    
    Parameters:
    bindings: conjunctive normal form of bindings
    cleaves: cleaves for fills
    
    Returns:
    numpy array indiciating which fills are tautologies
    """
    if len(bindings)==0:
        return np.ones_like(cleaves[:, 0, 0], dtype=bool)
    cleave_field = []
    for conjblock in bindings:
        if len(conjblock)==0:
            return np.zeros_like(cleaves[:, 0, 0], dtype=bool)
        sfield = []
        for eqst in conjblock:
            sfield.append(cleaves[:, eqst[0], eqst[1]])
        cleave_field.append(sfield)
    try:
        return np.logical_and.reduce([np.logical_or.reduce(x) for x in cleave_field])
    except:
        raise ValueError(bindings)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        