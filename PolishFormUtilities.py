#Various Utilities for handling the polish form of formulas

def generate_numbered_template(template):
    """
    Generates a template with variables as numbers instead of the blanks generated
    by a template generator function
    
    Parameter:
    template: template with blanks
    
    Returns:
    numbered template
    """
    i = 0
    numbered_template = ""
    for elem in template:
        if elem=="_":
            numbered_template += str(i)
            i += 1
        else:
            numbered_template += elem
    return numbered_template

def split_along_value(polish, arity={"C": 2, "K": 2, "D": 2, "N": 1}):
    """
    Splits the polish formula along the n-arry function.
    This is equivalent to getting the input parts of the top-most 
    function in left to right order.
    Temporarily doesn't work with non-default arity parameters.
    
    Parameters:
    polish: polish formula to split
    arity: arity of the functions used
    
    Returns:
    Tuple of the polish formula of the split
    """
    literals = -1
    split_count = -1 * arity[polish[0]] + 1
    last_split = 1
    splits = []
    if split_count==0: #special case
        return tuple([polish[1:]]) #?? why do i need to do this?
    
    for i in range(1, len(polish)):
        if polish[i] in arity.keys():
            literals -= arity[polish[i]] - 1 #We only add (arity - 1) needed literals, 
                                             #best example is classical negation, we don't add any literals
                                             #as it just needs 1 that we already are waiting for
        else:
            literals += 1
            
        if literals==0:
            splits.append(polish[last_split:i+1])
            last_split = i+1
            literals = -1
            split_count += 1
            if split_count==0:
                splits.append(polish[last_split:])
                return tuple(splits)
    
    raise ValueError("Error \"split_along_value\" fell off the polish string.\nParameters:\n"+polish+"\n"+str(arity))

def template_to_cnf(template, constants=["O"]):
    """
    Converts a template into conjunctive normal form.
    This particular form is a list of list of literals.
    The outer list is the conjunctive block.
    The inner lists are the disjunctive blocks.
    The literals are either: 
        integers representing the variable number
        characters representing the constant
        sets of literals representing negation
        
    Boy de la Tour, Thierry Boy. 
        "An optimality result for clause form translation." 
        Journal of Symbolic Computation 14.4 (1992): 283-301.
        
    Parameters:
    template: template in polish notation
    
    Returns:
    template in conjunctive normal form
    """
    if template[0].isdigit(): #isa variable
        return [[int(template[0])]]
        
    if template[0] in constants: #isa constant
        return [[template[0]]]
    
    if template[0]=="C":
        split = split_along_value(template)
        return template_to_cnf("DN"+split[0]+split[1], constants=constants)
    
    if template[0]=="K":
        split = split_along_value(template)
        return template_to_cnf(split[0], constants=constants)+template_to_cnf(split[1], constants=constants)
    
    if template[0]=="D":
        split = split_along_value(template)
        out = []
        for spc1 in template_to_cnf(split[0], constants=constants):
            for spc2 in template_to_cnf(split[1], constants=constants):
                out.append(spc1+spc2)
        return out
    
    if template[0]=="N":
        if template[1].isdigit():
            return [[[int(template[1])]]]
        if template[1] in constants:
            return [[[template[1]]]]
        if template[1]=="N":
            return template_to_cnf(template[2:], constants=constants)
        if template[1]=="C":
            split = split_along_value(template[1:])
            return template_to_cnf("NDN"+split[0]+split[1], constants=constants)
        if template[1]=="K":
            split = split_along_value(template[1:])
            return template_to_cnf("DN"+split[0]+"N"+split[1], constants=constants)
        if template[1]=="D":
            split = split_along_value(template[1:])
            return template_to_cnf("KN"+split[0]+"N"+split[1], constants=constants)
        
    raise ValueError(template)
    
def fillin_template_with_fill(template, fill):
    """
    Fills a template with given fill.
    
    Parameters:
    template: template to used
    fill: fill to used
    
    Returns:
    formula formed by filling the given template with the given fill
    """
    i = 0
    tform = ""
    for j in range(len(template)):
        if template[j].isdigit():
            tform += str(fill[i])
            i += 1
        else:
            tform += template[j]
    return tform
    
    
    
    
    
    
    
    
    
    
    