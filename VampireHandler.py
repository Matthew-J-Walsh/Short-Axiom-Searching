 22.49 KB

import numpy as np
import os
import copy
import sys
import itertools
import re
from datetime import datetime
 
from TautologyChecking import *
from PolishFormUtilities import *
from Models import *
 
 
#counter_axiom = 'i(i(i(X,Y),i(o,Z)),i(U,i(i(Z,X),i(V,i(W,X)))))' #C1 old
#counter_axiom = 'i(i(i(i(i(X,Y),i(Z,o)),U),V),i(i(V,X),i(Z,X)))' #C0?
#counter_axiom = 'i(i(i(X,Y),i(Z,i(o,U))),i(i(U,X),i(Z,i(V,X))))'
#counter_axiom = 'i(i(X,Y),i(i(Y,Z),i(X,Z)))' #C0 infinite transitivity
#counter_axiom = 'i(i(i(X,o),i(Y,o)),i(Y,X))' #C0 L4
#counter_axiom = 'i(X,i(Y,X))' #C0 L1
#counter_axiom = 'd(n(d(n(d(X,Y)),n(Z))),n(d(n(d(n(V),V)),d(n(Z),X))))=Z' #DN single axiom DN-13345
#counter_axiom = 'i(i(i(X,Y),i(o,Z)),i(U,i(i(Z,X),i(V,i(W,X)))))' #C1 old
#counter_axiom = 'i(i(i(i(i(X,Y),i(Z,o)),U),V),i(i(V,X),i(Z,X)))' #C0?
#counter_axiom = 'i(i(i(X,Y),i(Z,i(o,U))),i(i(U,X),i(Z,i(V,X))))'
 
 
def Run_Vampire_Elimination(input_file_folder, input_file_name, counter_formulas, type_, **kwargs):
    """
    Runs the vampire elimination process in its entirity.
 
    Parameters:
    input_file_folder: the folder where the input file is
    input_file_name: name of the input file in the folder
    counter_formulas: formulas to use to generate counter examples (models).
    Only the last one will be used for the individual files in the subfolder.
    type_: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
 
    Returns:
    Nothing.
    Makes a new file in the input_file_folder that countains the same lines as the 
    input file but with some lines containing counter examples (models). Makes a new 
    folder in the input_file_folder countaining a file for each of formulas without 
    counter examples.
    """
    for i in range(len(counter_formulas)):
        progressive_filtering(input_file_name, input_file_name[:-4]+"-CounterModels"+str(i)+".txt", 
            input_file_folder, type, fof_counter_formula=counter_formulas[i], **kwargs)
    print("There are "+str(count_remaining(input_file_folder+input_file_name[:-4]+"-CounterModels"+str(len(counter_formulas)-1)+".txt"))+" formulas remaining to be eliminated, creating subfolder and files")
    dump_remaning_into_file(input_file_folder+input_file_name[:-4]+"-CounterModels"+str(len(counter_formulas)-1)+".txt", input_file_folder+"specialcases"+input_file_name[:-4]+"-Remaining.txt")
    seperate_formula_into_files(input_file_folder+"specialcases"+input_file_name[:-4]+"-Remaining.txt", input_file_folder+"specialcases"+input_file_name[:-4], counter_formulas[-1], **kwargs)
 
 
def generate_polish_form(fof_form, **kwargs):
    """
    Generates the polish form of a fof formula.
 
    Parameters:
    fof_form: fof formula
 
    Returns:
    polish version of fof formula
    """
    fof_to_polish_translation = {} #i'm sorry, but this is going to barely slow anything down
    if 'polish_to_fof_translation' in kwargs.keys():
        polish_to_fof_translation = kwargs['polish_to_fof_translation']
    else:
        polish_to_fof_translation = {"E": "=", "C": ('i', 2), "D": ('d', 2), "N": ('n', 1), "O": "o"}
    for entry in polish_to_fof_translation:
        if type(entry[1])==type(()):
            k = entry[1][0]
        else:
            k = entry[1]
        fof_to_polish_translation.append({k: entry[0]})
 
    fof_stripped = fof_form.replace("(", "").replace(")", "").replace(",", "")
    polish = ""
    for c in fof_stripped:
        if c in fof_to_polish_translation.keys():
            polish.append(fof_to_polish_translation[c])
        else:
            polish.append(str(ord(c)-ord("A")))
 
    return polish
 
 
def generate_fof_form(polish, **kwargs):
    """
    Generates the fof form of a polish formula.
    Raises consecutive ValueErrors if input is not polish.
 
    Parameters:
    polish: polish formula
 
    Returns:
    fof version of polish formula
    """
    if 'fof_vars_list' in kwargs.keys():
        fof_vars_list = kwargs['fof_vars_list']
    else:
        fof_vars_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
 
    if 'polish_to_fof_translation' in kwargs.keys():
        polish_to_fof_translation = kwargs['polish_to_fof_translation']
    else:
        polish_to_fof_translation = {"E": "=", "C": ('i', 2), "D": ('d', 2), "N": ('n', 1), "O": "o"}
 
    return generate_fof_form_helper(polish, fof_vars_list, polish_to_fof_translation)
 
fof_vars_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
 
def generate_fof_form_helper(polish, fof_vars_list, polish_to_fof_translation):
    try:
        if polish[0] not in polish_to_fof_translation.keys():
            return fof_vars_list[int(polish)]
        elif polish_to_fof_translation[polish[0]]=="=":
            p = split_along_value(polish)
            return generate_fof_form_helper(p[0], fof_vars_list, polish_to_fof_translation)\
              +"="+generate_fof_form_helper(p[1], fof_vars_list, polish_to_fof_translation)
        elif type(polish_to_fof_translation[polish[0]])==type(""):
            return polish_to_fof_translation[polish[0]]
        elif type(polish_to_fof_translation[polish[0]])==type(()):
            p = split_along_value(polish)
            out = polish_to_fof_translation[polish[0]][0]+"("
            for ps in p:
                out += generate_fof_form_helper(ps, fof_vars_list, polish_to_fof_translation)+","
            return out[:-1]
 
        raise ValueError("Unknown Element in polish_to_fof_translation")
    except:
        raise ValueError(polish)
 
def get_fof_form_vars(fof_form, **kwargs):
    """
    Returns the list of vars in a fof form.
 
    Parameters:
    fof_form: The fof form
 
    Returns:
    List of variables used in formula
    """
    if 'disallowed_symbols' in kwargs.keys():
        disallowed_symbols = kwargs['disallowed_symbols']
    else:
        disallowed_symbols = ['i', 'n', '=', 'd', 't', 'o', '(', ')', ',']
 
    var_list = []
    for c in fof_form:
        if c not in disallowed_symbols:
            if c not in var_list:
                var_list.append(c)
    return var_list
 
def generate_fof_test_file_text(polishes, fof_counter_formula, **kwargs):
    """
    Generates the text to place in a fof file to (atempt to) disprove a 
    set of polish formulas' ability to be an axiom system.
 
    Parameters:
    polishes: List of polish formulas that make up the axioms of a logic.
    fof_counter_formula: Usually a tautology/axiom of a logic system, the
    conjecture that it will be shown cannot be derived from the axioms given.
    system_constants: The constants in the axiom system, in the form of a list.
    equational: Is the axiom system equational or not
 
    Returns:
    File text for an fof file that will be used to either prove or disprove 
    a set of polish formulas' ability to be an axiom system.
    """
    file = ''
    system_constants = []
    equational = False
 
    if 'polish_to_fof_translation' in kwargs.keys():
        for ptft in kwargs['polish_to_fof_translation'].values():
            if type(ptft)==type(0) and ptft!="=":
                system_constants.append(ptft)
        if "=" in kwargs['polish_to_fof_translation'].values():
            equational = True
 
    if not equational:
        file = 'fof(mp,axiom, ![X,Y]: ((t(X) & t(i(X,Y))) => t(Y))).\n' #MP
        file += 'fof(counter, conjecture, ![X,Y,Z,U,V]: t('+fof_counter_formula+')).\n'
    else:
        file = 'fof(counter, conjecture, ![X,Y,Z,U,V]: '+fof_counter_formula+').\n'
 
    for constant in system_constants:
        file += 'fof(const, axiom, t('+constant+')).\n'
 
    i = 0
    for polish in polishes:
        fof_form = generate_fof_form(polish, **kwargs)
        var_list = get_fof_form_vars(fof_form, **kwargs)
        if not equational:
            file += 'fof(cand'+str(i)+', axiom, !['+var_list[0]+''.join([','+c for c in var_list[1:]])\
                    +']: t('+fof_form+')).\n'
        else:
            file += 'fof(cand'+str(i)+', axiom, !['+var_list[0]+''.join([','+c for c in var_list[1:]])\
                    +']: '+fof_form+').\n'
        i += 1
    return file
 
def generate_fof_test_file(polishes, file_name, fof_counter_formula, **kwargs):
    """
    Generates an fof test file. See generate_fof_test_file_text for more details.
 
    Parameters:
    polishes: List of polish formulas that make up the axioms of a logic.
    fof_counter_formula: Usually a tautology/axiom of a logic system, the
    conjecture that it will be shown cannot be derived from the axioms given.
    system_constants: The constants in the axiom system, in the form of a list.
    equational: Is the axiom system equational or not
 
    Returns:
    File name of created file.
    """
    if os.path.exists(file_name):
        raise Warning("FILE ALREADY EXISTS. DELETE IT.")
    with open(file_name, "w") as f:
        f.write(generate_fof_test_file_text(polishes, fof_counter_formula, **kwargs))
    return file_name
 
def run_fof_test_on_formula(lines, 
        fof_counter_formula='i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X)))',
        verify_model=False, **kwargs):
    """
    Runs a set of lines in a file through vampire, each line is a potential axiom system.
 
    Parameters:
    lines: lines to run through vampire
    fof_counter_formula: the counter formula to be used, in fof form
    verify_model: whether or not to verify the model after it is found, only really used for
        double checking/debugging
 
    Returns:
    The model, if one was found, otherwise None
    """
    file_name = 'fof_'+str(datetime.now())
    for i in ['-', ' ', ':', '.']: 
        file_name = file_name.replace(i, '')
    file_name += '.p'
    polishes = []
    for l in lines:
        for s in l.split(","):
            polishes.append(s.strip())
    generate_fof_test_file(polishes, file_name, fof_counter_formula, **kwargs)
    time = 240
    if "vampire_time" in kwargs.keys():
        time = kwargs["vampire_time"]
    out_read = os.popen('./vampire --mode casc_sat -t '+str(time)+' --fmb_start_size 2 '+file_name)
    out = out_read.read()
    if False:
        out = out.replace(",o", ",fmb_$i_1").replace("o,", "fmb_$i_1,").replace(" = o", " = fmb_$i_1").replace("(o)", "(fmb_$i_1)")
 
    if "Finite Model Found!" in out:
        #print(out)
        try:
            seq = get_vampire_model(out)
        except:
            print(out)
            raise ValueError("Hmm")
 
        #Verify model actually works
        #This looks like a mess but its actually really nice output
        if verify_model:
            try:
                model = get_usable_model_of_sequence(seq)
            except:
                raise ValueError(out)
            print("Model checking: "+seq+"\n"+model+"\n\n")
            for i in range(len(polishes)):
                print("Checking model on: "+polishes[i])
                if not check_against_model(polishes[i], model):
                    print("Didn't work?\n"+str(polishes[i])+"\n"+str(model))
                    raise ValueError(out)
            if check_against_model(generate_polish_form(fof_counter_formula), model):
                print("It worked on the counterformula wut?")
 
        os.remove(file_name)
        return True, seq #change me
    else:
        os.remove(file_name)
        return False, None #change me
 
def readnextIncompleteLine(file, nfile, write_new_file=True):
    """
    Reads the next line of a file that doesn't already have a model (indicating that its
    been disproven)
 
    Parameters:
    file: main file
    nfile: new file being made
    write_new_file: if one needs to write to the new file
 
    Returns:
    The next line without a model
    """
    line = file.readline()
    while ":" in line:
        if write_new_file:
            nfile.write(line)
        line = file.readline()
    return line
 
 
def progressive_filtering(input_file_name, output_file_name, file_folder, type_,
        fof_counter_formula='i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X)))', **kwargs):
    """
    Progressivly filters a file of lines of potential axioms with a counter formula.
 
    Parameters:
    input_file_name: file of lines of potential axioms
    output_file_name: file to output filtered axioms
    file_folder: folder were the files are located
    type_: type of system this is
    fof_counter_formula: counter formula to base filtering off of
 
    Returns:
    Nothing. Creates the file file_folder+output_file_name and fills it line by line
    with lines from the input_file_name with ":" + model if there is a counter model.
    """
    models = []
    model_seqs = []
 
    with open(file_folder+"temp_"+output_file_name, 'w') as nfile:
        with open(file_folder+input_file_name, 'r') as file:
            line = readnextIncompleteLine(file, nfile)
            while line:
                modeled = False
                for i in range(len(models)):
                    model = models[i]
                    if check_against_model(line.strip(), model):
                        modeled = True
                        nfile.write(line.strip()+":"+model_seqs[i]+"\n")
                        break
                if not modeled:
                    b, seq = run_fof_test_on_formula([line], fof_counter_formula=fof_counter_formula, **kwargs)
                    if not b:
                        nfile.write(line)
                    else:
                        nfile.write(line.strip()+":"+seq+"\n")
                        model_seqs.append(seq)
                        models.append(model_from_string(seq, type_))
 
    with open(file_folder+output_file_name, 'w') as ofile:
        with open(file_folder+"temp_"+output_file_name, 'a') as file:
            line = readnextIncompleteLine(file, ofile)
            while line:
                modeled = False
                for i in range(len(models)):
                    model = models[i]
                    if check_against_model(line.strip(), model):
                        modeled = True
                        ofile.write(line.strip()+":"+model_seqs[i]+"\n")
                        break
                if not modeled:
                    ofile.write(line)
                line = readnextIncompleteLine(file, ofile)
 
    os.remove(file_folder+"temp_"+output_file_name)
 
def verify_dual_file_integrity(fileAname, fileBname, type_):
    """
    Verify's that two files refer to the same list of axiom systems.
    Raises error if they dont
 
    Parameters:
    fileAname: name of first file
    fileBname: name of second file
    type_: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
 
    Return:
    Nothing. Raises errors if the files aren't the same.
    """
    print("Verifying file integrity")
    with open(fileAname, 'r') as fileA:
        with open(fileBname, 'r') as fileB:
            lineA = fileA.readline().strip()
            lineB = fileB.readline().strip()
            while(lineA and lineB):
                if ":" in lineA:
                    if ":" in lineB:
                        if lineA != lineB:
                            print(lineA)
                            print(lineB)
                            raise ValueError("Wrong compare")
                    else:
                        if lineB not in lineA:
                            print(lineA)
                            print(lineB)
                            raise ValueError("Wrong compare")
                else:
                    if ":" in lineB:
                        if lineA not in lineB:
                            print(lineA)
                            print(lineB)
                            raise ValueError("Wrong compare")
                    else:
                        if lineA != lineB:
                            print(lineA)
                            print(lineB)
                            raise ValueError("Wrong compare")
                if ":" in lineA:
                    tmp = lineA.split(":")
                    if not check_against_model(tmp[0], model_from_string(tmp[1], type_)):
                        print(lineA)
                        raise ValueError("Wrong seq")
                if ":" in lineB:
                    tmp = lineB.split(":")
                    if not check_against_model(tmp[0], model_from_string(tmp[1], type_)):
                        print(lineB)
                        raise ValueError("Wrong seq")
                lineA = fileA.readline().strip()
                lineB = fileB.readline().strip()
 
def verify_single_file_integrity(filename, type_, forcefill=False):
    """
    Checks that a single file is properly formed and has consistent models
 
    Parameters:
    filename: name of file
    type_: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
    forcefill: should an error be thrown when there is an line without a model?
 
    Returns
    Nothing. Raises errors if the file not correct
    """
    print("Verifying the integrity of: "+filename)
    with open(filename, 'r') as file:
        line = file.readline().strip()
        while(line):
            if ":" in line:
                tmp = line.split(":")
                if not check_against_model(tmp[0], model_from_string(tmp[1], type_)):
                    print(line)
                    raise ValueError("Wrong seq")
            else:
                if forcefill:
                    raise ValueError("Not filled")
            line = file.readline().strip()
 
def count_remaining(filename):
    """
    Count how many potential axiom sets remain in a file.
 
    Parameters:
    filename: name of file
 
    Returns:
    How many lines there arn't counter examples for
    """
    c = 0
    with open(filename, 'r') as file:
        line = readnextIncompleteLine(file, None, write_new_file=False)
        c += 1
        while(line):
            line = readnextIncompleteLine(file, None, write_new_file=False)
            c += 1
    print(str(c)+" remaining in file: "+filename)
    return c
 
def seperate_formula_into_files(basefile, target_directory, counter_axiom, **kwargs):
    """
    Seperates the formulas in one file into the individual files in the target directory.
 
    Parameters:
    basefile: file from which to draw formulas from
    target_directory: directory to place individual formula files into
    counter_axiom: counter axiom to use in individual files
 
    Returns:
    Nothing. Seperates the remaining (no counter models) formulas into individual files in the
    target directory.
    """
    with open(basefile, 'r') as bfile:
        line = bfile.readline()
        while(line):
            content = generate_fof_test_file_text([line.strip()], 
                                                        counter_axiom, **kwargs)
            with open(target_directory+line.strip()+".p", 'w') as nf:
                nf.write(content)
            line = bfile.readline()
 
def dump_remaning_into_file(file_name, outfile_name):
    """
    Dump remaining formulas (no counter examples) to a new file. Usually done before using
    seperate_formula_into_files and then running each file individually.
    """
    with open(file_name, 'r') as file:
        with open(outfile_name, 'a') as ofile:
            line = file.readline()
            while(line):
                if ":" not in line:
                    polish = line.strip()
                    ofile.write(polish+'\n')
                line = file.readline()
 
domain_regex = re.compile("tff\(declare_\$i(\d+)")
i_regex = re.compile("(i\(fmb_\$i_\d+,fmb_\$i_\d+\) = fmb_\$i_\d+)")
i_regex_vals = re.compile("i\(fmb_\$i_(\d+),fmb_\$i_(\d+)\) = fmb_\$i_(\d+)")
n_regex = re.compile("(n\(fmb_\$i_\d+\) = fmb_\$i_\d+)")
n_regex_vals = re.compile("n\(fmb_\$i_(\d+)\) = fmb_\$i_(\d+)")
t_regex = re.compile("(\s~?)t\(fmb_\$i_(\d+)\)")
def get_vampire_model(vampire_txt, type_):
    """
    Generates a model from a vampire output file.
 
    Parameters:
    vampire_txt: text of the vampire file
    type_: \'N\', \'0\', \'1\', or \'O\'. Indicates what type of model to expect. \'N\'
    indicates that its a CN model. All the rest indicate its a C0 or C1 model.
 
    Returns:
    Sequence representing the model from the vampire file.
    """
 
    domainreses = domain_regex.findall(vampire_txt)
    domainsize = len(domainreses)
    for d in domainreses:
        if int(d) > domainsize:
            raise ValueError("Weird stuff happening")
    ireses = i_regex.findall(vampire_txt)
    i_functional_values = []
    for ir in ireses:
        irvals = i_regex_vals.findall(ir)
        i_functional_values.append(irvals[0])
    treses = t_regex.findall(vampire_txt)
    if type_=='N':
        nreses = n_regex.findall(vampire_txt)
        n_functional_values = []
        for nr in nreses:
            nrvals = n_regex_vals.findall(nr)
            n_functional_values.append(nrvals[0])
 
        return get_model_sequence([domainsize, i_functional_values, n_functional_values, treses], type_)
    return get_model_sequence([domainsize, i_functional_values, treses], type_)
 
def get_model_sequence(info, type_):
    """
    Turns info into a model sequence. See get_vampire_model.
 
    Parameters:
    See get_vampire_model
 
    Returns:
    See get_vampire_model
    """
    if type_=='N':
        domainsize, i_function, n_function, t_function = info[0], info[1], info[2], info[3]
    else:
        domainsize, i_function, t_function = info[0], info[1], info[2]
    if domainsize > 10:
        raise ValueError("domain too big to make compressed sequence")
    sequence = str(domainsize)
    for i in range(domainsize):
        if "~" in t_function[i][0]:
            sequence += str(0)
        else:
            sequence += str(1)
    for i in range(domainsize):
        for j in range(domainsize):
            sequence += str(int(i_function[i*domainsize + j][2]) - 1)
    if type_=='N':
        for i in range(domainsize):
            sequence += str(int(n_function[i][1]) - 1)
 
    return sequence