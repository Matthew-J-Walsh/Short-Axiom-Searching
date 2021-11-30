import datetime
from PolishFormUtilities import *
from TemplateGeneration import *
from MathUtilities import *
from ConjunctiveNormalForm import *
from FillHandling import *

def GenerateTautologyFile(length, output_file_name_base, generator, valid_input_generator = None, 
                    file_template_limit = None, constant_value = None):
    """
    Run the Conjunctive Normal Form Method on the specified length formulas to generate tautologies and
    save them with the specified filename.
    Skipping stuff is to be implemented.
    Will place every valid input into the generator into a different output file so that in the event
    of a crash not too much is lost.
    
    Parameters:
    length: length of formula being aimed at
    output_file_name_base: base file name to be used for output files
    generator: generator to use, can either be function or in ["CN", "C0", "C1"]
    valid_input_generator: function to get valid inputs to generator, not needed if builtins are used
    file_template_limit: limit to number of templates in a file, useful for making many small files
    constant_value: value of the "O" constant. To be implemented: multiple constants
    """
    if generator == "CN":
        generator = yield_all_theorem_templates_full_CN
        valid_input_generator = valid_CN_inputs_given_length
    elif generator in ["C0", "C1"]:
        generator = yield_all_theorem_templates_full_CO
        valid_input_generator = valid_CO_inputs_given_length
    if valid_input_generator is None:
        raise ValueError("Use builtin generator and input generator or give one.")
    
    for runs in valid_input_generator(length):
        print("Current run is: "+str(runs))
        variables = runs[0]
        fills = generate_fills(variables)
        cleaves = generate_cleaves(fills)
        spec_lists = generate_specification_lists(fills)
        
        count = 0
        file_count = 0 #how much stuff in the file
        file_number = 0 #what number file
        all_results = []
        for template in generator(runs[0], runs[1], runs[2]):
            if count%1000==0:
                print(datetime.datetime.now())
                print(str(count)+" Done")
            count += 1
            file_count += 1
            
            numbered_template = generate_numbered_template(template)
            bindings = cnf_to_bindings(template_to_cnf(numbered_template), Ovalue=constant_value)
            if bindings is not None:
                results = fills[remove_subsumptions( 
                                    bindings_to_tautologies(
                                        bindings,
                                        cleaves),
                                    spec_lists)]
                
                for j in range(results.shape[0]):
                    all_results.append(fillin_template_with_fill(numbered_template, results[j]))
                
            if file_template_limit is not None and file_count >= file_template_limit:
                file_count = 0
                
                with open(output_file_name_base+"_"+str(variables)+"vars_"+str(file_number)+".txt", "w") as ofile:
                    for res in all_results:
                        ofile.write(res+"\n")
                
                file_number += 1
                all_results = []
        if len(all_results) > 0:
            file_identifier = ""
            if file_number > 0:
                file_identifier = str(file_number)
            
            with open(output_file_name_base+"_"+str(variables)+"vars_"+file_identifier+".txt", "w") as ofile:
                for res in all_results:
                    ofile.write(res+"\n")
    
    
    
    
    
    
    
    
    
    
    
    