from Globals import *
from ModelTools import *
from TreeForms import *
from TheoremProverUtils import *

import time

BACFETargetLength = 14

def BACF():
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BACF, Models = setup()

    vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True, equational=True)
    for i in [BACFETargetLength]:
        print("Starting length: "+str(i))
        start_time = time.time()
        print(BOOLEAN_ALGEBRA_CF_SPEC)
        progress_tracker = ProgressTracker(150)
        save_file = save_file_base+str(i)+".txt"
        unsolved_count, processed_count = BACF.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "BACF"+str(i)+"Rem.txt"), progress_tracker, save_file)

        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))
    
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

def Hammer():
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BACF, Models = setup()
    vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True, equational=True, optional_args={"-t": "2"})
    for i in [BACFETargetLength]:
        print("Starting length: "+str(i))
        unsolved_file = os.path.join(unsolved_folder, "BACF"+str(i)+"Rem.txt")
        vampire_wrapper.hammer(unsolved_file)

def Prover9_Hammer():
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BACF, Models = setup()
    prover9_wrapper: TheoremProverWrapper = Prover9Wrapper(prover9_executable_file_path, Models.spec, equational=True)
    for i in [BACFETargetLength]:
        print("Starting length: "+str(i))
        unsolved_file = os.path.join(unsolved_folder, "BACF"+str(i)+"Rem.txt")
        new_unsolved_file = os.path.join(unsolved_folder, "BACF"+str(i)+"Rem-fastpass.txt")
        prover9_wrapper.hammer(unsolved_file, new_unsolved_file)

def setup():
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    prover9_executable_file_path = os.path.join("theorem_provers", "prover9")
    unsolved_folder = "BACFRemaining"
    counter_model_folder = "BACFCounterModels"
    save_file_base = os.path.join("partial_run_saves", "BACF")

    counter_modeling_formula_sets: list[list[str]] = [["i(i(P,f),i(Q,R))=i(i(R,P),i(Q,P))",
                                                       "i(i(P,Q),P)=P"]]

    BACF = TreeForm(BOOLEAN_ALGEBRA_CF_SPEC, 14)

    BACF.verify_formulas(8)

    #BACF.dump_formulas(7)

    Models = ModelTable(BOOLEAN_ALGEBRA_CF_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)
    return vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BACF, Models



if __name__ == "__main__":
    BACF()
    #Prover9_Hammer()

