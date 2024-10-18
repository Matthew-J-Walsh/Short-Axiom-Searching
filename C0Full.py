from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

import time

def FullC0() -> None:
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    unsolved_folder = "C0Remaining"
    counter_model_folder = "C0CounterModels"
    save_file_base = os.path.join("partial_run_saves", "C0")

    #wipe_counter_models(counter_model_folder)

    counter_modeling_formula_sets: list[list[str]] = [["t(i(i(i(i(i(X,Y),i(Z,f)),U),V),i(i(V,X),i(Z,X))))"]]

    C0 = TreeForm(C_OPERATIONS[1:]+C0_CONSTANTS, C_OPERATIONS[0], 15)

    Models = ModelTable(C0_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: VampireWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [11]:
        print("Starting length: "+str(i))
        start_time = time.time()
        progress_tracker = ProgressTracker(C0.form_count(i))
        save_file = save_file_base+str(i)+".txt"
        unsolved_count, processed_count = C0.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "C0"+str(i)+"Rem.txt"), progress_tracker, save_file)
        
        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))

    Models.verify_counter_model_sets(counter_modeling_formula_sets)


if __name__ == "__main__":
    FullC0()

