from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

import time

def FullC1() -> None:
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    unsolved_folder = "C1Remaining"
    counter_model_folder = "C1CounterModels"
    save_file_base = os.path.join("partial_run_saves", "C1")

    #wipe_counter_models(counter_model_folder)

    counter_modeling_formula_sets: list[list[str]] = [["t(i(i(i(X,Y),i(o,Z)),i(U,i(i(Z,X),i(V,i(W,X))))))"]]

    C1 = TreeForm(C1_SPEC, 15)

    Models = ModelTable(C1_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: TheoremProverWrapper = TheoremProverWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [15, 17]:
        print("Starting length: "+str(i))
        start_time = time.time()
        progress_tracker = ProgressTracker(C1.form_count(i))
        save_file = save_file_base+str(i)+".txt"
        unsolved_count, processed_count = C1.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "C1"+str(i)+"Rem.txt"), progress_tracker, save_file)
        
        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))

    Models.verify_counter_model_sets(counter_modeling_formula_sets)


if __name__ == "__main__":
    FullC1()

