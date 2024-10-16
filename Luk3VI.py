from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

import time

def FullLuk3VI() -> None:
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    unsolved_folder = "Luk3VIRemaining"
    counter_model_folder = "Luk3VICounterModels"
    save_file_base = os.path.join("partial_run_saves", "Luk3VI")

    #wipe_counter_models(counter_model_folder)

    counter_modeling_formula_sets: list[list[str]] = [["t(i(X,i(Y,X)))",
                                                       "t(i(i(X,Y),i(i(Y,Z),i(X,Z))))",
                                                       "t(i(i(X,i(Y,Z)),i(Y,i(X,Z))))",
                                                       "t(i(i(i(X,Y),Y),i(i(Y,X),X)))",
                                                       "t(i(i(i(i(i(X,Y),X),X),i(Y,Z)),i(Y,Z)))"]]

    Luk3VI = TreeForm(LUKASIEWICZ_3VI_OPERATIONS[1:], LUKASIEWICZ_3VI_OPERATIONS[0], 5)

    Luk3VI.verify_formulas(7)

    Models = ModelTable(LUKASIEWICZ_3VI_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: VampireWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [19]:
        print("Starting length: "+str(i))
        start_time = time.time()
        progress_tracker = ProgressTracker(Luk3VI.form_count(i))
        save_file = save_file_base+str(i)+".txt"
        unsolved_count, processed_count = Luk3VI.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "Luk3VI"+str(i)+"Rem.txt"), progress_tracker, save_file)

        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))
    
    Models.verify_counter_model_sets(counter_modeling_formula_sets)


if __name__ == "__main__":
    FullLuk3VI()

