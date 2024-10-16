from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

import time

def FullCN() -> None:
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    unsolved_folder = "CNRemaining"
    counter_model_folder = "CNCounterModels"

    #wipe_counter_models(counter_model_folder)

    counter_modeling_formula_sets: list[list[str]] = [["t(i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X))))"]]

    CN = TreeForm(CN_OPERATIONS[1:], CN_OPERATIONS[0], 14)

    CN.verify_formulas(8)

    Models = ModelTable(CN_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: VampireWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [20]:
        print("Starting length: "+str(i))
        start_time = time.time()
        unsolved_count, processed_count = CN.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "CN"+str(i)+"Rem.txt"), 0)

        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))
    
    Models.verify_counter_model_sets(counter_modeling_formula_sets)



if __name__ == "__main__":
    FullCN()

