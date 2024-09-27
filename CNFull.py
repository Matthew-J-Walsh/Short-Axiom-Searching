from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

import time

def FullCN() -> None:
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    unsolved_folder = "CNRemaining"
    counter_model_folder = "CNCounterModels"

    counter_modeling_formula_sets: list[list[str]] = [["t(i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X))))"]]

    CN = TreeForm(CN_OPERATIONS[1:], CN_OPERATIONS[0], 14)

    Models = ModelTable(CN_SPEC, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: VampireWrapper = create_vampire_countermodel_instance(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [10]:
        print("Starting length: "+str(i))
        start_time = time.time()
        CN.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "CN"+str(i)+"Rem.txt"))
    
        print("Execution time: "+str(time.time() - start_time))
    
    Models.verify_counter_model_sets(counter_modeling_formula_sets)



if __name__ == "__main__":
    FullCN()

