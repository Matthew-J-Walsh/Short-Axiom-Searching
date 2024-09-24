from Globals import *
from ModelTools import *
from TreeForms import *
from VampireUtils import *

def FullCN() -> None:
    vampire_executable_file_name = os.path.join("theorem_provers","vampire")
    unsolved_folder = "CNRemaining"
    counter_model_folder = "CNCounterModels"

    counter_model_sets: list[list[str]] = [["t(i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X))))"]]

    CN = TreeForm(CN_OPERATIONS[1:], CN_OPERATIONS[0], 14)

    Models = ModelTable(CN_SPEC, counter_model_folder=counter_model_folder)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper: VampireWrapper = create_vampire_countermodel_instance(vampire_executable_file_name, counter_model_sets, counter_model_folder, Models.spec, verify_models=True)

    for i in [15]:
        CN.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "CN"+str(i)+"Rem.txt"))



if __name__ == "__main__":
    FullCN()

