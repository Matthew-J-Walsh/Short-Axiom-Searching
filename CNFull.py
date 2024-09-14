from Globals import *
from ModelTools import *
from Forms import *

import os

from VampireUtils import create_vampire_countermodel_instance

def FullCN() -> None:
    vampire_executable_file_name = "./vampire"
    unsolved_folder = "CNRemaining"
    counter_model_folder = "CNCounterModels"

    counter_model_sets: list[list[str]] = [["t(i(i(i(i(i(X,Y),i(n(Z),n(U))),Z),V),i(i(V,X),i(U,X))))"]]

    Models = ModelTable(CN_SPEC, counter_model_folder=counter_model_folder)
    Length19 = FormTable(19, CLASSICAL_TRUTH, [CLASSICAL_IMPLICATION, CLASSICAL_NEGATION])
    #Length20 = FormTable(20, CLASSICAL_TRUTH, [CLASSICAL_CONJUNCTION, CLASSICAL_NEGATION])
    #Length21 = FormTable(21, CLASSICAL_TRUTH, [CLASSICAL_CONJUNCTION, CLASSICAL_NEGATION])

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)

    vampire_wrapper = create_vampire_countermodel_instance(vampire_executable_file_name, counter_model_sets, counter_model_folder, Models.spec, verify_models=True)

    Length19.process(Models, vampire_wrapper, os.path.join(unsolved_folder, "CN19Rem.txt"))
    #Length20.process(Models, vampire_wrapper, "CN20Rem")
    #Length21.process(Models, vampire_wrapper, "CN21Rem")









if __name__ == "__main__":
    FullCN()

