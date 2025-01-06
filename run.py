from Globals import *
from ModelTools import *
from TreeForms import *
from TheoremProverUtils import *

BACETargetLength = 20

def main(name: str, target_length: int, spec: ModelSpec, counter_modeling_formula_sets: list[list[str]], 
         progress_tracker_override: int = -1, run_type: str = "Standard"):
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BAC, Models = setup(name, spec, counter_modeling_formula_sets)

    vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True, equational=True)
    for i in [BACETargetLength]:
        print("Starting length: "+str(i))
        start_time = time.time()
        print(BOOLEAN_ALGEBRA_CF_SPEC)
        progress_tracker = ProgressTracker(1509)
        save_file = save_file_base+str(i)+".txt"
        unsolved_count, processed_count = BAC.process_tree(i, Models, vampire_wrapper, os.path.join(unsolved_folder, "BAC"+str(i)+"Rem.txt"), progress_tracker, save_file)

        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))
    
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

def Hammer(name: str, spec: ModelSpec, counter_modeling_formula_sets: list[list[str]]):
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BAC, Models = setup(name, spec, counter_modeling_formula_sets)
    vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, counter_modeling_formula_sets, counter_model_folder, Models.spec, verify_models=True, equational=True, optional_args={"-t": "2"})
    for i in [BACETargetLength]:
        print("Starting length: "+str(i))
        unsolved_file = os.path.join(unsolved_folder, "BAC"+str(i)+"Rem.txt")
        vampire_wrapper.hammer(unsolved_file)

def Prover9_Hammer(name: str, spec: ModelSpec, counter_modeling_formula_sets: list[list[str]]):
    vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, BAC, Models = setup(name, spec, counter_modeling_formula_sets)
    prover9_wrapper: TheoremProverWrapper = Prover9Wrapper(prover9_executable_file_path, Models.spec, equational=True)
    for i in [BACETargetLength]:
        print("Starting length: "+str(i))
        unsolved_file = os.path.join(unsolved_folder, name+str(i)+"Rem.txt")
        new_unsolved_file = os.path.join(unsolved_folder, name+str(i)+"Rem-fastpass.txt")
        prover9_wrapper.hammer(unsolved_file, new_unsolved_file)

def setup(name: str, spec: ModelSpec, counter_modeling_formula_sets: list[list[str]], cache_size: int = 14):
    vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
    prover9_executable_file_path = os.path.join("theorem_provers", "prover9")
    unsolved_folder = name+"Remaining"
    counter_model_folder = name+"CounterModels"
    save_file_base = os.path.join("partial_run_saves", name)

    treeForm = TreeForm(spec, cache_size)

    treeForm.verify_formulas(8)

    Models = ModelTable(spec, counter_model_folder=counter_model_folder)
    Models.verify_counter_model_sets(counter_modeling_formula_sets)

    if not os.path.exists(unsolved_folder):
        os.makedirs(unsolved_folder)
    return vampire_executable_file_path, prover9_executable_file_path, unsolved_folder, counter_model_folder, save_file_base, counter_modeling_formula_sets, treeForm, Models



if __name__ == "__main__":
    BAC()
    #Prover9_Hammer()

