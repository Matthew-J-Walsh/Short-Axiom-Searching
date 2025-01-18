from Globals import *
from ModelTools import *
from TreeForms import *
from TheoremProverUtils import *

BACETargetLength = 20

parser = argparse.ArgumentParser(description="General process runner. Able to handle standard computation, hammering out remaining with vampire, and filtering out with Prover9.")

parser.add_argument("name", type=str, help="Run name")
parser.add_argument("target_length", type=int, help="Targeted Length")
parser.add_argument("config_file", type=str, help="File for run configuration")
parser.add_argument("--progress_tracker_override", type=int, help="Progress tracker override", default=-1)
parser.add_argument("--run_type", type=str, help="Run type, 'Standard', 'Hammer', or 'Prover9Hammer'", default="Standard")
parser.add_argument("--cache_size", type=int, help="Cache Size. TODO'", default=14)

args = parser.parse_args()

name: str = args.name
target_length: int = args.target_length
with open(args.config_file, "r") as f:
    config = json.load(f)
    config_name: str = config.get("name")
    predicate: PredicateSpec = PredicateSpec.parse(config.get("predicate"))
    operators: tuple[OperationSpec, ...] = tuple([OperationSpec.parse(op) for op in config.get("operators")])
    constants: tuple[ConstantSpec, ...] = tuple([ConstantSpec.parse(cons) for cons in config.get("constants")])

    spec: ModelSpec = ModelSpec(predicate, operators, constants)

    vampire_template: str = "\n".join(config.get("vampire_template"))+"\n"
    prover9_template: str = "\n".join(config.get("prover9_template"))+"\n"

    known_tautologies: list[str] | None = config.get("known_tautologies")

progress_tracker_override: int = args.progress_tracker_override
run_type: str = args.run_type
cache_size: int = args.cache_size
print(name)
print(target_length)
print(spec)
print(vampire_template)
print(run_type)
print(cache_size)

vampire_executable_file_path = os.path.join("theorem_provers", "vampire")
prover9_executable_file_path = os.path.join("theorem_provers", "prover9")
unsolved_folder = name+"Remaining"
counter_model_folder = name+"CounterModels"
save_file_base = os.path.join("partial_run_saves", name)

tree_form = TreeForm(spec, cache_size)

tree_form.verify_formulas(8)

Models = ModelTable(spec, counter_model_folder=counter_model_folder)
if known_tautologies:
    Models.verify_known_formulas(known_tautologies)

if not os.path.exists(unsolved_folder):
    os.makedirs(unsolved_folder)

match run_type:
    case "Standard":
        vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, vampire_template, counter_model_folder, Models.spec, verify_models=True, equational=True)

        print("Starting length: "+str(target_length))
        start_time = time.time()
        progress_tracker = ProgressTracker(progress_tracker_override) if progress_tracker_override!=-1 else ProgressTracker(tree_form.form_count(target_length))
        save_file = save_file_base+str(target_length)+".txt"
        unsolved_count, processed_count = tree_form.process_tree(target_length, Models, vampire_wrapper, os.path.join(unsolved_folder, name+str(target_length)+"Rem.txt"), progress_tracker, save_file)

        print("Processed "+str(processed_count)+" formulas, Was unable to solve: "+str(unsolved_count))
    
        print("Execution time: "+str(time.time() - start_time))
    case "Hammer":
        vampire_wrapper: TheoremProverWrapper = VampireWrapper(vampire_executable_file_path, vampire_template, counter_model_folder, Models.spec, verify_models=True, equational=True, optional_args={"-t": "2"})
        
        print("Starting length: "+str(target_length))
        unsolved_file = os.path.join(unsolved_folder, "BAC"+str(target_length)+"Rem.txt")
        vampire_wrapper.hammer(unsolved_file)
    case "Prover9Hammer":
        prover9_wrapper: TheoremProverWrapper = Prover9Wrapper(prover9_executable_file_path, Models.spec, prover9_template, equational=True)

        print("Starting length: "+str(target_length))
        unsolved_file = os.path.join(unsolved_folder, name+str(target_length)+"Rem.txt")
        new_unsolved_file = os.path.join(unsolved_folder, name+str(target_length)+"Rem-fastpass.txt")
        prover9_wrapper.hammer(unsolved_file, new_unsolved_file)




