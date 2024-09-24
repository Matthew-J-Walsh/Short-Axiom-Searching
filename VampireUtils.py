from Globals import *
from ModelTools import *

import subprocess

_VAMPIRE_MODUS_PONENS = "((t(X) & t(i(X,Y))) => t(Y))"

class VampireWrapper(Protocol):
    def __call__(self, vampire_form: str) -> bool | Model:
        raise NotImplementedError
    
def BLANK_VAMPIRE_WRAPPER(vampire_form: str) -> bool | Model:
    return False

def vampire_expression_to_fof_line(expression: str, name: str, type: Literal["axiom"] | Literal["conjecture"]) -> str:
    vars: set[str] = {c for c in expression if c in VAMPIRE_VARIABLE_SYMBOLS}
    return 'fof('+name+", "+type+", "+"!["+','.join(vars)+"]: "+expression+").\n"

def _generate_vampire_input_file(vampire_form: str, counter_model_set: list[str]) -> str:
    file_name: str = "vampire_run_"+str(datetime.now())
    for i in ['-', ' ', ':', '.']: 
        file_name = file_name.replace(i, '')
    file_name += '.p'

    contents: str = ""
    contents += vampire_expression_to_fof_line(_VAMPIRE_MODUS_PONENS, "mp", "axiom")
    contents += vampire_expression_to_fof_line(vampire_form, "cand", "axiom")
    contents += ''.join(vampire_expression_to_fof_line(counter, "counter"+str(i), "axiom") for i, counter in enumerate(counter_model_set))

    with open(file_name, 'w') as input_file:
        input_file.write(contents)

    return file_name

def save_countermodel(result: str, folder: str) -> str:
    try:
        _, size, _ = VampireOutputTools.order_and_constants(result)
        base_file_name: str = "countermodel-"+str(size)+"-"
        c: int = 0
        while True:
            file_name: str = os.path.join(folder, base_file_name + str(c))
            if not os.path.exists(file_name):
                with open(file_name, 'w') as counter_model_file:
                    counter_model_file.write(result)
                    return file_name
            c += 1
    except:
        print(result)
        raise RuntimeError

def create_vampire_countermodel_instance(executable_file: str, counter_model_sets: list[list[str]], counter_model_folder: str, model_spec: ModelSpec,
                                         optional_args: dict[str, str] | None = None, optional_flags: list[str] | None = None, verify_models: bool = False) -> VampireWrapper:
    """Creates a vampire wrapper specified by inputs

    Parameters
    ----------
    executable_file : str
        vampire executable filepath
    counter_model_sets : list[list[str]]
        Lists of countermodeling lists (usually known tautologies forming axiom systems)
    optional_args : dict[str, str] | None, optional
        Added arguments for vampire executable, by default None
    optional_flags : list[str] | None, optional
        Added flags for vampire executable, by default None
    verify_models : bool, optional
        Should models be verified before returning (used for debugging), by default False

    Returns
    -------
    VampireWrapper
        Function that calculates counter-models
    """    
    baseline_args = {
        "-t": "240",
        "-sa": "fmb",
    }
    if optional_args:
        full_args: dict[str, str] = {**baseline_args, **optional_args}
    else:
        full_args = baseline_args

    baseline_flags: list[str] = []
    if optional_flags:
        flags: list[str] = list(set(baseline_flags + optional_flags))
    else:
        flags = baseline_flags

    command: list[str] = [executable_file]
    for flag in flags:
        command.append(flag)
    for arg, val in full_args.items():
        command.append(arg)
        command.append(val)

    assert not "--fmb_start_size" in command, "--fmb_start_size is iterated on, don't give as input please"

    if not os.path.exists(counter_model_folder):
        os.makedirs(counter_model_folder)
    
    def vampire_wrapper(vampire_form: str) -> bool | Model:
        for fmb_start_size in range(2, 8):
            for counter_model_set in counter_model_sets:
                input_file_name: str = _generate_vampire_input_file(vampire_form, counter_model_set)

                result: str = subprocess.run(command + ["--fmb_start_size", str(fmb_start_size)] + [input_file_name], capture_output=True, text=True).stdout

                if not "Finite Model Found!" in result:
                    #print(result)
                    #raise ValueError()
                    continue
                else:
                    model: Model = Model(model_spec, model_filename=save_countermodel(result, counter_model_folder))
                    if verify_models:
                        assert model(vampire_form), str(model)+"\n"+vampire_form

                    os.remove(input_file_name)
                    return model
                
                
        return False

    return vampire_wrapper


