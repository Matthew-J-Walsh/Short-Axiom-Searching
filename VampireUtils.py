from types import NotImplementedType
from Globals import *
from ModelTools import *

import subprocess

#counter_axiom = 'i(i(i(X,Y),i(o,Z)),i(U,i(i(Z,X),i(V,i(W,X)))))' #C1 old
#counter_axiom = 'i(i(i(i(i(X,Y),i(Z,o)),U),V),i(i(V,X),i(Z,X)))' #C0?
#counter_axiom = 'i(i(i(X,Y),i(Z,i(o,U))),i(i(U,X),i(Z,i(V,X))))'
#counter_axiom = 'i(i(X,Y),i(i(Y,Z),i(X,Z)))' #C0 infinite transitivity
#counter_axiom = 'i(i(i(X,o),i(Y,o)),i(Y,X))' #C0 L4
#counter_axiom = 'i(X,i(Y,X))' #C0 L1
#counter_axiom = 'd(n(d(n(d(X,Y)),n(Z))),n(d(n(d(n(V),V)),d(n(Z),X))))=Z' #DN single axiom DN-13345
#counter_axiom = 'i(i(i(X,Y),i(o,Z)),i(U,i(i(Z,X),i(V,i(W,X)))))' #C1 old
#counter_axiom = 'i(i(i(i(i(X,Y),i(Z,o)),U),V),i(i(V,X),i(Z,X)))' #C0?
#counter_axiom = 'i(i(i(X,Y),i(Z,i(o,U))),i(i(U,X),i(Z,i(V,X))))'

_VAMPIRE_MODUS_PONENS = "((t(X) & t(i(X,Y))) => t(Y))"

class VampireWrapper:
    excecutable_location: str
    counter_modeling_formula_sets: list[list[str]]
    counter_model_folder: str
    model_spec: ModelSpec
    command: list[str]
    verification: bool

    def __init__(self, executable_file: str, counter_modeling_formula_sets: list[list[str]], counter_model_folder: str, model_spec: ModelSpec,
                 optional_args: dict[str, str] | None = None, optional_flags: list[str] | None = None, verify_models: bool = False):
        self.excecutable_location = executable_file 
        self.counter_modeling_formula_sets = counter_modeling_formula_sets
        self.counter_model_folder = counter_model_folder
        self.model_spec = model_spec

        baseline_args = {
            "--time_limit": "60",
            "--saturation_algorithm": "fmb",
            "--memory_limit": "4096",
            "--cores": "1",
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

        self.command = [executable_file]
        for flag in flags:
            self.command.append(flag)
        for arg, val in full_args.items():
            self.command.append(arg)
            self.command.append(val)

        assert not "--fmb_start_size" in self.command, "--fmb_start_size is iterated on, don't give as input please"

        self.verification = verify_models
        
    def _vampire_expression_to_fof_line(self, expression: str, name: str, type: Literal["axiom"] | Literal["conjecture"]) -> str:
        vars: set[str] = {c for c in expression if c in VAMPIRE_VARIABLE_SYMBOLS}
        return 'fof('+name+", "+type+", "+("!["+','.join(vars)+"]: " if len(vars)>0 else "")+expression+").\n"

    def _generate_vampire_input_file(self, vampire_form: str, counter_model_set: list[str]) -> str:
        if not os.path.exists("input_tmp"):
            os.makedirs("input_tmp")

        file_name: str = os.path.join("input_tmp", "vampire_run_"+str(datetime.now()))
        for i in ['-', ' ', ':', '.']: 
            file_name = file_name.replace(i, '')
        file_name += '.p'

        contents: str = ""
        contents += self._vampire_expression_to_fof_line(_VAMPIRE_MODUS_PONENS, "mp", "axiom")
        contents += self._vampire_expression_to_fof_line(vampire_form, "cand", "axiom")
        contents += ''.join(self._vampire_expression_to_fof_line(("" if cons.predicate_orientation else "~")+self.model_spec.operators[0].vampire_symbol+"("+cons.vampire_symbol+")", "constant"+str(i), "axiom") 
                            for i, cons in enumerate(self.model_spec.constants) if not cons.predicate_orientation is None)
        contents += self._vampire_expression_to_fof_line('&'.join(counter_model_set), "counter", "conjecture")

        with open(file_name, 'w') as input_file:
            input_file.write(contents)

        return file_name

    def __call__(self, vampire_form: str) -> bool | Model:
        for fmb_start_size in [2, 6, 7, 8]:
            for counter_formula_set in self.counter_modeling_formula_sets:
                input_file_name: str = self._generate_vampire_input_file(vampire_form, counter_formula_set)

                result: str = subprocess.run(self.command + ["--fmb_start_size", str(fmb_start_size)] + [input_file_name], capture_output=True, text=True).stdout

                if not "Finite Model Found!" in result:
                    #print(result)
                    #raise ValueError(result)
                    #os.remove(input_file_name)
                    continue
                else:
                    model: Model = Model(self.model_spec, model_filename = self.save_countermodel(result))
                    if self.verification:
                        assert model(vampire_form), str(model)+"\n"+vampire_form
                        assert not all(model(counter_formula) for counter_formula in counter_formula_set), str(model)+"\n"+str(counter_formula_set)

                    os.remove(input_file_name)
                    return model
                
        return False
    
    def hammer(self, remaining_file_name: str) -> None:
        with open(remaining_file_name, 'r') as remaining_file:
            formulas = remaining_file.readlines()

        print("Starting processing of "+str(len(formulas))+" formulas individually.")
        
        continuing_remainders: list[str] = []
        held_models: list[Model] = []
        for formula in formulas:
            solved = False
            for hm in held_models:
                if hm(formula):
                    solved = True
                    break
            if not solved:
                result = self(formula)
                if result==False:
                    continuing_remainders.append(formula)
                else:
                    assert isinstance(result, Model)
                    held_models.append(result)
        
        with open(remaining_file_name, 'r') as remaining_file:
            remaining_file.writelines(continuing_remainders)

        print(str(len(formulas))+" formulas remaing after processing.")
    
    @staticmethod
    def _revariablize_count_model_set(counter_model_set: list[str]) -> list[str]:
        raise NotImplementedError

    def wipe_counter_models(self) -> None:
        if os.path.exists(self.counter_model_folder) and os.path.isdir(self.counter_model_folder):
            for counter_model_filename in os.listdir(self.counter_model_folder):
                os.remove(os.path.join(self.counter_model_folder, counter_model_filename))

    def save_countermodel(self, result: str) -> str:
        try:
            _, size, _ = VampireOutputTools.order_and_constants(result)
            base_file_name: str = "countermodel-"+str(size)+"-"
            c: int = 0
            while True:
                file_name: str = os.path.join(self.counter_model_folder, base_file_name + str(c))
                if not os.path.exists(file_name):
                    with open(file_name, 'w') as counter_model_file:
                        counter_model_file.write(result)
                        return file_name
                c += 1
        except:
            print(result)
            raise RuntimeError
        
class BlankVampireWrapper(VampireWrapper):
    def __init__(self) -> None:
        pass

    def __call__(self, vampire_form: str) -> bool | Model:
        return False

def create_vampire_countermodel_instance(executable_file: str, counter_modeling_formula_sets: list[list[str]], counter_model_folder: str, model_spec: ModelSpec,
                                         optional_args: dict[str, str] | None = None, optional_flags: list[str] | None = None, verify_models: bool = False) -> VampireWrapper:
    """Creates a vampire wrapper specified by inputs

    Parameters
    ----------
    executable_file : str
        vampire executable filepath
    counter_modeling_formula_sets : list[list[str]]
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
    raise DeprecationWarning
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
        for fmb_start_size in [2, 6, 7, 8]:
            for counter_formula_set in counter_modeling_formula_sets:
                input_file_name: str = _generate_vampire_input_file(vampire_form, counter_formula_set)

                result: str = subprocess.run(command + ["--fmb_start_size", str(fmb_start_size)] + [input_file_name], capture_output=True, text=True).stdout

                if not "Finite Model Found!" in result:
                    #print(result)
                    #raise ValueError()
                    continue
                else:
                    model: Model = Model(model_spec, model_filename=save_countermodel(result, counter_model_folder))
                    if verify_models:
                        assert model(vampire_form), str(model)+"\n"+vampire_form
                        assert not all(model(counter_formula) for counter_formula in counter_formula_set), str(model)+"\n"+str(counter_formula_set)

                    os.remove(input_file_name)
                    return model
                
                
        return False

    return vampire_wrapper

def wipe_counter_models(counter_model_folder: str) -> None:
    raise DeprecationWarning
    if os.path.exists(counter_model_folder) and os.path.isdir(counter_model_folder):
        for counter_model_filename in os.listdir(counter_model_folder):
            os.remove(os.path.join(counter_model_folder, counter_model_filename))
