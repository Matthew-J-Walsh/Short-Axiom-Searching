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

class TheoremProverWrapper:
    """Class for a theorem proving wrapper. Called on expressions to run them in a theorem prover."""
    excecutable_location: str
    """Location of executable file"""
    model_spec: ModelSpec
    """Model spec in use"""
    equational: bool
    """Is the model equational"""

    def __init__(self, executable_location: str, model_spec: ModelSpec, equational: bool = False):
        self.excecutable_location = executable_location
        self.model_spec = model_spec
        self.equational = equational
        
    def _fof_formula_to_fof_line(self, formula: str, name: str, type: Literal["axiom"] | Literal["conjecture"]) -> str:
        """Takes a formula and makes the appropriate single line

        Parameters
        ----------
        formula : str
            Formula to wrap
        name : str
            Name to have the formula be in the file ex: "candA", "mp".
        type : Literal[&quot;axiom&quot;] | Literal[&quot;conjecture&quot;]
            What tptp type it is, ex: "axiom", "conjecture"

        Returns
        -------
        str
            _description_
        """        
        vars: set[str] = {c for c in formula if c in VAMPIRE_VARIABLE_SYMBOLS}
        return 'fof('+name+", "+type+", "+("!["+','.join(vars)+"]: " if len(vars)>0 else "")+formula+").\n"

    def __call__(self, tptp_form: str) -> bool | Model:
        """Runs the theorem prover on the given formula

        Parameters
        ----------
        tptp_form : str
            TPTP form of the formula

        Returns
        -------
        bool | Model
            False if unmodeled
            True if unmodeled but should be removed (raises errors if used outside hammering)
            Model if countermodeled
        """
        raise NotImplementedError
    
    def encapsulate_candidate(self, tptp_form: str, i: int) -> str:
        """Puts a formula into a format associated for more easy access later

        Parameters
        ----------
        tptp_form : str
            _description_
        i : int
            _description_

        Returns
        -------
        str
            _description_
        """        
        return self._fof_formula_to_fof_line(tptp_form, "cand"+str(i), "axiom")
    
    @staticmethod
    def _strip_fof(fof_form: str) -> str:
        if ":" in fof_form:
            return fof_form.split(":")[1][1:-3]
        else: 
            return fof_form
    
    def hammer(self, remaining_file_name: str, target_file_name: str | None = None) -> None:
        """"Hammer out" remaining formulas with this solver.

        Parameters
        ----------
        remaining_file_name : str
            Location of the remaining formulas
        """        
        if target_file_name is None:
            target_file_name = remaining_file_name
        with open(remaining_file_name, 'r') as remaining_file:
            formulas: list[str] = [self._strip_fof(s) for s in remaining_file.readlines()]

        print("Starting processing of "+str(len(formulas))+" formulas individually.")
        
        continuing_remainders: list[str] = []
        held_models: list[Model] = []
        for formula in formulas:
            solved = False
            for hm in held_models:
                if hm(formula.strip()):
                    solved = True
                    break
            if not solved:
                print("Working on: "+formula.strip())
                result = self(formula.strip())
                if result==False:
                    continuing_remainders.append(formula.strip())
                elif isinstance(result, Model):
                    held_models.append(result)
        
        with open(target_file_name, 'w') as remaining_file:
            remaining_file.write("\n".join(continuing_remainders))

        print(str(len(formulas))+" formulas remaing after processing.")
        
class VampireWrapper(TheoremProverWrapper):
    """Wrapper for vampire theorem prover"""
    counter_modeling_formula_sets: list[list[str]]
    """Counter modeling formulas to use"""
    counter_model_folder: str
    """Location to put new counter models"""
    command: list[str]
    """Partially built command"""
    verification: bool
    """Should we verify countermodels before returning"""

    def __init__(self, executable_location: str, counter_modeling_formula_sets: list[list[str]], counter_model_folder: str, model_spec: ModelSpec, equational: bool = False,
                 optional_args: dict[str, str] | None = None, optional_flags: list[str] | None = None, verify_models: bool = False):
        super().__init__(executable_location, model_spec, equational)
        self.counter_modeling_formula_sets = counter_modeling_formula_sets
        self.counter_model_folder = counter_model_folder

        baseline_args = {
            "--time_limit": "10",
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

        self.command = [executable_location]
        for flag in flags:
            self.command.append(flag)
        for arg, val in full_args.items():
            self.command.append(arg)
            self.command.append(val)

        assert not "--fmb_start_size" in self.command, "--fmb_start_size is iterated on, don't give as input please"

        self.verification = verify_models

    def _generate_tptp_input_file(self, tptp_form: str, counter_model_set: list[str]) -> str:
        if not os.path.exists("input_tmp"):
            os.makedirs("input_tmp")

        file_name: str = os.path.join("input_tmp", "tptp_run_"+str(datetime.now()))
        for i in ['-', ' ', ':', '.']: 
            file_name = file_name.replace(i, '')
        file_name += '.p'

        contents: str = ""
        if not self.equational:
            contents += self._fof_formula_to_fof_line(_VAMPIRE_MODUS_PONENS, "mp", "axiom")
        contents += self._fof_formula_to_fof_line(tptp_form, "cand", "axiom")
        if not self.equational:
            contents += ''.join(self._fof_formula_to_fof_line(("" if cons.predicate_orientation else "~")+self.model_spec.prefix.tptp_symbol+"("+cons.tptp_symbol+")", "constant"+str(i), "axiom") 
                                for i, cons in enumerate(self.model_spec.constants) if not cons.predicate_orientation is None)
        contents += self._fof_formula_to_fof_line('&'.join(counter_model_set), "counter", "conjecture")

        with open(file_name, 'w') as input_file:
            input_file.write(contents)

        return file_name

    def wipe_counter_models(self) -> None:
        """Deletes the counter models in the folder"""
        if os.path.exists(self.counter_model_folder) and os.path.isdir(self.counter_model_folder):
            for counter_model_filename in os.listdir(self.counter_model_folder):
                os.remove(os.path.join(self.counter_model_folder, counter_model_filename))

    def save_countermodel(self, result: str) -> str:
        """Saves a vampire countermodel to a file

        Parameters
        ----------
        result : str
            Vampire output

        Returns
        -------
        str
            Filename it was saved in
        """
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

    def __call__(self, tptp_form: str) -> bool | Model:
        """Runs the vampire theorem prover on the given formula

        Parameters
        ----------
        tptp_form : str
            TPTP form of the formula

        Returns
        -------
        bool | Model
            False if unmodeled
            True if unmodeled but should be removed (raises errors if used outside hammering)
            Model if countermodeled
        """
        for counter_formula_set in self.counter_modeling_formula_sets:
            input_file_name: str = self._generate_tptp_input_file(tptp_form, counter_formula_set)
            for fmb_start_size in [2]:#, 6, 7, 8]:
                result: str = subprocess.run(self.command + ["--fmb_start_size", str(fmb_start_size)] + [input_file_name], capture_output=True, text=True).stdout

                if not "Finite Model Found!" in result:
                    #print(result)
                    #raise ValueError(result)
                    #os.remove(input_file_name)
                    continue
                else:
                    model: Model = Model(self.model_spec, model_filename = self.save_countermodel(result))
                    if self.verification:
                        assert model(tptp_form), str(model)+"\n"+tptp_form
                        assert not all(model(counter_formula) for counter_formula in counter_formula_set), str(model)+"\n"+str(counter_formula_set)

                    os.remove(input_file_name)
                    return model
                
        return False
        
class Prover9Wrapper(TheoremProverWrapper):
    preamble: str

    def __init__(self, executable_location: str, model_spec: ModelSpec, equational: bool = False):
        self.preamble = "set(auto).\nset(prolog_style_variables).\nassign(max_weight,48).\nassign(max_vars,8).\nset(restrict_denials).\nassign(max_given,300).\n\nformulas(usable).\ni(i(c1,c2),c1) != c1.\nend_of_list."
        super().__init__(executable_location, model_spec, equational)

    def _generate_prover9_input_file(self, tptp_form: str) -> str:
        if not os.path.exists("input_tmp"):
            os.makedirs("input_tmp")

        file_name: str = os.path.join("input_tmp", "prover9_run_"+str(datetime.now()))
        for i in ['-', ' ', ':', '.']: 
            file_name = file_name.replace(i, '')
        file_name += '.in'

        contents: str = self.preamble+"\n\n"
        contents += "formulas(sos).\n"
        contents += tptp_form+".\n"
        contents += "end_of_list."

        with open(file_name, 'w') as input_file:
            input_file.write(contents)

        return file_name

    def __call__(self, tptp_form: str) -> bool | Model:
        """Runs the prover9 theorem prover on the given formula

        Parameters
        ----------
        tptp_form : str
            TPTP form of the formula

        Returns
        -------
        bool | Model
            False if unmodeled
            True if unmodeled but should be removed (raises errors if used outside hammering)
            Model if countermodeled
        """
        input_file_name: str = self._generate_prover9_input_file(tptp_form)
        #print([self.excecutable_location, " < "] + [input_file_name])
        with open(input_file_name, "r") as inpt:
            result = subprocess.run([self.excecutable_location], stdin=inpt, capture_output=True, text=True).stdout
        #print(result)
        return "Exiting with failure." in result


class BlankTheoremProverWrapper(TheoremProverWrapper):
    """TheoremProverWrapper that returns False always
    """    
    def __init__(self, *args) -> None:
        pass

    def __call__(self, tptp_form: str) -> bool | Model:
        return False

