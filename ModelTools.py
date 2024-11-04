from __future__ import annotations

from Globals import *
from FillTools import *

def _get_generated_prefix_value(prefix: PrefixSpec, model: Model) -> np.ndarray:
    """Generates specific prefixes', with fixed values, default tables that are model dependent.
    Currently supported: Equality "="

    Parameters
    ----------
    prefix : PrefixSpec
        Prefix operator to make the value of
    model : Model
        Model that the value is being made from

    Returns
    -------
    np.ndarray
        Prefix constant value

    Raises
    ------
    NotImplementedError
        No implementation for that prefix
    """    
    if prefix.symbol == "=":
        arr: ModelArray = np.eye(model.order, dtype=np.bool_)
        arr.setflags(write=False)
        return arr
    raise NotImplementedError

class CompiledElement(NamedTuple):
    table: np.ndarray
    inputs: np.ndarray

def apply_fill_to_cache(cache: ModelArray, fill: DimensionalReference) -> tuple[ModelArray, DimensionalReference]:
    """This deserves a lot of documentation. 
    This applies a fill to a cache, the cache is asserted to be large enough to handle the fill. 
    This determination is made by checking if the values in the fill, up until there are cache.ndim - 1 values left, are all the same.

    For example a cache will be caching something like the fill: "[00]0123". The input fill is "[55]5233". 
    The brackets indicate that those two elements of the fill are "assumed" in the cache, meaning this function assumes that the cache was created for the fill size given,
    and only checks that the cache is actually big enough for the fill (we would error without the assert).

    Then we translate the fill into a standard form (more like the assumed cache fill form), and form a reverse translation to use at the end.
    Then we take diagonals out of the cache for every set of equal values in the fill onto the left most dimension in the equal values set, this gives us the proper array.
    Finally we use the reverse translation to calculate the output format for use by nodes higher up the chain.

    Parameters
    ----------
    cache : ModelArray
        Cached array
    fill : DimensionalReference
        Fill to calculate for

    Returns
    -------
    tuple[ModelArray, DimensionalReference]
        _description_
    """    
    #what if the cache is a different size?
    #cache.ndims = 4, [00]0123
    #fill: [55]5233
    #assert np.unique(fill[:len(fill) - cache.ndim]).size <= 1, fill[:len(fill) - cache.ndim]
    fill_renaming: dict[int, int] = {}
    inverse_fill_renaming: dict[int, int] = {}
    fixed_fill: list[int] = []
    #fixed_fill: [00]0122
    c = 0
    for e in fill[len(fill) - cache.ndim:]:
        if e not in fill_renaming.keys():
            fill_renaming[e] = c
            inverse_fill_renaming[c] = e
            c += 1
        fixed_fill.append(fill_renaming[e])

    reduced: ModelArray = cache

    #print(fixed_fill)

    for i in range(c):
        if fixed_fill.count(i) > 1:
            #print("============")
            #print(fixed_fill)
            dims: list[int] = sorted([j for j, e in enumerate(fixed_fill) if e==i])
            #print(dims)
            slices: list[slice | np.ndarray] = [slice(None)] * reduced.ndim
            for dim in dims:
                slices[dim] = np.arange(reduced.shape[dim])
            reduced = reduced[*slices]
            #print(output.shape)

            new_fixed_fill: list[int] = []
            found = False
            for e in fixed_fill:
                if e == i:
                    if not found:
                        new_fixed_fill.append(e)
                        found = True
                else:
                    new_fixed_fill.append(e)
            fixed_fill = new_fixed_fill
    
    #012
    assert reduced.ndim == len(fixed_fill)
    return reduced, tuple([inverse_fill_renaming[e] for e in fixed_fill])

def _apply_fill_to_cache_forced_iteration(arr: np.ndarray, fill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same as above, much slower, definantly works.
    """    
    assert np.unique(fill[:fill.size - arr.ndim]).size == 1, fill[:fill.size - arr.ndim]
    fill_renaming: dict[int, int] = {}
    inverse_fill_renaming: dict[int, int] = {}
    fixed_fill: list[int] = []
    c = 0
    for e in fill[fill.size - arr.ndim:]:
        if e not in fill_renaming.keys():
            fill_renaming[e] = c
            inverse_fill_renaming[c] = e
            c += 1
        fixed_fill.append(fill_renaming[e])

    print(fixed_fill)

    new_arr = np.empty((arr.shape[0],) * c, dtype = np.int8)
    for idx in np.ndindex(new_arr.shape):
        shadow = [idx[e] for e in fixed_fill]
        new_arr[idx] = arr[*shadow]

    set_fill: list[int] = []
    seen = set()
    for e in fixed_fill:
        if not e in seen:
            seen.add(e)
            set_fill.append(e)

    return new_arr, np.array([inverse_fill_renaming[e] for e in set_fill])

@functools.cache
def array_dimensional_resizer(target_size: int, current: tuple[int, ...]) -> tuple[slice | None, ...]:
    """Makes dimension adder to make current shape equal to target shape

    Parameters
    ----------
    target_size : int
        Target size shape
    current : tuple[int, ...]
        Current indexing

    Returns
    -------
    tuple[slice | None, ...]
        Reshaping tuple
    """    
    return tuple([slice(None) if i in current else np.newaxis for i in range(target_size)])

class VampireOutputTools:
    """Tools to process a tptp finite models
    """    
    _variable_regex: re.Pattern = re.compile(r"tff\('declare_\$i(\d+)',type,(\w|'\bfmb_\$i_\d+\b'):\$i\)")
    """Regex to pull out variables"""
    _function_def_regex: re.Pattern = re.compile(r"tff\(declare_(\w),type,\1: \((\$i(?: \* \$i)*)\) > \$i\)")
    """Regex to get the arity of functions"""
    _function_val_regex: re.Pattern = re.compile(r"(\w)\(([^)\s]+)\)\s*=\s*'fmb_\$i_(\d+)'")
    """Regex to get the value mapping of functions"""
    _function_input_regex: re.Pattern = re.compile(r"fmb_\$i_(\d+)")
    """Regex to parse the inputs out of function mappings"""
    _predicate_val_regex: re.Pattern = re.compile(r"(~?)(\w)\('fmb_\$i_(\d+)'\)")
    """Regex to parse the truthiness of predicates"""

    @staticmethod
    def order_and_constants(result: str) -> tuple[str, int, dict[str, int]]:
        """Determines the order and constant values, fixes the results to not have constants

        Parameters
        ----------
        result : str-
            Vampire output text

        Returns
        -------
        tuple[str, int, dict[str, int]]
            Fixed output text,
            Order of the countermodel,
            Constants dict
        """        
        result = copy.copy(result)
        matches = re.findall(VampireOutputTools._variable_regex, result)

        order: int = max([int(m[0]) for m in matches])
        constants: dict[str, int] = {c: i for i, (_, c) in enumerate(matches) if len(c)==1}

        for symbol, value in constants.items():
            result = re.sub(r'\b'+re.escape(symbol)+r'\b', "'fmb_$i_"+str(value + 1)+"'", result)

        return result, order, constants

    @staticmethod
    def functions_arity_verification(result: str, arities: dict[str, int]) -> None:
        """Verifies the arity of the functions in a tptp output

        Parameters
        ----------
        result : str
            Vampire output text
        arities : dict[str, int]
            Arity of functions
        """        
        matches = re.findall(VampireOutputTools._function_def_regex, result)
        for m in matches:
            assert arities[m[0]] == m[1].count(r"$i")
    
    @staticmethod
    def function_parse(result: str, function_identifier: str, order: int, arity: int, predicate: bool) -> np.ndarray:
        """Parses a function out of a tptp output

        Parameters
        ----------
        result : str
            Vampire output text
        function_identifier : str
            Function identifer for targeted function
        order : int
            Order of the model
        arity : int
            Arity of the function
        predicate : bool
            Is this function a predicate (returns a boolean) or a true function (returns a value)

        Returns
        -------
        np.ndarray
            _description_
        """        
        matches = re.findall(VampireOutputTools._predicate_val_regex if predicate else VampireOutputTools._function_val_regex, result)
        arr = np.full([order]*arity, -1, np.int8)
        func_id_idx = 1 if predicate else 0
        for m in matches:
            if m[func_id_idx]==function_identifier:
                if predicate:
                    inputs = tuple([int(m[2]) - 1 for g in m[2].split(',')])
                    assert len(inputs) == arity, "Incorrect arity assignment found."
                    if len(m[0])==0:
                        arr[inputs] = 1
                    else: #elif len(m[0])==1:
                        arr[inputs] = 0
                else:
                    inputs = tuple([int(re.search(r"fmb_\$i_(\d+)", g).group(1)) - 1 for g in m[1].split(',')]) # type: ignore
                    assert len(inputs) == arity, "Incorrect arity assignment found."
                    arr[inputs] = np.int8(m[-1]) - 1
        assert (arr==-1).sum() == 0, "Not all outputs were bound.\n"+str(arr)+"\nIdent: "+str(function_identifier)+"\nMatches: "+str([m for m in matches])+\
                "\nResult: "+result
        return arr.astype(np.bool_) if predicate else arr

class Model():
    """A model to check logical expressions
    """    
    order: int
    """Size of model"""
    prefix: PrefixSpec
    """Spec of the prefix operation in the model"""
    prefix_definition: np.ndarray
    """Definition of the prefix operation in the model"""
    operation_definitions: dict[OperationSpec, np.ndarray]
    """Definition of each operation in the model"""
    constant_definitions: dict[ConstantSpec, int]
    """Definition of each of the constants in the model"""

    def __init__(self, spec: ModelSpec, operation_definitions: dict[OperationSpec | PrefixSpec, np.ndarray] | None = None, 
                 constant_defnitions: dict[ConstantSpec, int] | None = None, model_filename: str | None = None) -> None:
        self.prefix = spec.prefix
        if not operation_definitions is None:
            self.operation_definitions = {op: arr for op, arr in operation_definitions.items() if isinstance(op, OperationSpec)}
            self.prefix_definition = [arr for op, arr in operation_definitions.items() if isinstance(op, PrefixSpec)][0]
            if not constant_defnitions is None:
                self.constant_definitions = constant_defnitions
            else:
                self.constant_definitions = {}
            self.order = self.operation_definitions[spec.operators[0]].shape[0]
        elif not model_filename is None:
            with open(model_filename, 'r') as f:
                result, order, constants = VampireOutputTools.order_and_constants(f.read())
                self.constant_definitions = {c: constants[c.tptp_symbol] for c in spec.constants}
                self.operation_definitions = {}
                VampireOutputTools.functions_arity_verification(result, {op.tptp_symbol: op.arity for op in spec.operators})
                for op in spec.operators:
                    self.operation_definitions[op] = VampireOutputTools.function_parse(result, op.tptp_symbol, order, op.arity, op.default_table.dtype == np.bool_)
                self.order = self.operation_definitions[spec.operators[0]].shape[0]
                if spec.prefix.default_table is None:
                    self.prefix_definition = _get_generated_prefix_value(spec.prefix, self)
                else:
                    self.prefix_definition = VampireOutputTools.function_parse(result, spec.prefix.tptp_symbol, order, spec.prefix.arity, spec.prefix.default_table.dtype == np.bool_)
        else:
            self.operation_definitions = {op: op.default_table for op in spec.operators}
            self.constant_definitions = {c: c.default_value for c in spec.constants}
            self.order = self.operation_definitions[spec.operators[0]].shape[0]
            if spec.prefix.default_table is None:
                self.prefix_definition = _get_generated_prefix_value(spec.prefix, self)
            else:
                self.prefix_definition = spec.prefix.default_table

    def calculate(self, op: OperationSpec | PrefixSpec, output_size: int, inputs: tuple[tuple[ModelArray, DimensionalReference], ...]) -> ModelArray:
        """Calculates the result table of an operation

        Parameters
        ----------
        op : OperationSpec | PrefixSpec
            Operation to apply
        output_size : int
            Expected size of output
        inputs : tuple[tuple[ModelArray, DimensionalReference], ...]
            Inputs

        Returns
        -------
        ModelArray
            Result
        """        
        assert len(inputs)==op.arity
        reorganized_models: list[ModelArray] = [arr[*array_dimensional_resizer(output_size, sub_dims)] for arr, sub_dims in inputs]
        out = self._apply_function(self.operation_definitions[op] if isinstance(op, OperationSpec) else self.prefix_definition, *reorganized_models)
        assert isinstance(out, np.ndarray)
        assert out.ndim == output_size
        return out

    def _get_values(self, tptp_form: str) -> list[PrefixSpec | OperationSpec | ConstantSpec | int]:
        """Gets the operations constants and variables making up a expression

        Parameters
        ----------
        tptp_form : str
            tptp string form of expression

        Returns
        -------
        list[PrefixSpec | OperationSpec | ConstantSpec | int]
            Ordered values of the expression
        """        
        stripped: str = tptp_form.replace('(', '').replace(')', '').replace(',', '')

        if '=' in stripped: #TODO: PLEASE FIX ME THIS IS SO BAD
            stripped = "=" + stripped.replace("=", '')

        out: list[PrefixSpec | OperationSpec | ConstantSpec | int] = []
        var_table: dict[str, int] = {}
        i = 0
        for c in stripped:
            found: bool = False
            if self.prefix.tptp_symbol==c:
                out.append(self.prefix)
                found = True
            for op in self.operation_definitions.keys():
                if op.tptp_symbol==c:
                    out.append(op)
                    found = True
                    break
            if found:
                continue
            for cons in self.constant_definitions.keys():
                if cons.tptp_symbol==c:
                    out.append(cons)
                    found = True
                    break
            if found:
                continue
            if not c in var_table.keys():
                var_table[c] = i
                i += 1
            out.append(var_table[c])

        return out

    def compile_expression(self, tptp_form: str) -> list[CompiledElement | int | None]:
        """Compiles the compression into a more easily managed form

        Parameters
        ----------
        tptp_form : str
            Vampire form of expression

        Returns
        -------
        list[CompiledElement | int | None]
            A compiled version for easy calculation
        """        
        values: list[PrefixSpec | OperationSpec | ConstantSpec | int] = self._get_values(tptp_form)
        try:
            assert isinstance(values[0], PrefixSpec), "Un-prefixed Expression" 
            assert values.count(values[0])==1, "Prefix must be unique"
        except:
            print(tptp_form)
            raise AssertionError
        function_stack: list[FunctionStackElement] = [FunctionStackElement(values[0], [values[0].arity], [])] #prefix arity
        var_count: int = max(v for v in values if isinstance(v, int)) + 1
        cons_list: list[ConstantSpec] = list(self.constant_definitions.keys())
        constant_indicies: dict[ConstantSpec, int] = {c: i+var_count for i, c in enumerate(cons_list)}
        compiled: list[CompiledElement | int | None] = [None] * var_count + [self.constant_definitions[c] for c in cons_list] # type: ignore

        for i in range(1, len(values)):
            value: PrefixSpec | OperationSpec | ConstantSpec | int = values[i]
            if isinstance(value, OperationSpec) or isinstance(value, PrefixSpec):
                function_stack.append(FunctionStackElement(value, [value.arity], []))
            else:
                if isinstance(value, ConstantSpec):
                    function_stack[-1].inpt_tab.append(constant_indicies[value])
                else: #variable
                    function_stack[-1].inpt_tab.append(value)
                function_stack[-1].rem_inpts[0] -= 1

                while function_stack[-1].rem_inpts[0]==0:
                    func_info: FunctionStackElement = function_stack.pop()
                    if isinstance(func_info.func, OperationSpec):
                        compiled.append(CompiledElement(self.operation_definitions[func_info.func], np.array(func_info.inpt_tab)))
                    else:
                        compiled.append(CompiledElement(self.prefix_definition, np.array(func_info.inpt_tab)))
                    if len(function_stack)>0:
                        function_stack[-1].inpt_tab.append(len(compiled)-1)
                        function_stack[-1].rem_inpts[0] -= 1
                    else:
                        break

        #print('\n'.join([str(i)+": "+str(c) for i, c in enumerate(compiled)]))
        return compiled
    
    @staticmethod
    def _apply_function(table: np.ndarray, *args: int | np.ndarray) -> int | np.ndarray:
        """Applys a function to its arguments

        Parameters
        ----------
        symbol : str
            Symbol of the function
        args : int | np.ndarray
            Inputs to calculate the function for.
            All should be the same type

        Returns
        -------
        int | np.ndarray
            Function result. Format depends on args
        """        
        assert len(args) == len(table.shape), "Not enough inputs to function."
        return table[*args]
    
    def _check_tautological_slow(self, compiled: list[CompiledElement | int | None]) -> tuple[bool, list[int | bool] | None]:
        """Brute force tautological check without numpy

        Parameters
        ----------
        compiled : list[None | CompiledElement]
            Compiled Expression

        Returns
        -------
        bool
            True if Tautological, otherwise False
        """        
        for vars in itertools.product(range(self.order), repeat=len([c for c in compiled if c is None])):
            result: list[int | bool] = []
            j = 0
            for i in range(len(compiled)):
                func: CompiledElement | int | None  = compiled[i]
                if func is None:
                    result.append(vars[j])
                    j += 1
                elif isinstance(func, int):
                    result.append(func)
                else:
                    result.append(func.table[*[result[j] for j in func.inputs]]) # type: ignore
            if not result[-1]:
                return False, result
            
        return True, None
    
    def _check_tautological_likely(self, compiled: list[CompiledElement | int | None]) -> tuple[bool, np.ndarray | None]:
        """Brute force tautological check with numpy

        Parameters
        ----------
        compiled : list[None | CompiledElement]
            Compiled Expression

        Returns
        -------
        bool
            True if Tautological, otherwise False
        """        
        #check all of them at once (quickly) via matrix operations
        var_count = sum([1 if c is None else 0 for c in compiled])
        results: list[np.ndarray] = []
        for i in range(len(compiled)):
            func: CompiledElement | int | None = compiled[i]
            if func is None:
                shape = [1] * var_count
                shape[i] = self.order
                results.append(np.arange(self.order).reshape(shape))
            elif isinstance(func, int):
                results.append(np.array(func))
            else:
                results.append(func.table[*[results[j] for j in func.inputs]])
        
        return bool(results[-1].all()), np.argwhere(results[-1]==0)
    
    def __call__(self, tptp_form: str, compiled: list[CompiledElement | int | None] | None = None,
                           probability: Union[Literal["Likely"], Literal["Slow"], Literal["Verify"], Literal["Fastest"]] = "Fastest") -> bool:
        """Checks if an expression is tautological

        Parameters
        ----------
        tptp_form : str
            Vampire form of expression
        compiled : list[CompiledElement | int | None] | None
            Already compiled form, by default None
        probability : Union[Literal["Likely"], Literal["Slow"], Literal["Verify"], Literal["Fastest"]]
            What tautology checker to use, by default "Fastest."
            In Verify mode all will be run and checked against eachother.

        Returns
        -------
        bool
            True if expression is tautological under this model, otherwise False

        Raises
        ------
        ValueError
            Improper inputs
        """        
        if compiled is None:
            compiled = self.compile_expression(tptp_form)

        if probability=="Likely" or probability=="Fastest":
            return self._check_tautological_likely(compiled)[0]
        elif probability=="Slow":
            return self._check_tautological_slow(compiled)[0]
        elif probability=="Verify":
            l: bool = self._check_tautological_likely(compiled)[0]
            sl: bool = self._check_tautological_slow(compiled)[0]
            assert l==sl, str(l)+", "+str(sl)
            return l
        else:
            raise ValueError
        
    def apply_function(self, op: OperationSpec | PrefixSpec, *arr: ModelArray) -> int | ModelArray:
        """Applies an function from this model to an array

        Parameters
        ----------
        op : OperationSpec | PrefixSpec
            Operator of the function to use
        arr : ModelArray
            Inputs

        Returns
        -------
        int | np.ndarray
            Result
        """        
        if isinstance(op, OperationSpec):
            return self._apply_function(self.operation_definitions[op], *arr)
        else:
            return self._apply_function(self.prefix_definition, *arr)
        
CN_STANDARD_MODEL = Model(CN_SPEC)
C0_STANDARD_MODEL = Model(C0_SPEC)
C1_STANDARD_MODEL = Model(C1_SPEC)

class ModelTable():
    """Table of counter-models with 1 target model. 
    A expression is only sent to counter modeling if:
    It's tautological under the target_model 
    """    
    spec: ModelSpec
    """Spec for the ModelTable"""
    target_model: Model
    """Targeted model to use"""
    counter_models: list[Model]
    """Counter models"""

    def __init__(self, spec: ModelSpec, target_model: Model | None = None, counter_model_folder: str | None = None) -> None:
        self.spec = spec
        if not target_model is None:
            self.target_model = target_model
        else:
            self.target_model = Model(self.spec)
        self.counter_models = []
        if not counter_model_folder is None:
            if not os.path.exists(counter_model_folder):
                os.makedirs(counter_model_folder)
            i = 0
            for filename in os.listdir(counter_model_folder):
                file_path = os.path.join(counter_model_folder, filename)
                
                if os.path.isfile(file_path):
                    i += 1
                    self += Model(self.spec, model_filename=file_path)
            
            print("Recovered "+str(i)+" models.")
    
    def __iadd__(self, new_model: Model | str) -> ModelTable:
        """Adds a new model in

        Parameters
        ----------
        new_model : Model | str
            New model to add, either a class instance or a file name

        Returns
        -------
        ModelTable
            Self with
        """        
        if isinstance(new_model, str):
            new_model = Model(self.spec, model_filename=new_model)
        for cm in self.counter_models:
            same: bool = new_model.order == cm.order
            if same:
                for cons in self.spec.constants:
                    same = same and (new_model.constant_definitions[cons] == cm.constant_definitions[cons])
            if same:
                for op in self.spec.operators:
                    same = same and (new_model.operation_definitions[op] == cm.operation_definitions[op]).all()
            assert not same
        self.counter_models.append(new_model)
        self.counter_models.sort(key = lambda x: x.order)
        return self
    
    def __call__(self, tptp_form: str) -> Literal["T"] | Literal["F"] | Literal["CM"]:
        """Determines if a expression is Tautological and Not-Countermodeled, Non-Tautological, or Countermodeled

        Parameters
        ----------
        tptp_form : str
            Vampire form of expression

        Returns
        -------
        Classification
            T = Tautological and Not-Countermodeled
            F = Non-Tautological
            CM = Countermodeled
        """        
        try:
            if not self.target_model(tptp_form):
                return "F"
            for counter_model in self.counter_models:
                if counter_model(tptp_form):
                    return "CM"
            return "T"
        except:
            raise RuntimeError(tptp_form)
        
    def verify_counter_model_sets(self, counter_modeling_formula_sets: list[list[str]]) -> None:
        """Checks if some sets of counter modeling formulas actually work with this model table.
        This means that all elements are tautological, and for all countermodels in this model table
        atleast 1 formula is non-tautological.

        Parameters
        ----------
        counter_modeling_formula_sets : list[list[str]]
            _description_
        """        
        for counter_modeling_formula_set in counter_modeling_formula_sets:
            for formula in counter_modeling_formula_set:
                assert self.target_model(formula)
        for counter_model in self.counter_models:
            valid_cm = False
            for counter_modeling_formula_set in counter_modeling_formula_sets:
                for formula in counter_modeling_formula_set:
                    if not counter_model(formula):
                        valid_cm = True
            assert valid_cm
    
    def counter_models_size_split(self, size: int) -> tuple[Iterable[Model], Iterable[Model]]:
        """Returns the counter models below and equal to or above a specific size

        Parameters
        ----------
        size : int
            Size to split at

        Returns
        -------
        tuple[Iterable[Model], Iterable[Model]]
            Iterable across Models below the size
            Iterable across Models equal to or above the size
        """        
        return [cm for cm in self.counter_models if cm.order < size], [cm for cm in self.counter_models if cm.order >= size]





