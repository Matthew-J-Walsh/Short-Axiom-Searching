from __future__ import annotations

from Globals import *
from FillTools import *

class OperationTableHandler(Protocol):
    def __call__(self, index: int, *inputs: tuple[ModelNode | np.ndarray, list[int]]) -> ModelNode | np.ndarray:
        raise NotImplementedError

class CompiledElement(NamedTuple):
    table: np.ndarray
    inputs: np.ndarray

CACHE_DEPTH_LIMIT = 5

class OperationDictionary(dict):
    """Dictonary for an operation in a model.
    """    
    _operation: np.ndarray
    _size: int

    def __init__(self, operation: np.ndarray, *args, **kwargs) -> None:
        self._operation = operation
        self._size = len(operation.shape)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: tuple[tuple[ModelNode, tuple[int, ...]], ...], value: ModelNode) -> None:
        assert isinstance(key, tuple)
        assert len(key)==self._size
        assert all(isinstance(k, ModelNode) or isinstance(k, np.ndarray) for k in key)
        assert isinstance(value, ModelTable)
        super().__setitem__(key, value)

    def __getitem__(self, key: tuple[tuple[ModelNode, tuple[int, ...]], ...]) -> ModelNode:
        assert isinstance(key, tuple)
        assert len(key)==self._size
        assert all(isinstance(node, ModelNode) and isinstance(rotation, tuple) and all(isinstance(i, int) for i in rotation) for node, rotation in key)
        return super().__getitem__(key)
    
    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            assert isinstance(key, tuple)
            assert len(key)==self._size
            assert all(isinstance(node, ModelNode) and isinstance(rotation, list) and all(isinstance(i, int) for i in rotation) for node, rotation in key)
            assert isinstance(value, ModelTable)
        super().update(*args, **kwargs)

    def calculate(self, *models: tuple[ModelNode | np.ndarray, list[int]]) -> tuple[ModelNode | np.ndarray, list[int]]:
        assert len(models)==self._size

        pure_nodes = True
        depth: int = 0
        cannonical_rotations: list[list[int]] = []
        rot_binding: dict[int, int] = {}
        inv_rot_binding: dict[int, int] = {}
        c = 0
        for model, rot in models:
            cannonical_rotations.append([])
            for r in rot:
                if not r in rot_binding.keys():
                    rot_binding[r] = c
                    inv_rot_binding[c] = r
                    c += 1
                cannonical_rotations[-1].append(rot_binding[r])
            if isinstance(model, ModelNode):
                depth = max(depth, model.depth)
            else:
                pure_nodes = False
                depth = CACHE_DEPTH_LIMIT
        depth += 1
        
        def _compute_new_arr() -> np.ndarray:
            raise NotImplementedError

        if pure_nodes:
            cannonical_models: tuple[tuple[ModelNode, tuple[int, ...]], ...] = tuple(zip([model for model, rot in models], [tuple(rot) for rot in cannonical_rotations])) # type: ignore

            if not cannonical_models in self.keys():
                arr: np.ndarray = _compute_new_arr()
                if depth <= CACHE_DEPTH_LIMIT:
                    self[cannonical_models] = ModelNode(arr, depth)
            node: ModelNode = self[cannonical_models]
            return node, list(inv_rot_binding.values())
        else:
            arr = _compute_new_arr()
            return arr, list(inv_rot_binding.values())
    
class ModelNode:
    """Node containing a cached model element
    """    
    arr: np.ndarray
    _hash: int | None
    depth: int

    def __init__(self, arr: np.ndarray, depth: int = 0):
        self.arr = arr.copy()
        self.arr.setflags(write=False)
        self._hash = None
        self.depth = depth

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(np.array_repr(self.arr))
        return self._hash

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
    """Same as above but much slower but definantly works.
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
def array_dimensional_reorganizer(target_size: int, current: tuple[int, ...]) -> tuple[Any, ...]:
    return tuple([slice(None) if i in current else np.newaxis for i in range(target_size)])

class VampireOutputTools:
    """Tools to process a vampire finite model
    """    
    variable_regex: re.Pattern = re.compile(r"tff\(declare_\$i(\d+),type,(\w+|\bfmb_\$i_\d+\b):\$i\)")
    function_def_regex: re.Pattern = re.compile(r"tff\(declare_(\w+),type,\1: (\$i(?: \* \$i)*) > \$i\)")
    function_val_regex: re.Pattern = re.compile(r"(\w)\(fmb_\$i_(\d+)(?:,fmb_\$i_(\d+))?\) = fmb_\$i_(\d+)")
    #TODO: Predicates

    @staticmethod
    def order_and_constants(result: str) -> tuple[str, int, dict[str, int]]:
        """Determines the order and constant values, fixes the results to not have constants

        Parameters
        ----------
        result : str
            Vampire output text

        Returns
        -------
        tuple[str, int, dict[str, int]]
            Fixed output text,
            Order of the countermodel,
            Constants dict
        """        
        result = copy.copy(result)
        matches = re.findall(VampireOutputTools.variable_regex, result)

        order: int = max([int(m[0]) for m in matches])
        constants: dict[str, int] = {c: i for i, (_, c) in enumerate(matches) if len(c)==1}

        for symbol, value in constants.items():
            result = re.sub(r'\b'+re.escape(symbol)+r'\b', "fmb_$i_"+str(value + 1), result)

        return result, order, constants

    @staticmethod
    def functions_arity_verification(result: str, arities: dict[str, int]) -> None:
        """Verifies the arity of the functions in a vampire output

        Parameters
        ----------
        result : str
            Vampire output text
        arities : dict[str, int]
            Arity of functions
        """        
        matches = re.findall(VampireOutputTools.function_def_regex, result)
        for m in matches:
            assert arities[m[0]] == m[1].count(r"$i")
    
    @staticmethod
    def function_parse(result: str, function_identifier: str, order: int, arity: int) -> np.ndarray:
        """Parses a function out of a vampire output

        Parameters
        ----------
        result : str
            Vampire output text
        function_identifier : str
            Function identifer for targeted function
        order : int
            Order of the model
        arity : int
            Ariuty of the function

        Returns
        -------
        np.ndarray
            _description_
        """        
        arr = np.full([order]*arity, -1, np.int8)
        matches = re.findall(VampireOutputTools.function_val_regex, result)
        for m in matches:
            if m[0]==function_identifier:
                assert len(m)-2 == arity, "Incorrect arity assignment found."
                arr[tuple(int(i) for i in m[1:-1])] = np.int8(m[-1])
        assert (arr==-1).sum() == 0, "Not all outputs were bound."
        return arr

class Model():
    """A model to check logical expressions.
    Assumes a prefix unitary proof operator

    Members
    -------
    order: int
        Order of the model
    operation_tables: list[np.ndarray]
        Tables for how each operation works
    constant_table : list[int]
        Table of the value of each constant
    _arities: np.ndarray
        Number of inputs for each operator (derived)
    """    
    order: int
    operation_definitions: dict[OperationSpec, np.ndarray]
    constant_definitions: dict[ConstantSpec, int]

    def __init__(self, spec: ModelSpec, operation_definitions: dict[OperationSpec, np.ndarray] | None = None, 
                 constant_defnitions: dict[ConstantSpec, int] | None = None, model_filename: str | None = None) -> None:
        if not operation_definitions is None:
            self.operation_definitions = operation_definitions
            if not constant_defnitions is None:
                self.constant_definitions = constant_defnitions
            else:
                self.constant_definitions = {}
        elif not model_filename is None:
            with open(model_filename, 'r') as f:
                result, order, constants = VampireOutputTools.order_and_constants(f.read())
                self.constant_definitions = {c: constants[c.vampire_symbol] for c in spec.constants}
                self.operation_definitions = {}
                VampireOutputTools.functions_arity_verification(result, {op.vampire_symbol: op.arity for op in spec.operators})
                for op in spec.operators:
                    self.operation_definitions[op] = VampireOutputTools.function_parse(result, op.vampire_symbol, order, op.arity)
        else:
            self.operation_definitions = {op: op.default_table for op in spec.operators}
            self.constant_definitions = {c: c.default_value for c in spec.constants}
        
        self.order = self.operation_definitions[spec.operators[0]].shape[0]

    def calculate(self, op: OperationSpec, output_size: int, inputs: tuple[tuple[ModelArray, DimensionalReference], ...]) -> ModelArray:
        #takes inputs, returns output with the fill so like [5, 2, 1] -> [3, 2, 4, 5] would give [5, 2, 1, 3, 4]
        assert len(inputs)==op.default_table.ndim
        #why wouldn't I just sort it?
        #full_dimensions: DimensionalReference = fill_dimensions(full_fill)
        reorganized_models: list[ModelArray] = [arr[*array_dimensional_reorganizer(output_size, sub_dims)] for arr, sub_dims in inputs]
        out = self._apply_function(self.operation_definitions[op], *reorganized_models)
        assert isinstance(out, np.ndarray)
        assert out.ndim == output_size
        return out

    def _get_values(self, vampire_form: str) -> list[OperationSpec | ConstantSpec | int]:
        stripped: str = vampire_form.replace('(', '').replace(')', '').replace(',', '')
        raise NotImplementedError

    def compile_expression(self, vampire_form: str) -> list[CompiledElement | int | None]:
        """Compiles the compression into a more easily managed form

        Parameters
        ----------
        vampire_form : str
            Vampire form of expression

        Returns
        -------
        TODO
        """        
        values: list[OperationSpec | ConstantSpec | int] = self._get_values(vampire_form)
        try:
            assert isinstance(values[0], OperationSpec), "Un-prefixed Expression" 
            assert self.operation_definitions[values[0]].dtype == np.bool_, "Un-prefixed Expression" 
            assert values.count(values[0])==1, "Prefix must be unique"
        except:
            print(vampire_form)
            raise AssertionError
        function_stack: list[FunctionStackElement] = [FunctionStackElement(values[0], [values[0].arity], [])] #prefix arity
        var_count = max(v for v in values if isinstance(v, int)) + 1
        variable_indicies: dict[int, int] = {i+1: i for i in range(var_count)}
        cons_list: list[ConstantSpec] = list(self.constant_definitions.keys())
        constant_indicies: dict[ConstantSpec, int] = {c: i+var_count for i, c in enumerate(cons_list)}
        compiled: list[CompiledElement | int | None] = [None] * var_count + [self.constant_definitions[c] for c in cons_list] # type: ignore

        for i in range(1, len(values)):
            value: OperationSpec | ConstantSpec | int = values[i]
            if isinstance(value, OperationSpec):
                function_stack.append(FunctionStackElement(value, [value.arity], []))
            else:
                if isinstance(value, ConstantSpec):
                    function_stack[-1].inpt_tab.append(constant_indicies[value])
                else: #variable
                    function_stack[-1].inpt_tab.append(variable_indicies[value])
                function_stack[-1].rem_inpts[0] -= 1

                while function_stack[-1].rem_inpts[0]==0:
                    func_info: FunctionStackElement = function_stack.pop()
                    compiled.append(CompiledElement(self.operation_definitions[func_info.func], np.array(func_info.inpt_tab)))
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
    
    def _check_tautological_slow(self, compiled: list[CompiledElement | int | None]) -> bool:
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
                return False#, np.array(result)
            
        return True#, np.zeros(0)
    
    def _check_tautological_likely(self, compiled: list[CompiledElement | int | None]) -> bool:
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
                shape = [1] * var_count
                shape[i] = self.order
                results.append(np.full(self.order, func).reshape(shape))
            else:
                results.append(func.table[*[results[j] for j in func.inputs]])
        
        return bool(results[-1].all())#, np.argwhere(results[-1]==0)
    
    def __call__(self, vampire_form: str, compiled: list[CompiledElement | int | None] | None = None,
                           probability: Union[Literal["Likely"], Literal["Slow"], Literal["Verify"], Literal["Fastest"]] = "Verify") -> bool:
        """Checks if an expression is tautological

        Parameters
        ----------
        vampire_form : str
            Vampire form of expression
        probability : Union[Literal[&quot;Likely&quot;], Literal[&quot;Slow&quot;], Literal[&quot;Verify&quot;]], optional
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
            compiled = self.compile_expression(vampire_form)

        if probability=="Likely" or probability=="Fastest":
            return self._check_tautological_likely(compiled)
        elif probability=="Slow":
            return self._check_tautological_slow(compiled)
        elif probability=="Verify":
            l: bool = self._check_tautological_likely(compiled)
            sl: bool = self._check_tautological_slow(compiled)
            assert l==sl, str(l)+", "+str(sl)
            return l
        else:
            raise ValueError
        
CN_STANDARD_MODEL = Model(CN_SPEC)
C0_STANDARD_MODEL = Model(C0_SPEC)
C1_STANDARD_MODEL = Model(C1_SPEC)

class ModelTable():
    """Table of counter-models with 1 target model. 
    A expression is only sent to counter modeling if:
    It's tautological under the target_model 
    """    
    spec: ModelSpec
    target_model: Model
    counter_models: list[Model]

    def __init__(self, spec: ModelSpec, target_model: Model | None = None, counter_model_folder: str | None = "counter_models") -> None:
        self.spec = spec
        if not target_model is None:
            self.target_model = target_model
        else:
            self.target_model = Model(self.spec)
        self.counter_models = []
        if not counter_model_folder is None:
            if not os.path.exists(counter_model_folder):
                os.makedirs(counter_model_folder)
            for filename in os.listdir(counter_model_folder):
                file_path = os.path.join(counter_model_folder, filename)
                
                if os.path.isfile(file_path):
                    self += Model(self.spec, model_filename=file_path)
    
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
        if isinstance(new_model, Model):
            self.counter_models.append(new_model)
        else: #isinstance(new_model, str)
            self.counter_models.append(Model(self.spec, model_filename=new_model))
        self.counter_models.sort(key = lambda x: x.order)
        return self
    
    def __call__(self, vampire_form: str) -> Literal["T"] | Literal["F"] | Literal["CM"]:
        """Determines if a expression is Tautological and Not-Countermodeled, Non-Tautological, or Countermodeled

        Parameters
        ----------
        vampire_form : str
            Vampire form of expression

        Returns
        -------
        Classification
            T = Tautological and Not-Countermodeled
            F = Non-Tautological
            CM = Countermodeled
        """        
        try:
            if not self.target_model(vampire_form):
                return "F"
            for counter_model in self.counter_models:
                if counter_model(vampire_form):
                    return "CM"
            return "T"
        except:
            raise RuntimeError(vampire_form)
    





