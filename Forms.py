from __future__ import annotations

from FillTools import FillTable
from Globals import *
from FillTools import *
from ModelTools import *

from sympy.utilities.iterables import multiset_permutations

from VampireUtils import VampireWrapper


class FormOpSpec(NamedTuple):
    """Specification for a operation in a form
    """    
    operation: OperationSpec 
    """Operation to use"""
    number: int 
    """Count of the operation in the form"""

class FormConstSpec(NamedTuple):
    """Specification for a constant in a form
    """    
    constant: ConstantSpec 
    """Symbol to use"""
    number: int 
    """Count of the constant in the form"""

class Form():
    operation_symbols: list[str]
    """List of symbols being used for operations in lexiographical order"""
    constant_symbols: list[str]
    """List of symbols being used for constants in lexiographical order"""
    arities: dict[str, int]
    """Arity of the operations"""
    _arity_lookup: np.ndarray
    """Quick lookup table for arities"""
    _arity_delta_lookup: np.ndarray
    """Quick lookup table for how an element changes the parens depth"""
    counts: dict[str, int]
    """Number of each operator"""
    _counts_lookup: np.ndarray
    """Quick lookup table for counts"""
    length: int
    """Length of whole form"""
    var_count: int
    """Number of variables in the form"""
    base_element: np.ndarray
    """First element (derived)"""
    _array: np.ndarray | None
    """Saved array for pickle recovery"""

    def __init__(self, operations: Sequence[FormOpSpec], constants: Sequence[FormConstSpec] = []):
        """
        Parameters
        ----------
        operations : Sequence[OperationSpec]
            Operations making up the Form
        constants : Sequence[FormConstSpec]
            Constants in the Form
        """        
        self.operation_symbols = [func.operation.symbol for func in operations]
        self.constant_symbols = [cons.constant.symbol for cons in constants]
        self.arities = {func.operation.symbol: func.operation.arity for func in operations}
        self._arity_lookup = np.array([self.arities[s] for s in self.operation_symbols]+[0 for _ in constants]+[0], dtype=np.int8)
        self._arity_delta_lookup = np.array([self.arities[s]-1 for s in self.operation_symbols]+[-1 for _ in constants]+[-1], dtype=np.int8)
        self.counts = {**{func.operation.symbol: func.number for func in operations}, **{cons.constant.symbol: cons.number for cons in constants}}
        self.length = sum([self.arities[s] * self.counts[s] for s in self.operation_symbols]) + 1
        self.var_count = self.length - sum(self.counts.values())
        self._counts_lookup = np.array([self.counts[s] for s in self.operation_symbols + self.constant_symbols]+[self.var_count], dtype=np.int8)
        self.base_element = np.array(sum([[i - len(self.operation_symbols) - len(self.constant_symbols)] * self.counts[s] for i, s in enumerate(self.operation_symbols + self.constant_symbols)], [])+[0] * self.var_count, dtype=np.int8)
        self.base_element.setflags(write = False)
        self._array = None

    def _super_element(self, element: np.ndarray, order: int) -> None:
        """Strips an element of symbols 'below' a specific lexographical order

        Parameters
        ----------
        element : np.ndarray
            Form element, can be partially formed
        order : int
            What (negative) order of operation symbols to strip
        """        
        element[element>order] = 0
    
    def _order_grounded(self, order: int) -> np.ndarray:
        count = sum([self.counts[s] for s in self.operation_symbols[:len(self.operation_symbols) + len(self.constant_symbols) + order + 1]]) +\
                     sum([self.counts[s] for s in self.constant_symbols[:len(self.constant_symbols) + order + 1]])
        return self.base_element[count:]
    
    def _ground_element(self, element: np.ndarray, order: int) -> None:
        """Grounds an element at an order
        If either input is default will give the base element

        Parameters
        ----------
        element : np.ndarray | None, optional
            Form element, by default None
        order : int, optional
            Order to take the ground at, by default 0
        """       
        try:
            self._super_element(element, order)
            element[element>order] = self._order_grounded(order)
        except:
            raise ValueError("")

    def _reset_right(self, element: np.ndarray, order: int, point: int) -> None:
        """Resets the at order elements to the right of a point.
        This will shift all symbols at the order to the right of a point as far left as possible

        Parameters
        ----------
        element : np.ndarray
            Form element to reset
        order : int
            What order is being targeted
        point : int
            What point to reset from
        """        
        left_count = (element[:point] == order).sum()
        empty_elements = np.where(element>=order)[0]
        empty_elements = empty_elements[empty_elements >= point]
        remaining_inject = self._order_grounded(order - 1)[left_count:]
        inject_count = min(len(remaining_inject), len(empty_elements))
        element[empty_elements[:inject_count]] = remaining_inject[:inject_count]
        
    def isa_member(self, element: np.ndarray) -> bool:
        """Check if a given element is a valid member of this Form

        Parameters
        ----------
        element: np.ndarray
            Form element

        Returns
        -------
        bool
            If the given element is a valid member of this Form
        """        
        fixed_elems = element + len(self.operation_symbols) + len(self.constant_symbols)
        arity_arr = self._arity_delta_lookup[(fixed_elems[:-1])]
        if (np.cumsum(arity_arr)<0).any():
            return False #Ends early
        if len(element)!=self.length:
            return False #Incorrect length
        if (np.bincount(fixed_elems) != self._counts_lookup).any():
            return False #Incorrect counts
        return True
    
    def _next(self, element: np.ndarray, order: int) -> None:
        """Turns element into the next lexicographically ordered element

        Parameters
        ----------
        element : np.ndarray
            Form element
        order : int
            Order being processed at, default: highest order
            Should only be set in this operations recursive call

        Returns
        -------
        str
            Next lexicographically ordered element
        """        
        if order < -1 * (len(self.operation_symbols) + len(self.constant_symbols)) - 2:
            return None #given element is the last element, there is no next, give a malformed element
        
        self._super_element(element, order+1)

        for i in range(len(element)-2, -1, -1):
            if element[i]==order:
                for j in range(1, len(element)-i):
                    if element[i+j] > element[i]: #symbol at order is right shiftable j over
                        element[i] = 0
                        element[i+j] = order
                        self._reset_right(element, order, i+j+1) #reset symbols at the same order back into a left state
                        self._ground_element(element, order) #ground everything of a higher order
                        if self.isa_member(element): #valid shift, element is now next
                            return None
                        else: #this wasn't a valid shift
                            break
        
        self._next(element, order - 1) #no new valid states exist at this order, move to a lower one

    def __iter__(self) -> Iterable[np.ndarray]:
        current_element = self.base_element.copy()
        while self.isa_member(current_element):
            yield current_element
            self._next(current_element, -1)
    
    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            arr = []
            for element in self.__iter__():
                arr.append(element.copy())
            self._array = np.vstack(arr)
        return self._array

    @property
    def raw_all(self) -> np.ndarray:
        arr = []
        for element in multiset_permutations(self.base_element.tolist()):
            parr = np.array(element, dtype=np.int8)
            if self.isa_member(parr):
                arr.append(parr)
        return np.vstack(arr)
    
    def check_array(self):
        """Verify array with raw_all()
        """        
        array = self.array
        raw_all = self.raw_all
        print(array.shape == raw_all.shape)
        print(np.array_equal(np.sort(array, axis=0), np.sort(raw_all, axis=0)))


class FormTable():
    """Forms of a specific length
    """
    length: int
    """Length of all forms"""
    operations: list[OperationSpec]
    """Operations in the table"""
    constants: list[ConstantSpec]
    """Constants in the table"""
    prefix: OperationSpec
    """Operation to attach to the front of all expressions (usually evaluates truth)"""
    sub_forms: list[Form]
    """Forms making up the table (derived)"""
    _translation_table: dict[int, str]
    """Translations from values to the corresponding strings"""

    def __init__(self, length: int, prefix: OperationSpec, operations: Sequence[OperationSpec], constants: Sequence[ConstantSpec] = []) -> None:
        self.length = length
        self.prefix = prefix
        self.operations = list(operations)
        self.constants = list(constants)

        self.sub_forms = []
        #ehh. just do brute force
        for op_counts in itertools.product(range(1, self.length), repeat=len(self.operations)):
            if 1 + sum([op.arity * count for op, count in zip(operations, op_counts)]) == self.length:
                form_ops: list[FormOpSpec] = [FormOpSpec(op, count) for op, count in zip(operations, op_counts)]
                if len(self.constants) > 0:
                    for const_count in itertools.product(range(1, self.length - sum(op_counts) - 1), repeat=len(self.constants)):
                        self.sub_forms.append(Form(form_ops, [FormConstSpec(cons, count) for cons, count in zip(self.constants, const_count)]))
                else:
                    self.sub_forms.append(Form(form_ops))
                assert self.sub_forms[-1].length==self.length, "Made an incorrect length form"
            
        self._translation_table = {}
        self._translation_table[-len(self.operations)-len(self.constants)-1] = self.prefix.symbol
        for i in range(len(self.operations)):
            self._translation_table[i-len(self.operations)-len(self.constants)] = self.operations[i].symbol
        for i in range(len(self.constants)):
            self._translation_table[i-len(self.constants)] = self.constants[i].symbol
        for i in range(1, 1 + len(VARIABLE_SYMBOLS)):
            self._translation_table[i] = VARIABLE_SYMBOLS[i-1]
    
    def __iter__(self) -> Iterable[str]:
        for form in self.sub_forms:
            for element in form.__iter__():
                yield self.prefix.symbol + element
        
    @property
    def array(self) -> list[tuple[np.ndarray, int]]:
        return [(np.hstack((np.full((sf.array.shape[0], 1), -len(self.operations)-len(self.constants)-1, dtype=np.int8), sf.array)), sf.var_count) for sf in self.sub_forms] #second dimension
    
    def get_polish(self, expression: np.ndarray) -> str:
        """Gets the polish form of an expression in this FormTable

        Parameters
        ----------
        expression : np.ndarray
            Expression

        Returns
        -------
        str
            Polish Form
        """        
        return ''.join(self._translation_table[e] for e in expression)

    def from_polish(self, polish: str) -> np.ndarray:
        """Gets the numpy form of a polish expression

        Parameters
        ----------
        polish : str
            Polish Form

        Returns
        -------
        np.ndarray
            Numpy Form
        """        
        raise NotImplementedError
        var_iter = 1
        variable_map: dict[str, int] = {}
        op_map: dict[str, int] = {op.symbol: i - len(self.operations) - len(self.constants) for i, op in enumerate(self.operations)}
        op_map[self.prefix.symbol] = - len(self.operations) - len(self.constants) - 1
        cons_map = {cons: i - len(self.constants) for i, cons in enumerate(self.constants)}

        def _convert(c: str) -> int:
            nonlocal var_iter
            if c in variable_map.keys():
                return variable_map[c]
            elif c in op_map.keys():
                return op_map[c]
            else:
                variable_map[c] = var_iter
                var_iter += 1
                return variable_map[c]
            
        return np.array([_convert(c) for c in polish], dtype=np.int8)
        
    def get_vampire(self, expression: np.ndarray | str) -> str:
        """Gets the full notation for vampire

        Parameters
        ----------
        expression : np.ndarray | str
            Numpy or Polish form

        Returns
        -------
        str
            Vampire Form
        """        
        raise NotImplementedError
        if isinstance(expression, str):
            expression = self.from_polish(expression)
        
        text: str = ""
        assert expression[0] == - len(self.operations) - len(self.constants) - 1, "Doesn't start with prefix."
        operator_symbol_offset: int = len(self.operations) + len(self.constants) + 1
        constant_symbol_offset: int = len(self.constants)
        var_count = expression.max()
        variable_indicies: dict[int, int] = {i+1: i for i in range(var_count)}
        constant_indicies: dict[int, int] = {i - constant_symbol_offset: i+var_count for i in range(constant_symbol_offset)}

        function_stack: list[FunctionStackElement] = [FunctionStackElement(expression[0], [self.prefix.arity], [])]
        for i in range(1, len(expression)):
            if expression[i] < -constant_symbol_offset:
                func_idx: int = expression[i] + operator_symbol_offset
                if func_idx==0:
                    op: OperationSpec = self.prefix
                else:
                    op: OperationSpec = self.operations[func_idx-1]
                text += op.vampire_symbol+"("
                function_stack.append(FunctionStackElement(func_idx, [op.arity], []))
            else: #expression[i] > 0:
                if expression[i] > 0: #variable
                    function_stack[-1].inpt_tab.append(variable_indicies[expression[i]])
                    text += VARIABLE_SYMBOLS[expression[i]-1]
                else: #constant
                    function_stack[-1].inpt_tab.append(constant_indicies[expression[i]])
                    text += self.constants[expression[i] + constant_symbol_offset].vampire_symbol
                function_stack[-1].rem_inpts[0] -= 1

                text += ")" if function_stack[-1].rem_inpts[0]==0 else ","
                while function_stack[-1].rem_inpts[0]==0:
                    function_stack.pop()
                    if len(function_stack)>0:
                        text += ")" if function_stack[-1].rem_inpts[0]==0 else ","
                        function_stack[-1].rem_inpts[0] -= 1
                    else:
                        text += ")"
                        break

        return text
            
    def process(self, model_table: ModelTable, vampire_wrapper: VampireWrapper, remaining_filename: str) -> int:
        """Processes this whole FormTable's expressions.
        Must be here so that we can access the prefix

        Parameters
        ----------
        model_table : ModelTable
            Tautology check and Countermodeling table
        vampire_wrapper : Callable[[str], bool  |  Model]
            Vampire function, returns False if no model was found, otherwise returns the model
        remaining_filename : str
            Filename to place the rest of the

        Returns
        -------
        int
            How many unsolved expressions were added to the remaining file
        """        
        unsolved_count = 0

        with open(remaining_filename, 'w') as reamaining_file:
            for arr, var_count in self.array:
                fills: FillTable = FillTable.get_fill_table(var_count)
                for i in range(arr.shape[0]):
                    farr = Fillin(arr[i], fills.fills)
                    for j in range(fills.fills.shape[0]):
                        cleaver = np.ones(fills.fills.shape[1])
                        if cleaver[j]:
                            match model_table(farr[j]):
                                case "F":
                                    cleaver *= fill_downward_cleave(j, var_count)
                                case "CM":
                                    cleaver *= fill_upward_cleave(j, var_count)
                                case "T":
                                    vampire_result: bool | Model = vampire_wrapper(self.get_vampire(farr[j]))#, farr[j])
                                    if vampire_result==False:
                                        reamaining_file.write(self.get_polish(farr[j])+"\n")
                                        unsolved_count += 1
                                    else:
                                        assert isinstance(vampire_result, Model), "Vampire wrapper shouldn't return True, only models or false."
                                        model_table += vampire_result
                                        cleaver *= fill_upward_cleave(j, var_count)

        return unsolved_count











