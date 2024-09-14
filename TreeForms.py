from __future__ import annotations

from Globals import *
from ModelTools import *
from VampireUtils import *
from FillTools import *

CountArray = np.ndarray[Any, np.dtype[np.int8]]

class TreeForm:
    """Class for making forms via a tree layout.
    This uses the idea that Expressions (or functions) can be described as a tree, where each node (non-leaf) of the tree is a operator, 
    each branch is an input to said function, and leaves are constants or variables.
    An instance of this is less of a "tree" than it is a "species of trees"
    """    
    LEXOGRAPHICAL_REFERENCE: tuple[OperationSpec | ConstantSpec, ...]
    """References for all Operations and Constants in a particular order (will define iteration order, not nessisarily efficiency, and definantly not the total possibilities)"""
    _PREFIX: OperationSpec
    """Prefix operator"""
    _MAXIMUM_FULLY_CACHED_NODE_SIZE: int
    """Maximum size Node to make PsudeoNodes for"""
    _PSUDEO_NODE_CACHE: dict[int, tuple[TreeForm.Node, ...]]
    """Cache of items used for PsudeoNode iteration"""

    def __init__(self, lexographical_reference: tuple[OperationSpec | ConstantSpec, ...], prefix: OperationSpec, MAXIMUM_FULLY_CACHED_NODE_SIZE: int) -> None:
        assert prefix.arity==1, "Non-Unitary prefixes not implemented at this time"
        self.LEXOGRAPHICAL_REFERENCE = lexographical_reference
        self._PREFIX = prefix
        self._MAXIMUM_FULLY_CACHED_NODE_SIZE = MAXIMUM_FULLY_CACHED_NODE_SIZE
        self._PSUDEO_NODE_CACHE = {}
        for i in range(1, self._MAXIMUM_FULLY_CACHED_NODE_SIZE+1):
            self._PSUDEO_NODE_CACHE[i] = self._build_psudeo_node_cache(i)

    def new_node(self, size: int) -> Node:
        """Creates (an appropriate) new Node in this Tree, will be a PsudeoNode if size is small enough

        Parameters
        ----------
        size : int
            Size of node to make

        Returns
        -------
        Node
            New Node
        """        
        if size <= self._MAXIMUM_FULLY_CACHED_NODE_SIZE:
            return self.PsudeoNode(self, size)
        return self.Node(self, size)

    @staticmethod
    def _valid_symbol(value: OperationSpec | ConstantSpec | None, size: int) -> bool:
        """Determines if a value is valid for the node of a specific size.
        For size = 1, only constants or variables are allowed
        For size > 1, only functions with arity < size - 1 are allowed

        Parameters
        ----------
        value : OperationSpec | ConstantSpec | None
            An operator, constant
        size : int
            Node size

        Returns
        -------
        bool
            If the value is valid for the node size
        """    
        if size==1:
            return not isinstance(value, OperationSpec)
        if not isinstance(value, OperationSpec):
            return False
        else:
            return size > value.arity

    class Node:
        """A node of a tree, must be attached to the tree, and have a fixed size.

        Must also implement the following functions: calculate, iterate, counts, copy, polish, and vampire
        """        
        tree: TreeForm
        """The tree (species) this node is a part of"""
        size: int
        """Size of the expression this node generates"""
        branches: list[TreeForm.Node]
        """Branches of this node"""
        _value: int
        """Current refrence in the TreeForm's Lexography that this Node is currently equal to"""
        _frozen: bool
        """If this Node is frozen and shouldn't be edited"""
        _cache: dict[Model, ModelArray]
        """Cache of (full) output arrays of this node"""
        _count_cache: CountArray | None
        """Cache of the count of this node"""
        _vampire_cache: str | None
        """Cache of the vampire string"""

        def __init__(self, tree: TreeForm, size: int, external_degeneracy: CountArray | None = None, frozen: bool = False, fast_init: bool = False) -> None:
            self.tree = tree
            self.size = size
            self.branches = []
            self._frozen = frozen
            self._reset_caches()
            if not fast_init:
                self._initilize()
                if not external_degeneracy is None:
                    raise NotImplementedError

        def _reset_caches(self) -> None:
            """Clears the cache
            """            
            self._cache = {}
            self._count_cache = None
            self._vampire_cache = None
        
        def freeze(self) -> None:
            """Freezes this node, will throw an error if it is iterated
            """            
            self._frozen = True

        @property
        def _current_value(self) -> OperationSpec | ConstantSpec | None:
            """The current Operation, Constant, or Variable (None)
            """            
            return self.tree.LEXOGRAPHICAL_REFERENCE[self._value] if self._value != len(self.tree.LEXOGRAPHICAL_REFERENCE) else None

        def _iterate_value(self) -> bool:
            """Iterates the value to next valid symbol

            Returns
            -------
            bool
                If there is a valid next symbol
            """            
            while True:
                self._value += 1
                if self._value > len(self.tree.LEXOGRAPHICAL_REFERENCE):
                    return False
                if TreeForm._valid_symbol(self._current_value if self._value < len(self.tree.LEXOGRAPHICAL_REFERENCE) else None, self.size):
                    if self.size > 1:
                        self._initialize_at_value()
                    return True
                
        def _initialize_at_value(self) -> None:
            """Initalizes the Node at a particular value
            """            
            self._reset_caches()
            op: OperationSpec | ConstantSpec | None = self._current_value
            assert isinstance(op, OperationSpec)
            full_node: TreeForm.Node = self.tree.new_node(self.size - op.arity)
            self.branches = [self.tree.new_node(1) for _ in range(op.arity - 1)]
            self.branches.insert(0, full_node)
            assert self.size - 1 - sum(b.size for b in self.branches) == 0

        def _initilize(self) -> None:
            """Initalizes the Node to the first valid Lexographical value.
            """            
            self._value = -1
            self._reset_caches()
            assert self._iterate_value(), "Malformation"

        def calculate(self, model: Model, fill: Fill) -> ModelArray:
            """Calculates the Node's truth table for a particular model

            Parameters
            ----------
            model : Model
                Model to calculate for
            fill : FillArray
                Fill to calculate with (currently just generates everything)

            Returns
            -------
            ModelArray
                Result array
            """            
            assert self.counts[-1]==fill.size, fill.size - self.counts[-1]
            
            if len(self.branches)==0:
                if not model in self._cache.keys():
                    val: OperationSpec | ConstantSpec | None = self._current_value if self._value < len(self.tree.LEXOGRAPHICAL_REFERENCE) else None
                    if isinstance(val, ConstantSpec):
                        assert fill.size==0
                        self._cache[model] = np.full(model.order, model.constant_definitions[val])
                    elif val is None:
                        assert fill.size==1
                        self._cache[model] = np.arange(model.order)
                    else:
                        raise AssertionError("Malformation")
                assert self._cache[model].ndim == fill.size, fill.size
                return self._cache[model]
            else:
                if not model in self._cache.keys():
                    op: OperationSpec | ConstantSpec | None = self._current_value
                    assert isinstance(op, OperationSpec)

                    fill_splits: tuple[tuple[Fill, DimensionalReference], ...] = split_fill(fill, self.branch_var_counts)
                    branch_calcs: tuple[tuple[ModelArray, DimensionalReference], ...] = tuple([(b.calculate(model, sub_fill), tuple([injection[i] for i in fill_dimensions(sub_fill)])) for b, (sub_fill, injection) in zip(self.branches, fill_splits)])

                    self._cache[model] = model.calculate(op, fill.size, branch_calcs)

                    assert self._cache[model].ndim == fill.size, str(fill.size - self._cache[model].ndim)+"\n"+str(fill_dimensions(fill))
                return self._cache[model]
            
        def _is_degenerate(self, external_degeneracy: CountArray) -> bool:
            """Determines if this Node's current value will cause the entire Tree to be degenerate, 
            lacking atleast one lexographical element

            Parameters
            ----------
            external_degeneracy : CountArray
                External counts of each lexographical elements

            Returns
            -------
            bool
                If the tree will be degenerate or not
            """            
            return bool((external_degeneracy + self.counts <= 0).any())
        
        def _iterate_inner(self, external_degeneracy: CountArray) -> bool:
            """Iterates this node, while ignore its own degeneracy

            Parameters
            ----------
            external_degeneracy : CountArray
                External counts of each lexographical elements

            Returns
            -------
            bool
                If the node was able to be iterated (before degeneragy check)

            Raises
            ------
            RuntimeError
                If the Node if frozen
            """            
            if self._frozen:
                raise RuntimeError("Attempted to iterate static Node.")
            self._reset_caches()
            
            #Try to change smallest branches first
            for b in sorted(self.branches, key = lambda x: x.size):
                if b.iterate(external_degeneracy + self.counts - b.counts):
                    self._reset_caches()
                    return True
                
            #Once unable to change the biggest branch, move element count rightward
            for i in range(len(self.branches)-1):
                if self.branches[i].size > 1:
                    #print(i)
                    self.branches[i+1] = self.tree.new_node(self.branches[i+1].size + 1)
                    #print(len(self.branches[i+1:]))
                    self.branches[0] = self.tree.new_node(self.size - 1 - i - sum(b.size for b in self.branches[i+1:]))
                    for j in range(1, i+1):
                        print(j)
                        self.branches[j] = self.tree.new_node(1)
                    assert self.size - 1 - sum(b.size for b in self.branches) == 0, [b.size for b in self.branches]
                    self._reset_caches()
                    return True
            
            #Once unable to move element count try to iterate value
            if self._iterate_value():
                self._reset_caches()
                return True

            #Remember to re-initialize when returning false so we return to "base" state.
            self._initilize()
            self._reset_caches()
                
            return False

        def iterate(self, external_degeneracy: CountArray) -> bool:
            """Iterates self until its non-degenerate (if possible)

            Parameters
            ----------
            external_degeneracy : CountArray
                External counts of each lexographical elements

            Returns
            -------
            bool
                True if it was able to be iterated, False if unable and had to be reset.
            """            
            while True:
                result: bool = self._iterate_inner(external_degeneracy)
                if not result:
                    return False
                if not self._is_degenerate(external_degeneracy):
                    return True
                
        @property
        def counts(self) -> CountArray:
            """Count of each lexographical element in this node and all branches

            Returns
            -------
            CountArray
                Counts of each lexographical element
            """            
            if self._count_cache is None:
                #typing hates this completely valid code
                if len(self.branches)==0:
                    self._count_cache = np.zeros(len(self.tree.LEXOGRAPHICAL_REFERENCE)+1, dtype=np.int8) # type: ignore
                else:
                    self._count_cache = sum([b.counts for b in self.branches]) # type: ignore
                self._count_cache[self._value] += 1 # type: ignore
                self._count_cache.setflags(write = False) # type: ignore
            return self._count_cache # type: ignore
        
        def copy(self) -> TreeForm.Node:
            """Copys this Node fully, except the frozen state

            Returns
            -------
            TreeForm.Node
                The copy
            """            
            assert all([isinstance(b, TreeForm.PsudeoNode) for b in self.branches]), "Cannot copy Nodes with true sub-Nodes."
            copyied = TreeForm.Node(self.tree, self.size, fast_init=True)
            copyied._value = self._value
            copyied.branches = [b.copy() for b in self.branches]
            return copyied
        
        @property
        def var_count(self) -> int:
            return self.counts[-1]
        
        @property
        def branch_var_counts(self) -> tuple[int, ...]:
            return tuple([b.var_count for b in self.branches])

        def __str__(self) -> str:
            return self.polish()
        
        def __repr__(self) -> str:
            return self.vampire()
        
        def polish(self, fill: Fill | None = None) -> str:
            """Get the polish expression this node represents

            Parameters
            ----------
            fill : Fill | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Polish form
            """            
            cv: OperationSpec | ConstantSpec | None = self._current_value
            polish = "_" if cv is None else cv.symbol + ''.join(b.polish(fill) for b in self.branches)

            if fill is None:
                return polish
            else:
                assert fill.size == polish.count("_")
                i = -1
                return ''.join([VARIABLE_SYMBOLS[(i := i + 1)] if c == "_" else c for c in polish])
        
        def vampire(self, fill: Fill | None = None) -> str:
            """Get the vampire expression this node represents

            Parameters
            ----------
            fill : Fill | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Vampire form
            """            
            if self._vampire_cache is None:
                cv: OperationSpec | ConstantSpec | None = self._current_value
                self._vampire_cache = "_" if cv is None else cv.vampire_symbol + "(" + ','.join(b.vampire(None) for b in self.branches) + ")"
            
            if fill is None:
                return self._vampire_cache
            else:
                assert fill.size == self._vampire_cache.count("_")
                i = -1
                return ''.join([VAMPIRE_VARIABLE_SYMBOLS[(i := i + 1)] if c == "_" else c for c in self._vampire_cache])
            
        def get_iterator(self, external_degeneracy: CountArray) -> Iterable[TreeForm.Node]:
            """Yields iterations self until self.iterate() returns False.

            Yields
            ------
            TreeForm.Node
                This node in increasingly iterated forms
            """            
            yield self
            while self.iterate(external_degeneracy):
                yield self

        def process(self, model_table: ModelTable, vampire_wrapper: VampireWrapper, remaining_file: TextIOWrapper) -> int:
            """Fully processes this Node in its current state (without iterating it), creating new countermodels using vampire as needed.
            Indeterminate expressions will be placed into the remaining file

            Parameters
            ----------
            model_table : ModelTable
                Tautology check and Countermodeling table
            vampire_wrapper : Callable[[str], bool  |  Model]
                Vampire function, returns False if no model was found, otherwise returns the model
            remaining_file : str
                File to place the indeterminate expressions (Tautological but Un-countermodeled) into

            Returns
            -------
            int
                How many unsolved expressions were added to the remaining file
            """            
            unsolved_count = 0

            var_count: int = self.counts[-1]
            full_target_evaluation: ModelArray = self.calculate(model_table.target_model, FullFill(var_count))
            full_cm_evals: dict[Model, ModelArray] = {}
            cleaver = Cleaver(var_count)
            for i, fill in enumerate(fill_iterator(var_count)):
                if cleaver[i]:
                    if not apply_fill_to_cache(full_target_evaluation, fill_dimensions(fill))[0].all():
                        #assert 
                        cleaver *= fill_downward_cleave(i, var_count)
                    for cm in model_table.counter_models:
                        if not cm in full_cm_evals.keys():
                            full_cm_evals[cm] = self.calculate(cm, FullFill(var_count))
                        if apply_fill_to_cache(full_cm_evals[cm], fill_dimensions(fill))[0].all():
                            cleaver *= fill_upward_cleave(i, var_count)
                if cleaver[i]:
                    vampire_result: bool | Model = vampire_wrapper(self.vampire(fill))
                    if vampire_result==False:
                        remaining_file.write(self.vampire(fill)+"\n")
                        unsolved_count += 1
                    else:
                        assert isinstance(vampire_result, Model), "Vampire wrapper shouldn't return True, only models or false."
                        model_table += vampire_result
                        cleaver *= fill_upward_cleave(i, var_count)
            
            return unsolved_count
    
    class PsudeoNode(Node):
        """A PsudeoNode acts like a Node but actually just indexes a known list of every possible node of that length 
        """        
        point: int
        """The current point in the list of possible Nodes that this Node is using"""
        cache: tuple[TreeForm.Node, ...]
        """'List' of possible Nodes that this Node is using"""

        def __init__(self, tree: TreeForm, size: int, external_degeneracy: CountArray | None = None) -> None:
            self.size = size
            self.tree = tree
            self.cache = tree._PSUDEO_NODE_CACHE[size]
            if external_degeneracy is None:
                self.point = 0
            else:
                self.point = -1
                self.iterate(external_degeneracy)

        def calculate(self, model: Model, fill: Fill) -> ModelArray: 
            """Calculates the Node's truth table for a particular model

            Parameters
            ----------
            model : Model
                Model to calculate for
            fill : FillArray
                Fill to calculate with (currently just generates everything)

            Returns
            -------
            ModelArray
                Result array
            """            
            return self.cache[self.point].calculate(model, fill)

        def iterate(self, external_degeneracy: CountArray) -> bool:
            """Iterates self until its non-degenerate (if possible)

            Parameters
            ----------
            external_degeneracy : CountArray
                External counts of each lexographical elements

            Returns
            -------
            bool
                True if it was able to be iterated, False if unable and had to be reset.
            """            
            while True:
                self.point += 1
                if self.point >= len(self.cache):
                    self.point = 0
                    return False
                if not self._is_degenerate(external_degeneracy):
                    return True
        
        @property
        def counts(self) -> CountArray:
            """Count of each lexographical element in this node and all branches

            Returns
            -------
            CountArray
                Counts of each lexographical element
            """            
            return self.cache[self.point].counts
                
        def copy(self) -> TreeForm.PsudeoNode:
            """Copys this Node fully

            Returns
            -------
            TreeForm.Node
                The copy
            """            
            copyied = TreeForm.PsudeoNode(self.tree, self.size)
            copyied.point = self.point
            return copyied
        
        def polish(self, fill: Fill | None = None) -> str:
            """Get the polish expression this node represents

            Parameters
            ----------
            fill : Fill | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Polish form
            """      
            return self.cache[self.point].polish(fill)
        
        def vampire(self, fill: Fill | None = None) -> str:
            """Get the vampire expression this node represents

            Parameters
            ----------
            fill : Fill | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Vampire form
            """            
            return self.cache[self.point].vampire(fill)

    def _build_psudeo_node_cache(self, size: int) -> tuple[TreeForm.Node, ...]:
        """Builds Nodes for the PSUDEO_NODE_CACHE of a particular size.
        All smaller PsudeoNodes need to already be generated

        Parameters
        ----------
        size : int
            Size to make for

        Returns
        -------
        tuple[TreeForm.Node, ...]
            Every possible Node of a particular size
        """        
        current_node: TreeForm.Node = self.Node(self, size)
        current_node.freeze()
        cache: list[TreeForm.Node] = [current_node]
        while True:
            new_node: TreeForm.Node = current_node.copy()
            current_node = new_node
            if not current_node.iterate(np.ones(len(self.LEXOGRAPHICAL_REFERENCE)+1, dtype=np.int8)):
                return tuple(cache)
            current_node.freeze()
            cache.append(current_node)

    def process_tree(self, size: int, model_table: ModelTable, vampire_wrapper: VampireWrapper, remaining_filename: str) -> int:
        """Processes this entire Tree species at a particular size into a file

        Parameters
        ----------
        size : int
            Size of tree to process at
        model_table : ModelTable
            Tautology check and Countermodeling table
        vampire_wrapper : Callable[[str], bool  |  Model]
            Vampire function, returns False if no model was found, otherwise returns the model
        remaining_filename : str
            Filename to place the indeterminate expressions (Tautological but Un-countermodeled) into

        Returns
        -------
        int
            How many unsolved expressions were added to the remaining file
        """        
        unsolved_count = 0
        default_degeneracy = np.zeros(len(self.LEXOGRAPHICAL_REFERENCE)+1, dtype=np.int8)

        i = 0
        with open(remaining_filename, 'w') as remaining_file:
            for state in self.new_node(size).get_iterator(default_degeneracy):
                i += 1
                unsolved_count += state.process(model_table, lambda vampire_form: vampire_wrapper(self._PREFIX.symbol+"("+vampire_form+")"), remaining_file)
        
        print("Processed "+str(i)+" formulas.")
        
        return unsolved_count


