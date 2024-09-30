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
    OPERATION_REFERENCE: tuple[OperationSpec, ...]
    """References for all Operations in a particular order (will define iteration order, not nessisarily efficiency, and definantly not the total possibilities)"""
    CONSTANT_REFERENCE: tuple[ConstantSpec, ...]
    """References for all Constants in a particular order (will define iteration order, not nessisarily efficiency, and definantly not the total possibilities)"""
    PREFIX: OperationSpec
    """Prefix operator"""
    _MAXIMUM_FULLY_CACHED_NODE_SIZE: int
    """Maximum size Node to make PsudeoNodes for"""
    _PSUDEO_NODE_CACHE: dict[int, tuple[TreeForm.Node, ...]]
    """Cache of items used for PsudeoNode iteration"""
    _RESONABLE_MAXIMUM_FULL_MODELING_SIZE: int
    """Above what size (order) model should full modeling be skipped and just individual formulas processed brute force"""

    def __init__(self, lexographical_reference: tuple[OperationSpec | ConstantSpec, ...], prefix: OperationSpec, MAXIMUM_FULLY_CACHED_NODE_SIZE: int, RESONABLE_MAXIMUM_FULL_MODELING_SIZE: int = 3) -> None:
        assert prefix.arity==1, "Non-Unitary prefixes not implemented at this time"
        self.OPERATION_REFERENCE = tuple([ref for ref in lexographical_reference if isinstance(ref, OperationSpec)])
        self.CONSTANT_REFERENCE = tuple([ref for ref in lexographical_reference if isinstance(ref, ConstantSpec)])
        self.PREFIX = prefix
        self._MAXIMUM_FULLY_CACHED_NODE_SIZE = MAXIMUM_FULLY_CACHED_NODE_SIZE
        self._PSUDEO_NODE_CACHE = {}
        self._RESONABLE_MAXIMUM_FULL_MODELING_SIZE = RESONABLE_MAXIMUM_FULL_MODELING_SIZE

        i: int = 1
        while i <= self._MAXIMUM_FULLY_CACHED_NODE_SIZE:
            self._PSUDEO_NODE_CACHE[i] = self._build_psudeo_node_cache(i)
            i = self._next_valid_node_size(i)

    @functools.cache
    def _next_valid_node_size(self, size: int) -> int:
        """Computes next allowable node size

        Parameters
        ----------
        size : int
            Previous Node Size

        Returns
        -------
        int
            Next allowable node size

        Raises
        ------
        RuntimeError
            Infinite loop
        """        
        for i in range(100): #anti-infinite-loop
            size += 1
            if self._valid_node_size(size):
                return size

        raise RuntimeError
        
    @staticmethod
    def _node_size_combos(remaining_slots: int, remaining_size: int) -> Iterable[tuple[int, ...]]:
        """Helper function for valid node size computation

        Parameters
        ----------
        remaining_slots : int
            Remaining number of node size groupings
        remaining_size : int
            Remaining number of subnodes to place into groupings

        Yields
        ------
        Iterator[Iterable[tuple[int, ...]]]
            All possible subnode groupings
        """        
        if remaining_slots==1:
            yield (remaining_size, )
        else:
            for i in range(1, remaining_size + 1 - remaining_slots + 1):
                for comb in TreeForm._node_size_combos(remaining_slots - 1, remaining_size - i):
                    yield (i,) + comb
    
    @functools.cache
    def _valid_node_size(self, size: int) -> bool:
        """Checks if a node size is valid

        Parameters
        ----------
        size : int
            Size to check

        Returns
        -------
        bool
            If size is a valid node size
        """        
        if size==1:
            return True
        
        for op in self.OPERATION_REFERENCE:
            for comb in self._node_size_combos(op.arity, size - 1):
                valid = True
                for s in comb:
                    if not self._valid_node_size(s):
                        valid = False
                        break
                if valid:
                    return True

        return False

    def new_node(self, size: int, external_degeneracy: CountArray | None = None) -> Node:
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
            return self.PsudeoNode(self, size, external_degeneracy)
        return self.Node(self, size, external_degeneracy)

    @staticmethod
    def _valid_symbol(value: OperationSpec | None, size: int) -> bool:
        """Determines if a value is valid for the node of a specific size.
        For size = 1, only variables (which are constant holders) are allowed
        For size > 1, only functions with arity < size - 1 are allowed

        Parameters
        ----------
        value : OperationSpec | None
            An operator
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
                self._initilize(external_degeneracy)

        def _reset_caches(self) -> None:
            """Clears the caches
            """            
            self._cache = {}
            self._count_cache = None
            self._vampire_cache = None
        
        def freeze(self) -> None:
            """Freezes this node, will throw an error if it is iterated
            """            
            self._frozen = True

        @property
        def _current_value(self) -> OperationSpec | None:
            """The current Operation, Constant, or Variable (None)
            """            
            return self.tree.OPERATION_REFERENCE[self._value] if self._value != len(self.tree.OPERATION_REFERENCE) else None

        def _initilize(self, external_degeneracy: CountArray | None = None) -> None:
            """Initalizes the Node to the first valid Lexographical value.

            Parameters
            ----------
            external_degeneracy : CountArray | None, optional
                The external counts of each possible node value to avoid degenerate trees, by default None
            """                      
            self._value = -1
            self._reset_caches()
            assert self._iterate_value(external_degeneracy), "Malformation"

        def _iterate_value(self, external_degeneracy: CountArray | None = None) -> bool:
            """Iterates the value to next valid symbol

            Parameters
            ----------
            external_degeneracy : CountArray | None, optional
                The external counts of each possible node value to avoid degenerate trees, by default None

            Returns
            -------
            bool
                If there is a valid next symbol
            """            
            while True:
                self._value += 1
                if self._value > len(self.tree.OPERATION_REFERENCE):
                    return False
                if TreeForm._valid_symbol(self._current_value if self._value < len(self.tree.OPERATION_REFERENCE) else None, self.size):
                    if self.size == 1:
                        return True
                    self._initialize_at_value(external_degeneracy)
                    return True
                
        def _initialize_at_value(self, external_degeneracy: CountArray | None = None) -> None:
            """Initalizes the Node at a particular value

            Parameters
            ----------
            external_degeneracy : CountArray | None, optional
                The external counts of each possible node value to avoid degenerate trees, by default None
            """            
            self._reset_caches()
            op: OperationSpec | ConstantSpec | None = self._current_value
            assert isinstance(op, OperationSpec)
            full_node: TreeForm.Node = self.tree.new_node(self.size - op.arity, external_degeneracy) # type: ignore
            self.branches = [self.tree.new_node(1) for _ in range(op.arity - 1)]
            self.branches.insert(0, full_node)
            assert self.size - 1 - sum(b.size for b in self.branches) == 0
            self._reset_caches()

        def calculate(self, model: Model, fill: FillPointer) -> ModelArray:
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
                    val: OperationSpec | None = self._current_value if self._value < len(self.tree.OPERATION_REFERENCE) else None
                    if val is None:
                        assert fill.size==1, "Malformation"
                        self._cache[model] = np.arange(model.order)
                    else:
                        raise AssertionError("Malformation")
                assert self._cache[model].ndim == fill.size, fill.size
                return self._cache[model]
            else:
                if not model in self._cache.keys():
                    op: OperationSpec | None = self._current_value
                    assert isinstance(op, OperationSpec)

                    fill_splits: tuple[tuple[FillPointer, DimensionalReference], ...] = split_fill(fill, self.branch_var_counts)
                    branch_calcs: tuple[tuple[ModelArray, DimensionalReference], ...] = tuple([(b.calculate(model, sub_fill), tuple([injection[i] for i in get_fill(sub_fill)])) for b, (sub_fill, injection) in zip(self.branches, fill_splits)])

                    self._cache[model] = model.calculate(op, fill.size, branch_calcs)

                    assert self._cache[model].ndim == fill.size, str(fill.size - self._cache[model].ndim)+"\n"+str(get_fill(fill))
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
                    #new_size = self.tree._next_valid_node_size(self.branches[i+1].size)
                    #diff = self.branches[i+1].size - new_size
                    self.branches[i+1] = self.tree.new_node(self.tree._next_valid_node_size(self.branches[i+1].size))
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
            """            
            if self._count_cache is None:
                #typing hates this completely valid code
                if len(self.branches)==0:
                    self._count_cache = np.zeros(len(self.tree.OPERATION_REFERENCE)+1, dtype=np.int8) # type: ignore
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
            """Number of variable slots in this node
            """            
            return self.counts[-1]
        
        @property
        def branch_var_counts(self) -> tuple[int, ...]:
            """Variable counts of each branch
            """            
            return tuple([b.var_count for b in self.branches])

        def __str__(self) -> str:
            return self.polish()
        
        def __repr__(self) -> str:
            return self.vampire()
        
        def polish(self, fillin: Sequence[Any] | None = None) -> str:
            """The polish expression this node represents

            Parameters
            ----------
            fill_dims : Sequence[Any] | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Polish form
            """            
            cv: OperationSpec | ConstantSpec | None = self._current_value
            polish = "_" if cv is None else cv.symbol + ''.join(b.polish(fillin) for b in self.branches)

            if fillin is None:
                return polish
            else:
                assert len(fillin) == polish.count("_")
                i = -1
                temp: list[Any] = []
                for c in polish:
                    if c == "_":
                        if fillin[i] >= 0:
                            temp.append(VARIABLE_SYMBOLS[fillin[i]])
                            i += 1
                        else:
                            temp.append(self.tree.CONSTANT_REFERENCE[- fillin[i] - 1].symbol)
                    else:
                        temp.append(c)

                return ''.join(temp)
        
        def vampire(self, fillin: Sequence[Any] | None = None) -> str:
            """The vampire expression this node represents

            Parameters
            ----------
            fill_dims : Sequence[Any] | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Vampire form
            """            
            if self._vampire_cache is None:
                cv: OperationSpec | ConstantSpec | None = self._current_value
                self._vampire_cache = "_" if cv is None else cv.vampire_symbol + "(" + ','.join(b.vampire(None) for b in self.branches) + ")"
            
            if fillin is None:
                return self._vampire_cache
            else:
                assert len(fillin) == self._vampire_cache.count("_")
                i = 0
                temp: list[Any] = []
                for c in self._vampire_cache:
                    if c == "_":
                        if fillin[i] >= 0:
                            temp.append(VAMPIRE_VARIABLE_SYMBOLS[fillin[i]])
                            i += 1
                        else:
                            temp.append(self.tree.CONSTANT_REFERENCE[- fillin[i] - 1].vampire_symbol)
                    else:
                        temp.append(c)

                return ''.join(temp)
            
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

        #@profile # type: ignore
        def process(self, model_table: ModelTable, vampire_wrapper: VampireWrapper, remaining_file: TextIOWrapper) -> tuple[int, int]:
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
            tuple[int, int]
                How many unsolved expressions were added to the remaining file
                How many expressions were processed
            """            

            var_count: int = self.counts[-1]
            cleaver = CleavingMatrix(var_count, len(self.tree.CONSTANT_REFERENCE))

            self._process_cleaver_helper(cleaver, model_table.target_model, "Downward")
            
            for cm in model_table.counter_models:
                if cleaver.empty:
                    break
                self._process_cleaver_helper(cleaver, cm, "Upward")

            #small_counter_models, big_counter_models = model_table.counter_models_size_split(self.tree._RESONABLE_MAXIMUM_FULL_MODELING_SIZE)
            #for cm in small_counter_models:
            #    if cleaver.empty:
            #        break
            #    self._process_cleaver_helper(cleaver, cm, "Upward")
            #
            #if not cleaver.empty:
            #    for k in cleaver.cleaves.keys(): 
            #        var_count = k.count(0)
            #        cleave = CleavingMatrix.base_cleaver(var_count)
            #        fill_iter = reversed(list(enumerate(fill_iterator(var_count))))
            #        for i, fill in fill_iter:
            #            if cleaver.constant_binding_empty(k):
            #                break
            #            if cleaver.cleaves[k][i]:
            #                fill_dims: DimensionalReference = get_fill(fill)
            #                j = -1
            #                fillin: list[Any] = [fill_dims[(j := j + 1)] if k[i]==0 else self.tree.CONSTANT_REFERENCE[- k[i] - 1].vampire_symbol for i in range(cleaver.full_size)]
            #                vamp: str = self.vampire(fillin)
            #                for cm in big_counter_models:
            #                    if cm("t("+vamp+")"):
            #                        cleave *= fill_downward_cleave(i, var_count).astype(np.bool_)

            for k in cleaver.cleaves.keys():
                for i, fill in enumerate(fill_iterator(k.count(0))):
                    fill_dims: DimensionalReference = get_fill(fill)
                    j = -1
                    fillin: list[Any] = [fill_dims[(j := j + 1)] if k[i]==0 else -k[i] for i in range(var_count)]

                    if cleaver.cleaves[k][i]:
                        vamp: str = self.vampire(fillin)

                        if VERIFY_ALL_FORMULAS:
                            assert not any(cm("t("+vamp+")") for cm in model_table.counter_models)
                            assert model_table.target_model("t("+vamp+")"), vamp+"\n"+str(i)+"\n"+str(fillin)

                        vampire_result: bool | Model = vampire_wrapper(vamp)
                        if vampire_result==False:
                            remaining_file.write(self.vampire(fillin)+"\n")
                        else:
                            assert isinstance(vampire_result, Model), "Vampire wrapper shouldn't return True, only models or false."
                            model_table += vampire_result
                            self._process_cleaver_helper(cleaver, vampire_result, "Upward")
                            if VERIFY_ALL_FORMULAS:
                                assert not cleaver.cleaves[k][i]
                    elif VERIFY_ALL_FORMULAS:
                        vamp: str = self.vampire(fillin)
                        assert not model_table.target_model("t("+vamp+")") or any(cm("t("+vamp+")") for cm in model_table.counter_models), vamp+"\n"+str(i)+"\n"+str(fillin)

            return sum(c.sum() for c in cleaver.cleaves.values()), sum(c.shape[0] for c in cleaver.cleaves.values())

        #@profile # type: ignore
        def _process_cleaver_helper(self, cleaver: CleavingMatrix, model: Model, cleave_direction: Literal["Upward"] | Literal["Downward"]) -> None:
            """Helper function to calculate cleaves from a model

            Parameters
            ----------
            cleaver : CleavingMatrix
                Cleaver to adjust
            model : Model
                Model being used
            cleave_direction : Literal[&quot;Upward&quot;] | Literal[&quot;Downward&quot;]
                Upward for counter models
                Downward for target models
            """            
            if model.order > self.tree._RESONABLE_MAXIMUM_FULL_MODELING_SIZE:
                for k in cleaver.cleaves.keys():
                    var_count = k.count(0)
                    cleave: CleavingArray = CleavingMatrix.base_cleaver(var_count) if cleave_direction == "Downward" else np.logical_not(CleavingMatrix.base_cleaver(var_count))
                    fill_iter: Iterable[tuple[int, FillPointer]] = enumerate(fill_iterator(var_count)) if cleave_direction == "Downward" else reversed(list(enumerate(fill_iterator(var_count))))
                    for i, fill in fill_iter:
                        assert fill.point==i #TODO
                        assert fill.size==var_count
                        fill_dims: DimensionalReference = get_fill(fill)
                        j = -1
                        fillin: list[Any] = [fill_dims[(j := j + 1)] if k[i]==0 else self.tree.CONSTANT_REFERENCE[- k[i] - 1].vampire_symbol for i in range(cleaver.full_size)]
                        if (cleave[i] if cleave_direction == "Downward" else not cleave[i]) and cleaver.cleaves[k][i]:
                            vamp: str = self.vampire(fillin)
                            if model("t("+vamp+")"):
                                if cleave_direction == "Downward":
                                    cleave *= fill_downward_cleave(fill).astype(np.bool_)
                                else: #cleave_direction == "Upward":
                                    cleave = np.logical_or(cleave, fill_downward_cleave(fill).astype(np.bool_))

                    cleaver.cleaves[k] *= cleave

            else:
                full_model_evaluation = model.apply_function(self.tree.PREFIX, self.calculate(model, full_fill(cleaver.full_size)))
                assert not isinstance(full_model_evaluation, int)
                assert full_model_evaluation.dtype == np.bool_
                cleaver *= fill_result_disassembly_application(full_model_evaluation, [model.constant_definitions[cons] for cons in self.tree.CONSTANT_REFERENCE], cleave_direction)
    
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

        def calculate(self, model: Model, fill: FillPointer) -> ModelArray: 
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
        
        def polish(self, fillin: Sequence[Any] | None = None) -> str:
            """The polish expression this node represents

            Parameters
            ----------
            fillin : Sequence[Any] | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Polish form
            """      
            return self.cache[self.point].polish(fillin)
        
        def vampire(self, fillin: Sequence[Any] | None = None) -> str:
            """The vampire expression this node represents

            Parameters
            ----------
            fillin : Sequence[Any] | None, optional
                Fill to use, otherwise provides the standard '_'s

            Returns
            -------
            str
                Vampire form
            """            
            return self.cache[self.point].vampire(fillin)

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
        default_degeneracy: CountArray = np.ones(len(self.OPERATION_REFERENCE)+1, dtype=np.int8)
        current_node: TreeForm.Node = self.Node(self, size)
        current_node.freeze()
        cache: list[TreeForm.Node] = [current_node]
        while True:
            new_node: TreeForm.Node = current_node.copy()
            current_node = new_node
            if not current_node.iterate(default_degeneracy):
                return tuple(cache)
            current_node.freeze()
            cache.append(current_node)

    def process_tree(self, size: int, model_table: ModelTable, vampire_wrapper: VampireWrapper, remaining_filename: str) -> tuple[int, int]:
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
        tuple[int, int]
            How many unsolved expressions were added to the remaining file
            How many expressions were processed
        """        
        unsolved_count = 0
        total_processed = 0
        default_degeneracy = np.zeros(len(self.OPERATION_REFERENCE)+1, dtype=np.int8)

        i = 0
        with open(remaining_filename, 'w') as remaining_file:
            for state in self.new_node(size, default_degeneracy).get_iterator(default_degeneracy):
                i += 1
                new_unsolved, new_processed = state.process(model_table, lambda vampire_form: vampire_wrapper(self.PREFIX.vampire_symbol+"("+vampire_form+")"), remaining_file)
                unsolved_count += new_unsolved
                total_processed += new_processed
        
        return unsolved_count, total_processed
    
    @functools.cache
    def _formula_count_helper(self, size: int) -> tuple[tuple[tuple[OperationSpec, ...], int], ...]:
        """Helper function for determining the number of possible formulas.
        Returns a mapping from operations used to formula counts, 
        must be in the form of a tuple rather than a dict for caching.

        Parameters
        ----------
        size : int
            Formula size to target

        Returns
        -------
        tuple[tuple[tuple[OperationSpec, ...], int], ...]
            Mapping from operations used to formula counts
        """        
        if size==1:
            return (((), 1),)
        counts: dict[tuple[OperationSpec, ...], int] = {}
        for op in self.OPERATION_REFERENCE:
            for comb in self._node_size_combos(op.arity, size - 1):
                valid = True
                for s in comb:
                    if not self._valid_node_size(s):
                        valid = False
                        break
                if valid:
                    comb_dicts: list[dict[tuple[OperationSpec, ...], int]] = [dict(self._formula_count_helper(s)) for s in comb]
                    for comb_set in itertools.product(*[list(d.items()) for d in comb_dicts]):
                        new_set = set(itertools.chain.from_iterable(set(s) for s, c in comb_set))
                        new_set.add(op)
                        new_s: tuple[OperationSpec, ...] = tuple([op for op in self.OPERATION_REFERENCE if op in new_set])
                        new_c: int = int(np.prod(np.array([c for s, c in comb_set])))

                        if new_s in counts.keys():
                            counts[new_s] += new_c
                        else:
                            counts[new_s] = new_c
        
        return tuple(counts.items())
    
    def formula_count(self, size: int, allow_degenerate: bool = False) -> int:
        """Calculates the number of formulas there should be of a particular size, including different variable setups

        Parameters
        ----------
        size : int
            Formula size target
        allow_degenerate : bool, optional
            If degenerate formulas (not containing all lexographical elements) should be counted, by default False

        Returns
        -------
        int
            Total Count

        Raises
        ------
        RuntimeError
            No nondegenerate formulas
        """         
        res: tuple[tuple[tuple[OperationSpec, ...], int], ...] = self._formula_count_helper(size)
        if not allow_degenerate:
            for s, c in res:
                if s == self.OPERATION_REFERENCE:
                    return c
                
            print(res)
            raise RuntimeError
        else:
            return sum([c for s, c in res])

    def verify_formulas(self, size: int) -> None:
        """Verifies the formulas generated by a tree.
        Could be incorrect in 3 ways:
        Repeated formulas,
        Invalid formulas,
        Not enough formulas

        If none of those 3 issues exists its assumed that all formulas are being properly generated

        Parameters
        ----------
        size : int
            Size to verify for

        Raises
        ------
        AssertionError
            If formulas generated are in some way invalid
        """        
        model: Model = Model(ModelSpec((self.PREFIX,) + self.OPERATION_REFERENCE, self.CONSTANT_REFERENCE))
        
        target_count: int = self.formula_count(size)

        default_degeneracy = np.zeros(len(self.OPERATION_REFERENCE)+1, dtype=np.int8)
        count = 0
        expressions: set[str] = set()
        for state in self.new_node(size, default_degeneracy).get_iterator(default_degeneracy):
            count += 1
            vamp = self.PREFIX.vampire_symbol+"("+state.vampire()+")"
            expressions.add(vamp)
            try:
                model.compile_expression(vamp)
            except:
                raise AssertionError("Unverifiable form: "+vamp)
            print(vamp)#str(count)+":"+
        
        assert count == target_count, "Counts: "+str(count)+" "+str(target_count)







