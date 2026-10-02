""""This module implements an SPPF implementation

This is used as the primary output mechanism for the Earley parser
in order to store complex ambiguities.

Full reference and more details is here:
https://web.archive.org/web/20190616123959/http://www.bramvandersanden.com/post/2014/06/shared-packed-parse-forest/
"""

import heapq
import math
from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial

from ..exceptions import LarkError
from ..grammar import Rule
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree

class ForestNode:
    pass

class SymbolNode(ForestNode):
    """
    A Symbol Node represents a symbol (or Intermediate LR0).

    Symbol nodes are keyed by the symbol (s). For intermediate nodes
    s will be an LR0, stored as a tuple of (rule, ptr). For completed symbol
    nodes, s will be a string representing the non-terminal origin (i.e.
    the left hand side of the rule).

    The children of a Symbol or Intermediate Node will always be Packed Nodes;
    with each Packed Node child representing a single derivation of a production.

    Hence a Symbol Node with a single child is unambiguous.

    Parameters:
        s: A Symbol, or a tuple of (rule, ptr) for an intermediate node.
        start: For dynamic lexers, the index of the start of the substring matched by this symbol (inclusive).
        end: For dynamic lexers, the index of the end of the substring matched by this symbol (exclusive).

    Properties:
        is_intermediate: True if this node is an intermediate node.
        priority: The priority of the node's symbol.
    """
    Set: Type[AbstractSet] = set   # Overridden by StableSymbolNode
    __slots__ = ('s', 'start', 'end', '_children', 'paths', 'paths_loaded', 'priority', 'is_intermediate')
    def __init__(self, s, start, end):
        self.s = s
        self.start = start
        self.end = end
        self._children = self.Set()
        self.paths = self.Set()
        self.paths_loaded = False

        ### We use inf here as it can be safely negated without resorting to conditionals,
        #   unlike None or float('NaN'), and sorts appropriately.
        self.priority = float('-inf')
        self.is_intermediate = isinstance(s, tuple)

    def add_family(self, lr0, rule, start, left, right):
        self._children.add(PackedNode(self, lr0, rule, start, left, right))

    def add_path(self, transitive, node):
        self.paths.add((transitive, node))

    def load_paths(self):
        for transitive, node in self.paths:
            if transitive.next_titem is not None:
                vn = type(self)(transitive.next_titem.s, transitive.next_titem.start, self.end)
                vn.add_path(transitive.next_titem, node)
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, vn)
            else:
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, node)
        self.paths_loaded = True

    @property
    def is_ambiguous(self):
        """Returns True if this node is ambiguous."""
        return len(self.children) > 1

    @property
    def children(self):
        """Returns a list of this node's children sorted from greatest to
        least priority."""
        if not self.paths_loaded:
            self.load_paths()
        return sorted(self._children, key=attrgetter('sort_key'))

    def __iter__(self):
        return iter(self._children)

    def __repr__(self):
        if self.is_intermediate:
            rule = self.s[0]
            ptr = self.s[1]
            before = ( expansion.name for expansion in rule.expansion[:ptr] )
            after = ( expansion.name for expansion in rule.expansion[ptr:] )
            symbol = "{} ::= {}* {}".format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return "({}, {}, {}, {})".format(symbol, self.start, self.end, self.priority)

class StableSymbolNode(SymbolNode):
    "A version of SymbolNode that uses OrderedSet for output stability"
    Set = OrderedSet

class PackedNode(ForestNode):
    """
    A Packed Node represents a single derivation in a symbol node.

    Parameters:
        rule: The rule associated with this node.
        parent: The parent of this node.
        left: The left child of this node. ``None`` if one does not exist.
        right: The right child of this node. ``None`` if one does not exist.
        priority: The priority of this node.
    """
    __slots__ = ('parent', 's', 'rule', 'start', 'left', 'right', 'priority', '_hash')
    def __init__(self, parent, s, rule, start, left, right):
        self.parent = parent
        self.s = s
        self.start = start
        self.rule = rule
        self.left = left
        self.right = right
        self.priority = float('-inf')
        self._hash = hash((self.left, self.right))

    @property
    def is_empty(self):
        return self.left is None and self.right is None

    @property
    def sort_key(self):
        """
        Used to sort PackedNode children of SymbolNodes.
        A SymbolNode has multiple PackedNodes if it matched
        ambiguously. Hence, we use the sort order to identify
        the order in which ambiguous children should be considered.
        """
        return self.is_empty, -self.priority, self.rule.order

    @property
    def children(self):
        """Returns a list of this node's children."""
        return [x for x in [self.left, self.right] if x is not None]

    def __iter__(self):
        yield self.left
        yield self.right

    def __eq__(self, other):
        if not isinstance(other, PackedNode):
            return False
        return self is other or (self.left == other.left and self.right == other.right)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        if isinstance(self.s, tuple):
            rule = self.s[0]
            ptr = self.s[1]
            before = ( expansion.name for expansion in rule.expansion[:ptr] )
            after = ( expansion.name for expansion in rule.expansion[ptr:] )
            symbol = "{} ::= {}* {}".format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return "({}, {}, {}, {})".format(symbol, self.start, self.priority, self.rule.order)

class TokenNode(ForestNode):
    """
    A Token Node represents a matched terminal and is always a leaf node.

    Parameters:
        token: The Token associated with this node.
        term: The TerminalDef matched by the token.
        priority: The priority of this node.
    """
    __slots__ = ('token', 'term', 'priority', '_hash')
    def __init__(self, token, term, priority=None):
        self.token = token
        self.term = term
        if priority is not None:
            self.priority = priority
        else:
            self.priority = term.priority if term is not None else 0
        self._hash = hash(token)

    def __eq__(self, other):
        if not isinstance(other, TokenNode):
            return False
        return self is other or (self.token == other.token)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return repr(self.token)

class ForestVisitor:
    """
    An abstract base class for building forest visitors.

    This class performs a controllable depth-first walk of an SPPF.
    The visitor will not enter cycles and will backtrack if one is encountered.
    Subclasses are notified of cycles through the ``on_cycle`` method.

    Behavior for visit events is defined by overriding the
    ``visit*node*`` functions.

    The walk is controlled by the return values of the ``visit*node_in``
    methods. Returning a node(s) will schedule them to be visited. The visitor
    will begin to backtrack if no nodes are returned.

    Parameters:
        single_visit: If ``True``, non-Token nodes will only be visited once.
    """

    def __init__(self, single_visit=False):
        self.single_visit = single_visit

    def visit_token_node(self, node):
        """Called when a ``Token`` is visited. ``Token`` nodes are always leaves."""
        pass

    def visit_symbol_node_in(self, node):
        """Called when a symbol node is visited. Nodes that are returned
        will be scheduled to be visited. If ``visit_intermediate_node_in``
        is not implemented, this function will be called for intermediate
        nodes as well."""
        pass

    def visit_symbol_node_out(self, node):
        """Called after all nodes returned from a corresponding ``visit_symbol_node_in``
        call have been visited. If ``visit_intermediate_node_out``
        is not implemented, this function will be called for intermediate
        nodes as well."""
        pass

    def visit_packed_node_in(self, node):
        """Called when a packed node is visited. Nodes that are returned
        will be scheduled to be visited. """
        pass

    def visit_packed_node_out(self, node):
        """Called after all nodes returned from a corresponding ``visit_packed_node_in``
        call have been visited."""
        pass

    def on_cycle(self, node, path):
        """Called when a cycle is encountered.

        Parameters:
            node: The node that causes a cycle.
            path: The list of nodes being visited: nodes that have been
                entered but not exited. The first element is the root in a forest
                visit, and the last element is the node visited most recently.
                ``path`` should be treated as read-only.
        """
        pass

    def get_cycle_in_path(self, node, path):
        """A utility function for use in ``on_cycle`` to obtain a slice of
        ``path`` that only contains the nodes that make up the cycle."""
        index = len(path) - 1
        while id(path[index]) != id(node):
            index -= 1
        return path[index:]

    def visit(self, root):
        # Visiting is a list of IDs of all symbol/intermediate nodes currently in
        # the stack. It serves two purposes: to detect when we 'recurse' in and out
        # of a symbol/intermediate so that we can process both up and down. Also,
        # since the SPPF can have cycles it allows us to detect if we're trying
        # to recurse into a node that's already on the stack (infinite recursion).
        visiting = set()

        # set of all nodes that have been visited
        visited = set()

        # a list of nodes that are currently being visited
        # used for the `on_cycle` callback
        path = []

        # We do not use recursion here to walk the Forest due to the limited
        # stack size in python. Therefore input_stack is essentially our stack.
        input_stack = deque([root])

        # It is much faster to cache these as locals since they are called
        # many times in large parses.
        vpno = getattr(self, 'visit_packed_node_out')
        vpni = getattr(self, 'visit_packed_node_in')
        vsno = getattr(self, 'visit_symbol_node_out')
        vsni = getattr(self, 'visit_symbol_node_in')
        vino = getattr(self, 'visit_intermediate_node_out', vsno)
        vini = getattr(self, 'visit_intermediate_node_in', vsni)
        vtn = getattr(self, 'visit_token_node')
        oc = getattr(self, 'on_cycle')

        while input_stack:
            current = next(reversed(input_stack))
            try:
                next_node = next(current)
            except StopIteration:
                input_stack.pop()
                continue
            except TypeError:
                ### If the current object is not an iterator, pass through to Token/SymbolNode
                pass
            else:
                if next_node is None:
                    continue

                if id(next_node) in visiting:
                    oc(next_node, path)
                    continue

                input_stack.append(next_node)
                continue

            if isinstance(current, TokenNode):
                vtn(current.token)
                input_stack.pop()
                continue

            current_id = id(current)
            if current_id in visiting:
                if isinstance(current, PackedNode):
                    vpno(current)
                elif current.is_intermediate:
                    vino(current)
                else:
                    vsno(current)
                input_stack.pop()
                path.pop()
                visiting.remove(current_id)
                visited.add(current_id)
            elif self.single_visit and current_id in visited:
                input_stack.pop()
            else:
                visiting.add(current_id)
                path.append(current)
                if isinstance(current, PackedNode):
                    next_node = vpni(current)
                elif current.is_intermediate:
                    next_node = vini(current)
                else:
                    next_node = vsni(current)
                if next_node is None:
                    continue

                if not isinstance(next_node, ForestNode):
                    next_node = iter(next_node)
                elif id(next_node) in visiting:
                    oc(next_node, path)
                    continue

                input_stack.append(next_node)

class ForestTransformer(ForestVisitor):
    """The base class for a bottom-up forest transformation. Most users will
    want to use ``TreeForestTransformer`` instead as it has a friendlier
    interface and covers most use cases.

    Transformations are applied via inheritance and overriding of the
    ``transform*node`` methods.

    ``transform_token_node`` receives a ``Token`` as an argument.
    All other methods receive the node that is being transformed and
    a list of the results of the transformations of that node's children.
    The return value of these methods are the resulting transformations.

    If ``Discard`` is raised in a node's transformation, no data from that node
    will be passed to its parent's transformation.
    """

    def __init__(self):
        super(ForestTransformer, self).__init__()
        # results of transformations
        self.data = dict()
        # used to track parent nodes
        self.node_stack = deque()

    def transform(self, root):
        """Perform a transformation on an SPPF."""
        self.node_stack.append('result')
        self.data['result'] = []
        self.visit(root)
        assert len(self.data['result']) <= 1
        if self.data['result']:
            return self.data['result'][0]

    def transform_symbol_node(self, node, data):
        """Transform a symbol node."""
        return node

    def transform_intermediate_node(self, node, data):
        """Transform an intermediate node."""
        return node

    def transform_packed_node(self, node, data):
        """Transform a packed node."""
        return node

    def transform_token_node(self, node):
        """Transform a ``Token``."""
        return node

    def visit_symbol_node_in(self, node):
        self.node_stack.append(id(node))
        self.data[id(node)] = []
        return node.children

    def visit_packed_node_in(self, node):
        self.node_stack.append(id(node))
        self.data[id(node)] = []
        return node.children

    def visit_token_node(self, node):
        transformed = self.transform_token_node(node)
        if transformed is not Discard:
            self.data[self.node_stack[-1]].append(transformed)

    def _visit_node_out_helper(self, node, method):
        self.node_stack.pop()
        transformed = method(node, self.data[id(node)])
        if transformed is not Discard:
            self.data[self.node_stack[-1]].append(transformed)
        del self.data[id(node)]

    def visit_symbol_node_out(self, node):
        self._visit_node_out_helper(node, self.transform_symbol_node)

    def visit_intermediate_node_out(self, node):
        self._visit_node_out_helper(node, self.transform_intermediate_node)

    def visit_packed_node_out(self, node):
        self._visit_node_out_helper(node, self.transform_packed_node)


class ForestSumVisitor(ForestVisitor):
    """
    A visitor for prioritizing ambiguous parts of the Forest.

    This visitor is used when support for explicit priorities on
    rules is requested (whether normal, or invert). It walks the
    forest (or subsets thereof) and cascades properties upwards
    from the leaves.

    It would be ideal to do this during parsing, however this would
    require processing each Earley item multiple times. That's
    a big performance drawback; so running a forest walk is the
    lesser of two evils: there can be significantly more Earley
    items created during parsing than there are SPPF nodes in the
    final tree.
    """
    def __init__(self):
        super(ForestSumVisitor, self).__init__(single_visit=True)

    def visit_packed_node_in(self, node):
        yield node.left
        yield node.right

    def visit_symbol_node_in(self, node):
        return iter(node.children)

    def visit_packed_node_out(self, node):
        priority = node.rule.options.priority if not node.parent.is_intermediate and node.rule.options.priority else 0
        priority += getattr(node.right, 'priority', 0)
        priority += getattr(node.left, 'priority', 0)
        node.priority = priority

    def visit_symbol_node_out(self, node):
        node.priority = max(child.priority for child in node.children)

class ForestCycleError(LarkError):
    """Raised when a parse forest contains a cycle (i.e. infinite ambiguity)
    and the requested operation cannot produce a finite, controlled result."""

def _node_tiebreak_key(node):
    """A deterministic, content-based key used to order forest nodes.

    Unlike identity-based ordering, this key is stable across runs, which
    keeps the tie-breaking of ``ForestTopKIterator`` reproducible.
    """
    if node is None:
        return (0, -1, -1, '')
    if isinstance(node, TokenNode):
        token = node.token
        start = token.start_pos if token.start_pos is not None else -1
        end = token.end_pos if token.end_pos is not None else -1
        return (2, start, end, str(token))
    return (1, node.start, node.end, str(node.s))

def _packed_tiebreak_key(packed):
    return (packed.sort_key, _node_tiebreak_key(packed.left), _node_tiebreak_key(packed.right))

def _collect_symbol_graph(root):
    """Iteratively collects all symbol nodes reachable from ``root``.

    Returns ``(symbols, packed_map)`` where ``packed_map`` maps ``id(symbol)``
    to a list of ``(packed_node, [symbol children])`` entries.
    """
    symbols = []
    packed_map = {}
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        symbols.append(node)
        entries = []
        for packed in node.children:
            child_symbols = []
            for child in (packed.left, packed.right):
                if isinstance(child, SymbolNode):
                    child_symbols.append(child)
                    stack.append(child)
            entries.append((packed, child_symbols))
        packed_map[id(node)] = entries
    return symbols, packed_map

def _analyze_cycles(root):
    """Detects cycles (infinite ambiguity) in the forest rooted at ``root``.

    Computes the strongly connected components of the symbol-node graph
    (using an iterative Tarjan traversal, so deep forests are safe).

    Returns ``(example, blocked)`` where ``example`` is a ``SymbolNode``
    participating in a cycle (or ``None`` if the forest is acyclic), and
    ``blocked`` is a set of ``id()`` of packed nodes whose use could
    close a cycle: packed nodes of a *non-intermediate* symbol node with a
    child in the same cyclic component. (Packed nodes of intermediate nodes
    are kept, because they also serve legitimate acyclic derivations.)
    Every cycle in the forest passes through at least one blocked packed
    node, so removing them makes the forest a DAG; only cycle-free
    derivations remain.
    """
    symbols, packed_map = _collect_symbol_graph(root)
    adjacency = {}
    for node in symbols:
        adjacency[id(node)] = [child for _packed, children in packed_map[id(node)] for child in children]

    index_of = {}
    lowlink = {}
    on_stack = set()
    stack = []
    sccs = []
    counter = 0
    for start in symbols:
        if id(start) in index_of:
            continue
        index_of[id(start)] = lowlink[id(start)] = counter
        counter += 1
        stack.append(start)
        on_stack.add(id(start))
        work = [(start, iter(adjacency[id(start)]))]
        while work:
            node, children_it = work[-1]
            descended = False
            for succ in children_it:
                succ_id = id(succ)
                if succ_id not in index_of:
                    index_of[succ_id] = lowlink[succ_id] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ_id)
                    work.append((succ, iter(adjacency[succ_id])))
                    descended = True
                    break
                elif succ_id in on_stack and index_of[succ_id] < lowlink[id(node)]:
                    lowlink[id(node)] = index_of[succ_id]
            if descended:
                continue
            work.pop()
            node_id = id(node)
            if work and lowlink[node_id] < lowlink[id(work[-1][0])]:
                lowlink[id(work[-1][0])] = lowlink[node_id]
            if lowlink[node_id] == index_of[node_id]:
                scc = []
                while True:
                    member = stack.pop()
                    on_stack.discard(id(member))
                    scc.append(member)
                    if member is node:
                        break
                sccs.append(scc)

    cyclic_scc = {}
    example = None
    for scc_index, scc in enumerate(sccs):
        if len(scc) > 1:
            cyclic = True
        else:
            node = scc[0]
            cyclic = any(child is node for child in adjacency[id(node)])
        if cyclic:
            if example is None:
                example = scc[0]
            for node in scc:
                cyclic_scc[id(node)] = scc_index

    blocked = set()
    if cyclic_scc:
        for node in symbols:
            if node.is_intermediate:
                continue
            scc_index = cyclic_scc.get(id(node))
            if scc_index is None:
                continue
            for packed, children in packed_map[id(node)]:
                if any(cyclic_scc.get(id(child)) == scc_index for child in children):
                    blocked.add(id(packed))
    return example, blocked

def _validate_cost(value, rule):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Cost for rule %r must be an int or float, got %r" % (rule, value))
    if math.isnan(value):
        raise ValueError("Cost for rule %r must not be NaN" % (rule,))
    if value < 0:
        raise ValueError("Cost for rule %r must be non-negative, got %r" % (rule, value))
    return value

class _Derivation:
    """A single enumerated derivation of a symbol node: which packed node
    was chosen, and which ranked sub-derivations of its children."""
    __slots__ = ('cost', 'packed', 'left_rank', 'right_rank')

    def __init__(self, cost, packed, left_rank, right_rank):
        self.cost = cost
        self.packed = packed
        self.left_rank = left_rank
        self.right_rank = right_rank

class _SymbolNodeExpander:
    """Lazily enumerates the derivations of a single symbol node, cheapest
    first, using a best-first heap over (packed node, left rank, right rank)
    candidates. Shared symbol nodes get exactly one expander, so shared
    subtrees are never re-expanded.
    """
    __slots__ = ('node', 'derivations', 'heap', 'pending', 'seen', 'exhausted')

    def __init__(self, node, packed_nodes):
        self.node = node
        self.derivations = []
        self.heap = []
        self.pending = deque()
        self.seen = set()
        self.exhausted = False
        for packed in packed_nodes:
            key = (id(packed), 0, 0)
            self.seen.add(key)
            self.pending.append((packed, 0, 0))

    def release(self):
        self.derivations = []
        self.heap = []
        self.pending = deque()
        self.seen = set()

class ForestTopKIterator:
    """Iterates over the derivations of an Earley parse forest (SPPF),
    cheapest first, so that callers can take the top-k derivations without
    materializing the whole (possibly exponential) ambiguity forest.

    The caller assigns a non-negative cost to each production; the cost of a
    derivation is the sum of the costs of the productions it applies.
    Derivations are yielded as ``(cost, tree)`` pairs in non-decreasing
    order of total cost.

    Sharing:
        Each symbol node of the forest is expanded lazily and at most once;
        its ranked derivations are memoized and reused by every parent, so
        shared subtrees are never fully re-expanded. The number of
        derivations and candidate expansions materialized so far is exposed
        via the ``nodes_expanded`` and ``candidates_generated`` attributes.

    Tie-breaking (stability):
        Ties are broken deterministically. At each symbol node, alternative
        derivations (packed nodes) are considered in the forest's canonical
        order: empty derivations first, then by descending priority, then by
        grammar rule order, then by span and symbol name. Combinations of
        sub-derivations are explored in lexicographic order of their ranks.
        Iterating the same forest twice with the same costs always yields
        the same sequence.

    Duplicates:
        A derivation is identified by its sequence of choices (which packed
        node is used at each symbol node), not by the tree it produces.
        Distinct derivations that build equal trees are all yielded.

    Cycles:
        If the forest contains a cycle (the grammar is infinitely ambiguous
        for the input), the default ``on_cycle='error'`` raises
        ``ForestCycleError`` when the iterator is created. With
        ``on_cycle='skip'``, packed nodes that could close a cycle (a
        production whose left-hand side can derive itself through one of
        the production's children) are excluded up front, reducing the
        forest to a DAG. Iteration then always terminates and yields only
        cycle-free derivations; note that this may also exclude a few
        technically acyclic derivations that share a cyclic component.

    Early termination:
        Callers that stop before exhaustion should call ``close()`` (or use
        the iterator as a context manager) to release the temporary
        candidate heaps and memoized derivations. Exhausting the iterator
        closes it automatically.

    Parameters:
        root: The root ``SymbolNode`` of a parse forest, as returned by
            parsing with ``ambiguity='forest'``.
        rule_costs: Optional. Either a callable ``rule -> non-negative
            number``, or a mapping keyed by ``Rule`` objects or rule names
            (strings, matched against the rule's alias or origin name).
            Productions without an explicit cost default to 0.
        on_cycle: ``'error'`` (default) or ``'skip'``.
        tree_class: The class used to build the resulting trees.

    Example:
        >>> from lark import Lark
        >>> from lark.parsers.earley_forest import ForestTopKIterator
        >>> parser = Lark('start: "a" "a"', parser='earley', ambiguity='forest')
        >>> forest = parser.parse('aa')
        >>> with ForestTopKIterator(forest) as top_k:
        ...     for cost, tree in top_k:
        ...         print(cost, tree.pretty())
    """

    def __init__(self, root, rule_costs=None, on_cycle='error', tree_class=Tree):
        self._closed = True
        self._expanders = {}
        if not isinstance(root, SymbolNode):
            raise TypeError("ForestTopKIterator expects the root SymbolNode of a parse forest, got %r" % (root,))
        if on_cycle not in ('error', 'skip'):
            raise ValueError("on_cycle must be 'error' or 'skip', got %r" % (on_cycle,))
        self._root = root
        self._tree_class = tree_class
        self._rule_cost = self._compile_costs(rule_costs)
        self._seq = 0
        self._yielded = 0
        self.nodes_expanded = 0
        self.candidates_generated = 0
        example, self._blocked = _analyze_cycles(root)
        if example is not None and on_cycle == 'error':
            raise ForestCycleError(
                "The parse forest contains a cycle (infinite ambiguity) involving node %s. "
                "Top-k iteration over an infinite number of derivations is not supported; "
                "pass on_cycle='skip' to enumerate only the cycle-free derivations." % (example,))
        self._closed = False

    @staticmethod
    def _compile_costs(rule_costs):
        if rule_costs is None:
            return lambda rule: 0
        if callable(rule_costs):
            def call_cost(rule):
                return _validate_cost(rule_costs(rule), rule)
            return call_cost
        if not hasattr(rule_costs, 'items'):
            raise TypeError("rule_costs must be a callable or a mapping, got %r" % (rule_costs,))
        by_rule = {}
        by_name = {}
        for key, value in rule_costs.items():
            value = _validate_cost(value, key)
            if isinstance(key, Rule):
                by_rule[key] = value
            elif isinstance(key, str):
                by_name[key] = value
            else:
                raise TypeError("rule_costs keys must be Rule instances or rule names (strings), got %r" % (key,))
        def lookup_cost(rule):
            if rule in by_rule:
                return by_rule[rule]
            name = rule.alias if rule.alias is not None else rule.origin.name
            if name in by_name:
                return by_name[name]
            return by_name.get(rule.origin.name, 0)
        return lookup_cost

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        expander = self._expander_for(self._root)
        if not self._ensure(expander, self._yielded):
            self.close()
            raise StopIteration
        derivation = expander.derivations[self._yielded]
        self._yielded += 1
        return derivation.cost, self._build_tree(self._root, derivation)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Releases all temporary candidates and memoized derivations.

        Safe to call multiple times; iteration cannot continue afterwards.
        """
        if self._closed:
            return
        self._closed = True
        for expander in self._expanders.values():
            expander.release()
        self._expanders.clear()

    def _expander_for(self, node):
        expander = self._expanders.get(id(node))
        if expander is None:
            packed_nodes = [p for p in node.children if id(p) not in self._blocked]
            packed_nodes.sort(key=_packed_tiebreak_key)
            expander = _SymbolNodeExpander(node, packed_nodes)
            self._expanders[id(node)] = expander
        return expander

    def _enqueue_candidate(self, expander, packed, left_rank, right_rank):
        if left_rank > 0 and packed.left is None:
            return
        if right_rank > 0 and not isinstance(packed.right, SymbolNode):
            return
        key = (id(packed), left_rank, right_rank)
        if key in expander.seen:
            return
        expander.seen.add(key)
        expander.pending.append((packed, left_rank, right_rank))

    def _candidate_cost(self, expander, packed, left_rank, right_rank):
        """Returns ``(cost, blocker)``: the candidate's total cost, or
        ``(None, (expander, rank))`` naming a child sub-derivation that must
        be enumerated first. ``(None, None)`` means the candidate is invalid
        (a child ran out of derivations)."""
        cost = 0
        if not expander.node.is_intermediate:
            cost += self._rule_cost(packed.rule)
        if packed.left is not None:
            left_expander = self._expander_for(packed.left)
            if left_rank >= len(left_expander.derivations):
                if left_expander.exhausted:
                    return None, None
                return None, (left_expander, left_rank)
            cost += left_expander.derivations[left_rank].cost
        if isinstance(packed.right, SymbolNode):
            right_expander = self._expander_for(packed.right)
            if right_rank >= len(right_expander.derivations):
                if right_expander.exhausted:
                    return None, None
                return None, (right_expander, right_rank)
            cost += right_expander.derivations[right_rank].cost
        return cost, None

    def _ensure(self, expander, rank):
        """Makes sure ``expander.derivations[rank]`` exists (returns True),
        or that the expander is exhausted (returns False).

        Uses an explicit work stack instead of recursion, so arbitrarily
        deep forests cannot overflow the Python call stack. The forest is
        guaranteed cycle-free at this point (cycles raise or are skipped in
        ``__init__``), so the work stack always drains.
        """
        stack = [(expander, rank)]
        while stack:
            current, needed = stack[-1]
            if needed < len(current.derivations) or current.exhausted:
                stack.pop()
                continue
            if current.pending:
                packed, left_rank, right_rank = current.pending[0]
                cost, blocker = self._candidate_cost(current, packed, left_rank, right_rank)
                if blocker is not None:
                    stack.append(blocker)
                    continue
                current.pending.popleft()
                if cost is not None:
                    self._seq += 1
                    heapq.heappush(current.heap, (cost, self._seq, packed, left_rank, right_rank))
                    self.candidates_generated += 1
                continue
            if current.heap:
                cost, _seq, packed, left_rank, right_rank = heapq.heappop(current.heap)
                current.derivations.append(_Derivation(cost, packed, left_rank, right_rank))
                self.nodes_expanded += 1
                self._enqueue_candidate(current, packed, left_rank + 1, right_rank)
                self._enqueue_candidate(current, packed, left_rank, right_rank + 1)
                continue
            current.exhausted = True
        return rank < len(expander.derivations)

    def _build_tree(self, node, derivation):
        """Materializes a derivation as a tree, iteratively (deep
        derivations of flat trees must not overflow the call stack).
        Intermediate nodes splice their children into the parent, mirroring
        ``ForestToParseTree``."""
        def make_frame(frame_node, frame_derivation):
            packed = frame_derivation.packed
            todo = []
            if packed.left is not None:
                todo.append((packed.left, frame_derivation.left_rank))
            if packed.right is not None:
                todo.append((packed.right, frame_derivation.right_rank))
            return [frame_node, packed, [], todo, 0]

        stack = [make_frame(node, derivation)]
        while stack:
            frame = stack[-1]
            frame_node, frame_packed, items, todo, index = frame
            if index < len(todo):
                frame[4] += 1
                child, child_rank = todo[index]
                if isinstance(child, TokenNode):
                    items.append(child.token)
                else:
                    child_derivation = self._expanders[id(child)].derivations[child_rank]
                    stack.append(make_frame(child, child_derivation))
                continue
            stack.pop()
            if frame_node.is_intermediate:
                result = items
            else:
                rule = frame_packed.rule
                result = self._tree_class(rule.alias or rule.origin.name, items)
            if not stack:
                return result
            parent_items = stack[-1][2]
            if frame_node.is_intermediate:
                parent_items.extend(result)
            else:
                parent_items.append(result)


class PackedData():
    """Used in transformationss of packed nodes to distinguish the data
    that comes from the left child and the right child.
    """

    class _NoData():
        pass

    NO_DATA = _NoData()

    def __init__(self, node, data):
        self.left = self.NO_DATA
        self.right = self.NO_DATA
        if data:
            if node.left is not None:
                self.left = data[0]
                if len(data) > 1:
                    self.right = data[1]
            else:
                self.right = data[0]

class ForestToParseTree(ForestTransformer):
    """Used by the earley parser when ambiguity equals 'resolve' or
    'explicit'. Transforms an SPPF into an (ambiguous) parse tree.

    Parameters:
        tree_class: The tree class to use for construction
        callbacks: A dictionary of rules to functions that output a tree
        prioritizer: A ``ForestVisitor`` that manipulates the priorities of ForestNodes
        resolve_ambiguity: If True, ambiguities will be resolved based on
                        priorities. Otherwise, `_ambig` nodes will be in the resulting tree.
        use_cache: If True, the results of packed node transformations will be cached.
    """

    def __init__(self, tree_class=Tree, callbacks=dict(), prioritizer=ForestSumVisitor(), resolve_ambiguity=True, use_cache=True):
        super(ForestToParseTree, self).__init__()
        self.tree_class = tree_class
        self.callbacks = callbacks
        self.prioritizer = prioritizer
        self.resolve_ambiguity = resolve_ambiguity
        self._use_cache = use_cache
        self._cache = {}
        self._on_cycle_retreat = False
        self._cycle_node = None
        self._successful_visits = set()

    def visit(self, root):
        if self.prioritizer:
            self.prioritizer.visit(root)
        super(ForestToParseTree, self).visit(root)
        self._cache = {}

    def on_cycle(self, node, path):
        logger.debug("Cycle encountered in the SPPF at node: %s. "
                "As infinite ambiguities cannot be represented in a tree, "
                "this family of derivations will be discarded.", node)
        self._cycle_node = node
        self._on_cycle_retreat = True

    def _check_cycle(self, node):
        if self._on_cycle_retreat:
            if id(node) == id(self._cycle_node) or id(node) in self._successful_visits:
                self._cycle_node = None
                self._on_cycle_retreat = False
            else:
                return Discard

    def _collapse_ambig(self, children):
        new_children = []
        for child in children:
            if hasattr(child, 'data') and child.data == '_ambig':
                new_children += child.children
            else:
                new_children.append(child)
        return new_children

    def _call_rule_func(self, node, data):
        # called when transforming children of symbol nodes
        # data is a list of trees or tokens that correspond to the
        # symbol's rule expansion
        return self.callbacks[node.rule](data)

    def _call_ambig_func(self, node, data):
        # called when transforming a symbol node
        # data is a list of trees where each tree's data is
        # equal to the name of the symbol or one of its aliases.
        if len(data) > 1:
            return self.tree_class('_ambig', data)
        elif data:
            return data[0]
        return Discard

    def transform_symbol_node(self, node, data):
        if id(node) not in self._successful_visits:
            return Discard
        r = self._check_cycle(node)
        if r is Discard:
            return r
        self._successful_visits.remove(id(node))
        data = self._collapse_ambig(data)
        return self._call_ambig_func(node, data)

    def transform_intermediate_node(self, node, data):
        if id(node) not in self._successful_visits:
            return Discard
        r = self._check_cycle(node)
        if r is Discard:
            return r
        self._successful_visits.remove(id(node))
        if len(data) > 1:
            children = [self.tree_class('_inter', c) for c in data]
            return self.tree_class('_iambig', children)
        return data[0]

    def transform_packed_node(self, node, data):
        r = self._check_cycle(node)
        if r is Discard:
            return r
        if self.resolve_ambiguity and id(node.parent) in self._successful_visits:
            return Discard
        if self._use_cache and id(node) in self._cache:
            return self._cache[id(node)]
        children = []
        assert len(data) <= 2
        data = PackedData(node, data)
        if data.left is not PackedData.NO_DATA:
            if node.left.is_intermediate and isinstance(data.left, list):
                children += data.left
            else:
                children.append(data.left)
        if data.right is not PackedData.NO_DATA:
            children.append(data.right)
        transformed = children if node.parent.is_intermediate else self._call_rule_func(node, children)
        if self._use_cache:
            self._cache[id(node)] = transformed
        return transformed

    def visit_symbol_node_in(self, node):
        super(ForestToParseTree, self).visit_symbol_node_in(node)
        if self._on_cycle_retreat:
            return
        return node.children

    def visit_packed_node_in(self, node):
        self._on_cycle_retreat = False
        to_visit = super(ForestToParseTree, self).visit_packed_node_in(node)
        if not self.resolve_ambiguity or id(node.parent) not in self._successful_visits:
            if not self._use_cache or id(node) not in self._cache:
                return to_visit

    def visit_packed_node_out(self, node):
        super(ForestToParseTree, self).visit_packed_node_out(node)
        if not self._on_cycle_retreat:
            self._successful_visits.add(id(node.parent))

def handles_ambiguity(func):
    """Decorator for methods of subclasses of ``TreeForestTransformer``.
    Denotes that the method should receive a list of transformed derivations."""
    func.handles_ambiguity = True
    return func

class TreeForestTransformer(ForestToParseTree):
    """A ``ForestTransformer`` with a tree ``Transformer``-like interface.
    By default, it will construct a tree.

    Methods provided via inheritance are called based on the rule/symbol
    names of nodes in the forest.

    Methods that act on rules will receive a list of the results of the
    transformations of the rule's children. By default, trees and tokens.

    Methods that act on tokens will receive a token.

    Alternatively, methods that act on rules may be annotated with
    ``handles_ambiguity``. In this case, the function will receive a list
    of all the transformations of all the derivations of the rule.
    By default, a list of trees where each tree.data is equal to the
    rule name or one of its aliases.

    Non-tree transformations are made possible by override of
    ``__default__``, ``__default_token__``, and ``__default_ambig__``.

    Note:
        Tree shaping features such as inlined rules and token filtering are
        not built into the transformation. Positions are also not propagated.

    Parameters:
        tree_class: The tree class to use for construction
        prioritizer: A ``ForestVisitor`` that manipulates the priorities of nodes in the SPPF.
        resolve_ambiguity: If True, ambiguities will be resolved based on priorities.
        use_cache (bool): If True, caches the results of some transformations,
                          potentially improving performance when ``resolve_ambiguity==False``.
                          Only use if you know what you are doing: i.e. All transformation
                          functions are pure and referentially transparent.
    """

    def __init__(self, tree_class=Tree, prioritizer=ForestSumVisitor(), resolve_ambiguity=True, use_cache=False):
        super(TreeForestTransformer, self).__init__(tree_class, dict(), prioritizer, resolve_ambiguity, use_cache)

    def __default__(self, name, data):
        """Default operation on tree (for override).

        Returns a tree with name with data as children.
        """
        return self.tree_class(name, data)

    def __default_ambig__(self, name, data):
        """Default operation on ambiguous rule (for override).

        Wraps data in an '_ambig_' node if it contains more than
        one element.
        """
        if len(data) > 1:
            return self.tree_class('_ambig', data)
        elif data:
            return data[0]
        return Discard

    def __default_token__(self, node):
        """Default operation on ``Token`` (for override).

        Returns ``node``.
        """
        return node

    def transform_token_node(self, node):
        return getattr(self, node.type, self.__default_token__)(node)

    def _call_rule_func(self, node, data):
        name = node.rule.alias or node.rule.options.template_source or node.rule.origin.name
        user_func = getattr(self, name, self.__default__)
        if user_func == self.__default__ or hasattr(user_func, 'handles_ambiguity'):
            user_func = partial(self.__default__, name)
        if not self.resolve_ambiguity:
            wrapper = partial(AmbiguousIntermediateExpander, self.tree_class)
            user_func = wrapper(user_func)
        return user_func(data)

    def _call_ambig_func(self, node, data):
        name = node.s.name
        user_func = getattr(self, name, self.__default_ambig__)
        if user_func == self.__default_ambig__ or not hasattr(user_func, 'handles_ambiguity'):
            user_func = partial(self.__default_ambig__, name)
        return user_func(data)

class ForestToPyDotVisitor(ForestVisitor):
    """
    A Forest visitor which writes the SPPF to a PNG.

    The SPPF can get really large, really quickly because
    of the amount of meta-data it stores, so this is probably
    only useful for trivial trees and learning how the SPPF
    is structured.
    """
    def __init__(self, rankdir="TB"):
        super(ForestToPyDotVisitor, self).__init__(single_visit=True)
        self.pydot = import_module('pydot')
        self.graph = self.pydot.Dot(graph_type='digraph', rankdir=rankdir)

    def visit(self, root, filename):
        super(ForestToPyDotVisitor, self).visit(root)
        try:
            self.graph.write_png(filename)
        except FileNotFoundError as e:
            logger.error("Could not write png: ", e)

    def visit_token_node(self, node):
        graph_node_id = str(id(node))
        graph_node_label = "\"{}\"".format(node.value.replace('"', '\\"'))
        graph_node_color = 0x808080
        graph_node_style = "\"filled,rounded\""
        graph_node_shape = "diamond"
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor="#{:06x}".format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)

    def visit_packed_node_in(self, node):
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 0x808080
        graph_node_style = "filled"
        graph_node_shape = "diamond"
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor="#{:06x}".format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        yield node.left
        yield node.right

    def visit_packed_node_out(self, node):
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in [node.left, node.right]:
            if child is not None:
                child_graph_node_id = str(id(child.token if isinstance(child, TokenNode) else child))
                child_graph_node = self.graph.get_node(child_graph_node_id)[0]
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))
            else:
                #### Try and be above the Python object ID range; probably impl. specific, but maybe this is okay.
                child_graph_node_id = str(randint(100000000000000000000000000000,123456789012345678901234567890))
                child_graph_node_style = "invis"
                child_graph_node = self.pydot.Node(child_graph_node_id, style=child_graph_node_style, label="None")
                child_edge_style = "invis"
                self.graph.add_node(child_graph_node)
                self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node, style=child_edge_style))

    def visit_symbol_node_in(self, node):
        graph_node_id = str(id(node))
        graph_node_label = repr(node)
        graph_node_color = 0x808080
        graph_node_style = "\"filled\""
        if node.is_intermediate:
            graph_node_shape = "ellipse"
        else:
            graph_node_shape = "rectangle"
        graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor="#{:06x}".format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
        self.graph.add_node(graph_node)
        return iter(node.children)

    def visit_symbol_node_out(self, node):
        graph_node_id = str(id(node))
        graph_node = self.graph.get_node(graph_node_id)[0]
        for child in node.children:
            child_graph_node_id = str(id(child))
            child_graph_node = self.graph.get_node(child_graph_node_id)[0]
            self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))
