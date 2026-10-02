""""This module implements an SPPF implementation

This is used as the primary output mechanism for the Earley parser
in order to store complex ambiguities.

Full reference and more details is here:
https://web.archive.org/web/20190616123959/http://www.bramvandersanden.com/post/2014/06/shared-packed-parse-forest/
"""

from typing import Type, AbstractSet, Callable
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from heapq import heappop, heappush

from ..exceptions import GrammarError

from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree


class ForestCycleError(GrammarError):
    """Raised when a top-k derivation iterator encounters a cyclic SPPF.

    Cyclic forests can represent infinitely many derivations. Use
    ``on_cycle='skip'`` to enumerate only derivations that do not use packed
    nodes participating in a cycle.
    """


class Derivation:
    """One selected derivation of a :class:`SymbolNode`.

    The derivation is represented as a shared DAG. The same shared child
    derivation object is reused by parent derivations instead of being copied.

    Attributes:
        symbol_node: The SPPF symbol node derived.
        packed_node: The packed node selected for this symbol node.
        rule: The production applied by this packed node.
        cost: Total non-negative cost of the derivation.
        left: The binary SPPF left child (``None``, a :class:`TokenNode`, or
            another ``Derivation``).
        right: The binary SPPF right child.
    """

    __slots__ = ('symbol_node', 'packed_node', 'rule', 'cost', 'left', 'right')

    def __init__(self, symbol_node, packed_node, rule, cost, left, right):
        self.symbol_node = symbol_node
        self.packed_node = packed_node
        self.rule = rule
        self.cost = cost
        self.left = left
        self.right = right

    @property
    def children(self):
        """The logical production children, in left-to-right order.

        Intermediate SPPF nodes used by the parser's binary representation are
        flattened. Values are either :class:`TokenNode` instances or final
        ``Derivation`` instances.
        """
        return _logical_children(self)

    def to_tree(self, tree_class=Tree):
        """Materialize this shared derivation as an ordinary parse tree.

        Repeated occurrences of a shared subtree are copied in the resulting
        tree, as required by a tree representation. This does not apply token
        filtering, rule inlining, or parse-tree builder callbacks.
        """
        return _derivation_to_tree(self, tree_class)

    def __repr__(self):
        return '{}(cost={}, rule={!r})'.format(type(self).__name__, self.cost, self.rule)


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


def _logical_children(derivation):
    result = []
    stack = [derivation.right, derivation.left]
    while stack:
        value = stack.pop()
        if value is None:
            continue
        if isinstance(value, TokenNode):
            result.append(value)
        elif value.symbol_node.is_intermediate:
            stack.append(value.right)
            stack.append(value.left)
        else:
            result.append(value)
    return tuple(result)


def _derivation_to_tree(derivation, tree_class):
    outputs = []
    tasks = [('enter', derivation)]
    while tasks:
        task = tasks.pop()
        if task[0] == 'enter':
            value = task[1]
            if isinstance(value, TokenNode):
                outputs.append(value.token)
                continue

            children = _logical_children(value)
            tasks.append(('build', value, len(children)))
            for child in reversed(children):
                tasks.append(('enter', child))
        else:
            _, value, child_count = task
            children = [outputs.pop() for _ in range(child_count)][::-1]
            rule = value.rule
            outputs.append(tree_class(rule.alias or rule.origin.name, children))
    return outputs[0]


_INVALID_RANK = object()
_MISSING_RANK = object()


class _FamilyEnumerator:
    __slots__ = ('packed_node', 'family_index', 'base_cost', '__weakref__')

    def __init__(self, packed_node, family_index, base_cost):
        self.packed_node = packed_node
        self.family_index = family_index
        self.base_cost = base_cost


class _SymbolDerivationStream:
    __slots__ = ('symbol_node', 'results', 'families', '_unopened', '_pending_candidates',
                 '_candidate_heap', '_initialized', '_exhausted', 'expanded_families',
                 '__weakref__')

    def __init__(self, symbol_node, cyclic_packed_nodes):
        self.symbol_node = symbol_node
        self.results = []
        self.families = []
        self._unopened = []
        self._pending_candidates = []
        self._candidate_heap = []
        self._initialized = False
        self._exhausted = False
        self.expanded_families = 0

        for family_index, packed_node in enumerate(symbol_node.children):
            if id(packed_node) not in cyclic_packed_nodes:
                self.families.append(_FamilyEnumerator(packed_node, len(self.families), 0))


class TopKDerivations:
    """Lazily enumerate the lowest-cost derivations of an Earley SPPF.

    Derivations are ordered by total production cost. Ties are resolved by
    packed-family order, then by the left-to-right ranks of the selected child
    derivations. Each shared :class:`SymbolNode` keeps its own ranked result
    list, so a shared subtree is enumerated once and then reused by reference.

    Two derivations are distinct if they select different packed nodes at a
    symbol occurrence. Repeated occurrences of the same shared derivation in
    one production remain the same object in the DAG representation; their
    positions distinguish them, and :meth:`Derivation.to_tree` copies each
    occurrence.

    Parameters:
        root: A ``SymbolNode`` returned by ``Lark(..., ambiguity='forest')``.
        k: Maximum number of root derivations to return.
        rule_costs: A mapping from production ``Rule`` objects to non-negative
            costs, or a callable receiving a rule and returning a cost. Missing
            rules cost zero. Intermediate binary SPPF families do not add
            another cost for the same production.
        on_cycle: ``'error'`` raises :class:`ForestCycleError`; ``'skip'``
            omits packed nodes that participate in an SPPF cycle and returns
            the finite acyclic derivations.

    The iterator is also a context manager. Closing it early drops buffered
    candidate heaps; already returned derivations remain available.
    """

    def __init__(self, root, k=5, rule_costs=None, *, on_cycle='error'):
        if not isinstance(root, SymbolNode):
            raise TypeError('Expected a SymbolNode, got %r' % (root,))
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError('k must be a positive integer')
        if on_cycle not in ('error', 'skip'):
            raise ValueError("on_cycle must be either 'error' or 'skip'")

        self.root = root
        self.k = k
        self.on_cycle = on_cycle
        self._rule_costs = rule_costs
        self._streams = {}
        self._stack = []
        self._suspend_after_current = False
        self._closed = False
        self.expanded_symbol_nodes = 0
        self.expanded_packed_nodes = 0

        cyclic_packed_nodes = _find_cyclic_packed_nodes(root, self._rule_cost)
        if cyclic_packed_nodes and on_cycle == 'error':
            raise ForestCycleError(
                'The parse forest contains a cycle and therefore has infinitely '
                "many derivations. Pass on_cycle='skip' to enumerate finite, "
                'acyclic derivations instead.')
        self._cyclic_packed_nodes = cyclic_packed_nodes
        self._root_stream = self._get_stream(root)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        rank = len(self._root_stream.results)
        if rank >= self.k:
            raise StopIteration
        task = ('index', self._root_stream, rank)
        if self._stack:
            self._stack.insert(0, task)
            self._process_stack()
        else:
            self._ensure_rank(self._root_stream, rank)
        try:
            return self._root_stream.results[rank]
        except IndexError:
            raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        if hasattr(self, '_streams'):
            self.close()

    def close(self):
        """Release buffered candidate heaps used by this iterator."""
        if self._closed:
            return
        self._closed = True
        self._streams.clear()
        self._stack.clear()
        self._root_stream = None

    @property
    def expanded_nodes(self):
        """Number of symbol and packed nodes expanded by the iterator."""
        return self.expanded_symbol_nodes + self.expanded_packed_nodes

    def _get_stream(self, symbol_node):
        node_id = id(symbol_node)
        stream = self._streams.get(node_id)
        if stream is None:
            stream = _SymbolDerivationStream(symbol_node, self._cyclic_packed_nodes)
            self._streams[node_id] = stream
        return stream

    def _rule_cost(self, rule):
        costs = self._rule_costs
        if costs is None:
            cost = 0
        elif callable(costs):
            cost = costs(rule)
        else:
            cost = costs.get(rule, 0)

        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise TypeError('Production costs must be int or float numbers')
        if cost < 0 or cost != cost:
            raise ValueError('Production costs must be non-negative and finite')
        return cost

    def _initialize_stream(self, stream):
        stream._initialized = True
        self.expanded_symbol_nodes += 1
        for family in stream.families:
            packed_node = family.packed_node
            family.base_cost = (
                0 if packed_node.parent.is_intermediate
                else self._rule_cost(packed_node.rule))
        stream._unopened = sorted(
            range(len(stream.families)),
            key=lambda family_index: (
                stream.families[family_index].base_cost, family_index),
            reverse=True)

    def _process_stack(self):
        self._suspend_after_current = False
        while self._stack:
            task = self._stack.pop()
            kind = task[0]
            if kind == 'candidate':
                self._prepare_candidate(*task[1:])
            else:
                self._produce_next(*task[1:])
            if self._suspend_after_current:
                break

    def _ensure_rank(self, stream, target_rank):
        self._stack.append(('index', stream, target_rank))
        self._process_stack()

    def _rank_value(self, forest_node, rank):
        if forest_node is None:
            return None if rank == 0 else _INVALID_RANK
        if isinstance(forest_node, TokenNode):
            return forest_node if rank == 0 else _INVALID_RANK

        stream = self._get_stream(forest_node)
        if rank < len(stream.results):
            return stream.results[rank]
        if stream._exhausted:
            return _INVALID_RANK
        return _MISSING_RANK

    def _wait_for_rank(self, family, left_rank, right_rank, child, rank):
        # The candidate stays below its dependency, so it is retried after
        # the requested child rank has been produced.
        self._stack.append(('candidate', family, left_rank, right_rank))
        self._stack.append(('index', self._get_stream(child), rank))

    def _child_rank_is_available(self, child, rank):
        if child is None:
            return rank == 0
        if isinstance(child, TokenNode):
            return rank == 0
        stream = self._streams.get(id(child))
        return stream is None or rank < len(stream.results) or not stream._exhausted

    def _prepare_candidate(self, family, left_rank, right_rank):
        packed_node = family.packed_node
        if not self._child_rank_is_available(packed_node.left, left_rank):
            return
        if not self._child_rank_is_available(packed_node.right, right_rank):
            return

        left_value = self._rank_value(packed_node.left, left_rank)
        if left_value is _MISSING_RANK:
            self._wait_for_rank(family, left_rank, right_rank,
                                packed_node.left, left_rank)
            return
        right_value = self._rank_value(packed_node.right, right_rank)
        if right_value is _MISSING_RANK:
            self._wait_for_rank(family, left_rank, right_rank,
                                packed_node.right, right_rank)
            return

        left_cost = left_value.cost if isinstance(left_value, Derivation) else 0
        right_cost = right_value.cost if isinstance(right_value, Derivation) else 0
        total_cost = family.base_cost + left_cost + right_cost
        stream = self._get_stream(packed_node.parent)
        heappush(stream._candidate_heap, (
            total_cost, family.family_index, left_rank, right_rank,
            left_value, right_value))

    def _open_family(self, stream, family_index):
        stream.expanded_families += 1
        self.expanded_packed_nodes += 1
        family = stream.families[family_index]
        self._stack.append(('candidate', family, 0, 0))

    def _schedule_candidate(self, family, left_rank, right_rank):
        self._stack.append(('candidate', family, left_rank, right_rank))

    def _next_lower_bound_source(self, stream):
        source = None
        if stream._unopened:
            family_index = stream._unopened[-1]
            family = stream.families[family_index]
            source = (family.base_cost, family_index, 'unopened', family_index)
        if stream._pending_candidates:
            lower_bound, _, sequence, family_index, _, _ = stream._pending_candidates[0]
            pending_key = (lower_bound, 0, 'pending', sequence, family_index)
            if source is None or pending_key < source:
                source = pending_key
        return source

    def _produce_next(self, stream, target_rank):
        if target_rank < len(stream.results):
            return
        if not stream._initialized:
            self._initialize_stream(stream)

        while True:
            source = self._next_lower_bound_source(stream)
            heap = stream._candidate_heap
            if heap and (source is None or heap[0][:2] < source[:2]):
                break

            if source is None:
                stream._exhausted = True
                return

            if source[2] == 'unopened':
                family_index = stream._unopened.pop()
                self._stack.append(('index', stream, target_rank))
                self._open_family(stream, family_index)
            else:
                _, family_index, sequence, family, left_rank, right_rank = heappop(
                    stream._pending_candidates)
                self._stack.append(('index', stream, target_rank))
                self._schedule_candidate(family, left_rank, right_rank)
            return

        total_cost, family_index, left_rank, right_rank, left_value, right_value = heappop(heap)
        family = stream.families[family_index]
        derivation = Derivation(
            stream.symbol_node, family.packed_node, family.packed_node.rule,
            total_cost, left_value, right_value)
        stream.results.append(derivation)

        # Each successor's total cost cannot fall below the selected total,
        # because all production costs are non-negative. Keep these as lazy
        # lower-bound candidates; child ranks are materialized only if selected.
        heappush(stream._pending_candidates,
                 (total_cost, family_index, 0, family, left_rank, right_rank + 1))
        if right_rank == 0:
            heappush(stream._pending_candidates,
                     (total_cost, family_index, 1, family, left_rank + 1, 0))

        if target_rank < len(stream.results):
            if stream is self._root_stream:
                self._suspend_after_current = True
            return

        self._stack.append(('index', stream, target_rank))


def top_k_derivations(root, k=5, rule_costs=None, *, on_cycle='error'):
    """Create a :class:`TopKDerivations` iterator."""
    return TopKDerivations(root, k=k, rule_costs=rule_costs, on_cycle=on_cycle)


def _find_cyclic_packed_nodes(root, rule_cost=None):
    """Return ids of packed nodes belonging to any reachable SPPF cycle."""
    nodes = {id(root): root}
    reverse_adjacency = {id(root): []}
    visit_stack = [(root, 0)]
    finish_order = []

    while visit_stack:
        node, edge_index = visit_stack[-1]
        if isinstance(node, TokenNode):
            visit_stack.pop()
            finish_order.append(id(node))
            continue

        children = node.children
        if edge_index < len(children):
            visit_stack[-1] = (node, edge_index + 1)
            child = children[edge_index]
            if (rule_cost is not None and isinstance(child, PackedNode)
                    and not child.parent.is_intermediate):
                rule_cost(child.rule)
            if isinstance(child, TokenNode):
                continue
            child_id = id(child)
            if child_id not in nodes:
                nodes[child_id] = child
                reverse_adjacency[child_id] = [id(node)]
                visit_stack.append((child, 0))
            else:
                reverse_adjacency[child_id].append(id(node))
        else:
            visit_stack.pop()
            finish_order.append(id(node))

    cyclic_packed_nodes = set()
    visited = set()
    for node_id in reversed(finish_order):
        if node_id in visited:
            continue
        component = []
        stack = [node_id]
        visited.add(node_id)
        while stack:
            current_id = stack.pop()
            component.append(current_id)
            for parent_id in reverse_adjacency[current_id]:
                if parent_id not in visited:
                    visited.add(parent_id)
                    stack.append(parent_id)
        if len(component) > 1:
            for member_id in component:
                member = nodes[member_id]
                if isinstance(member, PackedNode):
                    cyclic_packed_nodes.add(member_id)

    return frozenset(cyclic_packed_nodes)
