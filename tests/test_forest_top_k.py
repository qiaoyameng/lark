"""Tests for lark.parsers.earley_forest.ForestTopKIterator"""

from __future__ import absolute_import

import itertools
import unittest
from collections import Counter
from itertools import islice

from lark import Lark, Tree
from lark.grammar import Rule, NonTerminal, Terminal
from lark.lexer import Token
from lark.parsers.earley_forest import (
    ForestCycleError,
    ForestTopKIterator,
    SymbolNode,
    TokenNode,
)


JOIN_GRAMMAR = r"""
start: s
s: s s s -> join3
 | s s -> join2
 | "a" -> leaf
"""

CATALAN_GRAMMAR = r"""
start: s
s: s s -> join
 | "a" -> leaf
"""

ITEM_GRAMMAR = r"""
start: item item item item item item item item
item: "x" -> c0
    | wrap -> c1
    | wrap2 -> c2
wrap: "x"
wrap2: "x"
"""

NULLABLE_CYCLE_GRAMMAR = r"""
start: s
s: s s -> join
 | t -> leaf
 | -> empty
t: "a"
"""


def parse_forest(grammar, text):
    return Lark(grammar, parser='earley', ambiguity='forest').parse(text)


def reference_derivations(node, cost_of):
    """Brute-force enumeration of all (cost, tree) derivations of an
    acyclic forest, used to cross-check the iterator exhaustively."""
    if isinstance(node, TokenNode):
        return [(0, node.token)]
    results = []
    for packed in node.children:
        options = []
        for child in (packed.left, packed.right):
            if child is not None:
                options.append(reference_derivations(child, cost_of))
        for combo in itertools.product(*options):
            cost = sum(c for c, _ in combo)
            if not node.is_intermediate:
                cost += cost_of(packed.rule)
            values = []
            for _, value in combo:
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
            if node.is_intermediate:
                results.append((cost, values))
            else:
                name = packed.rule.alias or packed.rule.origin.name
                results.append((cost, Tree(name, values)))
    return results


def count_derivations(node, memo):
    """Counts the derivations of a forest node without building them."""
    if isinstance(node, TokenNode):
        return 1
    key = id(node)
    if key not in memo:
        total = 0
        for packed in node.children:
            count = 1
            for child in (packed.left, packed.right):
                if child is not None:
                    count *= count_derivations(child, memo)
            total += count
        memo[key] = total
    return memo[key]


def forest_rules(root):
    """Collects the rules used in a forest, keyed by alias or origin name."""
    rules = {}
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, TokenNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, SymbolNode):
            for packed in node.children:
                rules.setdefault(packed.rule.alias or packed.rule.origin.name, packed.rule)
                stack.append(packed)
        else:
            stack.extend(node.children)
    return rules


def make_deep_forest(depth):
    """Builds a deep (cycle-free) forest by hand: a long chain of
    intermediate nodes, so that recursive implementations would overflow
    the Python call stack."""
    item_rule = Rule(NonTerminal('item'), [Terminal('A')])
    chain_rule = Rule(NonTerminal('start'), [NonTerminal('start'), Terminal('A')], 1)
    start_rule = Rule(NonTerminal('start'), [NonTerminal('start')], 0)

    def token_node(pos):
        return TokenNode(Token('A', 'a', start_pos=pos, end_pos=pos + 1), None)

    bottom = SymbolNode(NonTerminal('item'), 0, 1)
    bottom.add_family(item_rule.origin, item_rule, 0, None, token_node(0))
    previous = bottom
    for i in range(depth):
        node = SymbolNode((chain_rule, 1), 0, i + 2)
        node.add_family(chain_rule.origin, chain_rule, 0, previous, token_node(i + 1))
        previous = node
    root = SymbolNode(NonTerminal('start'), 0, depth + 2)
    root.add_family(start_rule.origin, start_rule, 0, previous, token_node(depth + 1))
    return root


class TestForestTopK(unittest.TestCase):
    def test_ranking_exhaustive(self):
        # Exhaustively cross-check the ranking against a brute-force
        # enumeration of the whole forest.
        forest = parse_forest(JOIN_GRAMMAR, 'aaaa')
        costs = {'join3': 1, 'join2': 3, 'leaf': 0}
        results = list(ForestTopKIterator(forest, costs))

        self.assertEqual(len(results), 10)
        cost_list = [cost for cost, _ in results]
        self.assertEqual(cost_list, sorted(cost_list))
        self.assertEqual(cost_list, [4] * 5 + [9] * 5)

        reference = reference_derivations(forest, lambda rule: costs.get(rule.alias or rule.origin.name, 0))
        self.assertEqual(
            Counter((cost, repr(tree)) for cost, tree in results),
            Counter((cost, repr(tree)) for cost, tree in reference),
        )

    def test_stable_tie_order(self):
        # Equal-cost derivations must come out in a stable, reproducible
        # order, even across independently parsed forest instances.
        parser = Lark(JOIN_GRAMMAR, parser='earley', ambiguity='forest')
        costs = {'join3': 1, 'join2': 1, 'leaf': 0}
        first = [(c, repr(t)) for c, t in ForestTopKIterator(parser.parse('aaaa'), costs)]
        second = [(c, repr(t)) for c, t in ForestTopKIterator(parser.parse('aaaa'), costs)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))  # no derivation repeated

    def test_top_k_prefix_matches_full(self):
        forest = parse_forest(ITEM_GRAMMAR, 'x' * 8)
        costs = {'c0': 0, 'c1': 1, 'c2': 2}
        top5 = [(c, repr(t)) for c, t in islice(ForestTopKIterator(forest, costs), 5)]
        full = [(c, repr(t)) for c, t in ForestTopKIterator(forest, costs)]
        self.assertEqual(len(full), 3 ** 8)
        self.assertEqual([c for c, _ in full], sorted(c for c, _ in full))
        self.assertEqual(top5, full[:5])
        self.assertEqual([c for c, _ in top5], [0, 1, 1, 1, 1])

    def test_lazy_expansion_counts(self):
        # Catalan(13) = 742900 derivations; top-5 must only expand a tiny
        # fraction of the forest.
        forest = parse_forest(CATALAN_GRAMMAR, 'a' * 14)
        self.assertEqual(count_derivations(forest, {}), 742900)

        iterator = ForestTopKIterator(forest, {'join': 1, 'leaf': 0})
        top5 = list(islice(iterator, 5))
        self.assertEqual([cost for cost, _ in top5], [13] * 5)
        self.assertLess(iterator.nodes_expanded, 1000)
        self.assertLess(iterator.candidates_generated, 3000)
        iterator.close()

    def test_early_stop_releases_candidates(self):
        forest = parse_forest(ITEM_GRAMMAR, 'x' * 8)
        iterator = ForestTopKIterator(forest, {'c0': 0, 'c1': 1, 'c2': 2})
        top2 = list(islice(iterator, 2))
        self.assertEqual(len(top2), 2)

        expanders = list(iterator._expanders.values())
        self.assertTrue(any(e.heap or e.pending for e in expanders))
        iterator.close()
        self.assertTrue(all(not e.heap and not e.pending and not e.derivations for e in expanders))
        self.assertEqual(iterator._expanders, {})
        self.assertRaises(StopIteration, next, iterator)

    def test_context_manager(self):
        forest = parse_forest(ITEM_GRAMMAR, 'x' * 8)
        with ForestTopKIterator(forest) as iterator:
            cost, tree = next(iterator)
            self.assertEqual(cost, 0)
        self.assertTrue(iterator._closed)
        self.assertEqual(iterator._expanders, {})

    def test_exhaustion_auto_closes(self):
        forest = parse_forest(CATALAN_GRAMMAR, 'aaa')
        iterator = ForestTopKIterator(forest)
        self.assertEqual(len(list(iterator)), 2)
        self.assertTrue(iterator._closed)
        self.assertRaises(StopIteration, next, iterator)

    def test_cycle_error(self):
        forest = parse_forest('start: a? | start start\n!a: "a"', 'a')
        self.assertRaises(ForestCycleError, ForestTopKIterator, forest)
        try:
            ForestTopKIterator(forest)
        except ForestCycleError as e:
            self.assertIn('cycle', str(e))

        forest = parse_forest(NULLABLE_CYCLE_GRAMMAR, 'aaa')
        self.assertRaises(ForestCycleError, ForestTopKIterator, forest)

    def test_cycle_skip(self):
        # The cycle-free derivations are still enumerable, in cost order,
        # and iteration terminates despite the infinite ambiguity.
        forest = parse_forest(NULLABLE_CYCLE_GRAMMAR, 'aaa')
        results = list(ForestTopKIterator(forest, {'join': 1, 'leaf': 0, 'empty': 5}, on_cycle='skip'))
        self.assertEqual([cost for cost, _ in results], [2, 2])
        self.assertEqual([tree.data for _, tree in results], ['start', 'start'])

        forest = parse_forest('start: a? | start start\n!a: "a"', 'a')
        results = list(ForestTopKIterator(forest, on_cycle='skip'))
        self.assertEqual(results, [(0, Tree('start', [Tree('a', ['a'])]))])

        forest = parse_forest('start: start?', '')
        results = list(ForestTopKIterator(forest, on_cycle='skip'))
        self.assertEqual(results, [(0, Tree('start', []))])

    def test_duplicate_derivations(self):
        # Distinct derivations that build equal trees are all yielded.
        grammar = r"""
        start: x "a" -> same | y "a" -> same
        x: "a" -> leaf
        y: "a" -> leaf
        """
        forest = parse_forest(grammar, 'aa')
        results = list(ForestTopKIterator(forest))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][1], Tree('same', [Tree('leaf', ['a']), 'a']))

    def test_cost_callable(self):
        forest = parse_forest(JOIN_GRAMMAR, 'aaaa')
        by_name = list(ForestTopKIterator(forest, {'join3': 1, 'join2': 3, 'leaf': 0}))
        callable_costs = lambda rule: {'join3': 1, 'join2': 3}.get(rule.alias, 0)
        by_callable = list(ForestTopKIterator(forest, callable_costs))
        self.assertEqual([c for c, _ in by_name], [c for c, _ in by_callable])
        self.assertEqual([repr(t) for _, t in by_name], [repr(t) for _, t in by_callable])

    def test_cost_rule_objects(self):
        forest = parse_forest(JOIN_GRAMMAR, 'aaaa')
        rules = forest_rules(forest)
        costs = {rules['join2']: 3, rules['join3']: 1, 'leaf': 0}
        results = list(ForestTopKIterator(forest, costs))
        self.assertEqual([c for c, _ in results], [4] * 5 + [9] * 5)

    def test_cost_validation(self):
        forest = parse_forest(JOIN_GRAMMAR, 'aaaa')
        self.assertRaises(ValueError, ForestTopKIterator, forest, {'join2': -1})
        self.assertRaises(ValueError, ForestTopKIterator, forest, {'join2': float('nan')})
        self.assertRaises(TypeError, ForestTopKIterator, forest, {'join2': '3'})
        self.assertRaises(TypeError, ForestTopKIterator, forest, {3: 1})
        self.assertRaises(TypeError, ForestTopKIterator, forest, 42)
        self.assertRaises(ValueError, ForestTopKIterator, forest, None, 'bogus')
        self.assertRaises(TypeError, ForestTopKIterator, 'not a node')
        # Costs returned by a callable are validated lazily, on iteration.
        iterator = ForestTopKIterator(forest, lambda rule: -1)
        self.assertRaises(ValueError, next, iterator)

    def test_default_costs(self):
        forest = parse_forest(CATALAN_GRAMMAR, 'aaa')
        results = list(ForestTopKIterator(forest))
        self.assertEqual([cost for cost, _ in results], [0, 0])

    def test_deep_forest(self):
        # A 5000-deep forest: recursive implementations would raise
        # RecursionError, the iterative machinery must not.
        root = make_deep_forest(5000)
        iterator = ForestTopKIterator(root, {'item': 2, 'start': 3})
        cost, tree = next(iterator)
        self.assertEqual(cost, 5)
        self.assertEqual(tree.data, 'start')
        self.assertEqual(len(tree.children), 5002)
        self.assertRaises(StopIteration, next, iterator)


if __name__ == '__main__':
    unittest.main()
