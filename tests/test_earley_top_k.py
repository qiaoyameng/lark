import gc
import math
import unittest
import weakref

from lark import Lark, Tree
from lark.parsers.earley_forest import (
    Derivation,
    ForestCycleError,
    TokenNode,
    TopKDerivations,
    top_k_derivations,
)


def parse_forest(grammar, text="", lexer="dynamic"):
    parser = Lark(grammar, parser="earley", ambiguity="forest", lexer=lexer)
    return parser.parse(text)


def selected_terminals(derivation):
    terminals = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, TokenNode):
            terminals.append(node.term.name)
        elif isinstance(node, Derivation):
            walk(node.left)
            walk(node.right)

    walk(derivation)
    return tuple(terminals)


class TestTopKDerivations(unittest.TestCase):
    def test_small_grammar_exhaustive_ranking(self):
        grammar = """
        start: a a
        a: X | Y
        X: "x"
        Y: /x/
        """
        forest = parse_forest(grammar, "xx")

        def rule_cost(rule):
            if rule.origin.name == "a":
                return 1 if rule.expansion[0].name == "X" else 3
            return 0

        derivations = list(top_k_derivations(forest, 4, rule_cost))

        self.assertEqual([d.cost for d in derivations], [2, 4, 4, 6])
        self.assertEqual(
            [selected_terminals(d) for d in derivations],
            [("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")])

    def test_mapping_costs_and_tie_stability(self):
        grammar = """
        start: a a
        a: X | Y
        X: "x"
        Y: /x/
        """
        parser = Lark(grammar, parser="earley", ambiguity="forest", lexer="dynamic")
        forest = parser.parse("xx")
        cost_by_rule = {parser.rules[1]: 1, parser.rules[2]: 3}

        first = list(top_k_derivations(forest, 4, cost_by_rule))
        second = list(top_k_derivations(forest, 4, cost_by_rule))

        self.assertEqual(
            [selected_terminals(d) for d in first],
            [("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")])
        self.assertEqual(
            [(d.packed_node, d.left.packed_node, d.right.packed_node) for d in first],
            [(d.packed_node, d.left.packed_node, d.right.packed_node) for d in second]
        )

    def test_three_by_three_forest_exhaustive_ranking(self):
        grammar = """
        start: a a
        a: X | Y | Z
        X: "x"
        Y: /x/
        Z: /[x]/
        """
        forest = parse_forest(grammar, "xx")

        def rule_cost(rule):
            if rule.origin.name != "a":
                return 0
            return {"X": 3, "Y": 1, "Z": 2}[rule.expansion[0].name]

        derivations = list(top_k_derivations(forest, 9, rule_cost))

        self.assertEqual(
            [(d.cost, selected_terminals(d)) for d in derivations],
            [
                (2, ("Y", "Y")),
                (3, ("Y", "Z")),
                (3, ("Z", "Y")),
                (4, ("Y", "X")),
                (4, ("Z", "Z")),
                (4, ("X", "Y")),
                (5, ("Z", "X")),
                (5, ("X", "Z")),
                (6, ("X", "X")),
            ])

    def test_shared_subtree_is_reused_but_tree_occurrences_are_copied(self):
        forest = parse_forest("""
        start: e e
        e: | "x"
        """, "", lexer="basic")

        derivation = next(top_k_derivations(forest, 1))
        first, second = derivation.children

        self.assertIs(first.symbol_node, second.symbol_node)
        self.assertIs(first, second)
        tree = derivation.to_tree()
        self.assertIsNot(tree.children[0], tree.children[1])
        self.assertEqual(tree, Tree("start", [Tree("e", []), Tree("e", [])]))

    def test_large_combinatorial_forest_expands_lazily(self):
        grammar = """
        start: choice+
        choice: A | B
        A: "a"
        B: /a/
        """
        forest = parse_forest(grammar, "a" * 20)
        iterator = top_k_derivations(forest, 5, lambda rule: 0)
        derivations = list(iterator)

        self.assertEqual(len(derivations), 5)
        self.assertEqual(len({selected_terminals(d) for d in derivations}), 5)
        self.assertEqual(math.comb(20, 10), 184756)
        self.assertEqual(2 ** 20, 1048576)
        self.assertEqual(iterator.expanded_symbol_nodes, 60)
        self.assertEqual(iterator.expanded_packed_nodes, 80)
        self.assertEqual(iterator.expanded_nodes, 140)

    def test_cycle_raises_by_default_and_can_be_skipped(self):
        forest = parse_forest("start: start?", "", lexer="basic")

        with self.assertRaises(ForestCycleError):
            list(top_k_derivations(forest, 5))

        derivations = list(top_k_derivations(forest, 5, on_cycle="skip"))
        self.assertEqual([d.cost for d in derivations], [0])
        self.assertEqual(derivations[0].to_tree(), Tree("start", []))

    def test_negative_cost_is_rejected(self):
        forest = parse_forest('start: "a"', "a", lexer="basic")

        with self.assertRaises(ValueError):
            top_k_derivations(forest, 1, lambda rule: -1)

    def test_tree_materialization_does_not_recurse(self):
        parser = Lark(
            'start: start "a" | "a"',
            parser="earley", ambiguity="forest", lexer="basic")
        forest = parser.parse("a" * 1000)

        tree = next(top_k_derivations(forest, 1)).to_tree()

        self.assertEqual(sum(1 for _ in tree.find_data("start")), 1000)

    def test_invalid_k_is_rejected(self):
        forest = parse_forest('start: "a"', "a", lexer="basic")

        with self.assertRaises(ValueError):
            TopKDerivations(forest, 0)
        with self.assertRaises(ValueError):
            TopKDerivations(forest, True)

    def test_closing_early_releases_buffered_candidates(self):
        grammar = """
        start: choice+
        choice: A | B
        A: "a"
        B: /a/
        """
        forest = parse_forest(grammar, "a" * 20)
        iterator = TopKDerivations(forest, 5, lambda rule: 0)
        next(iterator)

        streams = list(iterator._streams.values())
        refs = [
            weakref.ref(stream)
            for stream in streams
        ] + [
            weakref.ref(family)
            for stream in streams
            for family in stream.families
        ]
        self.assertTrue(refs)
        self.assertTrue(any(stream._pending_candidates for stream in streams))
        del streams
        iterator.close()
        gc.collect()
        self.assertFalse(any(ref() is not None for ref in refs))


if __name__ == "__main__":
    unittest.main()
