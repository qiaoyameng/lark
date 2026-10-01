# -*- coding: utf-8 -*-
"""Tests for LALR shift/reduce conflict witnesses (strict mode)."""
from __future__ import absolute_import

import unittest
from unittest.mock import patch

from lark import Lark
from lark.common import ParserConf
from lark.exceptions import GrammarError
from lark.parsers import lalr_analysis
from lark.parsers.lalr_analysis import (
    LALR_Analyzer,
    compute_shift_reduce_witness,
    replay_token_prefix,
)


DANGLING_ELSE_GRAMMAR = r'''
start: stmt
stmt: IF expr THEN stmt
    | IF expr THEN stmt ELSE stmt
    | ID
expr: ID
IF: "if"
THEN: "then"
ELSE: "else"
ID: /[a-z]+/

%ignore " "
'''

# b, c and d are only nullable indirectly, through the chain d -> e -> (empty)
INDIRECT_NULLABLE_GRAMMAR = r'''
start: S a Y
a: b X
b: c
c: d
d: e
e: X |
S: "s"
X: "x"
Y: "y"
'''


def make_analyzer(grammar):
    """Build a LALR_Analyzer for the grammar, resolving conflicts as shifts."""
    lark = Lark(grammar, parser='lalr')
    parser_conf = ParserConf(lark.rules, {}, lark.options.start)
    analyzer = LALR_Analyzer(parser_conf, debug=True)
    analyzer.compute_lalr()
    return analyzer


def find_conflicts(analyzer):
    """Return (state, lookahead, reduce_rule) triples for shift/reduce conflicts."""
    conflicts = []
    for state in analyzer.lr0_itemsets:
        for la, rules in state.lookaheads.items():
            if la in state.transitions:
                rule ,= rules
                conflicts.append((state, la, rule))
    return conflicts


class TestLALRConflictWitness(unittest.TestCase):

    def assert_witness_replays(self, grammar, expected_lookahead, expected_prefix):
        analyzer = make_analyzer(grammar)
        conflicts = find_conflicts(analyzer)
        self.assertTrue(conflicts, "expected a shift/reduce conflict")
        (state, la, rule), = [(s, l, r) for s, l, r in conflicts if l.name == expected_lookahead]

        witness = compute_shift_reduce_witness(analyzer, state, la, rule)
        self.assertIsNotNone(witness, "expected a conflict witness")
        self.assertEqual(witness.lookahead.name, expected_lookahead)
        self.assertEqual([t.name for t in witness.prefix], expected_prefix)
        self.assertTrue(witness.shift_items)
        self.assertEqual(witness.reduce_rule, rule)

        # The witness must genuinely replay into the reported conflict state
        start_state ,= analyzer.lr0_start_states.values()
        self.assertTrue(
            replay_token_prefix(start_state, witness.prefix, state, witness.lookahead),
            "witness prefix does not reach the reported conflict state",
        )
        # ... and not into some other state
        other = next(s for s in analyzer.lr0_itemsets if s is not state)
        self.assertFalse(
            replay_token_prefix(start_state, witness.prefix, other, witness.lookahead),
        )
        return witness

    def test_dangling_else(self):
        self.assert_witness_replays(
            DANGLING_ELSE_GRAMMAR, 'ELSE', ['IF', 'ID', 'THEN', 'ID'])

    def test_indirect_nullable(self):
        self.assert_witness_replays(
            INDIRECT_NULLABLE_GRAMMAR, 'X', ['S'])

    def test_strict_error_message(self):
        with self.assertRaises(GrammarError) as cm:
            Lark(DANGLING_ELSE_GRAMMAR, parser='lalr', strict=True)
        msg = str(cm.exception)
        self.assertIn('Shift/Reduce conflict for terminal ELSE. [strict-mode]', msg)
        self.assertIn(' * Reduce: <stmt : IF expr THEN stmt>', msg)
        self.assertIn(' * Shift: <stmt : IF expr THEN stmt * ELSE stmt>', msg)
        self.assertIn('IF ID THEN ID', msg)
        self.assertIn('lookahead ELSE', msg)
        # The witness is token-level; it must not claim to be parseable text
        self.assertIn('token-level witness', msg)

    def test_strict_error_message_indirect_nullable(self):
        with self.assertRaises(GrammarError) as cm:
            Lark(INDIRECT_NULLABLE_GRAMMAR, parser='lalr', strict=True)
        msg = str(cm.exception)
        self.assertIn('Shift/Reduce conflict for terminal X. [strict-mode]', msg)
        self.assertIn(' * Reduce: <e : >', msg)
        self.assertIn(' * Shift: <e :  * X>', msg)
        self.assertIn('token prefix [S]', msg)

    def test_search_bounds_fall_back_to_plain_message(self):
        # When the bounded search cannot produce a witness, the original
        # conflict information must still be reported.
        with patch.object(lalr_analysis, '_WITNESS_MAX_PREFIX_TOKENS', 0):
            with self.assertRaises(GrammarError) as cm:
                Lark(DANGLING_ELSE_GRAMMAR, parser='lalr', strict=True)
        msg = str(cm.exception)
        self.assertIn('Shift/Reduce conflict for terminal ELSE. [strict-mode]', msg)
        self.assertIn(' * <stmt : IF expr THEN stmt>', msg)
        self.assertNotIn('Witness', msg)

    def test_non_strict_parser_unaffected(self):
        # Conflicts are still resolved as shifts outside strict mode
        parser = Lark(DANGLING_ELSE_GRAMMAR, parser='lalr')
        tree = parser.parse('if a then if b then c else d')
        self.assertEqual(tree.data, 'start')


if __name__ == '__main__':
    unittest.main()
