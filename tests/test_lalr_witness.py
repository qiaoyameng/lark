"""Tests for LALR shift/reduce conflict witnesses (strict mode)."""

import unittest

from lark import Lark
from lark.exceptions import GrammarError
from lark.parsers import lalr_witness


DANGLING_ELSE = r"""
start: stmt
stmt: IF expr THEN stmt
    | IF expr THEN stmt ELSE stmt
    | PASS
expr: ID
IF: "if"
THEN: "then"
ELSE: "else"
PASS: "pass"
ID: /[a-z]+/
%ignore " "
"""

# `opt` is only indirectly nullable, through `maybe`
INDIRECT_NULLABLE = r"""
start: P s
s: opt B C | B D
opt: maybe
maybe: A |
P: "p"
A: "a"
B: "b"
C: "c"
D: "d"
%ignore " "
"""


class TestLalrConflictWitness(unittest.TestCase):
    def test_dangling_else_witness(self):
        with self.assertRaises(GrammarError) as cm:
            Lark(DANGLING_ELSE, parser='lalr', strict=True)
        err = cm.exception
        msg = str(err)
        # original conflict information is preserved
        self.assertIn('Shift/Reduce conflict for terminal ELSE. [strict-mode]', msg)
        # witness: token prefix, lookahead and both competing items
        self.assertIn('Token prefix to conflict state: IF ID THEN PASS', msg)
        self.assertIn('Lookahead triggering the conflict: ELSE', msg)
        self.assertIn('Shift : <stmt : IF expr THEN stmt * ELSE stmt>', msg)
        self.assertIn('Reduce: <stmt : IF expr THEN stmt * >', msg)

        witness = err.conflict_witness
        self.assertIsNotNone(witness)
        self.assertEqual(witness.prefix, ('IF', 'ID', 'THEN', 'PASS'))
        self.assertEqual(witness.lookahead, 'ELSE')
        # replaying the witness must genuinely reach the reported conflict point
        self.assertTrue(witness.verify())

    def test_dangling_else_stays_token_level(self):
        # ID is a regex terminal, so no concrete lexemes may be fabricated;
        # the witness must remain token-level and say so.
        with self.assertRaises(GrammarError) as cm:
            Lark(DANGLING_ELSE, parser='lalr', strict=True)
        msg = str(cm.exception)
        self.assertIn('token names, not source text', msg)
        self.assertNotIn('Example lexemes', msg)
        self.assertIsNone(cm.exception.conflict_witness.example_lexemes)

    def test_indirect_nullable_witness(self):
        with self.assertRaises(GrammarError) as cm:
            Lark(INDIRECT_NULLABLE, parser='lalr', strict=True)
        err = cm.exception
        msg = str(err)
        self.assertIn('Shift/Reduce conflict for terminal B. [strict-mode]', msg)
        self.assertIn('Token prefix to conflict state: P', msg)
        self.assertIn('Lookahead triggering the conflict: B', msg)
        self.assertIn('Shift : <s :  * B D>', msg)
        self.assertIn('Reduce: <maybe :  * >', msg)

        witness = err.conflict_witness
        self.assertIsNotNone(witness)
        self.assertEqual(witness.prefix, ('P',))
        # replaying the witness must genuinely reach the reported conflict point
        self.assertTrue(witness.verify())

    def test_example_lexemes_for_string_terminals(self):
        # every terminal is a plain string, so concrete lexemes are available
        with self.assertRaises(GrammarError) as cm:
            Lark(INDIRECT_NULLABLE, parser='lalr', strict=True)
        witness = cm.exception.conflict_witness
        self.assertEqual(witness.example_lexemes, ('p', 'b'))
        self.assertIn("Example lexemes for the prefix + lookahead: 'p' 'b'", str(cm.exception))

    def test_bounded_search_falls_back_to_plain_message(self):
        # with a zero token budget the witness search aborts, but the
        # original conflict information must still be reported
        old_budget = lalr_witness.MAX_WITNESS_PREFIX_TOKENS
        lalr_witness.MAX_WITNESS_PREFIX_TOKENS = 0
        try:
            with self.assertRaises(GrammarError) as cm:
                Lark(DANGLING_ELSE, parser='lalr', strict=True)
        finally:
            lalr_witness.MAX_WITNESS_PREFIX_TOKENS = old_budget
        err = cm.exception
        self.assertIn('Shift/Reduce conflict for terminal ELSE. [strict-mode]', str(err))
        self.assertNotIn('Conflict witness', str(err))
        self.assertIsNone(err.conflict_witness)

    def test_non_strict_parser_unaffected(self):
        # conflicts still resolve as shift outside strict mode
        parser = Lark(DANGLING_ELSE, parser='lalr')
        tree = parser.parse('if x then if x then pass else pass')
        self.assertEqual(tree.data, 'start')
        parser = Lark(INDIRECT_NULLABLE, parser='lalr')
        tree = parser.parse('p b d')
        self.assertEqual(tree.data, 'start')


if __name__ == '__main__':
    unittest.main()
