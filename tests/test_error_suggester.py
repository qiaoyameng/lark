# -*- coding: utf-8 -*-
"""Tests for the LALR error-recovery suggester (lark.parsers.lalr_suggester)"""
from __future__ import absolute_import

import unittest

from lark import Lark, Token, Transformer, UnexpectedToken
from lark.parsers.lalr_suggester import (
    RepairSuggester, SuggestionResult,
    INSERT, DELETE, STOP_MAX_STEPS, STOP_EXHAUSTED,
)


NESTED_PARENS = r'''
    start: "(" start ")" | NAME
    NAME: /[a-z]+/
    %ignore " "
'''

OPTIONAL_SEPARATOR = r'''
    start: item ("," item)* ","?
    item: NAME
    NAME: /[a-z]+/
    %ignore " "
'''

MISSING_AT_EOF = r'''
    start: "begin" stmt ";"
    stmt: NAME
    NAME: /[a-z]+/
    %ignore " "
'''


def _make_parser(grammar, **options):
    options.setdefault('parser', 'lalr')
    return Lark(grammar, **options)


def _suggest(parser, text, **suggester_options):
    """Parse ``text``, expect it to fail, and return (error, SuggestionResult)."""
    try:
        parser.parse(text)
    except UnexpectedToken as e:
        return e, RepairSuggester(**suggester_options).suggest(e)
    raise AssertionError("Expected %r to fail parsing" % text)


class TestErrorSuggester(unittest.TestCase):
    def test_insert_missing_paren(self):
        parser = _make_parser(NESTED_PARENS)
        text = "( ( x )"
        e, res = _suggest(parser, text)

        self.assertIsNone(res.stop_reason)
        self.assertTrue(res.suggestions)

        suggestion = res.suggestions[0]
        self.assertEqual(len(suggestion.edits), 1)
        edit = suggestion.edits[0]
        self.assertEqual(edit.kind, INSERT)
        self.assertEqual(edit.token.type, 'RPAR')
        self.assertEqual(edit.token, ')')
        # Zero-width span at the end of the text
        self.assertEqual(edit.start_pos, len(text))
        self.assertEqual(edit.end_pos, len(text))

        # Applying the suggestion produces text that parses in one normal pass
        parser.parse(suggestion.apply(text))

    def test_delete_extra_separator(self):
        parser = _make_parser(OPTIONAL_SEPARATOR)
        text = "a,,b"
        e, res = _suggest(parser, text)

        self.assertIsNone(res.stop_reason)
        deletes = [s for s in res.suggestions
                   if len(s.edits) == 1 and s.edits[0].kind == DELETE]
        self.assertTrue(deletes, "Expected a deletion suggestion")
        edit = deletes[0].edits[0]
        self.assertEqual(edit.token.type, 'COMMA')
        self.assertEqual((edit.start_pos, edit.end_pos), (2, 3))
        self.assertEqual(edit.replacement, '')

        fixed = deletes[0].apply(text)
        self.assertEqual(fixed, "a,b")
        parser.parse(fixed)

    def test_missing_token_at_eof(self):
        parser = _make_parser(MISSING_AT_EOF)
        text = "begin x"
        e, res = _suggest(parser, text)

        self.assertIsNone(res.stop_reason)
        self.assertTrue(res.suggestions)
        edit = res.suggestions[0].edits[0]
        self.assertEqual(edit.kind, INSERT)
        self.assertEqual(edit.token.type, 'SEMICOLON')
        self.assertEqual(edit.start_pos, len(text))

        fixed = res.suggestions[0].apply(text)
        self.assertEqual(fixed, "begin x;")
        parser.parse(fixed)

    def test_multi_edit_repair_within_cost_limit(self):
        parser = _make_parser(NESTED_PARENS)
        text = "( ( x"
        e, res = _suggest(parser, text, max_cost=2.0)

        self.assertIsNone(res.stop_reason)
        self.assertTrue(res.suggestions)
        suggestion = res.suggestions[0]
        self.assertEqual(len(suggestion.edits), 2)
        self.assertEqual(suggested_cost(suggestion), 2.0)
        self.assertTrue(all(e.kind == INSERT and e.token.type == 'RPAR'
                            for e in suggestion.edits))
        parser.parse(suggestion.apply(text))

    def test_suggestions_sorted_by_cost_then_fixed_rule(self):
        parser = _make_parser(OPTIONAL_SEPARATOR)
        _, res = _suggest(parser, "a,,b", max_cost=2.0)

        costs = [s.cost for s in res.suggestions]
        self.assertEqual(costs, sorted(costs))
        # At equal cost, insertions sort before deletions (fixed rule)
        first_cost = [s for s in res.suggestions if s.cost == costs[0]]
        kinds = [s.edits[0].kind for s in first_cost]
        self.assertEqual(kinds, sorted(kinds, key={INSERT: 0, DELETE: 1}.get))

    def test_max_suggestions_is_respected(self):
        parser = _make_parser(OPTIONAL_SEPARATOR)
        _, res = _suggest(parser, "a,,b", max_cost=2.0, max_suggestions=1)
        self.assertEqual(len(res.suggestions), 1)

    def test_cost_limit_returns_empty_with_reason(self):
        parser = _make_parser(NESTED_PARENS)
        # Needs three inserted ')' but the budget allows only one edit
        _, res = _suggest(parser, "( ( ( x", max_cost=1.0)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.stop_reason, STOP_EXHAUSTED)

    def test_step_limit_returns_empty_with_reason(self):
        parser = _make_parser(NESTED_PARENS)
        _, res = _suggest(parser, "( ( ( x", max_steps=1)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.stop_reason, STOP_MAX_STEPS)

    def test_probing_does_not_run_transformer(self):
        calls = []

        class CountingTransformer(Transformer):
            def start(self, children):
                calls.append('start')
                return children

            def NAME(self, token):
                calls.append('NAME')
                return token

        parser = _make_parser(NESTED_PARENS, transformer=CountingTransformer())
        error = None
        try:
            parser.parse("( ( x )")
        except UnexpectedToken as e:
            error = e
        self.assertIsNotNone(error)
        calls_before = len(calls)

        res = RepairSuggester().suggest(error)
        self.assertTrue(res.suggestions)
        self.assertEqual(len(calls), calls_before,
                         "Probing must not run the user's transformer")

        # Sanity: applying the suggestion and parsing for real does run it
        parser.parse(res.suggestions[0].apply("( ( x )"))
        self.assertGreater(len(calls), calls_before)

    def test_interactive_parser_not_polluted(self):
        parser = _make_parser(NESTED_PARENS)
        error = None
        try:
            parser.parse("( ( x )")
        except UnexpectedToken as err:
            error = err
        ip = error.interactive_parser
        stack_before = list(ip.parser_state.state_stack)
        values_before = list(ip.parser_state.value_stack)
        pos_before = ip.lexer_thread.state.line_ctr.char_pos

        res = RepairSuggester().suggest(error)
        self.assertTrue(res.suggestions)

        self.assertEqual(list(ip.parser_state.state_stack), stack_before)
        self.assertEqual(list(ip.parser_state.value_stack), values_before)
        self.assertEqual(ip.lexer_thread.state.line_ctr.char_pos, pos_before)
        self.assertIsNone(ip.result)

    def test_accepts_interactive_parser_source(self):
        parser = _make_parser(NESTED_PARENS)
        ip = None
        try:
            parser.parse("( ( x )")
        except UnexpectedToken as e:
            ip = e.interactive_parser

        res = RepairSuggester().suggest(ip)
        self.assertTrue(res.suggestions)
        parser.parse(res.suggestions[0].apply("( ( x )"))

    def test_suggestion_does_not_modify_user_text(self):
        parser = _make_parser(NESTED_PARENS)
        text = "( ( x )"
        _, res = _suggest(parser, text)
        fixed = res.suggestions[0].apply(text)
        self.assertEqual(text, "( ( x )")      # original untouched
        self.assertNotEqual(fixed, text)
        self.assertIsInstance(fixed, str)


def suggested_cost(suggestion):
    return sum(e.cost for e in suggestion.edits)


if __name__ == '__main__':
    unittest.main()
