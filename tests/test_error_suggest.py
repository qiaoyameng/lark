import unittest

from lark import Lark, Token, Transformer, UnexpectedToken
from lark.error_suggest import (
    DELETE,
    INSERT,
    STOP_FOUND,
    STOP_MAX_COST,
    STOP_MAX_STEPS,
    STOP_UNSUPPORTED,
    suggest_fixes,
)


def parse_error(parser, text):
    try:
        parser.parse(text)
    except UnexpectedToken as e:
        return e
    raise AssertionError('Expected UnexpectedToken for %r' % text)


class TestErrorSuggest(unittest.TestCase):
    def setUp(self):
        self.parens = Lark(r'''
            start: "x" | "(" start ")"
            %ignore " "
        ''', parser='lalr')
        self.separators = Lark(r'''
            start: "a" ("," "a")* ","?
            %ignore " "
        ''', parser='lalr')
        self.list = Lark(r'''
            start: "(" "a" ("," "a")* ")"
            %ignore " "
        ''', parser='lalr')

    def test_missing_rparen_at_eof(self):
        text = '( x'
        e = parse_error(self.parens, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        self.assertEqual(len(res.suggestions), 1)
        sug = res.suggestions[0]
        self.assertEqual(sug.kind, INSERT)
        edit = sug.edits[0]
        self.assertEqual(edit.token.type, 'RPAR')
        self.assertEqual(edit.token.value, ')')
        self.assertEqual(edit.start_pos, len(text))
        self.assertEqual(edit.end_pos, edit.start_pos)
        self.assertEqual((edit.line, edit.column), (1, len(text) + 1))
        self.assertEqual(sug.apply(text), '( x)')
        self.parens.parse(sug.apply(text))

    def test_missing_two_rparens_at_eof(self):
        text = '( ( x'
        e = parse_error(self.parens, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        sug = res.suggestions[0]
        self.assertEqual(sug.cost, 2.0)
        self.assertEqual([ed.token.type for ed in sug.edits], ['RPAR', 'RPAR'])
        self.assertEqual(sug.apply(text), '( ( x))')
        self.parens.parse(sug.apply(text))

    def test_extra_rparen(self):
        text = '( x ) )'
        e = parse_error(self.parens, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        sug = res.suggestions[0]
        self.assertEqual(sug.kind, DELETE)
        edit = sug.edits[0]
        self.assertEqual(edit.token.type, 'RPAR')
        self.assertEqual((edit.start_pos, edit.end_pos), (6, 7))
        self.assertEqual(sug.apply(text), '( x ) ')
        self.parens.parse(sug.apply(text))

    def test_extra_separator(self):
        text = 'a,,a'
        e = parse_error(self.separators, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        kinds = [s.kind for s in res.suggestions]
        self.assertIn(DELETE, kinds)
        self.assertIn(INSERT, kinds)
        for sug in res.suggestions:
            self.separators.parse(sug.apply(text))
        deletes = [s for s in res.suggestions if s.kind == DELETE]
        self.assertEqual(deletes[0].edits[0].token.type, 'COMMA')
        self.assertEqual(deletes[0].apply(text), 'a,a')

    def test_missing_separator(self):
        text = 'a a'
        e = parse_error(self.separators, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        inserts = [s for s in res.suggestions if s.kind == INSERT]
        self.assertTrue(inserts)
        self.assertEqual(inserts[0].edits[0].token.type, 'COMMA')
        for sug in res.suggestions:
            self.separators.parse(sug.apply(text))

    def test_missing_rparen_after_list(self):
        text = '(a, a'
        e = parse_error(self.list, text)
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        sug = res.suggestions[0]
        self.assertEqual(sug.kind, INSERT)
        self.assertEqual(sug.edits[0].token.type, 'RPAR')
        self.assertEqual(sug.apply(text), '(a, a)')
        self.list.parse(sug.apply(text))

    def test_suggestions_sorted_by_cost(self):
        text = '( ( x'
        e = parse_error(self.parens, text)
        res = suggest_fixes(e, max_suggestions=5)
        costs = [s.cost for s in res.suggestions]
        self.assertEqual(costs, sorted(costs))

    def test_max_steps_limit(self):
        e = parse_error(self.parens, '( x')
        res = suggest_fixes(e, max_steps=0)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.stop_reason, STOP_MAX_STEPS)
        self.assertFalse(res.ok)

    def test_max_cost_limit(self):
        e = parse_error(self.parens, '( ( ( x')
        res = suggest_fixes(e, max_cost=1)
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.stop_reason, STOP_MAX_COST)

    def test_unsupported_error(self):
        res = suggest_fixes(ValueError('nope'))
        self.assertEqual(res.suggestions, [])
        self.assertEqual(res.stop_reason, STOP_UNSUPPORTED)

    def test_interactive_parser_input(self):
        ip = self.parens.parse_interactive('( x')
        ip.exhaust_lexer()
        end = Token.new_borrow_pos('$END', '', ip.lexer_thread.state.last_token)
        try:
            ip.feed_token(end)
        except UnexpectedToken as e:
            res = suggest_fixes(ip, e.token)
        else:
            self.fail('Expected UnexpectedToken')
        self.assertEqual(res.stop_reason, STOP_FOUND)
        self.assertEqual(res.suggestions[0].edits[0].token.type, 'RPAR')

    def test_probing_does_not_pollute_interactive_parser(self):
        e = parse_error(self.parens, '( x')
        ip = e.interactive_parser
        state_stack = list(ip.parser_state.state_stack)
        value_stack = list(ip.parser_state.value_stack)
        char_pos = ip.lexer_thread.state.line_ctr.char_pos
        callbacks = ip.parser_state.parse_conf.callbacks

        res = suggest_fixes(e)
        self.assertTrue(res.ok)

        self.assertEqual(ip.parser_state.state_stack, state_stack)
        self.assertEqual(ip.parser_state.value_stack, value_stack)
        self.assertEqual(ip.lexer_thread.state.line_ctr.char_pos, char_pos)
        self.assertIs(ip.parser_state.parse_conf.callbacks, callbacks)

    def test_probing_does_not_run_transformer(self):
        calls = []

        class CountingTransformer(Transformer):
            def start(self, children):
                calls.append('start')
                return children

            def __default_token__(self, token):
                calls.append(token.type)
                return token

        parser = Lark(r'''
            start: "x" | "(" start ")"
            %ignore " "
        ''', parser='lalr', transformer=CountingTransformer())
        e = parse_error(parser, '( x')
        # The failed parse itself may have triggered callbacks; probing must not add any.
        calls_before = list(calls)
        res = suggest_fixes(e)
        self.assertTrue(res.ok)
        self.assertEqual(calls, calls_before)

    def test_insert_without_known_lexeme(self):
        parser = Lark(r'''
            start: "a" NUMBER "b"
            NUMBER: /[0-9]+/
            %ignore " "
        ''', parser='lalr')
        e = parse_error(parser, 'a b')
        res = suggest_fixes(e)
        self.assertEqual(res.stop_reason, STOP_FOUND)
        inserts = [s for s in res.suggestions if s.kind == INSERT]
        self.assertTrue(inserts)
        edit = inserts[0].edits[0]
        self.assertEqual(edit.token.type, 'NUMBER')
        self.assertIsNone(edit.text)
        with self.assertRaises(ValueError):
            inserts[0].apply('a b')


if __name__ == '__main__':
    unittest.main()
