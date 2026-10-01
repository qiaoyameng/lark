# -*- coding: utf-8 -*-
"""Tests for Lark.edit_session() -- incremental reparsing on the LALR path."""

import random
import unittest

from lark import Lark, Tree, Token, Transformer
from lark.exceptions import ConfigurationError, UnexpectedInput, UnexpectedToken
from lark.edit_session import EditSession
from lark.indenter import Indenter


def assert_same_parse(test_case, expected, actual, path="root"):
    """Deep-compare two parse results, including token and meta positions."""
    if isinstance(expected, Tree):
        test_case.assertIsInstance(actual, Tree, path)
        test_case.assertEqual(expected.data, actual.data, path)
        m1, m2 = expected.meta, actual.meta
        for attr in ('line', 'column', 'start_pos', 'end_line', 'end_column', 'end_pos'):
            test_case.assertEqual(getattr(m1, attr, None), getattr(m2, attr, None),
                                  "%s: meta.%s" % (path, attr))
        test_case.assertEqual(len(expected.children), len(actual.children), path)
        for i, (c1, c2) in enumerate(zip(expected.children, actual.children)):
            assert_same_parse(test_case, c1, c2, "%s/%d" % (path, i))
    elif isinstance(expected, Token):
        test_case.assertIsInstance(actual, Token, path)
        for attr in ('type', 'value', 'start_pos', 'line', 'column', 'end_line', 'end_column', 'end_pos'):
            test_case.assertEqual(getattr(expected, attr), getattr(actual, attr),
                                  "%s: token.%s" % (path, attr))
    else:
        test_case.assertEqual(expected, actual, path)


class SessionHarness:
    """Applies edits to a session and checks each result against a full reparse."""

    def __init__(self, test_case, lark, text, **kwargs):
        self.test_case = test_case
        self.lark = lark
        self.session = lark.edit_session(text, **kwargs)
        assert_same_parse(test_case, lark.parse(text), self.session.result)

    @property
    def text(self):
        return self.session.text

    def edit(self, start, end, replacement):
        result = self.session.edit(start, end, replacement)
        assert_same_parse(self.test_case, self.lark.parse(self.session.text), result)
        return result


class TestEditSession(unittest.TestCase):
    def make_harness(self, grammar, text, **options):
        lark = Lark(grammar, parser='lalr', **options)
        return SessionHarness(self, lark, text)

    def test_basic_edits(self):
        h = self.make_harness(r"""
            ?start: expr
            ?expr: term (OP term)*
            ?term: NUMBER | NAME
            OP: "--" | "-" | "+"
            NUMBER: /\d+/
            NAME: /[a-z]+/
            %ignore " "
        """, "1 + abc - 22", propagate_positions=True)
        h.edit(4, 7, "xy")          # replace a token: "1 + xy - 22"
        h.edit(7, 8, "+")           # replace an operator
        h.edit(0, 0, "5 + ")        # insert at the start
        h.edit(15, 15, " + 7")      # insert at the end
        h.edit(0, 8, "")            # delete a prefix
        self.assertEqual(h.text, "xy + 22 + 7")

    def test_cross_token_edit(self):
        # Edits whose range spans several tokens
        h = self.make_harness(r"""
            start: item ("," item)*
            item: NAME | NUMBER
            NAME: /[a-z]+/
            NUMBER: /\d+/
            %ignore " "
        """, "aa, bb, cc, dd", propagate_positions=True)
        h.edit(3, 10, "1, 2")       # replace a span covering several tokens
        self.assertEqual(h.text, "aa,1, 2, dd")
        h.edit(0, 11, "zz")         # replace the whole document
        self.assertEqual(h.text, "zz")
        h.edit(0, 2, "aa, 2, cc")   # rebuild from scratch
        self.assertEqual(h.text, "aa, 2, cc")

    def test_edit_inside_token(self):
        h = self.make_harness(r"""
            start: NAME+
            NAME: /[a-z]+/
            %ignore " "
        """, "hello world")
        h.edit(0, 3, "j")           # "hello" -> "jlo"
        self.assertEqual(h.text, "jlo world")
        h.edit(4, 4, "wide ")       # insert between tokens
        self.assertEqual(h.text, "jlo wide world")

    def test_token_merge_maximal_munch(self):
        # Inserting '-' next to '-' must re-lex as '--': the checkpoint right
        # before the edit is unsafe, the session must back off.
        h = self.make_harness(r"""
            ?start: expr
            ?expr: term (OP term)*
            ?term: NUMBER
            OP: "--" | "-"
            NUMBER: /\d+/
            %ignore " "
        """, "1 - 2")
        tree = h.edit(3, 3, "-")    # "1 -- 2"
        ops = list(tree.scan_values(lambda t: isinstance(t, Token) and t.type == 'OP'))
        self.assertEqual(ops, ['--'])

    def test_token_merge_across_gap(self):
        # A token can merge across unchanged text: "axc" -> "axb" must re-lex
        # from scratch, because 'a' alone is no longer a valid token.
        h = self.make_harness(r"""
            start: AB | A X last
            last: B | C
            AB: "axb"
            A: "a"
            X: "x"
            B: "b"
            C: "c"
            %ignore " "
        """, "axc")
        tree = h.edit(2, 3, "b")    # "axc" -> "axb"
        self.assertEqual(tree.children[0], "axb")
        self.assertEqual(tree.children[0].type, "AB")

    def test_newline_insertion_positions(self):
        h = self.make_harness(r"""
            start: item ("," item)*
            item: NAME | NUMBER
            NAME: /[a-z]+/
            NUMBER: /\d+/
            %ignore " " | "\n"
        """, "aa, bb, cc", propagate_positions=True)
        tree = h.edit(3, 3, "\n\n")
        lines = [tok.line for tok in tree.scan_values(lambda t: isinstance(t, Token))]
        self.assertEqual(lines, [1, 3, 3])
        h.edit(0, 0, "\n")          # push everything one line down
        lines = [tok.line for tok in h.session.result.scan_values(lambda t: isinstance(t, Token))]
        self.assertEqual(lines, [2, 4, 4])

    def test_syntax_error_recovery(self):
        h = self.make_harness(r"""
            ?start: expr
            ?expr: term (OP term)*
            ?term: NUMBER | NAME
            OP: "--" | "-" | "+"
            NUMBER: /\d+/
            NAME: /[a-z]+/
            %ignore " "
        """, "1 + abc - 22", propagate_positions=True)
        h.edit(4, 7, "xy")
        # Break the document
        with self.assertRaises(UnexpectedInput):
            h.session.edit(8, 9, "+")
        self.assertEqual(h.text, "1 + xy -+22")
        self.assertIsNone(h.session.result)
        # The session is still usable: repair the document
        tree = h.edit(8, 9, "")
        self.assertEqual(h.text, "1 + xy -22")
        self.assertIs(tree, h.session.result)
        # Break it lexically (no terminal matches)
        with self.assertRaises(UnexpectedInput):
            h.session.edit(0, 0, "&")
        self.assertIsNone(h.session.result)
        h.edit(0, 1, "")            # remove the offending character
        self.assertEqual(h.text, "1 + xy -22")

    def test_contextual_lexer(self):
        # Only the contextual lexer can parse this: at the start state, "x"
        # must lex as the literal, not as part of a NAME.
        h = self.make_harness(r"""
            start: "x" NAME "y" NAME
            NAME: /[a-z]+/
            %ignore " "
        """, "xhello yworld")
        h.edit(1, 6, "abc")         # "xabc yworld"
        h.edit(6, 11, "foo")        # "xabc yfoo"
        h.edit(1, 4, "hi")          # "xhi yfoo"
        self.assertEqual(h.text, "xhi yfoo")

    def test_contextual_lexer_basic(self):
        # Same grammar also works with the basic lexer path of the session
        h = self.make_harness(r"""
            start: NAME ("," NAME)*
            NAME: /[a-z]+/
            %ignore " "
        """, "aa, bb", lexer='basic')
        h.edit(3, 6, " cc, dd")
        self.assertEqual(h.text, "aa, cc, dd")

    def test_transformer_side_effects(self):
        counts = {'NUMBER': [], 'number': 0, 'start': 0}

        class CountT(Transformer):
            def NUMBER(self, t):
                counts['NUMBER'].append(str(t))
                return int(t)

            def number(self, children):
                counts['number'] += 1
                return children[0]

            def start(self, children):
                counts['start'] += 1
                return sum(children)

        lark = Lark("""
            start: number ("+" number)*
            number: NUMBER
            NUMBER: /\d+/
            %ignore " "
        """, parser='lalr', transformer=CountT())

        session = lark.edit_session("1 + 22 + 333")
        self.assertEqual(session.result, 356)
        self.assertEqual(counts['NUMBER'], ['1', '22', '333'])
        self.assertEqual(counts['number'], 3)
        self.assertEqual(counts['start'], 1)

        # Edit the last token: only it may be re-lexed and re-transformed
        result = session.edit(9, 12, "44")
        self.assertEqual(result, 67)
        self.assertEqual(counts['NUMBER'], ['1', '22', '333', '44'])
        self.assertEqual(counts['number'], 4)
        self.assertEqual(counts['start'], 2)

        # Edit the first token: everything is re-lexed
        result = session.edit(0, 1, "5")
        self.assertEqual(result, 71)
        self.assertEqual(counts['NUMBER'], ['1', '22', '333', '44', '5', '22', '44'])

    def test_prefix_nodes_are_reused(self):
        h = self.make_harness(r"""
            start: item ("," item)*
            item: NAME
            NAME: /[a-z]+/
            %ignore " "
        """, "aa, bb, cc")
        before = h.session.result
        after = h.edit(8, 10, "dd")     # only the last item changes
        self.assertIs(before.children[0], after.children[0])
        # The retained token is reused, though its parent node may be
        # re-reduced (LALR reduces with one token of lookahead)
        self.assertIs(before.children[1].children[0], after.children[1].children[0])
        self.assertIsNot(before.children[2], after.children[2])

    def test_random_edits_match_reparse(self):
        grammar = r"""
            ?start: sum
            ?sum: product (("+"|"-") product)*
            ?product: atom (("*"|"/") atom)*
            ?atom: NUMBER | "(" sum ")"
            NUMBER: /\d+/
            %ignore " " | "\n"
        """
        lark = Lark(grammar, parser='lalr', propagate_positions=True)
        text = "1 + (22 - 3) * 4 + 555 - (6 + 77)"
        session = lark.edit_session(text)
        assert_same_parse(self, lark.parse(text), session.result)

        rng = random.Random(1234)
        alphabet = "0123456789+-*/() \n"
        for i in range(200):
            text = session.text
            start = rng.randrange(len(text) + 1)
            end = rng.randrange(start, len(text) + 1)
            replacement = "".join(rng.choice(alphabet) for _ in range(rng.randrange(4)))
            try:
                result = session.edit(start, end, replacement)
            except UnexpectedInput:
                # Broken document: a full reparse must fail too
                with self.assertRaises(UnexpectedInput):
                    lark.parse(session.text)
                continue
            assert_same_parse(self, lark.parse(session.text), result)

    def test_random_edits_json(self):
        grammar = r"""
            ?start: value
            ?value: object
                  | array
                  | string
                  | SIGNED_NUMBER      -> number
                  | "true"             -> true
                  | "false"            -> false
                  | "null"             -> null
            array  : "[" _WS? [value ("," _WS? value)*] "]"
            object : "{" _WS? [pair ("," _WS? pair)*] "}"
            pair   : string ":" _WS value
            string : ESCAPED_STRING
            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS -> _WS
        """
        lark = Lark(grammar, parser='lalr', propagate_positions=True)
        text = '{"name": "lark", "tags": ["parser", "lalr"], "version": 1.5, "stable": true}'
        session = lark.edit_session(text)
        assert_same_parse(self, lark.parse(text), session.result)

        rng = random.Random(42)
        alphabet = '{}[],":0123456789.eE+-truefalsn \n'
        for i in range(150):
            text = session.text
            start = rng.randrange(len(text) + 1)
            end = rng.randrange(start, len(text) + 1)
            replacement = "".join(rng.choice(alphabet) for _ in range(rng.randrange(5)))
            try:
                result = session.edit(start, end, replacement)
            except UnexpectedInput:
                with self.assertRaises(UnexpectedInput):
                    lark.parse(session.text)
                continue
            assert_same_parse(self, lark.parse(session.text), result)

    def test_lexer_callbacks_not_refired(self):
        # User lexer callbacks must not fire again for the retained prefix,
        # not even for the checkpoint validation probe.
        calls = []

        def to_int(token):
            calls.append(str(token))
            return token.update(value=int(token))

        lark = Lark(r"""
            start: NUMBER ("," NUMBER)*
            NUMBER: /\d+/
            %ignore " "
        """, parser='lalr', lexer_callbacks={'NUMBER': to_int})
        session = lark.edit_session("1, 22, 333")
        self.assertEqual(calls, ['1', '22', '333'])
        tree = session.edit(3, 5, "5")      # "1, 5, 333"
        # '1' is retained and doesn't fire again; '5' and '333' are re-lexed
        self.assertEqual(calls, ['1', '22', '333', '5', '333'])
        assert_same_parse(self, lark.parse("1, 5, 333"), tree)
        # Token values were transformed by the callback
        self.assertEqual([tok.value for tok in tree.scan_values(lambda t: isinstance(t, Token))],
                         [1, 5, 333])

    def test_unsupported_configurations(self):
        with self.assertRaises(ConfigurationError):
            Lark('start: "a"', parser='earley').edit_session("a")
        with self.assertRaises(ConfigurationError):
            Lark('start: "a"', parser='cyk').edit_session("a")

        class MyIndenter(Indenter):
            NL_type = '_NL'
            OPEN_PAREN_types = []
            CLOSE_PAREN_types = []
            INDENT_type = '_INDENT'
            DEDENT_type = '_DEDENT'
            tab_len = 4

        grammar = "\n".join([
            'start: "a"',
            '%ignore " "',
        ])
        with self.assertRaises(ConfigurationError):
            Lark(grammar, parser='lalr', postlex=MyIndenter()).edit_session("a")

        from lark.lexer import Lexer
        class CustomLexer(Lexer):
            def __init__(self, lexer_conf): pass
            def lex(self, lexer_state, parser_state): return iter(())
        with self.assertRaises(ConfigurationError):
            Lark('start: "a"', parser='lalr', lexer=CustomLexer).edit_session("a")

    def test_invalid_edit_range(self):
        session = Lark('start: "a"', parser='lalr').edit_session("a")
        self.assertRaises(ValueError, session.edit, -1, 1, "b")
        self.assertRaises(ValueError, session.edit, 1, 0, "b")
        self.assertRaises(ValueError, session.edit, 0, 2, "b")
        # The session is unaffected
        self.assertEqual(session.text, "a")
        session.edit(0, 1, "a")

    def test_multiple_start_symbols(self):
        lark = Lark("""
            a: "x"+
            b: "y"+
        """, parser='lalr', start=['a', 'b'])
        session = lark.edit_session("yyy", start='b')
        tree = session.edit(0, 0, "yy")
        self.assertEqual(tree.data, 'b')
        self.assertRaises(ConfigurationError, lark.edit_session, "yyy")

    def test_bytes_input(self):
        lark = Lark("start: /[ab]+/", parser='lalr', use_bytes=True)
        session = lark.edit_session(b"aabb")
        session.edit(1, 3, b"ba")
        self.assertEqual(session.text, b"abab")
        assert_same_parse(self, lark.parse(b"abab"), session.result)


if __name__ == '__main__':
    unittest.main()
