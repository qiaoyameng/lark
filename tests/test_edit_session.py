# -*- coding: utf-8 -*-
"""Tests for the incremental LALR editing session (``Lark.edit_session``).

Every test compares the incremental result against a complete re-parse of the
edited document, including token and tree positions.
"""

import pytest

from lark import Lark, Token, Tree, Transformer
from lark.exceptions import (
    ConfigurationError, UnexpectedCharacters, UnexpectedInput, UnexpectedToken,
)


GRAMMAR = r"""
start: statement*
statement: assign | call
assign: NAME "=" expr _NL
call: NAME "(" args? ")" _NL
?expr: expr "+" term -> add
     | term
?term: NUMBER -> num
     | NAME -> ref
args: expr ("," expr)*

NAME: /[a-zA-Z_]\w*/
NUMBER: /\d+/
%ignore /[ \t]+/
_NL: /\r?\n/
"""


_POS_FIELDS = ('line', 'column', 'end_line', 'end_column', 'start_pos', 'end_pos',
               'container_line', 'container_column',
               'container_end_line', 'container_end_column',
               'container_start_pos', 'container_end_pos', 'empty')


def _serialize(node, out):
    if isinstance(node, Tree):
        meta = node.meta
        out.append(('T', node.data) + tuple(getattr(meta, f, None) for f in _POS_FIELDS))
        for child in node.children:
            _serialize(child, out)
    elif isinstance(node, Token):
        out.append(('K', node.type, str(node),
                    node.start_pos, node.end_pos, node.line, node.column,
                    node.end_line, node.end_column))
    else:
        out.append(('O', repr(node)))


def assert_trees_equal(t1, t2):
    s1, s2 = [], []
    _serialize(t1, s1)
    _serialize(t2, s2)
    assert s1 == s2


def all_tokens(tree):
    result = []
    for child in tree.children:
        if isinstance(child, Token):
            result.append(child)
        else:
            result.extend(all_tokens(child))
    return result


def make_lark(lexer='contextual', **kwargs):
    options = dict(parser='lalr', lexer=lexer, propagate_positions=True)
    options.update(kwargs)
    return Lark(GRAMMAR, **options)


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_initial_parse_matches_full_parse(lexer):
    lark = make_lark(lexer)
    text = "x = 1\nfoo(a, b)\ny = 2\n"
    session = lark.edit_session(text)
    assert session.error is None
    assert session.text == text
    assert_trees_equal(session.tree, lark.parse(text))


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_edit_single_character_in_token(lexer):
    lark = make_lark(lexer)
    text = "x = 1\ny = 2\n"
    session = lark.edit_session(text)

    session.replace(text.index('1'), text.index('1') + 1, '9')
    assert session.text == "x = 9\ny = 2\n"
    assert_trees_equal(session.tree, lark.parse("x = 9\ny = 2\n"))

    session.replace(0, 1, 'z')
    assert_trees_equal(session.tree, lark.parse("z = 9\ny = 2\n"))


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_cross_token_edit_merges_and_splits_tokens(lexer):
    lark = make_lark(lexer)
    text = "x = 123\n"
    session = lark.edit_session(text)

    # Replace across the NAME/"=" boundary, lengthening the NAME.
    session.replace(0, 2, 'ab')
    assert_trees_equal(session.tree, lark.parse("ab= 123\n"))

    # Split a NUMBER by inserting an operator across its middle.
    pos = session.text.index('12')
    session.replace(pos + 1, pos + 1, ' + 2')
    assert session.text == "ab= 1 + 223\n"
    assert_trees_equal(session.tree, lark.parse("ab= 1 + 223\n"))

    # Delete across several tokens again, re-merging into one NUMBER.
    session.replace(session.text.index('1'), session.text.index('223') + 3, '456')
    assert session.text == "ab= 456\n"
    assert_trees_equal(session.tree, lark.parse("ab= 456\n"))


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_insert_newline_recalculates_positions(lexer):
    lark = make_lark(lexer)
    text = "x = 1\ny = 2\nz = 3\n"
    session = lark.edit_session(text)
    original = session.tree

    # Insert a whole new statement (itself containing a newline) before "y":
    # an edit that adds a line break in the middle of the document.
    pos = text.index('y')
    session.insert(pos, "w = 0\n")
    new_text = "x = 1\nw = 0\ny = 2\nz = 3\n"
    assert session.text == new_text
    assert_trees_equal(session.tree, lark.parse(new_text))

    # Everything after the inserted line shifted down by one line and forward
    # in absolute position.
    y_token = [t for t in all_tokens(session.tree) if t.type == 'NAME' and t == 'y'][0]
    assert y_token.line == 3
    assert y_token.start_pos == new_text.index('y')
    z_token = [t for t in all_tokens(session.tree) if t.type == 'NAME' and t == 'z'][0]
    assert z_token.line == 4

    # Removing the inserted line restores every original position exactly.
    session.delete(pos, pos + len("w = 0\n"))
    assert session.text == text
    assert_trees_equal(session.tree, original)


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_syntax_error_matches_full_parse(lexer):
    lark = make_lark(lexer)
    session = lark.edit_session("x = 1\n")

    with pytest.raises(UnexpectedToken) as exc_incr:
        session.replace(0, 1, "1")
    assert session.tree is None
    assert isinstance(session.error, UnexpectedToken)

    with pytest.raises(UnexpectedToken) as exc_full:
        lark.parse("1 = 1\n")
    assert type(exc_incr.value) is type(exc_full.value)
    assert exc_incr.value.pos_in_stream == exc_full.value.pos_in_stream


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_recover_from_syntax_error(lexer):
    lark = make_lark(lexer)
    session = lark.edit_session("x = 1\n")

    with pytest.raises(UnexpectedInput):
        session.replace(0, 1, "1")
    assert session.error is not None

    # A follow-up edit repairs the document.
    session.replace(0, 1, "x")
    assert session.error is None
    assert_trees_equal(session.tree, lark.parse("x = 1\n"))

    # Delete the value, leaving a dangling assignment, then restore it.
    value_pos = session.text.index('1')
    with pytest.raises(UnexpectedInput):
        session.delete(value_pos, value_pos + 1)
    session.insert(value_pos, "9")
    assert_trees_equal(session.tree, lark.parse("x = 9\n"))


def test_initial_invalid_document():
    lark = make_lark('contextual')
    session = lark.edit_session("???")
    assert session.tree is None
    assert isinstance(session.error, UnexpectedCharacters)

    session.replace(0, 3, "x = 1\n")
    assert session.error is None
    assert_trees_equal(session.tree, lark.parse("x = 1\n"))


def test_lexer_error_matches_full_parse():
    lark = make_lark('contextual')
    session = lark.edit_session("x = 1\n")

    with pytest.raises(UnexpectedCharacters) as exc_incr:
        session.insert(0, "@")
    with pytest.raises(UnexpectedCharacters) as exc_full:
        lark.parse("@x = 1\n")
    assert exc_incr.value.pos_in_stream == exc_full.value.pos_in_stream
    assert exc_incr.value.line == exc_full.value.line
    assert session.text == "@x = 1\n"

    session.delete(0, 1)
    assert_trees_equal(session.tree, lark.parse("x = 1\n"))


class _CountingTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.calls = []

    def assign(self, children):
        self.calls.append(('assign', str(children[0])))
        return Tree('assign', children)

    def statement(self, children):
        return Tree('statement', children)

    def start(self, children):
        self.calls.append(('start', len(children)))
        return Tree('start', children)


def test_rule_callbacks_do_not_refire_on_retained_prefix():
    transformer = _CountingTransformer()
    lark = Lark(GRAMMAR, parser='lalr', lexer='contextual', transformer=transformer)
    text = "a = 1\nb = 2\nc = 3\nd = 4\n"
    session = lark.edit_session(text)

    # Edit only the second line: the first assignment must not be rebuilt.
    pos = session.text.index('2')
    transformer.calls.clear()
    session.replace(pos, pos + 1, '8')

    fired = [name for kind, name in transformer.calls if kind == 'assign']
    assert 'a' not in fired            # retained prefix
    assert fired == ['b', 'c', 'd']    # edited point and suffix
    assert ('start', 4) in transformer.calls
    assert_trees_equal(session.tree, lark.parse("a = 1\nb = 8\nc = 3\nd = 4\n"))

    # Editing the first token rebuilds the whole tree.
    transformer.calls.clear()
    session.replace(0, 1, 'z')
    fired = [name for kind, name in transformer.calls if kind == 'assign']
    assert fired == ['z', 'b', 'c', 'd']


def test_lexer_callbacks_only_refire_on_relexed_region():
    lark = make_lark('contextual')
    seen = []
    lark.lexer_conf.callbacks['NUMBER'] = lambda token: (seen.append(token.value), token)[1]

    text = "a = 1\nb = 2\nc = 3\n"
    session = lark.edit_session(text)

    seen.clear()
    pos = session.text.index('3')
    session.replace(pos, pos + 1, '7')

    # Tokens before the resume checkpoint are never lexed again.
    assert '1' not in seen
    assert '2' not in seen
    assert '7' in seen
    assert_trees_equal(session.tree, lark.parse("a = 1\nb = 2\nc = 7\n"))


def test_contextual_lexer_candidates_resume_correctly():
    # "if" only matches the IF terminal where an if-statement can start;
    # after an "if" was shifted, that parser context accepts NAME only.
    grammar = r"""
start: ifstmt | assign
ifstmt: "if" NAME ":" NAME
assign: NAME "=" NAME
NAME: /[a-z]+/
%ignore / /
"""
    lark = Lark(grammar, parser='lalr', lexer='contextual')
    session = lark.edit_session("if aa:bb")

    # Edit fully inside the NAME-only context.
    session.replace(6, 8, "cc")
    assert_trees_equal(session.tree, lark.parse("if aa:cc"))

    # Rewrite across contexts: if-statement -> assignment.
    session.replace(0, 8, "xx=yy")
    assert_trees_equal(session.tree, lark.parse("xx=yy"))

    # And back, restoring the IF candidate at position 0.
    session.replace(0, 5, "if q:r")
    assert_trees_equal(session.tree, lark.parse("if q:r"))


def test_bytes_input():
    lark = Lark(r"""
start: WORD+
WORD: /\w+/
%ignore / /
""", parser='lalr', use_bytes=True, propagate_positions=True)
    session = lark.edit_session(b"abc def")
    session.replace(4, 7, b"ghi jkl")
    assert_trees_equal(session.tree, lark.parse(b"abc ghi jkl"))


def test_multiple_start_rules():
    lark = Lark(r"""
a: NAME+
b: NUMBER+
NAME: /[a-z]+/
NUMBER: /\d+/
%ignore / /
""", parser='lalr', start=['a', 'b'], propagate_positions=True)

    session_a = lark.edit_session("x y", start='a')
    session_a.insert(3, " z")
    assert_trees_equal(session_a.tree, lark.parse("x y z", start='a'))

    session_b = lark.edit_session("1 2", start='b')
    # Inserting inside the first NUMBER merges "1" and the inserted "3".
    session_b.insert(1, "3")
    assert session_b.text == "13 2"
    assert_trees_equal(session_b.tree, lark.parse("13 2", start='b'))


def test_insert_and_delete_conveniences():
    lark = make_lark()
    session = lark.edit_session("x = 1\n")
    session.insert(0, "y = 2\n")
    assert_trees_equal(session.tree, lark.parse("y = 2\nx = 1\n"))
    session.delete(0, len("y = 2\n"))
    assert_trees_equal(session.tree, lark.parse("x = 1\n"))


def test_invalid_arguments():
    lark = make_lark()
    session = lark.edit_session("x = 1\n")
    with pytest.raises(ValueError):
        session.replace(0, 99, "z")
    with pytest.raises(ValueError):
        session.replace(2, 1, "z")
    with pytest.raises(TypeError):
        session.replace(0, 1, b"z")


class _IdentityPostLex:
    always_accept = ()

    def process(self, stream):
        return stream


def test_unsupported_configurations():
    earley = Lark(r"""
start: WORD+
WORD: /\w+/
""", parser='earley')
    with pytest.raises(ConfigurationError):
        earley.edit_session("ab")

    lark_post = Lark(GRAMMAR, parser='lalr', postlex=_IdentityPostLex())
    with pytest.raises(ConfigurationError):
        lark_post.edit_session("x = 1\n")


@pytest.mark.parametrize('lexer', ['basic', 'contextual'])
def test_random_edits_match_full_reparse(lexer):
    """Property test: for hundreds of random edit sequences (including
    cross-token edits, inserted newlines and error states), the incremental
    result is identical to a complete reparse, or the same error occurs."""
    import random
    lark = make_lark(lexer)
    seed_text = "x = 1\nfoo(a, b)\ny = 22 + z\n"
    alphabet = 'abcxyz=1234567890()+," \n\t'
    random.seed(20260925 if lexer == 'contextual' else 20260926)
    for _ in range(200):
        edits = []
        staged = seed_text
        for _edits in range(random.randint(1, 5)):
            n = len(staged)
            start = random.randint(0, n)
            end = random.randint(start, n)
            replacement = ''.join(random.choice(alphabet)
                                  for _ in range(random.randint(0, 6)))
            edits.append((start, end, replacement))
            staged = staged[:start] + replacement + staged[end:]

        session = lark.edit_session(seed_text)
        current = seed_text
        for start, end, replacement in edits:
            current = current[:start] + replacement + current[end:]
            try:
                incremental = session.replace(start, end, replacement)
            except UnexpectedInput as incr_error:
                with pytest.raises(UnexpectedInput) as full_error:
                    lark.parse(current)
                assert type(incr_error) is type(full_error.value)
                assert incr_error.pos_in_stream == full_error.value.pos_in_stream
                assert session.tree is None
                assert session.error is incr_error
            else:
                assert_trees_equal(incremental, lark.parse(current))
                assert session.error is None
        assert session.text == staged
