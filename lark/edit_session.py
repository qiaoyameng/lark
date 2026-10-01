"""Incremental parsing sessions for the LALR parser.

This module implements ``Lark.edit_session()``. An ``EditSession`` parses an
initial document, and then applies small text edits (replace a range with new
text), reusing as much of the previous parse as possible, instead of
reparsing the whole document on every keystroke.

Only the unchanged prefix is reused. The session records a *checkpoint* at
every token boundary, capturing the full parser state (state stack and value
stack) and the full lexer state (line/column counter and last token), rather
than just a list of tokens. Because the parser state is part of the
checkpoint, resuming from it also restores the contextual lexer's candidate
terminal set, which is derived from the parser's position. Everything at and
after the edit is re-lexed from the new text and re-parsed, so all tokens,
tree nodes and position information past the edit point are recomputed.
Since the retained prefix is neither re-lexed nor re-reduced, lexer callbacks
and transformer callbacks never fire twice for it.
"""

from typing import Any, List, Optional, Tuple, Union

from .exceptions import ConfigurationError, UnexpectedInput
from .lexer import BasicLexer, ContextualLexer, LexerState, LexerThread, Token
from .parsers.lalr_interactive_parser import InteractiveParser
from .parsers.lalr_parser_state import ParseConf, ParserState
from .tree import Tree
from .utils import TextSlice

###{standalone
from bisect import bisect_left
from contextlib import contextmanager
from copy import copy


class _Checkpoint:
    """A snapshot of the parser and lexer state at a token boundary.

    ``parser_state`` is a copy in which the lexer state sits right after
    ``token`` (the token that was just fed), so parsing can resume from it
    without re-lexing or re-reducing anything before it.
    """
    __slots__ = ('token', 'raw_token', 'parser_state')

    def __init__(self, token: Optional[Token], raw_token: Optional[Token], parser_state: ParserState):
        self.token = token              # The token just fed (None for the initial checkpoint)
        self.raw_token = raw_token      # The same token before lexer callbacks were applied
        self.parser_state = parser_state


def _freeze_value(value):
    """Copy a value retained by a checkpoint, giving it a private children list.

    ``ChildFilterLALR`` optimizes left-recursion by aliasing the first child's
    ``children`` list and appending to it in place. That is safe for a single
    parse (the old node is dead afterwards), but it would corrupt nodes that a
    checkpoint still references. Feeding the callback a frozen copy instead
    keeps the retained node immutable.
    """
    frozen = copy(value)
    frozen.children = list(value.children)
    return frozen


class EditSession:
    """An incremental parsing session over a document that is being edited.

    Create one with ``Lark.edit_session()``. The session parses the initial
    text eagerly; ``edit()`` then applies a replacement and returns the
    updated parse result, reusing the unchanged prefix from its checkpoints.

    A checkpoint is recorded at every token boundary, so memory use is
    proportional to the token count. Edits near the end of the document are
    the cheapest; an edit at offset ``p`` re-parses everything after ``p``.

    Not thread-safe. The parse results of consecutive edits may share
    (unchanged) subtrees; treat returned trees as read-only, or copy them
    before mutating.
    """

    def __init__(self, parsing_frontend, text: Union[str, bytes], start: Optional[str] = None):
        frontend = parsing_frontend
        if frontend.parser_conf.parser_type != 'lalr':
            raise ConfigurationError("edit_session() requires parser='lalr'")
        if frontend.skip_lexer:
            raise ConfigurationError("edit_session() does not support lexer='dynamic'/'dynamic_complete'")
        if frontend.lexer_conf.postlex is not None:
            # postlex carries state across the stream (indent depth, paren nesting)
            raise ConfigurationError("edit_session() does not support postlex")
        if not isinstance(text, (str, bytes)):
            raise TypeError("edit_session() expects text as str or bytes, got %r" % type(text).__name__)

        self._basic_lexers: List[BasicLexer] = []
        self._raw_records: List[Tuple[Token, Token]] = []
        lexer = frontend.lexer
        if isinstance(lexer, ContextualLexer):
            self._lexer = self._clone_contextual_lexer(lexer)
        elif isinstance(lexer, BasicLexer):
            self._lexer = self._clone_basic_lexer(lexer)
        else:
            raise ConfigurationError("edit_session() does not support custom lexers")

        self._parser_impl = frontend.parser.parser  # lark.parsers.lalr_parser._Parser
        self.start = frontend._verify_start(start)
        self._retained_value_ids = set()
        self._parse_conf = ParseConf(self._parser_impl.parse_table,
                                     self._protect_callbacks(self._parser_impl.callbacks),
                                     self.start)

        self._checkpoints: List[_Checkpoint] = []
        self._positions: List[int] = []
        self.text = text
        self.result: Any = None

        parser_state = ParserState(self._parse_conf, LexerThread.from_text(self._lexer, text))
        self._record_checkpoint(parser_state, None, None)
        self.result = self._parse(parser_state)

    # -- Value protection ---------------------------------------------------
    # Checkpoints share tree nodes with the ongoing parse. Lark's LALR child
    # filter mutates the children list of left-recursion targets in place,
    # which would retroactively corrupt the nodes a checkpoint retains. The
    # session therefore wraps the rule callbacks: when a reduction is about
    # to expand a retained node in place, it is first swapped for a frozen
    # copy. Normal parses (outside a session) keep the fast path untouched.

    def _protect_callbacks(self, callbacks):
        protected = {}
        for key, callback in callbacks.items():
            if isinstance(key, str) or not self._rule_may_alias(key):
                protected[key] = callback
            else:
                protected[key] = self._make_protecting_callback(callback)
        return protected

    @staticmethod
    def _rule_may_alias(rule) -> bool:
        # Mirrors maybe_create_child_filter: ChildFilterLALR aliases the
        # children list of the first kept symbol when it is expandable
        # (the left-recursion optimization). Only those reductions can
        # mutate a retained node in place.
        keep_all_tokens = rule.options.keep_all_tokens
        for sym in rule.expansion:
            if keep_all_tokens or not (sym.is_term and sym.filter_out):
                return not sym.is_term and sym.name.startswith('_')
        return False

    def _make_protecting_callback(self, callback):
        def protecting_callback(children):
            if children:
                first = children[0]
                if id(first) in self._retained_value_ids:
                    children = list(children)
                    children[0] = _freeze_value(first)
            return callback(children)

        return protecting_callback

    def _register_retained_values(self, parser_state: ParserState):
        for value in parser_state.value_stack:
            if isinstance(value, Tree):
                self._retained_value_ids.add(id(value))

    def _rebuild_retained_values(self):
        self._retained_value_ids.clear()
        for checkpoint in self._checkpoints:
            self._register_retained_values(checkpoint.parser_state)

    # -- Lexer cloning ----------------------------------------------------
    # The session lexes through private shallow copies of the lexers (they
    # share the compiled scanners with the originals). This lets the session
    # wrap the token callbacks to record raw (pre-callback) tokens, and
    # disable callbacks while probing, without affecting concurrent parses.

    def _recording_callback(self, callback):
        records = self._raw_records

        def record_and_call(token):
            # Copy the token: callbacks such as UnlessCallback mutate it in place
            raw = Token.new_borrow_pos(token.type, token.value, token)
            result = callback(token)
            records.append((raw, result))
            return result

        return record_and_call

    def _clone_basic_lexer(self, lexer: BasicLexer) -> BasicLexer:
        lexer.scanner  # Make sure `callback` has been built
        clone = copy(lexer)
        clone.callback = {name: self._recording_callback(cb) for name, cb in lexer.callback.items()}
        self._basic_lexers.append(clone)
        return clone

    def _clone_contextual_lexer(self, lexer: ContextualLexer) -> ContextualLexer:
        clone = copy(lexer)
        clone.lexers = {state: self._clone_basic_lexer(sub) for state, sub in lexer.lexers.items()}
        clone.root_lexer = self._clone_basic_lexer(lexer.root_lexer)
        return clone

    @contextmanager
    def _callbacks_disabled(self):
        lexers = self._basic_lexers
        saved = [lexer.callback for lexer in lexers]
        for lexer in lexers:
            lexer.callback = {}
        try:
            yield
        finally:
            for lexer, callback in zip(lexers, saved):
                lexer.callback = callback

    # -- Parse loop ---------------------------------------------------------

    @staticmethod
    def _snapshot_state(parser_state: ParserState) -> ParserState:
        # Like parser_state.copy(deepcopy_values=False), but avoids the
        # generic copy() machinery (it dominates the checkpointing cost).
        lexer_thread = parser_state.lexer
        lexer_state = lexer_thread.state
        new_lexer_state = LexerState(lexer_state.text, copy(lexer_state.line_ctr), lexer_state.last_token)
        return ParserState(
            parser_state.parse_conf,
            LexerThread(lexer_thread.lexer, new_lexer_state),
            list(parser_state.state_stack),
            list(parser_state.value_stack),
        )

    def _record_checkpoint(self, parser_state: ParserState, token: Optional[Token], raw_token: Optional[Token]):
        checkpoint_state = self._snapshot_state(parser_state)
        self._checkpoints.append(_Checkpoint(token, raw_token, checkpoint_state))
        self._positions.append(checkpoint_state.lexer.state.line_ctr.char_pos)
        self._register_retained_values(checkpoint_state)

    def _match_raw_token(self, token: Token) -> Token:
        for raw, result in self._raw_records:
            if result is token:
                return raw
        return token

    def _parse(self, parser_state: ParserState):
        """Run the LALR parse loop, recording a checkpoint after every token."""
        try:
            token = None
            token_stream = parser_state.lexer.lex(parser_state)
            while True:
                del self._raw_records[:]
                token = next(token_stream, None)
                if token is None:
                    break
                raw_token = self._match_raw_token(token)
                parser_state.feed_token(token)
                self._record_checkpoint(parser_state, token, raw_token)

            end_token = Token.new_borrow_pos('$END', '', token) if token is not None else Token('$END', '', 0, 1, 1)
            return parser_state.feed_token(end_token, True)
        except UnexpectedInput as e:
            e.interactive_parser = InteractiveParser(self._parser_impl, parser_state, parser_state.lexer)
            raise

    # -- Editing ------------------------------------------------------------

    def _next_raw_token(self, lexer_state: LexerState, parser_state: ParserState) -> Token:
        lexer = self._lexer
        if isinstance(lexer, ContextualLexer):
            return lexer.lexers[parser_state.position].next_token(lexer_state, parser_state)
        return lexer.next_token(lexer_state, parser_state)

    def _validate_checkpoint(self, index: int, new_slice: TextSlice) -> bool:
        """Check that the token right before checkpoint ``index`` is still lexed
        identically in the new text.

        Maximal munch can merge a token with the edited text that follows it
        (e.g. inserting ``-`` after ``-`` when ``--`` is a terminal), even
        when the token ends before the edit. Re-lexing one token (with lexer
        callbacks disabled, so user callbacks don't fire again) detects that;
        the caller then backs off to an earlier checkpoint.
        """
        previous = self._checkpoints[index - 1].parser_state
        old_lexer_state = previous.lexer.state
        lexer_state = LexerState(new_slice, copy(old_lexer_state.line_ctr), old_lexer_state.last_token)
        try:
            with self._callbacks_disabled():
                token = self._next_raw_token(lexer_state, previous)
        except (EOFError, UnexpectedInput):
            return False
        raw_token = self._checkpoints[index].raw_token
        return (token.type == raw_token.type
                and token.value == raw_token.value
                and token.start_pos == raw_token.start_pos
                and token.end_pos == raw_token.end_pos)

    def edit(self, start: int, end: int, replacement: Union[str, bytes]):
        """Replace ``text[start:end]`` with ``replacement`` and reparse.

        Only the unchanged prefix is reused: parsing resumes from the last
        safe checkpoint before ``start``, and everything after it is re-lexed
        and re-parsed. Returns the updated parse result (a ``Tree`` by
        default, or the transformer's output).

        On a parse error, the ``UnexpectedInput`` exception propagates (as
        with ``Lark.parse()``), ``result`` is set to ``None``, but the
        session stays valid: ``text`` reflects the edit, and further calls to
        ``edit()`` keep working, so the document can be repaired.
        """
        text = self.text
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid edit range (%r, %r) for text of length %d" % (start, end, len(text)))
        new_text = text[:start] + replacement + text[end:]
        new_slice = TextSlice.cast_from(new_text)

        # Find the latest checkpoint whose token ends strictly before the edit.
        # (A token ending exactly at `start` could merge with the new text.)
        index = bisect_left(self._positions, start) - 1
        if index < 0:
            index = 0
        while index > 0 and not self._validate_checkpoint(index, new_slice):
            index -= 1

        del self._checkpoints[index + 1:]
        del self._positions[index + 1:]
        self._rebuild_retained_values()

        parser_state = self._snapshot_state(self._checkpoints[index].parser_state)
        parser_state.lexer.state.text = new_slice
        self.text = new_text
        try:
            self.result = self._parse(parser_state)
        except UnexpectedInput:
            self.result = None
            raise
        return self.result
###}
