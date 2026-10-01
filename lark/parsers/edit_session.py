"""Incremental editing sessions for the LALR(1) parser.

An ``EditSession`` parses a document once and then accepts a series of
(replacement-range, new-text) edits without re-parsing the whole document.
Each edit resumes from a checkpoint taken at the token boundary immediately
before the change, so only the affected region and everything after it is
re-lexed and re-parsed.

A checkpoint captures more than a cached token list:

* the parser's state *and* value stacks (the LALR automaton position and the
  already-built tree nodes of the unchanged prefix), and
* the full lexer state: the scan position, the line/column counter and the
  last produced token. With the contextual lexer the candidate terminal set
  is derived from the restored parser position, so context-sensitive
  tokenization resumes correctly as well.

This is a first version: only the unchanged prefix is reused. Every tree node
and every token position at or after the edit is recomputed from scratch, and
the checkpoints only cover boundaries of tokens that were actually shifted by
the parser.

Rule-reduction callbacks (and therefore ``transformer`` rule methods) never
run a second time for the retained prefix: its tree nodes are shared from the
value stack as-is. Terminal/lexer callbacks may re-fire for the small region
between the resume checkpoint and the edit, because that text has to be
re-lexed to find the first token boundary; they only ever see text that is
being lexed again.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

from ..exceptions import ConfigurationError, UnexpectedInput
from ..lexer import LexerState, LexerThread, LineCounter, TextSlice, Token, BasicLexer, ContextualLexer
from ..tree import Tree
from ..parse_tree_builder import ParseTreeBuilder
from .lalr_parser_state import ParseConf, ParserState

###{standalone

if TYPE_CHECKING:
    from ..lark import Lark


@dataclass(frozen=True)
class _Checkpoint:
    """A resumable point in the parse, taken after a token was shifted.

    ``pos`` is the absolute character offset of the first unconsumed character
    (i.e. the end of the shifted token). ``line`` and ``line_start_pos`` are
    the line-counter values at that offset, which lets the lexer resume
    position tracking without recounting the prefix.
    """
    pos: int
    line: int
    line_start_pos: int
    last_token: Optional[Token]
    state_stack: list
    value_stack: list


class EditSession:
    """An incremental parse session for the LALR(1) parser.

    Create one with :meth:`Lark.edit_session`, then apply edits with
    :meth:`replace` (or the convenience :meth:`insert` / :meth:`delete`).
    Every result is identical to parsing the resulting document from scratch;
    when the edited document is invalid, the same exception that a full parse
    would raise is raised, while the session still records the valid prefix so
    that a follow-up edit can recover.

    Only ``parser='lalr'`` with the built-in ``'basic'`` or ``'contextual'``
    lexer is supported. Post-lexers (such as the indentation post-lexer),
    custom lexer classes and ``lexer='dynamic'`` are rejected.
    """

    def __init__(self, lark_instance: 'Lark', text: Union[str, bytes], start: Optional[str] = None):
        options = lark_instance.options
        if options.parser != 'lalr':
            raise ConfigurationError("EditSession requires parser='lalr'")
        if options.postlex is not None:
            raise ConfigurationError("EditSession does not support postlex")

        frontend = lark_instance.parser
        lexer = getattr(frontend, 'lexer', None)
        if not isinstance(lexer, (BasicLexer, ContextualLexer)):
            raise ConfigurationError(
                "EditSession only supports the built-in 'basic' and 'contextual' lexers")

        chosen_start = frontend._verify_start(start)

        # The LALR tree builder normally installs in-place child filters
        # (ChildFilterLALR), which mutate a child tree's children while
        # reducing. Nodes consumed by such a reduction are still shared with
        # earlier checkpoints, so resuming from them could hand the parser
        # already-mutated trees. Building callbacks with ambiguous=True selects
        # the copying ChildFilter, which builds exactly the same parse tree for
        # an unambiguous LALR grammar while leaving every reduced node intact.
        parse_tree_builder = ParseTreeBuilder(
            lark_instance.rules,
            options.tree_class or Tree,
            options.propagate_positions,
            ambiguous=True,
            maybe_placeholders=options.maybe_placeholders,
        )
        callbacks = parse_tree_builder.create_callback(options.transformer)
        # Reuse the terminal (lexer) callbacks that the normal parse path uses
        callbacks.update(frontend.lexer_conf.callbacks)

        self._lexer = lexer
        self._parse_conf = ParseConf(
            frontend.parser._parse_table,
            callbacks,
            chosen_start,
        )
        self._callbacks = callbacks

        self._text = text
        self._checkpoints: List[_Checkpoint] = []
        self.tree: Any = None
        self.error: Optional[UnexpectedInput] = None

        self._run_initial()

    # -- public API --------------------------------------------------------

    @property
    def text(self) -> Union[str, bytes]:
        "The current document text."
        return self._text

    def replace(self, start: int, end: int, new_text: Union[str, bytes]) -> Any:
        """Replace ``self.text[start:end]`` with ``new_text`` and re-parse incrementally.

        Indices refer to the text *before* this edit. Returns the updated tree
        (or the transformer result). Raises the same ``UnexpectedInput``
        subclass that ``Lark.parse`` would raise on the new document; in that
        case ``self.error`` holds the exception and ``self.tree`` is ``None``.

        A failed edit is still applied to the session text, so coordinates for
        the next edit refer to the edited document.
        """
        old_text = self._text
        if not 0 <= start <= end <= len(old_text):
            raise ValueError("Replacement range (%d, %d) is out of bounds for text of length %d"
                             % (start, end, len(old_text)))
        if type(new_text) is not type(old_text):
            raise TypeError("new_text must have the same type as the session text (%s)"
                            % type(old_text).__name__)

        new_full_text = old_text[:start] + new_text + old_text[end:]
        return self._edit(new_full_text, start)

    def insert(self, pos: int, new_text: Union[str, bytes]) -> Any:
        "Insert ``new_text`` at offset ``pos``."
        return self.replace(pos, pos, new_text)

    def delete(self, start: int, end: int) -> Any:
        "Delete ``self.text[start:end]``."
        empty = b'' if isinstance(self._text, bytes) else ''
        return self.replace(start, end, empty)

    # -- internals ---------------------------------------------------------

    def _run_initial(self) -> None:
        checkpoints: List[_Checkpoint] = []
        slice_ = TextSlice.cast_from(self._text)
        line_ctr = LineCounter.from_text_slice(slice_)
        lexer_state = LexerState(slice_, line_ctr)
        parser_state = ParserState(self._parse_conf,
                                   LexerThread(self._lexer, lexer_state))
        # Checkpoint at the very beginning, before anything is consumed.
        checkpoints.append(_Checkpoint(
            slice_.start, line_ctr.line, line_ctr.line_start_pos,
            lexer_state.last_token,
            list(parser_state.state_stack), list(parser_state.value_stack)))

        result, error = self._consume(parser_state, lexer_state, checkpoints, len(checkpoints))
        self._commit(result, error, checkpoints)

    def _edit(self, new_text: Union[str, bytes], edit_start: int) -> Any:
        # Find the latest checkpoint strictly before the edit: its boundary is
        # the end of a token that cannot be affected by the edit. Resuming from
        # there re-lexes that token's successor onwards, which is exactly the
        # suffix whose tokenization and positions may have changed.
        cps = self._checkpoints
        new_len = len(new_text)
        lo, hi = 0, len(cps)
        while lo < hi:
            mid = (lo + hi) // 2
            if cps[mid].pos < edit_start:
                lo = mid + 1
            else:
                hi = mid
        # lo points just past the rightmost checkpoint strictly before the
        # edit; the zero-position checkpoint (empty stacks) is always safe,
        # even though its position isn't strictly before an edit at 0.
        lo = max(lo, 1)
        # The replacement may delete text beyond the resume boundary: walk
        # back until the chosen checkpoint also lies inside the new document.
        # Both conditions must hold for the same index (the prefix and the
        # retained checkpoints must describe the same parser state).
        while lo > 1 and cps[lo - 1].pos > new_len:
            lo -= 1
        cp = cps[lo - 1]
        new_checkpoints = cps[:lo]

        slice_ = TextSlice.cast_from(new_text)
        line_ctr = LineCounter(b'\n' if isinstance(new_text, bytes) else '\n')
        line_ctr.char_pos = cp.pos
        line_ctr.line = cp.line
        line_ctr.line_start_pos = cp.line_start_pos
        line_ctr.column = cp.pos - cp.line_start_pos + 1

        lexer_state = LexerState(slice_, line_ctr, cp.last_token)
        parser_state = ParserState(
            self._parse_conf,
            LexerThread(self._lexer, lexer_state),
            list(cp.state_stack),
            list(cp.value_stack),
        )

        base_len = len(new_checkpoints)
        result, error = self._consume(parser_state, lexer_state, new_checkpoints, base_len)
        self._text = new_text
        self._commit(result, error, new_checkpoints)
        if error is not None:
            raise error
        return result

    def _consume(self, parser_state, lexer_state, checkpoints, base_len):
        """Run the LALR loop over the lexer from the given state, checkpointing
        after every shifted token. Returns ``(result, error)``; exactly one is
        ``None``. ``checkpoints`` is extended in place.
        """
        token = None
        try:
            for token in parser_state.lexer.lex(parser_state):
                assert token is not None
                parser_state.feed_token(token)
                line_ctr = lexer_state.line_ctr
                checkpoints.append(_Checkpoint(
                    line_ctr.char_pos, line_ctr.line, line_ctr.line_start_pos,
                    lexer_state.last_token,
                    list(parser_state.state_stack), list(parser_state.value_stack)))

            if token is not None:
                end_token = Token.new_borrow_pos('$END', '', token)
            else:
                # The lexer yielded nothing after the resume point; borrow the
                # position from the checkpoint's last token (or start fresh).
                lt = lexer_state.last_token
                if lt is not None:
                    end_token = Token.new_borrow_pos('$END', '', lt)
                else:
                    end_token = Token('$END', '', 0, 1, 1)
            return parser_state.feed_token(end_token, True), None
        except UnexpectedInput as e:
            # The parse failed. None of the checkpoints produced by this run
            # is known to lie on a valid derivation (they may be mid-rule
            # states that wrongly accept the suffix of a later edit), so only
            # the safe prefix inherited from the previous document survives.
            del checkpoints[base_len:]
            return None, e

    def _commit(self, result, error, checkpoints):
        self.tree = result
        self.error = error
        self._checkpoints = checkpoints


###}
