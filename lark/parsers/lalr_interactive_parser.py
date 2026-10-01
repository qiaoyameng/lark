# This module provides a LALR interactive parser, which is used for debugging and error handling

from typing import Iterator, List
from copy import copy
import warnings

from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
from .lalr_parser_state import ParserState

###{standalone

class CompletionCandidate:
    """The result of analyzing a single accepted token for completability.

    Attributes:
        token_type: The name of the accepted token type.
        suffix: A finite sequence of token types that, when fed after ``token_type``
            (followed by EOF), leads to a complete parse. ``None`` when no such
            suffix could be found within the search limits.
        completable: True iff ``suffix`` is not None.
    """
    __slots__ = 'token_type', 'suffix', 'completable'

    def __init__(self, token_type, suffix):
        self.token_type = token_type
        self.suffix = suffix
        self.completable = suffix is not None

    def __repr__(self):
        return '%s(%r, suffix=%r)' % (type(self).__name__, self.token_type, self.suffix)

    def __eq__(self, other):
        if not isinstance(other, CompletionCandidate):
            return NotImplemented
        return self.token_type == other.token_type and self.suffix == other.suffix

    def __hash__(self):
        return hash((self.token_type, self.suffix))


class InteractiveParser:
    """InteractiveParser gives you advanced control over parsing and error handling when parsing with LALR.

    For a simpler interface, see the ``on_error`` argument to ``Lark.parse()``.
    """
    def __init__(self, parser, parser_state: ParserState, lexer_thread: LexerThread):
        self.parser = parser
        self.parser_state = parser_state
        self.lexer_thread = lexer_thread
        self.result = None

    @property
    def lexer_state(self) -> LexerThread:
        warnings.warn("lexer_state will be removed in subsequent releases. Use lexer_thread instead.", DeprecationWarning)
        return self.lexer_thread

    def feed_token(self, token: Token):
        """Feed the parser with a token, and advance it to the next state, as if it received it from the lexer.

        Note that ``token`` has to be an instance of ``Token``.
        """
        return self.parser_state.feed_token(token, token.type == '$END')

    def iter_parse(self) -> Iterator[Token]:
        """Step through the different stages of the parse, by reading tokens from the lexer
        and feeding them to the parser, one per iteration.

        Returns an iterator of the tokens it encounters.

        When the parse is over, the resulting tree can be found in ``InteractiveParser.result``.
        """
        for token in self.lexer_thread.lex(self.parser_state):
            yield token
            self.result = self.feed_token(token)

    def exhaust_lexer(self) -> List[Token]:
        """Try to feed the rest of the lexer state into the interactive parser.

        Note that this modifies the instance in place and does not feed an '$END' Token
        """
        return list(self.iter_parse())


    def feed_eof(self, last_token=None):
        """Feed a '$END' Token. Borrows from 'last_token' if given."""
        eof = Token.new_borrow_pos('$END', '', last_token) if last_token is not None else self.lexer_thread._Token('$END', '', 0, 1, 1)
        return self.feed_token(eof)


    def __copy__(self):
        """Create a new interactive parser with a separate state.

        Calls to feed_token() won't affect the old instance, and vice-versa.
        """
        return self.copy()

    def copy(self, deepcopy_values=True):
        parser_state = self.parser_state.copy(deepcopy_values=deepcopy_values)
        return type(self)(self.parser, parser_state, parser_state.lexer)

    def __eq__(self, other):
        if not isinstance(other, InteractiveParser):
            return False

        return self.parser_state == other.parser_state and self.lexer_thread == other.lexer_thread

    def as_immutable(self):
        """Convert to an ``ImmutableInteractiveParser``."""
        p = copy(self)
        return ImmutableInteractiveParser(p.parser, p.parser_state, p.lexer_thread)

    def pretty(self):
        """Print the output of ``choices()`` in a way that's easier to read."""
        out = ["Parser choices:"]
        for k, v in self.choices().items():
            out.append('\t- %s -> %r' % (k, v))
        out.append('stack size: %s' % len(self.parser_state.state_stack))
        return '\n'.join(out)

    def choices(self):
        """Returns a dictionary of token types, matched to their action in the parser.

        Only returns token types that are accepted by the current state.

        Updated by ``feed_token()``.
        """
        return self.parser_state.parse_conf.parse_table.states[self.parser_state.position]

    def accepts(self):
        """Returns the set of possible tokens that will advance the parser into a new valid state."""
        accepts = set()
        conf_no_callbacks = copy(self.parser_state.parse_conf)
        # We don't want to call callbacks here since those might have arbitrary side effects
        # and are unnecessarily slow.
        conf_no_callbacks.callbacks = {}
        for t in self.choices():
            if t.isupper(): # is terminal?
                new_cursor = self.copy(deepcopy_values=False)
                new_cursor.parser_state.parse_conf = conf_no_callbacks
                try:
                    new_cursor.feed_token(self.lexer_thread._Token(t, ''))
                except UnexpectedToken:
                    pass
                else:
                    accepts.add(t)
        return accepts

    def completion_analysis(self, max_steps=1000) -> List[CompletionCandidate]:
        """Analyze whether each token in ``accepts()`` can lead to a complete parse.

        For every accepted token, search (using only the grammar and the current
        parser state, without invoking any user callbacks) for a finite suffix of
        token types that completes the parse.

        Returns a list of ``CompletionCandidate``, one per accepted token.
        Candidates for which no completion could be proven (e.g. dead branches
        that can be shifted but never finished, or completions that exceed
        ``max_steps``) are marked with ``completable == False`` and a ``suffix``
        of ``None``.

        The search is breadth-first and bounded by ``max_steps`` state
        expansions, so recursion, nullable cycles and empty productions cannot
        make it loop forever.
        """
        return [
            CompletionCandidate(t, self._find_completion_suffix(t, max_steps))
            for t in sorted(self.accepts())
        ]

    def _find_completion_suffix(self, token_type, max_steps):
        """Breadth-first search for a finite suffix of token types that, when fed
        after ``token_type`` (followed by EOF), results in a complete parse.

        Returns a tuple of token types (possibly empty), or None if no completion
        was found within ``max_steps`` state expansions.
        """
        conf_no_callbacks = copy(self.parser_state.parse_conf)
        # We don't want to call callbacks here since those might have arbitrary
        # side effects and are unnecessarily slow.
        conf_no_callbacks.callbacks = {}
        make_token = self.lexer_thread._Token
        states = conf_no_callbacks.states

        def trial_state(state):
            s = state.copy(deepcopy_values=False)
            s.parse_conf = conf_no_callbacks
            return s

        start = trial_state(self.parser_state)
        try:
            start.feed_token(make_token(token_type, ''), token_type == '$END')
        except UnexpectedToken:
            return None
        if token_type == '$END':
            return ()

        # The LALR automaton is fully determined by its state stack (the value
        # stack only feeds callbacks, which are disabled), so visited stacks
        # never need to be expanded again. Recursive grammars can still produce
        # unboundedly many distinct stacks, hence the max_steps bound.
        visited = {tuple(start.state_stack)}
        queue = [(start, ())]
        steps = 0
        while queue and steps < max_steps:
            state, suffix = queue.pop(0)
            steps += 1

            eof_state = trial_state(state)
            try:
                eof_state.feed_token(make_token('$END', ''), True)
            except UnexpectedToken:
                pass
            else:
                return suffix

            for t in states[state.position]:
                if not t.isupper() or t == '$END':  # not a feedable terminal
                    continue
                next_state = trial_state(state)
                try:
                    next_state.feed_token(make_token(t, ''))
                except UnexpectedToken:
                    continue
                key = tuple(next_state.state_stack)
                if key not in visited:
                    visited.add(key)
                    queue.append((next_state, suffix + (t,)))
        return None

    def resume_parse(self):
        """Resume automated parsing from the current state.
        """
        return self.parser.parse_from_state(self.parser_state, last_token=self.lexer_thread.state.last_token)



class ImmutableInteractiveParser(InteractiveParser):
    """Same as ``InteractiveParser``, but operations create a new instance instead
    of changing it in-place.
    """

    result = None

    def __hash__(self):
        return hash((self.parser_state, self.lexer_thread))

    def feed_token(self, token):
        c = copy(self)
        c.result = InteractiveParser.feed_token(c, token)
        return c

    def exhaust_lexer(self):
        """Try to feed the rest of the lexer state into the parser.

        Note that this returns a new ImmutableInteractiveParser and does not feed an '$END' Token"""
        cursor = self.as_mutable()
        cursor.exhaust_lexer()
        return cursor.as_immutable()

    def as_mutable(self):
        """Convert to an ``InteractiveParser``."""
        p = copy(self)
        return InteractiveParser(p.parser, p.parser_state, p.lexer_thread)

###}
