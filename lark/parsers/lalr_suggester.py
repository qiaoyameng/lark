"""Bounded error-recovery suggestions for LALR parsing.

Given a failed LALR parse (an ``UnexpectedToken`` exception, or its
``InteractiveParser``), this module searches for small repairs -- token
insertions and deletions -- that would allow the parse to succeed.

The search is bounded by a step limit and a cost limit. All probing is done
on private copies of the parser state with the parse callbacks disabled, so
the user's transformer is never executed, and the given exception /
``InteractiveParser`` is never modified.

Suggestions only *describe* the repair (a span in the original text plus a
virtual token); they never modify the user's text.
"""

from copy import copy
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from typing import Dict, Iterator, List, Optional, Tuple

from ..exceptions import UnexpectedInput
from ..lexer import Token, PatternStr, TerminalDef

INSERT = 'insert'
DELETE = 'delete'

# Stop reasons reported in ``SuggestionResult.stop_reason`` when no
# suggestion could be found.
STOP_MAX_STEPS = 'max_steps_exceeded'
STOP_EXHAUSTED = 'search_exhausted'

_EOF_TYPES = ('$END', '<EOF>')

_KIND_RANK = {INSERT: 0, DELETE: 1}


@dataclass(frozen=True)
class Edit:
    """A single token insertion or deletion.

    Attributes:
        kind: ``'insert'`` or ``'delete'``.
        token: For insertions, the virtual token to insert. For deletions,
            the token that should be removed.
        start_pos: Start of the affected span in the original text.
            Zero-width (``start_pos == end_pos``) for insertions.
        end_pos: End of the affected span in the original text.
        cost: The cost of this edit, as configured on the suggester.
    """
    kind: str
    token: Token
    start_pos: Optional[int]
    end_pos: Optional[int]
    cost: float

    @property
    def replacement(self) -> str:
        """The text to splice into ``text[start_pos:end_pos]`` when applying this edit.

        For insertions of terminals with a literal (string) pattern, this is
        the literal itself. For pattern (regex) terminals it is empty -- the
        virtual token only carries the token type and position.
        """
        return str(self.token) if self.kind == INSERT else ''

    def _signature(self):
        return (self.kind, self.token.type, self.start_pos, self.end_pos)

    def _sort_key(self):
        return (_KIND_RANK[self.kind], self.token.type,
                self.start_pos if self.start_pos is not None else -1)

    def __str__(self):
        if self.kind == INSERT:
            return "insert %s at position %s" % (self.token.type, self.start_pos)
        return "delete %r (%s:%s)" % (self.token, self.start_pos, self.end_pos)


@dataclass(frozen=True)
class Suggestion:
    """A repair suggestion: a set of edits with a total cost.

    Applying a suggestion is up to the caller; ``apply()`` returns a *new*
    string and never modifies the given text.
    """
    edits: Tuple[Edit, ...]
    cost: float

    def apply(self, text: str) -> str:
        """Return a copy of ``text`` with all edits applied."""
        result = text
        for edit in sorted(self.edits, key=lambda e: e.start_pos or 0, reverse=True):
            start = edit.start_pos or 0
            end = edit.end_pos if edit.end_pos is not None else start
            result = result[:start] + edit.replacement + result[end:]
        return result

    def _signature(self):
        return frozenset(e._signature() for e in self.edits)

    def _sort_key(self):
        return (self.cost, len(self.edits), tuple(sorted(e._sort_key() for e in self.edits)))

    def __str__(self):
        return '%s (cost %s)' % (', '.join(str(e) for e in self.edits), self.cost)


@dataclass
class SuggestionResult:
    """The outcome of a repair search.

    Attributes:
        suggestions: The repairs found, sorted by cost and then by a fixed
            rule (fewer edits first, insertions before deletions, then by
            token type and position). Empty when nothing was found.
        stop_reason: ``None`` when at least one suggestion was found.
            Otherwise, why the search stopped: ``'max_steps_exceeded'`` or
            ``'search_exhausted'``.
        steps_used: How many search nodes were expanded.
    """
    suggestions: List[Suggestion]
    stop_reason: Optional[str]
    steps_used: int = 0

    def __bool__(self):
        return bool(self.suggestions)


class RepairSuggester:
    """Searches for token-level repairs for a failed LALR parse.

    The search is a best-first search (cheapest first) over parser states,
    where each step either inserts a virtual token accepted by the current
    state, or deletes a token from the remaining input. It is bounded by
    ``max_steps`` (total nodes expanded) and ``max_cost`` (total edit cost
    of a single repair), and returns at most ``max_suggestions`` repairs.

    Probing never runs the parse callbacks (i.e. the user's transformer) and
    never mutates the given exception or ``InteractiveParser``.
    """

    def __init__(self,
                 insert_cost: float = 1.0,
                 delete_cost: float = 1.0,
                 max_steps: int = 1000,
                 max_cost: float = 2.0,
                 max_suggestions: int = 3):
        assert insert_cost > 0 and delete_cost > 0, "Edit costs must be positive"
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.max_suggestions = max_suggestions

    def suggest(self, source) -> SuggestionResult:
        """Search for repairs, starting from the state of a failed parse.

        Parameters:
            source: Either the ``UnexpectedToken`` exception of the failed
                parse (recommended - its token gives precise edit spans), or
                an ``InteractiveParser`` positioned at the point of failure.
                Neither is modified.
        """
        if isinstance(source, UnexpectedInput):
            error_token = getattr(source, 'token', None)
            interactive_parser = source.interactive_parser
        else:
            error_token = None
            interactive_parser = source

        start_state = interactive_parser.parser_state.copy(deepcopy_values=False)
        # Disable all parse callbacks, so probing can't run the user's
        # transformer (or any other callback with side effects).
        conf = copy(start_state.parse_conf)
        conf.callbacks = {}
        start_state.parse_conf = conf

        terminals = self._terminals_by_name(interactive_parser)
        at_token_error = error_token is not None and error_token.type not in _EOF_TYPES

        queue: List[Tuple[float, int, object, Tuple[Edit, ...]]] = []
        tie = count()
        heappush(queue, (0.0, next(tie), start_state, ()))
        visited = {self._fingerprint(start_state, ())}
        seen_signatures = set()
        suggestions: List[Suggestion] = []
        steps = 0
        stop_reason: Optional[str] = STOP_EXHAUSTED

        while queue:
            cost, _, state, edits = heappop(queue)
            if steps >= self.max_steps:
                stop_reason = STOP_MAX_STEPS
                break
            steps += 1

            if edits and self._parses_to_end(state):
                signature = frozenset(e._signature() for e in edits)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    suggestions.append(Suggestion(edits, cost))
                    if len(suggestions) >= self.max_suggestions:
                        break
                # A state that parses to the end is a goal; extending it with
                # more edits would only produce costlier, redundant repairs.
                continue

            if not edits and at_token_error:
                # The lexer already consumed the offending token, so the root
                # expands differently: an insertion must re-feed the offending
                # token afterwards, and a deletion is just the state as-is.
                expansion = self._expand_at_error_token(state, terminals, error_token)
            else:
                expansion = self._expand(state, edits, cost, terminals)

            for child_cost, child_state, child_edits in expansion:
                fingerprint = self._fingerprint(child_state, child_edits)
                if fingerprint not in visited:
                    visited.add(fingerprint)
                    heappush(queue, (child_cost, next(tie), child_state, child_edits))

        suggestions.sort(key=lambda s: s._sort_key())
        return SuggestionResult(suggestions, None if suggestions else stop_reason, steps)

    ### Search primitives

    def _expand_at_error_token(self, state, terminals: Dict[str, TerminalDef],
                               error_token: Token) -> Iterator[Tuple[float, object, Tuple[Edit, ...]]]:
        """Root expansion when the offending token is known.

        The lexer of ``state`` is positioned *after* ``error_token`` (it was
        consumed before the error was raised), so:
          - insert: feed a virtual token, then re-feed the offending token
          - delete: keep the state as-is; the edit records the removed token
        """
        if self.insert_cost <= self.max_cost:
            for token_type in sorted(self._accepted_terminals(state)):
                if token_type == '$END':
                    continue
                child = state.copy(deepcopy_values=False)
                virtual = self._virtual_token(token_type, terminals, error_token, state)
                try:
                    child.feed_token(virtual)
                    child.feed_token(error_token)
                except UnexpectedInput:
                    continue
                pos = error_token.start_pos
                edit = Edit(INSERT, virtual, pos, pos, self.insert_cost)
                yield self.insert_cost, child, (edit,)

        if self.delete_cost <= self.max_cost:
            child = state.copy(deepcopy_values=False)
            edit = Edit(DELETE, error_token, error_token.start_pos, error_token.end_pos, self.delete_cost)
            yield self.delete_cost, child, (edit,)

    def _expand(self, state, edits: Tuple[Edit, ...], cost: float,
                terminals: Dict[str, TerminalDef]) -> Iterator[Tuple[float, object, Tuple[Edit, ...]]]:
        next_token = self._peek_token(state)

        if cost + self.insert_cost <= self.max_cost:
            for token_type in sorted(self._accepted_terminals(state)):
                if token_type == '$END':
                    continue
                child = state.copy(deepcopy_values=False)
                virtual = self._virtual_token(token_type, terminals, next_token, state)
                try:
                    child.feed_token(virtual)
                except UnexpectedInput:
                    continue
                pos = virtual.start_pos
                edit = Edit(INSERT, virtual, pos, pos, self.insert_cost)
                yield cost + self.insert_cost, child, edits + (edit,)

        if next_token is not None and cost + self.delete_cost <= self.max_cost:
            child = state.copy(deepcopy_values=False)
            deleted = self._consume_one(child)
            if deleted is not None:
                edit = Edit(DELETE, deleted, deleted.start_pos, deleted.end_pos, self.delete_cost)
                yield cost + self.delete_cost, child, edits + (edit,)

    def _parses_to_end(self, state) -> bool:
        """Return True if a probe copy of ``state`` can finish the parse."""
        probe = state.copy(deepcopy_values=False)
        last_token = None
        try:
            for token in probe.lexer.lex(probe):
                last_token = token
                probe.feed_token(token)
            if last_token is not None:
                end_token = Token.new_borrow_pos('$END', '', last_token)
            else:
                end_token = Token('$END', '', 0, 1, 1)
            probe.feed_token(end_token, True)
        except UnexpectedInput:
            return False
        return True

    def _accepted_terminals(self, state) -> List[str]:
        """Terminal types that the state can shift, probed without callbacks."""
        accepts = []
        for token_type in state.parse_conf.states[state.position]:
            if not token_type.isupper():    # not a terminal
                continue
            child = state.copy(deepcopy_values=False)
            try:
                child.feed_token(Token(token_type, ''))
            except UnexpectedInput:
                pass
            else:
                accepts.append(token_type)
        return accepts

    @staticmethod
    def _fingerprint(state, edits: Tuple[Edit, ...]):
        edits_key = frozenset(e._signature() for e in edits)
        return (tuple(state.state_stack), state.lexer.state.line_ctr.char_pos, edits_key)

    @staticmethod
    def _peek_token(state) -> Optional[Token]:
        """Return the next lexer token without advancing ``state``."""
        lexer_thread = copy(state.lexer)
        try:
            for token in lexer_thread.lex(state):
                return token
        except UnexpectedInput:
            pass
        return None

    @staticmethod
    def _consume_one(state) -> Optional[Token]:
        """Advance ``state``'s lexer past one token and return it."""
        try:
            for token in state.lexer.lex(state):
                return token
        except UnexpectedInput:
            pass
        return None

    @staticmethod
    def _virtual_token(token_type: str, terminals: Dict[str, TerminalDef],
                       next_token: Optional[Token], state) -> Token:
        """Create a virtual token, positioned where it would be inserted.

        The value is the literal pattern for string terminals, and empty
        otherwise (the parse tables only care about the token type).
        """
        value = ''
        terminal = terminals.get(token_type)
        if terminal is not None and isinstance(terminal.pattern, PatternStr):
            value = terminal.pattern.value
        if next_token is not None:
            return Token.new_borrow_pos(token_type, value, next_token)
        last_token = state.lexer.state.last_token
        if last_token is not None:
            return Token(token_type, value, last_token.end_pos, last_token.end_line,
                         last_token.end_column, last_token.end_line, last_token.end_column,
                         last_token.end_pos)
        return Token(token_type, value, 0, 1, 1, 1, 1, 0)

    @staticmethod
    def _terminals_by_name(interactive_parser) -> Dict[str, TerminalDef]:
        lexer = interactive_parser.lexer_thread.lexer
        terminals = getattr(lexer, 'terminals', None)
        if terminals is None:
            terminals = getattr(getattr(lexer, 'root_lexer', None), 'terminals', None)
        return {t.name: t for t in terminals} if terminals else {}
