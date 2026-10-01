"""Bounded search for token-level fixes after a LALR parse error.

Given an UnexpectedToken error (or an InteractiveParser positioned at
the point of failure), this module performs a bounded best-first search over
small edits - inserting a virtual token before the offending token, or
deleting the offending token - and returns a list of suggestions that would
let the parse complete.

The search is bounded by a step limit (how many search nodes are expanded)
and a cost limit (the maximum total edit cost of a suggestion). Probing is
done on copies of the interactive parser with all parse callbacks disabled,
so the user's transformer is never invoked and the original parser state is
never modified. Suggestions only describe edits (source span + virtual
token); they never modify the user's text.
"""

from copy import copy
from dataclasses import dataclass, field
from heapq import heappush, heappop
from itertools import count
from typing import Dict, List, Optional, Tuple, Union

from .exceptions import UnexpectedInput, UnexpectedToken
from .lexer import Token, PatternStr
from .parsers.lalr_interactive_parser import InteractiveParser

INSERT = 'insert'
DELETE = 'delete'

# Stop reasons for SuggestionResult.stop_reason
STOP_FOUND = 'found'                            # at least one suggestion was found
STOP_MAX_STEPS = 'max_steps_exceeded'           # step limit reached before exhausting the search
STOP_MAX_COST = 'max_cost_exceeded'             # remaining candidates all exceed the cost limit
STOP_EXHAUSTED = 'search_exhausted'             # search space exhausted, nothing found
STOP_NO_PARSER = 'no_interactive_parser'        # error carries no interactive parser to probe with
STOP_UNSUPPORTED = 'unsupported_error'          # not an UnexpectedToken / InteractiveParser

_KIND_RANK = {INSERT: 0, DELETE: 1}


@dataclass(frozen=True)
class Edit:
    """A single token-level edit.

    Attributes:
        kind: INSERT or DELETE.
        token: For inserts, the virtual token to insert (its value is a
            best-effort guess at a valid lexeme). For deletes, the existing
            token that should be removed.
        start_pos/end_pos: Character span in the original text. Zero-width
            for inserts (the insertion point).
        line/column/end_line/end_column: The same span, in line/column form.
        text: For inserts, the text to insert (None if no concrete
            lexeme could be synthesized for the terminal). For deletes, the
            text that would be removed.
    """
    kind: str
    token: Token
    start_pos: Optional[int]
    end_pos: Optional[int]
    line: Optional[int]
    column: Optional[int]
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    text: Optional[str] = None

    def apply(self, text: str) -> str:
        """Return a new string with this edit applied. Does not mutate text."""
        start = self.start_pos or 0
        if self.kind == INSERT:
            if self.text is None:
                raise ValueError("Cannot apply insert of %r: no concrete lexeme is known for this terminal"
                                 % (self.token.type,))
            return text[:start] + self.text + text[start:]
        end = self.end_pos if self.end_pos is not None else start
        return text[:start] + text[end:]

    def describe(self) -> str:
        where = 'line %s, column %s' % (self.line, self.column)
        if self.kind == INSERT:
            what = repr(self.text) if self.text is not None else 'a %s token' % self.token.type
            return 'Insert %s at %s' % (what, where)
        return 'Delete %r (%s) at %s' % (str(self.token), self.token.type, where)


@dataclass(frozen=True)
class Suggestion:
    """A sequence of edits that would let the parse complete, with its total cost."""
    edits: Tuple[Edit, ...]
    cost: float

    @property
    def kind(self) -> str:
        kinds = {e.kind for e in self.edits}
        return kinds.pop() if len(kinds) == 1 else 'mixed'

    def apply(self, text: str) -> str:
        """Return a new string with all edits applied. Does not mutate text."""
        # Apply from the end of the text backwards, so earlier offsets stay valid.
        ordered = sorted(enumerate(self.edits), key=lambda p: (p[1].start_pos or 0, p[0]), reverse=True)
        for _, edit in ordered:
            text = edit.apply(text)
        return text

    def describe(self) -> str:
        return '; '.join(e.describe() for e in self.edits)

    def _sort_key(self):
        return (self.cost, len(self.edits),
                tuple(_KIND_RANK[e.kind] for e in self.edits),
                tuple(e.token.type for e in self.edits))


@dataclass
class SuggestionResult:
    """The outcome of suggest_fixes.

    Attributes:
        suggestions: Valid fix suggestions, sorted by cost and then by fixed
            rules (fewer edits, inserts before deletes, token type name).
        stop_reason: Why the search stopped. STOP_FOUND when suggestions
            were found; otherwise the reason the search came up empty.
        steps: How many search nodes were expanded.
    """
    suggestions: List[Suggestion] = field(default_factory=list)
    stop_reason: str = STOP_EXHAUSTED
    steps: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.suggestions)


def _terminals_by_name(ip: InteractiveParser) -> Dict[str, object]:
    lexer = ip.lexer_thread.lexer
    terminals = getattr(lexer, 'terminals', None)
    if terminals is None:
        terminals = getattr(getattr(lexer, 'root_lexer', None), 'terminals', None)
    if not terminals:
        return {}
    return {t.name: t for t in terminals}


def _probe_copy(ip: InteractiveParser, conf_no_callbacks) -> InteractiveParser:
    """Copy the interactive parser and disable parse callbacks on the copy.

    The original instance is never fed tokens and never sees a modified
    parse configuration, so probing can't pollute it, and the user's
    transformer callbacks are never invoked during probing.
    """
    probe = ip.copy(deepcopy_values=False)
    probe.parser_state.parse_conf = conf_no_callbacks
    return probe


def _try_complete(ip: InteractiveParser, pending: Token, conf_no_callbacks) -> bool:
    """Return True if feeding pending and the rest of the input finishes the parse."""
    probe = _probe_copy(ip, conf_no_callbacks)
    try:
        if (pending.type == '$END'
                and probe.parser_state.position == probe.parser_state.parse_conf.end_state):
            # The start rule already fully reduced; only the extra token stood in the way.
            return True
        last = pending
        probe.feed_token(pending)
        for token in probe.lexer_thread.lex(probe.parser_state):
            probe.feed_token(token)
            last = token
        if pending.type != '$END':
            probe.feed_eof(last)
    except UnexpectedInput:
        return False
    return True


def _next_token(ip: InteractiveParser) -> Optional[Token]:
    """Pull the next token out of the probe's own lexer copy (None at EOF)."""
    try:
        return next(ip.lexer_thread.lex(ip.parser_state))
    except StopIteration:
        return None


def _insert_text(terminals, token_type: str) -> Optional[str]:
    terminal = terminals.get(token_type)
    if terminal is not None and isinstance(terminal.pattern, PatternStr):
        return terminal.pattern.value
    return None


def _virtual_token(token_type: str, text: Optional[str], before: Token) -> Token:
    if before.type == '$END':
        # '$END' borrows the span of the last real token; an insertion at
        # end-of-input belongs after it, not before it.
        pos, line, column = before.end_pos, before.end_line, before.end_column
    else:
        pos, line, column = before.start_pos, before.line, before.column
    return Token(token_type, text if text is not None else '',
                 start_pos=pos, line=line, column=column,
                 end_line=line, end_column=column, end_pos=pos)


def suggest_fixes(error: Union[UnexpectedToken, InteractiveParser],
                  token: Optional[Token] = None,
                  *,
                  max_steps: int = 200,
                  max_cost: float = 3.0,
                  max_suggestions: int = 3,
                  insert_cost: float = 1.0,
                  delete_cost: float = 1.0) -> SuggestionResult:
    """Search for small token-level fixes for a LALR parse error.

    Parameters:
        error: The UnexpectedToken raised by the parser, or an
            InteractiveParser positioned at the point of failure.
        token: The offending token. Only needed (and only used) when error
            is an InteractiveParser.
        max_steps: Maximum number of search nodes to expand.
        max_cost: Maximum total cost of a suggestion.
        max_suggestions: Maximum number of suggestions to return.
        insert_cost: Cost of inserting one virtual token.
        delete_cost: Cost of deleting one token.

    Returns:
        A SuggestionResult. When no fix is found within the limits,
        suggestions is empty and stop_reason explains why.
    """
    if isinstance(error, UnexpectedToken):
        ip, token = error.interactive_parser, error.token
    elif isinstance(error, InteractiveParser):
        ip = error
    else:
        return SuggestionResult(stop_reason=STOP_UNSUPPORTED)

    if ip is None or token is None:
        return SuggestionResult(stop_reason=STOP_NO_PARSER)

    conf_no_callbacks = copy(ip.parser_state.parse_conf)
    conf_no_callbacks.callbacks = {}
    terminals = _terminals_by_name(ip)

    suggestions: List[Suggestion] = []
    seen_suggestions = set()
    visited = set()
    steps = 0
    cost_limited = False
    counter = count()
    # Heap items: (cost, edit count, tie-breaker, (interactive_parser, pending_token, edits))
    heap: List[Tuple[float, int, int, Tuple[InteractiveParser, Token, Tuple[Edit, ...]]]] = []
    heappush(heap, (0.0, 0, next(counter), (ip, token, ())))

    while heap and steps < max_steps and len(suggestions) < max_suggestions:
        cost, _, _, (node_ip, pending, edits) = heappop(heap)
        key = (tuple(node_ip.parser_state.state_stack), pending.type, pending.start_pos)
        if key in visited:
            continue
        visited.add(key)
        steps += 1

        if edits and _try_complete(node_ip, pending, conf_no_callbacks):
            sig = tuple((e.kind, e.token.type, e.start_pos) for e in edits)
            if sig not in seen_suggestions:
                seen_suggestions.add(sig)
                suggestions.append(Suggestion(edits, cost))
            continue

        min_op_cost = min(insert_cost, delete_cost)
        if cost + min_op_cost > max_cost:
            cost_limited = True
            continue

        # Try inserting each accepted virtual token before the pending token
        if cost + insert_cost <= max_cost:
            for token_type in sorted(node_ip.choices()):
                if not token_type.isupper() or token_type == '$END':
                    continue
                text = _insert_text(terminals, token_type)
                virtual = _virtual_token(token_type, text, pending)
                probe = _probe_copy(node_ip, conf_no_callbacks)
                try:
                    probe.feed_token(virtual)
                except UnexpectedInput:
                    continue
                edit = Edit(INSERT, virtual, virtual.start_pos, virtual.end_pos,
                            virtual.line, virtual.column, virtual.end_line, virtual.end_column, text)
                heappush(heap, (cost + insert_cost, len(edits) + 1, next(counter),
                                (probe, pending, edits + (edit,))))

        # Try deleting the pending token (never delete end-of-input)
        if pending.type != '$END' and cost + delete_cost <= max_cost:
            probe = _probe_copy(node_ip, conf_no_callbacks)
            try:
                nxt = _next_token(probe)
            except UnexpectedInput:
                nxt = None
            if nxt is None:
                nxt = Token.new_borrow_pos('$END', '', pending)
            edit = Edit(DELETE, pending, pending.start_pos, pending.end_pos,
                        pending.line, pending.column, pending.end_line, pending.end_column, str(pending))
            heappush(heap, (cost + delete_cost, len(edits) + 1, next(counter),
                            (probe, nxt, edits + (edit,))))

    suggestions.sort(key=Suggestion._sort_key)
    if suggestions:
        stop_reason = STOP_FOUND
    elif steps >= max_steps:
        stop_reason = STOP_MAX_STEPS
    elif cost_limited:
        stop_reason = STOP_MAX_COST
    else:
        stop_reason = STOP_EXHAUSTED
    return SuggestionResult(suggestions, stop_reason, steps)
