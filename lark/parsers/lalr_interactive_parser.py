# This module provides a LALR interactive parser, which is used for debugging and error handling

from typing import Dict, Iterator, List, Optional, Set, Tuple
from copy import copy
import warnings

from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
from .lalr_analysis import Shift, StateT
from .lalr_parser_state import ParserState

###{standalone

class InteractiveParser:
    """InteractiveParser gives you advanced control over parsing and error handling when parsing with LALR.

    For a simpler interface, see the ``on_error`` argument to ``Lark.parse()``.
    """

    @staticmethod
    def _is_shift(action) -> bool:
        return action == 0 if isinstance(action, int) else action is Shift

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

    def _advance_state_stack_without_callbacks(
            self,
            state_stack: Tuple[StateT, ...],
            terminal: str,
            is_end: bool=False
    ) -> Optional[Tuple[StateT, ...]]:
        states = self.parser_state.parse_conf.states
        end_state = self.parser_state.parse_conf.end_state
        next_stack = list(state_stack)
        reductions: Set[Tuple[Tuple[StateT, ...], str]] = set()
        empty_reductions: Set[Tuple[StateT, str]] = set()

        while True:
            state = next_stack[-1]
            try:
                action, arg = states[state][terminal]
            except KeyError:
                return None

            if self._is_shift(action):
                if is_end:
                    return None
                next_stack.append(arg)
                return tuple(next_stack)

            rule = arg
            size = len(rule.expansion)
            reduce_key = (tuple(next_stack), terminal)
            if reduce_key in reductions:
                return None
            reductions.add(reduce_key)

            if size == 0:
                empty_key = (state, terminal)
                if empty_key in empty_reductions:
                    return None
                empty_reductions.add(empty_key)
            else:
                empty_reductions.clear()

            if len(next_stack) < size:
                return None
            if size:
                del next_stack[-size:]

            try:
                goto_action, new_state = states[next_stack[-1]][rule.origin.name]
            except KeyError:
                return None
            if not self._is_shift(goto_action):
                return None

            next_stack.append(new_state)
            if is_end and next_stack[-1] == end_state:
                return tuple(next_stack)

    def completions(
            self, max_completion_tokens: int=100
    ) -> Dict[str, Optional[Tuple[str, ...]]]:
        """Return a finite completion suffix for each currently accepted terminal.

        The suffix does not include the synthetic ``$END`` terminal. A value of
        ``None`` means that no completion could be proven within
        ``max_completion_tokens`` terminals.

        Only the parser table and state stack are used, so parser callbacks are
        not called.
        """
        initial_stack = tuple(self.parser_state.state_stack)
        completions: Dict[str, Optional[Tuple[str, ...]]] = {}
        advance = self._advance_state_stack_without_callbacks

        for token in self.accepts():
            if token == '$END':
                stack = advance(initial_stack, token, is_end=True)
                completions[token] = () if stack is not None else None
                continue
            else:
                stack = advance(initial_stack, token)

            if stack is None:
                completions[token] = None
                continue

            if advance(stack, '$END', is_end=True) is not None:
                completions[token] = ()
                continue

            queue: List[Tuple[Tuple[StateT, ...], Tuple[str, ...]]] = [(stack, ())]
            queue_head = 0
            visited = {stack}
            suffix: Optional[Tuple[str, ...]] = None

            while queue_head < len(queue) and suffix is None:
                current_stack, current_suffix = queue[queue_head]
                queue_head += 1
                if len(current_suffix) >= max_completion_tokens:
                    continue

                for next_token in self.parser_state.parse_conf.states[current_stack[-1]]:
                    if next_token == '$END' or not next_token.isupper():
                        continue

                    next_stack = advance(current_stack, next_token)
                    if next_stack is None:
                        continue

                    next_suffix = current_suffix + (next_token,)
                    completed = advance(next_stack, '$END', is_end=True)
                    if completed is not None:
                        suffix = next_suffix
                        break

                    if (len(next_suffix) < max_completion_tokens
                            and next_stack not in visited):
                        visited.add(next_stack)
                        queue.append((next_stack, next_suffix))

            completions[token] = suffix

        return completions

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
