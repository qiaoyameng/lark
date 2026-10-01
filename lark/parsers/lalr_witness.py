"""Token-level witnesses for LALR shift/reduce conflicts.

When strict-mode LALR construction hits a shift/reduce conflict, this module
searches the LR(0) automaton for a token prefix that drives the parser into
the conflicting state, and reports it together with the triggering lookahead
and the two competing items (the item that would shift, and the item that
would reduce).

The witness is *token-level*: it is a sequence of terminal names, not source
text.  Concrete example lexemes are only reported when every terminal involved
is defined by a plain string pattern; otherwise no lexemes are fabricated.

All searches are bounded.  When a bound is exceeded (or no witness can be
found or verified), ``build_shift_reduce_witness`` returns ``None`` and the
caller falls back to the plain conflict message.
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..grammar import Terminal, NonTerminal, Rule, Symbol
from .grammar_analysis import LR0ItemSet, RulePtr

# Search bounds.  Exceeding any of them aborts witness generation; the
# original conflict information is still reported by the caller.
MAX_WITNESS_PREFIX_TOKENS = 40    # max tokens in the reported prefix
MAX_DERIVATION_TOKENS = 16        # max tokens when expanding one nonterminal
MAX_SEARCH_STATES = 100000        # max LR(0) states visited by the BFS
MAX_DERIVATION_PASSES = 10000     # max fixpoint passes for derivations
MAX_REPLAY_STEPS = 100000         # max parser moves during witness replay


class ConflictWitness:
    """A token-level witness for a shift/reduce conflict.

    Attributes:
        prefix: Token names that lead the parser to the conflict state.
        lookahead: The token name on which shift and reduce compete.
        shift_items: Items in the conflict state that shift ``lookahead``.
        reduce_item: The satisfied item whose rule would be reduced.
        example_lexemes: Concrete lexemes for ``prefix`` + ``lookahead``,
            or ``None`` when they cannot be derived from string patterns.
    """

    def __init__(self,
                 prefix: Sequence[str],
                 lookahead: str,
                 shift_items: Sequence[RulePtr],
                 reduce_item: RulePtr,
                 example_lexemes: Optional[Sequence[str]] = None):
        self.prefix = tuple(prefix)
        self.lookahead = lookahead
        self.shift_items = tuple(shift_items)
        self.reduce_item = reduce_item
        self.example_lexemes = tuple(example_lexemes) if example_lexemes is not None else None
        # Context needed to replay the witness (see verify()).
        self._analyzer = None
        self._start_itemset: Optional[LR0ItemSet] = None
        self._conflict_itemset: Optional[LR0ItemSet] = None

    def format(self) -> str:
        lines = ['Conflict witness (token-level; token names, not source text):']
        if self.prefix:
            lines.append('  Token prefix to conflict state: %s' % ' '.join(self.prefix))
        else:
            lines.append('  Token prefix to conflict state: (empty; conflict is in the start state)')
        lines.append('  Lookahead triggering the conflict: %s' % self.lookahead)
        lines.append('  Competing items in the conflict state:')
        for item in self.shift_items:
            lines.append('    Shift : %s' % (item,))
        lines.append('    Reduce: %s' % (self.reduce_item,))
        if self.example_lexemes is not None:
            lexemes = ' '.join(repr(x) for x in self.example_lexemes)
            lines.append('  Example lexemes for the prefix + lookahead: %s' % lexemes)
        return '\n'.join(lines)

    def verify(self) -> bool:
        """Replay the witness through the LALR automaton.

        Returns True iff replaying ``prefix`` from the start state, with
        conflicts resolved as shift, arrives at the conflict state where both
        the shift and the reduce action apply to ``lookahead``.
        """
        if self._analyzer is None or self._start_itemset is None or self._conflict_itemset is None:
            return False
        return replay_witness(self._analyzer, self._start_itemset, self)


def _terminals_by_name(analyzer) -> Dict[str, Terminal]:
    terminals: Dict[str, Terminal] = {'$END': Terminal('$END')}
    for rules in analyzer.lr0_rules_by_origin.values():
        for rule in rules:
            for sym in rule.expansion:
                if sym.is_term:
                    terminals.setdefault(sym.name, sym)
    return terminals


def _bfs_path_to_state(analyzer, target: LR0ItemSet) -> Optional[Tuple[LR0ItemSet, List[Symbol]]]:
    """Find a shortest symbol path from any start state to ``target``.

    Returns ``(start_itemset, symbols)`` or ``None`` when the state is not
    reachable within ``MAX_SEARCH_STATES`` visited states.
    """
    starts = list(analyzer.lr0_start_states.values())
    parents: Dict[int, Tuple[LR0ItemSet, Symbol]] = {}
    visited: Set[int] = {id(s) for s in starts}
    queue = deque(starts)
    while queue:
        state = queue.popleft()
        if state is target:
            symbols: List[Symbol] = []
            node = state
            while id(node) in parents:
                prev, sym = parents[id(node)]
                symbols.append(sym)
                node = prev
            symbols.reverse()
            return node, symbols
        if len(visited) >= MAX_SEARCH_STATES:
            return None
        for sym, next_state in state.transitions.items():
            if id(next_state) not in visited:
                visited.add(id(next_state))
                parents[id(next_state)] = (state, sym)
                queue.append(next_state)
    return None


def _minimal_terminal_derivations(analyzer) -> Dict[NonTerminal, Tuple[Terminal, ...]]:
    """Compute a shortest terminal derivation for each nonterminal.

    Derivations longer than ``MAX_DERIVATION_TOKENS`` are treated as
    unavailable, keeping the search bounded.
    """
    derivs: Dict[NonTerminal, Tuple[Terminal, ...]] = {}
    for _ in range(MAX_DERIVATION_PASSES):
        changed = False
        for nt, rules in analyzer.lr0_rules_by_origin.items():
            best = derivs.get(nt)
            for rule in rules:
                candidate: List[Terminal] = []
                for sym in rule.expansion:
                    if sym.is_term:
                        candidate.append(sym)
                    elif sym in derivs:
                        candidate.extend(derivs[sym])
                    else:
                        break
                    if len(candidate) > MAX_DERIVATION_TOKENS:
                        break
                else:
                    if best is None or len(candidate) < len(best):
                        best = tuple(candidate)
                        derivs[nt] = best
                        changed = True
        if not changed:
            break
    return derivs


def _resolve_reduce(rules: Set[Rule]) -> Rule:
    "Mirror the conflict resolution of compute_lalr1_states (priority, then deterministic)."
    by_priority = sorted(rules, key=lambda r: (r.options.priority or 0), reverse=True)
    best = by_priority[0]
    if len(by_priority) > 1 and (by_priority[1].options.priority or 0) == (best.options.priority or 0):
        return min(rules, key=str)
    return best


def replay_witness(analyzer, start_itemset: LR0ItemSet, witness: ConflictWitness) -> bool:
    """Replay ``witness`` through the LALR automaton (conflicts resolve as shift).

    Simulates the parser on ``prefix`` followed by ``lookahead`` and returns
    True iff the conflict point is genuinely reached: the conflict state is on
    top of the stack, the lookahead both shifts there and is a valid reduce
    lookahead for the witness' reduce rule.
    """
    terminals = _terminals_by_name(analyzer)
    try:
        tokens = [terminals[name] for name in witness.prefix]
        lookahead = terminals[witness.lookahead]
    except KeyError:
        return False

    conflict_state = witness._conflict_itemset
    reduce_rule = witness.reduce_item.rule
    stack: List[LR0ItemSet] = [start_itemset]
    steps = 0
    for i, token in enumerate(tokens + [lookahead]):
        is_lookahead = i == len(tokens)
        while True:
            steps += 1
            if steps > MAX_REPLAY_STEPS:
                return False
            state = stack[-1]
            if (is_lookahead and state is conflict_state
                    and token in state.transitions
                    and reduce_rule in state.lookaheads.get(token, ())):
                return True
            if token in state.transitions:
                stack.append(state.transitions[token])
                break
            rules = state.lookaheads.get(token)
            if not rules:
                return False
            rule = _resolve_reduce(rules)
            if rule.expansion:
                if len(rule.expansion) >= len(stack):
                    return False
                del stack[-len(rule.expansion):]
            goto = stack[-1].transitions.get(rule.origin)
            if goto is None:
                return False
            stack.append(goto)
    return False


def _example_lexemes(token_names: Sequence[str], terminals_by_name) -> Optional[List[str]]:
    "Concrete lexemes, only when every token is a plain string pattern."
    lexemes = []
    for name in token_names:
        term_def = terminals_by_name.get(name)
        if term_def is None or term_def.pattern.type != 'str':
            return None
        lexemes.append(term_def.pattern.value)
    return lexemes


def build_shift_reduce_witness(analyzer, itemset: LR0ItemSet, terminal: Terminal,
                               rule: Rule, terminals_by_name=None) -> Optional[ConflictWitness]:
    """Build (and verify) a token-level witness for a shift/reduce conflict.

    ``itemset`` is the conflict state, ``terminal`` the competing lookahead
    and ``rule`` the rule that would be reduced.  Returns ``None`` when no
    bounded, verifiable witness can be produced.
    """
    found = _bfs_path_to_state(analyzer, itemset)
    if found is None:
        return None
    start_itemset, symbols = found

    derivs = _minimal_terminal_derivations(analyzer)
    prefix: List[str] = []
    for sym in symbols:
        if sym.is_term:
            prefix.append(sym.name)
        else:
            derivation = derivs.get(sym)
            if derivation is None:
                return None
            prefix.extend(t.name for t in derivation)
        if len(prefix) > MAX_WITNESS_PREFIX_TOKENS:
            return None

    shift_items = sorted(
        (rp for rp in itemset.closure if not rp.is_satisfied and rp.next == terminal),
        key=str)
    if not shift_items:
        return None
    reduce_item = RulePtr(rule, len(rule.expansion))

    lexemes = None
    if terminals_by_name:
        lexemes = _example_lexemes(list(prefix) + [terminal.name], terminals_by_name)

    witness = ConflictWitness(prefix, terminal.name, shift_items, reduce_item, lexemes)
    witness._analyzer = analyzer
    witness._start_itemset = start_itemset
    witness._conflict_itemset = itemset

    # Only report witnesses that genuinely replay to the conflict point.
    if not witness.verify():
        return None
    return witness
