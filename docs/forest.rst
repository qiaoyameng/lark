Working with the SPPF
=====================

When parsing with Earley, Lark provides the ``ambiguity='forest'`` option
to obtain the shared packed parse forest (SPPF) produced by the parser as
an alternative to it being automatically converted to a tree.

Lark provides a few tools to facilitate working with the SPPF. Here are some
things to consider when deciding whether or not to use the SPPF.

**Pros**

- Efficient storage of highly ambiguous parses
- Precise handling of ambiguities
- Custom rule prioritizers
- Ability to handle infinite ambiguities
- Directly transform forest -> object instead of forest -> tree -> object

**Cons**

- More complex than working with a tree
- SPPF may contain nodes corresponding to rules generated internally
- Loss of Lark grammar features:

  - Rules starting with '_' are not inlined in the SPPF
  - Rules starting with '?' are never inlined in the SPPF
  - All tokens will appear in the SPPF

SymbolNode
----------

.. autoclass:: lark.parsers.earley_forest.SymbolNode
   :members: is_ambiguous, children

PackedNode
----------

.. autoclass:: lark.parsers.earley_forest.PackedNode
   :members: children

ForestVisitor
-------------

.. autoclass:: lark.parsers.earley_forest.ForestVisitor
   :members: visit, visit_symbol_node_in, visit_symbol_node_out,
             visit_packed_node_in, visit_packed_node_out,
             visit_token_node, on_cycle, get_cycle_in_path

ForestTransformer
-----------------

.. autoclass:: lark.parsers.earley_forest.ForestTransformer
   :members: transform, transform_symbol_node, transform_intermediate_node,
             transform_packed_node, transform_token_node

TreeForestTransformer
---------------------

.. autoclass:: lark.parsers.earley_forest.TreeForestTransformer
   :members: __default__, __default_token__, __default_ambig__

TopKDerivations
---------------

When a forest represents many parse trees, ``top_k_derivations`` lazily
enumerates the lowest-cost derivations instead of materializing the whole set.
Costs are assigned to production rules with a mapping or callable. All costs
must be non-negative.

The returned derivations remain shared DAGs; shared subtrees are reused by
reference. ``Derivation.to_tree()`` materializes one derivation as an ordinary
parse tree and copies repeated occurrences. Ties use packed-family order and
the left-to-right ranks of child derivations, so repeated enumeration is stable.

Cyclic forests represent infinitely many derivations. By default a
``ForestCycleError`` is raised; pass ``on_cycle='skip'`` to return only finite,
acyclic derivations. Use ``close()`` or the iterator as a context manager to
release buffered candidates when stopping early.

.. autofunction:: lark.parsers.earley_forest.top_k_derivations
.. autoclass:: lark.parsers.earley_forest.TopKDerivations
   :members: close, expanded_nodes
.. autoclass:: lark.parsers.earley_forest.Derivation
   :members: cost, rule, children, to_tree
.. autoclass:: lark.parsers.earley_forest.ForestCycleError

handles_ambiguity
-----------------

.. autofunction:: lark.parsers.earley_forest.handles_ambiguity
