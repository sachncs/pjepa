"""Hyperedge-replacement grammar (HRG).

A hyperedge-replacement grammar is the rewrite class chosen by the
framework (see the paper, §7.6.1). Productions replace a single
non-terminal hyperedge with a hypergraph, which makes DPO-style
verification well-defined and supports both edge substitution and
node merging via hyperedge fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from pjepa.exceptions import GraphError

__all__ = ["HRG", "HRGProduction"]


@dataclass(frozen=True)
class HRGProduction:
    """A single production rule ``X -> R`` in the grammar.

    Attributes:
        lhs: The non-terminal label on the left-hand side.
        rhs_edge_features: A ``[E_rhs, d_e]`` tensor describing the
          right-hand-side edges.
        rhs_edge_index: A ``[2, E_rhs]`` ``long`` tensor in COO format
          describing the right-hand-side hypergraph connectivity.

    Example:
        >>> prod = HRGProduction(lhs="Stmt", rhs_edge_index=torch.zeros((2, 0), dtype=torch.long), rhs_edge_features=torch.zeros((0, 1)))
    """

    lhs: str
    rhs_edge_index: torch.Tensor
    rhs_edge_features: torch.Tensor

    def __post_init__(self) -> None:
        if not self.lhs:
            raise GraphError("HRGProduction: lhs must be a non-empty string")
        if self.rhs_edge_index.dtype != torch.long:
            raise GraphError("HRGProduction: rhs_edge_index must be long")


@dataclass(frozen=True)
class HRG:
    """A hyperedge-replacement grammar ``(N, T, P, S)``.

    Attributes:
        nonterminals: Tuple of non-terminal labels.
        terminals: Tuple of terminal labels.
        productions: Tuple of :class:`HRGProduction` rules.
        start: The start non-terminal.

    Example:
        >>> hrg = HRG(nonterminals=("Stmt",), terminals=("Tok",), productions=(), start="Stmt")
    """

    nonterminals: tuple[str, ...]
    terminals: tuple[str, ...]
    productions: tuple[HRGProduction, ...]
    start: str

    def __post_init__(self) -> None:
        if not self.nonterminals:
            raise GraphError("HRG: at least one non-terminal is required")
        if self.start not in self.nonterminals:
            raise GraphError(f"HRG: start {self.start!r} is not in nonterminals")
        labels = set(self.nonterminals) | set(self.terminals)
        if len(labels) != len(self.nonterminals) + len(self.terminals):
            raise GraphError("HRG: non-terminal and terminal labels overlap")
        for prod in self.productions:
            if prod.lhs not in self.nonterminals:
                raise GraphError(
                    f"HRG: production lhs {prod.lhs!r} is not a non-terminal"
                )

    def productions_for(self, label: str) -> tuple[HRGProduction, ...]:
        """Return every production whose left-hand side is ``label``."""
        if label not in self.nonterminals and label not in self.terminals:
            raise GraphError(f"HRG.productions_for: unknown label {label!r}")
        return tuple(p for p in self.productions if p.lhs == label)

    def is_nonterminal(self, label: str) -> bool:
        """Return whether ``label`` is a non-terminal in this grammar."""
        return label in self.nonterminals

    def is_terminal(self, label: str) -> bool:
        """Return whether ``label`` is a terminal in this grammar."""
        return label in self.terminals