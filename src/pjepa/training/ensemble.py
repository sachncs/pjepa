"""Model ensemble with three aggregation strategies.

A k-model ensemble runs each model on the input and combines
predictions via:

* ``soft_vote`` (default): mean of logits, then argmax.
* ``hard_vote``: mode of per-model argmax.
* ``rank_avg``: average of per-model ranks.
"""

from __future__ import annotations

from collections import Counter

import torch

from pjepa.exceptions import ConfigError

__all__ = ["Aggregator", "Ensemble"]


class Aggregator:
    """Enumeration of ensemble aggregation strategies."""

    SOFT_VOTE = "soft_vote"
    HARD_VOTE = "hard_vote"
    RANK_AVG = "rank_avg"


class Ensemble(torch.nn.Module):
    """A k-model ensemble.

    Attributes:
        models: The list of models to ensemble.
        aggregator: One of :class:`Aggregator`'s values.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        aggregator: str = Aggregator.SOFT_VOTE,
    ) -> None:
        super().__init__()
        if len(models) == 0:
            raise ConfigError("Ensemble: at least one model is required")
        if aggregator not in (Aggregator.SOFT_VOTE, Aggregator.HARD_VOTE, Aggregator.RANK_AVG):
            raise ConfigError(f"Ensemble: unknown aggregator {aggregator!r}")
        self.models = torch.nn.ModuleList(models)
        self.aggregator = aggregator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ensemble and return the aggregated output.

        Args:
            x: The input tensor.

        Returns:
            For ``soft_vote``: the mean of per-model logits.
            For ``hard_vote``: the per-sample mode of per-model argmax.
            For ``rank_avg``: the per-sample mean of per-model ranks.
        """
        if self.aggregator == Aggregator.SOFT_VOTE:
            logits = torch.stack([m(x) for m in self.models], dim=0)
            return logits.mean(dim=0)
        if self.aggregator == Aggregator.HARD_VOTE:
            preds = torch.stack([m(x).argmax(dim=-1) for m in self.models], dim=0)
            # Per-sample mode; ties broken by smallest index.
            result = []
            for sample_idx in range(preds.shape[1]):
                counts = Counter(preds[:, sample_idx].tolist())
                result.append(counts.most_common(1)[0][0])
            return torch.tensor(result, dtype=torch.long, device=x.device)
        # RANK_AVG
        ranks = []
        for m in self.models:
            logits = m(x)
            order = torch.argsort(logits, dim=-1, descending=True)
            sample_ranks = torch.argsort(order, dim=-1).float()
            ranks.append(sample_ranks)
        return torch.stack(ranks, dim=0).mean(dim=0)
