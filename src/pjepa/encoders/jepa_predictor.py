"""Predictor and EMA target for self-supervised training.

The :class:`Predictor` produces predicted target embeddings from
context embeddings. The :class:`Target` is an exponential moving
average of an online encoder (Grill et al. 2020) and is updated
by the trainer after each optimisation step.

The :class:`Target` is deliberately *not* a
:class:`torch.nn.Module`: its :meth:`forward` runs under
:func:`torch.no_grad` so the autograd graph never includes the
target branch, which is essential for stable self-supervised
training.

Both classes are polymorphic subclasses of :class:`Head`. The
:class:`Head` base class is the polymorphic root of the head
hierarchy and supplies the shared contract that the trainer
relies on: every head accepts a context tensor and returns a
target-shaped tensor; every head exposes :meth:`update` (a no-op
for :class:`Predictor`).

Example:
    >>> import torch
    >>> from pjepa.encoders.jepa_predictor import Predictor, Target
    >>> pred = Predictor(input_dim=4, hidden_dim=8, output_dim=4)
    >>> online = torch.nn.Linear(4, 4)
    >>> tgt = Target(online=online, momentum=0.5)
    >>> x = torch.randn((2, 4))
    >>> torch.isfinite(pred(x)).all().item()
    True
    >>> torch.isfinite(tgt(x)).all().item()
    True
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch
from torch import nn

from pjepa.exceptions import NumericalError

__all__ = ["Head", "Predictor", "Target"]


class Head(ABC, nn.Module):
    """Abstract base class for head modules.

    The :class:`Head` class is the polymorphic root of the head
    hierarchy. The two concrete subclasses are :class:`Predictor`
    (the learned predictor) and :class:`Target` (the EMA shadow
    encoder, which is not an ``nn.Module`` but wraps one).

    The base class unifies the contract that every head accepts a
    context tensor and returns a target-shaped tensor. The
    :meth:`update` method is declared on the base class so the
    trainer can iterate over a list of heads uniformly; the default
    implementation is a no-op so subclasses that do not need
    state updates (like :class:`Predictor`) do not have to
    override it.
    """

    def __init__(self) -> None:
        """Initialise the underlying :class:`torch.nn.Module`."""
        super().__init__()

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Map a context tensor to a target-shaped tensor.

        Args:
            context: A ``[..., input_dim]`` context tensor.

        Returns:
            A ``[..., output_dim]`` target-shape tensor.
        """
        ...

    def update(self) -> None:
        """Update the head's internal state.

        Default implementation is a no-op. Overridden by
        :class:`Target` to perform the EMA update after every
        optimiser step.
        """
        return


class Predictor(Head):
    """Predictor head that maps context features to predicted target features.

    The predictor is a ``Linear -> GELU -> Linear`` stack. Its
    weights are learned; the target branch never propagates
    gradients through them.

    Attributes:
        input_dim: Dimension of context features.
        hidden_dim: Hidden width of the predictor MLP.
        output_width: Dimension of predicted target features.

    Args:
        input_dim: Dimension of context features.
        hidden_dim: Hidden width of the predictor MLP.
        output_dim: Dimension of predicted target features.

    Raises:
        ValueError: At construction if any dimension is non-positive.

    Example:
        >>> from pjepa.encoders.jepa_predictor import Predictor
        >>> pred = Predictor(input_dim=4, hidden_dim=8, output_dim=4)
        >>> pred.output_dim
        4
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        """Initialise the predictor.

        Args:
            input_dim: Dimension of context features.
            hidden_dim: Hidden width of the predictor MLP.
            output_dim: Dimension of predicted target features.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Predictor: dims must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_width = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    @property
    def output_dim(self) -> int:
        """Return the trailing dimension of the predictor's output.

        Returns:
            The output width stored in ``self.output_width``.
        """
        return int(self.output_width)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict the target embedding from the context embedding.

        Args:
            context: A ``[..., input_dim]`` context tensor.

        Returns:
            A ``[..., output_dim]`` predicted target tensor.
        """
        return self.net(context)


class Target:
    """Exponential moving average of an online encoder.

    The target is a delayed copy of an online encoder whose
    parameters are updated after each training step as::

        theta_target = momentum * theta_target + (1 - momentum) * theta_online

    with the convention that ``momentum`` near ``1`` keeps the
    target close to its previous state. Gradients are always
    disabled on the shadow parameters so they never appear in
    any optimiser step.

    The class is *not* a :class:`torch.nn.Module`; it is a thin
    wrapper that owns a frozen copy of the online encoder. The
    polymorphic :class:`Head` contract is satisfied by providing
    :meth:`forward` and :meth:`update`.

    Attributes:
        online: The live encoder whose gradients are tracked.
        shadow: A :func:`copy.deepcopy` of ``online`` with
            ``requires_grad=False`` on every parameter.
        momentum: The momentum constant above; default ``0.996``.

    Args:
        online: The live encoder.
        momentum: The momentum constant in ``[0, 1]``.

    Raises:
        ValueError: At construction if ``momentum`` is outside
            ``[0, 1]``.
        NumericalError: At update time if the new parameters are
            not finite.

    Example:
        >>> import torch
        >>> from pjepa.encoders.jepa_predictor import Target
        >>> online = torch.nn.Linear(4, 4)
        >>> tgt = Target(online=online, momentum=0.5)
        >>> tgt(torch.randn((2, 4))).shape
        torch.Size([2, 4])
    """

    def __init__(self, online: nn.Module, momentum: float = 0.996) -> None:
        """Initialise the EMA target.

        Args:
            online: The live encoder whose gradients are tracked.
            momentum: The momentum constant in ``[0, 1]``.

        Raises:
            ValueError: If ``momentum`` is outside ``[0, 1]``.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Target: momentum must be in [0, 1]; got {momentum}")
        self.online = online
        self.momentum = momentum
        self.shadow = copy.deepcopy(online)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        """Update the target parameters via EMA.

        Complexity is ``O(P)`` where ``P`` is the number of
        scalar parameters in ``online``. The update is
        performed in place on ``self.shadow`` and never
        allocates new tensors.

        Returns:
            ``None``. The method mutates ``self.shadow`` in
            place.

        Raises:
            NumericalError: If any updated parameter is non-finite.
        """
        for online_param, shadow_param in zip(self.online.parameters(), self.shadow.parameters()):
            new_value = (
                self.momentum * shadow_param.data + (1.0 - self.momentum) * online_param.data
            )
            if not torch.isfinite(new_value).all():
                raise NumericalError("Target.update: produced non-finite parameters")
            shadow_param.data.copy_(new_value)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Forward through the target encoder without gradients.

        The return type is intentionally untyped because the
        shadow network can be any :class:`torch.nn.Module`.

        Args:
            *args: Positional arguments forwarded to
                ``self.shadow``.
            **kwargs: Keyword arguments forwarded to
                ``self.shadow``.

        Returns:
            Whatever the wrapped encoder returns.
        """
        with torch.no_grad():
            return self.shadow(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward through the target encoder.

        The :class:`Target` class is not a :class:`torch.nn.Module`,
        so :meth:`__call__` is wired manually to dispatch to
        :meth:`forward`. This lets callers treat a :class:`Target`
        instance as a callable encoder (the same way they would
        treat a module instance).

        Args:
            *args: Positional arguments forwarded to
                :meth:`forward`.
            **kwargs: Keyword arguments forwarded to
                :meth:`forward`.

        Returns:
            Whatever the wrapped encoder returns.
        """
        return self.forward(*args, **kwargs)
