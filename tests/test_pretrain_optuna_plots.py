"""Tests for pjepa.training.pretrain and pjepa.eval.plots additions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from pjepa.augmentations import AugmentationPipeline, DropEdge, TensorDropFeature
from pjepa.encoders import JEPAPredictor, TargetEncoder
from pjepa.eval.plots import plot_heatmap, plot_radar, render_svg_fallback
from pjepa.exceptions import ConfigError
from pjepa.training import (
    OPTUNA_SEARCH_SPACE,
    OptunaSearch,
    OptunaSearchConfig,
    PretrainConfig,
    augmentation_call,
    build_augmentation_from_name,
    enable_sqlite_wal,
    load_best_config,
    pretrain_loop,
    suggest_hyperparameters,
)

__all__ = [
    "test_bad_augmentation_call_rejects_pipeline",
    "test_bad_augmentation_call_shape_mismatch",
    "test_bad_pretrain_negative_val_every",
    "test_bad_pretrain_unknown_augmentation",
    "test_build_augmentation_from_name_known",
    "test_build_augmentation_from_name_unknown",
    "test_build_augmentation_none",
    "test_enable_sqlite_wal_creates_file",
    "test_happy_augmentation_call_drops_features",
    "test_happy_augmentation_call_identity_like",
    "test_happy_pretrain_loop_with_augmentation",
    "test_happy_pretrain_loop_with_cadence_early_stop",
    "test_happy_pretrain_loop_with_validation_callback",
    "test_load_best_config_missing_file",
    "test_load_best_config_round_trip_yaml",
    "test_optuna_search_evaluate_smoke",
    "test_optuna_search_run_smoke",
    "test_optuna_search_space_keys",
    "test_plot_heatmap_writes_file",
    "test_plot_radar_invalid_dimensions",
    "test_plot_radar_writes_file",
    "test_render_svg_fallback_writes_file",
    "test_render_svg_fallback_wrong_extension",
    "test_suggest_hyperparameters_returns_all_keys",
    "test_suggest_hyperparameters_uses_space",
]


class _ToyEncoder(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _toy_batches(num_batches: int = 2, batch_size: int = 4, dim: int = 4):
    for _ in range(num_batches):
        yield torch.randn(batch_size, dim), torch.randn(batch_size, dim)


def test_happy_augmentation_call_drops_features() -> None:
    aug = TensorDropFeature(strength=0.5, generator=torch.Generator().manual_seed(0))
    x = torch.ones((3, 4))
    out = augmentation_call(aug, x)
    assert out.shape == x.shape
    assert (out == 0.0).any(dim=0).sum() >= 1


def test_bad_augmentation_call_rejects_pipeline() -> None:
    pipeline = AugmentationPipeline([DropEdge(strength=0.1)], mode="sequential", k=1)
    with pytest.raises(ConfigError):
        augmentation_call(pipeline, torch.zeros((2, 2)))


def test_bad_augmentation_call_shape_mismatch() -> None:
    class _BadAug:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:1]

    with pytest.raises(ConfigError):
        augmentation_call(_BadAug(), torch.zeros((3, 4)))


def test_happy_augmentation_call_identity_like() -> None:
    out = augmentation_call(lambda x: x, torch.zeros((2, 3)))
    assert out.shape == (2, 3)


def test_happy_pretrain_loop_with_augmentation() -> None:
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    aug = TensorDropFeature(strength=0.3, generator=torch.Generator().manual_seed(1))
    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=2),
        config=PretrainConfig(epochs=2, checkpoint_dir=tempfile.mkdtemp(), log_every=0),
        augmentation=aug,
    )
    assert len(losses) == 2
    assert all(loss >= 0.0 for loss in losses)


def test_happy_pretrain_loop_with_validation_callback() -> None:
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    val_calls: list[int] = []

    def val_fn(enc, pred, epoch):
        val_calls.append(epoch)
        return 0.5 + 0.01 * epoch

    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=1),
        config=PretrainConfig(
            epochs=4, checkpoint_dir=tempfile.mkdtemp(), val_every=2, log_every=0
        ),
        val_fn=val_fn,
    )
    assert len(losses) == 4
    assert val_calls == [2, 4]


def test_happy_pretrain_loop_with_cadence_early_stop() -> None:
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    class _AlwaysSleep:
        def __init__(self) -> None:
            self.calls = 0

        def should_sleep(self) -> bool:
            self.calls += 1
            return True

    cadence = _AlwaysSleep()
    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=1),
        config=PretrainConfig(
            epochs=5, checkpoint_dir=tempfile.mkdtemp(), cadence=cadence, log_every=0
        ),
    )
    assert len(losses) == 1
    assert cadence.calls == 1


def test_bad_pretrain_negative_val_every() -> None:
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    with pytest.raises(ConfigError):
        pretrain_loop(
            encoder=encoder,
            predictor=predictor,
            target=target,
            optimizer=optimizer,
            batches=_toy_batches(num_batches=1),
            config=PretrainConfig(epochs=1, val_every=-1),
        )


def test_bad_pretrain_unknown_augmentation() -> None:
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    with pytest.raises(ConfigError):
        pretrain_loop(
            encoder=encoder,
            predictor=predictor,
            target=target,
            optimizer=optimizer,
            batches=_toy_batches(num_batches=1),
            config=PretrainConfig(epochs=1, augmentation="bogus"),
        )


class _FakeTrial:
    def __init__(self, params: dict[str, object]) -> None:
        self._params = params
        self.number = 0

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return float(self._params.get(name, (low + high) / 2))

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(self._params.get(name, (low + high) // 2))

    def suggest_categorical(self, name: str, choices: list) -> object:
        return self._params.get(name, choices[0])


def test_optuna_search_space_keys() -> None:
    expected = {
        "lr",
        "weight_decay",
        "hidden_dim",
        "num_layers",
        "dropout",
        "B",
        "beta_ib",
        "lambda_mdl",
        "gamma_forward",
        "ema_momentum",
        "augmentation",
        "label_smoothing",
    }
    assert set(OPTUNA_SEARCH_SPACE.keys()) == expected


def test_suggest_hyperparameters_returns_all_keys() -> None:
    params = suggest_hyperparameters(_FakeTrial({}))
    assert set(params.keys()) == set(OPTUNA_SEARCH_SPACE.keys())


def test_suggest_hyperparameters_uses_space() -> None:
    space = {
        "alpha": {"type": "uniform", "low": 0.0, "high": 1.0},
        "kind": {"type": "categorical", "choices": ["a", "b"]},
    }
    params = suggest_hyperparameters(_FakeTrial({}), space)
    assert params["kind"] in ("a", "b")


def test_enable_sqlite_wal_creates_file(tmp_path: Path) -> None:
    path = enable_sqlite_wal(tmp_path / "optuna.db")
    assert path.exists()
    import sqlite3

    with sqlite3.connect(str(path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_load_best_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_best_config(tmp_path / "does_not_exist.yaml")


def test_load_best_config_round_trip_yaml(tmp_path: Path) -> None:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyYAML not installed")
    payload = {
        "dataset": "MUTAG",
        "best_params": {"lr": 0.01, "hidden_dim": 64, "augmentation": "dropfeat"},
        "best_value": 0.82,
    }
    target = tmp_path / "best_config.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_best_config(target)
    assert loaded["best_value"] == pytest.approx(0.82)
    assert loaded["best_params"]["hidden_dim"] == 64


def test_build_augmentation_none() -> None:
    assert build_augmentation_from_name("none") is None
    assert build_augmentation_from_name("") is None
    assert build_augmentation_from_name(None) is None


def test_build_augmentation_from_name_known() -> None:
    assert isinstance(build_augmentation_from_name("dropfeat"), TensorDropFeature)
    assert isinstance(build_augmentation_from_name("dropedge"), DropEdge)
    pipeline = build_augmentation_from_name("composite")
    assert isinstance(pipeline, AugmentationPipeline)


def test_build_augmentation_from_name_unknown() -> None:
    with pytest.raises(ConfigError):
        build_augmentation_from_name("mystery")


def test_optuna_search_evaluate_smoke() -> None:
    from pjepa.graphs import TypedAttributedGraph

    search = OptunaSearch(OptunaSearchConfig(n_trials=1, epochs=2))
    pairs = []
    for i in range(6):
        g = TypedAttributedGraph(
            vertex_features=torch.randn((4, 3)),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        )
        pairs.append((g, i % 2))
    acc = search.evaluate(
        params={
            "lr": 1e-2,
            "weight_decay": 1e-4,
            "hidden_dim": 32,
            "num_layers": 2,
            "B": 16,
            "label_smoothing": 0.0,
        },
        train_pairs=pairs[:4],
        test_pairs=pairs[4:],
        num_classes=2,
    )
    assert 0.0 <= acc <= 1.0


def test_optuna_search_run_smoke(tmp_path: Path) -> None:
    from pjepa.graphs import TypedAttributedGraph

    pairs = []
    for i in range(6):
        g = TypedAttributedGraph(
            vertex_features=torch.randn((4, 3)),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        )
        pairs.append((g, i % 2))
    train, test = pairs[:4], pairs[4:]
    cfg = OptunaSearchConfig(
        study_name="test-study",
        storage_path=str(tmp_path),
        n_trials=2,
        epochs=1,
        random_seed=0,
    )
    search = OptunaSearch(cfg)
    summary = search.run("MUTAG", train, test, num_classes=2)
    assert summary["n_trials"] >= 1
    assert Path(summary["best_config_path"]).exists()


def test_plot_radar_writes_file(tmp_path: Path) -> None:
    out = plot_radar(
        method_means={"A": [0.6, 0.7, 0.8], "B": [0.7, 0.6, 0.75]},
        datasets=["D1", "D2", "D3"],
        output_path=tmp_path / "radar.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_radar_fallback_bar_with_two_datasets(tmp_path: Path) -> None:
    out = plot_radar(
        method_means={"A": [0.6, 0.7], "B": [0.5, 0.8]},
        datasets=["D1", "D2"],
        output_path=tmp_path / "bar.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_radar_invalid_dimensions() -> None:
    with pytest.raises(ConfigError):
        plot_radar(method_means={"A": [0.1]}, datasets=["D1", "D2", "D3"], output_path="/tmp/x.png")


def test_plot_heatmap_writes_file(tmp_path: Path) -> None:
    out = plot_heatmap(
        matrix=[[0.6, 0.7], [0.8, 0.65]],
        row_labels=["D1", "D2"],
        col_labels=["M1", "M2"],
        output_path=tmp_path / "heat.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_svg_fallback_writes_file(tmp_path: Path) -> None:
    out = render_svg_fallback(40, 30, output_path=tmp_path / "x.svg")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text


def test_render_svg_fallback_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        render_svg_fallback(40, 30, output_path=tmp_path / "x.png")
