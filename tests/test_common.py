import json

import pytest
import torch
import torch.nn as nn

from headlines.common import (
    EarlyStopping,
    count_parameters,
    get_logger,
    make_generator,
    resolve_device,
    save_json,
    seed_everything,
    seed_worker,
)


class TestSeedEverything:
    def test_reproducible(self):
        seed_everything(123)
        a = torch.rand(5)
        seed_everything(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            seed_everything(-1)


class TestResolveDevice:
    def test_default_is_valid(self):
        assert resolve_device().type in {"cuda", "cpu"}

    def test_cpu_explicit(self):
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        if not torch.cuda.is_available():
            assert resolve_device("cuda").type == "cpu"


class TestCountParameters:
    def test_counts_trainable(self):
        model = nn.Linear(4, 3)  # 4*3 weights + 3 bias = 15
        assert count_parameters(model) == 15

    def test_frozen_excluded(self):
        model = nn.Linear(4, 3)
        for p in model.parameters():
            p.requires_grad_(False)
        assert count_parameters(model, trainable_only=True) == 0
        assert count_parameters(model, trainable_only=False) == 15


class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        path = save_json({"b": 2, "a": 1}, tmp_path / "sub" / "metrics.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == {"a": 1, "b": 2}


def test_get_logger_single_handler():
    logger = get_logger("test.common")
    logger2 = get_logger("test.common")
    assert logger is logger2
    assert len(logger.handlers) == 1


class TestMakeGenerator:
    def test_reproducible_shuffle(self):
        g1 = make_generator(7)
        g2 = make_generator(7)
        a = torch.randperm(10, generator=g1)
        b = torch.randperm(10, generator=g2)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        a = torch.randperm(50, generator=make_generator(1))
        b = torch.randperm(50, generator=make_generator(2))
        assert not torch.equal(a, b)


class TestSeedWorker:
    def test_runs_and_is_deterministic(self):
        import random

        import numpy as np

        seed_everything(0)
        seed_worker(0)
        first = (random.random(), float(np.random.rand()))
        seed_everything(0)
        seed_worker(0)
        second = (random.random(), float(np.random.rand()))
        assert first == second


class TestEarlyStopping:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="lower")

    def test_first_step_is_improvement(self):
        es = EarlyStopping(patience=2, mode="min")
        assert es.step(1.0) is True
        assert es.best == 1.0
        assert es.should_stop is False

    def test_min_mode_stops_after_patience(self):
        es = EarlyStopping(patience=2, mode="min")
        assert es.step(1.0) is True  # best
        assert es.step(2.0) is False  # bad 1
        assert es.should_stop is False
        assert es.step(2.0) is False  # bad 2 -> stop
        assert es.should_stop is True

    def test_max_mode_tracks_higher(self):
        es = EarlyStopping(patience=1, mode="max")
        assert es.step(0.5) is True
        assert es.step(0.7) is True  # improvement resets counter
        assert es.should_stop is False
        assert es.step(0.6) is False  # no improvement -> stop (patience=1)
        assert es.should_stop is True

    def test_min_delta_requires_real_improvement(self):
        es = EarlyStopping(patience=1, mode="min", min_delta=0.1)
        assert es.step(1.0) is True
        # 0.95 is better but not by min_delta, so it does not count.
        assert es.step(0.95) is False
        assert es.should_stop is True
        assert es.best == 1.0

    def test_improvement_resets_bad_epoch_counter(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0)
        es.step(1.5)  # bad 1
        assert es.num_bad_epochs == 1
        assert es.step(0.5) is True  # improvement resets
        assert es.num_bad_epochs == 0
        assert es.should_stop is False
