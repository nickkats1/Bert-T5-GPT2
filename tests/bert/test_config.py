from headlines.bert.config import CONFIG, BertConfig


class TestBertConfig:
    """Validate the default ``BertConfig`` values."""

    def test_is_dataclass_instance(self):
        assert isinstance(CONFIG, BertConfig)

    def test_model_name(self):
        assert CONFIG.model_name == "bert-base-uncased"

    def test_epochs(self):
        assert isinstance(CONFIG.epochs, int)
        assert CONFIG.epochs > 0

    def test_learning_rate(self):
        assert isinstance(CONFIG.learning_rate, float)
        assert 0 < CONFIG.learning_rate < 1

    def test_data_path(self):
        assert CONFIG.data_path == "data/guardian_headlines.csv"

    def test_max_length(self):
        assert isinstance(CONFIG.max_length, int)
        assert CONFIG.max_length >= 16

    def test_batch_size(self):
        assert isinstance(CONFIG.batch_size, int)
        assert CONFIG.batch_size > 0

    def test_device(self):
        assert CONFIG.device in {"cuda", "cpu", "cuda:0"}

    def test_split_sizes(self):
        assert CONFIG.holdout_size == 0.50
        assert CONFIG.test_size_from_holdout == 0.20

    def test_immutable(self):
        """Frozen dataclass: cannot reassign fields."""
        import dataclasses

        try:
            CONFIG.epochs = 999  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("BertConfig should be frozen")
