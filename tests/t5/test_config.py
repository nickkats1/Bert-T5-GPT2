from headlines.t5.config import CONFIG, T5Config


class TestT5Config:
    """Validate the default ``T5Config`` values."""

    def test_is_dataclass_instance(self):
        assert isinstance(CONFIG, T5Config)

    def test_source_length(self):
        assert CONFIG.source_length == 128

    def test_target_length(self):
        assert CONFIG.target_length == 32

    def test_batch_size(self):
        assert CONFIG.batch_size == 12

    def test_epochs(self):
        assert CONFIG.epochs == 2

    def test_model_name(self):
        assert CONFIG.model_name == "t5-base"

    def test_learning_rate(self):
        assert CONFIG.learning_rate == 5e-5

    def test_data_path(self):
        assert CONFIG.data_path == "data/reuters_headlines.csv"

    def test_device(self):
        assert CONFIG.device == "cuda:0"

    def test_output_dir(self):
        assert CONFIG.output_dir == "artifacts/t5/"

    def test_source_prefix(self):
        assert CONFIG.source_prefix.endswith(": ")
