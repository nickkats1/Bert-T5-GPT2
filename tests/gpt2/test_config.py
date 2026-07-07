from headlines.gpt2.config import CONFIG, GPT2Config


class TestGPT2Config:
    """Validate the default ``GPT2Config`` values."""

    def test_is_dataclass_instance(self):
        assert isinstance(CONFIG, GPT2Config)

    def test_model_name(self):
        assert CONFIG.model_name == "gpt2"

    def test_epochs(self):
        assert CONFIG.epochs == 3

    def test_device(self):
        assert CONFIG.device == "cuda"

    def test_learning_rate(self):
        assert CONFIG.learning_rate == 5e-5

    def test_output_dir_under_artifacts(self):
        """Artifacts live outside the source tree."""
        assert CONFIG.output_dir.startswith("artifacts/")

    def test_special_tokens(self):
        assert CONFIG.pad_token and CONFIG.bos_token and CONFIG.eos_token
