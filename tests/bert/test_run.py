from headlines.bert.config import BertConfig
from headlines.bert.run import parse_args


class TestParseArgs:
    def test_defaults_match_config(self):
        cfg = parse_args([])
        assert isinstance(cfg, BertConfig)
        assert cfg.epochs == BertConfig().epochs
        assert cfg.model_name == BertConfig().model_name

    def test_overrides_applied(self):
        cfg = parse_args(["--epochs", "1", "--device", "cpu", "--batch-size", "4", "--seed", "7"])
        assert cfg.epochs == 1
        assert cfg.device == "cpu"
        assert cfg.batch_size == 4
        assert cfg.seed == 7

    def test_returns_new_frozen_instance(self):
        cfg = parse_args(["--epochs", "2"])
        assert cfg is not BertConfig()
        assert cfg.epochs == 2
