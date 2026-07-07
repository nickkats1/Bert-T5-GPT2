from headlines.t5.config import T5Config
from headlines.t5.run import parse_args


class TestParseArgs:
    def test_defaults_match_config(self):
        cfg = parse_args([])
        assert isinstance(cfg, T5Config)
        assert cfg.epochs == T5Config().epochs
        assert cfg.report_bertscore == T5Config().report_bertscore

    def test_overrides_applied(self):
        cfg = parse_args(
            ["--epochs", "1", "--device", "cpu", "--source-length", "64", "--target-length", "16"]
        )
        assert cfg.epochs == 1
        assert cfg.device == "cpu"
        assert cfg.source_length == 64
        assert cfg.target_length == 16

    def test_bertscore_flag_enables_report(self):
        assert parse_args(["--bertscore"]).report_bertscore is True
        assert parse_args([]).report_bertscore is False
