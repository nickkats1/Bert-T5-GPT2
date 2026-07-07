from headlines.gpt2.config import GPT2Config
from headlines.gpt2.run import parse_args


class TestParseArgs:
    def test_defaults_match_config(self):
        cfg = parse_args([])
        assert isinstance(cfg, GPT2Config)
        assert cfg.epochs == GPT2Config().epochs
        assert cfg.use_amp == GPT2Config().use_amp

    def test_overrides_applied(self):
        cfg = parse_args(["--epochs", "1", "--device", "cpu", "--accum-steps", "2"])
        assert cfg.epochs == 1
        assert cfg.device == "cpu"
        assert cfg.accum_steps == 2

    def test_no_amp_flag_disables_amp(self):
        assert parse_args(["--no-amp"]).use_amp is False
        assert parse_args([]).use_amp == GPT2Config().use_amp
