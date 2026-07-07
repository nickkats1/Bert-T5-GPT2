import pandas as pd
import pytest

from headlines.t5.utils import load_data


class TestLoadData:
    def test_none_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data(None)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("does/not/exist.csv")

    def test_reads_required_columns(self, temp_reuters_headlines):
        df = load_data(temp_reuters_headlines)
        assert "Headlines" in df.columns
        assert "Description" in df.columns
        assert "Time" not in df.columns
        assert len(df) == 3
        assert list(df.index) == [0, 1, 2]

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Headlines": ["h"]}).to_csv(path, index=False)
        with pytest.raises(KeyError):
            load_data(path)

    def test_nans_become_empty_strings(self, tmp_path):
        path = tmp_path / "nan.csv"
        pd.DataFrame(
            {"Headlines": ["h1", None], "Description": ["d1", "d2"]}
        ).to_csv(path, index=False)
        df = load_data(path)
        assert (df["Headlines"] == "").any()
        assert df.isna().sum().sum() == 0
