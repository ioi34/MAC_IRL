from src.data.loaders import read_table


def test_csv_loader_strips_header_whitespace(tmp_path):
    source = tmp_path / "daily.csv"
    source.write_text(" date , close \n2025-01-01,100\n", encoding="utf-8")

    frame = read_table(source)

    assert frame.columns.tolist() == ["date", "close"]
