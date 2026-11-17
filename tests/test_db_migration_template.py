from pathlib import Path


def test_schema_has_dimension_template():
    sql = Path("sql/001_schema.sql").read_text(encoding="utf-8")
    assert "VECTOR({{EMBED_DIM}})" in sql
    assert "USING hnsw" in sql
    assert "USING gin" in sql
