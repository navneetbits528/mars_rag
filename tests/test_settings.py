import pytest
from pydantic import ValidationError
from marsrag.settings import Settings


def test_embedding_dimension_is_bounded():
    with pytest.raises(ValidationError):
        Settings(embed_dim=3000)


def test_demo_request_budget_default():
    settings = Settings()
    assert settings.openai_max_requests == 10
