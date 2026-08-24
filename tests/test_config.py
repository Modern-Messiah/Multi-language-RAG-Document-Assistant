"""Settings validation and the .env -> component wiring that startup depends on."""
import pytest
from pydantic import ValidationError

from app.config import Settings
from tests.conftest import make_settings


def _settings(**overrides):
    return Settings(_env_file=None, openai_api_key="k", **overrides)


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError) as exc:
        _settings(chunk_size=100, chunk_overlap=200)

    message = str(exc.value)
    assert "CHUNK_OVERLAP" in message and "CHUNK_SIZE" in message


def test_chunk_overlap_equal_to_chunk_size_is_rejected():
    with pytest.raises(ValidationError):
        _settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k_results": 0},
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"max_file_size": 0},
        {"temperature": -0.5},
        {"temperature": 2.5},
    ],
)
def test_out_of_range_values_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _settings(**overrides)


def test_defaults_are_valid():
    settings = _settings()

    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k_results >= 1


def test_dotenv_only_startup_works(tmp_path, monkeypatch, fake_openai_embeddings):
    """A .env file with no exported env vars must be enough to boot.

    pydantic-settings does NOT export .env values into os.environ, so any
    component reading os.getenv("OPENAI_API_KEY") crashed startup for anyone
    following the documented `cp .env.template .env` flow.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = make_settings(tmp_path, openai_api_key="sk-only-in-dotenv")

    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200


def test_rag_chain_receives_key_from_settings(tmp_path, monkeypatch, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(make_settings(tmp_path, openai_api_key="sk-only-in-dotenv"))

    with TestClient(app):
        assert app.state.rag_chain.client.api_key == "sk-only-in-dotenv"


def test_settings_are_wired_into_components(tmp_path, fake_openai_embeddings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = make_settings(
        tmp_path,
        model_name="gpt-4o",
        temperature=0.7,
        top_k_results=9,
        chunk_size=321,
        chunk_overlap=21,
    )
    app = create_app(settings)

    with TestClient(app):
        assert app.state.rag_chain.model == "gpt-4o"
        assert app.state.rag_chain.temperature == 0.7
        assert app.state.rag_chain.top_k == 9
        assert app.state.chunker.chunk_size == 321
        assert app.state.chunker.chunk_overlap == 21
