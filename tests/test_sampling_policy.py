"""Prevent model controls that the documented provider ignores from reaching inference."""
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from portfolio.api import create_app
from portfolio.config import Generation, Settings, sampling_controls
from portfolio.models import build_model


@pytest.mark.parametrize('model', ['gemini-3.6-flash', 'gemini-3.8-flash',
                                  'models/gemini-3.8-flash', 'gemini-4-carbon'])
@pytest.mark.parametrize('option,value', [('temperature', 1), ('top_p', .8), ('top_k', 20)])
def test_provider_controlled_sampling_rejected(model, option, value):
    with pytest.raises(ValueError, match='provider-controlled sampling'):
        Generation(model=model, **{option: value})


def test_current_default_omits_sampling_and_credentials_stay_private(settings):
    configured = settings.model_copy(update={'google_api_key': SecretStr('private-test-placeholder')})
    with patch('portfolio.models.init_chat_model') as initialize:
        build_model(configured)
    options = initialize.call_args.kwargs
    assert not {'temperature', 'top_p', 'top_k'} & options.keys()
    assert options['vertexai'] is False
    assert sampling_controls('google_genai', 'gemini-3.8-flash') == []
    with TestClient(create_app(configured)) as client:
        result = client.get('/v1/config', headers={'Authorization': 'Bearer demo-alice-key'})
        assert result.status_code == 200
        assert result.json()['model_controls']['google_genai']['gemini-3.8-flash'] == []
        assert 'private-test-placeholder' not in result.text


def test_legacy_sampling_and_openai_known_restriction():
    spec = Generation(model='gemini-2.5-flash', temperature=0, top_p=.8, top_k=20)
    assert spec.temperature == 0 and spec.top_k == 20
    assert 'temperature' in sampling_controls('google_genai', spec.model)
    assert sampling_controls('openai', 'gpt-4o-mini') == ['temperature', 'top_p']
    with pytest.raises(ValueError, match='top_k'):
        Generation(provider='openai', model='gpt-4o-mini', top_k=20)


def test_blank_environment_uses_provider_defaults(tmp_path):
    env = tmp_path/'settings.env'
    env.write_text('LLM_TEMPERATURE=\nLLM_TOP_P=\nLLM_TOP_K=\n', encoding='utf-8')
    settings = Settings(_env_file=env, app_env='test', data_dir=tmp_path/'runtime')
    assert settings.generation().temperature is None
    assert settings.generation().top_p is None
    assert settings.generation().top_k is None


def test_api_rejects_ineffective_override_before_graph_execution(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post('/v1/chat', headers={'Authorization': 'Bearer demo-alice-key'},
                               json={'session_id': 'unused', 'message': 'calculate 1+1',
                                     'generation': {'model': 'gemini-3.8-flash', 'top_p': .8}})
        assert response.status_code == 422
        assert 'provider-controlled sampling' in response.text
