import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from jupyter_mynerva.routes import (
    encrypt_api_key,
    decrypt_api_key,
    load_config,
    save_config,
    resolve_chat_config,
    _fetch_openai_models,
    _openai_models_cache,
    _NotebookStore,
)



def test_encrypt_decrypt_roundtrip(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv('MYNERVA_SECRET_KEY', key)

    original = 'sk-test-key-12345'
    encrypted = encrypt_api_key(original)
    assert encrypted.startswith('encrypted:')
    assert original not in encrypted

    decrypted = decrypt_api_key(encrypted)
    assert decrypted == original


def test_decrypt_empty():
    assert decrypt_api_key('') == ''
    assert decrypt_api_key(None) == ''


def test_decrypt_unencrypted():
    assert decrypt_api_key('plain-key') == 'plain-key'


def test_decrypt_without_secret_key_raises(monkeypatch):
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)
    with pytest.raises(ValueError, match='MYNERVA_SECRET_KEY'):
        decrypt_api_key('encrypted:somedata')


def test_load_save_config(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    config = {'provider': 'enki-gate', 'model': '', 'apiKey': '', 'enkiGateUrl': 'https://example.com'}
    save_config(config)
    assert config_file.exists()

    loaded = load_config()
    assert loaded['provider'] == 'enki-gate'
    assert loaded['enkiGateUrl'] == 'https://example.com'


def test_load_config_missing_fields(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({'apiKey': 'sk-test'}))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = load_config()
    assert loaded['provider'] == 'openai'
    assert loaded['model'] == 'gpt-5.2'
    assert 'configWarning' in loaded
    assert 'provider' in loaded['configWarning']
    assert 'model' in loaded['configWarning']


def test_load_config_missing_fields_with_use_default(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({'apiKey': '', 'useDefault': True}))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = load_config()
    assert 'configWarning' not in loaded


def test_load_config_decrypt_error(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({
        'provider': 'openai',
        'model': 'gpt-5.2',
        'apiKey': 'encrypted:invalid_data'
    }))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = load_config()
    assert loaded['apiKey'] == ''
    assert 'decryptError' in loaded



def test_notebook_store_eviction_cleans_up_temp_file(tmp_path):
    store = _NotebookStore(maxsize=2)

    def write_store(key, content):
        path = str(tmp_path / key)
        with open(path, 'w') as f:
            json.dump(content, f)
        store[key] = path
        return path

    path_a = write_store('a.ipynb', {'cells': [{'source': 'x=1'}], 'nbformat': 4})
    path_b = write_store('b.ipynb', {'cells': [{'source': 'y=2'}], 'nbformat': 4})

    # Cache hit: same key returns same path
    assert store['a.ipynb'] == path_a
    assert store['b.ipynb'] == path_b

    # Cached content is readable and correct
    with open(store['a.ipynb']) as f:
        assert json.load(f)['cells'][0]['source'] == 'x=1'
    with open(store['b.ipynb']) as f:
        assert json.load(f)['cells'][0]['source'] == 'y=2'

    # Third entry evicts a.ipynb
    path_c = write_store('c.ipynb', {'cells': [{'source': 'z=3'}], 'nbformat': 4})

    assert not os.path.exists(path_a), "evicted temp file should be deleted"
    assert 'a.ipynb' not in store

    # b and c still cached with correct content
    with open(store['b.ipynb']) as f:
        assert json.load(f)['cells'][0]['source'] == 'y=2'
    with open(store['c.ipynb']) as f:
        assert json.load(f)['cells'][0]['source'] == 'z=3'


def test_notebook_store_eviction_warns_on_missing_file(tmp_path, caplog):
    store = _NotebookStore(maxsize=1)

    store['a.ipynb'] = '/nonexistent/path.ipynb'

    with caplog.at_level(logging.WARNING):
        store['b.ipynb'] = str(tmp_path / 'b.ipynb')

    assert any('Failed to remove store file' in r.message for r in caplog.records)


# --- _fetch_openai_models ---

def test_fetch_openai_models(monkeypatch):
    _openai_models_cache.clear()

    mock_model_a = MagicMock()
    mock_model_a.id = 'model-a'
    mock_model_b = MagicMock()
    mock_model_b.id = 'model-b'
    mock_response = MagicMock()
    mock_response.data = [mock_model_b, mock_model_a]

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list.return_value = mock_response
        result = _fetch_openai_models('key', 'http://localhost:8000/v1')

    assert result == ['model-a', 'model-b']
    MockOpenAI.assert_called_once_with(api_key='key', base_url='http://localhost:8000/v1')


def test_fetch_openai_models_cache():
    _openai_models_cache.clear()
    _openai_models_cache['http://cached/v1'] = ['cached-model']

    result = _fetch_openai_models('key', 'http://cached/v1')
    assert result == ['cached-model']


def test_fetch_openai_models_empty_raises():
    _openai_models_cache.clear()

    mock_response = MagicMock()
    mock_response.data = []

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list.return_value = mock_response
        with pytest.raises(ValueError, match='No models available'):
            _fetch_openai_models('key', 'http://localhost:8000/v1')


# --- resolve_chat_config ---

def test_resolve_chat_config_use_default(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
        'openai_base_url': 'http://admin-endpoint/v1',
        'provider': 'openai',
    })
    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        lambda: {'provider': 'openai', 'model': 'admin-model'})

    config = {
        'useDefault': True,
        'provider': 'openai',
        'apiKey': 'user-key',
        'openaiBaseUrl': 'http://evil-server/v1',
    }
    provider, model, api_key, base_url = resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'admin-model'
    assert api_key == 'admin-key'
    assert base_url == 'http://admin-endpoint/v1'


def test_resolve_chat_config_use_default_ignores_user_base_url(monkeypatch):
    """Ensure useDefault=true never uses user-supplied base_url (credential leak prevention)."""
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
    })
    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        lambda: {'provider': 'openai', 'model': 'gpt-5.2'})

    config = {
        'useDefault': True,
        'openaiBaseUrl': 'http://evil-server/v1',
    }
    _, _, api_key, base_url = resolve_chat_config(config)

    assert api_key == 'admin-key'
    assert base_url is None  # Not the user's evil URL


def test_resolve_chat_config_user_config(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
        'openai_base_url': 'http://admin-endpoint/v1',
    })

    config = {
        'provider': 'openai',
        'model': 'my-model',
        'apiKey': 'user-key',
        'openaiBaseUrl': 'http://user-endpoint/v1',
    }
    provider, model, api_key, base_url = resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'my-model'
    assert api_key == 'user-key'
    assert base_url == 'http://user-endpoint/v1'


def test_resolve_chat_config_defaults_only(monkeypatch):
    """defaults_only ignores user config even when useDefault is false."""
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
        'openai_base_url': 'http://admin-endpoint/v1',
        'defaults_only': True,
    })
    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        lambda: {'provider': 'openai', 'model': 'admin-model'})

    config = {
        'provider': 'anthropic',
        'model': 'claude-sonnet-4-5-20250929',
        'apiKey': 'user-key',
    }
    provider, model, api_key, base_url = resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'admin-model'
    assert api_key == 'admin-key'
    assert base_url == 'http://admin-endpoint/v1'


def test_resolve_chat_config_no_defaults_raises(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {})
    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config', lambda: None)

    with pytest.raises(ValueError, match='Default configuration not available'):
        resolve_chat_config({'useDefault': True})


