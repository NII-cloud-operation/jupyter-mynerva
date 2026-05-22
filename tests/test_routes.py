import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

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
    _fetch_bedrock_models,
    _bedrock_models_cache,
    _validate_bedrock_region,
    _load_bedrock_regions,
    _fetch_chat_models,
    _chat_models_cache,
    _filter_models,
    _load_model_spec,
    _get_provider_models,
    _build_providers_with_models,
    OpenAIModelsHandler,
    BedrockModelsHandler,
    ProviderModelsHandler,
    _NotebookStore,
    _convert_messages_for_responses_api,
    _build_anthropic_params,
    _build_bedrock_converse_body,
    _extract_json_content,
    _send_sse,
    sse_serializer,
    chat_openai,
    chat_anthropic,
    chat_bedrock_converse,
)
from jupyter_mynerva.echo_agent import chat_echo



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


def test_decrypt_with_wrong_key_raises(monkeypatch):
    old_key = Fernet.generate_key().decode()
    monkeypatch.setenv('MYNERVA_SECRET_KEY', old_key)
    encrypted = encrypt_api_key('sk-secret')

    new_key = Fernet.generate_key().decode()
    monkeypatch.setenv('MYNERVA_SECRET_KEY', new_key)
    with pytest.raises(ValueError, match='may have changed'):
        decrypt_api_key(encrypted)


async def test_load_config_recovers_from_wrong_key(monkeypatch, tmp_path):
    old_key = Fernet.generate_key().decode()
    monkeypatch.setenv('MYNERVA_SECRET_KEY', old_key)
    config_file = tmp_path / '.mynerva' / 'config.json'
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    save_config({'provider': 'openai', 'model': 'gpt-5.2', 'apiKey': 'sk-secret'})

    new_key = Fernet.generate_key().decode()
    monkeypatch.setenv('MYNERVA_SECRET_KEY', new_key)
    loaded = await load_config()
    assert loaded['apiKey'] == ''
    assert 'decryptError' in loaded


async def test_load_save_config(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    config = {'provider': 'enki-gate', 'model': '', 'apiKey': '', 'enkiGateUrl': 'https://example.com'}
    save_config(config)
    assert config_file.exists()

    loaded = await load_config()
    assert loaded['provider'] == 'enki-gate'
    assert loaded['enkiGateUrl'] == 'https://example.com'


async def test_load_config_missing_fields(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({'apiKey': 'sk-test'}))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = await load_config()
    assert loaded['provider'] == 'openai'
    assert loaded['model'] == 'gpt-5.2'
    assert 'configWarning' in loaded
    assert 'provider' in loaded['configWarning']
    assert 'model' in loaded['configWarning']


async def test_load_config_missing_fields_with_use_default(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({'apiKey': '', 'useDefault': True}))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = await load_config()
    assert 'configWarning' not in loaded


async def test_load_config_decrypt_error(monkeypatch, tmp_path):
    config_file = tmp_path / '.mynerva' / 'config.json'
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({
        'provider': 'openai',
        'model': 'gpt-5.2',
        'apiKey': 'encrypted:invalid_data'
    }))
    monkeypatch.setattr('jupyter_mynerva.routes.get_config_path', lambda: config_file)
    monkeypatch.delenv('MYNERVA_SECRET_KEY', raising=False)

    loaded = await load_config()
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

async def test_fetch_openai_models(monkeypatch):
    _openai_models_cache.clear()

    mock_model_a = MagicMock()
    mock_model_a.id = 'model-a'
    mock_model_b = MagicMock()
    mock_model_b.id = 'model-b'
    mock_response = MagicMock()
    mock_response.data = [mock_model_b, mock_model_a]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=mock_response)
        result = await _fetch_openai_models('key', 'http://localhost:8000/v1')

    assert result == ['model-a', 'model-b']
    MockOpenAI.assert_called_once_with(api_key='key', base_url='http://localhost:8000/v1')


async def test_fetch_openai_models_cache():
    _openai_models_cache.clear()
    _openai_models_cache[('http://cached/v1', 'key')] = ['cached-model']

    result = await _fetch_openai_models('key', 'http://cached/v1')
    assert result == ['cached-model']


async def test_fetch_openai_models_cache_keyed_by_api_key(monkeypatch):
    """Cache must not return one key's models when a different key is supplied.

    Regression: cache_key was previously base_url only, so swapping the key
    against the same endpoint silently returned the prior result (broken UX
    when user wipes the key field expecting a re-auth).
    """
    _openai_models_cache.clear()
    _openai_models_cache[('http://x/v1', 'key-A')] = ['model-from-A']

    mock_response = MagicMock()
    mock_response.data = [MagicMock(id='model-from-B')]
    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=mock_response)
        result = await _fetch_openai_models('key-B', 'http://x/v1')

    assert result == ['model-from-B']
    MockOpenAI.assert_called_once_with(api_key='key-B', base_url='http://x/v1')


async def test_fetch_openai_models_empty_raises():
    _openai_models_cache.clear()

    mock_response = MagicMock()
    mock_response.data = []

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=mock_response)
        with pytest.raises(ValueError, match='No models available'):
            await _fetch_openai_models('key', 'http://localhost:8000/v1')


# --- OpenAIModelsHandler ---

def _make_models_handler(body):
    """MagicMock handler instance pre-wired for OpenAIModelsHandler.post()."""
    h = MagicMock()
    h.current_user = 'user'  # bypass @tornado.web.authenticated
    h.get_json_body.return_value = body
    return h


async def test_openai_models_handler_success(monkeypatch):
    async def fake_fetch(key, url):
        return ['model-x', 'model-y']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models',
                        fake_fetch)
    handler = _make_models_handler({'baseUrl': 'http://x/v1', 'apiKey': 'k'})

    await OpenAIModelsHandler.post(handler)

    written = handler.finish.call_args[0][0]
    assert json.loads(written) == {'models': ['model-x', 'model-y']}
    handler.set_status.assert_not_called()


async def test_openai_models_handler_passes_baseurl_and_apikey(monkeypatch):
    """Frontend body keys 'baseUrl' / 'apiKey' must be honored (camelCase contract)."""
    captured = {}

    async def fake_fetch(key, url):
        captured['args'] = (key, url)
        return ['m']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models', fake_fetch)
    handler = _make_models_handler({'baseUrl': 'http://endpoint/v1', 'apiKey': 'sk-x'})

    await OpenAIModelsHandler.post(handler)

    assert captured['args'] == ('sk-x', 'http://endpoint/v1')


async def test_openai_models_handler_returns_500_with_error_body(monkeypatch):
    """Auth / network errors must surface as JSON {error: ...} not bare 500."""
    async def fake_fetch(key, url):
        raise RuntimeError('401: Missing bearer authentication in header')

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models', fake_fetch)
    handler = _make_models_handler({'baseUrl': 'http://x/v1', 'apiKey': ''})

    await OpenAIModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(500)
    written = handler.finish.call_args[0][0]
    body = json.loads(written)
    assert 'Missing bearer authentication' in body['error']


# --- ProviderModelsHandler ---

async def test_provider_models_handler_openai_official_requires_api_key(monkeypatch):
    handler = _make_models_handler({'provider': 'openai', 'apiKey': '', 'baseUrl': ''})

    await ProviderModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(400)
    assert json.loads(handler.finish.call_args[0][0]) == {'error': 'API key is required'}


async def test_provider_models_handler_openai_default_base_url_requires_api_key(monkeypatch):
    handler = _make_models_handler({
        'provider': 'openai',
        'apiKey': '',
        'baseUrl': 'https://api.openai.com/v1/',
    })

    await ProviderModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(400)
    assert json.loads(handler.finish.call_args[0][0]) == {'error': 'API key is required'}


async def test_provider_models_handler_openai_custom_allows_empty_api_key(monkeypatch):
    captured = {}

    async def fake_get_models(provider, api_key='', base_url='', region=''):
        captured['args'] = (provider, api_key, base_url)
        return ['custom-model']

    monkeypatch.setattr('jupyter_mynerva.routes._get_provider_models',
                        fake_get_models)
    handler = _make_models_handler({
        'provider': 'openai',
        'apiKey': '',
        'baseUrl': 'http://custom/v1',
    })

    await ProviderModelsHandler.post(handler)

    assert captured['args'] == ('openai', '', 'http://custom/v1')
    assert json.loads(handler.finish.call_args[0][0]) == {'models': ['custom-model']}


async def test_provider_models_handler_anthropic_uses_user_api_key(monkeypatch):
    captured = {}

    async def fake_get_models(provider, api_key='', base_url='', region=''):
        captured['args'] = (provider, api_key, base_url)
        return ['claude-x']

    monkeypatch.setattr('jupyter_mynerva.routes._get_provider_models',
                        fake_get_models)
    handler = _make_models_handler({'provider': 'anthropic', 'apiKey': 'user-key'})

    await ProviderModelsHandler.post(handler)

    assert captured['args'] == ('anthropic', 'user-key', '')
    assert json.loads(handler.finish.call_args[0][0]) == {'models': ['claude-x']}


async def test_provider_models_handler_rejects_unsupported_provider(monkeypatch):
    handler = _make_models_handler({'provider': 'enki-gate', 'apiKey': 'token'})

    await ProviderModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(400)
    assert json.loads(handler.finish.call_args[0][0]) == {
        'error': 'Unsupported provider: enki-gate'
    }


# --- _load_model_spec / _filter_models ---

def test_load_model_spec_has_required_keys():
    spec = _load_model_spec()
    assert 'openai' in spec
    assert 'anthropic' in spec
    assert 'allow' in spec['openai']
    assert 'allow' in spec['anthropic']


def test_filter_models_allow_glob():
    ids = ['gpt-5.2', 'gpt-5-mini', 'gpt-4', 'text-embedding-3', 'whisper-1']
    result = _filter_models(ids, ['gpt-5*', 'gpt-4'], [])
    assert result == ['gpt-4', 'gpt-5-mini', 'gpt-5.2']


def test_filter_models_deny_takes_precedence():
    ids = ['gpt-4o', 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18']
    result = _filter_models(ids, ['gpt-4o*'], ['*-????-??-??'])
    assert result == ['gpt-4o']


def test_filter_models_no_match_returns_empty():
    ids = ['text-embedding-3', 'whisper-1']
    result = _filter_models(ids, ['gpt-*'], [])
    assert result == []


def test_filter_models_returns_sorted():
    ids = ['gpt-5-nano', 'gpt-5.2', 'gpt-5-mini']
    result = _filter_models(ids, ['gpt-5*'], [])
    assert result == sorted(['gpt-5-nano', 'gpt-5.2', 'gpt-5-mini'])


def test_filter_models_anthropic_alias_only_strategy():
    """Real-world spec: keep only alias-form Anthropic models, drop dated snapshots.

    Regression: a naive deny pattern of '*-????????' would also match alias IDs
    like 'claude-opus-4-6' (because '?' matches any char, not digits only).
    Use [0-9] character class to constrain to digits.
    """
    ids = [
        'claude-haiku-4-5-20251001',
        'claude-opus-4-1-20250805',
        'claude-opus-4-20250514',
        'claude-opus-4-5-20251101',
        'claude-opus-4-6',
        'claude-opus-4-7',
        'claude-sonnet-4-20250514',
        'claude-sonnet-4-5-20250929',
        'claude-sonnet-4-6',
    ]
    result = _filter_models(
        ids, ['claude-*-4-*'],
        ['*-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'])
    assert result == ['claude-opus-4-6', 'claude-opus-4-7', 'claude-sonnet-4-6']


# --- _fetch_chat_models ---

def _mock_models_response(ids, *, anthropic=False):
    """Build a mock `client.models.list()` response.

    Assigns an incrementing `created` (OpenAI) or `created_at` (Anthropic) so
    tests of the date-DESC sort can specify order by argument index without
    extra setup. Pass `(id, ts)` tuples to override.
    """
    from datetime import datetime, timezone
    mocks = []
    for i, item in enumerate(ids):
        if isinstance(item, tuple):
            mid, ts = item
        else:
            mid, ts = item, 1700000000 + i
        m = MagicMock()
        m.id = mid
        if anthropic:
            m.created_at = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            m.created = ts
        mocks.append(m)
    response = MagicMock()
    response.data = mocks
    return response


async def test_fetch_chat_models_openai_filters_and_caches(monkeypatch):
    _chat_models_cache.clear()
    monkeypatch.setattr('jupyter_mynerva.routes._load_model_spec',
                        lambda: {'openai': {'allow': ['gpt-5*'], 'deny': []}})

    response = _mock_models_response(['gpt-5.2', 'text-embedding-3', 'gpt-5-mini'])

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=response)
        result = await _fetch_chat_models('openai', 'admin-key')

    assert result == ['gpt-5-mini', 'gpt-5.2']
    MockOpenAI.assert_called_once_with(api_key='admin-key')

    # Cached: second call must not re-invoke the client
    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI2:
        cached = await _fetch_chat_models('openai', 'admin-key')
    assert cached == ['gpt-5-mini', 'gpt-5.2']
    MockOpenAI2.assert_not_called()


async def test_fetch_chat_models_anthropic_filters_and_caches(monkeypatch):
    _chat_models_cache.clear()
    monkeypatch.setattr('jupyter_mynerva.routes._load_model_spec',
                        lambda: {'anthropic': {'allow': ['claude-*-4-*'], 'deny': []}})

    response = _mock_models_response([
        'claude-sonnet-4-5-20250929',
        'claude-3-opus-20240229',
        'claude-haiku-4-5-20251001',
    ], anthropic=True)

    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.models.list = AsyncMock(return_value=response)
        result = await _fetch_chat_models('anthropic', 'admin-key')

    assert result == ['claude-haiku-4-5-20251001', 'claude-sonnet-4-5-20250929']
    MockAnthropic.assert_called_once_with(api_key='admin-key')


async def test_fetch_chat_models_sorted_by_created_desc(monkeypatch):
    """Newer releases come first regardless of alphabetic ID order."""
    _chat_models_cache.clear()
    monkeypatch.setattr('jupyter_mynerva.routes._load_model_spec',
                        lambda: {'openai': {'allow': ['gpt-*'], 'deny': []}})
    # Alphabetic ASC would put gpt-4.1 first; created DESC puts gpt-5.5 first
    response = _mock_models_response([
        ('gpt-4.1', 1_700_000_000),
        ('gpt-5', 1_750_000_000),
        ('gpt-5.5', 1_800_000_000),
    ])
    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=response)
        result = await _fetch_chat_models('openai', 'admin-key')
    assert result == ['gpt-5.5', 'gpt-5', 'gpt-4.1']


async def test_fetch_chat_models_cache_keyed_by_api_key():
    _chat_models_cache.clear()
    _chat_models_cache[('openai', 'key-A')] = ['model-from-A']

    mock_response = _mock_models_response(['gpt-5.2'])
    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.models.list = AsyncMock(return_value=mock_response)
        result = await _fetch_chat_models('openai', 'key-B')

    assert result == ['gpt-5.2']
    MockOpenAI.assert_called_once_with(api_key='key-B')


# --- _get_provider_models ---

async def test_get_provider_models_openai_uses_api_key(monkeypatch):
    async def fake_fetch_chat_models(pid, key):
        return ['gpt-5.2'] if (pid, key) == ('openai', 'user-key') else []

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_chat_models',
                        fake_fetch_chat_models)

    assert await _get_provider_models('openai', 'user-key') == ['gpt-5.2']


async def test_get_provider_models_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {})
    assert await _get_provider_models('openai') == []
    assert await _get_provider_models('anthropic') == []


async def test_get_provider_models_unknown_provider_returns_empty(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG',
                        {'openai_api_key': 'k', 'anthropic_api_key': 'k'})
    assert await _get_provider_models('enki-gate') == []
    assert await _get_provider_models('echo') == []


async def test_get_provider_models_openai_with_base_url(monkeypatch):
    """When openai_base_url is set, route through _fetch_openai_models (raw, no filter)."""
    captured = {}

    async def fake_fetch_openai_models(api_key, base_url):
        captured['args'] = (api_key, base_url)
        return ['custom-model-a', 'custom-model-b']

    async def fake_fetch_chat_models(pid, key):
        captured['chat_called'] = True
        return []

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models',
                        fake_fetch_openai_models)
    monkeypatch.setattr('jupyter_mynerva.routes._fetch_chat_models',
                        fake_fetch_chat_models)

    result = await _get_provider_models(
        'openai', 'admin-key', 'http://custom-endpoint/v1')
    assert result == ['custom-model-a', 'custom-model-b']
    assert captured['args'] == ('admin-key', 'http://custom-endpoint/v1')
    assert 'chat_called' not in captured  # filter path must not be invoked


async def test_get_provider_models_openai_default_base_url_uses_chat_models(monkeypatch):
    captured = {}

    async def fake_fetch_openai_models(api_key, base_url):
        captured['openai_models_called'] = True
        return []

    async def fake_fetch_chat_models(pid, key):
        captured['chat_args'] = (pid, key)
        return ['gpt-5.2']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models',
                        fake_fetch_openai_models)
    monkeypatch.setattr('jupyter_mynerva.routes._fetch_chat_models',
                        fake_fetch_chat_models)

    result = await _get_provider_models(
        'openai', 'user-key', 'https://api.openai.com/v1/')

    assert result == ['gpt-5.2']
    assert captured['chat_args'] == ('openai', 'user-key')
    assert 'openai_models_called' not in captured


async def test_get_provider_models_openai_with_base_url_no_api_key(monkeypatch):
    """base_url without api_key still hits the custom endpoint (auth-less endpoints)."""
    captured = {}

    async def fake_fetch_openai_models(api_key, base_url):
        captured['args'] = (api_key, base_url)
        return ['m']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_openai_models',
                        fake_fetch_openai_models)

    assert await _get_provider_models('openai', '', 'http://no-auth-endpoint/v1') == ['m']
    assert captured['args'] == ('', 'http://no-auth-endpoint/v1')


async def test_get_provider_models_anthropic_ignores_openai_base_url(monkeypatch):
    """openai_base_url must not affect anthropic provider."""
    async def fake_fetch_chat_models(pid, key):
        return ['claude-x'] if (pid, key) == ('anthropic', 'a-key') else []

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_chat_models',
                        fake_fetch_chat_models)

    assert await _get_provider_models('anthropic', 'a-key', 'http://custom/v1') == ['claude-x']


# --- _build_providers_with_models ---

async def test_build_providers_with_models_attaches_model_lists(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes.PROVIDERS', [
        {'id': 'openai', 'displayName': 'OpenAI'},
        {'id': 'anthropic', 'displayName': 'Anthropic'},
    ])
    async def fake_default_models(pid):
        return ['m1', 'm2'] if pid == 'openai' else ['c1']

    monkeypatch.setattr('jupyter_mynerva.routes._get_default_provider_models',
                        fake_default_models)

    result = await _build_providers_with_models()
    assert result == [
        {'id': 'openai', 'displayName': 'OpenAI', 'models': ['m1', 'm2']},
        {'id': 'anthropic', 'displayName': 'Anthropic', 'models': ['c1']},
    ]


async def test_build_providers_with_models_uses_config_provider(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes.PROVIDERS', [
        {'id': 'openai', 'displayName': 'OpenAI'},
        {'id': 'anthropic', 'displayName': 'Anthropic'},
    ])
    captured = []

    async def fake_get_provider_models(provider, api_key='', base_url='', region=''):
        captured.append((provider, api_key, base_url))
        return ['claude-x']

    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {})
    monkeypatch.setattr('jupyter_mynerva.routes._get_provider_models',
                        fake_get_provider_models)

    result = await _build_providers_with_models({
        'provider': 'anthropic',
        'model': 'claude-x',
        'apiKey': 'user-anthropic-key',
    })

    assert result == [
        {'id': 'openai', 'displayName': 'OpenAI', 'models': []},
        {'id': 'anthropic', 'displayName': 'Anthropic', 'models': ['claude-x']},
    ]
    assert captured == [('anthropic', 'user-anthropic-key', '')]


# --- resolve_chat_config ---

async def test_resolve_chat_config_use_default(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
        'openai_base_url': 'http://admin-endpoint/v1',
        'provider': 'openai',
    })
    async def fake_default_config():
        return {'provider': 'openai', 'model': 'admin-model'}

    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        fake_default_config)

    config = {
        'useDefault': True,
        'provider': 'openai',
        'apiKey': 'user-key',
        'openaiBaseUrl': 'http://evil-server/v1',
    }
    provider, model, api_key, base_url = await resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'admin-model'
    assert api_key == 'admin-key'
    assert base_url == 'http://admin-endpoint/v1'


async def test_resolve_chat_config_use_default_ignores_user_base_url(monkeypatch):
    """Ensure useDefault=true never uses user-supplied base_url (credential leak prevention)."""
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
    })
    async def fake_default_config():
        return {'provider': 'openai', 'model': 'gpt-5.2'}

    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        fake_default_config)

    config = {
        'useDefault': True,
        'openaiBaseUrl': 'http://evil-server/v1',
    }
    _, _, api_key, base_url = await resolve_chat_config(config)

    assert api_key == 'admin-key'
    assert base_url is None  # Not the user's evil URL


async def test_resolve_chat_config_user_config(monkeypatch):
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
    provider, model, api_key, base_url = await resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'my-model'
    assert api_key == 'user-key'
    assert base_url == 'http://user-endpoint/v1'


async def test_resolve_chat_config_defaults_only(monkeypatch):
    """defaults_only ignores user config even when useDefault is false."""
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'admin-key',
        'openai_base_url': 'http://admin-endpoint/v1',
        'defaults_only': True,
    })
    async def fake_default_config():
        return {'provider': 'openai', 'model': 'admin-model'}

    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        fake_default_config)

    config = {
        'provider': 'anthropic',
        'model': 'claude-sonnet-4-5-20250929',
        'apiKey': 'user-key',
    }
    provider, model, api_key, base_url = await resolve_chat_config(config)

    assert provider == 'openai'
    assert model == 'admin-model'
    assert api_key == 'admin-key'
    assert base_url == 'http://admin-endpoint/v1'


async def test_resolve_chat_config_no_defaults_raises(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {})
    async def fake_default_config():
        return None

    monkeypatch.setattr('jupyter_mynerva.routes.get_default_config',
                        fake_default_config)

    with pytest.raises(ValueError, match='Default configuration not available'):
        await resolve_chat_config({'useDefault': True})


# --- _convert_messages_for_responses_api ---

def test_convert_messages_system_to_developer():
    messages = [
        {'role': 'system', 'content': 'You are an assistant.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = _convert_messages_for_responses_api(messages)
    assert result[0]['role'] == 'developer'
    assert result[0]['content'] == 'You are an assistant.'
    assert result[1]['role'] == 'user'
    assert result[1]['content'] == 'Hello'


def test_convert_messages_preserves_other_roles():
    messages = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello'},
    ]
    result = _convert_messages_for_responses_api(messages)
    assert result[0]['role'] == 'user'
    assert result[1]['role'] == 'assistant'


def test_convert_messages_missing_role_defaults_to_user():
    messages = [{'content': 'No role specified'}]
    result = _convert_messages_for_responses_api(messages)
    assert result[0]['role'] == 'user'


def test_convert_messages_missing_content_defaults_to_empty():
    messages = [{'role': 'user'}]
    result = _convert_messages_for_responses_api(messages)
    assert result[0]['content'] == ''


# --- _extract_json_content ---

def test_extract_json_content_basic():
    raw = '{"messages":[{"role":"assistant","content":"Hello world"}],"actions":[]}'
    assert _extract_json_content(raw) == 'Hello world'


def test_extract_json_content_partial():
    raw = '{"messages":[{"role":"assistant","content":"Hello'
    assert _extract_json_content(raw) == 'Hello'


def test_extract_json_content_escaped():
    raw = '{"messages":[{"role":"assistant","content":"line1\\nline2"}]}'
    assert _extract_json_content(raw) == 'line1\nline2'


def test_extract_json_content_no_content_yet():
    raw = '{"messages":[{"role":'
    assert _extract_json_content(raw) == ''


def test_extract_json_content_empty():
    assert _extract_json_content('') == ''


# --- _send_sse ---

def test_send_sse_writes_correct_format():
    handler = MagicMock()
    _send_sse(handler, {'type': 'content_block_delta', 'content_type': 'text', 'delta': 'hello'})

    handler.write.assert_called_once()
    written = handler.write.call_args[0][0]
    assert written.startswith('data: ')
    assert written.endswith('\n\n')
    payload = json.loads(written[6:-2])
    assert payload == {'type': 'content_block_delta', 'content_type': 'text', 'delta': 'hello'}
    handler.flush.assert_called_once()


def _parse_sse_payloads(handler):
    """Extract parsed SSE payloads from mock handler write calls."""
    written = [call[0][0] for call in handler.write.call_args_list]
    payloads = []
    for w in written:
        if w.startswith('data: ') and not w.startswith('data: [DONE]'):
            payloads.append(json.loads(w[6:-2]))
    return payloads, written


# --- async helpers for streaming mocks ---

def _async_iter(items):
    """Wrap a sync iterable so it can be consumed via `async for`."""
    async def _gen():
        for item in items:
            yield item
    return _gen()


class _AsyncStreamCtx:
    """Async context manager that yields events from a list and exposes
    Anthropic's async final-state methods (get_final_message/text)."""
    def __init__(self, events, final_text='', stop_reason='end_turn'):
        self._events = list(events)
        self._final_text = final_text
        self._stop_reason = stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return _async_iter(self._events)

    async def get_final_text(self):
        return self._final_text

    async def get_final_message(self):
        msg = MagicMock()
        msg.stop_reason = self._stop_reason
        return msg


# --- chat_openai ---

def _make_event(event_type, **kwargs):
    """Create a mock streaming event."""
    event = MagicMock()
    event.type = event_type
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


@pytest.mark.asyncio
async def test_chat_openai_basic_flow():
    handler = MagicMock()
    # Simulate realistic JSON token stream from OpenAI
    json_text = '{"messages":[{"role":"assistant","content":"Hi there!"}],"actions":[]}'
    events = [
        _make_event('response.created'),
        _make_event('response.in_progress'),
        _make_event('response.output_item.added'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.delta', delta='{"messages":[{"role":"assistant","content":"Hi'),
        _make_event('response.output_text.delta', delta=' there!"}],"actions":[]}'),
        _make_event('response.output_text.done', text=json_text),
        _make_event('response.completed', response=MagicMock(status='completed', incomplete_details=None)),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, written = _parse_sse_payloads(handler)

    types = [p['type'] for p in payloads]
    assert 'content_block_start' in types
    assert 'content_block_delta' in types
    assert 'content_block_stop' in types
    assert 'message_done' in types

    starts = [p for p in payloads if p['type'] == 'content_block_start']
    assert starts[0]['content_type'] == 'thinking'
    assert starts[1]['content_type'] == 'text'

    # _extract_json_content extracts accumulated content from JSON
    deltas = [p for p in payloads if p['type'] == 'content_block_delta']
    assert deltas[0]['content_type'] == 'text'
    assert deltas[0]['delta'] == 'Hi'  # First chunk: partial content
    assert deltas[1]['delta'] == 'Hi there!'  # Second chunk: full content so far

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['text'] == json_text  # Full JSON for processLLMResponse

    stops = [p for p in payloads if p['type'] == 'content_block_stop']
    assert any(s['content_type'] == 'thinking' for s in stops)
    assert any(s['content_type'] == 'text' for s in stops)

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_openai_reasoning():
    handler = MagicMock()
    events = [
        _make_event('response.in_progress'),
        _make_event('response.reasoning_summary_text.delta', delta='Let me think'),
        _make_event('response.reasoning_summary_text.delta', delta=' about this'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.delta', delta='Answer'),
        _make_event('response.output_text.done', text='Answer'),
        _make_event('response.completed', response=MagicMock(status='completed', incomplete_details=None)),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, _ = _parse_sse_payloads(handler)

    thinking_deltas = [p for p in payloads
                       if p['type'] == 'content_block_delta' and p['content_type'] == 'thinking']
    assert len(thinking_deltas) == 2
    assert thinking_deltas[0]['delta'] == 'Let me think'
    assert thinking_deltas[1]['delta'] == ' about this'


@pytest.mark.asyncio
async def test_chat_openai_api_error():
    handler = MagicMock()

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(side_effect=Exception('API key invalid'))
        await chat_openai(handler, 'bad-key', 'gpt-4o', [])

    payloads, written = _parse_sse_payloads(handler)

    assert len(payloads) == 1
    assert payloads[0]['type'] == 'error'
    assert 'API key invalid' in payloads[0]['error']
    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_openai_failed_event():
    handler = MagicMock()
    events = [
        _make_event('response.in_progress'),
        _make_event('response.failed', error='rate limit exceeded'),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, _ = _parse_sse_payloads(handler)
    error_events = [p for p in payloads if p['type'] == 'error']
    assert len(error_events) == 1
    assert 'rate limit exceeded' in error_events[0]['error']


@pytest.mark.asyncio
async def test_chat_openai_system_role_converted():
    handler = MagicMock()
    messages = [
        {'role': 'system', 'content': 'You are helpful'},
        {'role': 'user', 'content': 'Hi'},
    ]
    events = [
        _make_event('response.output_text.done', text='Hello'),
        _make_event('response.completed', response=MagicMock(status='completed', incomplete_details=None)),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', messages)

    call_kwargs = MockOpenAI.return_value.responses.create.call_args
    api_input = call_kwargs[1]['input']
    assert api_input[0]['role'] == 'developer'
    assert api_input[1]['role'] == 'user'


@pytest.mark.asyncio
async def test_chat_openai_with_base_url():
    handler = MagicMock()
    events = [
        _make_event('response.output_text.done', text='ok'),
        _make_event('response.completed', response=MagicMock(status='completed', incomplete_details=None)),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [],
                                 base_url='http://custom/v1')

    MockOpenAI.assert_called_once_with(api_key='key', base_url='http://custom/v1')


# --- _build_anthropic_params ---

def test_build_anthropic_params_system_extraction():
    messages = [
        {'role': 'system', 'content': 'Be helpful'},
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello', 'actions': [{'type': 'getToc'}]},
    ]
    params = _build_anthropic_params(messages)
    assert params['system'] == 'Be helpful'
    assert len(params['messages']) == 2
    assert params['messages'][0] == {'role': 'user', 'content': 'Hi'}
    assert '[Actions proposed]' in params['messages'][1]['content']


def test_build_anthropic_params_no_system():
    messages = [{'role': 'user', 'content': 'Hi'}]
    params = _build_anthropic_params(messages)
    assert 'system' not in params
    assert params['max_tokens'] == 32000
    assert params['thinking'] == {'type': 'enabled', 'budget_tokens': 2000}


# --- chat_anthropic ---

def _make_anthropic_event(event_type, **kwargs):
    event = MagicMock()
    event.type = event_type
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


def _make_content_block(block_type, **kwargs):
    block = MagicMock()
    block.type = block_type
    for k, v in kwargs.items():
        setattr(block, k, v)
    return block


def _make_delta(delta_type, **kwargs):
    delta = MagicMock()
    delta.type = delta_type
    for k, v in kwargs.items():
        setattr(delta, k, v)
    return delta


@pytest.mark.asyncio
async def test_chat_anthropic_basic_flow():
    handler = MagicMock()
    # chat_anthropic extracts the content field from a Mynerva JSON envelope,
    # so the mocked text deltas form a partial JSON that resolves to "Hello world".
    json_text = '{"messages":[{"role":"assistant","content":"Hello world"}],"actions":[]}'
    events = [
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta',
                                                text='{"messages":[{"role":"assistant","content":"Hello')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta',
                                                text=' world"}],"actions":[]}')),
        _make_anthropic_event('content_block_stop'),
        _make_anthropic_event('message_stop'),
    ]

    mock_stream = _AsyncStreamCtx(events, final_text=json_text,
                                  stop_reason='end_turn')
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, written = _parse_sse_payloads(handler)

    types = [p['type'] for p in payloads]
    assert 'content_block_start' in types
    assert 'content_block_delta' in types
    assert 'content_block_stop' in types
    assert 'message_done' in types

    # _extract_json_content emits accumulated content (cumulative, not incremental)
    text_deltas = [p for p in payloads
                   if p['type'] == 'content_block_delta' and p['content_type'] == 'text']
    assert text_deltas[0]['delta'] == 'Hello'
    assert text_deltas[1]['delta'] == 'Hello world'

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['text'] == json_text
    assert done[0]['stop_reason'] == 'end_turn'

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_anthropic_thinking():
    handler = MagicMock()
    events = [
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('thinking')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('thinking_delta', thinking='Reasoning...')),
        _make_anthropic_event('content_block_stop'),
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text='Answer')),
        _make_anthropic_event('content_block_stop'),
    ]

    mock_stream = _AsyncStreamCtx(events, final_text='Answer',
                                  stop_reason='end_turn')
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-opus', [])

    payloads, _ = _parse_sse_payloads(handler)

    thinking_deltas = [p for p in payloads
                       if p['type'] == 'content_block_delta' and p['content_type'] == 'thinking']
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]['delta'] == 'Reasoning...'

    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    assert starts == ['thinking', 'text']  # No duplicate thinking start

    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']
    assert 'thinking' in stops
    assert 'text' in stops


@pytest.mark.asyncio
async def test_chat_anthropic_api_error():
    handler = MagicMock()

    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(side_effect=Exception('Auth failed'))
        await chat_anthropic(handler, 'bad-key', 'claude-sonnet', [])

    payloads, _ = _parse_sse_payloads(handler)

    error_events = [p for p in payloads if p['type'] == 'error']
    assert len(error_events) == 1
    assert 'Auth failed' in error_events[0]['error']
    handler.finish.assert_called_once()


# --- New serializer features ---

@pytest.mark.asyncio
async def test_chat_openai_stop_reason():
    handler = MagicMock()
    completed_response = MagicMock()
    completed_response.status = 'completed'
    completed_response.incomplete_details = None
    events = [
        _make_event('response.in_progress'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.delta', delta='{"messages":[{"role":"assistant","content":"ok"}],"actions":[]}'),
        _make_event('response.output_text.done', text='{"messages":[{"role":"assistant","content":"ok"}],"actions":[]}'),
        _make_event('response.completed', response=completed_response),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done']
    assert len(done) == 1
    assert done[0]['stop_reason'] == 'completed'


@pytest.mark.asyncio
async def test_chat_openai_stop_reason_incomplete():
    handler = MagicMock()
    incomplete = MagicMock()
    incomplete.reason = 'max_tokens'
    completed_response = MagicMock()
    completed_response.status = 'incomplete'
    completed_response.incomplete_details = incomplete
    events = [
        _make_event('response.in_progress'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.done', text='partial'),
        _make_event('response.completed', response=completed_response),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['stop_reason'] == 'max_tokens'


@pytest.mark.asyncio
async def test_chat_openai_reasoning_done():
    handler = MagicMock()
    events = [
        _make_event('response.in_progress'),
        _make_event('response.reasoning_summary_text.delta', delta='Step 1'),
        _make_event('response.reasoning_summary_text.done', text='Step 1. Step 2.'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.done', text='answer'),
        _make_event('response.completed', response=MagicMock(status='completed', incomplete_details=None)),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [])

    payloads, _ = _parse_sse_payloads(handler)

    thinking_stops = [p for p in payloads
                      if p['type'] == 'content_block_stop' and p['content_type'] == 'thinking']
    assert any(s.get('text') == 'Step 1. Step 2.' for s in thinking_stops)


@pytest.mark.asyncio
async def test_chat_anthropic_stop_reason():
    handler = MagicMock()
    events = [
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text='Hi')),
        _make_anthropic_event('content_block_stop'),
    ]

    mock_stream = _AsyncStreamCtx(events, final_text='Hi',
                                  stop_reason='max_tokens')
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['stop_reason'] == 'max_tokens'




# --- chat_echo (streaming) ---

@pytest.mark.asyncio
async def test_chat_echo_trigger_action():
    handler = MagicMock()
    messages = [{'role': 'user', 'content': 'show me the toc'}]

    await chat_echo(handler, messages)

    payloads, written = _parse_sse_payloads(handler)

    # Lifecycle: thinking -> text -> message_done
    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    assert starts == ['thinking', 'text']

    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']
    assert 'thinking' in stops
    assert 'text' in stops

    done = [p for p in payloads if p['type'] == 'message_done']
    assert len(done) == 1
    body = json.loads(done[0]['text'])
    assert body['actions'][0]['type'] == 'getToc'
    assert body['messages'][0]['role'] == 'assistant'

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_echo_action_results_passthrough():
    handler = MagicMock()
    messages = [{'role': 'user', 'content': '[Action Results]\n{"toc": [...]}'}]

    await chat_echo(handler, messages)

    payloads, _ = _parse_sse_payloads(handler)

    done = [p for p in payloads if p['type'] == 'message_done']
    body = json.loads(done[0]['text'])
    assert body['actions'] == []
    assert '[Action Results]' in body['messages'][0]['content']


@pytest.mark.asyncio
async def test_chat_echo_default_action_when_no_trigger():
    handler = MagicMock()
    messages = [{'role': 'user', 'content': 'hello world'}]

    await chat_echo(handler, messages)

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done']
    body = json.loads(done[0]['text'])
    # Default trigger is 'toc'
    assert body['actions'][0]['type'] == 'getToc'


# --- sse_serializer decorator ---

@pytest.mark.asyncio
async def test_sse_serializer_calls_init_and_finish():
    handler = MagicMock()

    @sse_serializer
    async def serializer(h):
        _send_sse(h, {'type': 'content_block_start', 'content_type': 'text'})

    await serializer(handler)

    # headers set, [DONE] emitted, finish called
    handler.set_header.assert_any_call('Content-Type', 'text/event-stream')
    written = [call[0][0] for call in handler.write.call_args_list]
    assert any(w == 'data: [DONE]\n\n' for w in written)
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_sse_serializer_emits_error_and_finishes_on_exception():
    handler = MagicMock()

    @sse_serializer
    async def serializer(h):
        raise RuntimeError('boom')

    await serializer(handler)

    payloads, written = _parse_sse_payloads(handler)
    errors = [p for p in payloads if p['type'] == 'error']
    assert len(errors) == 1
    assert 'boom' in errors[0]['error']
    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


# --- Anthropic: unknown block types are ignored ---

@pytest.mark.asyncio
async def test_chat_anthropic_unknown_block_type_ignored():
    handler = MagicMock()
    events = [
        # Unsupported block type should neither emit start nor crash
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('tool_use', name='foo')),
        _make_anthropic_event('content_block_stop'),
        # Regular text block still works
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text='ok')),
        _make_anthropic_event('content_block_stop'),
    ]

    mock_stream = _AsyncStreamCtx(events, final_text='ok',
                                  stop_reason='end_turn')
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, _ = _parse_sse_payloads(handler)

    # Only text block should appear, no tool_use or empty content_type
    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']
    assert starts == ['text']
    assert stops == ['text']


# --- chat_bedrock_converse ---

def _encode_es_string_header(name, value):
    name_b = name.encode('utf-8')
    value_b = value.encode('utf-8')
    return (
        bytes([len(name_b)]) + name_b
        + bytes([7])
        + len(value_b).to_bytes(2, 'big') + value_b
    )


def _encode_es_frame(headers, payload):
    """Build an AWS event-stream frame (zeroed CRCs)."""
    headers_blob = b''.join(_encode_es_string_header(k, v) for k, v in headers.items())
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    total_length = 12 + len(headers_blob) + len(payload) + 4
    return (
        total_length.to_bytes(4, 'big')
        + len(headers_blob).to_bytes(4, 'big')
        + b'\x00\x00\x00\x00'
        + headers_blob
        + payload
        + b'\x00\x00\x00\x00'
    )


class _FakeStreamResponse:
    def __init__(self, status_code=200, chunks=(), error_body=b''):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._error_body = error_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def aread(self):
        return self._error_body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient(...) — async ctx mgr returning self."""
    def __init__(self, response, captured):
        self._response = response
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, headers=None, content=None):
        self._captured['method'] = method
        self._captured['url'] = url
        self._captured['headers'] = headers
        self._captured['content'] = content
        return self._response


def test_build_bedrock_converse_body_basic():
    body = _build_bedrock_converse_body(
        [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'system', 'content': 'Be concise.'},
            {'role': 'user', 'content': 'Hello',
             'actions': [{'type': 'getToc'}]},
            {'role': 'assistant', 'content': 'Hi!'},
        ],
        model='us.anthropic.claude-haiku-4-5-20251001-v1:0',
    )
    assert body['system'] == [
        {'text': 'You are helpful.'},
        {'text': 'Be concise.'},
    ]
    assert body['inferenceConfig'] == {'maxTokens': 32000}
    assert body['messages'][0]['role'] == 'user'
    user_text = body['messages'][0]['content'][0]['text']
    assert user_text.startswith('Hello')
    assert '[Actions proposed]' in user_text
    assert '"getToc"' in user_text
    assert body['messages'][1] == {'role': 'assistant', 'content': [{'text': 'Hi!'}]}
    # Claude in model id -> thinking enabled
    assert body['additionalModelRequestFields'] == {
        'thinking': {'type': 'enabled', 'budget_tokens': 2000}
    }


def test_build_bedrock_converse_body_no_thinking_for_non_claude():
    body = _build_bedrock_converse_body(
        [{'role': 'user', 'content': 'Hi'}],
        model='meta.llama3-8b-instruct-v1:0',
    )
    assert 'additionalModelRequestFields' not in body


@pytest.mark.asyncio
async def test_chat_bedrock_converse_basic_flow():
    handler = MagicMock()
    json_text = '{"messages":[{"role":"assistant","content":"Hi there!"}],"actions":[]}'

    # Stream simulates Bedrock Converse: text deltas, block stop, message stop.
    chunks = [
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'delta': {'text': '{"messages":[{"role":"assistant","content":"Hi'}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'delta': {'text': ' there!"}],"actions":[]}'}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockStop', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0}),
        ),
        _encode_es_frame(
            {':event-type': 'messageStop', ':message-type': 'event'},
            json.dumps({'stopReason': 'end_turn'}),
        ),
    ]

    captured = {}
    response = _FakeStreamResponse(status_code=200, chunks=chunks)
    with patch('jupyter_mynerva.routes.httpx.AsyncClient',
               return_value=_FakeAsyncClient(response, captured)):
        await chat_bedrock_converse(
            handler, 'sk-bedrock', 'us-west-2',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            [{'role': 'user', 'content': 'Hi'}],
        )

    # URL encodes the model ID (colon escaped); points at bedrock-runtime.
    assert captured['url'] == (
        'https://bedrock-runtime.us-west-2.amazonaws.com'
        '/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream'
    )
    assert captured['method'] == 'POST'
    assert captured['headers']['Authorization'] == 'Bearer sk-bedrock'
    assert captured['headers']['Accept'] == 'application/vnd.amazon.eventstream'

    payloads, written = _parse_sse_payloads(handler)
    types = [p['type'] for p in payloads]
    assert 'content_block_start' in types
    assert 'content_block_delta' in types
    assert 'content_block_stop' in types
    assert 'message_done' in types

    text_deltas = [p['delta'] for p in payloads
                   if p['type'] == 'content_block_delta'
                   and p['content_type'] == 'text']
    # _extract_json_content extracts the running 'content' field value.
    assert text_deltas[0] == 'Hi'
    assert text_deltas[-1] == 'Hi there!'

    done = [p for p in payloads if p['type'] == 'message_done'][0]
    assert done['text'] == json_text
    assert done['stop_reason'] == 'end_turn'

    assert written[-1] == 'data: [DONE]\n\n'


@pytest.mark.asyncio
async def test_chat_bedrock_converse_thinking_then_text():
    handler = MagicMock()
    chunks = [
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'delta': {'reasoningContent': {'text': 'Let me think'}}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'delta': {'reasoningContent': {'text': ' about this.'}}}),
        ),
        # Switch to text block; emitter must close thinking and open text.
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'delta': {'text': '{"messages":[{"role":"assistant","content":"Done"}]}'}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockStop', ':message-type': 'event'},
            json.dumps({}),
        ),
        _encode_es_frame(
            {':event-type': 'messageStop', ':message-type': 'event'},
            json.dumps({'stopReason': 'end_turn'}),
        ),
    ]
    response = _FakeStreamResponse(status_code=200, chunks=chunks)
    with patch('jupyter_mynerva.routes.httpx.AsyncClient',
               return_value=_FakeAsyncClient(response, {})):
        await chat_bedrock_converse(
            handler, 'k', 'us-east-1',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            [],
        )

    payloads, _ = _parse_sse_payloads(handler)
    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']

    # Thinking opened first, then closed when text starts; text closed at end.
    assert starts == ['thinking', 'text']
    assert stops == ['thinking', 'text']

    thinking_deltas = [p['delta'] for p in payloads
                       if p['type'] == 'content_block_delta'
                       and p['content_type'] == 'thinking']
    assert thinking_deltas == ['Let me think', ' about this.']


@pytest.mark.asyncio
async def test_chat_bedrock_converse_http_error():
    handler = MagicMock()
    response = _FakeStreamResponse(
        status_code=404,
        chunks=[],
        error_body=b'{"message":"use case not approved"}',
    )
    with patch('jupyter_mynerva.routes.httpx.AsyncClient',
               return_value=_FakeAsyncClient(response, {})):
        await chat_bedrock_converse(
            handler, 'k', 'us-east-1',
            'us.anthropic.claude-haiku-4-5-20251001-v1:0',
            [],
        )

    payloads, _ = _parse_sse_payloads(handler)
    errors = [p for p in payloads if p['type'] == 'error']
    assert errors
    assert 'Bedrock Converse error (404)' in errors[0]['error']
    assert 'use case not approved' in errors[0]['error']


@pytest.mark.asyncio
async def test_chat_bedrock_converse_exception_frame():
    handler = MagicMock()
    chunks = [
        _encode_es_frame(
            {':message-type': 'exception',
             ':exception-type': 'ValidationException'},
            json.dumps({'message': 'bad input'}),
        ),
    ]
    response = _FakeStreamResponse(status_code=200, chunks=chunks)
    with patch('jupyter_mynerva.routes.httpx.AsyncClient',
               return_value=_FakeAsyncClient(response, {})):
        await chat_bedrock_converse(
            handler, 'k', 'us-east-1',
            'us.anthropic.claude-haiku-4-5-20251001-v1:0',
            [],
        )

    payloads, _ = _parse_sse_payloads(handler)
    errors = [p for p in payloads if p['type'] == 'error']
    assert errors
    assert 'ValidationException' in errors[0]['error']
    assert 'bad input' in errors[0]['error']


# --- _load_bedrock_regions / _validate_bedrock_region ---

def test_load_bedrock_regions_returns_list_with_id_and_name():
    regions = _load_bedrock_regions()
    assert len(regions) > 0
    for r in regions:
        assert 'id' in r
        assert 'name' in r


@pytest.mark.parametrize('region', [
    'us-east-1', 'ap-northeast-1', 'eu-central-2', 'me-south-1', 'il-central-1',
])
def test_validate_bedrock_region_accepts_valid(region):
    _validate_bedrock_region(region)  # should not raise


@pytest.mark.parametrize('region', [
    'evil.com/', '../foo', 'us east 1', 'US-EAST-1', '', 'us-east-1;rm -rf /',
    'xx-fake-99',
])
def test_validate_bedrock_region_rejects_invalid(region):
    with pytest.raises(ValueError, match='Invalid AWS region'):
        _validate_bedrock_region(region)


# --- _fetch_bedrock_models ---

class _FakeSyncResponse:
    def __init__(self, status_code=200, json_data=None, text=''):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class _FakeSyncClient:
    def __init__(self, response, captured=None):
        self._response = response
        self._captured = captured if captured is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None):
        self._captured['url'] = url
        self._captured['headers'] = headers
        return self._response


def test_fetch_bedrock_models_filters_active_system_defined():
    _bedrock_models_cache.clear()
    captured = {}
    response = _FakeSyncResponse(
        status_code=200,
        json_data={
            'inferenceProfileSummaries': [
                # included
                {'inferenceProfileId': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
                 'status': 'ACTIVE', 'type': 'SYSTEM_DEFINED'},
                # included
                {'inferenceProfileId': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                 'status': 'ACTIVE', 'type': 'SYSTEM_DEFINED'},
                # excluded: APPLICATION_PROFILE (user-created)
                {'inferenceProfileId': 'user-defined-1',
                 'status': 'ACTIVE', 'type': 'APPLICATION_PROFILE'},
                # excluded: INACTIVE
                {'inferenceProfileId': 'us.anthropic.claude-3-0-legacy-v1:0',
                 'status': 'INACTIVE', 'type': 'SYSTEM_DEFINED'},
            ]
        },
    )
    with patch('jupyter_mynerva.routes.httpx.Client',
               return_value=_FakeSyncClient(response, captured)):
        models = _fetch_bedrock_models('sk-key', 'us-west-2')

    assert captured['url'] == 'https://bedrock.us-west-2.amazonaws.com/inference-profiles'
    assert captured['headers']['Authorization'] == 'Bearer sk-key'
    assert models == [
        'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    ]


def test_fetch_bedrock_models_cache():
    _bedrock_models_cache.clear()
    _bedrock_models_cache[('eu-central-1', 'k')] = ['cached-model']

    result = _fetch_bedrock_models('k', 'eu-central-1')
    assert result == ['cached-model']


def test_fetch_bedrock_models_cache_keyed_by_api_key():
    """Regression guard: swapping the API key against the same region must
    re-fetch rather than return another user's cached models."""
    _bedrock_models_cache.clear()
    _bedrock_models_cache[('us-east-1', 'key-A')] = ['from-A']

    response = _FakeSyncResponse(
        status_code=200,
        json_data={'inferenceProfileSummaries': [
            {'inferenceProfileId': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
             'status': 'ACTIVE', 'type': 'SYSTEM_DEFINED'},
        ]},
    )
    with patch('jupyter_mynerva.routes.httpx.Client',
               return_value=_FakeSyncClient(response)):
        result = _fetch_bedrock_models('key-B', 'us-east-1')

    assert result == ['us.anthropic.claude-sonnet-4-5-20250929-v1:0']


def test_fetch_bedrock_models_empty_after_filter_raises():
    _bedrock_models_cache.clear()
    response = _FakeSyncResponse(
        status_code=200,
        json_data={'inferenceProfileSummaries': [
                {'inferenceProfileId': 'us.anthropic.claude-3-0-legacy-v1:0',
                 'status': 'INACTIVE', 'type': 'SYSTEM_DEFINED'},
        ]},
    )
    with patch('jupyter_mynerva.routes.httpx.Client',
               return_value=_FakeSyncClient(response)):
        with pytest.raises(ValueError, match='No matching inference profiles'):
            _fetch_bedrock_models('k', 'us-east-1')


def test_fetch_bedrock_models_http_error_raises():
    _bedrock_models_cache.clear()
    response = _FakeSyncResponse(
        status_code=403,
        json_data=None,
        text='{"message":"not authorized"}',
    )
    with patch('jupyter_mynerva.routes.httpx.Client',
               return_value=_FakeSyncClient(response)):
        with pytest.raises(ValueError, match=r'Bedrock list-profiles error \(403\).*not authorized'):
            _fetch_bedrock_models('k', 'us-east-1')


# --- BedrockModelsHandler ---

def test_bedrock_models_handler_success(monkeypatch):
    monkeypatch.setattr('jupyter_mynerva.routes._fetch_bedrock_models',
                        lambda key, region: ['us.anthropic.claude-haiku-4-5-20251001-v1:0'])
    handler = _make_models_handler({'region': 'us-east-1', 'apiKey': 'k'})

    BedrockModelsHandler.post(handler)

    written = handler.finish.call_args[0][0]
    assert json.loads(written) == {
        'models': ['us.anthropic.claude-haiku-4-5-20251001-v1:0']
    }
    handler.set_status.assert_not_called()


def test_bedrock_models_handler_missing_key_returns_400():
    handler = _make_models_handler({'region': 'us-east-1'})

    BedrockModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(400)
    body = json.loads(handler.finish.call_args[0][0])
    assert 'apiKey' in body['error']


def test_bedrock_models_handler_default_region(monkeypatch):
    captured = {}

    def fake_fetch(key, region):
        captured['args'] = (key, region)
        return ['m']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_bedrock_models', fake_fetch)
    handler = _make_models_handler({'apiKey': 'k'})  # no region

    BedrockModelsHandler.post(handler)

    assert captured['args'] == ('k', 'us-east-1')


def test_bedrock_models_handler_returns_500_with_error_body(monkeypatch):
    def fake_fetch(key, region):
        raise RuntimeError('use case not approved')

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_bedrock_models', fake_fetch)
    handler = _make_models_handler({'region': 'us-east-1', 'apiKey': 'k'})

    BedrockModelsHandler.post(handler)

    handler.set_status.assert_called_once_with(500)
    body = json.loads(handler.finish.call_args[0][0])
    assert 'use case not approved' in body['error']


# --- _get_provider_models bedrock admin-default path ---

async def test_get_provider_models_bedrock_uses_admin_key_and_region(monkeypatch):
    captured = {}

    def fake_fetch(key, region):
        captured['args'] = (key, region)
        return ['us.anthropic.claude-sonnet-4-5-20250929-v1:0']

    monkeypatch.setattr('jupyter_mynerva.routes._fetch_bedrock_models', fake_fetch)

    result = await _get_provider_models('bedrock', api_key='admin-bedrock', region='ap-northeast-1')
    assert result == ['us.anthropic.claude-sonnet-4-5-20250929-v1:0']
    assert captured['args'] == ('admin-bedrock', 'ap-northeast-1')


async def test_get_provider_models_bedrock_no_key_returns_empty(monkeypatch):
    assert await _get_provider_models('bedrock') == []


async def test_get_provider_models_bedrock_defaults_region_to_us_east_1(monkeypatch):
    captured = {}
    monkeypatch.setattr('jupyter_mynerva.routes._fetch_bedrock_models',
                        lambda key, region: captured.setdefault('region', region) or ['m'])

    await _get_provider_models('bedrock', api_key='k')
    assert captured['region'] == 'us-east-1'


# --- get_default_config bedrock-as-default ---

async def test_get_default_config_picks_bedrock_when_only_bedrock_configured(monkeypatch):
    from jupyter_mynerva.routes import get_default_config
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'bedrock_api_key': 'k', 'bedrock_region': 'us-west-2',
    })

    async def fake_get_models(provider_id):
        return ['us.anthropic.claude-haiku-4-5-20251001-v1:0']

    monkeypatch.setattr('jupyter_mynerva.routes._get_default_provider_models',
                        fake_get_models)

    defaults = await get_default_config()
    assert defaults['provider'] == 'bedrock'
    assert defaults['model'] == 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
    assert defaults['bedrockRegion'] == 'us-west-2'


async def test_get_default_config_multi_provider_requires_explicit(monkeypatch):
    from jupyter_mynerva.routes import get_default_config
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'o',
        'bedrock_api_key': 'b',
    })
    assert await get_default_config() is None


async def test_get_default_config_multi_provider_with_explicit(monkeypatch):
    from jupyter_mynerva.routes import get_default_config
    monkeypatch.setattr('jupyter_mynerva.routes._DEFAULT_CONFIG', {
        'openai_api_key': 'o',
        'bedrock_api_key': 'b',
        'provider': 'bedrock',
    })

    async def fake_get_models(provider_id):
        return ['us.anthropic.claude-haiku-4-5-20251001-v1:0']

    monkeypatch.setattr('jupyter_mynerva.routes._get_default_provider_models',
                        fake_get_models)

    defaults = await get_default_config()
    assert defaults['provider'] == 'bedrock'

