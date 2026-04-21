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
    _convert_messages_for_responses_api,
    _build_anthropic_params,
    _extract_json_content,
    _send_sse,
    sse_serializer,
    chat_openai,
    chat_anthropic,
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.side_effect = Exception('API key invalid')
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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
    assert params['max_tokens'] == 4096


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
    events = [
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text='Hello')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text=' world')),
        _make_anthropic_event('content_block_stop'),
        _make_anthropic_event('message_stop'),
    ]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.get_final_text = MagicMock(return_value='Hello world')
    final_msg = MagicMock()
    final_msg.stop_reason = 'end_turn'
    mock_stream.get_final_message = MagicMock(return_value=final_msg)

    with patch('jupyter_mynerva.routes.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = mock_stream
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, written = _parse_sse_payloads(handler)

    types = [p['type'] for p in payloads]
    assert 'content_block_start' in types
    assert 'content_block_delta' in types
    assert 'content_block_stop' in types
    assert 'message_done' in types

    text_deltas = [p for p in payloads
                   if p['type'] == 'content_block_delta' and p['content_type'] == 'text']
    assert text_deltas[0]['delta'] == 'Hello'
    assert text_deltas[1]['delta'] == ' world'

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['text'] == 'Hello world'
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

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.get_final_text = MagicMock(return_value='Answer')

    with patch('jupyter_mynerva.routes.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = mock_stream
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

    with patch('jupyter_mynerva.routes.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream.side_effect = Exception('Auth failed')
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    with patch('jupyter_mynerva.routes.OpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create.return_value = events
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

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.get_final_text = MagicMock(return_value='Hi')

    final_msg = MagicMock()
    final_msg.stop_reason = 'max_tokens'
    mock_stream.get_final_message = MagicMock(return_value=final_msg)

    with patch('jupyter_mynerva.routes.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = mock_stream
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

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.get_final_text = MagicMock(return_value='ok')
    final_msg = MagicMock()
    final_msg.stop_reason = 'end_turn'
    mock_stream.get_final_message = MagicMock(return_value=final_msg)

    with patch('jupyter_mynerva.routes.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = mock_stream
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, _ = _parse_sse_payloads(handler)

    # Only text block should appear, no tool_use or empty content_type
    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']
    assert starts == ['text']
    assert stops == ['text']
