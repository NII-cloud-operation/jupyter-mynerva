import asyncio
import json
import logging
import os
from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tornado.web
from cryptography.fernet import Fernet
from tornado.httputil import HTTPServerRequest

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
    _build_openai_tools,
    _build_openai_input,
    _build_anthropic_params,
    _anthropic_thinking_config,
    _build_bedrock_converse_body,
    _send_sse,
    sse_serializer,
    chat_openai,
    chat_anthropic,
    chat_bedrock_converse,
)
from jupyter_mynerva.echo_agent import chat_echo
from jupyter_mynerva.handlers.nbsearch import (
    _NBSEARCH_CELLS_PAGE_LIMIT_BYTES,
    _NBSEARCH_REFERENCE_CACHE,
    NBSearchHandler,
    _build_nbsearch_filter_queries,
    _build_nbsearch_query,
    _paginate_cells_result,
    _source_only_cells,
    _NBSEARCH_SUMMARY_MAX_DEPTH,
    _is_context_exceeded,
    _truncate_for_floor,
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


# --- nbsearch query helpers ---

class _FakeConnection:
    def set_close_callback(self, callback):
        pass


def _make_nbsearch_handler(config):
    app = tornado.web.Application([], config=config)
    request = HTTPServerRequest(method='POST', uri='/jupyter-mynerva/nbsearch/notebooks')
    request.connection = _FakeConnection()
    return NBSearchHandler(app, request)


def test_build_nbsearch_query_uses_raw_solr_query():
    query = _build_nbsearch_query(
        {
            'query': 'filename:*BinderHub* AND source__markdown__heading_1:構築',
        }
    )
    assert query == 'filename:*BinderHub* AND source__markdown__heading_1:構築'


def test_build_nbsearch_filter_queries_uses_normalized_datetime_range():
    filters = _build_nbsearch_filter_queries(
        {
            'dateFrom': '2026-01-01',
            'dateTo': '2026-01-31',
            'dateTimeFrom': '2025-12-31T15:00:00.000Z',
            'dateTimeTo': '2026-01-31T15:00:00.000Z',
        },
        {
            'owner': 'owner',
            'filename': 'filename',
            'mtime': 'mtime',
        },
    )
    assert filters == [
        'mtime:[2025-12-31T15:00:00.000Z TO 2026-01-31T15:00:00.000Z]',
        '-mtime:"2026-01-31T15:00:00.000Z"',
    ]


def test_source_only_cells_flattens_and_drops_outputs():
    cells_result = {
        'cells': [
            {
                '_index': 10,
                'cell_type': 'code',
                'source': ['print(1)'],
                'outputs': [{'text': ['big output']}],
                'metadata': {'lc_cell_meme': {'current': 'meme'}},
                'execution_count': 1,
                '_hash': 'hash',
            },
            {
                '_index': 11,
                'cell_type': 'markdown',
                'source': ['# title'],
            },
        ],
    }

    assert _source_only_cells(cells_result) == [
        {'_index': 10, 'cell_type': 'code', 'source': ['print(1)']},
        {'_index': 11, 'cell_type': 'markdown', 'source': ['# title']},
    ]


@pytest.mark.asyncio
async def test_summarize_feeds_flat_source_only_cell_list_to_map_reduce(monkeypatch):
    # Regression: map-reduce must receive the actual cell list (so it can split
    # to fit context), not a single wrapper object.
    async def fake_load_config():
        return {}

    async def fake_resolve(config):
        return ('openai', 'm', 'k', '')

    monkeypatch.setattr('jupyter_mynerva.routes.load_config', fake_load_config)
    monkeypatch.setattr('jupyter_mynerva.routes.resolve_chat_config', fake_resolve)

    captured = {}

    async def fake_adaptive(provider, model, api_key, base_url, config,
                            build_messages, segments):
        captured['segments'] = segments
        captured['payload'] = json.loads(build_messages(segments)[1]['content'])
        return 'summary'

    handler = MagicMock()
    handler._summarize_adaptive = fake_adaptive

    cells_result = {
        'cells': [
            {'_index': 0, 'cell_type': 'code', 'source': ['a'], 'outputs': ['x']},
            {'_index': 1, 'cell_type': 'code', 'source': ['b']},
        ],
    }

    out = await NBSearchHandler._summarize(
        handler, 'focus', 'f.ipynb', cells_result, 'instruction', 'echo-text')

    assert out == 'summary'
    # Flat list of source-only cells (outputs dropped), each keeping _index.
    assert captured['segments'] == [
        {'_index': 0, 'cell_type': 'code', 'source': ['a']},
        {'_index': 1, 'cell_type': 'code', 'source': ['b']},
    ]
    assert captured['payload']['cells'] == captured['segments']


def test_paginate_cells_result_uses_budget_and_next_start():
    result = _paginate_cells_result(
        {
            'cells': [
                {'_index': 0, 'source': ['a']},
                {'_index': 1, 'source': ['x' * _NBSEARCH_CELLS_PAGE_LIMIT_BYTES]},
                {'_index': 2, 'source': ['c']},
            ],
        },
        start=0,
    )

    assert result == {
        'cells': [{'_index': 0, 'source': ['a']}],
        'total': 3,
        'hasMore': True,
        'nextStart': 1,
    }


def test_paginate_cells_result_honors_start_and_limit():
    result = _paginate_cells_result(
        {
            'cells': [
                {'_index': 0, 'source': ['a']},
                {'_index': 1, 'source': ['b']},
                {'_index': 2, 'source': ['c']},
            ],
        },
        start=1,
        limit=1,
    )

    assert result == {
        'cells': [{'_index': 1, 'source': ['b']}],
        'total': 3,
        'hasMore': True,
        'nextStart': 2,
    }


async def test_search_nbsearch_notebooks_queries_notebook_core(monkeypatch):
    class FakeDB:
        solr_base_url = 'http://solr:8983'
        solr_basic_auth_username = ''
        solr_basic_auth_password = ''
        solr_notebook = 'jupyter-notebook'

        def __init__(self, config):
            self.config = config

    response = MagicMock()
    response.code = 200
    response.body = json.dumps({
        'response': {
            'numFound': 1,
            'start': 0,
            'docs': [{
                'id': 'notebook',
                'filename': 'foo.ipynb',
                'owner': 'alice',
                'server': 'http://localhost:8000/',
                'mtime': '2026-05-01T00:00:00Z',
                'ctime': '2026-04-30T00:00:00Z',
                'atime': '2026-05-02T00:00:00Z',
                'source__markdown__heading': '## Data',
                'source__markdown__heading_count': '1',
                'score': 1.0,
            }],
        },
    }).encode()
    client = MagicMock()
    client.fetch = AsyncMock(return_value=response)

    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.NBSearchDB', FakeDB)
    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.AsyncHTTPClient', lambda: client)
    async def fake_get_search_reference_cells(self, db, reference, no_filter):
        assert reference['query'] == {'start': 0}
        assert reference['count'] >= 10000
        return {'cells': [{'cell_type': 'code', 'source': ['print(1)']}]}

    async def fake_summarize_result(self, focus, path, cells):
        return 'focus に関連する notebook です。'

    monkeypatch.setattr(NBSearchHandler, '_get_search_reference_cells', fake_get_search_reference_cells)
    monkeypatch.setattr(NBSearchHandler, '_summarize_result', fake_summarize_result)

    handler = _make_nbsearch_handler(
        {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
    )
    result = await handler._search_notebooks(
        'notebooks',
        {
            'query': 'filename:*foo* AND pandas',
            'focus': 'pandas usage',
            'dateFrom': '2026-05-01',
            'dateTo': '2026-05-01',
            'dateTimeFrom': '2026-04-30T15:00:00.000Z',
            'dateTimeTo': '2026-05-01T15:00:00.000Z',
            'limit': 5,
            'sort': 'mtime desc',
        },
    )

    request = client.fetch.call_args[0][0]
    requested_url = request.url
    assert '/solr/jupyter-notebook/select?' in requested_url
    params = parse_qs(urlparse(requested_url).query)
    assert 'group' not in params
    assert 'group.field' not in params
    assert params['q'] == ['filename:*foo* AND pandas']
    assert params['fl'][0] == (
        'id,filename,owner,server,mtime,ctime,atime,source__markdown__heading,'
        'source__markdown__heading_count,lc_cell_memes,lc_cell_meme__execution_end_time,score'
    )
    assert params['fq'] == [
        'mtime:[2026-04-30T15:00:00.000Z TO 2026-05-01T15:00:00.000Z]',
        '-mtime:"2026-05-01T15:00:00.000Z"',
    ]
    assert params['sort'] == ['mtime desc']
    assert result['focus'] == 'pandas usage'
    assert result['query'] == 'filename:*foo* AND pandas'
    assert result['results'][0] == {
        'filename': 'foo.ipynb',
        'owner': 'alice',
        'server': 'http://localhost:8000/',
        'mtime': '2026-05-01T00:00:00Z',
        'ctime': '2026-04-30T00:00:00Z',
        'atime': '2026-05-02T00:00:00Z',
        'summary': 'focus に関連する notebook です。',
        'references': [
            {
                'type': 'summaryCellsFromSearch',
                'referenceId': result['results'][0]['references'][0]['referenceId'],
            },
        ],
    }
    assert result['results'][0]['references'][0]['referenceId'].endswith('/ref1')


@pytest.mark.asyncio
async def test_summary_cells_from_search_returns_locator_summary(monkeypatch):
    class FakeDB:
        def __init__(self, config):
            self.solr_base_url = 'http://solr:8983'
            self.solr_notebook = 'notebooks'
            self.solr_basic_auth_username = None
            self.solr_basic_auth_password = None

    captured = {}

    async def fake_get_search_reference_cells(self, db, reference, no_filter):
        captured['no_filter'] = no_filter
        return {
            'cells': [
                {'_index': 12, 'cell_type': 'markdown', 'source': 'Galaxy 更新前確認'},
                {'_index': 13, 'cell_type': 'code', 'source': 'run_update()'},
            ],
        }

    async def fake_summarize_search_cells(self, focus, filename, cells, coverage):
        captured['focus'] = focus
        captured['filename'] = filename
        captured['cells'] = cells
        captured['coverage'] = coverage
        return 'セル12-13でGalaxy更新前確認と更新処理をしている。'

    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.NBSearchDB', FakeDB)
    monkeypatch.setattr(NBSearchHandler, '_get_search_reference_cells', fake_get_search_reference_cells)
    monkeypatch.setattr(NBSearchHandler, '_summarize_search_cells', fake_summarize_search_cells)

    reference_id = 'search-test/r1/ref1'
    _NBSEARCH_REFERENCE_CACHE[reference_id] = {
        'filename': 'galaxy.ipynb',
        'notebookId': 'notebook',
        'query': {'start': 0},
        'count': 10000,
    }
    try:
        handler = _make_nbsearch_handler(
            {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
        )
        result = await handler._summary_cells_from_search({
            'referenceId': reference_id,
            'focus': 'Galaxy更新の手順を確認したい',
            'noFilter': True,
        })
    finally:
        _NBSEARCH_REFERENCE_CACHE.pop(reference_id, None)

    assert captured['no_filter'] is True
    assert captured['focus'] == 'Galaxy更新の手順を確認したい'
    assert captured['filename'] == 'galaxy.ipynb'
    assert captured['coverage'] == 'full 2/2 cells'
    assert result == {
        'type': 'summaryCellsFromSearch',
        'referenceId': reference_id,
        'filename': 'galaxy.ipynb',
        'cellCount': 2,
        'coverage': 'full 2/2 cells',
        'summary': 'セル12-13でGalaxy更新前確認と更新処理をしている。',
        'readAction': {
            'type': 'getCellsFromSearch',
            'referenceId': reference_id,
        },
    }


@pytest.mark.asyncio
async def test_search_nbsearch_notebooks_requires_focus(monkeypatch):
    class FakeDB:
        def __init__(self, config):
            self.solr_base_url = 'http://solr:8983'
            self.solr_notebook = 'notebooks'
            self.solr_basic_auth_username = None
            self.solr_basic_auth_password = None

    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.NBSearchDB', FakeDB)

    with pytest.raises(tornado.web.HTTPError) as error:
        handler = _make_nbsearch_handler(
            {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
        )
        await handler._search_notebooks(
            'notebooks',
            {'query': 'pandas'},
        )

    assert error.value.status_code == 400
    assert error.value.reason == 'focus is required'


@pytest.mark.asyncio
async def test_search_nbsearch_notebooks_requires_normalized_date_range(monkeypatch):
    class FakeDB:
        def __init__(self, config):
            self.solr_base_url = 'http://solr:8983'
            self.solr_notebook = 'notebooks'
            self.solr_basic_auth_username = None
            self.solr_basic_auth_password = None

    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.NBSearchDB', FakeDB)

    with pytest.raises(tornado.web.HTTPError) as error:
        handler = _make_nbsearch_handler(
            {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
        )
        await handler._search_notebooks(
            'notebooks',
            {'query': 'pandas', 'focus': 'pandas usage', 'dateFrom': '2026-01-01'},
        )

    assert error.value.status_code == 400
    assert error.value.reason == 'dateFrom must be normalized to dateTimeFrom by the client'


@pytest.mark.asyncio
async def test_search_nbsearch_notebooks_rejects_unsupported_sort(monkeypatch):
    class FakeDB:
        def __init__(self, config):
            self.solr_base_url = 'http://solr:8983'
            self.solr_notebook = 'notebooks'
            self.solr_basic_auth_username = None
            self.solr_basic_auth_password = None

    monkeypatch.setattr('jupyter_mynerva.handlers.nbsearch.NBSearchDB', FakeDB)

    with pytest.raises(tornado.web.HTTPError) as error:
        handler = _make_nbsearch_handler(
            {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
        )
        await handler._search_notebooks(
            'notebooks',
            {
                'query': 'pandas',
                'focus': 'pandas usage',
                'sort': 'score desc',
            },
        )

    assert error.value.status_code == 400
    assert error.value.reason == 'unsupported sort: score desc'


@pytest.mark.asyncio
async def test_get_search_reference_cells_returns_empty_for_invalid_payload(monkeypatch, caplog):
    class FakeDB:
        async def download_file(self, notebook_id, data):
            data.write(b'not valid json')

    handler = _make_nbsearch_handler(
        {'NBSearchDB': {'solr_base_url': 'http://solr:8983'}},
    )

    with caplog.at_level(logging.ERROR):
        result = await handler._get_search_reference_cells(
            FakeDB(),
            {
                'notebookId': 'broken-notebook',
                'filename': 'broken.ipynb',
                'query': {'start': 0},
                'count': 3,
            },
            False,
        )

    assert result == {
        'cells': [],
        'error': 'notebook payload is not valid JSON',
    }
    assert any(
        'Invalid nbsearch notebook payload: notebook_id=broken-notebook filename=broken.ipynb' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_nbsearch_handler_returns_json_for_unhandled_errors():
    # The cheap LLM-free target stays a plain JSON request.
    handler = MagicMock()
    handler._run_target = AsyncMock(side_effect=RuntimeError('provider unavailable'))

    await NBSearchHandler._post_json(handler, 'cells-from-search', {'referenceId': 'r1'})

    handler.log.exception.assert_called_once_with(
        'Unhandled nbsearch request failure: target=%s',
        'cells-from-search',
    )
    handler.set_status.assert_called_once_with(500)
    assert json.loads(handler.finish.call_args[0][0]) == {
        'error': 'nbsearch request failed; see server logs',
    }


@pytest.mark.asyncio
async def test_nbsearch_handler_streams_error_event_for_unhandled_errors():
    # LLM targets stream SSE, so errors arrive as an SSE error event.
    handler = MagicMock()
    handler._run_target = AsyncMock(side_effect=RuntimeError('provider unavailable'))
    handler._sse_heartbeat = AsyncMock()

    await NBSearchHandler._post_streaming(handler, 'notebooks', {'query': 'q', 'focus': 'f'})

    handler.log.exception.assert_called_once_with(
        'Unhandled nbsearch request failure: target=%s',
        'notebooks',
    )
    writes = ''.join(call.args[0] for call in handler.write.call_args_list)
    assert '"type": "error"' in writes
    assert '[DONE]' in writes


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


# --- _build_openai_tools ---

def test_build_openai_tools_wraps_function_shape():
    tools = [
        {'name': 'getToc', 'description': 'Get the table of contents',
         'parameters': {'type': 'object', 'properties': {}}},
    ]
    result = _build_openai_tools(tools)
    assert result == [
        {'type': 'function', 'name': 'getToc',
         'description': 'Get the table of contents',
         'parameters': {'type': 'object', 'properties': {}}},
    ]


# --- _build_openai_input ---

def test_build_openai_input_system_to_developer():
    messages = [
        {'role': 'system', 'content': 'You are an assistant.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = _build_openai_input(messages)
    assert result[0] == {'role': 'developer', 'content': 'You are an assistant.'}
    assert result[1] == {'role': 'user', 'content': 'Hello'}


def test_build_openai_input_plain_assistant_and_user():
    messages = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello'},
    ]
    result = _build_openai_input(messages)
    assert result[0] == {'role': 'user', 'content': 'Hi'}
    assert result[1] == {'role': 'assistant', 'content': 'Hello'}


def test_build_openai_input_missing_role_defaults_to_user():
    result = _build_openai_input([{'content': 'No role specified'}])
    assert result[0] == {'role': 'user', 'content': 'No role specified'}


def test_build_openai_input_missing_content_defaults_to_empty():
    result = _build_openai_input([{'role': 'user'}])
    assert result[0] == {'role': 'user', 'content': ''}


def test_build_openai_input_assistant_blocks_spliced_verbatim():
    blocks = [
        {'type': 'reasoning', 'summary': []},
        {'type': 'function_call', 'call_id': 'call_1', 'name': 'getToc',
         'arguments': '{}'},
    ]
    messages = [{'role': 'assistant', 'assistantBlocks': blocks}]
    result = _build_openai_input(messages)
    assert result == blocks


def test_build_openai_input_tool_results_to_function_call_output():
    messages = [
        {'role': 'user', 'toolResults': [
            {'id': 'call_1', 'result': '{"toc": []}'},
            {'id': 'call_2', 'result': 'oops'},
        ]},
    ]
    result = _build_openai_input(messages)
    assert result == [
        {'type': 'function_call_output', 'call_id': 'call_1',
         'output': '{"toc": []}'},
        {'type': 'function_call_output', 'call_id': 'call_2', 'output': 'oops'},
    ]


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
    def __init__(self, events, final_text='', stop_reason='end_turn',
                 final_content=None):
        self._events = list(events)
        self._final_text = final_text
        self._stop_reason = stop_reason
        self._final_content = final_content

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
        if self._final_content is not None:
            msg.content = self._final_content
        else:
            msg.content = [_make_anthropic_block('text', text=self._final_text)]
        return msg


# --- chat_openai ---

def _make_event(event_type, **kwargs):
    """Create a mock streaming event."""
    event = MagicMock()
    event.type = event_type
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


def _make_output_message(dump=None):
    """A non-function_call Responses output item (e.g. an assistant message)."""
    o = MagicMock()
    o.type = 'message'
    o.model_dump.return_value = dump or {'type': 'message', 'role': 'assistant'}
    return o


def _make_output_function_call(call_id, name, arguments, dump=None):
    """A function_call Responses output item."""
    o = MagicMock()
    o.type = 'function_call'
    o.call_id = call_id
    o.name = name
    o.arguments = arguments
    o.model_dump.return_value = dump or {
        'type': 'function_call', 'call_id': call_id, 'name': name,
        'arguments': arguments,
    }
    return o


def _make_completed_response(output=None, status='completed', incomplete_details=None):
    resp = MagicMock()
    resp.output = output if output is not None else [_make_output_message()]
    resp.status = status
    resp.incomplete_details = incomplete_details
    return resp


@pytest.mark.asyncio
async def test_chat_openai_basic_flow():
    handler = MagicMock()
    # Streaming now emits RAW assistant text (no JSON-envelope extraction).
    events = [
        _make_event('response.created'),
        _make_event('response.in_progress'),
        _make_event('response.output_item.added'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.delta', delta='Hi'),
        _make_event('response.output_text.delta', delta=' there!'),
        _make_event('response.output_text.done', text='Hi there!'),
        _make_event('response.completed', response=_make_completed_response()),
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

    # Raw text deltas are emitted verbatim (incremental, no extraction).
    deltas = [p for p in payloads if p['type'] == 'content_block_delta']
    assert deltas[0]['content_type'] == 'text'
    assert deltas[0]['delta'] == 'Hi'
    assert deltas[1]['delta'] == ' there!'

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['text'] == 'Hi there!'  # accumulated raw text
    assert done[0]['tool_calls'] == []
    assert done[0]['assistant_blocks'] == [{'type': 'message', 'role': 'assistant'}]

    stops = [p for p in payloads if p['type'] == 'content_block_stop']
    assert any(s['content_type'] == 'thinking' for s in stops)
    assert any(s['content_type'] == 'text' for s in stops)

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_openai_tool_call():
    handler = MagicMock()
    fc = _make_output_function_call('call_1', 'getToc', '{"depth": 2}')
    events = [
        _make_event('response.in_progress'),
        _make_event('response.completed',
                    response=_make_completed_response(output=[fc], status='completed')),
    ]

    with patch('jupyter_mynerva.routes.AsyncOpenAI') as MockOpenAI:
        MockOpenAI.return_value.responses.create = AsyncMock(return_value=_async_iter(events))
        await chat_openai(handler, 'key', 'gpt-4o', [], tools=[
            {'name': 'getToc', 'description': 'd',
             'parameters': {'type': 'object'}}])

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done'][0]
    assert done['stop_reason'] == 'tool_use'
    assert done['tool_calls'] == [
        {'id': 'call_1', 'name': 'getToc', 'input': {'depth': 2}}]
    assert done['assistant_blocks'] == [{
        'type': 'function_call', 'call_id': 'call_1', 'name': 'getToc',
        'arguments': '{"depth": 2}'}]


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
        _make_event('response.completed', response=_make_completed_response()),
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
        _make_event('response.completed', response=_make_completed_response()),
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
        _make_event('response.completed', response=_make_completed_response()),
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
        {'role': 'assistant', 'content': 'Hello'},
    ]
    params = _build_anthropic_params(messages)
    assert params['system'] == 'Be helpful'
    assert len(params['messages']) == 2
    assert params['messages'][0] == {'role': 'user', 'content': 'Hi'}
    # Plain assistant content passes through verbatim (no action annotations).
    assert params['messages'][1] == {'role': 'assistant', 'content': 'Hello'}


def test_build_anthropic_params_multiple_system_messages_folded():
    messages = [
        {'role': 'system', 'content': 'Be helpful'},
        {'role': 'system', 'content': 'Be concise'},
        {'role': 'user', 'content': 'Hi'},
    ]
    params = _build_anthropic_params(messages)
    assert params['system'] == 'Be helpful\n\nBe concise'


def test_build_anthropic_params_no_system():
    messages = [{'role': 'user', 'content': 'Hi'}]
    params = _build_anthropic_params(messages)
    assert 'system' not in params
    assert params['max_tokens'] == 32000
    # No model given -> conservative budget_tokens form.
    assert params['thinking'] == {'type': 'enabled', 'budget_tokens': 2000}


@pytest.mark.parametrize('model', [
    'claude-opus-4-6', 'claude-opus-4-7', 'claude-opus-4-8',
    'claude-sonnet-4-6', 'us.anthropic.claude-opus-4-8-v1:0',
])
def test_anthropic_thinking_adaptive_for_4_6_plus(model):
    assert _anthropic_thinking_config(model) == {'type': 'adaptive'}


@pytest.mark.parametrize('model', [
    'claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5',
    'claude-3-7-sonnet', 'anthropic.claude-3-5-sonnet-20240620-v1:0', '',
])
def test_anthropic_thinking_budget_tokens_for_older(model):
    assert _anthropic_thinking_config(model) == {
        'type': 'enabled', 'budget_tokens': 2000}


def test_build_anthropic_params_uses_adaptive_for_new_model():
    params = _build_anthropic_params([{'role': 'user', 'content': 'Hi'}],
                                     model='claude-opus-4-8')
    assert params['thinking'] == {'type': 'adaptive'}


def test_build_anthropic_params_tools_blocks_and_results():
    blocks = [
        {'type': 'text', 'text': 'Let me check'},
        {'type': 'tool_use', 'id': 'tu_1', 'name': 'getToc', 'input': {}},
    ]
    messages = [
        {'role': 'assistant', 'assistantBlocks': blocks},
        {'role': 'user', 'toolResults': [
            {'id': 'tu_1', 'result': '{"toc": []}'},
            {'id': 'tu_2', 'result': 'boom', 'isError': True},
        ]},
    ]
    params = _build_anthropic_params(messages, tools=[
        {'name': 'getToc', 'description': 'd', 'parameters': {'type': 'object'}}])

    assert params['tools'] == [
        {'name': 'getToc', 'description': 'd',
         'input_schema': {'type': 'object'}}]
    assert params['messages'][0] == {'role': 'assistant', 'content': blocks}
    assert params['messages'][1] == {'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'tu_1', 'content': '{"toc": []}'},
        {'type': 'tool_result', 'tool_use_id': 'tu_2', 'content': 'boom',
         'is_error': True},
    ]}


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


def _make_anthropic_block(block_type, *, dump=None, **kwargs):
    """A final-message content block (exposes .type, .model_dump, and attrs).

    For tool_use blocks pass id/name/input; for text pass text.
    """
    block = MagicMock()
    block.type = block_type
    for k, v in kwargs.items():
        setattr(block, k, v)
    if dump is None:
        dump = {'type': block_type, **kwargs}
    block.model_dump.return_value = dump
    return block


@pytest.mark.asyncio
async def test_chat_anthropic_basic_flow():
    handler = MagicMock()
    # Streaming now emits RAW text deltas (no JSON-envelope extraction).
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

    final_content = [_make_anthropic_block('text', text='Hello world')]
    mock_stream = _AsyncStreamCtx(events, final_text='Hello world',
                                  stop_reason='end_turn',
                                  final_content=final_content)
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-sonnet', [])

    payloads, written = _parse_sse_payloads(handler)

    types = [p['type'] for p in payloads]
    assert 'content_block_start' in types
    assert 'content_block_delta' in types
    assert 'content_block_stop' in types
    assert 'message_done' in types

    # Raw text deltas emitted verbatim (incremental).
    text_deltas = [p for p in payloads
                   if p['type'] == 'content_block_delta' and p['content_type'] == 'text']
    assert text_deltas[0]['delta'] == 'Hello'
    assert text_deltas[1]['delta'] == ' world'

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['text'] == 'Hello world'
    assert done[0]['stop_reason'] == 'end_turn'
    assert done[0]['tool_calls'] == []
    assert done[0]['assistant_blocks'] == [{'type': 'text', 'text': 'Hello world'}]

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_anthropic_tool_use():
    handler = MagicMock()
    events = [
        _make_anthropic_event('content_block_start',
                              content_block=_make_content_block('text')),
        _make_anthropic_event('content_block_delta',
                              delta=_make_delta('text_delta', text='Checking')),
        _make_anthropic_event('content_block_stop'),
    ]
    final_content = [
        _make_anthropic_block('text', text='Checking'),
        _make_anthropic_block(
            'tool_use', id='tu_1', name='getToc', input={'depth': 2},
            dump={'type': 'tool_use', 'id': 'tu_1', 'name': 'getToc',
                  'input': {'depth': 2}}),
    ]
    mock_stream = _AsyncStreamCtx(events, final_text='Checking',
                                  stop_reason='tool_use',
                                  final_content=final_content)
    with patch('jupyter_mynerva.routes.AsyncAnthropic') as MockAnthropic:
        MockAnthropic.return_value.messages.stream = MagicMock(return_value=mock_stream)
        await chat_anthropic(handler, 'key', 'claude-sonnet', [], tools=[
            {'name': 'getToc', 'description': 'd',
             'parameters': {'type': 'object'}}])

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done'][0]
    assert done['stop_reason'] == 'tool_use'
    assert done['tool_calls'] == [
        {'id': 'tu_1', 'name': 'getToc', 'input': {'depth': 2}}]
    assert done['assistant_blocks'] == [
        {'type': 'text', 'text': 'Checking'},
        {'type': 'tool_use', 'id': 'tu_1', 'name': 'getToc',
         'input': {'depth': 2}}]


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
    completed_response = _make_completed_response(status='completed')
    events = [
        _make_event('response.in_progress'),
        _make_event('response.content_part.added'),
        _make_event('response.output_text.delta', delta='ok'),
        _make_event('response.output_text.done', text='ok'),
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
    completed_response = _make_completed_response(
        status='incomplete', incomplete_details=incomplete)
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
        _make_event('response.completed', response=_make_completed_response()),
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

    # Lifecycle: text -> message_done
    starts = [p['content_type'] for p in payloads if p['type'] == 'content_block_start']
    assert starts == ['text']

    stops = [p['content_type'] for p in payloads if p['type'] == 'content_block_stop']
    assert 'text' in stops

    done = [p for p in payloads if p['type'] == 'message_done']
    assert len(done) == 1
    # Native tool calls on message_done; assistant_blocks is None for echo.
    assert done[0]['stop_reason'] == 'tool_use'
    assert done[0]['tool_calls'][0]['name'] == 'getToc'
    assert done[0]['assistant_blocks'] is None
    assert done[0]['text'] == 'Echo: requesting getToc'

    assert written[-1] == 'data: [DONE]\n\n'
    handler.finish.assert_called_once()


@pytest.mark.asyncio
async def test_chat_echo_action_results_passthrough():
    handler = MagicMock()
    # A turn carrying toolResults -> echo finishes with text and no more calls.
    messages = [{'role': 'user', 'toolResults': [
        {'id': 'echo_getToc', 'result': '{"toc": [...]}'}]}]

    await chat_echo(handler, messages)

    payloads, _ = _parse_sse_payloads(handler)

    done = [p for p in payloads if p['type'] == 'message_done']
    assert done[0]['tool_calls'] == []
    assert done[0]['stop_reason'] == 'end_turn'
    assert '{"toc": [...]}' in done[0]['text']


@pytest.mark.asyncio
async def test_chat_echo_default_action_when_no_trigger():
    handler = MagicMock()
    messages = [{'role': 'user', 'content': 'hello world'}]

    await chat_echo(handler, messages)

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done']
    # Default trigger is 'toc'
    assert done[0]['tool_calls'][0]['name'] == 'getToc'


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
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!'},
        ],
        model='us.anthropic.claude-haiku-4-5-20251001-v1:0',
    )
    assert body['system'] == [
        {'text': 'You are helpful.'},
        {'text': 'Be concise.'},
    ]
    # maxTokens is omitted so Converse uses each model's own maximum.
    assert 'inferenceConfig' not in body
    # User content is a plain text block (no action annotations).
    assert body['messages'][0] == {'role': 'user', 'content': [{'text': 'Hello'}]}
    assert body['messages'][1] == {'role': 'assistant', 'content': [{'text': 'Hi!'}]}
    # Claude 4.5 in model id -> conservative budget_tokens form.
    assert body['additionalModelRequestFields'] == {
        'thinking': {'type': 'enabled', 'budget_tokens': 2000}
    }


def test_build_bedrock_converse_body_adaptive_for_4_6_plus():
    # Claude 4.6+ requires adaptive thinking; enabled/budget_tokens 400s.
    body = _build_bedrock_converse_body(
        [{'role': 'user', 'content': 'Hi'}],
        model='jp.anthropic.claude-sonnet-4-6',
    )
    assert body['additionalModelRequestFields'] == {
        'thinking': {'type': 'adaptive'}
    }


def test_build_bedrock_converse_body_no_thinking_for_non_claude():
    body = _build_bedrock_converse_body(
        [{'role': 'user', 'content': 'Hi'}],
        model='meta.llama3-8b-instruct-v1:0',
    )
    assert 'additionalModelRequestFields' not in body


def test_build_bedrock_converse_body_tools_blocks_and_results():
    blocks = [
        {'text': 'Let me check'},
        {'toolUse': {'toolUseId': 'tu_1', 'name': 'getToc', 'input': {}}},
    ]
    body = _build_bedrock_converse_body(
        [
            {'role': 'assistant', 'assistantBlocks': blocks},
            {'role': 'user', 'toolResults': [
                {'id': 'tu_1', 'result': '{"toc": []}'},
                {'id': 'tu_2', 'result': 'boom', 'isError': True},
            ]},
        ],
        model='us.anthropic.claude-haiku-4-5-20251001-v1:0',
        tools=[{'name': 'getToc', 'description': 'd',
                'parameters': {'type': 'object'}}],
    )
    assert body['toolConfig'] == {'tools': [
        {'toolSpec': {'name': 'getToc', 'description': 'd',
                      'inputSchema': {'json': {'type': 'object'}}}}]}
    assert body['messages'][0] == {'role': 'assistant', 'content': blocks}
    assert body['messages'][1] == {'role': 'user', 'content': [
        {'toolResult': {'toolUseId': 'tu_1',
                        'content': [{'text': '{"toc": []}'}],
                        'status': 'success'}},
        {'toolResult': {'toolUseId': 'tu_2', 'content': [{'text': 'boom'}],
                        'status': 'error'}},
    ]}


@pytest.mark.asyncio
async def test_chat_bedrock_converse_basic_flow():
    handler = MagicMock()

    # Stream simulates Bedrock Converse: raw text deltas, block stop, msg stop.
    chunks = [
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0, 'delta': {'text': 'Hi'}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0, 'delta': {'text': ' there!'}}),
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
    # Raw text deltas emitted verbatim (incremental).
    assert text_deltas == ['Hi', ' there!']

    done = [p for p in payloads if p['type'] == 'message_done'][0]
    assert done['text'] == 'Hi there!'
    assert done['stop_reason'] == 'end_turn'
    assert done['tool_calls'] == []
    assert done['assistant_blocks'] == [{'text': 'Hi there!'}]

    assert written[-1] == 'data: [DONE]\n\n'


@pytest.mark.asyncio
async def test_chat_bedrock_converse_thinking_then_text():
    handler = MagicMock()
    chunks = [
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0,
                        'delta': {'reasoningContent': {'text': 'Let me think'}}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0,
                        'delta': {'reasoningContent': {'text': ' about this.'}}}),
        ),
        # Switch to text block; emitter must close thinking and open text.
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 1, 'delta': {'text': 'Done'}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockStop', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 1}),
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
async def test_chat_bedrock_converse_tool_use():
    handler = MagicMock()
    chunks = [
        _encode_es_frame(
            {':event-type': 'contentBlockStart', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0, 'start': {'toolUse': {
                'toolUseId': 'tu_1', 'name': 'getToc'}}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0,
                        'delta': {'toolUse': {'input': '{"dep'}}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockDelta', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0,
                        'delta': {'toolUse': {'input': 'th": 2}'}}}),
        ),
        _encode_es_frame(
            {':event-type': 'contentBlockStop', ':message-type': 'event'},
            json.dumps({'contentBlockIndex': 0}),
        ),
        _encode_es_frame(
            {':event-type': 'messageStop', ':message-type': 'event'},
            json.dumps({'stopReason': 'tool_use'}),
        ),
    ]
    response = _FakeStreamResponse(status_code=200, chunks=chunks)
    with patch('jupyter_mynerva.routes.httpx.AsyncClient',
               return_value=_FakeAsyncClient(response, {})):
        await chat_bedrock_converse(
            handler, 'k', 'us-east-1',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            [], tools=[{'name': 'getToc', 'description': 'd',
                        'parameters': {'type': 'object'}}],
        )

    payloads, _ = _parse_sse_payloads(handler)
    done = [p for p in payloads if p['type'] == 'message_done'][0]
    assert done['stop_reason'] == 'tool_use'
    assert done['tool_calls'] == [
        {'id': 'tu_1', 'name': 'getToc', 'input': {'depth': 2}}]
    assert done['assistant_blocks'] == [
        {'toolUse': {'toolUseId': 'tu_1', 'name': 'getToc',
                     'input': {'depth': 2}}}]


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


# --- nbsearch summary context-overflow map-reduce ---


@pytest.mark.parametrize('message', [
    'request (12700 tokens) exceeds the available context size (4096 tokens)',
    "This model's maximum context length is 4096 tokens",
    'Error code: 400 - context_length_exceeded',
    'prompt is larger than the context window',
])
def test_is_context_exceeded_matches_known_errors(message):
    assert _is_context_exceeded(RuntimeError(message))


@pytest.mark.parametrize('message', [
    'invalid api key',
    'rate limit reached',
    'connection refused',
])
def test_is_context_exceeded_ignores_other_errors(message):
    assert not _is_context_exceeded(RuntimeError(message))


def test_truncate_for_floor_clips_nested_strings():
    value = {'source': 'x' * 5000, 'cells': [{'text': 'y' * 5000}], 'n': 3}
    clipped = _truncate_for_floor(value, limit=10)
    assert clipped['source'] == 'x' * 10
    assert clipped['cells'][0]['text'] == 'y' * 10
    assert clipped['n'] == 3


class _FakeSummarizer:
    """Duck-typed stand-in exercising NBSearchHandler._summarize_adaptive.

    `_run_summary` fails with a context-overflow error whenever it is given
    more than `max_fit` cells, so the recursion is forced to split.
    """

    _summarize_adaptive = NBSearchHandler._summarize_adaptive
    _emit_progress = NBSearchHandler._emit_progress
    _on_progress = None

    def __init__(self, max_fit):
        self.max_fit = max_fit
        self.log = logging.getLogger('test-nbsearch')
        self.sizes = []

    async def _run_summary(self, provider, model, api_key, base_url, config,
                           messages):
        cells = json.loads(messages[1]['content'])['cells']
        self.sizes.append(len(cells))
        if len(cells) > self.max_fit:
            raise RuntimeError('exceeds the available context size')
        return f'summary-of-{len(cells)}'


def _build_summary_messages(segments):
    return [
        {'role': 'system', 'content': 'summarize'},
        {'role': 'user', 'content': json.dumps({'cells': segments})},
    ]


async def test_summarize_adaptive_single_shot_when_it_fits():
    fake = _FakeSummarizer(max_fit=10)
    result = await fake._summarize_adaptive(
        'openai', 'm', 'k', '', {}, _build_summary_messages, [1, 2, 3])
    assert result == 'summary-of-3'
    assert fake.sizes == [3]  # no split


async def test_summarize_adaptive_map_reduce_on_overflow():
    # Chunks larger than 2 cells overflow, forcing a split of the 4-cell input.
    fake = _FakeSummarizer(max_fit=2)
    result = await fake._summarize_adaptive(
        'openai', 'm', 'k', '', {}, _build_summary_messages, [1, 2, 3, 4])
    assert result.startswith('summary-of-')
    # First attempt (4) overflows; halves of 2 fit; reduce of 2 partials fits.
    assert fake.sizes[0] == 4
    assert 2 in fake.sizes


async def test_summarize_adaptive_propagates_non_context_errors():
    class _Boom(_FakeSummarizer):
        async def _run_summary(self, *args, **kwargs):
            raise ValueError('bad api key')

    fake = _Boom(max_fit=2)
    with pytest.raises(ValueError):
        await fake._summarize_adaptive(
            'openai', 'm', 'k', '', {}, _build_summary_messages, [1, 2, 3, 4])
    # No split attempted on a non-context error.
    assert fake.sizes == []


async def test_summarize_adaptive_depth_is_bounded():
    # Nothing ever fits, so the floor (truncate + final attempt) must raise
    # rather than recurse forever.
    fake = _FakeSummarizer(max_fit=0)
    with pytest.raises(RuntimeError):
        await fake._summarize_adaptive(
            'openai', 'm', 'k', '', {}, _build_summary_messages, [1, 2, 3, 4])
    # Bounded: never explodes into an unbounded number of calls.
    assert len(fake.sizes) <= 2 ** (_NBSEARCH_SUMMARY_MAX_DEPTH + 2)


async def test_summarize_adaptive_propagates_cancellation():
    # A client-disconnect cancellation must abort, never be mistaken for a
    # context overflow and trigger map-reduce.
    class _Cancelled(_FakeSummarizer):
        async def _run_summary(self, *args, **kwargs):
            self.sizes.append(0)
            raise asyncio.CancelledError()

    fake = _Cancelled(max_fit=2)
    with pytest.raises(asyncio.CancelledError):
        await fake._summarize_adaptive(
            'openai', 'm', 'k', '', {}, _build_summary_messages, [1, 2, 3, 4])
    # Only the first attempt ran; no split.
    assert fake.sizes == [0]

