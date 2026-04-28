import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import cachetools

_log = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado

from .echo_agent import chat_echo


# Lazy import wrappers for heavy SDKs. The actual modules are loaded only on
# first call, keeping `import jupyter_mynerva` fast (avoids ~1-3s of openai /
# anthropic / pydantic / cryptography import time during JupyterHub spawn).
# unittest.mock.patch('jupyter_mynerva.routes.OpenAI', ...) replaces the
# wrapper with a Mock, so existing tests keep working.

def OpenAI(*args, **kwargs):
    from openai import OpenAI as _OpenAI
    return _OpenAI(*args, **kwargs)


def AsyncOpenAI(*args, **kwargs):
    from openai import AsyncOpenAI as _AsyncOpenAI
    return _AsyncOpenAI(*args, **kwargs)


def AsyncAnthropic(*args, **kwargs):
    from anthropic import AsyncAnthropic as _AsyncAnthropic
    return _AsyncAnthropic(*args, **kwargs)


def Fernet(*args, **kwargs):
    from cryptography.fernet import Fernet as _Fernet
    return _Fernet(*args, **kwargs)


PROVIDERS = [
    {
        'id': 'openai',
        'displayName': 'OpenAI',
        'models': [
            'gpt-5.2',
            'gpt-5-mini',
            'gpt-5-nano',
            'gpt-4.1',
            'gpt-4.1-mini',
            'gpt-4.1-nano'
        ]
    },
    {
        'id': 'anthropic',
        'displayName': 'Anthropic',
        'models': [
            'claude-sonnet-4-5-20250929',
            'claude-haiku-4-5-20251001',
            'claude-opus-4-5-20251101',
            'claude-sonnet-4-20250514',
            'claude-opus-4-1-20250805'
        ]
    },
    {
        'id': 'enki-gate',
        'displayName': 'Enki Gate',
        'models': []
    }
]

if os.environ.get('MYNERVA_ECHO_AGENT'):
    PROVIDERS.append({'id': 'echo', 'displayName': 'Echo (Testing)', 'models': []})

DEFAULT_PROVIDER = 'openai'
DEFAULT_MODEL = 'gpt-5.2'
ENCRYPTED_PREFIX = 'encrypted:'

# Default privacy filters (same as nbfilter)
DEFAULT_FILTERS = [
    {
        'pattern': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'label': '[IPv4_#]'
    },
    {
        'pattern': r'[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.(com|org|net|jp|io|dev|local|internal)',
        'label': '[DOMAIN_#]'
    }
]


def load_filters():
    """Load filters from ~/.nbfilterrc.toml or return defaults.

    Raises ValueError if config file exists but is invalid.
    """
    config_path = Path.home() / '.nbfilterrc.toml'
    if not config_path.exists():
        return DEFAULT_FILTERS

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    filters = config.get('filters', [])
    if not filters:
        return DEFAULT_FILTERS

    result = []
    for i, f in enumerate(filters):
        if 'pattern' not in f:
            raise ValueError(f"Filter {i}: missing 'pattern' field")
        if 'label' not in f:
            raise ValueError(f"Filter {i}: missing 'label' field")

        try:
            re.compile(f['pattern'])
        except re.error as e:
            raise ValueError(f"Filter {i}: invalid regex '{f['pattern']}': {e}")

        result.append({
            'pattern': f['pattern'],
            'label': f['label']
        })

    return result

# Load default config from environment variables (and delete secrets)
_DEFAULT_CONFIG = {}

if 'MYNERVA_OPENAI_API_KEY' in os.environ:
    _DEFAULT_CONFIG['openai_api_key'] = os.environ['MYNERVA_OPENAI_API_KEY']
    del os.environ['MYNERVA_OPENAI_API_KEY']

if 'MYNERVA_ANTHROPIC_API_KEY' in os.environ:
    _DEFAULT_CONFIG['anthropic_api_key'] = os.environ['MYNERVA_ANTHROPIC_API_KEY']
    del os.environ['MYNERVA_ANTHROPIC_API_KEY']

if 'MYNERVA_DEFAULT_PROVIDER' in os.environ:
    _DEFAULT_CONFIG['provider'] = os.environ['MYNERVA_DEFAULT_PROVIDER']

if 'MYNERVA_DEFAULT_MODEL' in os.environ:
    _DEFAULT_CONFIG['model'] = os.environ['MYNERVA_DEFAULT_MODEL']

if 'MYNERVA_OPENAI_BASE_URL' in os.environ:
    _DEFAULT_CONFIG['openai_base_url'] = os.environ['MYNERVA_OPENAI_BASE_URL']

if os.environ.get('MYNERVA_DEFAULTS_ONLY'):
    _DEFAULT_CONFIG['defaults_only'] = True


def _get_provider_models(provider_id):
    """Returns model list for the given provider."""
    for p in PROVIDERS:
        if p['id'] == provider_id:
            return p['models']
    return []


_openai_models_cache = cachetools.TTLCache(maxsize=8, ttl=300)


def _fetch_openai_models(api_key, base_url):
    """Fetch model list from an OpenAI-compatible v1/models endpoint."""
    cache_key = base_url
    if cache_key in _openai_models_cache:
        return _openai_models_cache[cache_key]
    client = OpenAI(api_key=api_key or '', base_url=base_url)
    response = client.models.list()
    models = sorted([m.id for m in response.data])
    if not models:
        raise ValueError(f'No models available from {base_url}')
    _openai_models_cache[cache_key] = models
    return models


def get_default_config():
    """Returns default config if available.

    - If only one API key (or base_url) is set, auto-select that provider
    - If both API keys are set, MYNERVA_DEFAULT_PROVIDER is required
    - If model is not specified, use first model from provider's list
      (fetched from v1/models when openai_base_url is set)
    """
    has_openai = bool(_DEFAULT_CONFIG.get('openai_api_key') or
                      _DEFAULT_CONFIG.get('openai_base_url'))
    has_anthropic = bool(_DEFAULT_CONFIG.get('anthropic_api_key'))

    if not has_openai and not has_anthropic:
        return None

    # Determine provider
    explicit_provider = _DEFAULT_CONFIG.get('provider')
    if has_openai and has_anthropic:
        # Both keys present - require explicit provider
        if not explicit_provider:
            return None
        provider = explicit_provider
    elif has_openai:
        provider = 'openai'
    else:
        provider = 'anthropic'

    # Determine model
    model = _DEFAULT_CONFIG.get('model')
    if not model:
        if provider == 'openai' and _DEFAULT_CONFIG.get('openai_base_url'):
            base_url = _DEFAULT_CONFIG['openai_base_url']
            models = _fetch_openai_models(
                _DEFAULT_CONFIG.get('openai_api_key'), base_url)
            model = models[0]
            _log.info('Fetched %d models from %s, using %s',
                      len(models), base_url, model)
        else:
            models = _get_provider_models(provider)
            model = models[0] if models else ''

    result = {
        'provider': provider,
        'model': model,
    }
    if _DEFAULT_CONFIG.get('openai_base_url'):
        result['openaiBaseUrl'] = _DEFAULT_CONFIG['openai_base_url']
    return result


def get_default_api_key(provider):
    """Returns default API key for the given provider."""
    if provider == 'openai':
        return _DEFAULT_CONFIG.get('openai_api_key')
    elif provider == 'anthropic':
        return _DEFAULT_CONFIG.get('anthropic_api_key')
    return None


def resolve_chat_config(config):
    """Resolve provider, model, api_key, base_url from config.

    All fields come from the same source (defaults or user config)
    to prevent credential leakage across trust boundaries.
    """
    if _DEFAULT_CONFIG.get('defaults_only') or config.get('useDefault'):
        defaults = get_default_config()
        if not defaults:
            raise ValueError('Default configuration not available')
        provider = defaults['provider']
        model = defaults['model']
        api_key = get_default_api_key(provider)
        base_url = _DEFAULT_CONFIG.get('openai_base_url')
    else:
        provider = config['provider']
        model = config['model']
        api_key = config.get('apiKey')
        base_url = config.get('openaiBaseUrl')
    return provider, model, api_key, base_url


def get_fernet():
    secret_key = os.environ.get('MYNERVA_SECRET_KEY')
    if secret_key:
        return Fernet(secret_key.encode())
    return None


def encrypt_api_key(api_key):
    if not api_key:
        return ''
    fernet = get_fernet()
    if fernet:
        encrypted = fernet.encrypt(api_key.encode()).decode()
        return ENCRYPTED_PREFIX + encrypted
    return api_key


def decrypt_api_key(stored_value):
    if not stored_value:
        return ''
    if stored_value.startswith(ENCRYPTED_PREFIX):
        fernet = get_fernet()
        if not fernet:
            raise ValueError('MYNERVA_SECRET_KEY is required to decrypt stored API key')
        encrypted = stored_value[len(ENCRYPTED_PREFIX):]
        return fernet.decrypt(encrypted.encode()).decode()
    return stored_value


def get_config_path():
    return Path.home() / '.mynerva' / 'config.json'


def load_config():
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Validate required fields (only when not using defaults)
        if not config.get('useDefault'):
            missing = [f for f in ('provider', 'model') if f not in config]
            if missing:
                _log.warning('Config missing required fields: %s', ', '.join(missing))
                config['provider'] = DEFAULT_PROVIDER
                config['model'] = DEFAULT_MODEL
                config['configWarning'] = f'Config missing required fields: {", ".join(missing)}'
        try:
            config['apiKey'] = decrypt_api_key(config.get('apiKey', ''))
        except ValueError:
            config['apiKey'] = ''
            config['decryptError'] = 'MYNERVA_SECRET_KEY is required to decrypt stored API key'
        return config

    # Config doesn't exist - check if defaults are available
    defaults = get_default_config()
    if defaults:
        # Auto-generate config with useDefault=true
        config = {
            'provider': defaults['provider'],
            'model': defaults['model'],
            'apiKey': '',
            'useDefault': True
        }
        save_config(config)
        return config

    return {'provider': DEFAULT_PROVIDER, 'model': DEFAULT_MODEL, 'apiKey': ''}


def save_config(config):
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_to_save = config.copy()
    config_to_save['apiKey'] = encrypt_api_key(config.get('apiKey', ''))
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f)


def is_encryption_configured():
    return bool(os.environ.get('MYNERVA_SECRET_KEY'))


class ProvidersHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        try:
            filters = load_filters()
        except (ValueError, tomllib.TOMLDecodeError) as e:
            self.set_status(500)
            self.finish(json.dumps({'error': f'Filter config error: {e}'}))
            return

        result = {
            'providers': PROVIDERS,
            'encryption': is_encryption_configured(),
            'defaults': get_default_config(),
            'filters': filters
        }
        if _DEFAULT_CONFIG.get('defaults_only'):
            result['defaultsOnly'] = True
        self.finish(json.dumps(result))


class ConfigHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        config = load_config()
        self.finish(json.dumps(config))

    @tornado.web.authenticated
    def post(self):
        config = self.get_json_body()
        save_config(config)
        self.finish(json.dumps({'status': 'ok'}))


def _init_sse(handler):
    """Set SSE response headers."""
    handler.set_header('Content-Type', 'text/event-stream')
    handler.set_header('Cache-Control', 'no-cache')
    handler.set_header('Connection', 'keep-alive')


def _send_sse(handler, data):
    """Send a single SSE event."""
    handler.write(f'data: {json.dumps(data)}\n\n')
    handler.flush()


def _finish_sse(handler):
    """Send SSE termination and finish response."""
    handler.write('data: [DONE]\n\n')
    handler.finish()


def _block_start(handler, content_type, **metadata):
    event = {'type': 'content_block_start', 'content_type': content_type}
    event.update(metadata)
    _send_sse(handler, event)


def _block_delta(handler, content_type, delta):
    _send_sse(handler, {'type': 'content_block_delta', 'content_type': content_type, 'delta': delta})


def _block_stop(handler, content_type, **metadata):
    event = {'type': 'content_block_stop', 'content_type': content_type}
    event.update(metadata)
    _send_sse(handler, event)


def sse_serializer(func):
    """Decorator: wraps a serializer with init_sse / error handling / finish_sse.

    The decorated function must be an async coroutine taking (handler, ...).
    Any exception is caught and emitted as an SSE error event.
    _finish_sse runs in finally.
    """
    @functools.wraps(func)
    async def wrapper(handler, *args, **kwargs):
        _init_sse(handler)
        try:
            await func(handler, *args, **kwargs)
        except Exception as e:
            _send_sse(handler, {'type': 'error', 'error': str(e)})
        finally:
            _finish_sse(handler)

    return wrapper


def _convert_messages_for_responses_api(messages):
    """Convert Chat Completions messages to Responses API input format.

    The Responses API uses 'developer' role instead of 'system'.
    """
    result = []
    for m in messages:
        role = m.get('role', 'user')
        if role == 'system':
            role = 'developer'
        result.append({'role': role, 'content': m.get('content', '')})
    return result


def _extract_json_content(raw):
    """Extract the 'content' field value from a partial JSON string.

    The LLM responds with JSON like {"messages":[{"role":"assistant","content":"TEXT"}],...}.
    This extracts TEXT for display during streaming, handling escape sequences.
    Returns empty string if content field is not yet found.
    """
    match = re.search(r'"content"\s*:\s*"', raw)
    if not match:
        return ''
    start = match.end()
    result = []
    i = start
    while i < len(raw):
        c = raw[i]
        if c == '\\' and i + 1 < len(raw):
            nxt = raw[i + 1]
            result.append({'n': '\n', 't': '\t', '"': '"', '\\': '\\'}.get(nxt, nxt))
            i += 2
        elif c == '"':
            break
        else:
            result.append(c)
            i += 1
    return ''.join(result)


@sse_serializer
async def chat_openai(handler, api_key, model, messages, base_url=None):
    """Serializer for OpenAI Responses API (used also for Enki Gate)."""
    kwargs = {'api_key': api_key or ''}
    if base_url:
        kwargs['base_url'] = base_url
    client = AsyncOpenAI(**kwargs)

    api_input = _convert_messages_for_responses_api(messages)
    text_accumulated = ''

    stream = await client.responses.create(
        model=model, input=api_input, stream=True
    )
    async for event in stream:
        if event.type == 'response.in_progress':
            _block_start(handler, 'thinking')

        elif event.type == 'response.content_part.added':
            _block_stop(handler, 'thinking')
            _block_start(handler, 'text')

        elif event.type == 'response.reasoning_summary_text.delta':
            _block_delta(handler, 'thinking', event.delta)

        elif event.type == 'response.reasoning_summary_text.done':
            _block_stop(handler, 'thinking', text=event.text)

        elif event.type == 'response.output_text.delta':
            text_accumulated += event.delta
            display = _extract_json_content(text_accumulated)
            if display:
                _block_delta(handler, 'text', display)

        elif event.type == 'response.output_text.done':
            _block_stop(handler, 'text')

        elif event.type == 'response.completed':
            resp = event.response
            stop_reason = getattr(resp, 'status', 'completed')
            incomplete = getattr(resp, 'incomplete_details', None)
            if incomplete:
                stop_reason = str(getattr(incomplete, 'reason', stop_reason))
            _send_sse(handler, {'type': 'message_done',
                                'text': text_accumulated,
                                'stop_reason': stop_reason})

        elif event.type == 'response.failed':
            error_msg = str(getattr(event, 'error', 'Unknown error'))
            _send_sse(handler, {'type': 'error', 'error': error_msg})


def _build_anthropic_params(messages):
    """Build Anthropic API parameters from message list.

    Extracts system messages into a separate system param and appends
    action data to message content.
    """
    api_messages = []
    system_text = None
    for m in messages:
        role = m.get('role')
        content = m.get('content', '')
        actions = m.get('actions')
        if actions:
            content += '\n\n[Actions proposed]\n' + json.dumps(actions)

        if role == 'system':
            if system_text is None:
                system_text = content
            else:
                system_text += '\n\n' + content
        else:
            api_messages.append({'role': role, 'content': content})

    kwargs = {
        'max_tokens': 32000,
        'messages': api_messages,
        'thinking': {'type': 'enabled', 'budget_tokens': 2000}
    }
    if system_text:
        kwargs['system'] = system_text
    return kwargs


@sse_serializer
async def chat_anthropic(handler, api_key, model, messages):
    """Serializer for Anthropic messages.stream API."""
    client = AsyncAnthropic(api_key=api_key)
    kwargs = _build_anthropic_params(messages)

    text_accumulated = ''
    async with client.messages.stream(model=model, **kwargs) as stream:
        # Anthropic の content_block.type は Mynerva の content_type と一致する
        # サポート対象: 'thinking', 'text'
        current_block_type = ''
        async for event in stream:
            if event.type == 'content_block_start':
                block_type = event.content_block.type
                if block_type in ('thinking', 'text'):
                    current_block_type = block_type
                    _block_start(handler, block_type)

            elif event.type == 'content_block_delta':
                delta = event.delta
                if delta.type == 'thinking_delta':
                    _block_delta(handler, 'thinking', delta.thinking)
                elif delta.type == 'text_delta':
                    text_accumulated += delta.text
                    display = _extract_json_content(text_accumulated)
                    if display:
                        _block_delta(handler, 'text', display)

            elif event.type == 'content_block_stop':
                if current_block_type:
                    _block_stop(handler, current_block_type)
                    current_block_type = ''

        final_msg = await stream.get_final_message()
        final_text = await stream.get_final_text()
        stop_reason = getattr(final_msg, 'stop_reason', 'end_turn')
        _send_sse(handler, {'type': 'message_done',
                            'text': final_text,
                            'stop_reason': stop_reason or 'end_turn'})


class ChatHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        data = self.get_json_body()
        messages = data.get('messages', [])

        config = load_config()
        provider, model, api_key, base_url = resolve_chat_config(config)
        self.log.info('Chat request: provider=%s, model=%s, base_url=%s',
                      provider, model, base_url)

        if provider == 'echo':
            await chat_echo(self, messages)
            return

        if provider == 'enki-gate':
            enki_token = config.get('enkiGateToken')
            enki_url = config.get('enkiGateUrl')
            enki_model = config.get('enkiGateModel', '')
            if not enki_token or not enki_url:
                self.set_status(500)
                self.finish(json.dumps({'error': 'Enki Gate not configured. Run device flow first.'}))
                return
            enki_base = enki_url.rstrip('/') + '/v1'
            await chat_openai(self, enki_token, enki_model, messages,
                              base_url=enki_base)
            return

        if provider == 'openai':
            if not api_key and not base_url:
                self.set_status(500)
                self.finish(json.dumps({'error': 'API key not configured'}))
                return
            await chat_openai(self, api_key, model, messages, base_url=base_url)
            return

        if provider == 'anthropic':
            if not api_key:
                self.set_status(500)
                self.finish(json.dumps({'error': 'API key not configured'}))
                return
            await chat_anthropic(self, api_key, model, messages)
            return

        self.set_status(400)
        self.finish(json.dumps({'error': f'Unknown provider: {provider}'}))


# Session management
def get_sessions_dir():
    return Path.home() / '.mynerva' / 'sessions'


def generate_session_id():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_id = uuid.uuid4().hex[:8]
    return f'{timestamp}_{short_id}'


def list_sessions():
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return {'sessions': [], 'errors': []}

    sessions = []
    errors = []
    for path in sessions_dir.glob('*.mnchat'):
        try:
            with open(path) as f:
                data = json.load(f)
            sessions.append({
                'id': path.stem,
                'created': data.get('created'),
                'updated': data.get('updated'),
                'messageCount': len(data.get('messages', []))
            })
        except (json.JSONDecodeError, IOError) as e:
            errors.append({'file': path.name, 'error': str(e)})

    sessions.sort(key=lambda s: s.get('updated') or s.get('created') or '', reverse=True)
    return {'sessions': sessions, 'errors': errors}


def get_session(session_id):
    sessions_dir = get_sessions_dir()
    path = sessions_dir / f'{session_id}.mnchat'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_session(session_id, data):
    sessions_dir = get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f'{session_id}.mnchat'

    # Preserve existing created timestamp
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        data['created'] = existing.get('created', datetime.now().isoformat())
    elif 'created' not in data:
        data['created'] = datetime.now().isoformat()

    data['updated'] = datetime.now().isoformat()

    with open(path, 'w') as f:
        json.dump(data, f)


def delete_session(session_id):
    sessions_dir = get_sessions_dir()
    path = sessions_dir / f'{session_id}.mnchat'
    if path.exists():
        path.unlink()
        return True
    return False


class SessionsHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        result = list_sessions()
        self.finish(json.dumps(result))

    @tornado.web.authenticated
    def post(self):
        session_id = generate_session_id()
        data = {
            'messages': [],
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat()
        }
        save_session(session_id, data)
        self.finish(json.dumps({'id': session_id}))


class SessionHandler(APIHandler):
    @tornado.web.authenticated
    def get(self, session_id):
        data = get_session(session_id)
        if data is None:
            self.set_status(404)
            self.finish(json.dumps({'error': 'Session not found'}))
            return
        self.finish(json.dumps({'id': session_id, **data}))

    @tornado.web.authenticated
    def put(self, session_id):
        data = self.get_json_body()
        save_session(session_id, data)
        self.finish(json.dumps({'status': 'ok'}))

    @tornado.web.authenticated
    def delete(self, session_id):
        if delete_session(session_id):
            self.finish(json.dumps({'status': 'ok'}))
        else:
            self.set_status(404)
            self.finish(json.dumps({'error': 'Session not found'}))


# Per-notebook store for live document content.
# LRU eviction cleans up temp files when capacity is exceeded.
class _NotebookStore(cachetools.LRUCache):
    def __init__(self, maxsize=16):
        super().__init__(maxsize=maxsize)
        self._log = logging.getLogger(__name__)

    def __delitem__(self, key):
        path = self[key]
        super().__delitem__(key)
        try:
            os.unlink(path)
        except OSError as e:
            self._log.warning("Failed to remove store file %s: %s", path, e)

_notebook_stores = _NotebookStore()


def _get_store_path(notebook_path):
    if notebook_path not in _notebook_stores:
        fd, path = tempfile.mkstemp(suffix='.ipynb', prefix='nblibram-')
        os.close(fd)
        _notebook_stores[notebook_path] = path
    return _notebook_stores[notebook_path]


_NBLIBRAM_COMMANDS = frozenset(['toc', 'section', 'cells', 'outputs'])
_NBLIBRAM_FORMATS = frozenset(['md', 'json', 'py', 'text', 'raw'])


class NblibramHandler(APIHandler):

    def _validate_path(self, path):
        """Resolve path against content root. Rejects traversal and hidden files."""
        root_dir = os.path.realpath(self.contents_manager.root_dir)
        resolved = os.path.realpath(os.path.join(root_dir, path))
        if not resolved.startswith(root_dir + os.sep):
            raise ValueError('path escapes content root')
        rel = os.path.relpath(resolved, root_dir)
        for part in rel.split(os.sep):
            if part.startswith('.'):
                raise ValueError('hidden files are not accessible')
        return resolved

    @tornado.web.authenticated
    def post(self):
        nblibram_path = shutil.which('nblibram')
        if not nblibram_path:
            self.set_status(500)
            self.finish(json.dumps({'error': 'nblibram not found in PATH'}))
            return

        data = self.get_json_body()
        command = data.get('command', '')
        if command not in _NBLIBRAM_COMMANDS:
            self.set_status(400)
            self.finish(json.dumps({'error': f'unknown command: {command}'}))
            return

        # Resolve file path
        path = data.get('path', '')
        live = data.get('live', False)
        notebook_content = data.get('notebookContent')

        if live:
            # Live notebook query: use temp store
            store_key = os.path.normpath(path)
            if notebook_content is not None:
                store_path = _get_store_path(store_key)
                with open(store_path, 'w') as f:
                    json.dump(notebook_content, f)
            if store_key not in _notebook_stores:
                self.set_status(400)
                self.finish(json.dumps({'error': 'No notebook content in store. Send notebookContent first.'}))
                return
            file_arg = _notebook_stores[store_key]
        elif path:
            # File-based query: read from disk
            try:
                file_arg = self._validate_path(path)
            except ValueError as e:
                self.set_status(400)
                self.finish(json.dumps({'error': str(e)}))
                return
        else:
            self.set_status(400)
            self.finish(json.dumps({'error': 'path is required'}))
            return

        # Build CLI args from structured params
        args = ['-file', file_arg]

        fmt = data.get('format')
        if fmt:
            if fmt not in _NBLIBRAM_FORMATS:
                self.set_status(400)
                self.finish(json.dumps({'error': f'unknown format: {fmt}'}))
                return
            args += ['-format', fmt]

        query = data.get('query')
        if query:
            if not isinstance(query, str):
                self.set_status(400)
                self.finish(json.dumps({'error': 'query must be a string'}))
                return
            args += ['-query', query]

        count = data.get('count')
        if count is not None:
            args += ['-count', str(int(count))]

        if data.get('noFilter'):
            args.append('-no-filter')

        if data.get('excludeOutputs'):
            args.append('-exclude-outputs')

        cmd = [nblibram_path, command] + args
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.set_status(400)
            self.finish(json.dumps({'error': result.stderr.strip()}))
            return

        try:
            parsed = json.loads(result.stdout)
            self.finish(json.dumps(parsed))
        except json.JSONDecodeError:
            self.finish(json.dumps({'output': result.stdout}))


class EnkiGateDeviceFlowHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        data = self.get_json_body()
        enki_url = data.get('enkiGateUrl', '').rstrip('/')
        if not enki_url:
            self.set_status(400)
            self.finish(json.dumps({'error': 'enkiGateUrl is required'}))
            return


        req = urllib.request.Request(f'{enki_url}/api/device-flows', method='POST')
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
            self.finish(json.dumps(body))
        except urllib.error.HTTPError as e:
            self.set_status(e.code)
            self.finish(json.dumps({'error': e.read().decode()}))


class EnkiGateDeviceFlowPollHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self, device_code):
        data = self.get_json_body()
        enki_url = data.get('enkiGateUrl', '').rstrip('/')
        if not enki_url:
            self.set_status(400)
            self.finish(json.dumps({'error': 'enkiGateUrl is required'}))
            return


        req = urllib.request.Request(
            f'{enki_url}/api/device-flows/{device_code}/poll',
            method='POST'
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
            self.finish(json.dumps(body))
        except urllib.error.HTTPError as e:
            self.set_status(e.code)
            self.finish(json.dumps({'error': e.read().decode()}))


class OpenAIModelsHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        data = self.get_json_body()
        base_url = data.get('baseUrl')
        api_key = data.get('apiKey')
        models = _fetch_openai_models(api_key, base_url)
        self.finish(json.dumps({'models': models}))


def setup_route_handlers(web_app):
    host_pattern = '.*$'
    base_url = web_app.settings['base_url']

    providers_pattern = url_path_join(base_url, 'jupyter-mynerva', 'providers')
    config_pattern = url_path_join(base_url, 'jupyter-mynerva', 'config')
    chat_pattern = url_path_join(base_url, 'jupyter-mynerva', 'chat')
    openai_models_pattern = url_path_join(base_url, 'jupyter-mynerva', 'openai-models')
    sessions_pattern = url_path_join(base_url, 'jupyter-mynerva', 'sessions')
    session_pattern = url_path_join(base_url, 'jupyter-mynerva', 'sessions', '([^/]+)')
    nblibram_pattern = url_path_join(base_url, 'jupyter-mynerva', 'nblibram')
    enki_device_flow_pattern = url_path_join(base_url, 'jupyter-mynerva', 'enki-gate', 'device-flows')
    enki_device_flow_poll_pattern = url_path_join(base_url, 'jupyter-mynerva', 'enki-gate', 'device-flows', '([^/]+)', 'poll')
    handlers = [
        (providers_pattern, ProvidersHandler),
        (config_pattern, ConfigHandler),
        (chat_pattern, ChatHandler),
        (openai_models_pattern, OpenAIModelsHandler),
        (sessions_pattern, SessionsHandler),
        (session_pattern, SessionHandler),
        (nblibram_pattern, NblibramHandler),
        (enki_device_flow_pattern, EnkiGateDeviceFlowHandler),
        (enki_device_flow_poll_pattern, EnkiGateDeviceFlowPollHandler)
    ]

    web_app.add_handlers(host_pattern, handlers)
