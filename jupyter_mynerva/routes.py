import asyncio
import fnmatch
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import cachetools
from cryptography.fernet import InvalidToken
import httpx

_log = logging.getLogger(__name__)

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado

from .echo_agent import chat_echo
from .eventstream import EventStreamParser


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


def Anthropic(*args, **kwargs):
    from anthropic import Anthropic as _Anthropic
    return _Anthropic(*args, **kwargs)


def Fernet(*args, **kwargs):
    from cryptography.fernet import Fernet as _Fernet
    return _Fernet(*args, **kwargs)


PROVIDERS = [
    {'id': 'openai', 'displayName': 'OpenAI'},
    {'id': 'anthropic', 'displayName': 'Anthropic'},
    {'id': 'bedrock', 'displayName': 'Amazon Bedrock (Converse)'},
    {'id': 'enki-gate', 'displayName': 'Enki Gate'}
]

if os.environ.get('MYNERVA_ECHO_AGENT'):
    PROVIDERS.append({'id': 'echo', 'displayName': 'Echo (Testing)'})

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

if 'MYNERVA_BEDROCK_API_KEY' in os.environ:
    _DEFAULT_CONFIG['bedrock_api_key'] = os.environ['MYNERVA_BEDROCK_API_KEY']
    del os.environ['MYNERVA_BEDROCK_API_KEY']

if 'MYNERVA_BEDROCK_REGION' in os.environ:
    _DEFAULT_CONFIG['bedrock_region'] = os.environ['MYNERVA_BEDROCK_REGION']

if os.environ.get('MYNERVA_DEFAULTS_ONLY'):
    _DEFAULT_CONFIG['defaults_only'] = True


def _load_model_spec():
    """Load allow/deny glob patterns from models.json."""
    spec_file = files('jupyter_mynerva').joinpath('models.json')
    with spec_file.open() as f:
        return json.load(f)


def _filter_models(model_ids, allow, deny):
    """Filter model IDs by glob allow/deny patterns. Returns sorted list."""
    result = []
    for model_id in model_ids:
        if not any(fnmatch.fnmatch(model_id, p) for p in allow):
            continue
        if any(fnmatch.fnmatch(model_id, p) for p in deny):
            continue
        result.append(model_id)
    return sorted(result)


_chat_models_cache = cachetools.TTLCache(maxsize=8, ttl=300)

_PROVIDER_KEY_FIELD = {
    'openai': 'openai_api_key',
    'anthropic': 'anthropic_api_key',
}


def _is_openai_default_base_url(base_url):
    return base_url.strip().rstrip('/') == 'https://api.openai.com/v1'


async def _fetch_chat_models(provider_id, api_key):
    """Fetch chat models, filter via models.json spec, sort by release date DESC.

    Uses the provider's own `created` / `created_at` timestamp so newest
    releases bubble to the top of the UI dropdown automatically.
    """
    cache_key = (provider_id, api_key)
    if cache_key in _chat_models_cache:
        return _chat_models_cache[cache_key]
    spec = _load_model_spec()[provider_id]
    if provider_id == 'openai':
        # m.created is a Unix int seconds since epoch
        response = await AsyncOpenAI(api_key=api_key).models.list()
        created = {m.id: m.created for m in response.data}
    elif provider_id == 'anthropic':
        # m.created_at is a datetime
        response = await AsyncAnthropic(api_key=api_key).models.list()
        created = {m.id: m.created_at.timestamp() for m in response.data}
    else:
        raise ValueError(f'Unknown chat provider: {provider_id}')
    filtered = _filter_models(list(created), spec['allow'], spec.get('deny', []))
    models = sorted(filtered, key=lambda m: created[m], reverse=True)
    _chat_models_cache[cache_key] = models
    return models


async def _get_provider_models(provider_id, api_key='', base_url='', region=''):
    """Returns model list for the given provider credentials.

    OpenAI-compatible custom endpoints return raw /v1/models results.
    Official OpenAI and Anthropic providers apply the models.json filter.
    Bedrock provider fetches from the inference-profiles management API.
    """
    if provider_id == 'openai':
        if base_url and not _is_openai_default_base_url(base_url):
            return await _fetch_openai_models(api_key, base_url)
        if not api_key:
            return []
        return await _fetch_chat_models(provider_id, api_key)
    if provider_id == 'anthropic':
        if not api_key:
            return []
        return await _fetch_chat_models(provider_id, api_key)
    if provider_id == 'bedrock':
        if not api_key:
            return []
        return _fetch_bedrock_models(api_key, region or 'us-east-1')
    return []


async def _get_default_provider_models(provider_id):
    key_field = _PROVIDER_KEY_FIELD.get(provider_id)
    api_key = _DEFAULT_CONFIG.get(key_field, '') if key_field else ''
    base_url = _DEFAULT_CONFIG.get('openai_base_url', '') if provider_id == 'openai' else ''
    region = _DEFAULT_CONFIG.get('bedrock_region', 'us-east-1') if provider_id == 'bedrock' else ''
    return await _get_provider_models(provider_id, api_key, base_url, region)


async def _get_config_provider_models(provider_id, config):
    if _DEFAULT_CONFIG.get('defaults_only') or config.get('useDefault'):
        return await _get_default_provider_models(provider_id)
    if provider_id != config.get('provider'):
        return []
    base_url = config.get('openaiBaseUrl', '') if provider_id == 'openai' else ''
    region = config.get('bedrockRegion', '') if provider_id == 'bedrock' else ''
    return await _get_provider_models(provider_id, config['apiKey'], base_url, region)


_openai_models_cache = cachetools.TTLCache(maxsize=8, ttl=300)
_bedrock_models_cache = cachetools.TTLCache(maxsize=8, ttl=300)


def _load_bedrock_regions():
    """Load Bedrock region definitions from regions.json."""
    spec_file = files('jupyter_mynerva').joinpath('regions.json')
    with spec_file.open() as f:
        return json.load(f)['bedrock']


def _validate_bedrock_region(region):
    valid_ids = {r['id'] for r in _load_bedrock_regions()}
    if region not in valid_ids:
        raise ValueError(f'Invalid AWS region: {region}')


def _fetch_bedrock_models(api_key, region):
    """Fetch active inference profile IDs from Bedrock management API.

    Hits GET https://bedrock.{region}.amazonaws.com/inference-profiles with
    bearer-token auth, keeps profiles with status=ACTIVE and
    type=SYSTEM_DEFINED, then applies the models.json bedrock allow/deny
    filter so the dropdown stays focused on chat-capable models.

    Sync because model listing is a fast one-shot — unlike chat which is
    long-lived and must not block the Tornado event loop.

    Bedrock does not expose per-account grant info on this endpoint, so all
    matching profiles in the region are returned; access failures surface
    later at Converse invocation time.
    """
    _validate_bedrock_region(region)
    cache_key = (region, api_key or '')
    if cache_key in _bedrock_models_cache:
        return _bedrock_models_cache[cache_key]
    url = f'https://bedrock.{region}.amazonaws.com/inference-profiles'
    headers = {'Authorization': f'Bearer {api_key}'}
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(
            f'Bedrock list-profiles error ({resp.status_code}): {resp.text}')
    data = resp.json()
    summaries = data.get('inferenceProfileSummaries', [])
    candidate_ids = [
        p['inferenceProfileId']
        for p in summaries
        if p.get('status') == 'ACTIVE'
        and p.get('type') == 'SYSTEM_DEFINED'
        and p.get('inferenceProfileId')
    ]
    spec = _load_model_spec().get('bedrock', {'allow': ['*'], 'deny': []})
    models = _filter_models(candidate_ids, spec['allow'], spec.get('deny', []))
    if not models:
        raise ValueError(f'No matching inference profiles available in {region}')
    _bedrock_models_cache[cache_key] = models
    return models


async def _fetch_openai_models(api_key, base_url):
    """Fetch model list from an OpenAI-compatible v1/models endpoint.

    Used by the /openai-models endpoint and Enki Gate (custom base_url).
    Returns the raw list without allow/deny filtering, since custom
    endpoints may expose private model catalogs.

    Cache key includes api_key so swapping keys against the same endpoint
    re-fetches (different keys may have different model access).
    """
    cache_key = (base_url, api_key or '')
    if cache_key in _openai_models_cache:
        return _openai_models_cache[cache_key]
    client = AsyncOpenAI(api_key=api_key or '', base_url=base_url)
    response = await client.models.list()
    models = sorted([m.id for m in response.data])
    if not models:
        raise ValueError(f'No models available from {base_url}')
    _openai_models_cache[cache_key] = models
    return models


async def get_default_config():
    """Returns default config if available.

    - If only one API key (or base_url) is set, auto-select that provider
    - If multiple keys are set, MYNERVA_DEFAULT_PROVIDER is required and must
      name one of the configured providers
    - If model is not specified, use first model from _get_provider_models()
    """
    has_openai = bool(_DEFAULT_CONFIG.get('openai_api_key') or
                      _DEFAULT_CONFIG.get('openai_base_url'))
    has_anthropic = bool(_DEFAULT_CONFIG.get('anthropic_api_key'))
    has_bedrock = bool(_DEFAULT_CONFIG.get('bedrock_api_key'))

    candidates = [name for name, present in (
        ('openai', has_openai),
        ('anthropic', has_anthropic),
        ('bedrock', has_bedrock),
    ) if present]
    if not candidates:
        return None

    # Determine provider
    explicit_provider = _DEFAULT_CONFIG.get('provider')
    if len(candidates) == 1:
        provider = candidates[0]
    else:
        if not explicit_provider or explicit_provider not in candidates:
            return None
        provider = explicit_provider

    # Determine model
    model = _DEFAULT_CONFIG.get('model')
    if not model:
        models = await _get_default_provider_models(provider)
        model = models[0] if models else ''

    result = {
        'provider': provider,
        'model': model,
    }
    if _DEFAULT_CONFIG.get('openai_base_url'):
        result['openaiBaseUrl'] = _DEFAULT_CONFIG['openai_base_url']
    if _DEFAULT_CONFIG.get('bedrock_region'):
        result['bedrockRegion'] = _DEFAULT_CONFIG['bedrock_region']
    return result


def get_default_api_key(provider):
    """Returns default API key for the given provider."""
    if provider == 'openai':
        return _DEFAULT_CONFIG.get('openai_api_key')
    elif provider == 'anthropic':
        return _DEFAULT_CONFIG.get('anthropic_api_key')
    elif provider == 'bedrock':
        return _DEFAULT_CONFIG.get('bedrock_api_key')
    return None


async def resolve_chat_config(config):
    """Resolve provider, model, api_key, base_url from config.

    All fields come from the same source (defaults or user config)
    to prevent credential leakage across trust boundaries.
    """
    if _DEFAULT_CONFIG.get('defaults_only') or config.get('useDefault'):
        defaults = await get_default_config()
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
        try:
            return fernet.decrypt(encrypted.encode()).decode()
        except InvalidToken as e:
            _log.warning('Failed to decrypt stored API key: %s', e)
            raise ValueError(
                'Failed to decrypt API key — MYNERVA_SECRET_KEY may have changed') from e
    return stored_value


def get_config_path():
    return Path.home() / '.mynerva' / 'config.json'


async def load_config():
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
        except ValueError as e:
            config['apiKey'] = ''
            config['decryptError'] = str(e)
        return config

    # Config doesn't exist - check if defaults are available
    defaults = await get_default_config()
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


async def _build_providers_with_models(config=None):
    """Assemble PROVIDERS list with dynamically fetched model IDs."""
    get_models = _get_config_provider_models if config else None

    async def with_models(provider):
        try:
            models = await get_models(provider['id'], config) if get_models else \
                await _get_default_provider_models(provider['id'])
            return {**provider, 'models': models}
        except Exception as e:
            _log.warning('Failed to fetch models for %s: %s', provider['id'], e)
            return {**provider, 'models': [], 'modelsError': str(e)}

    return await asyncio.gather(*(with_models(p) for p in PROVIDERS))


class ProvidersHandler(APIHandler):
    @tornado.web.authenticated
    async def get(self):
        try:
            filters = load_filters()
        except (ValueError, tomllib.TOMLDecodeError) as e:
            self.set_status(500)
            self.finish(json.dumps({'error': f'Filter config error: {e}'}))
            return

        try:
            config = await load_config()
            providers = await _build_providers_with_models(config)
        except Exception as e:
            _log.exception('Failed to fetch provider models')
            self.set_status(500)
            self.finish(json.dumps({'error': f'Failed to fetch models: {e}'}))
            return

        defaults = None
        defaults_error = None
        try:
            defaults = await get_default_config()
        except Exception as e:
            _log.warning('Default config unavailable: %s', e)
            defaults_error = str(e)

        result = {
            'providers': providers,
            'encryption': is_encryption_configured(),
            'defaults': defaults,
            'filters': filters,
            'bedrockRegions': _load_bedrock_regions(),
        }
        if defaults_error:
            result['defaultsError'] = defaults_error
        if _DEFAULT_CONFIG.get('defaults_only'):
            result['defaultsOnly'] = True
        self.finish(json.dumps(result))


class ConfigHandler(APIHandler):
    @tornado.web.authenticated
    async def get(self):
        config = await load_config()
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


def _build_openai_tools(tools):
    """Wrap provider-neutral tool defs into the Responses API function shape."""
    return [
        {
            'type': 'function',
            'name': t['name'],
            'description': t['description'],
            'parameters': t['parameters'],
        }
        for t in tools
    ]


def _build_openai_input(messages):
    """Build the Responses API `input` array from the message history.

    System -> developer message. Assistant turns carry their native output
    items verbatim (`assistantBlocks`) so reasoning/function_call items round
    trip unchanged. Tool results become `function_call_output` items.
    """
    result = []
    for m in messages:
        role = m.get('role', 'user')
        if role == 'system':
            result.append({'role': 'developer', 'content': m.get('content', '')})
        elif role == 'assistant':
            blocks = m.get('assistantBlocks')
            if blocks:
                result.extend(blocks)
            else:
                result.append({'role': 'assistant', 'content': m.get('content', '')})
        else:
            tool_results = m.get('toolResults')
            if tool_results:
                for r in tool_results:
                    result.append({
                        'type': 'function_call_output',
                        'call_id': r['id'],
                        'output': r['result'],
                    })
            else:
                result.append({'role': 'user', 'content': m.get('content', '')})
    return result


@sse_serializer
async def chat_openai(handler, api_key, model, messages, tools=None, base_url=None):
    """Serializer for OpenAI Responses API (used also for Enki Gate)."""
    kwargs = {'api_key': api_key or ''}
    if base_url:
        kwargs['base_url'] = base_url
    client = AsyncOpenAI(**kwargs)

    create_kwargs = {
        'model': model,
        'input': _build_openai_input(messages),
        'stream': True,
    }
    if tools:
        create_kwargs['tools'] = _build_openai_tools(tools)

    text_accumulated = ''
    stream = await client.responses.create(**create_kwargs)
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
            _block_delta(handler, 'text', event.delta)

        elif event.type == 'response.output_text.done':
            _block_stop(handler, 'text')

        elif event.type == 'response.completed':
            resp = event.response
            output = resp.output or []
            assistant_blocks = [
                o.model_dump(mode='json', exclude_none=True) for o in output
            ]
            tool_calls = [
                {'id': o.call_id, 'name': o.name,
                 'input': json.loads(o.arguments or '{}')}
                for o in output if o.type == 'function_call'
            ]
            if tool_calls:
                stop_reason = 'tool_use'
            else:
                stop_reason = getattr(resp, 'status', 'completed')
                incomplete = getattr(resp, 'incomplete_details', None)
                if incomplete:
                    stop_reason = str(getattr(incomplete, 'reason', stop_reason))
            _send_sse(handler, {'type': 'message_done',
                                'text': text_accumulated,
                                'stop_reason': stop_reason,
                                'tool_calls': tool_calls,
                                'assistant_blocks': assistant_blocks})

        elif event.type == 'response.failed':
            error_msg = str(getattr(event, 'error', 'Unknown error'))
            _send_sse(handler, {'type': 'error', 'error': error_msg})


def _anthropic_thinking_config(model):
    """Choose the thinking config by model version.

    Adaptive thinking is supported from Claude 4.6 onward, where the older
    `budget_tokens` form is removed (it 400s on Opus 4.7/4.8). Earlier models
    use `budget_tokens`.
    """
    m = re.search(r'(\d+)-(\d+)', model)
    if m and (int(m.group(1)), int(m.group(2))) >= (4, 6):
        return {'type': 'adaptive'}
    return {'type': 'enabled', 'budget_tokens': 2000}


def _build_anthropic_params(messages, tools=None, model=''):
    """Build Anthropic API parameters from message list.

    System messages fold into the `system` param. Assistant turns carry their
    native content blocks verbatim (`assistantBlocks`) so thinking signatures
    round trip. Tool results become `tool_result` blocks.
    """
    api_messages = []
    system_text = None
    for m in messages:
        role = m.get('role')
        if role == 'system':
            content = m.get('content', '')
            if system_text is None:
                system_text = content
            else:
                system_text += '\n\n' + content
        elif role == 'assistant':
            blocks = m.get('assistantBlocks')
            if blocks:
                api_messages.append({'role': 'assistant', 'content': blocks})
            else:
                api_messages.append({'role': 'assistant',
                                     'content': m.get('content', '')})
        else:
            tool_results = m.get('toolResults')
            if tool_results:
                content = [
                    {'type': 'tool_result', 'tool_use_id': r['id'],
                     'content': r['result'],
                     **({'is_error': True} if r.get('isError') else {})}
                    for r in tool_results
                ]
                api_messages.append({'role': 'user', 'content': content})
            else:
                api_messages.append({'role': 'user',
                                     'content': m.get('content', '')})

    kwargs = {
        'max_tokens': 32000,
        'messages': api_messages,
        'thinking': _anthropic_thinking_config(model)
    }
    if tools:
        kwargs['tools'] = [
            {'name': t['name'], 'description': t['description'],
             'input_schema': t['parameters']}
            for t in tools
        ]
    if system_text:
        kwargs['system'] = system_text
    return kwargs


@sse_serializer
async def chat_anthropic(handler, api_key, model, messages, tools=None):
    """Serializer for Anthropic messages.stream API."""
    client = AsyncAnthropic(api_key=api_key)
    kwargs = _build_anthropic_params(messages, tools, model)

    async with client.messages.stream(model=model, **kwargs) as stream:
        # content_block.type matches Mynerva's content_type for thinking/text
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
                    _block_delta(handler, 'text', delta.text)

            elif event.type == 'content_block_stop':
                if current_block_type:
                    _block_stop(handler, current_block_type)
                    current_block_type = ''

        final_msg = await stream.get_final_message()
        # Not get_final_text(): it raises when the turn has no text block
        # (tool_use only). Concatenate text blocks instead (empty if none).
        final_text = ''.join(
            b.text for b in final_msg.content if b.type == 'text'
        )
        assistant_blocks = [b.model_dump(mode='json', exclude_none=True)
                            for b in final_msg.content]
        tool_calls = [
            {'id': b.id, 'name': b.name, 'input': b.input}
            for b in final_msg.content if b.type == 'tool_use'
        ]
        stop_reason = getattr(final_msg, 'stop_reason', 'end_turn') or 'end_turn'
        _send_sse(handler, {'type': 'message_done',
                            'text': final_text,
                            'stop_reason': stop_reason,
                            'tool_calls': tool_calls,
                            'assistant_blocks': assistant_blocks})


def _build_bedrock_converse_body(messages, model, tools=None):
    """Build the JSON body for a Bedrock Converse request.

    System messages fold into the top-level `system` field. Assistant turns
    carry their native content blocks verbatim (`assistantBlocks`) so reasoning
    signatures round trip. Tool results become `toolResult` blocks.

    Extended thinking is enabled only for Claude/Anthropic models, since
    non-Anthropic Bedrock models reject `additionalModelRequestFields.thinking`.
    """
    api_messages = []
    system_blocks = []
    for m in messages:
        role = m.get('role')
        if role == 'system':
            system_blocks.append({'text': m.get('content', '')})
        elif role == 'assistant':
            blocks = m.get('assistantBlocks')
            if blocks:
                api_messages.append({'role': 'assistant', 'content': blocks})
            else:
                api_messages.append({'role': 'assistant',
                                     'content': [{'text': m.get('content', '')}]})
        else:
            tool_results = m.get('toolResults')
            if tool_results:
                content = [
                    {'toolResult': {
                        'toolUseId': r['id'],
                        'content': [{'text': r['result']}],
                        'status': 'error' if r.get('isError') else 'success',
                    }}
                    for r in tool_results
                ]
                api_messages.append({'role': 'user', 'content': content})
            else:
                api_messages.append({'role': 'user',
                                     'content': [{'text': m.get('content', '')}]})

    body = {
        'messages': api_messages,
        'inferenceConfig': {'maxTokens': 32000}
    }
    if tools:
        body['toolConfig'] = {
            'tools': [
                {'toolSpec': {
                    'name': t['name'],
                    'description': t['description'],
                    'inputSchema': {'json': t['parameters']},
                }}
                for t in tools
            ]
        }
    if system_blocks:
        body['system'] = system_blocks

    model_lower = model.lower()
    if 'claude' in model_lower or 'anthropic' in model_lower:
        body['additionalModelRequestFields'] = {
            'thinking': {'type': 'enabled', 'budget_tokens': 2000}
        }
    return body


@sse_serializer
async def chat_bedrock_converse(handler, api_key, region, model, messages,
                               tools=None):
    """Serializer for Bedrock Converse Stream API.

    Uses bearer-token auth against bedrock-runtime.{region}.amazonaws.com.
    Short-term and long-term Bedrock API keys work directly as Bearer
    tokens, so no AWS SigV4 or boto3 is needed.
    """
    _validate_bedrock_region(region)
    body = _build_bedrock_converse_body(messages, model, tools)
    url = (f'https://bedrock-runtime.{region}.amazonaws.com'
           f'/model/{urllib.parse.quote(model, safe="")}/converse-stream')
    req_headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.amazon.eventstream',
    }

    parser = EventStreamParser()
    # Accumulate content blocks by index so reasoning/tool_use round trip and
    # parallel tool calls stay distinct.
    blocks = {}            # contentBlockIndex -> accumulator dict
    display_block = None   # currently streaming display block ('text'|'thinking')
    stop_reason = 'end_turn'

    def set_display(kind):
        nonlocal display_block
        if display_block != kind:
            if display_block:
                _block_stop(handler, display_block)
            _block_start(handler, kind)
            display_block = kind

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream('POST', url, headers=req_headers,
                                 content=json.dumps(body)) as resp:
            if resp.status_code != 200:
                error_body = (await resp.aread()).decode('utf-8', errors='replace')
                raise ValueError(
                    f'Bedrock Converse error ({resp.status_code}): {error_body}')

            async for chunk in resp.aiter_bytes():
                for hdrs, payload in parser.feed(chunk):
                    if hdrs.get(':message-type') == 'exception':
                        try:
                            err = json.loads(payload)
                            msg = err.get('message', payload.decode('utf-8', errors='replace'))
                        except json.JSONDecodeError:
                            msg = payload.decode('utf-8', errors='replace')
                        exc_type = hdrs.get(':exception-type', 'Exception')
                        raise ValueError(f'Bedrock Converse {exc_type}: {msg}')

                    event_type = hdrs.get(':event-type')
                    if event_type == 'contentBlockStart':
                        data_obj = json.loads(payload)
                        idx = data_obj.get('contentBlockIndex')
                        start = data_obj.get('start', {})
                        if 'toolUse' in start:
                            tu = start['toolUse']
                            blocks[idx] = {'kind': 'toolUse',
                                           'toolUseId': tu['toolUseId'],
                                           'name': tu['name'], 'input': ''}
                    elif event_type == 'contentBlockDelta':
                        data_obj = json.loads(payload)
                        idx = data_obj.get('contentBlockIndex')
                        delta = data_obj.get('delta', {})
                        if 'text' in delta:
                            b = blocks.setdefault(idx, {'kind': 'text', 'text': ''})
                            b['text'] += delta['text']
                            set_display('text')
                            _block_delta(handler, 'text', delta['text'])
                        elif 'reasoningContent' in delta:
                            rc = delta['reasoningContent']
                            b = blocks.setdefault(idx, {'kind': 'reasoning',
                                                        'text': '',
                                                        'signature': None,
                                                        'redacted': None})
                            if 'text' in rc:
                                b['text'] += rc['text']
                                set_display('thinking')
                                _block_delta(handler, 'thinking', rc['text'])
                            elif 'signature' in rc:
                                b['signature'] = rc['signature']
                            elif 'redactedContent' in rc:
                                b['redacted'] = rc['redactedContent']
                        elif 'toolUse' in delta:
                            b = blocks.setdefault(idx, {'kind': 'toolUse',
                                                        'toolUseId': '',
                                                        'name': '', 'input': ''})
                            b['input'] += delta['toolUse'].get('input', '')
                    elif event_type == 'messageStop':
                        data_obj = json.loads(payload)
                        stop_reason = data_obj.get('stopReason', stop_reason)

    if display_block:
        _block_stop(handler, display_block)

    assistant_blocks = []
    tool_calls = []
    text_parts = []
    for idx in sorted(blocks):
        b = blocks[idx]
        if b['kind'] == 'text':
            assistant_blocks.append({'text': b['text']})
            text_parts.append(b['text'])
        elif b['kind'] == 'reasoning':
            if b['redacted'] is not None:
                assistant_blocks.append(
                    {'reasoningContent': {'redactedContent': b['redacted']}})
            else:
                rt = {'text': b['text']}
                if b['signature'] is not None:
                    rt['signature'] = b['signature']
                assistant_blocks.append({'reasoningContent': {'reasoningText': rt}})
        elif b['kind'] == 'toolUse':
            parsed = json.loads(b['input'] or '{}')
            assistant_blocks.append({'toolUse': {
                'toolUseId': b['toolUseId'], 'name': b['name'], 'input': parsed}})
            tool_calls.append({'id': b['toolUseId'], 'name': b['name'],
                               'input': parsed})

    _send_sse(handler, {'type': 'message_done',
                        'text': ''.join(text_parts),
                        'stop_reason': stop_reason,
                        'tool_calls': tool_calls,
                        'assistant_blocks': assistant_blocks})


class ChatHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        data = self.get_json_body()
        messages = data.get('messages', [])
        tools = data.get('tools', [])

        config = await load_config()
        provider, model, api_key, base_url = await resolve_chat_config(config)
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
            await chat_openai(self, enki_token, enki_model, messages, tools,
                              base_url=enki_base)
            return

        if provider == 'openai':
            if not api_key and not base_url:
                self.set_status(500)
                self.finish(json.dumps({'error': 'API key not configured'}))
                return
            await chat_openai(self, api_key, model, messages, tools,
                              base_url=base_url)
            return

        if provider == 'anthropic':
            if not api_key:
                self.set_status(500)
                self.finish(json.dumps({'error': 'API key not configured'}))
                return
            await chat_anthropic(self, api_key, model, messages, tools)
            return

        if provider == 'bedrock':
            if not api_key:
                self.set_status(500)
                self.finish(json.dumps({'error': 'API key not configured'}))
                return
            if config.get('useDefault') or _DEFAULT_CONFIG.get('defaults_only'):
                region = _DEFAULT_CONFIG.get('bedrock_region', 'us-east-1')
            else:
                region = config.get('bedrockRegion', 'us-east-1')
            await chat_bedrock_converse(self, api_key, region, model, messages,
                                        tools)
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
    async def post(self):
        data = self.get_json_body()
        base_url = data.get('baseUrl')
        api_key = data.get('apiKey')
        try:
            models = await _fetch_openai_models(api_key, base_url)
        except Exception as e:
            _log.warning('Failed to fetch models from %s: %s', base_url, e)
            self.set_status(500)
            self.finish(json.dumps({'error': str(e)}))
            return
        self.finish(json.dumps({'models': models}))


class BedrockModelsHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        data = self.get_json_body()
        api_key = data.get('apiKey')
        region = data.get('region') or 'us-east-1'
        if not api_key:
            self.set_status(400)
            self.finish(json.dumps({'error': 'apiKey is required'}))
            return
        try:
            models = _fetch_bedrock_models(api_key, region)
        except Exception as e:
            _log.warning('Failed to fetch Bedrock models in %s: %s', region, e)
            self.set_status(500)
            self.finish(json.dumps({'error': str(e)}))
            return
        self.finish(json.dumps({'models': models}))


class ProviderModelsHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self):
        data = self.get_json_body()
        provider = data['provider']
        api_key = data.get('apiKey', '')
        base_url = data.get('baseUrl', '') if provider == 'openai' else ''
        region = data.get('region', '') if provider == 'bedrock' else ''

        if provider not in ('openai', 'anthropic', 'bedrock'):
            self.set_status(400)
            self.finish(json.dumps({'error': f'Unsupported provider: {provider}'}))
            return
        openai_custom_base_url = (
            provider == 'openai' and base_url and
            not _is_openai_default_base_url(base_url)
        )
        if not api_key and not openai_custom_base_url:
            self.set_status(400)
            self.finish(json.dumps({'error': 'API key is required'}))
            return

        try:
            models = await _get_provider_models(provider, api_key, base_url, region)
        except Exception as e:
            _log.warning('Failed to fetch %s models: %s', provider, e)
            self.set_status(500)
            self.finish(json.dumps({'error': str(e)}))
            return
        self.finish(json.dumps({'models': models}))


def setup_route_handlers(web_app):
    host_pattern = '.*$'
    base_url = web_app.settings['base_url']

    providers_pattern = url_path_join(base_url, 'jupyter-mynerva', 'providers')
    config_pattern = url_path_join(base_url, 'jupyter-mynerva', 'config')
    chat_pattern = url_path_join(base_url, 'jupyter-mynerva', 'chat')
    openai_models_pattern = url_path_join(base_url, 'jupyter-mynerva', 'openai-models')
    bedrock_models_pattern = url_path_join(base_url, 'jupyter-mynerva', 'bedrock-models')
    provider_models_pattern = url_path_join(base_url, 'jupyter-mynerva', 'provider-models')
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
        (bedrock_models_pattern, BedrockModelsHandler),
        (provider_models_pattern, ProviderModelsHandler),
        (sessions_pattern, SessionsHandler),
        (session_pattern, SessionHandler),
        (nblibram_pattern, NblibramHandler),
        (enki_device_flow_pattern, EnkiGateDeviceFlowHandler),
        (enki_device_flow_poll_pattern, EnkiGateDeviceFlowPollHandler)
    ]

    web_app.add_handlers(host_pattern, handlers)
