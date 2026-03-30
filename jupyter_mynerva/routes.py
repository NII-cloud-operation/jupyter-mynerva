import ipaddress
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
from openai import OpenAI
from anthropic import Anthropic
from cryptography.fernet import Fernet

from .echo_agent import chat_echo


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


def _get_provider_models(provider_id):
    """Returns model list for the given provider."""
    for p in PROVIDERS:
        if p['id'] == provider_id:
            return p['models']
    return []


def get_default_config():
    """Returns default config if available.

    - If only one API key is set, auto-select that provider
    - If both API keys are set, MYNERVA_DEFAULT_PROVIDER is required
    - If model is not specified, use first model from provider's list
    """
    has_openai = bool(_DEFAULT_CONFIG.get('openai_api_key'))
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
        models = _get_provider_models(provider)
        model = models[0] if models else ''

    return {
        'provider': provider,
        'model': model,
    }


def get_default_api_key(provider):
    """Returns default API key for the given provider."""
    if provider == 'openai':
        return _DEFAULT_CONFIG.get('openai_api_key')
    elif provider == 'anthropic':
        return _DEFAULT_CONFIG.get('anthropic_api_key')
    return None


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

        self.finish(json.dumps({
            'providers': PROVIDERS,
            'encryption': is_encryption_configured(),
            'defaults': get_default_config(),
            'filters': filters,
            'agentMode': bool(os.environ.get('MYNERVA_AGENT_MODE'))
        }))


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


def chat_openai(api_key, model, messages, base_url=None):
    kwargs = {'api_key': api_key}
    if base_url:
        kwargs['base_url'] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(model=model, messages=messages)
    return {'provider': 'openai', 'response': response.model_dump()}


def chat_anthropic(api_key, model, messages):
    client = Anthropic(api_key=api_key)

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
        'model': model,
        'max_tokens': 4096,
        'messages': api_messages
    }
    if system_text:
        kwargs['system'] = system_text

    response = client.messages.create(**kwargs)
    return {'provider': 'anthropic', 'response': response.model_dump()}


class ChatHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        data = self.get_json_body()
        messages = data.get('messages', [])

        config = load_config()

        if config.get('useDefault'):
            defaults = get_default_config()
            if not defaults:
                self.set_status(500)
                self.finish(json.dumps({'error': 'Default configuration not available'}))
                return
            provider = defaults['provider']
            model = defaults['model']
            api_key = get_default_api_key(provider)
        else:
            provider = config.get('provider', DEFAULT_PROVIDER)
            model = config.get('model', DEFAULT_MODEL)
            api_key = config.get('apiKey')

        if provider == 'echo':
            result = chat_echo(messages)
            self.finish(json.dumps(result))
            return

        if provider == 'enki-gate':
            enki_token = config.get('enkiGateToken')
            enki_url = config.get('enkiGateUrl')
            enki_model = config.get('enkiGateModel', '')
            if not enki_token or not enki_url:
                self.set_status(500)
                self.finish(json.dumps({'error': 'Enki Gate not configured. Run device flow first.'}))
                return
            result = chat_openai(enki_token, enki_model, messages,
                                 base_url=enki_url.rstrip('/') + '/v1')
            self.finish(json.dumps(result))
            return

        if not api_key:
            self.set_status(500)
            self.finish(json.dumps({'error': 'API key not configured'}))
            return

        if provider == 'openai':
            result = chat_openai(api_key, model, messages)
        elif provider == 'anthropic':
            result = chat_anthropic(api_key, model, messages)
        else:
            self.set_status(400)
            self.finish(json.dumps({'error': f'Unknown provider: {provider}'}))
            return

        self.finish(json.dumps(result))


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


class AgentServerHandler(APIHandler):
    """Start/stop a Named Server for agent isolation via JupyterHub API."""

    def _hub_api(self, method, path, body=None):
        hub_url = os.environ.get('JUPYTERHUB_API_URL', '')
        token = os.environ.get('JUPYTERHUB_API_TOKEN', '')
        if not hub_url or not token:
            raise RuntimeError('JupyterHub API not available (not running inside JupyterHub)')
        url = f'{hub_url}{path}'
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header('Authorization', f'token {token}')
        if data:
            req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body)

    @tornado.web.authenticated
    async def post(self):
        data = self.get_json_body()
        ssh_entries = data.get('ssh', [])
        host_list = []
        for entry in ssh_entries:
            host = entry.get('host', '')
            try:
                host_list.append(str(ipaddress.ip_address(host)))
            except ValueError:
                self.set_status(400)
                self.finish(json.dumps({'error': f'Invalid IP address: {host}'}))
                return

        user = os.environ.get('JUPYTERHUB_USER', '')
        if not user:
            self.set_status(500)
            self.finish(json.dumps({'error': 'Not running inside JupyterHub'}))
            return

        server_name = 'mynerva-agent'
        try:
            # ssh_hosts is passed as user_options → spawner.user_options
            self._hub_api('POST', f'/users/{user}/servers/{server_name}',
                          body={'ssh_hosts': host_list})
        except urllib.error.HTTPError as e:
            if e.code != 400:  # 400 = already running
                self.set_status(e.code)
                self.finish(json.dumps({'error': f'Failed to start server: {e.read().decode()}'}))
                return

        base_url = os.environ.get('JUPYTERHUB_BASE_URL', '/').rstrip('/')
        agent_url = f'{base_url}/user/{user}/{server_name}/lab'

        self.finish(json.dumps({
            'status': 'started',
            'url': agent_url,
            'ssh_hosts': host_list
        }))

    @tornado.web.authenticated
    async def delete(self):
        user = os.environ.get('JUPYTERHUB_USER', '')
        if not user:
            self.set_status(500)
            self.finish(json.dumps({'error': 'Not running inside JupyterHub'}))
            return

        server_name = 'mynerva-agent'
        try:
            self._hub_api('DELETE', f'/users/{user}/servers/{server_name}')
        except urllib.error.HTTPError as e:
            self.set_status(e.code)
            self.finish(json.dumps({'error': f'Failed to stop server: {e.read().decode()}'}))
            return

        self.finish(json.dumps({'status': 'stopped'}))


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


def setup_route_handlers(web_app):
    host_pattern = '.*$'
    base_url = web_app.settings['base_url']

    providers_pattern = url_path_join(base_url, 'jupyter-mynerva', 'providers')
    config_pattern = url_path_join(base_url, 'jupyter-mynerva', 'config')
    chat_pattern = url_path_join(base_url, 'jupyter-mynerva', 'chat')
    sessions_pattern = url_path_join(base_url, 'jupyter-mynerva', 'sessions')
    session_pattern = url_path_join(base_url, 'jupyter-mynerva', 'sessions', '([^/]+)')
    nblibram_pattern = url_path_join(base_url, 'jupyter-mynerva', 'nblibram')
    agent_server_pattern = url_path_join(base_url, 'jupyter-mynerva', 'agent-server')
    enki_device_flow_pattern = url_path_join(base_url, 'jupyter-mynerva', 'enki-gate', 'device-flows')
    enki_device_flow_poll_pattern = url_path_join(base_url, 'jupyter-mynerva', 'enki-gate', 'device-flows', '([^/]+)', 'poll')
    handlers = [
        (providers_pattern, ProvidersHandler),
        (config_pattern, ConfigHandler),
        (chat_pattern, ChatHandler),
        (sessions_pattern, SessionsHandler),
        (session_pattern, SessionHandler),
        (nblibram_pattern, NblibramHandler),
        (agent_server_pattern, AgentServerHandler),
        (enki_device_flow_pattern, EnkiGateDeviceFlowHandler),
        (enki_device_flow_poll_pattern, EnkiGateDeviceFlowPollHandler)
    ]

    web_app.add_handlers(host_pattern, handlers)
