import json
import logging
import os

import pytest
from cryptography.fernet import Fernet

from jupyter_mynerva.routes import (
    encrypt_api_key,
    decrypt_api_key,
    load_config,
    save_config,
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


