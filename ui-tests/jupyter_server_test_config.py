import json
import os
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

os.environ['MYNERVA_ECHO_AGENT'] = '1'

# Write echo config so Mynerva panel starts with echo provider
config_path = Path.home() / '.mynerva' / 'config.json'
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps({
    'provider': 'echo',
    'model': '',
    'apiKey': 'echo'
}))

configure_jupyter_server(c)
