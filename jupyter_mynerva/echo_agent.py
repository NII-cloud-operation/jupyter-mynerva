"""Echo agent for testing the agent loop without a real LLM."""

import json
import logging

_log = logging.getLogger(__name__)

# Map trigger words in user message to actions
_TRIGGERS = {
    'toc': {'type': 'getToc'},
    'cells': {'type': 'getCells', 'query': {'start': 0}, 'count': 2},
    'section': {'type': 'getSection', 'query': {'start': 0}},
    'output': {'type': 'getOutput', 'query': {'start': 0}},
}


def chat_echo(messages):
    last_msg = messages[-1].get('content', '') if messages else ''
    _log.info("chat_echo: %d messages, last=%s", len(messages), last_msg[:200])

    if '[Action Results]' in last_msg:
        # Echo back the action results as-is
        body = {
            'messages': [{'role': 'assistant', 'content': last_msg}],
            'actions': []
        }
    else:
        # Pick action based on trigger word in user message
        lower = last_msg.lower()
        action = None
        for trigger, act in _TRIGGERS.items():
            if trigger in lower:
                action = act
                break
        if action is None:
            action = _TRIGGERS['toc']

        body = {
            'messages': [{'role': 'assistant', 'content': f'Echo: requesting {action["type"]}'}],
            'actions': [action]
        }

    _log.info("chat_echo: responding with %d actions", len(body['actions']))
    return {
        'provider': 'openai',
        'response': {'choices': [{'message': {'content': json.dumps(body)}}]}
    }
