"""Echo agent for testing the agent loop without a real LLM.

Emits the same unified SSE format as the real LLM serializers.
"""

import json
import logging

_log = logging.getLogger(__name__)

# Map trigger words in user message to actions
_TRIGGERS = {
    'toc': {'type': 'getToc'},
    'cells': {'type': 'getCells', 'query': {'start': 0}, 'count': 2},
    'section': {'type': 'getSection', 'query': {'start': 0}},
    'output': {'type': 'getOutput', 'query': {'start': 0}},
    'list help': {'type': 'listHelp'},
    'help': {'type': 'help', 'action': 'getToc'},
}


def _build_echo_body(messages):
    last_msg = messages[-1].get('content', '') if messages else ''
    _log.info("chat_echo: %d messages, last=%s", len(messages), last_msg[:200])

    if '[Action Results]' in last_msg:
        return {
            'messages': [{'role': 'assistant', 'content': last_msg}],
            'actions': []
        }

    lower = last_msg.lower()
    action = None
    for trigger, act in _TRIGGERS.items():
        if trigger in lower:
            action = act
            break
    if action is None:
        action = _TRIGGERS['toc']

    return {
        'messages': [{'role': 'assistant', 'content': f'Echo: requesting {action["type"]}'}],
        'actions': [action]
    }


async def chat_echo(handler, messages):
    """Emit a canned LLM-style response as unified SSE events.

    The body is a Mynerva action-protocol JSON so processLLMResponse on the
    frontend receives it unchanged, matching how real LLM serializers work.

    The SSE init / error handling / finish is delegated to the sse_serializer
    decorator in routes. Imported lazily to avoid a circular import at module
    load time (routes imports chat_echo at module top).
    """
    from .routes import (  # lazy import to break the circular dep
        sse_serializer, _send_sse, _block_start, _block_delta, _block_stop,
    )

    @sse_serializer
    async def _run(h):
        body = _build_echo_body(messages)
        text = json.dumps(body)
        assistant_text = body['messages'][0]['content'] if body['messages'] else ''

        _block_start(h, 'thinking')
        _block_stop(h, 'thinking')
        _block_start(h, 'text')
        if assistant_text:
            _block_delta(h, 'text', assistant_text)
        _block_stop(h, 'text')
        _send_sse(h, {'type': 'message_done', 'text': text,
                      'stop_reason': 'completed'})
        _log.info("chat_echo: responded with %d actions", len(body.get('actions', [])))

    await _run(handler)
