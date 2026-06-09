"""Echo agent for testing the agent loop without a real LLM.

Emits the same unified SSE format as the real LLM serializers, including native
tool calls (`tool_calls` + `assistant_blocks` on `message_done`).
"""

import logging

_log = logging.getLogger(__name__)

# Map trigger words in the user message to a tool call (name + input).
_TRIGGERS = {
    'toc': ('getToc', {}),
    'cells': ('getCells', {'query': {'start': 0}, 'count': 2}),
    'section': ('getSection', {'query': {'start': 0}}),
    'output': ('getOutput', {'query': {'start': 0}}),
    'list help': ('listHelp', {}),
    'help': ('help', {'action': 'getToc'}),
}


def _build_echo(messages):
    last = messages[-1] if messages else {}

    # Tool results came back -> finish with a plain text summary, no more calls.
    if last.get('toolResults'):
        summary = '; '.join(r.get('result', '') for r in last['toolResults'])
        return {'text': f'Echo done. {summary}', 'tool_calls': []}

    content = last.get('content', '')
    _log.info('chat_echo: %d messages, last=%s', len(messages), content[:200])
    lower = content.lower()
    name, tool_input = _TRIGGERS['toc']
    for trigger, spec in _TRIGGERS.items():
        if trigger in lower:
            name, tool_input = spec
            break

    return {
        'text': f'Echo: requesting {name}',
        'tool_calls': [{'id': f'echo_{name}', 'name': name, 'input': tool_input}],
    }


async def chat_echo(handler, messages):
    """Emit a canned native-tool-calling response as unified SSE events.

    The SSE init / error handling / finish is delegated to the sse_serializer
    decorator in routes. Imported lazily to avoid a circular import at module
    load time (routes imports chat_echo at module top).
    """
    from .routes import (  # lazy import to break the circular dep
        sse_serializer, _send_sse, _block_start, _block_delta, _block_stop,
    )

    @sse_serializer
    async def _run(h):
        body = _build_echo(messages)
        _block_start(h, 'text')
        if body['text']:
            _block_delta(h, 'text', body['text'])
        _block_stop(h, 'text')
        _send_sse(h, {'type': 'message_done',
                      'text': body['text'],
                      'stop_reason': 'tool_use' if body['tool_calls'] else 'end_turn',
                      'tool_calls': body['tool_calls'],
                      'assistant_blocks': None})
        _log.info('chat_echo: responded with %d tool calls',
                  len(body['tool_calls']))

    await _run(handler)
