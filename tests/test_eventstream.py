"""Tests for jupyter_mynerva.eventstream.EventStreamParser."""

from jupyter_mynerva.eventstream import EventStreamParser


def _encode_string_header(name, value):
    name_b = name.encode('utf-8')
    value_b = value.encode('utf-8')
    return (
        bytes([len(name_b)]) + name_b
        + bytes([7])  # value_type = string
        + len(value_b).to_bytes(2, 'big') + value_b
    )


def _encode_frame(headers, payload):
    """Build an AWS-event-stream frame; CRCs are zeroed (parser doesn't verify)."""
    headers_blob = b''.join(_encode_string_header(k, v) for k, v in headers.items())
    total_length = 12 + len(headers_blob) + len(payload) + 4
    return (
        total_length.to_bytes(4, 'big')
        + len(headers_blob).to_bytes(4, 'big')
        + b'\x00\x00\x00\x00'  # prelude CRC (not verified)
        + headers_blob
        + payload
        + b'\x00\x00\x00\x00'  # message CRC (not verified)
    )


def test_single_frame():
    parser = EventStreamParser()
    frame = _encode_frame(
        {':event-type': 'contentBlockDelta', ':message-type': 'event'},
        b'{"delta":{"text":"Hi"}}',
    )
    frames = parser.feed(frame)
    assert len(frames) == 1
    headers, payload = frames[0]
    assert headers == {':event-type': 'contentBlockDelta', ':message-type': 'event'}
    assert payload == b'{"delta":{"text":"Hi"}}'


def test_multiple_frames_in_one_feed():
    parser = EventStreamParser()
    a = _encode_frame({':event-type': 'contentBlockStart'}, b'{}')
    b = _encode_frame({':event-type': 'contentBlockDelta'}, b'{"delta":{"text":"x"}}')
    frames = parser.feed(a + b)
    assert [h[':event-type'] for h, _ in frames] == [
        'contentBlockStart', 'contentBlockDelta',
    ]


def test_partial_frame_buffered_across_feeds():
    parser = EventStreamParser()
    frame = _encode_frame(
        {':event-type': 'messageStop'},
        b'{"stopReason":"end_turn"}',
    )
    mid = len(frame) // 2
    assert parser.feed(frame[:mid]) == []
    frames = parser.feed(frame[mid:])
    assert len(frames) == 1
    headers, payload = frames[0]
    assert headers[':event-type'] == 'messageStop'
    assert payload == b'{"stopReason":"end_turn"}'


def test_exception_message_type():
    """The :message-type: exception frames carry error details in the payload."""
    parser = EventStreamParser()
    frame = _encode_frame(
        {':message-type': 'exception',
         ':exception-type': 'ValidationException'},
        b'{"message":"bad input"}',
    )
    headers, payload = parser.feed(frame)[0]
    assert headers[':message-type'] == 'exception'
    assert headers[':exception-type'] == 'ValidationException'
    assert payload == b'{"message":"bad input"}'


def test_unknown_header_type_stops_header_parsing_for_that_frame():
    """Unknown header types are unsafe to skip without length info; we stop
    parsing remaining headers in that frame rather than misinterpret bytes."""
    # Build a frame whose first header is string (parsed) and second header
    # has an unknown type byte. Parser should yield the first header only.
    name_b = b':event-type'
    value_b = b'contentBlockDelta'
    first_header = (
        bytes([len(name_b)]) + name_b
        + bytes([7])
        + len(value_b).to_bytes(2, 'big') + value_b
    )
    bad_name = b':custom'
    bad_header = bytes([len(bad_name)]) + bad_name + bytes([99])  # unknown type
    headers_blob = first_header + bad_header
    payload = b'{}'
    total_length = 12 + len(headers_blob) + len(payload) + 4
    frame = (
        total_length.to_bytes(4, 'big')
        + len(headers_blob).to_bytes(4, 'big')
        + b'\x00\x00\x00\x00'
        + headers_blob
        + payload
        + b'\x00\x00\x00\x00'
    )
    parser = EventStreamParser()
    headers, payload_out = parser.feed(frame)[0]
    assert headers == {':event-type': 'contentBlockDelta'}
    assert payload_out == b'{}'


def test_empty_feed_yields_nothing():
    assert EventStreamParser().feed(b'') == []


def test_byte_at_a_time_feed():
    parser = EventStreamParser()
    frame = _encode_frame({':event-type': 'messageStop'}, b'{}')
    out = []
    for byte in frame:
        out.extend(parser.feed(bytes([byte])))
    assert len(out) == 1
    assert out[0][0][':event-type'] == 'messageStop'


def test_zero_total_length_does_not_loop():
    """A malformed frame with total_length=0 must not cause an infinite loop."""
    parser = EventStreamParser()
    bad = b'\x00\x00\x00\x00' + b'\x00' * 8  # 12 bytes, total_length=0
    assert parser.feed(bad) == []
