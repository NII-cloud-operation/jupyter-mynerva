"""Incremental parser for the AWS Event Stream binary frame format.

Used by the Bedrock Converse Stream response (Content-Type
application/vnd.amazon.eventstream). The format is documented at
https://docs.aws.amazon.com/transcribe/latest/dg/event-stream.html.

Frame layout (all integers big-endian):
    [4] total_length  - total bytes including these 12 prelude bytes and the
                        trailing CRC
    [4] headers_length
    [4] prelude_crc   - not verified here; TLS already guarantees integrity
    [...] headers     - sequence of (name_len:u8, name:utf8, value_type:u8,
                        value:...) tuples
    [...] payload     - usually JSON (utf-8)
    [4] message_crc   - not verified here

Only string-typed headers (value_type=7, uint16-length-prefixed utf-8) are
needed for Converse Stream events; other types are skipped.
"""


_HEADER_TYPE_STRING = 7


class EventStreamParser:
    """Feed bytes; receive complete (headers, payload) frames.

    Partial frames are buffered across feed() calls.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, chunk):
        """Append bytes and return whatever complete frames are now available.

        Returns a list of (headers_dict, payload_bytes) tuples. Headers are
        decoded as utf-8 strings; non-string header types are skipped.
        """
        self._buffer.extend(chunk)
        frames = []
        while len(self._buffer) >= 12:
            total_length = int.from_bytes(self._buffer[0:4], 'big')
            if total_length < 12:
                break
            headers_length = int.from_bytes(self._buffer[4:8], 'big')
            if len(self._buffer) < total_length:
                break
            frame = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            headers = self._parse_headers(frame[12:12 + headers_length])
            payload = frame[12 + headers_length:total_length - 4]
            frames.append((headers, payload))
        return frames

    @staticmethod
    def _parse_headers(raw):
        headers = {}
        i = 0
        n = len(raw)
        while i < n:
            name_len = raw[i]
            i += 1
            name = raw[i:i + name_len].decode('utf-8')
            i += name_len
            value_type = raw[i]
            i += 1
            if value_type == _HEADER_TYPE_STRING:
                value_len = int.from_bytes(raw[i:i + 2], 'big')
                i += 2
                headers[name] = raw[i:i + value_len].decode('utf-8')
                i += value_len
            else:
                # Other types unused by Converse Stream events; stop parsing
                # the rest of this header block rather than guess at lengths.
                break
        return headers
