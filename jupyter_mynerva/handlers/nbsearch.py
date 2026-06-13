import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode, urljoin

import cachetools
import tornado
from jupyter_server.base.handlers import APIHandler
from tornado.httpclient import AsyncHTTPClient, HTTPRequest


def NBSearchDB(*args, **kwargs):
    from nbsearch.db import NBSearchDB as _NBSearchDB
    return _NBSearchDB(*args, **kwargs)


# A tool with no useful action, attached to summary requests when
# summaryToolPriming is enabled: its mere presence shifts Gemma out of its
# chain-of-thought narration mode. tool_choice stays 'auto' (the priming has no
# effect under 'none'); a guard below handles the model occasionally calling it.
_SUMMARY_PRIMING_TOOL = {
    'type': 'function',
    'function': {
        'name': 'noop',
        'description': 'Unused. Do not call this; answer directly.',
        'parameters': {'type': 'object', 'properties': {}},
    },
}

_NBSEARCH_NOTEBOOK_FL = [
    'id',
    'filename',
    'owner',
    'server',
    'mtime',
    'ctime',
    'atime',
    'source__markdown__heading',
    'source__markdown__heading_count',
    'lc_cell_memes',
    'lc_cell_meme__execution_end_time',
    'score',
]

_NBSEARCH_NOTEBOOK_FIELDS = {
    'mtime': 'mtime',
}

_NBSEARCH_SORTS = {
    'notebooks': {
        'mtime desc',
        'mtime asc',
        'ctime desc',
        'ctime asc',
        'atime desc',
        'atime asc',
        'lc_cell_meme__execution_end_time desc',
        'lc_cell_meme__execution_end_time asc',
    },
}

_NBSEARCH_REFERENCE_CACHE = cachetools.TTLCache(maxsize=512, ttl=3600)
_NBSEARCH_REFERENCE_CELL_COUNT = 10000
_NBSEARCH_CELLS_PAGE_LIMIT_BYTES = 100 * 1024
_NBSEARCH_CELLS_MAX_LIMIT = 50

# Summaries feed the search-matched cells plus a neighbor window of `±k` cells
# (whole cells, never mid-cell truncated — partial code invites hallucination)
# together with the notebook's heading outline for global context. On a context
# overflow the window shrinks down this ladder; at k=0 (matched cells only) a
# residual overflow drops matched cells from the tail.
_NBSEARCH_WINDOW_K_LADDER = (10, 5, 2, 1, 0)

# System prompts for the two summary flavors (searchNotebooks relevance line vs
# summaryCellsFromSearch content paragraph). Both cite cell indexes/ranges.
# The notebook is given as Markdown (headings = structure, fenced blocks =
# code), shared per the user-approved Privacy filter. "!matched!" flags a part
# that matched the search. Summaries refer to locations by section heading,
# never by cell number (small models miscount indexes).
_NBSEARCH_RESULT_INSTRUCTION = (
    'The notebook below is given as Markdown; headings show its structure, '
    'fenced blocks are code, and a "!matched!" line marks a part that matched '
    'the search. Summarize why this notebook is relevant to the user focus, '
    'using only the provided content and referring to locations by section '
    'heading (never by cell number). Return one concise Japanese sentence.'
)
_NBSEARCH_CELLS_INSTRUCTION = (
    'The notebook below is given as Markdown; headings show its structure, '
    'fenced blocks are code, and "!matched!" lines mark parts that matched the '
    'search. Summarize the content relevant to the user focus, using only the '
    'provided content and referring to locations by section heading (never by '
    'cell number). Do not copy cell contents verbatim. Return one concise '
    'Japanese paragraph.'
)

# Substrings identifying a "prompt exceeds context window" error across
# providers and local OpenAI-compatible servers. Matching is best-effort: a
# miss re-raises and a false match only triggers a needless but still-correct
# window-shrink pass.
_CONTEXT_EXCEEDED_MARKERS = (
    'exceed_context_size_error',
    'exceeds the available context size',
    'context_length_exceeded',
    'maximum context length',
    'context window',
)


def _is_context_exceeded(exc):
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_EXCEEDED_MARKERS)


def _cell_source_text(cell):
    source = cell.get('source')
    if isinstance(source, list):
        return ''.join(source)
    return source or ''


def _cell_headings(cell):
    """ATX heading lines of a markdown cell (the structural backbone)."""
    if cell.get('cell_type') != 'markdown':
        return []
    return [
        line.rstrip() for line in _cell_source_text(cell).splitlines()
        if line.lstrip().startswith('#')
    ]


def _window_indexes(matched_indexes, k):
    """Cell indexes within `±k` of each matched index (or of the top when there
    is no match)."""
    keep = set()
    for anchor in matched_indexes or [0]:
        keep.update(range(anchor - k, anchor + k + 1))
    return keep


def _render_markdown(cells, matched_indexes, window_keep, backbone_keep):
    """Render cells as a Markdown document for summarization.

    Cells in `window_keep` render in full (markdown verbatim, code fenced);
    cells in `backbone_keep` keep only their heading lines (the structural
    backbone for global context); others are dropped. A matched cell is
    preceded by a "!matched!" marker line. No cell indexes appear — small models
    cite them poorly, so relevance is carried by the marker, not numbers.
    """
    matched = set(matched_indexes)
    parts = []
    for cell in cells:
        index = cell.get('_index', 0)
        if index in window_keep:
            text = _cell_source_text(cell).strip()
            block = text if cell.get('cell_type') == 'markdown' else f'```\n{text}\n```'
        elif index in backbone_keep:
            headings = _cell_headings(cell)
            if not headings:
                continue
            block = '\n'.join(headings)
        else:
            continue
        if index in matched:
            parts.append('!matched!')
        parts.append(block)
    return '\n\n'.join(parts)


def _solr_phrase(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _build_nbsearch_query(data):
    return data.get('query') or '*:*'


def _build_nbsearch_filter_queries(data, fields):
    parts = []
    if data.get('dateTimeFrom') or data.get('dateTimeTo'):
        date_from = data.get('dateTimeFrom') or '*'
        date_to = data.get('dateTimeTo') or '*'
        parts.append(f"{fields['mtime']}:[{date_from} TO {date_to}]")
        if data.get('dateTimeTo'):
            parts.append(f"-{fields['mtime']}:{_solr_phrase(date_to)}")
    return parts


def _validate_utc_datetime(value, field_name):
    if not isinstance(value, str) or not value.endswith('Z'):
        raise tornado.web.HTTPError(400, reason=f'{field_name} must be a UTC ISO datetime ending with Z')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as e:
        raise tornado.web.HTTPError(400, reason=f'{field_name} must be a valid UTC ISO datetime') from e
    if parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise tornado.web.HTTPError(400, reason=f'{field_name} must be UTC')


def _validate_nbsearch_date_range(data):
    if data.get('dateFrom') and not data.get('dateTimeFrom'):
        raise tornado.web.HTTPError(400, reason='dateFrom must be normalized to dateTimeFrom by the client')
    if data.get('dateTo') and not data.get('dateTimeTo'):
        raise tornado.web.HTTPError(400, reason='dateTo must be normalized to dateTimeTo by the client')
    if data.get('dateTimeFrom'):
        _validate_utc_datetime(data['dateTimeFrom'], 'dateTimeFrom')
    if data.get('dateTimeTo'):
        _validate_utc_datetime(data['dateTimeTo'], 'dateTimeTo')


def has_nbsearch_config(config):
    return config is not None and 'NBSearchDB' in config


def _shape_nbsearch_notebook_reference(doc, query):
    return {
        'filename': doc['filename'],
        'notebookId': doc['id'],
        # nblibram fetch (all cells); kept distinct from the Solr query below.
        'query': {'start': 0},
        # The original Solr query, reused to locate matched cells for summaries.
        'solrQuery': query,
        'start': None,
        'count': _NBSEARCH_REFERENCE_CELL_COUNT,
    }


def _estimate_json_size(value, limit):
    seen = set()

    def walk(item):
        object_id = id(item)
        if isinstance(item, (dict, list, tuple)):
            if object_id in seen:
                return 0
            seen.add(object_id)

        if item is None:
            return 4
        if isinstance(item, bool):
            return 4 if item else 5
        if isinstance(item, (int, float)):
            return len(str(item))
        if isinstance(item, str):
            return len(item.encode('utf-8')) + 2
        if isinstance(item, dict):
            size = 2
            for key, child in item.items():
                size += walk(str(key)) + 1 + walk(child) + 1
                if size > limit:
                    return size
            return size
        if isinstance(item, (list, tuple)):
            size = 2
            for child in item:
                size += walk(child) + 1
                if size > limit:
                    return size
            return size
        return len(str(item).encode('utf-8')) + 2

    return walk(value)


def _source_only_cell(cell):
    if not isinstance(cell, dict):
        return cell
    # nblibram omits `_index` for cell 0 (JSON omitempty on the zero int), so
    # default it to 0 — windowing and TOC matching key off this index.
    result = {'_index': cell.get('_index', 0)}
    if 'cell_type' in cell:
        result['cell_type'] = cell['cell_type']
    if 'source' in cell:
        result['source'] = cell['source']
    return result


def _source_only_cells(cells_result):
    """Flatten an nblibram cells_result into a source-only cell list.

    Outputs are dropped (summaries judge relevance from source, and outputs
    bloat the context). Each cell keeps its `_index` so the summary window can
    select whole cells and cite ranges.
    """
    return [_source_only_cell(cell) for cell in cells_result['cells']]


def _parse_non_negative_int(value, name, default=0):
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise tornado.web.HTTPError(400, reason=f'{name} must be an integer') from e
    if parsed < 0:
        raise tornado.web.HTTPError(400, reason=f'{name} must be >= 0')
    return parsed


def _parse_optional_limit(value):
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise tornado.web.HTTPError(400, reason='limit must be an integer') from e
    if parsed < 1 or parsed > _NBSEARCH_CELLS_MAX_LIMIT:
        raise tornado.web.HTTPError(400, reason=f'limit must be between 1 and {_NBSEARCH_CELLS_MAX_LIMIT}')
    return parsed


def _paginate_cells_result(cells_result, start=0, limit=None, budget=_NBSEARCH_CELLS_PAGE_LIMIT_BYTES):
    if not isinstance(cells_result, dict):
        return cells_result

    cells = cells_result.get('cells')
    if not isinstance(cells, list):
        return cells_result

    total = len(cells)
    selected = []
    index = min(start, total)
    while index < total:
        if limit is not None and len(selected) >= limit:
            break
        candidate = cells[index]
        candidate_page = selected + [candidate]
        if selected and _estimate_json_size(candidate_page, budget) > budget:
            break
        selected = candidate_page
        index += 1

    result = {key: value for key, value in cells_result.items() if key != 'cells'}
    result['cells'] = selected
    result['total'] = total
    result['hasMore'] = index < total
    if index < total:
        result['nextStart'] = index
    return result


# Targets whose work calls the LLM and can run for minutes (map-reduce). These
# stream SSE so a reverse proxy's idle timeout never fires and the UI can show
# progress; the cheap, LLM-free target stays a plain JSON request.
_NBSEARCH_STREAMING_TARGETS = ('notebooks', 'summary-cells-from-search')
_NBSEARCH_HEARTBEAT_INTERVAL = 15


class NBSearchHandler(APIHandler):
    # Per-request progress sink, set only while streaming (see _post_streaming).
    _on_progress = None

    def on_connection_close(self):
        """Cancel in-flight work when the client disconnects (e.g. user abort).

        Summarization can fan out into many LLM calls (map-reduce over a large
        notebook); cancelling stops that work instead of running it to waste.
        """
        super().on_connection_close()
        task = getattr(self, '_work_task', None)
        if task is not None and not task.done():
            self.log.info('nbsearch client disconnected; cancelling in-flight work')
            task.cancel()

    def _emit_progress(self, event):
        """Report progress to the SSE stream, if one is active."""
        if self._on_progress is not None:
            self._on_progress(event)

    async def _run_target(self, target, data):
        if target == 'cells-from-search':
            return await self._get_cells_from_search(data)
        if target == 'summary-cells-from-search':
            return await self._summary_cells_from_search(data)
        return await self._search_notebooks(target, data)

    @tornado.web.authenticated
    async def post(self, target):
        data = self.get_json_body()
        self._work_task = asyncio.current_task()
        if target in _NBSEARCH_STREAMING_TARGETS:
            await self._post_streaming(target, data)
        else:
            await self._post_json(target, data)

    async def _post_json(self, target, data):
        try:
            result = await self._run_target(target, data)
        except asyncio.CancelledError:
            self.log.info('nbsearch request cancelled by client: target=%s', target)
            raise
        except tornado.web.HTTPError as e:
            self.set_status(e.status_code)
            self.finish(json.dumps({'error': e.reason or e.log_message}))
            return
        except Exception:
            self.log.exception('Unhandled nbsearch request failure: target=%s', target)
            self.set_status(500)
            self.finish(json.dumps({'error': 'nbsearch request failed; see server logs'}))
            return
        self.finish(json.dumps(result))

    async def _post_streaming(self, target, data):
        from jupyter_mynerva import routes

        routes._init_sse(self)
        self._on_progress = lambda event: routes._send_sse(
            self, {'type': 'progress', **event})
        heartbeat = asyncio.ensure_future(self._sse_heartbeat())
        try:
            result = await self._run_target(target, data)
            routes._send_sse(self, {'type': 'done', 'result': result})
        except asyncio.CancelledError:
            self.log.info('nbsearch request cancelled by client: target=%s', target)
            raise
        except tornado.web.HTTPError as e:
            routes._send_sse(self, {'type': 'error', 'error': e.reason or e.log_message})
        except Exception:
            self.log.exception('Unhandled nbsearch request failure: target=%s', target)
            routes._send_sse(
                self, {'type': 'error', 'error': 'nbsearch request failed; see server logs'})
        finally:
            heartbeat.cancel()
            self._on_progress = None
            try:
                routes._finish_sse(self)
            except Exception:
                # Connection already closed (e.g. client aborted); nothing to flush.
                pass

    async def _sse_heartbeat(self):
        """Emit SSE comments so an idle reverse proxy never times out the stream.

        A single LLM call can exceed a proxy's read timeout on its own, so the
        heartbeat runs on its own timer rather than only between work steps.
        """
        try:
            while True:
                await asyncio.sleep(_NBSEARCH_HEARTBEAT_INTERVAL)
                self.write(': keepalive\n\n')
                self.flush()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Stream closed; stop heartbeating.
            return

    def _handler_config(self):
        config = getattr(self, 'config', None)
        if config is not None:
            return config
        return self.settings.get('config')

    def _create_db(self):
        config = self._handler_config()
        if not has_nbsearch_config(config):
            raise tornado.web.HTTPError(503, reason='NBSearchDB config is not available')
        try:
            return NBSearchDB(config=config)
        except ModuleNotFoundError as e:
            raise tornado.web.HTTPError(503, reason='nbsearch is not installed') from e

    def _http_kwargs(self, db):
        if db.solr_basic_auth_username or db.solr_basic_auth_password:
            return {
                'auth_mode': 'basic',
                'auth_username': db.solr_basic_auth_username,
                'auth_password': db.solr_basic_auth_password,
            }
        return {}

    async def _search_notebooks(self, target, data):
        db = self._create_db()
        if target != 'notebooks':
            raise tornado.web.HTTPError(404, reason=f'Unknown nbsearch target: {target}')
        if not data.get('query'):
            raise tornado.web.HTTPError(400, reason='query is required')
        if not data.get('focus'):
            raise tornado.web.HTTPError(400, reason='focus is required')
        _validate_nbsearch_date_range(data)

        query = _build_nbsearch_query(data)
        filters = _build_nbsearch_filter_queries(data, _NBSEARCH_NOTEBOOK_FIELDS)
        start = int(data.get('start', 0))
        limit = int(data.get('limit', 10))
        if start < 0:
            raise tornado.web.HTTPError(400, reason='start must be >= 0')
        if limit < 1 or limit > 50:
            raise tornado.web.HTTPError(400, reason='limit must be between 1 and 50')

        params = {
            'q': query,
            'q.op': data.get('qOp', 'AND'),
            'wt': 'json',
            'start': start,
            'rows': limit,
            'fl': ','.join(_NBSEARCH_NOTEBOOK_FL),
        }
        if filters:
            params['fq'] = filters
        if data.get('sort'):
            sort = data['sort']
            if sort not in _NBSEARCH_SORTS[target]:
                raise tornado.web.HTTPError(400, reason=f'unsupported sort: {sort}')
            params['sort'] = sort

        url = urljoin(db.solr_base_url + '/', f'solr/{db.solr_notebook}/select?{urlencode(params, doseq=True)}')
        try:
            response = await AsyncHTTPClient().fetch(
                HTTPRequest(url, method='GET', **self._http_kwargs(db)),
                raise_error=False,
            )
        except Exception:
            self.log.exception(
                'Failed to query nbsearch Solr: target=%s query=%r start=%s limit=%s',
                target,
                query,
                start,
                limit,
            )
            raise
        if response.code >= 400:
            self.log.error(
                'nbsearch Solr query failed: status=%s target=%s query=%r start=%s limit=%s',
                response.code,
                target,
                query,
                start,
                limit,
            )
            raise tornado.web.HTTPError(
                502,
                reason=f'Solr query failed with status {response.code}',
            )

        result = json.loads(response.body)
        response_body = result['response']
        docs = response_body['docs']
        num_found = response_body.get('numFound', len(docs))
        search_id = f"search-{uuid.uuid4().hex}"
        shaped_docs = []
        for index, doc in enumerate(docs):
            self._emit_progress({
                'phase': 'notebook',
                'current': index + 1,
                'total': len(docs),
            })
            shaped_docs.append(await self._shape_notebook_result(
                db,
                search_id,
                index,
                doc,
                data['query'],
                data['focus'],
                bool(data.get('noFilter')),
            ))
        return {
            'target': target,
            'searchId': search_id,
            'query': query,
            'focus': data['focus'],
            'start': start,
            'limit': limit,
            'returned': len(shaped_docs),
            'numFound': num_found,
            'results': shaped_docs,
        }

    async def _get_cells_from_search(self, data):
        reference_id = data.get('referenceId')
        if not reference_id:
            raise tornado.web.HTTPError(400, reason='referenceId is required')
        if reference_id not in _NBSEARCH_REFERENCE_CACHE:
            raise tornado.web.HTTPError(404, reason='search reference not found')

        start = _parse_non_negative_int(data.get('start'), 'start')
        limit = _parse_optional_limit(data.get('limit'))
        db = self._create_db()
        reference = _NBSEARCH_REFERENCE_CACHE[reference_id]
        cells = await self._get_search_reference_cells(
            db,
            reference,
            bool(data.get('noFilter')),
        )
        result = _paginate_cells_result(cells, start=start, limit=limit)
        return {
            'type': 'getCellsFromSearch',
            'referenceId': reference_id,
            'filename': reference['filename'],
            'result': result,
        }

    async def _summary_cells_from_search(self, data):
        reference_id = data.get('referenceId')
        if not reference_id:
            raise tornado.web.HTTPError(400, reason='referenceId is required')
        if reference_id not in _NBSEARCH_REFERENCE_CACHE:
            raise tornado.web.HTTPError(404, reason='search reference not found')
        if not data.get('focus'):
            raise tornado.web.HTTPError(400, reason='focus is required')

        db = self._create_db()
        reference = _NBSEARCH_REFERENCE_CACHE[reference_id]
        cells = await self._get_search_reference_cells(
            db,
            reference,
            bool(data.get('noFilter')),
        )
        cell_count = len(cells['cells'])
        matched_indexes = await self._matched_cell_indexes(
            db, reference['notebookId'], reference.get('solrQuery'),
        )
        coverage = f'{len(matched_indexes)} matched / {cell_count} cells'
        self._emit_progress({'phase': 'summarize', 'detail': 'summarizing cells'})
        return {
            'type': 'summaryCellsFromSearch',
            'referenceId': reference_id,
            'filename': reference['filename'],
            'cellCount': cell_count,
            'coverage': coverage,
            'summary': await self._summarize_search_cells(
                data['focus'],
                reference['filename'],
                cells,
                matched_indexes,
                coverage,
            ),
            'readAction': {
                'type': 'getCellsFromSearch',
                'referenceId': reference_id,
            },
        }

    async def _download_notebook_to_tempfile(self, db, notebook_id):
        fd, path = tempfile.mkstemp(suffix='.ipynb', prefix='nbsearch-')
        try:
            with os.fdopen(fd, 'wb') as f:
                await db.download_file(notebook_id, f)
        except Exception:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            self.log.exception('Failed to download nbsearch notebook payload: notebook_id=%s', notebook_id)
            raise
        return path

    def _is_valid_notebook_payload(self, path):
        result = subprocess.run(
            [sys.executable, '-m', 'json.tool', path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0

    def _run_nblibram_cells(self, path, query, count, no_filter):
        nblibram_path = shutil.which('nblibram')
        if not nblibram_path:
            raise tornado.web.HTTPError(500, reason='nblibram not found in PATH')

        query_key, query_value = next(iter(query.items()))
        args = [
            nblibram_path,
            'cells',
            '-file',
            path,
            '-format',
            'json',
            '-query',
            f'{query_key}:{query_value}',
            '-count',
            str(count),
        ]
        if no_filter:
            args.append('-no-filter')
        result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode != 0:
            self.log.error('nblibram cells failed for nbsearch reference: stderr=%s', result.stderr.strip())
            raise tornado.web.HTTPError(400, reason=result.stderr.strip())
        return json.loads(result.stdout)

    async def _get_search_reference_cells(self, db, reference, no_filter):
        notebook_path = await self._download_notebook_to_tempfile(db, reference['notebookId'])
        try:
            if not await asyncio.to_thread(self._is_valid_notebook_payload, notebook_path):
                self.log.error(
                    'Invalid nbsearch notebook payload: notebook_id=%s filename=%s',
                    reference['notebookId'],
                    reference['filename'],
                )
                return {
                    'cells': [],
                    'error': 'notebook payload is not valid JSON',
                }
            return await asyncio.to_thread(
                self._run_nblibram_cells,
                notebook_path,
                reference['query'],
                reference['count'],
                no_filter,
            )
        finally:
            os.unlink(notebook_path)

    async def _run_summary(self, provider, model, api_key, base_url, config, messages):
        """Send one summarization request and return the text.

        Exceptions propagate unchanged so the adaptive caller can detect a
        context-window overflow and fall back to map-reduce.
        """
        from jupyter_mynerva import routes

        if provider == 'openai':
            kwargs = {'api_key': api_key or ''}
            if base_url:
                kwargs['base_url'] = base_url
            client = routes.AsyncOpenAI(**kwargs)
            if base_url:
                # Custom endpoint (vLLM etc.): use chat/completions like the
                # chat path; its Responses API leaks reasoning into the summary.
                create = {'model': model, 'messages': messages}
                if routes._advanced_flag(
                        config, 'summaryToolPriming', routes._SUMMARY_TOOL_PRIMING):
                    create['tools'] = [_SUMMARY_PRIMING_TOOL]
                    create['tool_choice'] = 'auto'
                msg = (await client.chat.completions.create(**create)).choices[0].message
                if not (msg.content or '').strip() and msg.tool_calls:
                    # Model called the priming tool instead of answering; retry
                    # once without it so the summary isn't lost.
                    msg = (await client.chat.completions.create(
                        model=model, messages=messages)).choices[0].message
                return (msg.content or '').strip()
            response = await client.responses.create(model=model, input=messages)
            return response.output_text.strip()
        if provider == 'enki-gate':
            enki_token = config.get('enkiGateToken')
            enki_url = config.get('enkiGateUrl')
            enki_model = config.get('enkiGateModel', '')
            client = routes.AsyncOpenAI(
                api_key=enki_token, base_url=enki_url.rstrip('/') + '/v1'
            )
            response = await client.responses.create(model=enki_model, input=messages)
            return response.output_text.strip()
        if provider == 'anthropic':
            client = routes.AsyncAnthropic(api_key=api_key)
            kwargs = routes._build_anthropic_params(messages)
            response = await client.messages.create(model=model, **kwargs)
            return ''.join(
                block.text for block in response.content if block.type == 'text'
            ).strip()
        raise tornado.web.HTTPError(400, reason=f'Unknown provider: {provider}')

    async def _matched_cell_indexes(self, db, notebook_id, solr_query):
        """Cell indexes in `notebook_id` matching the search query, via the
        jupyter-cell core. Returns [] on no match or query error, so the window
        falls back to the notebook's start."""
        params = {
            'q': solr_query or '*:*',
            'fq': f'notebook_id:{_solr_phrase(notebook_id)}',
            'fl': 'index',
            'rows': _NBSEARCH_REFERENCE_CELL_COUNT,
            'wt': 'json',
        }
        url = urljoin(
            db.solr_base_url + '/',
            f'solr/{db.solr_cell}/select?{urlencode(params, doseq=True)}',
        )
        response = await AsyncHTTPClient().fetch(
            HTTPRequest(url, method='GET', **self._http_kwargs(db)),
            raise_error=False,
        )
        if response.code >= 400:
            self.log.warning(
                'nbsearch cell-match query failed (status=%s, notebook_id=%s); '
                'summarizing from notebook start', response.code, notebook_id,
            )
            return []
        docs = json.loads(response.body)['response']['docs']
        return sorted({d['index'] for d in docs if 'index' in d})

    async def _summarize(self, focus, filename, cells_result, matched_indexes,
                         instruction, echo_text, extra=None):
        """Summarize a notebook's cells for `focus`, fitting the model context.

        The notebook is rendered as Markdown: a heading backbone for global
        context, plus the search-matched cells with a neighbor window shown in
        full and flagged with a marker. On a context overflow the input shrinks
        until it fits — first the full-content window (`_NBSEARCH_WINDOW_K_LADDER`
        then dropping matched cells from the tail), then the heading backbone —
        so even the smallest payload is guaranteed to fit. `extra` adds header
        lines (e.g. coverage).
        """
        from jupyter_mynerva import routes

        config = await routes.load_config()
        provider, model, api_key, base_url = await routes.resolve_chat_config(config)
        if provider == 'echo':
            return echo_text
        cells = _source_only_cells(cells_result)
        backbone = {c['_index'] for c in cells if _cell_headings(c)}

        async def attempt(window_keep, backbone_keep):
            header = f'focus: {focus}\nnotebook: {filename}'
            if extra and extra.get('coverage'):
                header += f"\ncoverage: {extra['coverage']}"
            document = _render_markdown(
                cells, matched_indexes, window_keep, backbone_keep)
            return await self._run_summary(
                provider, model, api_key, base_url, config,
                [
                    {'role': 'system', 'content': instruction},
                    {'role': 'user', 'content': f'{header}\n\n{document}'},
                ],
            )

        async def shrink(stages):
            last_exc = None
            for window_keep, backbone_keep in stages:
                try:
                    return await attempt(window_keep, backbone_keep)
                except Exception as exc:
                    if not _is_context_exceeded(exc):
                        raise
                    last_exc = exc
            raise last_exc

        def tail_drop(indexes):
            ordered = sorted(indexes)
            while ordered:
                ordered = ordered[:-1]
                yield set(ordered)

        stages = [
            (_window_indexes(matched_indexes, k), backbone)
            for k in _NBSEARCH_WINDOW_K_LADDER
        ]
        # k=0 still overflows: drop matched cells (full content) from the tail.
        stages += [(w, backbone) for w in tail_drop(_window_indexes(matched_indexes, 0))]
        # Backbone alone still overflows: drop heading cells from the tail.
        stages += [(set(), bb) for bb in tail_drop(backbone)]
        return await shrink(stages)

    async def _summarize_result(self, focus, filename, cells_result, matched_indexes):
        return await self._summarize(
            focus, filename, cells_result, matched_indexes,
            _NBSEARCH_RESULT_INSTRUCTION,
            f'{filename} は検索 focus に関連する notebook です。'
            '関連箇所は提示されたセル番号を確認してください。',
        )

    async def _summarize_search_cells(self, focus, filename, cells_result,
                                      matched_indexes, coverage):
        return await self._summarize(
            focus, filename, cells_result, matched_indexes,
            _NBSEARCH_CELLS_INSTRUCTION,
            f'{filename} は focus に関連するセル範囲を含みます。'
            '要約内のセル番号を使って getCellsFromSearch で詳細を確認してください。',
            extra={'coverage': coverage},
        )

    def _shape_notebook_metadata(self, doc):
        return {
            key: doc[key]
            for key in ['owner', 'server', 'mtime', 'ctime', 'atime']
            if key in doc
        }

    async def _shape_notebook_result(self, db, search_id, index, doc, query, focus, no_filter):
        reference_id = f'{search_id}/r{index + 1}/ref1'
        reference = _shape_nbsearch_notebook_reference(doc, query)
        _NBSEARCH_REFERENCE_CACHE[reference_id] = reference
        cells = await self._get_search_reference_cells(db, reference, no_filter)
        matched_indexes = await self._matched_cell_indexes(db, doc['id'], query)
        references = [{
            'type': 'summaryCellsFromSearch',
            'referenceId': reference_id,
        }]
        return {
            'filename': doc['filename'],
            **self._shape_notebook_metadata(doc),
            'summary': await self._summarize_result(
                focus,
                doc['filename'],
                cells,
                matched_indexes,
            ),
            'references': references,
        }
