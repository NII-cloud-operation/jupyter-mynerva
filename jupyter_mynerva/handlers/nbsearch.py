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
    'owner': 'owner',
    'filename': 'filename',
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
_NBSEARCH_SUMMARY_FULL_CELLS_LIMIT_BYTES = 100 * 1024
_NBSEARCH_CELLS_PAGE_LIMIT_BYTES = 100 * 1024
_NBSEARCH_CELLS_MAX_LIMIT = 50


def _solr_phrase(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _nbsearch_query_value(value, exact):
    return _solr_phrase(value) if exact else value


def _build_nbsearch_query(data, fields):
    exact = bool(data.get('exact'))
    parts = []
    if data.get('query'):
        parts.append(_nbsearch_query_value(data['query'], exact))
    if data.get('owner'):
        parts.append(f"{fields['owner']}:{_solr_phrase(data['owner'])}")
    if data.get('filename'):
        parts.append(f"{fields['filename']}:{_solr_phrase(data['filename'])}")
    return ' AND '.join(parts) if parts else '*:*'


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
        'path': doc['filename'],
        'notebookId': doc['id'],
        'query': {'start': 0},
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
    result = {}
    if '_index' in cell:
        result['_index'] = cell['_index']
    if 'cell_type' in cell:
        result['cell_type'] = cell['cell_type']
    if 'source' in cell:
        result['source'] = cell['source']
    return result


def _source_only_cells_result(cells_result):
    if not isinstance(cells_result, dict):
        return cells_result
    result = {}
    if 'error' in cells_result:
        result['error'] = cells_result['error']
    if 'cells' in cells_result:
        result['cells'] = [_source_only_cell(cell) for cell in cells_result['cells']]
    return result


def _prepare_summary_cells(cells):
    if _estimate_json_size(cells, _NBSEARCH_SUMMARY_FULL_CELLS_LIMIT_BYTES) <= _NBSEARCH_SUMMARY_FULL_CELLS_LIMIT_BYTES:
        return cells
    summary_cells = []
    for item in cells:
        if not isinstance(item, dict):
            summary_cells.append(item)
            continue
        summary_item = {}
        if 'referenceId' in item:
            summary_item['referenceId'] = item['referenceId']
        if 'cells' in item:
            summary_item['cells'] = _source_only_cells_result(item['cells'])
        summary_cells.append(summary_item)
    return summary_cells


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


class NBSearchHandler(APIHandler):
    @tornado.web.authenticated
    async def post(self, target):
        data = self.get_json_body()
        try:
            if target == 'cells-from-search':
                result = await self._get_cells_from_search(data)
            else:
                result = await self._search_notebooks(target, data)
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

        query = _build_nbsearch_query(data, _NBSEARCH_NOTEBOOK_FIELDS)
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
        shaped_docs = [
            await self._shape_notebook_result(
                db,
                search_id,
                index,
                doc,
                data['query'],
                data['focus'],
                bool(data.get('noFilter')),
            )
            for index, doc in enumerate(docs)
        ]
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
            'path': reference['path'],
            'result': result,
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
                    'Invalid nbsearch notebook payload: notebook_id=%s path=%s',
                    reference['notebookId'],
                    reference['path'],
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

    async def _summarize_result(self, focus, path, cells):
        from jupyter_mynerva import routes

        config = await routes.load_config()
        provider, model, api_key, base_url = await routes.resolve_chat_config(config)
        cells = _prepare_summary_cells(cells)
        messages = [
            {
                'role': 'system',
                'content': (
                    'Summarize why this notebook is relevant to the user focus. '
                    'Use only the provided privacy-filtered cells. '
                    'Include the supporting cell index or range in the sentence, '
                    'such as "セル12-14では...". '
                    'Return one concise Japanese sentence.'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'focus': focus,
                    'path': path,
                    'cells': cells,
                }, ensure_ascii=False),
            },
        ]
        if provider == 'openai':
            kwargs = {'api_key': api_key or ''}
            if base_url:
                kwargs['base_url'] = base_url
            client = routes.AsyncOpenAI(**kwargs)
            try:
                response = await client.responses.create(model=model, input=messages)
            except Exception:
                self.log.exception(
                    'Failed to summarize nbsearch result: provider=%s model=%s path=%s',
                    provider,
                    model,
                    path,
                )
                raise
            return response.output_text.strip()
        if provider == 'enki-gate':
            enki_token = config.get('enkiGateToken')
            enki_url = config.get('enkiGateUrl')
            enki_model = config.get('enkiGateModel', '')
            client = routes.AsyncOpenAI(api_key=enki_token, base_url=enki_url.rstrip('/') + '/v1')
            try:
                response = await client.responses.create(model=enki_model, input=messages)
            except Exception:
                self.log.exception(
                    'Failed to summarize nbsearch result: provider=%s model=%s path=%s',
                    provider,
                    enki_model,
                    path,
                )
                raise
            return response.output_text.strip()
        if provider == 'anthropic':
            client = routes.AsyncAnthropic(api_key=api_key)
            kwargs = routes._build_anthropic_params(messages)
            try:
                response = await client.messages.create(model=model, **kwargs)
            except Exception:
                self.log.exception(
                    'Failed to summarize nbsearch result: provider=%s model=%s path=%s',
                    provider,
                    model,
                    path,
                )
                raise
            return ''.join(block.text for block in response.content if block.type == 'text').strip()
        if provider == 'echo':
            return f'{path} は検索 focus に関連する notebook です。関連箇所は提示されたセル番号を確認してください。'
        raise tornado.web.HTTPError(400, reason=f'Unknown provider: {provider}')

    async def _shape_notebook_result(self, db, search_id, index, doc, query, focus, no_filter):
        reference_id = f'{search_id}/r{index + 1}/ref1'
        reference = _shape_nbsearch_notebook_reference(doc, query)
        _NBSEARCH_REFERENCE_CACHE[reference_id] = reference
        cells = await self._get_search_reference_cells(db, reference, no_filter)
        references = [{
            'type': 'getCellsFromSearch',
            'referenceId': reference_id,
        }]
        return {
            'path': doc['filename'],
            'summary': await self._summarize_result(
                focus,
                doc['filename'],
                [{
                    'referenceId': reference_id,
                    'cells': cells,
                }],
            ),
            'references': references,
        }
