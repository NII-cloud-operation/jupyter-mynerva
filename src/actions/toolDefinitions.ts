/**
 * Single source of truth for the notebook tools exposed to the LLM.
 *
 * The frontend is the tool host (it executes every action against the live
 * notebook), so it declares the available tools. These provider-neutral
 * definitions are sent to the backend with each chat request; the backend
 * wraps them into each provider's native `tools` shape.
 */

/** JSON Schema (Draft-07 subset understood by all providers). */
export type IJSONSchema = Record<string, unknown>;

export interface IToolDefinition {
  name: string;
  description: string;
  parameters: IJSONSchema;
}

const CELL_QUERY_SCHEMA: IJSONSchema = {
  description:
    'How to locate a cell. Provide exactly one of the matching keys.',
  anyOf: [
    {
      type: 'object',
      properties: {
        match: {
          type: 'string',
          description: 'Regex tested against cell/heading content'
        }
      },
      required: ['match'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: {
        contains: { type: 'string', description: 'Substring match' }
      },
      required: ['contains'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: { start: { type: 'integer', description: 'Cell index' } },
      required: ['start'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: { id: { type: 'string', description: 'Cell ID' } },
      required: ['id'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: {
        meme: { type: 'string', description: 'nblineage cell meme ID (UUID)' }
      },
      required: ['meme'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: {
        active: {
          const: true,
          description: 'Currently focused cell (active notebook only)'
        }
      },
      required: ['active'],
      additionalProperties: false
    },
    {
      type: 'object',
      properties: {
        selected: {
          const: true,
          description: 'Selected cells (active notebook only)'
        }
      },
      required: ['selected'],
      additionalProperties: false
    }
  ]
};

function obj(
  properties: Record<string, IJSONSchema>,
  required: string[]
): IJSONSchema {
  return { type: 'object', properties, required, additionalProperties: false };
}

const TOOL_DEFINITIONS: IToolDefinition[] = [
  // Query (active notebook) — results include the notebook "path"
  {
    name: 'getToc',
    description: 'Get the heading structure of the current notebook.',
    parameters: obj({}, [])
  },
  {
    name: 'getSection',
    description:
      'Get the cells under a matched heading in the current notebook.',
    parameters: obj({ query: CELL_QUERY_SCHEMA }, ['query'])
  },
  {
    name: 'getCells',
    description:
      'Get a range of cells from a matched position in the current notebook.',
    parameters: obj(
      {
        query: CELL_QUERY_SCHEMA,
        count: { type: 'integer', description: 'Number of cells to return' }
      },
      ['query']
    )
  },
  {
    name: 'getOutput',
    description: 'Get the output of a matched cell in the current notebook.',
    parameters: obj({ query: CELL_QUERY_SCHEMA }, ['query'])
  },
  // Query (other files)
  {
    name: 'listNotebookFiles',
    description: 'List notebook files in a directory (defaults to root).',
    parameters: obj(
      { path: { type: 'string', description: 'Directory path' } },
      []
    )
  },
  {
    name: 'getTocFromFile',
    description: 'Get the heading structure of a notebook file.',
    parameters: obj({ path: { type: 'string' } }, ['path'])
  },
  {
    name: 'getSectionFromFile',
    description: 'Get the cells under a matched heading in a notebook file.',
    parameters: obj({ path: { type: 'string' }, query: CELL_QUERY_SCHEMA }, [
      'path',
      'query'
    ])
  },
  {
    name: 'getCellsFromFile',
    description: 'Get a range of cells from a notebook file.',
    parameters: obj(
      {
        path: { type: 'string' },
        query: CELL_QUERY_SCHEMA,
        count: { type: 'integer' }
      },
      ['path', 'query']
    )
  },
  {
    name: 'getOutputFromFile',
    description: 'Get the output of a matched cell in a notebook file.',
    parameters: obj({ path: { type: 'string' }, query: CELL_QUERY_SCHEMA }, [
      'path',
      'query'
    ])
  },
  // Mutate (active notebook)
  {
    name: 'insertCell',
    description: 'Insert a new cell into the current notebook.',
    parameters: obj(
      {
        position: {
          description:
            'Where to insert: a cell query (inserts after the match) or "end".',
          anyOf: [CELL_QUERY_SCHEMA, { type: 'string', enum: ['end'] }]
        },
        cellType: { type: 'string', enum: ['code', 'markdown'] },
        source: { type: 'string' }
      },
      ['position', 'cellType', 'source']
    )
  },
  {
    name: 'updateCell',
    description:
      "Update a cell's content. Requires _hash from a prior read of the cell.",
    parameters: obj(
      {
        query: CELL_QUERY_SCHEMA,
        source: { type: 'string' },
        _hash: {
          type: 'string',
          description: 'Hash from a prior read, for optimistic locking'
        }
      },
      ['query', 'source', '_hash']
    )
  },
  {
    name: 'deleteCell',
    description: 'Delete a cell. Requires _hash from a prior read of the cell.',
    parameters: obj(
      {
        query: CELL_QUERY_SCHEMA,
        _hash: {
          type: 'string',
          description: 'Hash from a prior read, for optimistic locking'
        }
      },
      ['query', '_hash']
    )
  },
  {
    name: 'runCell',
    description: 'Execute a cell in the current notebook.',
    parameters: obj({ query: CELL_QUERY_SCHEMA }, ['query'])
  },
  // Help
  {
    name: 'listHelp',
    description: 'List the available notebook tools.',
    parameters: obj({}, [])
  },
  {
    name: 'help',
    description: 'Show usage details for a specific tool.',
    parameters: obj({ action: { type: 'string', description: 'Tool name' } }, [
      'action'
    ])
  }
];

// nbsearch tools — only offered when the server reports nbsearch is configured.
// Guidance that used to live in the system prompt lives here, in the tool and
// parameter descriptions, so the always-on prompt stays lean.
const NBSEARCH_SORT_VALUES = [
  'mtime desc',
  'mtime asc',
  'ctime desc',
  'ctime asc',
  'atime desc',
  'atime asc',
  'lc_cell_meme__execution_end_time desc',
  'lc_cell_meme__execution_end_time asc'
];

const NBSEARCH_TOOL_DEFINITIONS: IToolDefinition[] = [
  {
    name: 'searchNotebooks',
    description:
      'Search indexed notebooks across the server (nbsearch / Solr) and ' +
      'return per-notebook summaries plus a referenceId for reading cells. ' +
      'Prefer this over listNotebookFiles whenever the user wants to search, ' +
      'find, discover, or look across notebooks — listNotebookFiles only ' +
      'lists paths in one directory and does not search content or the index. ' +
      'Returns filename, owner, server, mtime/ctime/atime, summaries, and a ' +
      'referenceId per result; it does NOT return raw cells (use ' +
      'summaryCellsFromSearch / getCellsFromSearch with the referenceId). ' +
      'Cells are read through the user-approved Privacy filter before being ' +
      'summarized. If numFound exceeds start + returned, tell the user more ' +
      'results are available and re-query with a larger start.',
    parameters: obj(
      {
        query: {
          type: 'string',
          description:
            'Solr/Lucene query sent as q to the jupyter-notebook core. Use ' +
            'fielded queries when the user names a field: filename:*BinderHub* ' +
            '(file name contains), owner:alice, ' +
            'source__markdown__heading_1:検索 (top-level title/heading); plain ' +
            'terms like BinderHub for normal content search.'
        },
        focus: {
          type: 'string',
          description:
            "The user's purpose, used to judge relevance when summarizing. " +
            'Required.'
        },
        dateFrom: {
          type: 'string',
          description:
            'Local calendar date (YYYY-MM-DD) lower bound; converted to a UTC ' +
            'datetime by the client.'
        },
        dateTo: {
          type: 'string',
          description: 'Local calendar date (YYYY-MM-DD) upper bound.'
        },
        start: {
          type: 'integer',
          description: 'Result offset for pagination.'
        },
        limit: {
          type: 'integer',
          description: 'Max results to return (default 10).'
        },
        sort: {
          type: 'string',
          enum: NBSEARCH_SORT_VALUES,
          description:
            'Prefer "mtime desc" for recently updated notebooks and ' +
            '"lc_cell_meme__execution_end_time desc" for recently executed ' +
            'notebooks.'
        }
      },
      ['query', 'focus']
    )
  },
  {
    name: 'summaryCellsFromSearch',
    description:
      'Summarize the cells of one search result (by referenceId) and identify ' +
      'the relevant cell indexes/ranges. Call this before getCellsFromSearch ' +
      'when a notebook may be large or you need to locate the relevant part. ' +
      'Only call after a searchNotebooks result has returned a concrete ' +
      'referenceId — never invent a referenceId. Returns cellCount, coverage, ' +
      'and a summary with cell numbers to choose start/limit for ' +
      'getCellsFromSearch.',
    parameters: obj(
      {
        referenceId: {
          type: 'string',
          description: 'referenceId from a prior searchNotebooks result.'
        },
        focus: {
          type: 'string',
          description: "The user's purpose, used to judge relevance. Required."
        }
      },
      ['referenceId', 'focus']
    )
  },
  {
    name: 'getCellsFromSearch',
    description:
      'Read raw cells from one search result (by referenceId), respecting the ' +
      'user-approved Privacy filter. Only call after searchNotebooks returned ' +
      'the referenceId — never invent one. Do not read many cells just to ' +
      'locate content: use summaryCellsFromSearch first, then read only the ' +
      'relevant range. If the result has hasMore=true, call again with the ' +
      'same referenceId and start=nextStart to continue; do not give the final ' +
      'answer until you have read the range you need.',
    parameters: obj(
      {
        referenceId: {
          type: 'string',
          description: 'referenceId from a prior searchNotebooks result.'
        },
        start: {
          type: 'integer',
          description:
            'Cell offset within the reference (use nextStart to continue).'
        },
        limit: { type: 'integer', description: 'Max cells to read.' }
      },
      ['referenceId']
    )
  }
];

export interface IToolOptions {
  nbsearchAvailable?: boolean;
}

export function getToolDefinitions(
  options: IToolOptions = {}
): IToolDefinition[] {
  if (options.nbsearchAvailable) {
    return [...TOOL_DEFINITIONS, ...NBSEARCH_TOOL_DEFINITIONS];
  }
  return TOOL_DEFINITIONS;
}
