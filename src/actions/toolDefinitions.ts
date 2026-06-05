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

export function getToolDefinitions(): IToolDefinition[] {
  return TOOL_DEFINITIONS;
}
