const QUERY_SYNTAX = `Query syntax:
  { "match": "regex" } - regex against heading/content
  { "contains": "text" } - substring match
  { "start": N } - cell index
  { "id": "cellId" } - cell ID
  { "meme": "UUID" } - nblineage cell meme ID
  { "active": true } - currently focused cell (active notebook only)
  { "selected": true } - selected cells (active notebook only)`;

interface IActionDetail {
  description: string;
  required: string[];
  optional: string[];
  usesQuery: boolean;
}

const ACTION_DETAILS: Record<string, IActionDetail> = {
  getToc: {
    description: 'Get heading structure of current notebook',
    required: [],
    optional: [],
    usesQuery: false
  },
  getSection: {
    description: 'Get cells under matched heading',
    required: ['query'],
    optional: [],
    usesQuery: true
  },
  getCells: {
    description: 'Get cell range from matched position',
    required: ['query'],
    optional: ['count'],
    usesQuery: true
  },
  getOutput: {
    description: 'Get output of matched cell',
    required: ['query'],
    optional: [],
    usesQuery: true
  },
  listNotebookFiles: {
    description: 'List notebook files in directory',
    required: [],
    optional: ['path'],
    usesQuery: false
  },
  getTocFromFile: {
    description: 'Get heading structure from file',
    required: ['path'],
    optional: [],
    usesQuery: false
  },
  getSectionFromFile: {
    description: 'Get cells under matched heading in file',
    required: ['path', 'query'],
    optional: [],
    usesQuery: true
  },
  getCellsFromFile: {
    description: 'Get cell range from file',
    required: ['path', 'query'],
    optional: ['count'],
    usesQuery: true
  },
  getOutputFromFile: {
    description: 'Get output of matched cell in file',
    required: ['path', 'query'],
    optional: [],
    usesQuery: true
  },
  searchNotebooks: {
    description:
      'Search indexed notebooks and return focus-oriented summaries with search references',
    required: ['query', 'focus'],
    optional: ['dateFrom', 'dateTo', 'start', 'limit', 'sort'],
    usesQuery: false
  },
  getCellsFromSearch: {
    description:
      'Read cells from a search reference using the approved Privacy filter setting',
    required: ['referenceId'],
    optional: ['start', 'limit'],
    usesQuery: false
  },
  summaryCellsFromSearch: {
    description:
      'Summarize cells from a search reference and identify cell ranges to read',
    required: ['referenceId', 'focus'],
    optional: [],
    usesQuery: false
  },
  insertCell: {
    description: 'Insert new cell',
    required: ['position', 'cellType', 'source'],
    optional: [],
    usesQuery: false
  },
  updateCell: {
    description: 'Update cell content (requires _hash from prior read)',
    required: ['query', 'source', '_hash'],
    optional: [],
    usesQuery: true
  },
  deleteCell: {
    description: 'Delete cell (requires _hash from prior read)',
    required: ['query', '_hash'],
    optional: [],
    usesQuery: true
  },
  runCell: {
    description: 'Execute cell',
    required: ['query'],
    optional: [],
    usesQuery: true
  },
  listHelp: {
    description: 'Show the system prompt again',
    required: [],
    optional: [],
    usesQuery: false
  },
  help: {
    description: 'Show details for specific action',
    required: ['action'],
    optional: [],
    usesQuery: false
  }
};

export function getActionHelp(actionName: string): string {
  const detail = ACTION_DETAILS[actionName];
  if (!detail) {
    const known = Object.keys(ACTION_DETAILS).join(', ');
    return `Unknown action: "${actionName}". Available actions: ${known}`;
  }

  const lines: string[] = [
    `${actionName}: ${detail.description}`,
    '',
    `Required: ${detail.required.length > 0 ? detail.required.join(', ') : '(none)'}`,
    `Optional: ${detail.optional.length > 0 ? detail.optional.join(', ') : '(none)'}`
  ];

  if (detail.usesQuery) {
    lines.push('', QUERY_SYNTAX);
  }

  return lines.join('\n');
}

export function buildSystemPrompt(): string {
  return `You are Mynerva, a Jupyter notebook assistant.

Use the provided tools to inspect and edit the user's notebooks, and reply to
the user in natural language. Call tools when you need notebook context or to
make changes; you may call several tools.

Tool calls are shown to the user as approval requests. Before calling any tool,
always tell the user in natural language what you are about to do and why. Never
propose actions without an accompanying explanation.

You may request several actions in one turn; they run in order. Prefer stable
cell references (id, or a heading/content match) over absolute indices, since
earlier changes shift later indices. Split into separate turns only when a later
step depends on observing an earlier step's result (e.g. read a cell's output
before editing based on it).

Cell-locating queries take exactly one of:
${QUERY_SYNTAX}

updateCell and deleteCell require the cell's _hash from a prior read
(getCells/getSection). Read the cell first if you don't already have its _hash.`;
}
