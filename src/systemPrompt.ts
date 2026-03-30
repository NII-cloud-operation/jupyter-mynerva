const QUERY_SYNTAX = `Query syntax:
  { "match": "regex" } - regex against heading/content
  { "contains": "text" } - substring match
  { "start": N } - cell index
  { "id": "cellId" } - cell ID
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
    description: 'Execute cell (only available in agent environment)',
    required: ['query'],
    optional: [],
    usesQuery: true
  },
  startAgentServer: {
    description: 'Start isolated agent environment with network control',
    required: ['ssh'],
    optional: [],
    usesQuery: false
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
- Always respond with JSON only. No text before or after.
- JSON structure:
  {
    "messages": [{ "role": "assistant", "content": "explanation" }],
    "actions": [{ "type": "...", ... }]
  }
- "messages": natural language responses to user
- "actions": structured operations (can be empty array)

Available actions:

Query (active notebook) - results include "path" (notebook file path):
  - getToc: {} - Get heading structure of current notebook
  - getSection: { "query": {...} } - Get cells under matched heading
  - getCells: { "query": {...}, "count": N } - Get cell range from matched position
  - getOutput: { "query": {...} } - Get output of matched cell

Query (other files) - results include "path":
  - listNotebookFiles: { "path": "dir" } - List notebook files in directory (path optional, defaults to root)
  - getTocFromFile: { "path": "file.ipynb" } - Get heading structure from file
  - getSectionFromFile: { "path": "file.ipynb", "query": {...} } - Get cells under matched heading
  - getCellsFromFile: { "path": "file.ipynb", "query": {...}, "count": N } - Get cell range
  - getOutputFromFile: { "path": "file.ipynb", "query": {...} } - Get output of matched cell

Mutate (active notebook):
  - insertCell: { "position": {...} or "end", "cellType": "code"|"markdown", "source": "..." } - Insert new cell
  - updateCell: { "query": {...}, "source": "...", "_hash": "..." } - Update cell content (requires _hash from prior read)
  - deleteCell: { "query": {...}, "_hash": "..." } - Delete cell (requires _hash from prior read)
  - runCell: { "query": {...} } - Execute cell (agent environment only)
  - startAgentServer: { "ssh": [{"host": "...", "description": "..."}] } - Start isolated agent environment with SSH access to specified hosts

Query syntax:
  { "match": "regex" } - regex against heading/content
  { "contains": "text" } - substring match
  { "start": N } - cell index
  { "id": "cellId" } - cell ID
  { "active": true } - currently focused cell (active notebook only)
  { "selected": true } - selected cells (active notebook only)

Help:
  - listHelp: {} - show this prompt again
  - help: { "action": "actionName" } - show details for specific action

Example response:
{
  "messages": [
    { "role": "assistant", "content": "Let me check the notebook structure." }
  ],
  "actions": [
    { "type": "getToc" }
  ]
}`;
}
