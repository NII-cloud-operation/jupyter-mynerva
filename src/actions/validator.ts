/**
 * Action schemas for validation
 */
const ACTION_SCHEMAS: Record<
  string,
  { required: string[]; optional: string[] }
> = {
  // Query actions
  getToc: {
    required: [],
    optional: []
  },
  getSection: {
    required: ['query'],
    optional: []
  },
  getCells: {
    required: ['query'],
    optional: ['count']
  },
  getOutput: {
    required: ['query'],
    optional: []
  },
  // File query actions
  listNotebookFiles: {
    required: [],
    optional: ['path']
  },
  getTocFromFile: {
    required: ['path'],
    optional: []
  },
  getSectionFromFile: {
    required: ['path', 'query'],
    optional: []
  },
  getCellsFromFile: {
    required: ['path', 'query'],
    optional: ['count']
  },
  getOutputFromFile: {
    required: ['path', 'query'],
    optional: []
  },
  // Mutate actions
  insertCell: {
    required: ['position', 'cellType', 'source'],
    optional: []
  },
  updateCell: {
    required: ['query', 'source', '_hash'],
    optional: []
  },
  deleteCell: {
    required: ['query', '_hash'],
    optional: []
  },
  runCell: {
    required: ['query'],
    optional: []
  },
  startAgentServer: {
    required: ['ssh'],
    optional: []
  },
  // Help actions
  listHelp: {
    required: [],
    optional: []
  },
  help: {
    required: ['action'],
    optional: []
  }
};

function findSimilarField(
  unknownField: string,
  knownFields: string[]
): string | null {
  const lower = unknownField.toLowerCase();
  for (const field of knownFields) {
    if (field.toLowerCase() === lower) {
      return field;
    }
    if (
      lower.includes(field.toLowerCase()) ||
      field.toLowerCase().includes(lower)
    ) {
      return field;
    }
  }
  return null;
}

function validateAction(action: unknown, index: number): string[] {
  const errors: string[] = [];

  if (!action || typeof action !== 'object') {
    errors.push(`Action ${index}: Invalid action format (expected object)`);
    return errors;
  }

  const actionObj = action as Record<string, unknown>;
  const { type } = actionObj;

  if (!type || typeof type !== 'string') {
    errors.push(`Action ${index}: Missing required field "type"`);
    return errors;
  }

  const schema = ACTION_SCHEMAS[type];
  if (!schema) {
    const knownTypes = Object.keys(ACTION_SCHEMAS).join(', ');
    errors.push(
      `Action ${index}: Unknown action type "${type}". Valid types: ${knownTypes}`
    );
    return errors;
  }

  const allKnownFields = ['type', ...schema.required, ...schema.optional];

  for (const field of schema.required) {
    if (actionObj[field] === undefined || actionObj[field] === null) {
      errors.push(
        `Action ${index} (${type}): Missing required field "${field}"`
      );
    }
  }

  const actionKeys = Object.keys(actionObj);
  for (const key of actionKeys) {
    if (!allKnownFields.includes(key)) {
      const similar = findSimilarField(key, allKnownFields);
      if (similar) {
        errors.push(
          `Action ${index} (${type}): Unknown field "${key}" (did you mean "${similar}"?)`
        );
      } else {
        errors.push(
          `Action ${index} (${type}): Unknown field "${key}". Valid fields: ${allKnownFields.join(', ')}`
        );
      }
    }
  }

  return errors;
}

export interface IValidationResult {
  valid: boolean;
  errors: string[];
  feedbackMessage: string | null;
}

export function validateActions(actions: unknown): IValidationResult {
  if (!Array.isArray(actions)) {
    return {
      valid: false,
      errors: ['Actions must be an array'],
      feedbackMessage:
        '[Action Validation Error]\n\nActions must be an array.\n\nPlease correct the format and resend.'
    };
  }

  const allErrors: string[] = [];
  actions.forEach((action, index) => {
    const errors = validateAction(action, index);
    allErrors.push(...errors);
  });

  if (allErrors.length === 0) {
    return { valid: true, errors: [], feedbackMessage: null };
  }

  const feedbackMessage = [
    '[Action Validation Error]',
    '',
    ...allErrors,
    '',
    'Please correct the action format and resend.'
  ].join('\n');

  return { valid: false, errors: allErrors, feedbackMessage };
}
