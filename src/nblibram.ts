import { ServerConnection } from '@jupyterlab/services';
import { URLExt } from '@jupyterlab/coreutils';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ICellQuery } from './context';

interface INblibramRequest {
  command: string;
  path: string;
  notebookContent?: unknown;
  format?: string;
  query?: string;
  count?: number;
  noFilter?: boolean;
  excludeOutputs?: boolean;
}

async function callNblibram(req: INblibramRequest): Promise<unknown> {
  const settings = ServerConnection.makeSettings();
  const url = URLExt.join(settings.baseUrl, 'jupyter-mynerva', 'nblibram');

  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify(req)
    },
    settings
  );

  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.error || `nblibram ${req.command} failed`);
  }

  return response.json();
}

function queryToString(query: ICellQuery): string {
  if ('start' in query) {
    return `start:${query.start}`;
  }
  if ('id' in query) {
    return `id:${query.id}`;
  }
  if ('contains' in query) {
    return `contains:${query.contains}`;
  }
  if ('match' in query) {
    return `match:${query.match}`;
  }
  throw new Error(`Unsupported query for nblibram: ${JSON.stringify(query)}`);
}

// --- File-based queries (no live doc needed) ---

export async function nblibramTocFromFile(path: string): Promise<unknown> {
  return callNblibram({ command: 'toc', path, format: 'json' });
}

export async function nblibramSectionFromFile(
  path: string,
  query: ICellQuery
): Promise<unknown> {
  return callNblibram({
    command: 'section',
    path,
    format: 'json',
    query: queryToString(query)
  });
}

export async function nblibramCellsFromFile(
  path: string,
  query: ICellQuery,
  count?: number
): Promise<unknown> {
  return callNblibram({
    command: 'cells',
    path,
    format: 'json',
    query: queryToString(query),
    count
  });
}

export async function nblibramOutputsFromFile(
  path: string,
  query: ICellQuery
): Promise<unknown> {
  return callNblibram({
    command: 'outputs',
    path,
    format: 'json',
    query: queryToString(query)
  });
}

// --- Live notebook queries (with dirty sync) ---

export class NblibramLiveQuery {
  private dirty = true;
  private disposeSignal: (() => void) | null = null;
  filterEnabled = true;

  constructor(private notebookTracker: INotebookTracker) {
    this.watchChanges();
  }

  private watchChanges(): void {
    this.notebookTracker.currentChanged.connect(() => {
      this.dirty = true;
      this.rebindContentChanged();
    });
    this.rebindContentChanged();
  }

  private rebindContentChanged(): void {
    if (this.disposeSignal) {
      this.disposeSignal();
      this.disposeSignal = null;
    }
    const model = this.notebookTracker.currentWidget?.model;
    if (!model) {
      return;
    }
    const handler = () => {
      this.dirty = true;
    };
    model.contentChanged.connect(handler);
    this.disposeSignal = () => {
      model.contentChanged.disconnect(handler);
    };
  }

  markDirty(): void {
    this.dirty = true;
  }

  private getNotebookPath(): string {
    const path = this.notebookTracker.currentWidget?.context?.path;
    if (!path) {
      throw new Error('No notebook is open');
    }
    return path;
  }

  private getNotebookContent(): unknown | undefined {
    if (!this.dirty) {
      return undefined;
    }
    const model = this.notebookTracker.currentWidget?.model;
    if (!model) {
      throw new Error('No notebook is open');
    }
    this.dirty = false;
    return model.toJSON();
  }

  private resolveQuery(query: ICellQuery): ICellQuery {
    if ('active' in query) {
      const index = this.notebookTracker.currentWidget?.content.activeCellIndex;
      if (index === undefined || index < 0) {
        throw new Error('No active cell');
      }
      return { start: index };
    }
    if ('selected' in query) {
      const notebook = this.notebookTracker.currentWidget?.content;
      if (!notebook) {
        throw new Error('No notebook is open');
      }
      for (const cell of notebook.selectedCells) {
        const idx = notebook.widgets.indexOf(cell);
        if (idx >= 0) {
          return { start: idx };
        }
      }
      throw new Error('No selected cell');
    }
    return query;
  }

  private buildRequest(
    command: string,
    query?: string,
    opts?: { count?: number }
  ): INblibramRequest {
    const req: INblibramRequest = {
      command,
      path: this.getNotebookPath(),
      format: 'json',
      noFilter: !this.filterEnabled || undefined
    };
    const content = this.getNotebookContent();
    if (content !== undefined) {
      req.notebookContent = content;
    }
    if (query) {
      req.query = query;
    }
    if (opts?.count) {
      req.count = opts.count;
    }
    return req;
  }

  async getToc(): Promise<unknown> {
    return callNblibram(this.buildRequest('toc'));
  }

  async getSection(query: ICellQuery): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    return callNblibram(this.buildRequest('section', queryToString(resolved)));
  }

  async getCells(query: ICellQuery, count?: number): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    return callNblibram(
      this.buildRequest('cells', queryToString(resolved), { count })
    );
  }

  async getOutput(query: ICellQuery): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    return callNblibram(this.buildRequest('outputs', queryToString(resolved)));
  }
}
