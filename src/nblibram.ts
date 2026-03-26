import { ServerConnection } from '@jupyterlab/services';
import { URLExt } from '@jupyterlab/coreutils';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ICellQuery } from './context';

interface INblibramRequest {
  command: string;
  args: string[];
  notebookPath?: string;
  notebookContent?: unknown;
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

function queryToArgs(query: ICellQuery): string[] {
  if ('start' in query) {
    return ['-query', `start:${query.start}`];
  }
  if ('id' in query) {
    return ['-query', `id:${query.id}`];
  }
  if ('contains' in query) {
    return ['-query', `contains:${query.contains}`];
  }
  if ('match' in query) {
    return ['-query', `match:${query.match}`];
  }
  throw new Error(
    `Unsupported query for nblibram: ${JSON.stringify(query)}`
  );
}

// --- File-based queries (no live doc needed) ---

export async function nblibramTocFromFile(path: string): Promise<unknown> {
  return callNblibram({
    command: 'toc',
    args: ['-file', path, '-format', 'json']
  });
}

export async function nblibramSectionFromFile(
  path: string,
  query: ICellQuery
): Promise<unknown> {
  return callNblibram({
    command: 'section',
    args: ['-file', path, '-format', 'json', ...queryToArgs(query)]
  });
}

export async function nblibramCellsFromFile(
  path: string,
  query: ICellQuery,
  count?: number
): Promise<unknown> {
  const args = ['-file', path, '-format', 'json', ...queryToArgs(query)];
  if (count) {
    args.push('-count', String(count));
  }
  return callNblibram({ command: 'cells', args });
}

export async function nblibramOutputsFromFile(
  path: string,
  query: ICellQuery
): Promise<unknown> {
  return callNblibram({
    command: 'outputs',
    args: ['-file', path, '-format', 'json', ...queryToArgs(query)]
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
    args: string[]
  ): INblibramRequest {
    if (!this.filterEnabled) {
      args = ['-no-filter', ...args];
    }
    return {
      command,
      args,
      notebookPath: this.getNotebookPath(),
      notebookContent: this.getNotebookContent()
    };
  }

  async getToc(): Promise<unknown> {
    return callNblibram(
      this.buildRequest('toc', ['-format', 'json'])
    );
  }

  async getSection(query: ICellQuery): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    return callNblibram(
      this.buildRequest('section', ['-format', 'json', ...queryToArgs(resolved)])
    );
  }

  async getCells(query: ICellQuery, count?: number): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    const args = ['-format', 'json', ...queryToArgs(resolved)];
    if (count) {
      args.push('-count', String(count));
    }
    return callNblibram(this.buildRequest('cells', args));
  }

  async getOutput(query: ICellQuery): Promise<unknown> {
    const resolved = this.resolveQuery(query);
    return callNblibram(
      this.buildRequest('outputs', ['-format', 'json', ...queryToArgs(resolved)])
    );
  }
}
