import { INotebookTracker } from '@jupyterlab/notebook';
import { ICellModel } from '@jupyterlab/cells';

/**
 * Query types for matching cells
 */
export interface IMatchQuery {
  match: string;
}

export interface IContainsQuery {
  contains: string;
}

export interface IStartQuery {
  start: number;
}

export interface IIdQuery {
  id: string;
}

export interface IActiveQuery {
  active: true;
}

export interface ISelectedQuery {
  selected: true;
}

export type ICellQuery =
  | IMatchQuery
  | IContainsQuery
  | IStartQuery
  | IIdQuery
  | IActiveQuery
  | ISelectedQuery;

/**
 * Table of contents entry
 */
export interface ITocEntry {
  level: number;
  text: string;
  cellIndex: number;
  cellId: string;
}

/**
 * Cell data for LLM (outputs excluded - use getOutput for outputs)
 */
export interface ICellData {
  index: number;
  id: string;
  type: 'code' | 'markdown' | 'raw';
  source: string;
  isActive: boolean;
  isSelected: boolean;
  _hash: string;
}

/**
 * Output data for LLM
 */
export interface IOutputData {
  outputType: string;
  text?: string;
  data?: Record<string, unknown>;
}

/**
 * Compute hash for cell content (djb2 algorithm)
 */
export function computeCellHash(type: string, source: string): string {
  const str = type + '\0' + source;
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  return (hash >>> 0).toString(16);
}

/**
 * Context Engine - handles live notebook mutations and UI state.
 * Read queries (getToc, getSection, getCells, getOutput) are handled by nblibram.
 */
export class ContextEngine {
  constructor(private notebookTracker: INotebookTracker) {}

  private getNotebookWidget() {
    const panel = this.notebookTracker.currentWidget;
    if (!panel) {
      throw new Error('No notebook is open');
    }
    return panel.content;
  }

  private getNotebookModel() {
    const notebook = this.getNotebookWidget();
    const model = notebook.model;
    if (!model) {
      throw new Error('Notebook model is not available');
    }
    return model;
  }

  private getActiveCellIndex(): number {
    return this.getNotebookWidget().activeCellIndex;
  }

  private getSelectedCellIndices(): Set<number> {
    const notebook = this.getNotebookWidget();
    const indices = new Set<number>();
    for (const cell of notebook.selectedCells) {
      const index = notebook.widgets.indexOf(cell);
      if (index >= 0) {
        indices.add(index);
      }
    }
    return indices;
  }

  private findCellIndex(query: ICellQuery): number {
    const model = this.getNotebookModel();
    const activeCellIndex = this.getActiveCellIndex();
    const selectedIndices = this.getSelectedCellIndices();

    if ('start' in query) {
      return query.start;
    }
    if ('active' in query) {
      if (activeCellIndex < 0) {
        throw new Error('No active cell');
      }
      return activeCellIndex;
    }
    if ('selected' in query) {
      for (const idx of selectedIndices) {
        return idx;
      }
      throw new Error('No selected cell');
    }
    if ('id' in query) {
      for (let i = 0; i < model.cells.length; i++) {
        if (model.cells.get(i).id === query.id) {
          return i;
        }
      }
      throw new Error(`No cell with id: ${query.id}`);
    }
    if ('contains' in query) {
      for (let i = 0; i < model.cells.length; i++) {
        if (model.cells.get(i).sharedModel.source.includes(query.contains)) {
          return i;
        }
      }
      throw new Error(`No cell contains: ${query.contains}`);
    }
    if ('match' in query) {
      const re = new RegExp(query.match);
      for (let i = 0; i < model.cells.length; i++) {
        if (re.test(model.cells.get(i).sharedModel.source)) {
          return i;
        }
      }
      throw new Error(`No cell matches: ${query.match}`);
    }
    throw new Error(`Invalid query: ${JSON.stringify(query)}`);
  }

  private cellToData(cell: ICellModel, index: number): ICellData {
    const type = cell.type;
    if (type !== 'code' && type !== 'markdown' && type !== 'raw') {
      throw new Error(`Unknown cell type: ${type}`);
    }
    const source = cell.sharedModel.source;
    const activeCellIndex = this.getActiveCellIndex();
    const selectedIndices = this.getSelectedCellIndices();
    return {
      index,
      id: cell.id,
      type,
      source,
      isActive: index === activeCellIndex,
      isSelected: selectedIndices.has(index),
      _hash: computeCellHash(type, source)
    };
  }

  hasActiveNotebook(): boolean {
    return this.notebookTracker.currentWidget !== null;
  }

  getNotebookPath(): string {
    const notebook = this.notebookTracker.currentWidget;
    if (!notebook) {
      throw new Error('No notebook is open');
    }
    const path = notebook.context?.path;
    if (!path) {
      throw new Error('Notebook path is not available');
    }
    return path;
  }

  insertCell(
    position: ICellQuery | 'end',
    cellType: 'code' | 'markdown',
    source: string
  ): ICellData {
    const panel = this.notebookTracker.currentWidget;
    if (!panel) {
      throw new Error('No notebook is open');
    }
    const notebook = panel.content;
    const model = this.getNotebookModel();

    let insertIndex: number;
    if (position === 'end') {
      insertIndex = model.cells.length;
    } else {
      insertIndex = this.findCellIndex(position) + 1;
    }

    model.sharedModel.insertCell(insertIndex, {
      cell_type: cellType,
      source
    });
    const cell = model.cells.get(insertIndex);

    notebook.activeCellIndex = insertIndex;
    notebook.scrollToItem(insertIndex);

    return this.cellToData(cell, insertIndex);
  }

  updateCell(query: ICellQuery, source: string, _hash: string): ICellData {
    const panel = this.notebookTracker.currentWidget;
    if (!panel) {
      throw new Error('No notebook is open');
    }
    const notebook = panel.content;
    const model = this.getNotebookModel();
    const index = this.findCellIndex(query);
    const cell = model.cells.get(index);

    const currentHash = computeCellHash(cell.type, cell.sharedModel.source);
    if (currentHash !== _hash) {
      throw new Error(
        `Hash mismatch: cell has been modified (expected ${_hash}, got ${currentHash})`
      );
    }

    cell.sharedModel.source = source;

    notebook.activeCellIndex = index;
    notebook.scrollToItem(index);

    return this.cellToData(cell, index);
  }

  deleteCell(query: ICellQuery, _hash: string): { index: number; id: string } {
    const panel = this.notebookTracker.currentWidget;
    if (!panel) {
      throw new Error('No notebook is open');
    }
    const notebook = panel.content;
    const model = this.getNotebookModel();
    const index = this.findCellIndex(query);
    const cell = model.cells.get(index);

    const currentHash = computeCellHash(cell.type, cell.sharedModel.source);
    if (currentHash !== _hash) {
      throw new Error(
        `Hash mismatch: cell has been modified (expected ${_hash}, got ${currentHash})`
      );
    }

    const id = cell.id;
    model.sharedModel.deleteCell(index);

    const newCellCount = model.cells.length;
    if (newCellCount > 0) {
      const newIndex = Math.min(index, newCellCount - 1);
      notebook.activeCellIndex = newIndex;
      notebook.scrollToItem(newIndex);
    }

    return { index, id };
  }

  async runCell(query: ICellQuery): Promise<{ index: number; id: string }> {
    const panel = this.notebookTracker.currentWidget;
    if (!panel) {
      throw new Error('No notebook is open');
    }

    const notebook = panel.content;
    const model = this.getNotebookModel();
    const index = this.findCellIndex(query);
    const cell = model.cells.get(index);

    notebook.activeCellIndex = index;

    const { NotebookActions } = await import('@jupyterlab/notebook');
    await NotebookActions.run(notebook, panel.sessionContext);

    return { index, id: cell.id };
  }
}
