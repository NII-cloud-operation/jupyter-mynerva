import { ICellQuery } from '../context';

/**
 * LLM response structure
 */
export interface ILLMResponse {
  messages: Array<{ role: 'assistant'; content: string }>;
  actions: IAction[];
}

/**
 * Query action types (read-only, displayed on user side)
 */
export type IQueryAction =
  | IGetTocAction
  | IGetSectionAction
  | IGetCellsAction
  | IGetOutputAction
  | IListNotebookFilesAction
  | IGetTocFromFileAction
  | IGetSectionFromFileAction
  | IGetCellsFromFileAction
  | IGetOutputFromFileAction
  | IListHelpAction
  | IHelpDetailAction;

/**
 * Mutate action types (modify notebook, displayed on assistant side)
 */
export type IMutateAction =
  | IInsertCellAction
  | IUpdateCellAction
  | IDeleteCellAction
  | IRunCellAction;

/**
 * All action types
 */
export type IAction = IQueryAction | IMutateAction;

export interface IGetTocAction {
  type: 'getToc';
}

export interface IGetSectionAction {
  type: 'getSection';
  query: ICellQuery;
}

export interface IGetCellsAction {
  type: 'getCells';
  query: ICellQuery;
  count?: number;
}

export interface IGetOutputAction {
  type: 'getOutput';
  query: ICellQuery;
}

export interface IListNotebookFilesAction {
  type: 'listNotebookFiles';
  path?: string;
}

export interface IGetTocFromFileAction {
  type: 'getTocFromFile';
  path: string;
}

export interface IGetSectionFromFileAction {
  type: 'getSectionFromFile';
  path: string;
  query: ICellQuery;
}

export interface IGetCellsFromFileAction {
  type: 'getCellsFromFile';
  path: string;
  query: ICellQuery;
  count?: number;
}

export interface IGetOutputFromFileAction {
  type: 'getOutputFromFile';
  path: string;
  query: ICellQuery;
}

export interface IListHelpAction {
  type: 'listHelp';
}

export interface IHelpDetailAction {
  type: 'help';
  action: string;
}

/**
 * Mutate action interfaces
 */
export interface IInsertCellAction {
  type: 'insertCell';
  position: ICellQuery | 'end';
  cellType: 'code' | 'markdown';
  source: string;
}

export interface IUpdateCellAction {
  type: 'updateCell';
  query: ICellQuery;
  source: string;
  _hash: string;
}

export interface IDeleteCellAction {
  type: 'deleteCell';
  query: ICellQuery;
  _hash: string;
}

export interface IRunCellAction {
  type: 'runCell';
  query: ICellQuery;
}

/**
 * Action status for UI
 * pending → approved → executed
 *        ↘ rejected → notified
 */
export type ActionStatus =
  | 'pending'
  | 'approved'
  | 'executed'
  | 'rejected'
  | 'notified';

/**
 * Action with status for tracking
 */
export interface IActionWithStatus {
  action: IAction;
  status: ActionStatus;
}
