import * as React from 'react';
import { IQueryAction, ActionStatus } from './types';
import { DropdownButton } from './DropdownButton';

interface IQueryActionCardProps {
  action: IQueryAction;
  status: ActionStatus;
  onApprove: () => void;
  onApproveAlways: () => void;
  onReject: () => void;
}

function getActionLabel(action: IQueryAction): string {
  switch (action.type) {
    case 'getToc':
      return 'Get Table of Contents';
    case 'getSection':
      return `Get Section: ${JSON.stringify(action.query)}`;
    case 'getCells':
      return `Get Cells: ${JSON.stringify(action.query)}${action.count ? ` (count: ${action.count})` : ''}`;
    case 'getOutput':
      return `Get Output: ${JSON.stringify(action.query)}`;
    case 'listNotebookFiles':
      return `List Notebooks: ${action.path || '(root)'}`;
    case 'getTocFromFile':
      return `Get TOC from: ${action.path}`;
    case 'getSectionFromFile':
      return `Get Section from ${action.path}: ${JSON.stringify(action.query)}`;
    case 'getCellsFromFile':
      return `Get Cells from ${action.path}: ${JSON.stringify(action.query)}${action.count ? ` (count: ${action.count})` : ''}`;
    case 'getOutputFromFile':
      return `Get Output from ${action.path}: ${JSON.stringify(action.query)}`;
    case 'listHelp':
      return 'Show available actions';
    case 'help':
      return `Help: ${action.action}`;
    default:
      return 'Unknown Action';
  }
}

export function QueryActionCard({
  action,
  status,
  onApprove,
  onApproveAlways,
  onReject
}: IQueryActionCardProps): React.ReactElement {
  const label = getActionLabel(action);

  const isCompleted = status === 'executed' || status === 'notified';

  return (
    <div
      className={`jp-Mynerva-action-card jp-Mynerva-query-action${isCompleted ? ' jp-Mynerva-action-completed' : ''}`}
    >
      <div className="jp-Mynerva-action-header">
        <span className="jp-Mynerva-action-icon">📋</span>
        <span className="jp-Mynerva-action-type">{action.type}</span>
      </div>
      <div className="jp-Mynerva-action-label">{label}</div>
      {status === 'pending' && (
        <div className="jp-Mynerva-action-buttons">
          <DropdownButton
            options={[
              { label: 'Share', onClick: onApprove },
              { label: 'Share & Always', onClick: onApproveAlways }
            ]}
          />
          <button
            className="jp-Mynerva-action-button jp-Mynerva-dismiss-button"
            onClick={onReject}
          >
            Dismiss
          </button>
        </div>
      )}
      {status === 'approved' && (
        <div className="jp-Mynerva-action-badge jp-Mynerva-approved-badge">
          Approved
        </div>
      )}
      {status === 'executed' && (
        <div className="jp-Mynerva-action-badge jp-Mynerva-shared-badge">
          Shared
        </div>
      )}
      {(status === 'rejected' || status === 'notified') && (
        <div className="jp-Mynerva-action-badge jp-Mynerva-dismissed-badge">
          Dismissed
        </div>
      )}
    </div>
  );
}
