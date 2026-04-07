import * as React from 'react';
import { IMutateAction, ActionStatus } from './types';
import { DropdownButton } from './DropdownButton';

interface IMutateActionCardProps {
  action: IMutateAction;
  status: ActionStatus;
  resultUrl?: string;
  onApprove: () => void;
  onApproveAlways: () => void;
  onReject: () => void;
}

function getActionLabel(action: IMutateAction): string {
  switch (action.type) {
    case 'insertCell':
      return `Insert ${action.cellType} cell`;
    case 'updateCell':
      return `Update cell: ${JSON.stringify(action.query)}`;
    case 'deleteCell':
      return `Delete cell: ${JSON.stringify(action.query)}`;
    case 'runCell':
      return `Run cell: ${JSON.stringify(action.query)}`;
    case 'startAgentServer': {
      const hosts = action.ssh
        .map(s => `${s.host} (${s.description})`)
        .join(', ');
      return `Start agent server — SSH: ${hosts}`;
    }
    default:
      return 'Unknown Action';
  }
}

function getPreviewContent(action: IMutateAction): string | null {
  switch (action.type) {
    case 'insertCell':
    case 'updateCell':
      return action.source;
    default:
      return null;
  }
}

export function MutateActionCard({
  action,
  status,
  resultUrl,
  onApprove,
  onApproveAlways,
  onReject
}: IMutateActionCardProps): React.ReactElement {
  const [showPreview, setShowPreview] = React.useState(false);
  const label = getActionLabel(action);
  const previewContent = getPreviewContent(action);
  const isCompleted = status === 'executed' || status === 'notified';

  return (
    <div
      className={`jp-Mynerva-action-card jp-Mynerva-mutate-action${isCompleted ? ' jp-Mynerva-action-completed' : ''}`}
    >
      <div className="jp-Mynerva-action-header">
        <span className="jp-Mynerva-action-icon">✏️</span>
        <span className="jp-Mynerva-action-type">{action.type}</span>
      </div>
      <div className="jp-Mynerva-action-label">{label}</div>
      {status === 'pending' && (
        <>
          {previewContent && (
            <button
              className="jp-Mynerva-action-button jp-Mynerva-preview-toggle"
              onClick={() => setShowPreview(!showPreview)}
            >
              {showPreview ? 'Hide Preview' : 'Show Preview'}
            </button>
          )}
          {showPreview && previewContent && (
            <pre className="jp-Mynerva-action-preview">{previewContent}</pre>
          )}
          <div className="jp-Mynerva-action-buttons">
            <DropdownButton
              options={[
                { label: 'Accept', onClick: onApprove },
                { label: 'Accept & Always', onClick: onApproveAlways }
              ]}
            />
            <button
              className="jp-Mynerva-action-button jp-Mynerva-cancel-button"
              onClick={onReject}
            >
              Reject
            </button>
          </div>
        </>
      )}
      {status === 'approved' && (
        <div className="jp-Mynerva-action-badge jp-Mynerva-approved-badge">
          Approved
        </div>
      )}
      {status === 'executed' && (
        <>
          <div className="jp-Mynerva-action-badge jp-Mynerva-applied-badge">
            Applied
          </div>
          {resultUrl && (
            <a
              className="jp-Mynerva-action-link"
              href={resultUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open Agent Environment
            </a>
          )}
        </>
      )}
      {(status === 'rejected' || status === 'notified') && (
        <div className="jp-Mynerva-action-badge jp-Mynerva-cancelled-badge">
          Rejected
        </div>
      )}
    </div>
  );
}
