import { ILabShell } from '@jupyterlab/application';
import { ServerConnection } from '@jupyterlab/services';
import { ReactWidget, settingsIcon, copyIcon } from '@jupyterlab/ui-components';
import * as React from 'react';
import { marked } from 'marked';

import { ContextEngine } from './context';
import {
  IAction,
  IQueryAction,
  IMutateAction,
  IListNotebookFilesAction,
  ActionStatus,
  parseRawContent,
  validateActions,
  QueryActionCard,
  MutateActionCard,
  DropdownButton
} from './actions';
import { buildSystemPrompt, getActionHelp } from './systemPrompt';
import {
  NblibramLiveQuery,
  nblibramTocFromFile,
  nblibramSectionFromFile,
  nblibramCellsFromFile,
  nblibramOutputsFromFile
} from './nblibram';
import { mynervaIcon } from './icons';

const PANEL_CLASS = 'jp-Mynerva-panel';

interface IMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  actions?: IAction[];
  generated?: boolean; // Auto-generated messages (show brief in UI)
}

interface IConfig {
  provider: string;
  model: string;
  decryptError?: string;
  apiKey: string;
  useDefault?: boolean;
  enkiGateUrl?: string;
  enkiGateToken?: string;
  enkiGateModel?: string;
  enkiGateExpiresAt?: number;
}

interface IDefaultConfig {
  provider: string;
  model: string;
}

interface IProvider {
  id: string;
  displayName: string;
  models: string[];
}

interface IProvidersResponse {
  providers: IProvider[];
  encryption: boolean;
  defaults: IDefaultConfig | null;
  agentMode: boolean;
}

async function getProviders(): Promise<IProvidersResponse> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/providers`;
  const response = await ServerConnection.makeRequest(url, {}, settings);

  if (!response.ok) {
    console.error(
      'Failed to load providers',
      response.status,
      response.statusText
    );
    throw new Error(`Failed to load providers (${response.status})`);
  }
  return response.json();
}

async function getConfig(): Promise<IConfig> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/config`;
  const response = await ServerConnection.makeRequest(url, {}, settings);

  if (!response.ok) {
    console.error(
      'Failed to load config',
      response.status,
      response.statusText
    );
    throw new Error(`Failed to load config (${response.status})`);
  }
  return response.json();
}

async function saveConfig(config: IConfig): Promise<void> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/config`;
  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify(config)
    },
    settings
  );

  if (!response.ok) {
    const data = await response.json();
    console.error('Failed to save config', response.status, data);
    throw new Error(data.error || `Failed to save config (${response.status})`);
  }
}

interface IChatResponse {
  provider: string;
  response: Record<string, unknown>;
}

function parseAssistantContent(data: IChatResponse): string {
  const { provider, response } = data;
  if (provider === 'openai') {
    const choices = response.choices as Array<{
      message?: { content?: string };
    }>;
    return choices?.[0]?.message?.content || JSON.stringify(response);
  } else if (provider === 'anthropic') {
    const content = response.content as Array<{ text?: string }>;
    return content?.[0]?.text || JSON.stringify(response);
  }
  return JSON.stringify(response);
}

async function sendChat(
  messages: IMessage[],
  signal?: AbortSignal
): Promise<string> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/chat`;

  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify({ messages }),
      ...(signal && { signal })
    },
    settings
  );

  if (!response.ok) {
    const data = await response.json();
    console.error('Chat request failed', response.status, data);
    throw new Error(data.error || `Request failed (${response.status})`);
  }

  const data: IChatResponse = await response.json();
  return parseAssistantContent(data);
}

function humanizeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) {
    return 'just now';
  }
  if (diffMin < 60) {
    return `${diffMin}m ago`;
  }
  if (diffHour < 24) {
    return `${diffHour}h ago`;
  }
  if (diffDay < 7) {
    return `${diffDay}d ago`;
  }
  return date.toLocaleDateString();
}

// Session API
interface ISessionSummary {
  id: string;
  created: string;
  updated: string;
  messageCount: number;
}

interface ISession {
  id: string;
  created: string;
  updated: string;
  messages: IMessage[];
}

interface ISessionsResponse {
  sessions: ISessionSummary[];
  errors: Array<{ file: string; error: string }>;
}

async function getSessions(): Promise<ISessionsResponse> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/sessions`;
  const response = await ServerConnection.makeRequest(url, {}, settings);
  if (!response.ok) {
    throw new Error(`Failed to get sessions (${response.status})`);
  }
  return response.json();
}

async function getSession(sessionId: string): Promise<ISession> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/sessions/${sessionId}`;
  const response = await ServerConnection.makeRequest(url, {}, settings);
  if (!response.ok) {
    throw new Error(`Failed to get session (${response.status})`);
  }
  return response.json();
}

async function createSession(): Promise<string> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/sessions`;
  const response = await ServerConnection.makeRequest(
    url,
    { method: 'POST' },
    settings
  );
  if (!response.ok) {
    throw new Error(`Failed to create session (${response.status})`);
  }
  const data = await response.json();
  return data.id;
}

async function saveSession(
  sessionId: string,
  messages: IMessage[]
): Promise<void> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/sessions/${sessionId}`;
  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'PUT',
      body: JSON.stringify({ messages })
    },
    settings
  );
  if (!response.ok) {
    throw new Error(`Failed to save session (${response.status})`);
  }
}

interface ISettingsViewProps {
  config: IConfig;
  providers: IProvider[];
  encryption: boolean;
  defaults: IDefaultConfig | null;
  defaultsUnavailable: boolean;
  onSave: (config: IConfig) => void;
  warning?: string;
}

function CopyableCode({ code }: { code: string }): React.ReactElement {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '0.85em', color: '#666', marginBottom: '4px' }}>
        and enter this code
      </div>
      <div
        style={{
          fontSize: '1.8em',
          fontWeight: 'bold',
          letterSpacing: '0.1em',
          color: '#333',
          cursor: 'pointer'
        }}
        title="Click to copy"
        onClick={handleCopy}
      >
        {code}{' '}
        <span
          style={{
            verticalAlign: 'middle',
            marginLeft: '4px',
            display: 'inline-block'
          }}
        >
          <copyIcon.react tag="span" width="16px" height="16px" />
        </span>
      </div>
      {copied && (
        <div style={{ fontSize: '0.8em', color: '#4a86c8', marginTop: '4px' }}>
          Copied!
        </div>
      )}
    </div>
  );
}

function EnkiGateSettings({
  config,
  onSave
}: {
  config: IConfig;
  onSave: (config: IConfig) => void;
}): React.ReactElement {
  const [enkiUrl, setEnkiUrl] = React.useState(
    config.enkiGateUrl || 'https://enki-gate.web.app'
  );
  const [connecting, setConnecting] = React.useState(false);
  const [verificationUri, setVerificationUri] = React.useState('');
  const [userCode, setUserCode] = React.useState('');
  const [error, setError] = React.useState('');

  const tokenValid =
    config.enkiGateToken &&
    config.enkiGateExpiresAt &&
    config.enkiGateExpiresAt > Date.now();

  const startDeviceFlow = async () => {
    setConnecting(true);
    setError('');
    setVerificationUri('');
    setUserCode('');
    try {
      const settings = ServerConnection.makeSettings();
      const url = `${settings.baseUrl}jupyter-mynerva/enki-gate/device-flows`;
      const resp = await ServerConnection.makeRequest(
        url,
        { method: 'POST', body: JSON.stringify({ enkiGateUrl: enkiUrl }) },
        settings
      );
      if (!resp.ok) {
        const body = await resp.json();
        throw new Error(body.error || 'Failed to start device flow');
      }
      const data = await resp.json();
      setVerificationUri(data.verification_uri);
      setUserCode(data.user_code);

      const interval = (data.interval || 5) * 1000;
      const pollUrl = `${settings.baseUrl}jupyter-mynerva/enki-gate/device-flows/${encodeURIComponent(data.device_code)}/poll`;

      const poll = async (): Promise<void> => {
        const pollResp = await ServerConnection.makeRequest(
          pollUrl,
          { method: 'POST', body: JSON.stringify({ enkiGateUrl: enkiUrl }) },
          settings
        );
        if (!pollResp.ok) {
          throw new Error('Polling failed');
        }
        const pollData = await pollResp.json();
        if (pollData.status === 'pending') {
          await new Promise(r => setTimeout(r, interval));
          return poll();
        }
        if (pollData.status === 'completed') {
          const newConfig: IConfig = {
            ...config,
            provider: 'enki-gate',
            model: '',
            apiKey: '',
            enkiGateUrl: enkiUrl,
            enkiGateToken: pollData.access_token,
            enkiGateModel: pollData.selected_model || '',
            enkiGateExpiresAt: Date.now() + (pollData.expires_in || 3600) * 1000
          };
          await saveConfig(newConfig);
          onSave(newConfig);
          setVerificationUri('');
          setUserCode('');
          return;
        }
        throw new Error(`Unexpected status: ${pollData.status}`);
      };

      await poll();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Device flow failed');
    } finally {
      setConnecting(false);
    }
  };

  const remainingMinutes = tokenValid
    ? Math.ceil(((config.enkiGateExpiresAt ?? 0) - Date.now()) / 60000)
    : 0;

  return (
    <>
      <div className="jp-Mynerva-settings-field">
        <label>Enki Gate URL</label>
        <input
          type="text"
          value={enkiUrl}
          onChange={e => setEnkiUrl(e.target.value)}
          placeholder="https://enki-gate.web.app"
        />
      </div>
      {tokenValid && (
        <div className="jp-Mynerva-settings-warning">
          Connected ({config.enkiGateModel}). Token expires in{' '}
          {remainingMinutes}m.
        </div>
      )}
      {connecting && verificationUri && (
        <div
          style={{
            padding: '12px',
            background: '#f0f4ff',
            border: '1px solid #4a86c8',
            borderRadius: '4px'
          }}
        >
          <a
            href={verificationUri}
            target="_blank"
            rel="noreferrer"
            style={{
              display: 'block',
              textAlign: 'center',
              padding: '10px',
              marginBottom: '12px',
              background: '#4a86c8',
              color: 'white',
              borderRadius: '4px',
              textDecoration: 'none',
              fontWeight: 'bold'
            }}
          >
            Open Enki Gate
          </a>
          <CopyableCode code={userCode} />
        </div>
      )}
      {error && <div className="jp-Mynerva-settings-error">{error}</div>}
      {!connecting && (
        <button className="jp-Mynerva-settings-save" onClick={startDeviceFlow}>
          {tokenValid ? 'Reconnect' : 'Connect'}
        </button>
      )}
    </>
  );
}

function SettingsView({
  config,
  providers,
  encryption,
  defaults,
  defaultsUnavailable,
  onSave,
  warning
}: ISettingsViewProps): React.ReactElement {
  const [useDefault, setUseDefault] = React.useState(
    defaultsUnavailable ? false : (config.useDefault ?? false)
  );
  const [provider, setProvider] = React.useState(config.provider);
  const [model, setModel] = React.useState(config.model);
  const [apiKey, setApiKey] = React.useState(config.apiKey);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState('');

  const currentProvider =
    providers.find(p => p.id === provider) || providers[0];
  const models = currentProvider?.models || [];

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    const newProviderData = providers.find(p => p.id === newProvider);
    if (newProviderData && !newProviderData.models.includes(model)) {
      setModel(newProviderData.models[0] || '');
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError('');
    try {
      const newConfig = { provider, model, apiKey, useDefault };
      await saveConfig(newConfig);
      onSave(newConfig);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="jp-Mynerva-settings">
      {defaultsUnavailable && (
        <div className="jp-Mynerva-settings-warning">
          Default settings are no longer available. Please configure your own
          API key.
        </div>
      )}
      {defaults && (
        <div className="jp-Mynerva-settings-field jp-Mynerva-settings-checkbox">
          <label>
            <input
              type="checkbox"
              checked={useDefault}
              onChange={e => setUseDefault(e.target.checked)}
            />
            Use default settings ({defaults.provider} / {defaults.model})
          </label>
        </div>
      )}
      {!useDefault && (
        <>
          {!encryption && !warning && provider !== 'enki-gate' && (
            <div className="jp-Mynerva-settings-warning">
              API keys are stored unencrypted. Set MYNERVA_SECRET_KEY for
              encryption, or use Enki Gate for short-lived tokens.
            </div>
          )}
          <div className="jp-Mynerva-settings-field">
            <label>Provider</label>
            <select
              value={provider}
              onChange={e => handleProviderChange(e.target.value)}
            >
              {providers.map(p => (
                <option key={p.id} value={p.id}>
                  {p.displayName}
                </option>
              ))}
            </select>
          </div>
          {provider === 'enki-gate' ? (
            <EnkiGateSettings config={config} onSave={onSave} />
          ) : (
            <>
              <div className="jp-Mynerva-settings-field">
                <label>Model</label>
                <select value={model} onChange={e => setModel(e.target.value)}>
                  {models.map(m => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="jp-Mynerva-settings-field">
                <label>API Key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  placeholder="Enter API key"
                />
              </div>
            </>
          )}
        </>
      )}
      {warning && <div className="jp-Mynerva-settings-error">{warning}</div>}
      {error && <div className="jp-Mynerva-settings-error">{error}</div>}
      {provider !== 'enki-gate' && (
        <button
          className="jp-Mynerva-settings-save"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
      )}
    </div>
  );
}

const QUERY_ACTION_TYPES = [
  'getToc',
  'getSection',
  'getCells',
  'getOutput',
  'listNotebookFiles',
  'getTocFromFile',
  'getSectionFromFile',
  'getCellsFromFile',
  'getOutputFromFile',
  'listHelp',
  'help'
];
const MUTATE_ACTION_TYPES = [
  'insertCell',
  'updateCell',
  'deleteCell',
  'runCell',
  'startAgentServer'
];

// Query hierarchy: higher level permits lower levels
// getOutput > getCells > getSection > getToc
const QUERY_HIERARCHY: Record<string, string[]> = {
  getOutput: ['getCells', 'getSection', 'getToc'],
  getCells: ['getSection', 'getToc'],
  getSection: ['getToc'],
  getToc: []
};

function isQueryAction(action: IAction): action is IQueryAction {
  return QUERY_ACTION_TYPES.includes(action.type);
}

function isMutateAction(action: IAction): action is IMutateAction {
  return MUTATE_ACTION_TYPES.includes(action.type);
}

type QueryActionType = 'getToc' | 'getSection' | 'getCells' | 'getOutput';
type MutateActionType =
  | 'insertCell'
  | 'updateCell'
  | 'deleteCell'
  | 'runCell'
  | 'startAgentServer';
type ActionType = QueryActionType | MutateActionType;

function isQueryAutoApproved(
  approvedTypes: Set<ActionType>,
  actionType: QueryActionType
): boolean {
  if (approvedTypes.has(actionType)) {
    return true;
  }
  // Check hierarchy: if a higher-level action is approved, this one is too
  for (const [approved, permitted] of Object.entries(QUERY_HIERARCHY)) {
    if (
      approvedTypes.has(approved as ActionType) &&
      permitted.includes(actionType)
    ) {
      return true;
    }
  }
  return false;
}

interface IChatViewProps {
  messages: IMessage[];
  onSendMessage: (content: string) => void;
  onActionApprove: (msgIndex: number, actionIndex: number) => void;
  onActionApproveAlways: (
    msgIndex: number,
    actionIndex: number,
    action: IAction
  ) => void;
  onActionReject: (msgIndex: number, actionIndex: number) => void;
  onAcceptAll: (msgIndex: number) => void;
  onAcceptAllAlways: (msgIndex: number) => void;
  onRejectAll: (msgIndex: number) => void;
  getActionStatus: (msgIndex: number, actionIndex: number) => ActionStatus;
  actionResultUrls: Map<string, string>;
  loading: boolean;
  onCancelLoading: () => void;
  hasPendingActions: boolean;
  filterEnabled: boolean;
  onFilterToggle: (enabled: boolean) => void;
}

function getDisplayContent(msg: IMessage): string {
  if (!msg.generated) {
    return msg.content;
  }
  // For generated messages, show only the first line (e.g., "[Action Results]")
  const firstLine = msg.content.split('\n')[0];
  return firstLine;
}

function ChatView({
  messages,
  onSendMessage,
  onActionApprove,
  onActionApproveAlways,
  onActionReject,
  onAcceptAll,
  onAcceptAllAlways,
  onRejectAll,
  getActionStatus,
  actionResultUrls,
  loading,
  onCancelLoading,
  hasPendingActions,
  filterEnabled,
  onFilterToggle
}: IChatViewProps): React.ReactElement {
  const [input, setInput] = React.useState('');
  const inputDisabled = loading || hasPendingActions;
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLTextAreaElement>(null);

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  React.useEffect(() => {
    if (!loading && !hasPendingActions) {
      inputRef.current?.focus();
    }
  }, [loading, hasPendingActions]);

  const handleSend = () => {
    if (!input.trim() || inputDisabled) {
      return;
    }
    onSendMessage(input);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'Enter') {
      return;
    }
    // Shift+Enter: newline
    // isComposing: IME composing (Chrome/Firefox)
    // keyCode 229: IME input (Safari workaround)
    if (e.shiftKey || e.nativeEvent.isComposing || e.keyCode === 229) {
      return;
    }
    e.preventDefault();
    handleSend();
  };

  return (
    <>
      <div className="jp-Mynerva-messages">
        {messages.map((msg, msgIndex) => {
          const actions = msg.actions || [];
          const pendingCount = actions.filter(
            (_, i) => getActionStatus(msgIndex, i) === 'pending'
          ).length;
          const hasPending = pendingCount > 0;

          return (
            <React.Fragment key={msgIndex}>
              {/* Message */}
              <div className={`jp-Mynerva-message jp-Mynerva-${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <div
                    className="jp-Mynerva-message-content jp-Mynerva-markdown"
                    dangerouslySetInnerHTML={{
                      __html: marked.parse(getDisplayContent(msg)) as string
                    }}
                  />
                ) : (
                  <div className="jp-Mynerva-message-content">
                    {getDisplayContent(msg)}
                  </div>
                )}
              </div>
              {/* Actions with bulk header */}
              {actions.length > 0 && (
                <div className="jp-Mynerva-actions-container">
                  {hasPending && actions.length > 1 && (
                    <div className="jp-Mynerva-actions-header">
                      <span className="jp-Mynerva-actions-count">
                        {pendingCount} action{pendingCount > 1 ? 's' : ''}
                      </span>
                      <div className="jp-Mynerva-actions-bulk">
                        <DropdownButton
                          className="jp-Mynerva-accept-all"
                          options={[
                            {
                              label: 'Accept All',
                              onClick: () => onAcceptAll(msgIndex)
                            },
                            {
                              label: 'Accept All & Always',
                              onClick: () => onAcceptAllAlways(msgIndex)
                            }
                          ]}
                        />
                        <button
                          className="jp-Mynerva-bulk-button jp-Mynerva-reject-all"
                          onClick={() => onRejectAll(msgIndex)}
                        >
                          Reject All
                        </button>
                      </div>
                    </div>
                  )}
                  {/* Mutate actions (right side) */}
                  {actions.some(isMutateAction) && (
                    <div className="jp-Mynerva-actions jp-Mynerva-user">
                      {actions.map((action, actionIndex) =>
                        isMutateAction(action) ? (
                          <MutateActionCard
                            key={actionIndex}
                            action={action}
                            status={getActionStatus(msgIndex, actionIndex)}
                            resultUrl={actionResultUrls.get(`${msgIndex}:${actionIndex}`)}
                            onApprove={() =>
                              onActionApprove(msgIndex, actionIndex)
                            }
                            onApproveAlways={() =>
                              onActionApproveAlways(
                                msgIndex,
                                actionIndex,
                                action
                              )
                            }
                            onReject={() =>
                              onActionReject(msgIndex, actionIndex)
                            }
                          />
                        ) : null
                      )}
                    </div>
                  )}
                  {/* Query actions (right side) */}
                  {actions.some(isQueryAction) && (
                    <div className="jp-Mynerva-actions jp-Mynerva-user">
                      {actions.map((action, actionIndex) =>
                        isQueryAction(action) ? (
                          <QueryActionCard
                            key={actionIndex}
                            action={action}
                            status={getActionStatus(msgIndex, actionIndex)}
                            onApprove={() =>
                              onActionApprove(msgIndex, actionIndex)
                            }
                            onApproveAlways={() =>
                              onActionApproveAlways(
                                msgIndex,
                                actionIndex,
                                action
                              )
                            }
                            onReject={() =>
                              onActionReject(msgIndex, actionIndex)
                            }
                          />
                        ) : null
                      )}
                    </div>
                  )}
                </div>
              )}
            </React.Fragment>
          );
        })}
        {loading && (
          <div className="jp-Mynerva-message jp-Mynerva-assistant">
            <div className="jp-Mynerva-message-content">...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="jp-Mynerva-input-area">
        <textarea
          ref={inputRef}
          className="jp-Mynerva-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            hasPendingActions
              ? 'Please respond to pending actions...'
              : 'Ask Mynerva...'
          }
          rows={2}
          disabled={inputDisabled}
        />
        <div className="jp-Mynerva-input-controls">
          <label className="jp-Mynerva-filter-toggle">
            <input
              type="checkbox"
              checked={filterEnabled}
              onChange={e => onFilterToggle(e.target.checked)}
            />
            Privacy filter
          </label>
          {loading ? (
            <button className="jp-Mynerva-cancel" onClick={onCancelLoading}>
              Cancel
            </button>
          ) : (
            <button
              className="jp-Mynerva-send"
              onClick={handleSend}
              disabled={hasPendingActions}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </>
  );
}

interface IMynervaComponentProps {
  contextEngine: ContextEngine;
  liveQuery: NblibramLiveQuery;
}

function MynervaComponent({
  contextEngine,
  liveQuery
}: IMynervaComponentProps): React.ReactElement {
  const [providers, setProviders] = React.useState<IProvider[]>([]);
  const [encryption, setEncryption] = React.useState(false);
  const [defaults, setDefaults] = React.useState<IDefaultConfig | null>(null);
  const [config, setConfig] = React.useState<IConfig | null>(null);
  const [showSettings, setShowSettings] = React.useState(false);
  const [messages, setMessages] = React.useState<IMessage[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [initializing, setInitializing] = React.useState(true);
  const [initError, setInitError] = React.useState<string | null>(null);
  const [filterEnabled, setFilterEnabled] = React.useState(true);
  liveQuery.filterEnabled = filterEnabled;
  // Auto-approval for active notebook: Map<notebookPath, Set<actionType>>
  const [autoApproved, setAutoApproved] = React.useState<
    Map<string, Set<ActionType>>
  >(new Map());
  // Auto-approval for file queries: Map<targetPath, Set<fileQueryActionType>>
  const [fileAutoApproved, setFileAutoApproved] = React.useState<
    Map<string, Set<string>>
  >(new Map());
  // Session management
  const [sessionId, setSessionId] = React.useState<string | null>(null);
  const [sessions, setSessions] = React.useState<ISessionSummary[]>([]);
  const [sessionLoadErrors, setSessionLoadErrors] = React.useState<
    Array<{ file: string; error: string }>
  >([]);
  const [sessionError, setSessionError] = React.useState<string | null>(null);
  const [showSessions, setShowSessions] = React.useState(false);
  const [agentMode, setAgentMode] = React.useState(false);
  // Map from "msgIndex:actionIndex" to result URL (for startAgentServer)
  const [actionResultUrls, setActionResultUrls] = React.useState<
    Map<string, string>
  >(new Map());

  // AbortController for cancelling chat requests
  const abortControllerRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    Promise.all([getProviders(), getConfig(), getSessions()])
      .then(async ([providersRes, cfg, sessionsRes]) => {
        setProviders(providersRes.providers);
        setEncryption(providersRes.encryption);
        setDefaults(providersRes.defaults);
        setAgentMode(providersRes.agentMode ?? false);
        setConfig(cfg);
        setSessions(sessionsRes.sessions);
        setSessionLoadErrors(sessionsRes.errors);

        if (cfg.decryptError) {
          setShowSettings(true);
        }

        // Show settings if:
        // - no API key and not using defaults, OR
        // - useDefault is set but defaults are not available
        const defaultsUnavailable = cfg.useDefault && !providersRes.defaults;
        if ((!cfg.apiKey && !cfg.useDefault) || defaultsUnavailable) {
          setShowSettings(true);
        }
      })
      .catch(e => {
        setInitError(e instanceof Error ? e.message : 'Failed to initialize');
      })
      .finally(() => {
        setInitializing(false);
      });
  }, []);

  // Track action statuses: messageIndex -> actionIndex -> status
  const [actionStatuses, setActionStatuses] = React.useState<
    Map<number, Map<number, ActionStatus>>
  >(new Map());

  // Queue of action results waiting to be sent
  const [pendingResults, setPendingResults] = React.useState<string[]>([]);

  // Flag to prevent duplicate execution from useEffect during batch operations
  const executingActionsRef = React.useRef(false);

  // Auto-save session when messages change
  React.useEffect(() => {
    if (sessionId && messages.length > 0) {
      saveSession(sessionId, messages).catch(e => {
        setSessionError(
          `Failed to save session: ${e instanceof Error ? e.message : 'Unknown error'}`
        );
      });
    }
  }, [sessionId, messages]);

  // Session switching
  const handleSessionSwitch = async (newSessionId: string) => {
    if (newSessionId === sessionId) {
      return;
    }
    try {
      setSessionError(null);
      const session = await getSession(newSessionId);
      setSessionId(session.id);
      setMessages(session.messages);
      // Mark all actions in loaded session as executed (already processed)
      const statuses = new Map<number, Map<number, ActionStatus>>();
      session.messages.forEach((msg, msgIndex) => {
        const actions = msg.actions || [];
        if (actions.length > 0) {
          const actionMap = new Map<number, ActionStatus>();
          actions.forEach((_, actionIndex) => {
            actionMap.set(actionIndex, 'executed');
          });
          statuses.set(msgIndex, actionMap);
        }
      });
      setActionStatuses(statuses);
      setPendingResults([]);
    } catch (e) {
      setSessionError(
        `Failed to load session: ${e instanceof Error ? e.message : 'Unknown error'}`
      );
    }
  };

  // Create new session
  const handleNewSession = async () => {
    try {
      setSessionError(null);
      const newId = await createSession();
      setSessionId(newId);
      setMessages([]);
      setActionStatuses(new Map());
      setPendingResults([]);
      const sessionsRes = await getSessions();
      setSessions(sessionsRes.sessions);
      setSessionLoadErrors(sessionsRes.errors);
    } catch (e) {
      setSessionError(
        `Failed to create session: ${e instanceof Error ? e.message : 'Unknown error'}`
      );
    }
  };

  const executeQueryAction = async (action: IAction): Promise<string> => {
    let result: string;
    switch (action.type) {
      case 'getToc': {
        const toc = await liveQuery.getToc();
        result = JSON.stringify({ type: 'getToc', result: toc }, null, 2);
        break;
      }
      case 'getSection': {
        const cells = await liveQuery.getSection(action.query);
        result = JSON.stringify({ type: 'getSection', result: cells }, null, 2);
        break;
      }
      case 'getCells': {
        const cells = await liveQuery.getCells(action.query, action.count);
        result = JSON.stringify({ type: 'getCells', result: cells }, null, 2);
        break;
      }
      case 'getOutput': {
        const outputs = await liveQuery.getOutput(action.query);
        result = JSON.stringify(
          { type: 'getOutput', result: outputs },
          null,
          2
        );
        break;
      }
      case 'listHelp': {
        result = JSON.stringify(
          { type: 'listHelp', result: buildSystemPrompt(agentMode) },
          null,
          2
        );
        break;
      }
      case 'help': {
        result = JSON.stringify(
          { type: 'help', result: getActionHelp(action.action) },
          null,
          2
        );
        break;
      }
      case 'listNotebookFiles': {
        const { ContentsManager } = await import('@jupyterlab/services');
        const contents = new ContentsManager();
        const model = await contents.get(action.path || '');
        const files = (model.content as { type: string; path: string }[])
          .filter(item => item.type === 'notebook')
          .map(item => item.path);
        result = JSON.stringify(
          { type: 'listNotebookFiles', path: action.path || '', result: files },
          null,
          2
        );
        break;
      }
      case 'getTocFromFile': {
        const toc = await nblibramTocFromFile(action.path);
        result = JSON.stringify(
          { type: 'getTocFromFile', path: action.path, result: toc },
          null,
          2
        );
        break;
      }
      case 'getSectionFromFile': {
        const cells = await nblibramSectionFromFile(action.path, action.query);
        result = JSON.stringify(
          { type: 'getSectionFromFile', path: action.path, result: cells },
          null,
          2
        );
        break;
      }
      case 'getCellsFromFile': {
        const cells = await nblibramCellsFromFile(
          action.path,
          action.query,
          action.count
        );
        result = JSON.stringify(
          { type: 'getCellsFromFile', path: action.path, result: cells },
          null,
          2
        );
        break;
      }
      case 'getOutputFromFile': {
        const outputs = await nblibramOutputsFromFile(
          action.path,
          action.query
        );
        result = JSON.stringify(
          { type: 'getOutputFromFile', path: action.path, result: outputs },
          null,
          2
        );
        break;
      }
      default:
        result = JSON.stringify(
          { type: 'unknown', error: 'Unknown action type' },
          null,
          2
        );
    }

    return result;
  };

  const executeMutateAction = async (action: IAction): Promise<string> => {
    switch (action.type) {
      case 'insertCell': {
        const result = contextEngine.insertCell(
          action.position,
          action.cellType,
          action.source
        );
        return JSON.stringify({ type: 'insertCell', result }, null, 2);
      }
      case 'updateCell': {
        const result = contextEngine.updateCell(
          action.query,
          action.source,
          action._hash
        );
        return JSON.stringify({ type: 'updateCell', result }, null, 2);
      }
      case 'deleteCell': {
        const result = contextEngine.deleteCell(action.query, action._hash);
        return JSON.stringify({ type: 'deleteCell', result }, null, 2);
      }
      case 'runCell': {
        if (!agentMode) {
          return JSON.stringify(
            {
              type: 'runCell',
              error:
                'Cell execution requires an agent environment. Use startAgentServer first.'
            },
            null,
            2
          );
        }
        const result = await contextEngine.runCell(action.query);
        return JSON.stringify({ type: 'runCell', result }, null, 2);
      }
      case 'startAgentServer': {
        const settings = ServerConnection.makeSettings();
        const url = `${settings.baseUrl}jupyter-mynerva/agent-server`;
        const resp = await ServerConnection.makeRequest(
          url,
          {
            method: 'POST',
            body: JSON.stringify({ ssh: action.ssh })
          },
          settings
        );
        const result = await resp.json();
        return JSON.stringify({ type: 'startAgentServer', result }, null, 2);
      }
      default:
        return JSON.stringify(
          { type: 'unknown', error: 'Unknown action type' },
          null,
          2
        );
    }
  };

  const getActionStatus = (
    msgIndex: number,
    actionIndex: number
  ): ActionStatus => {
    return actionStatuses.get(msgIndex)?.get(actionIndex) ?? 'pending';
  };

  const hasPendingActions = messages.some((msg, msgIndex) =>
    (msg.actions || []).some((_, actionIndex) => {
      const status = getActionStatus(msgIndex, actionIndex);
      return status === 'pending' || status === 'approved';
    })
  );

  const setActionStatus = (
    msgIndex: number,
    actionIndex: number,
    status: ActionStatus
  ) => {
    setActionStatuses(prev => {
      const newMap = new Map(prev);
      if (!newMap.has(msgIndex)) {
        newMap.set(msgIndex, new Map());
      }
      newMap.get(msgIndex)!.set(actionIndex, status);
      return newMap;
    });
  };

  const handleActionApprove = (msgIndex: number, actionIndex: number) => {
    setActionStatus(msgIndex, actionIndex, 'approved');
  };

  const handleActionReject = (msgIndex: number, actionIndex: number) => {
    setActionStatus(msgIndex, actionIndex, 'rejected');
  };

  const executeApprovedAction = async (
    msgIndex: number,
    actionIndex: number,
    action: IAction
  ) => {
    let result: string;
    try {
      if (isQueryAction(action)) {
        result = await executeQueryAction(action);
      } else {
        result = await executeMutateAction(action);
      }
    } catch (e) {
      console.error('Action failed:', action.type, e);
      result = JSON.stringify(
        {
          type: action.type,
          error: e instanceof Error ? e.message : 'Unknown error'
        },
        null,
        2
      );
    }

    if (action.type === 'startAgentServer') {
      const parsed = JSON.parse(result);
      setActionResultUrls(prev => {
        const next = new Map(prev);
        next.set(`${msgIndex}:${actionIndex}`, parsed.result.url);
        return next;
      });
    }

    setPendingResults(prev => [...prev, result]);
    setActionStatus(msgIndex, actionIndex, 'executed');
  };

  const FILE_QUERY_TYPES = [
    'listNotebookFiles',
    'getTocFromFile',
    'getSectionFromFile',
    'getCellsFromFile',
    'getOutputFromFile'
  ];

  const isFileQueryAction = (action: IAction): boolean => {
    return FILE_QUERY_TYPES.includes(action.type);
  };

  const getFileQueryTargetPath = (action: IAction): string => {
    if (action.type === 'listNotebookFiles') {
      return (action as IListNotebookFilesAction).path || '';
    }
    return (action as { path: string }).path;
  };

  const addAutoApproval = (action: IAction) => {
    if (isFileQueryAction(action)) {
      const targetPath = getFileQueryTargetPath(action);
      setFileAutoApproved(prev => {
        const newMap = new Map(prev);
        const types = newMap.get(targetPath) ?? new Set<string>();
        types.add(action.type);
        newMap.set(targetPath, types);
        return newMap;
      });
    } else {
      const path = contextEngine.getNotebookPath();
      setAutoApproved(prev => {
        const newMap = new Map(prev);
        const types = newMap.get(path) ?? new Set<ActionType>();
        types.add(action.type as ActionType);
        newMap.set(path, types);
        return newMap;
      });
    }
  };

  const isActionAutoApproved = (action: IAction): boolean => {
    if (isFileQueryAction(action)) {
      const targetPath = getFileQueryTargetPath(action);
      const approved = fileAutoApproved.get(targetPath);
      return approved?.has(action.type) ?? false;
    }

    if (!contextEngine.hasActiveNotebook()) {
      return false;
    }
    const path = contextEngine.getNotebookPath();
    const approved = autoApproved.get(path);
    if (!approved) {
      return false;
    }
    if (isQueryAction(action)) {
      return isQueryAutoApproved(approved, action.type as QueryActionType);
    }
    return approved.has(action.type as ActionType);
  };

  const handleActionApproveAlways = (
    msgIndex: number,
    actionIndex: number,
    action: IAction
  ) => {
    handleActionApprove(msgIndex, actionIndex);
    addAutoApproval(action);
  };

  const handleAcceptAll = (msgIndex: number) => {
    const msg = messages[msgIndex];
    const actions = msg.actions || [];

    for (let i = 0; i < actions.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionApprove(msgIndex, i);
    }
  };

  const handleRejectAll = (msgIndex: number) => {
    const msg = messages[msgIndex];
    const actions = msg.actions || [];

    for (let i = 0; i < actions.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionReject(msgIndex, i);
    }
  };

  const handleAcceptAllAlways = (msgIndex: number) => {
    const msg = messages[msgIndex];
    const actions = msg.actions || [];

    for (let i = 0; i < actions.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionApprove(msgIndex, i);
      addAutoApproval(actions[i]);
    }
  };

  // Execute approved actions when all actions in a message are decided
  React.useEffect(() => {
    if (loading || executingActionsRef.current) {
      return;
    }

    const executeBatch = async () => {
      executingActionsRef.current = true;
      try {
        for (let msgIndex = 0; msgIndex < messages.length; msgIndex++) {
          const msg = messages[msgIndex];
          const actions = msg.actions || [];
          if (actions.length === 0) {
            continue;
          }

          // Check if all actions are decided (not pending)
          const allDecided = actions.every(
            (_, i) => getActionStatus(msgIndex, i) !== 'pending'
          );
          if (!allDecided) {
            continue;
          }

          // Check if any actions need processing
          const hasApproved = actions.some(
            (_, i) => getActionStatus(msgIndex, i) === 'approved'
          );
          const hasRejected = actions.some(
            (_, i) => getActionStatus(msgIndex, i) === 'rejected'
          );
          if (!hasApproved && !hasRejected) {
            continue;
          }

          // Process actions in order: execute approved, notify rejected
          for (let i = 0; i < actions.length; i++) {
            const status = getActionStatus(msgIndex, i);
            if (status === 'approved') {
              await executeApprovedAction(msgIndex, i, actions[i]);
            } else if (status === 'rejected') {
              const result = JSON.stringify(
                { type: actions[i].type, rejected: true },
                null,
                2
              );
              setPendingResults(prev => [...prev, result]);
              setActionStatus(msgIndex, i, 'notified');
            }
          }
        }
      } finally {
        executingActionsRef.current = false;
      }
    };

    executeBatch();
  }, [messages, actionStatuses, loading]);

  // Send results when all actions are resolved
  React.useEffect(() => {
    if (hasPendingActions || pendingResults.length === 0 || loading) {
      return;
    }

    const sendResults = async () => {
      setLoading(true);
      const results = pendingResults;
      setPendingResults([]);

      const feedbackMessage: IMessage = {
        role: 'user',
        content: `[Action Results]\n${results.join('\n\n')}`,
        generated: true
      };
      const newMessages = [...messages, feedbackMessage];
      setMessages(newMessages);

      const chatMessages = [
        { role: 'system' as const, content: buildSystemPrompt(agentMode) },
        ...newMessages
      ];

      try {
        const response = await sendChat(chatMessages);
        const finalMessages = await processLLMResponse(
          response,
          newMessages,
          0
        );
        setMessages(finalMessages);
      } catch (e) {
        const errorMessage: IMessage = {
          role: 'assistant',
          content: `Error: ${e instanceof Error ? e.message : 'Unknown error'}`
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    };

    sendResults();
  }, [hasPendingActions, pendingResults, loading]);

  // Auto-approve actions when all actions in a batch are auto-approvable
  React.useEffect(() => {
    if (loading || executingActionsRef.current) {
      return;
    }

    for (let msgIndex = 0; msgIndex < messages.length; msgIndex++) {
      const msg = messages[msgIndex];
      const actions = msg.actions || [];
      if (actions.length === 0) {
        continue;
      }

      // Check if any pending actions exist
      const hasPending = actions.some(
        (_, i) => getActionStatus(msgIndex, i) === 'pending'
      );
      if (!hasPending) {
        continue;
      }

      // Check if ALL pending actions are auto-approvable
      const allAutoApprovable = actions.every((action, i) => {
        const status = getActionStatus(msgIndex, i);
        return status !== 'pending' || isActionAutoApproved(action);
      });

      if (allAutoApprovable) {
        // Approve all pending actions
        for (let i = 0; i < actions.length; i++) {
          if (getActionStatus(msgIndex, i) === 'pending') {
            handleActionApprove(msgIndex, i);
          }
        }
      }
      // If not all auto-approvable, do nothing (show buttons for all)
    }
  }, [messages, autoApproved, fileAutoApproved]);

  const processLLMResponse = async (
    rawContent: string,
    currentMessages: IMessage[],
    retryCount: number,
    signal?: AbortSignal
  ): Promise<IMessage[]> => {
    const MAX_RETRIES = 2;
    const parseResult = parseRawContent(rawContent);

    // JSON parse error - retry with feedback
    if (parseResult.warning) {
      if (retryCount < MAX_RETRIES) {
        const assistantMessage: IMessage = {
          role: 'assistant',
          content: '(Format error - retrying...)'
        };
        const feedbackMessage: IMessage = {
          role: 'user',
          content: `[Format Error]\n\nYour response was not valid JSON. You must respond with JSON only, no text before or after.\n\nError: ${parseResult.warning.message}\n\nPlease retry with correct JSON format.`,
          generated: true
        };
        const newMessages = [
          ...currentMessages,
          assistantMessage,
          feedbackMessage
        ];
        setMessages(newMessages);

        const chatMessages = [
          { role: 'system' as const, content: buildSystemPrompt(agentMode) },
          ...newMessages
        ];

        const nextResponse = await sendChat(chatMessages, signal);
        return processLLMResponse(
          nextResponse,
          newMessages,
          retryCount + 1,
          signal
        );
      }
      // Max retries reached - show raw content
      return [...currentMessages, { role: 'assistant', content: rawContent }];
    }

    const llmResponse = parseResult.response!;
    const validation = validateActions(llmResponse.actions);

    const assistantContent = llmResponse.messages
      .map(m => m.content)
      .filter(Boolean)
      .join('\n\n');

    // Action validation error - retry with feedback
    if (!validation.valid && retryCount < MAX_RETRIES) {
      const assistantMessage: IMessage = {
        role: 'assistant',
        content: assistantContent || '(Action format error - retrying...)'
      };
      const feedbackMessage: IMessage = {
        role: 'user',
        content: validation.feedbackMessage!,
        generated: true
      };
      const newMessages = [
        ...currentMessages,
        assistantMessage,
        feedbackMessage
      ];
      setMessages(newMessages);

      const chatMessages = [
        { role: 'system' as const, content: buildSystemPrompt(agentMode) },
        ...newMessages
      ];

      const nextResponse = await sendChat(chatMessages, signal);
      return processLLMResponse(
        nextResponse,
        newMessages,
        retryCount + 1,
        signal
      );
    }

    // Validation passed - include actions
    const assistantMessage: IMessage = {
      role: 'assistant',
      content: assistantContent || '(no message)',
      actions: llmResponse.actions as IAction[]
    };

    return [...currentMessages, assistantMessage];
  };

  const handleCancelLoading = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    // Remove the last user message
    setMessages(prev => prev.slice(0, -1));
    setLoading(false);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: IMessage = { role: 'user', content };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Create session on first message
    if (!sessionId) {
      try {
        const newId = await createSession();
        setSessionId(newId);
        const sessionsRes = await getSessions();
        setSessions(sessionsRes.sessions);
        setSessionLoadErrors(sessionsRes.errors);
      } catch (e) {
        setSessionError(
          `Failed to create session: ${e instanceof Error ? e.message : 'Unknown error'}`
        );
      }
    }

    try {
      const chatMessages = [
        { role: 'system' as const, content: buildSystemPrompt(agentMode) },
        ...newMessages
      ];
      const response = await sendChat(chatMessages, controller.signal);
      const finalMessages = await processLLMResponse(
        response,
        newMessages,
        0,
        controller.signal
      );
      setMessages(finalMessages);
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        return; // Already handled by handleCancelLoading
      }
      const errorMessage: IMessage = {
        role: 'assistant',
        content: `Error: ${e instanceof Error ? e.message : 'Unknown error'}`
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      abortControllerRef.current = null;
      setLoading(false);
    }
  };

  const handleConfigSave = (newConfig: IConfig) => {
    setConfig(newConfig);
    setShowSettings(false);
  };

  if (initializing) {
    return (
      <div className={PANEL_CLASS}>
        <div className="jp-Mynerva-header">
          <span className="jp-Mynerva-title">
            <mynervaIcon.react tag="span" className="jp-Mynerva-title-icon" />
            MYNERVA
          </span>
        </div>
        <div className="jp-Mynerva-loading">Loading...</div>
      </div>
    );
  }

  if (initError) {
    return (
      <div className={PANEL_CLASS}>
        <div className="jp-Mynerva-header">
          <span className="jp-Mynerva-title">
            <mynervaIcon.react tag="span" className="jp-Mynerva-title-icon" />
            MYNERVA
          </span>
        </div>
        <div className="jp-Mynerva-settings">
          <div className="jp-Mynerva-settings-error">{initError}</div>
        </div>
      </div>
    );
  }

  const defaultConfig: IConfig = {
    provider: providers[0]?.id || 'openai',
    model: providers[0]?.models[0] || '',
    apiKey: ''
  };

  const currentSession = sessions.find(s => s.id === sessionId);

  return (
    <div className={PANEL_CLASS}>
      <div className="jp-Mynerva-header">
        <span className="jp-Mynerva-title">
          <mynervaIcon.react tag="span" className="jp-Mynerva-title-icon" />
          MYNERVA
        </span>
        <div className="jp-Mynerva-header-buttons">
          <div className="jp-Mynerva-session-dropdown">
            <button
              className="jp-Mynerva-header-button"
              onClick={() => setShowSessions(!showSessions)}
              title="Sessions"
            >
              {currentSession
                ? `Started ${humanizeTime(currentSession.created)}`
                : 'Not started'}
            </button>
            {showSessions && (
              <div className="jp-Mynerva-session-menu">
                {sessions.map(s => (
                  <button
                    key={s.id}
                    className={`jp-Mynerva-session-item ${s.id === sessionId ? 'jp-Mynerva-session-active' : ''}`}
                    onClick={() => {
                      handleSessionSwitch(s.id);
                      setShowSessions(false);
                    }}
                  >
                    <span className="jp-Mynerva-session-time">
                      Started {humanizeTime(s.created)}
                    </span>
                    <span className="jp-Mynerva-session-count">
                      {s.messageCount} msg
                    </span>
                  </button>
                ))}
                <button
                  className="jp-Mynerva-session-item jp-Mynerva-session-new"
                  onClick={() => {
                    handleNewSession();
                    setShowSessions(false);
                  }}
                >
                  + New session
                </button>
              </div>
            )}
          </div>
          <button
            className="jp-Mynerva-header-button"
            onClick={() => setShowSettings(!showSettings)}
            title="Settings"
          >
            <settingsIcon.react tag="span" />
          </button>
        </div>
      </div>
      {sessionError && (
        <div className="jp-Mynerva-session-error">{sessionError}</div>
      )}
      {sessionLoadErrors.length > 0 && (
        <div className="jp-Mynerva-session-errors">
          {sessionLoadErrors.map((err, i) => (
            <div key={i}>
              Failed to load {err.file}: {err.error}
            </div>
          ))}
        </div>
      )}
      {showSettings ? (
        <SettingsView
          config={config || defaultConfig}
          providers={providers}
          encryption={encryption}
          defaults={defaults}
          defaultsUnavailable={!!(config?.useDefault && !defaults)}
          onSave={handleConfigSave}
          warning={config?.decryptError}
        />
      ) : (
        <ChatView
          messages={messages}
          onSendMessage={handleSendMessage}
          onActionApprove={handleActionApprove}
          onActionApproveAlways={handleActionApproveAlways}
          onActionReject={handleActionReject}
          onAcceptAll={handleAcceptAll}
          onAcceptAllAlways={handleAcceptAllAlways}
          onRejectAll={handleRejectAll}
          getActionStatus={getActionStatus}
          actionResultUrls={actionResultUrls}
          loading={loading}
          onCancelLoading={handleCancelLoading}
          hasPendingActions={hasPendingActions}
          filterEnabled={filterEnabled}
          onFilterToggle={setFilterEnabled}
        />
      )}
    </div>
  );
}

export class MynervaPanel extends ReactWidget {
  private _contextEngine: ContextEngine;
  private _liveQuery: NblibramLiveQuery;

  constructor(contextEngine: ContextEngine, liveQuery: NblibramLiveQuery) {
    super();
    this._contextEngine = contextEngine;
    this._liveQuery = liveQuery;
    this.id = 'mynerva-panel';
    this.title.icon = mynervaIcon;
    this.title.caption = 'Mynerva';
    this.addClass(PANEL_CLASS);
  }

  render(): React.ReactElement {
    return (
      <MynervaComponent
        contextEngine={this._contextEngine}
        liveQuery={this._liveQuery}
      />
    );
  }
}

export function activatePanel(
  shell: ILabShell,
  contextEngine: ContextEngine,
  liveQuery: NblibramLiveQuery
): void {
  const panel = new MynervaPanel(contextEngine, liveQuery);
  shell.add(panel, 'right', { rank: 1000 });
}
