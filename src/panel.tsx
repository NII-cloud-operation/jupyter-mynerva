import { JupyterFrontEnd } from '@jupyterlab/application';
import { ServerConnection } from '@jupyterlab/services';
import {
  ReactWidget,
  settingsIcon,
  copyIcon,
  refreshIcon
} from '@jupyterlab/ui-components';
import * as React from 'react';
import { marked } from 'marked';
import { Streamdown } from 'streamdown';

import { ContextEngine } from './context';
import {
  IAction,
  IQueryAction,
  IMutateAction,
  IListNotebookFilesAction,
  IToolCall,
  IToolResult,
  ISearchNotebooksAction,
  IGetCellsFromSearchAction,
  ISummaryCellsFromSearchAction,
  ActionStatus,
  IToolDefinition,
  getToolDefinitions,
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
  // Assistant: tool calls extracted for the UI / execution.
  toolCalls?: IToolCall[];
  // Assistant: provider-native content blocks, kept opaque and resent verbatim
  // so reasoning/thinking signatures round trip.
  assistantBlocks?: unknown;
  // User: results of executed tool calls, sent back to the LLM.
  toolResults?: IToolResult[];
  generated?: boolean; // Auto-generated messages (show brief in UI)
}

/** The completed result of one streamed chat turn. */
interface IChatResult {
  text: string;
  stopReason?: string;
  toolCalls: IToolCall[];
  assistantBlocks?: unknown;
}

interface IConfig {
  provider: string;
  model: string;
  decryptError?: string;
  configWarning?: string;
  apiKey: string;
  useDefault?: boolean;
  openaiBaseUrl?: string;
  bedrockRegion?: string;
  enkiGateUrl?: string;
  enkiGateToken?: string;
  enkiGateModel?: string;
  enkiGateExpiresAt?: number;
}

interface IDefaultConfig {
  provider: string;
  model: string;
  openaiBaseUrl?: string;
  bedrockRegion?: string;
}

interface IProvider {
  id: string;
  displayName: string;
  models: string[];
  modelsError?: string;
}

interface IBedrockRegion {
  id: string;
  name: string;
}

interface IProvidersResponse {
  providers: IProvider[];
  encryption: boolean;
  defaults: IDefaultConfig | null;
  defaultsOnly?: boolean;
  defaultsError?: string;
  nbsearchAvailable?: boolean;
  bedrockRegions: IBedrockRegion[];
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

async function fetchProviderModels(
  provider: string,
  apiKey: string,
  baseUrl: string,
  region?: string
): Promise<string[]> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/provider-models`;
  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify({ provider, apiKey, baseUrl, region })
    },
    settings
  );

  if (!response.ok) {
    const data = await response.json();
    throw new Error(
      data.error || `Failed to fetch models (${response.status})`
    );
  }
  const data = await response.json();
  return data.models;
}

function isOpenAIDefaultBaseUrl(baseUrl: string): boolean {
  return baseUrl.trim().replace(/\/+$/, '') === 'https://api.openai.com/v1';
}

interface IStreamCallbacks {
  onContentBlockStart: (
    contentType: string,
    metadata?: Record<string, unknown>
  ) => void;
  onContentBlockDelta: (contentType: string, delta: string) => void;
  onContentBlockStop: (
    contentType: string,
    metadata?: Record<string, unknown>
  ) => void;
  onMessageDone: (result: IChatResult) => void;
}

async function sendChat(
  messages: IMessage[],
  tools: IToolDefinition[],
  callbacks: IStreamCallbacks,
  signal?: AbortSignal
): Promise<IChatResult> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/chat`;

  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify({ messages, tools }),
      ...(signal && { signal })
    },
    settings
  );

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.error || `Request failed (${response.status})`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let result: IChatResult = { text: '', toolCalls: [] };
  let buffer = '';
  let reading = true;

  while (reading) {
    const { done, value } = await reader.read();
    if (done) {
      reading = false;
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) {
        continue;
      }
      const payload = trimmed.slice(6);
      if (payload === '[DONE]') {
        return result;
      }
      try {
        const parsed = JSON.parse(payload);

        if (parsed.type === 'error') {
          throw new Error(parsed.error);
        } else if (parsed.type === 'content_block_start') {
          callbacks.onContentBlockStart(parsed.content_type, parsed);
        } else if (parsed.type === 'content_block_delta') {
          callbacks.onContentBlockDelta(parsed.content_type, parsed.delta);
        } else if (parsed.type === 'content_block_stop') {
          callbacks.onContentBlockStop(parsed.content_type, parsed);
        } else if (parsed.type === 'message_done') {
          result = {
            text: parsed.text,
            stopReason: parsed.stop_reason,
            toolCalls: parsed.tool_calls || [],
            assistantBlocks: parsed.assistant_blocks
          };
          callbacks.onMessageDone(result);
        }
      } catch (e) {
        if (e instanceof SyntaxError) {
          continue;
        }
        throw e;
      }
    }
  }

  return result;
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

// LLM targets stream SSE (so a reverse proxy never times out a multi-minute
// summarization and we can show progress); the LLM-free target is plain JSON.
const NBSEARCH_STREAMING_TARGETS = ['notebooks', 'summary-cells-from-search'];

async function readNBSearchSSE(
  response: Response,
  onProgress?: (event: Record<string, unknown>) => void
): Promise<unknown> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: unknown;
  let done = false;
  while (!done) {
    const { done: streamDone, value } = await reader.read();
    if (streamDone) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      // Ignore SSE comments (heartbeats) and non-data lines.
      if (!trimmed || !trimmed.startsWith('data: ')) {
        continue;
      }
      const payload = trimmed.slice('data: '.length);
      if (payload === '[DONE]') {
        done = true;
        break;
      }
      const event = JSON.parse(payload);
      if (event.type === 'progress') {
        onProgress?.(event);
      } else if (event.type === 'done') {
        result = event.result;
      } else if (event.type === 'error') {
        throw new Error(event.error || 'nbsearch request failed');
      }
    }
  }
  return result;
}

async function callNBSearch(
  target: 'notebooks' | 'summary-cells-from-search' | 'cells-from-search',
  action: IAction,
  filterEnabled: boolean,
  signal?: AbortSignal,
  onProgress?: (event: Record<string, unknown>) => void
): Promise<unknown> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}jupyter-mynerva/nbsearch/${target}`;
  const response = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify({
        ...normalizeSearchNotebooksAction(action),
        noFilter: !filterEnabled || undefined
      }),
      ...(signal && { signal })
    },
    settings
  );
  if (NBSEARCH_STREAMING_TARGETS.includes(target)) {
    if (!response.ok) {
      let message = `nbsearch ${target} failed`;
      try {
        message = (await response.json()).error || message;
      } catch {
        // Non-JSON transport error; keep the generic message.
      }
      throw new Error(message);
    }
    return readNBSearchSSE(response, onProgress);
  }
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || `nbsearch ${target} failed`);
  }
  return body;
}

function formatActionProgress(event: Record<string, unknown>): string {
  if (event.phase === 'notebook') {
    return `notebook ${event.current}/${event.total}`;
  }
  if (event.phase === 'summarize') {
    return typeof event.detail === 'string' ? event.detail : 'summarizing…';
  }
  return '';
}

function parseLocalDate(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) {
    return null;
  }

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(year, month - 1, day);

  if (
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day
  ) {
    return null;
  }

  return date;
}

function normalizeSearchNotebooksAction(action: IAction): IAction {
  if (action.type !== 'searchNotebooks') {
    return action;
  }

  const normalized = { ...action };
  if (action.dateFrom) {
    const dateFrom = parseLocalDate(action.dateFrom);
    if (!dateFrom) {
      throw new Error('searchNotebooks dateFrom must be YYYY-MM-DD');
    }
    normalized.dateTimeFrom = dateFrom.toISOString();
  }

  if (action.dateTo) {
    const dateTo = parseLocalDate(action.dateTo);
    if (!dateTo) {
      throw new Error('searchNotebooks dateTo must be YYYY-MM-DD');
    }
    dateTo.setDate(dateTo.getDate() + 1);
    normalized.dateTimeTo = dateTo.toISOString();
  }

  return normalized;
}

interface ISettingsViewProps {
  config: IConfig;
  providers: IProvider[];
  bedrockRegions: IBedrockRegion[];
  encryption: boolean;
  defaults: IDefaultConfig | null;
  defaultsUnavailable: string | null;
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
      {!verificationUri && (
        <button
          className="jp-Mynerva-settings-save"
          onClick={startDeviceFlow}
          disabled={connecting}
        >
          {connecting ? 'Connecting...' : tokenValid ? 'Reconnect' : 'Connect'}
        </button>
      )}
    </>
  );
}

function SettingsView({
  config,
  providers,
  bedrockRegions,
  encryption,
  defaults,
  defaultsUnavailable,
  onSave,
  warning
}: ISettingsViewProps): React.ReactElement {
  const [useDefault, setUseDefault] = React.useState(
    defaultsUnavailable !== null ? false : (config.useDefault ?? false)
  );
  const initialProvider = providers.some(p => p.id === config.provider)
    ? config.provider
    : providers[0]?.id || config.provider;
  const initialModels =
    providers.find(p => p.id === initialProvider)?.models || [];
  const [provider, setProvider] = React.useState(initialProvider);
  const [model, setModel] = React.useState(
    config.model || initialModels[0] || ''
  );
  const [apiKey, setApiKey] = React.useState(config.apiKey);
  const [openaiBaseUrl, setOpenaiBaseUrl] = React.useState(
    config.openaiBaseUrl || ''
  );
  const [bedrockRegion, setBedrockRegion] = React.useState(
    config.bedrockRegion || 'us-east-1'
  );
  const [customModels, setCustomModels] = React.useState<string[] | null>(null);
  const [fetchingModels, setFetchingModels] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  const initialModelsError =
    providers.find(p => p.id === initialProvider)?.modelsError || '';
  const [error, setError] = React.useState(initialModelsError);
  const hasOpenAICustomBaseUrl =
    !!openaiBaseUrl && !isOpenAIDefaultBaseUrl(openaiBaseUrl);
  const canFetchModels =
    provider === 'openai'
      ? !!(apiKey || hasOpenAICustomBaseUrl)
      : provider === 'anthropic'
        ? !!apiKey
        : provider === 'bedrock'
          ? !!apiKey
          : false;

  const loadModels = async () => {
    if (!canFetchModels) {
      return;
    }
    setFetchingModels(true);
    setError('');
    try {
      const fetched = await fetchProviderModels(
        provider,
        apiKey,
        openaiBaseUrl,
        provider === 'bedrock' ? bedrockRegion : undefined
      );
      setCustomModels(fetched);
      setModel(current =>
        fetched.includes(current) ? current : fetched[0] || ''
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch models');
    } finally {
      setFetchingModels(false);
    }
  };

  React.useEffect(() => {
    if (canFetchModels) {
      void loadModels();
    }
  }, []);

  const currentProvider =
    providers.find(p => p.id === provider) || providers[0];
  const models = customModels ? customModels : currentProvider?.models || [];

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    setCustomModels(null);
    setError('');
    const newProviderData = providers.find(p => p.id === newProvider);
    if (newProviderData && !newProviderData.models.includes(model)) {
      setModel(newProviderData.models[0] || '');
    }
  };

  const handleFetchModels = async () => {
    await loadModels();
  };

  const handleModelSourceBlur = () => {
    if (canFetchModels) {
      void loadModels();
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError('');
    try {
      const newConfig: IConfig = {
        provider,
        model,
        apiKey,
        useDefault,
        openaiBaseUrl:
          provider === 'openai' && openaiBaseUrl ? openaiBaseUrl : undefined,
        bedrockRegion: provider === 'bedrock' ? bedrockRegion : undefined
      };
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
      {defaultsUnavailable !== null && (
        <div className="jp-Mynerva-settings-error">{defaultsUnavailable}</div>
      )}
      {defaults && (
        <div className="jp-Mynerva-settings-field jp-Mynerva-settings-checkbox">
          <label>
            <input
              type="checkbox"
              checked={useDefault}
              onChange={e => setUseDefault(e.target.checked)}
            />
            Use default settings ({defaults.provider} / {defaults.model}
            {defaults.openaiBaseUrl && ` @ ${defaults.openaiBaseUrl}`}
            {defaults.bedrockRegion && ` @ ${defaults.bedrockRegion}`})
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
              {provider === 'openai' && (
                <div className="jp-Mynerva-settings-field">
                  <label>Base URL (optional)</label>
                  <input
                    type="text"
                    value={openaiBaseUrl}
                    onChange={e => {
                      setOpenaiBaseUrl(e.target.value);
                      setCustomModels(null);
                    }}
                    onBlur={handleModelSourceBlur}
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
              )}
              {provider === 'bedrock' && (
                <div className="jp-Mynerva-settings-field">
                  <label>AWS Region</label>
                  <select
                    value={bedrockRegion}
                    onChange={e => setBedrockRegion(e.target.value)}
                  >
                    {bedrockRegions.map(r => (
                      <option key={r.id} value={r.id}>
                        {r.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <div className="jp-Mynerva-settings-field">
                <label>API Key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={e => {
                    setApiKey(e.target.value);
                    setCustomModels(null);
                  }}
                  onBlur={handleModelSourceBlur}
                  placeholder="Enter API key"
                />
              </div>
              <div className="jp-Mynerva-settings-field">
                <label>Model</label>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <select
                    value={model}
                    onChange={e => setModel(e.target.value)}
                    style={{ flex: 1 }}
                  >
                    {models.map(m => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  {provider !== 'enki-gate' && (
                    <button
                      className="jp-Mynerva-settings-model-refresh"
                      onClick={handleFetchModels}
                      disabled={fetchingModels || !canFetchModels}
                      title="Fetch models"
                    >
                      <refreshIcon.react
                        tag="span"
                        width="16px"
                        height="16px"
                      />
                    </button>
                  )}
                </div>
              </div>
            </>
          )}
        </>
      )}
      {warning && !useDefault && (
        <div className="jp-Mynerva-settings-error">{warning}</div>
      )}
      {error && <div className="jp-Mynerva-settings-error">{error}</div>}
      {provider !== 'enki-gate' && (
        <button
          className="jp-Mynerva-settings-save"
          onClick={handleSave}
          disabled={saving || !model}
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
  'searchNotebooks',
  'summaryCellsFromSearch',
  'getCellsFromSearch',
  'listHelp',
  'help'
];
const MUTATE_ACTION_TYPES = [
  'insertCell',
  'updateCell',
  'deleteCell',
  'runCell'
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

/** Reconstruct an executable IAction from a native tool call. */
function toolCallToAction(toolCall: IToolCall): IAction {
  return { type: toolCall.name, ...toolCall.input } as IAction;
}

// How many times to feed validation errors back to the model before giving up
// and presenting the turn as-is.
const MAX_RETRIES = 2;

/**
 * Validate the model's turn before presenting it to the user. Returns the
 * problems to feed back to the model, or [] if the turn is acceptable. Add
 * rules here as needed; the retry loop is generic.
 */
function validateAssistantResult(result: IChatResult): string[] {
  const errors: string[] = [];
  if (result.toolCalls.length > 0 && result.text.trim() === '') {
    errors.push(
      'You proposed actions with no explanation. Tool calls are shown to the ' +
        'user as approval requests, so first state in natural language what ' +
        'you intend to do and why, then request the action(s) again.'
    );
  }
  return errors;
}

type QueryActionType = (typeof QUERY_ACTION_TYPES)[number];
type MutateActionType = 'insertCell' | 'updateCell' | 'deleteCell' | 'runCell';
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
  onActionCancel: () => void;
  actionProgress: string;
  onAcceptAll: (msgIndex: number) => void;
  onAcceptAllAlways: (msgIndex: number) => void;
  onRejectAll: (msgIndex: number) => void;
  getActionStatus: (msgIndex: number, actionIndex: number) => ActionStatus;
  loading: boolean;
  streamingContent: string;
  activeContentType: string;
  thinkingContent: string;
  stopReason: string;
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
  onActionCancel,
  actionProgress,
  onAcceptAll,
  onAcceptAllAlways,
  onRejectAll,
  getActionStatus,
  loading,
  streamingContent,
  activeContentType,
  thinkingContent,
  stopReason,
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
  }, [messages, loading, streamingContent]);

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
          const actions = (msg.toolCalls || []).map(toolCallToAction);
          const pendingCount = actions.filter(
            (_, i) => getActionStatus(msgIndex, i) === 'pending'
          ).length;
          const hasPending = pendingCount > 0;

          return (
            <React.Fragment key={msgIndex}>
              {/* Message (skip empty bubbles, e.g. tool-only assistant turns) */}
              {getDisplayContent(msg).trim() !== '' && (
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
              )}
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
                            onCancel={onActionCancel}
                            progress={actionProgress}
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
                            onCancel={onActionCancel}
                            progress={actionProgress}
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
            <div className="jp-Mynerva-message-content">
              {activeContentType === 'thinking' && !streamingContent && (
                <div className="jp-Mynerva-streaming-status">
                  Thinking...
                  {thinkingContent && (
                    <div className="jp-Mynerva-reasoning">
                      {thinkingContent}
                    </div>
                  )}
                </div>
              )}
              {activeContentType === 'text' && !streamingContent && (
                <div className="jp-Mynerva-streaming-status">Generating...</div>
              )}
              {streamingContent ? (
                <div className="jp-Mynerva-streamdown">
                  <Streamdown
                    animated
                    controls={{ code: false, table: false, mermaid: false }}
                  >
                    {streamingContent}
                  </Streamdown>
                </div>
              ) : (
                !activeContentType && '...'
              )}
              {stopReason &&
                (stopReason === 'max_tokens' || stopReason === 'length') && (
                  <div className="jp-Mynerva-stop-warning">
                    Response was truncated (max tokens reached)
                  </div>
                )}
            </div>
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
  const [bedrockRegions, setBedrockRegions] = React.useState<IBedrockRegion[]>(
    []
  );
  const [encryption, setEncryption] = React.useState(false);
  const [defaults, setDefaults] = React.useState<IDefaultConfig | null>(null);
  const [defaultsError, setDefaultsError] = React.useState<string | null>(null);
  const [defaultsOnly, setDefaultsOnly] = React.useState(false);
  const [nbsearchAvailable, setNbsearchAvailable] = React.useState(false);
  const [config, setConfig] = React.useState<IConfig | null>(null);
  const [showSettings, setShowSettings] = React.useState(false);
  const [messages, setMessages] = React.useState<IMessage[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [streamingContent, setStreamingContent] = React.useState('');
  const [activeContentType, setActiveContentType] = React.useState('');
  const [thinkingContent, setThinkingContent] = React.useState('');
  const [stopReason, setStopReason] = React.useState('');
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
  // Auto-approval for nbsearch queries: Map<query|referenceId, Set<actionType>>
  const [searchAutoApproved, setSearchAutoApproved] = React.useState<
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

  // AbortController for cancelling chat requests
  const abortControllerRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    Promise.all([getProviders(), getConfig(), getSessions()])
      .then(async ([providersRes, cfg, sessionsRes]) => {
        setProviders(providersRes.providers);
        setBedrockRegions(providersRes.bedrockRegions || []);
        setEncryption(providersRes.encryption);
        setDefaults(providersRes.defaults);
        if (providersRes.defaultsError) {
          setDefaultsError(providersRes.defaultsError);
        }
        setNbsearchAvailable(!!providersRes.nbsearchAvailable);
        if (providersRes.defaultsOnly) {
          setDefaultsOnly(true);
        }
        setConfig(cfg);
        setSessions(sessionsRes.sessions);
        setSessionLoadErrors(sessionsRes.errors);

        if (providersRes.defaultsOnly) {
          // Settings screen is not available in defaults-only mode
        } else if (cfg.decryptError || cfg.configWarning) {
          setShowSettings(true);
        } else {
          // Show settings if:
          // - no API key (and no base URL) and not using defaults, OR
          // - useDefault is set but defaults are not available
          const needsSettings =
            (cfg.useDefault && !providersRes.defaults) ||
            providersRes.defaultsError;
          const enkiGateValid =
            cfg.enkiGateToken &&
            cfg.enkiGateExpiresAt &&
            cfg.enkiGateExpiresAt > Date.now();
          const hasAuth = cfg.apiKey || cfg.openaiBaseUrl || enkiGateValid;
          if ((!hasAuth && !cfg.useDefault) || needsSettings) {
            setShowSettings(true);
          }
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

  // Queue of tool results waiting to be sent back to the LLM
  const [pendingResults, setPendingResults] = React.useState<IToolResult[]>([]);

  // Flag to prevent duplicate execution from useEffect during batch operations
  const executingActionsRef = React.useRef(false);
  // Aborts the action currently being executed (e.g. a slow nbsearch summary)
  const actionAbortRef = React.useRef<AbortController | null>(null);
  // Progress text for the action currently executing (shown on its card)
  const [actionProgress, setActionProgress] = React.useState('');

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
        const actions = msg.toolCalls || [];
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

  const executeQueryAction = async (
    action: IAction,
    signal?: AbortSignal,
    onProgress?: (event: Record<string, unknown>) => void
  ): Promise<string> => {
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
          {
            type: 'listHelp',
            result: buildSystemPrompt()
          },
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
      case 'searchNotebooks': {
        const searchResult = await callNBSearch(
          'notebooks',
          action,
          filterEnabled,
          signal,
          onProgress
        );
        result = JSON.stringify(
          { type: 'searchNotebooks', result: searchResult },
          null,
          2
        );
        break;
      }
      case 'summaryCellsFromSearch': {
        const searchResult = await callNBSearch(
          'summary-cells-from-search',
          action,
          filterEnabled,
          signal,
          onProgress
        );
        result = JSON.stringify(
          { type: 'summaryCellsFromSearch', result: searchResult },
          null,
          2
        );
        break;
      }
      case 'getCellsFromSearch': {
        const searchResult = await callNBSearch(
          'cells-from-search',
          action,
          filterEnabled,
          signal
        );
        result = JSON.stringify(
          { type: 'getCellsFromSearch', result: searchResult },
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
        const result = await contextEngine.runCell(action.query);
        return JSON.stringify({ type: 'runCell', result }, null, 2);
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
    (msg.toolCalls || []).some((_, actionIndex) => {
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

  const handleActionCancel = () => {
    actionAbortRef.current?.abort();
  };

  const executeApprovedAction = async (
    msgIndex: number,
    actionIndex: number,
    action: IAction,
    toolCallId: string,
    signal?: AbortSignal
  ) => {
    let toolResult: IToolResult;
    setActionProgress('');
    try {
      const result = isQueryAction(action)
        ? await executeQueryAction(action, signal, event =>
            setActionProgress(formatActionProgress(event))
          )
        : await executeMutateAction(action);
      toolResult = { id: toolCallId, result };
    } catch (e) {
      // User-initiated cancellation: record it and stop, don't surface as error.
      if (signal?.aborted || (e instanceof Error && e.name === 'AbortError')) {
        setPendingResults(prev => [
          ...prev,
          {
            id: toolCallId,
            result: JSON.stringify(
              { type: action.type, cancelled: true },
              null,
              2
            ),
            isError: true
          }
        ]);
        setActionStatus(msgIndex, actionIndex, 'cancelled');
        return;
      }
      console.error('Action failed:', action.type, e);
      toolResult = {
        id: toolCallId,
        result: JSON.stringify(
          {
            type: action.type,
            error: e instanceof Error ? e.message : 'Unknown error'
          },
          null,
          2
        ),
        isError: true
      };
    }

    setPendingResults(prev => [...prev, toolResult]);
    setActionStatus(msgIndex, actionIndex, 'executed');
  };

  const FILE_QUERY_TYPES = [
    'listNotebookFiles',
    'getTocFromFile',
    'getSectionFromFile',
    'getCellsFromFile',
    'getOutputFromFile'
  ];

  const SEARCH_QUERY_TYPES = [
    'searchNotebooks',
    'summaryCellsFromSearch',
    'getCellsFromSearch'
  ];

  const isFileQueryAction = (action: IAction): boolean => {
    return FILE_QUERY_TYPES.includes(action.type);
  };

  const isSearchQueryAction = (action: IAction): boolean => {
    return SEARCH_QUERY_TYPES.includes(action.type);
  };

  const getFileQueryTargetPath = (action: IAction): string => {
    if (action.type === 'listNotebookFiles') {
      return (action as IListNotebookFilesAction).path || '';
    }
    return (action as { path: string }).path;
  };

  const getSearchQueryKey = (action: IAction): string => {
    if (action.type === 'searchNotebooks') {
      return (action as ISearchNotebooksAction).query;
    }
    return (action as IGetCellsFromSearchAction | ISummaryCellsFromSearchAction)
      .referenceId;
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
    } else if (isSearchQueryAction(action)) {
      const key = getSearchQueryKey(action);
      setSearchAutoApproved(prev => {
        const newMap = new Map(prev);
        const types = newMap.get(key) ?? new Set<string>();
        types.add(action.type);
        newMap.set(key, types);
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

    if (isSearchQueryAction(action)) {
      const key = getSearchQueryKey(action);
      const approved = searchAutoApproved.get(key);
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
    const toolCalls = messages[msgIndex].toolCalls || [];

    for (let i = 0; i < toolCalls.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionApprove(msgIndex, i);
    }
  };

  const handleRejectAll = (msgIndex: number) => {
    const toolCalls = messages[msgIndex].toolCalls || [];

    for (let i = 0; i < toolCalls.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionReject(msgIndex, i);
    }
  };

  const handleAcceptAllAlways = (msgIndex: number) => {
    const toolCalls = messages[msgIndex].toolCalls || [];

    for (let i = 0; i < toolCalls.length; i++) {
      if (getActionStatus(msgIndex, i) !== 'pending') {
        continue;
      }
      handleActionApprove(msgIndex, i);
      addAutoApproval(toolCallToAction(toolCalls[i]));
    }
  };

  // Execute approved actions when all actions in a message are decided
  React.useEffect(() => {
    if (loading || executingActionsRef.current) {
      return;
    }

    const executeBatch = async () => {
      executingActionsRef.current = true;
      const controller = new AbortController();
      actionAbortRef.current = controller;
      try {
        for (let msgIndex = 0; msgIndex < messages.length; msgIndex++) {
          const toolCalls = messages[msgIndex].toolCalls || [];
          if (toolCalls.length === 0) {
            continue;
          }

          // Check if all tool calls are decided (not pending)
          const allDecided = toolCalls.every(
            (_, i) => getActionStatus(msgIndex, i) !== 'pending'
          );
          if (!allDecided) {
            continue;
          }

          // Check if any tool calls need processing
          const hasApproved = toolCalls.some(
            (_, i) => getActionStatus(msgIndex, i) === 'approved'
          );
          const hasRejected = toolCalls.some(
            (_, i) => getActionStatus(msgIndex, i) === 'rejected'
          );
          if (!hasApproved && !hasRejected) {
            continue;
          }

          // Process in order: execute approved, notify rejected. Every tool
          // call must get a result (provider pairing), so on cancellation the
          // remaining ones are answered with a cancelled result too.
          for (let i = 0; i < toolCalls.length; i++) {
            if (controller.signal.aborted) {
              for (let j = i; j < toolCalls.length; j++) {
                const st = getActionStatus(msgIndex, j);
                if (
                  st === 'executed' ||
                  st === 'notified' ||
                  st === 'cancelled'
                ) {
                  continue;
                }
                setPendingResults(prev => [
                  ...prev,
                  {
                    id: toolCalls[j].id,
                    result: JSON.stringify(
                      { type: toolCalls[j].name, cancelled: true },
                      null,
                      2
                    ),
                    isError: true
                  }
                ]);
                setActionStatus(msgIndex, j, 'cancelled');
              }
              break;
            }
            const status = getActionStatus(msgIndex, i);
            if (status === 'approved') {
              setActionStatus(msgIndex, i, 'executing');
              await executeApprovedAction(
                msgIndex,
                i,
                toolCallToAction(toolCalls[i]),
                toolCalls[i].id,
                controller.signal
              );
            } else if (status === 'rejected') {
              const result = JSON.stringify(
                { type: toolCalls[i].name, rejected: true },
                null,
                2
              );
              setPendingResults(prev => [
                ...prev,
                { id: toolCalls[i].id, result }
              ]);
              setActionStatus(msgIndex, i, 'notified');
            }
          }
        }
      } finally {
        executingActionsRef.current = false;
        actionAbortRef.current = null;
        setActionProgress('');
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
        content: '[Tool results]',
        toolResults: results,
        generated: true
      };
      const newMessages = [...messages, feedbackMessage];
      setMessages(newMessages);

      try {
        const finalMessages = await runChatWithRetry(newMessages);
        setMessages(finalMessages);
      } catch (e) {
        const errorMessage: IMessage = {
          role: 'assistant',
          content: `Error: ${e instanceof Error ? e.message : 'Unknown error'}`
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        clearStreamingState();
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
      const actions = (messages[msgIndex].toolCalls || []).map(
        toolCallToAction
      );
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
  }, [messages, autoApproved, fileAutoApproved, searchAutoApproved]);

  const clearStreamingState = () => {
    setStreamingContent('');
    setActiveContentType('');
    setThinkingContent('');
    setStopReason('');
  };

  const runChat = async (
    chatMessages: IMessage[],
    signal?: AbortSignal
  ): Promise<IChatResult> => {
    clearStreamingState();
    try {
      return await sendChat(
        chatMessages,
        getToolDefinitions({ nbsearchAvailable }),
        {
          onContentBlockStart: (ct: string) => setActiveContentType(ct),
          onContentBlockDelta: (ct: string, delta: string) => {
            // Deltas are incremental tokens now (the backend streams raw text),
            // so accumulate rather than replace.
            if (ct === 'text') {
              setStreamingContent(prev => prev + delta);
            } else if (ct === 'thinking') {
              setThinkingContent(prev => prev + delta);
            }
          },
          onContentBlockStop: (
            ct: string,
            metadata?: Record<string, unknown>
          ) => {
            if (ct === 'thinking' && metadata?.text) {
              setThinkingContent(metadata.text as string);
            }
          },
          onMessageDone: (result: IChatResult) => {
            if (result.stopReason) {
              setStopReason(result.stopReason);
            }
          }
        },
        signal
      );
    } finally {
      clearStreamingState();
    }
  };

  const buildAssistantMessage = (result: IChatResult): IMessage => ({
    role: 'assistant',
    // Empty for tool-only turns (the bubble is hidden); only fall back to a
    // placeholder when there is no text and no tool calls at all.
    content: result.text || (result.toolCalls.length > 0 ? '' : '(no message)'),
    toolCalls: result.toolCalls.length > 0 ? result.toolCalls : undefined,
    assistantBlocks: result.assistantBlocks
  });

  // Feed validation errors back to the model. Tool calls must be answered with
  // tool results (provider pairing), so the errors ride on those; otherwise a
  // plain generated user message carries them.
  const buildRetryFeedback = (
    result: IChatResult,
    errors: string[]
  ): IMessage => {
    const text = errors.join('\n');
    if (result.toolCalls.length > 0) {
      return {
        role: 'user',
        content: '[Retry requested]',
        toolResults: result.toolCalls.map(tc => ({
          id: tc.id,
          result: text,
          isError: true
        })),
        generated: true
      };
    }
    return { role: 'user', content: text, generated: true };
  };

  // Run one chat turn, feeding validation errors back to the model and retrying
  // (bounded) until the turn is acceptable. Returns the message list to commit:
  // only the final accepted assistant turn, not the intermediate correction
  // round-trips (which are sent to the provider in-loop but never committed, so
  // the approval machinery never sees a rejected turn's tool calls). The loop
  // runs while `loading` is true, so the approval effects stay dormant.
  const runChatWithRetry = async (
    history: IMessage[],
    signal?: AbortSignal
  ): Promise<IMessage[]> => {
    const withSystem = (msgs: IMessage[]): IMessage[] => [
      { role: 'system', content: buildSystemPrompt() },
      ...msgs
    ];

    let apiMessages = history;
    let result = await runChat(withSystem(apiMessages), signal);
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      const errors = validateAssistantResult(result);
      if (errors.length === 0) {
        break;
      }
      apiMessages = [
        ...apiMessages,
        buildAssistantMessage(result),
        buildRetryFeedback(result, errors)
      ];
      result = await runChat(withSystem(apiMessages), signal);
    }
    return [...history, buildAssistantMessage(result)];
  };

  const handleCancelLoading = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    // Remove the last user message
    setMessages(prev => prev.slice(0, -1));
    clearStreamingState();
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
      const finalMessages = await runChatWithRetry(
        newMessages,
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
      clearStreamingState();
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
          {!defaultsOnly && (
            <button
              className="jp-Mynerva-header-button"
              onClick={() => setShowSettings(!showSettings)}
              title="Settings"
            >
              <settingsIcon.react tag="span" />
            </button>
          )}
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
          bedrockRegions={bedrockRegions}
          encryption={encryption}
          defaults={defaults}
          defaultsUnavailable={
            defaultsError
              ? `Default settings unavailable: ${defaultsError}`
              : config?.useDefault && !defaults
                ? 'Default settings are no longer available. Please configure your own API key.'
                : null
          }
          onSave={handleConfigSave}
          warning={config?.configWarning || config?.decryptError}
        />
      ) : (
        <ChatView
          messages={messages}
          onSendMessage={handleSendMessage}
          onActionApprove={handleActionApprove}
          onActionApproveAlways={handleActionApproveAlways}
          onActionReject={handleActionReject}
          onActionCancel={handleActionCancel}
          actionProgress={actionProgress}
          onAcceptAll={handleAcceptAll}
          onAcceptAllAlways={handleAcceptAllAlways}
          onRejectAll={handleRejectAll}
          getActionStatus={getActionStatus}
          loading={loading}
          streamingContent={streamingContent}
          activeContentType={activeContentType}
          thinkingContent={thinkingContent}
          stopReason={stopReason}
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
  shell: JupyterFrontEnd.IShell,
  contextEngine: ContextEngine,
  liveQuery: NblibramLiveQuery
): MynervaPanel {
  const panel = new MynervaPanel(contextEngine, liveQuery);
  shell.add(panel, 'right', { rank: 1000 });
  return panel;
}
