import {
  ILabShell,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { INotebookTracker } from '@jupyterlab/notebook';

import { activatePanel } from './panel';
import { ContextEngine } from './context';
import { NblibramLiveQuery } from './nblibram';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-mynerva:plugin',
  description:
    'A JupyterLab extension that provides an LLM-powered assistant with deep understanding of notebook structure.',
  autoStart: true,
  requires: [ILabShell, INotebookTracker],
  optional: [ISettingRegistry],
  activate: (
    app: JupyterFrontEnd,
    shell: ILabShell,
    notebookTracker: INotebookTracker,
    settingRegistry: ISettingRegistry | null
  ) => {
    console.log('JupyterLab extension jupyter-mynerva is activated!');

    const contextEngine = new ContextEngine(notebookTracker);
    const liveQuery = new NblibramLiveQuery(notebookTracker);
    activatePanel(shell, contextEngine, liveQuery);

    if (settingRegistry) {
      settingRegistry
        .load(plugin.id)
        .then(settings => {
          console.log('jupyter-mynerva settings loaded:', settings.composite);
        })
        .catch(reason => {
          console.error('Failed to load settings for jupyter-mynerva.', reason);
        });
    }
  }
};

export default plugin;
