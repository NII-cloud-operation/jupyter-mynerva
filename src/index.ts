import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILabShell
} from '@jupyterlab/application';

import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { INotebookTracker } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';
import { MessageLoop } from '@lumino/messaging';

import { activatePanel } from './panel';
import { ContextEngine } from './context';
import { NblibramLiveQuery } from './nblibram';
import { mynervaIcon } from './icons';

const TOGGLE_COMMAND = 'mynerva:toggle';

/**
 * Floating button anchored to the right viewport edge.
 *
 * JupyterLab shows an always-visible Mynerva icon in its right side tab bar,
 * but Notebook 7's shell has no such tab bar — the panel can only be reached
 * via View > Right Sidebar > Show Mynerva. This recreates the one-click
 * affordance as a viewport overlay, so it works uniformly on every Notebook 7
 * page (tree / notebook / console / terminal) with a single implementation.
 */
function addFloatingToggle(app: JupyterFrontEnd, panel: Widget): void {
  const button = document.createElement('button');
  button.className = 'jp-Mynerva-floatingToggle';
  button.title = 'Show Mynerva';
  button.setAttribute('aria-label', 'Show Mynerva');
  button.innerHTML = mynervaIcon.svgstr;
  button.addEventListener('click', () => {
    void app.commands.execute(TOGGLE_COMMAND);
  });
  document.body.appendChild(button);

  // Hide the handle while the panel is open (covers both the button-driven and
  // the menu-driven open/close paths), show it again once collapsed.
  const sync = (): void => {
    button.style.display = panel.isVisible ? 'none' : 'flex';
  };
  MessageLoop.installMessageHook(panel, (_handler, msg) => {
    if (msg.type === 'after-show' || msg.type === 'after-hide') {
      window.setTimeout(sync, 0);
    }
    return true;
  });
  sync();
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-mynerva:plugin',
  description:
    'A JupyterLab extension that provides an LLM-powered assistant with deep understanding of notebook structure.',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ISettingRegistry, ILabShell],
  activate: (
    app: JupyterFrontEnd,
    notebookTracker: INotebookTracker,
    settingRegistry: ISettingRegistry | null,
    labShell: ILabShell | null
  ) => {
    console.log('JupyterLab extension jupyter-mynerva is activated!');

    const contextEngine = new ContextEngine(notebookTracker);
    const liveQuery = new NblibramLiveQuery(notebookTracker);
    const panel = activatePanel(app.shell, contextEngine, liveQuery);

    app.commands.addCommand(TOGGLE_COMMAND, {
      label: 'Show Mynerva',
      icon: mynervaIcon,
      execute: () => {
        if (panel.isVisible) {
          // NotebookShell exposes collapseRight(); JupyterLab's shell too.
          const shell = app.shell as unknown as { collapseRight?: () => void };
          shell.collapseRight?.();
        } else {
          app.shell.activateById(panel.id);
        }
      }
    });

    // Notebook 7 has no right side tab bar; add a floating edge handle.
    // In JupyterLab (labShell present) the tab bar already provides the icon.
    if (!labShell) {
      addFloatingToggle(app, panel);
    }

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
