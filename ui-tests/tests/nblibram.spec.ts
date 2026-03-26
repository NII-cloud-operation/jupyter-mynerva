import { expect, test } from '@jupyterlab/galata';

test.describe('NblibramHandler', () => {
  test.beforeEach(async ({ page }) => {
    await page.notebook.createNew('test.ipynb');
    await page.notebook.setCell(0, 'markdown', '# Hello World');
    await page.notebook.addCell('code', 'x = 1');
    await page.notebook.addCell('markdown', '## Section Two');
    await page.notebook.addCell('code', 'y = 2');
    await page.notebook.save();
  });

  test('toc returns headings', async ({ page, tmpPath }) => {
    const response = await page.evaluate(async (path: string) => {
      const settings = (window as any).jupyterapp.serviceManager.serverSettings;
      const res = await fetch(`${settings.baseUrl}jupyter-mynerva/nblibram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'toc',
          path: path,
          format: 'json'
        })
      });
      return { status: res.status, body: await res.json() };
    }, tmpPath + '/test.ipynb');

    expect(response.status).toBe(200);
    const cells = response.body.cells;
    expect(cells).toBeInstanceOf(Array);
    expect(cells.length).toBe(2);
    expect(cells[0].source).toContain('# Hello World');
    expect(cells[1].source).toContain('## Section Two');
  });

  test('cells returns matched cells', async ({ page, tmpPath }) => {
    const response = await page.evaluate(async (path: string) => {
      const settings = (window as any).jupyterapp.serviceManager.serverSettings;
      const res = await fetch(`${settings.baseUrl}jupyter-mynerva/nblibram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'cells',
          path: path,
          format: 'json',
          query: 'start:1',
          count: 1
        })
      });
      return { status: res.status, body: await res.json() };
    }, tmpPath + '/test.ipynb');

    expect(response.status).toBe(200);
    const cells = response.body.cells;
    expect(cells).toBeInstanceOf(Array);
    expect(cells[0].source).toContain('x = 1');
    expect(cells[0].cell_type).toBe('code');
  });

  test('invalid command returns 400', async ({ page }) => {
    const response = await page.evaluate(async () => {
      const settings = (window as any).jupyterapp.serviceManager.serverSettings;
      const res = await fetch(`${settings.baseUrl}jupyter-mynerva/nblibram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'nonexistent-command',
          path: 'dummy.ipynb'
        })
      });
      return { status: res.status, body: await res.json() };
    });

    expect(response.status).toBe(400);
    expect(response.body.error).toBeTruthy();
  });

  test('dirty sync stores notebook content', async ({ page }) => {
    const notebookContent = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {},
      cells: [
        {
          cell_type: 'markdown',
          source: '# From dirty sync',
          metadata: {}
        },
        {
          cell_type: 'code',
          source: 'z = 42',
          metadata: {}
        }
      ]
    };

    const response = await page.evaluate(
      async ({ content, path }) => {
        const settings = (window as any).jupyterapp.serviceManager
          .serverSettings;
        const res = await fetch(`${settings.baseUrl}jupyter-mynerva/nblibram`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command: 'toc',
            path: path,
            format: 'json',
            notebookContent: content
          })
        });
        return { status: res.status, body: await res.json() };
      },
      { content: notebookContent, path: 'dirty-test.ipynb' }
    );

    expect(response.status).toBe(200);
    const cells = response.body.cells;
    expect(cells).toBeInstanceOf(Array);
    expect(cells.length).toBe(1);
    expect(cells[0].source).toContain('# From dirty sync');
  });

  test('filters mask IP addresses by default', async ({ page }) => {
    const notebookContent = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {},
      cells: [
        {
          cell_type: 'code',
          source: 'server = "192.168.1.1"',
          metadata: {},
          outputs: []
        }
      ]
    };

    const response = await page.evaluate(
      async ({ content, path }) => {
        const settings = (window as any).jupyterapp.serviceManager
          .serverSettings;
        const res = await fetch(`${settings.baseUrl}jupyter-mynerva/nblibram`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command: 'cells',
            path: path,
            format: 'json',
            query: 'start:0',
            count: 1,
            notebookContent: content
          })
        });
        return { status: res.status, body: await res.json() };
      },
      { content: notebookContent, path: 'filter-test.ipynb' }
    );

    expect(response.status).toBe(200);
    const source = response.body.cells[0].source.join('');
    expect(source).not.toContain('192.168.1.1');
    expect(source).toContain('[ipv4-address_');
  });
});
