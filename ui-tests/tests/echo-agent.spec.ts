import { expect, test } from '@jupyterlab/galata';

test.describe('Echo Agent', () => {
  test.beforeEach(async ({ page }) => {
    await page.notebook.createNew('test.ipynb');
    await page.notebook.setCell(0, 'markdown', '# Hello World');
    await page.notebook.addCell('code', 'x = 1');
    await page.notebook.addCell('markdown', '## Section Two');
    await page.notebook.addCell('code', 'y = 2');
    await page.notebook.save();
  });

  async function sendAndApprove(
    page: any,
    message: string,
    expectedActionType: string
  ): Promise<string> {
    const panel = page.locator('#mynerva-panel');

    // Open panel if not yet open
    if (!(await panel.isVisible())) {
      await page.getByRole('tab', { name: 'Mynerva' }).click();
    }
    await expect(panel).toBeVisible();

    const input = panel.locator('.jp-Mynerva-input');
    await expect(input).toBeVisible({ timeout: 10000 });
    await input.fill(message);
    await panel.locator('.jp-Mynerva-send').click();

    // Verify action type
    const actionCard = panel.locator('.jp-Mynerva-action-card');
    await expect(actionCard).toBeVisible({ timeout: 10000 });
    await expect(actionCard.locator('.jp-Mynerva-action-type')).toHaveText(
      expectedActionType
    );

    // Approve
    const shareButton = actionCard.getByText('Share');
    await shareButton.click();

    // Wait for echo response (the last assistant message after action results)
    // The echo agent echoes back the [Action Results] content
    const allMessages = panel.locator(
      '.jp-Mynerva-message.jp-Mynerva-assistant .jp-Mynerva-message-content'
    );
    // Wait for at least 2 assistant messages (first = "Echo: requesting ...", second = echo of results)
    await expect(allMessages).toHaveCount(2, { timeout: 15000 });
    const lastMessage = allMessages.last();
    return await lastMessage.textContent();
  }

  test('getToc returns headings only', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const result = await sendAndApprove(page, 'show toc', 'getToc');
    expect(result).toContain('Hello World');
    expect(result).toContain('Section Two');
    expect(result).not.toContain('x = 1');
  });

  test('getCells returns count-limited cells', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const result = await sendAndApprove(page, 'get cells', 'getCells');
    expect(result).toContain('Hello World');
    expect(result).toContain('x = 1');
    expect(result).not.toContain('Section Two');
  });

  test('getSection returns full section', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const result = await sendAndApprove(page, 'get section', 'getSection');
    expect(result).toContain('Hello World');
    expect(result).toContain('x = 1');
    expect(result).toContain('Section Two');
    expect(result).toContain('y = 2');
  });
});
