import { expect, test } from '@jupyterlab/galata';

test.describe('jupyter-mynerva extension', () => {
  test('should show the Mynerva tab in the right sidebar', async ({ page }) => {
    const tab = page.getByRole('tab', { name: 'Mynerva' });
    await expect(tab).toBeVisible();
  });

  test('should open the Mynerva panel', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const panel = page.locator('#mynerva-panel');
    await expect(panel).toBeVisible();
  });

  test('should toggle settings view', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const panel = page.locator('#mynerva-panel');
    const settingsButton = panel.locator('button[title="Settings"]');
    await settingsButton.click();
    await expect(panel.locator('.jp-Mynerva-settings')).toBeVisible();
    await settingsButton.click();
    await expect(panel.locator('.jp-Mynerva-settings')).not.toBeVisible();
  });
});
