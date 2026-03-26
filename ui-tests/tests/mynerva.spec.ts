import { expect, test } from '@jupyterlab/galata';

test.describe('jupyter-mynerva extension', () => {
  test('should show the Mynerva tab in the right sidebar', async ({
    page
  }) => {
    const tab = page.getByRole('tab', { name: 'Mynerva' });
    await expect(tab).toBeVisible();
  });

  test('should open the Mynerva panel', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const panel = page.locator('#mynerva-panel');
    await expect(panel).toBeVisible();
  });

  test('should show settings view initially', async ({ page }) => {
    await page.getByRole('tab', { name: 'Mynerva' }).click();
    const panel = page.locator('#mynerva-panel');
    // Settings view should be present (provider selection)
    await expect(panel.locator('.jp-Mynerva-settings')).toBeVisible();
  });
});
