import { test, expect } from "@playwright/test";

// This is the single most important test in M2: it proves the whole stack
// (Rust server, WebSocket protocol, generated types, React UI) works
// together by playing an entire game of 7 Wonders Duel from the home screen
// to the game-over screen, driven only through rendered elements a real
// person could click - never by calling an internal API.
//
// The "policy" here is deliberately dumb (always click the first available
// option), which is fine: the goal is coverage of every phase the UI has to
// render (wonder draft, ordinary turns, pending-choice modals, game over),
// not to play well.

test("plays a full game against the random bot from the home screen to game over", async ({ page }) => {
  test.setTimeout(120_000);

  await page.goto("/");
  await page.getByTestId("start-vs-bot").click();

  // The game screen (and its first server-pushed state) has loaded.
  await expect(page.getByTestId("nav-leave-game")).toBeVisible({ timeout: 15_000 });

  const maxSteps = 600;
  for (let i = 0; i < maxSteps; i++) {
    if (await page.getByTestId("leave-game").isVisible()) {
      break;
    }

    // A modal is open: either the Build/Discard/BuildWonder action menu for
    // a slot just clicked, or a pending-effect choice (progress token, the
    // Great Library, a destroy effect, the Mausoleum, or who begins the
    // next age). Every button in it except the close (X) is a legal action.
    const modalButton = page.locator('[data-testid="modal"] button:not([data-testid="modal-close"])').first();
    if (await modalButton.isVisible()) {
      await modalButton.click();
      await page.waitForTimeout(30);
      continue;
    }

    // The wonder draft: pick whichever offered wonder is still pickable.
    const offeredWonder = page.locator('[data-testid^="wonder-"]:not([disabled])').first();
    if (await offeredWonder.isVisible()) {
      await offeredWonder.click();
      continue;
    }

    // An ordinary turn: click an accessible (highlighted, enabled) card in
    // the structure, which opens the action menu handled above.
    const accessibleSlot = page.locator('[data-testid^="slot-"] button:not([disabled])').first();
    if (await accessibleSlot.isVisible()) {
      await accessibleSlot.click();
      continue;
    }

    // Nothing clickable yet (e.g. the server is still resolving an agent's
    // turn) - give the WebSocket message a moment to arrive.
    await page.waitForTimeout(100);
  }

  // Reached game over via the rendered breakdown/banner, not by inspecting
  // any internal state.
  await expect(page.getByTestId("leave-game")).toBeVisible();
  await expect(page.getByText(/wins!|it's a draw!/i)).toBeVisible();

  await page.getByTestId("leave-game").click();
  await expect(page.getByTestId("start-vs-bot")).toBeVisible();
});
