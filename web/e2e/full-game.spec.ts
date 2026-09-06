import { test, expect, type Page, type Locator } from "@playwright/test";

// The single most important test in the suite: it proves the whole stack
// (Rust server, WebSocket protocol, generated types, React UI) works together
// by playing an entire game of 7 Wonders Duel from the home screen to the
// end screen, driven only through rendered elements a real person could
// click - never by calling an internal API.
//
// The "policy" is deliberately dumb (always take the first available option),
// which is fine: the goal is coverage of every phase the UI has to render
// (wonder draft, ordinary turns, forced mid-turn choices, the end screen),
// not to play well.
//
// The table plays each move back as a sequence (take -> resolve -> reveal ->
// hand over) and disables the controls while it does, so the loop below waits
// for an interactive control to reappear rather than clicking blind.

/** Turn the table's animations down through the real settings menu. Every
 * move still plays its full take -> resolve -> reveal -> hand-over sequence;
 * this only shortens each step, so a whole game fits in a test budget. */
async function useFastAnimations(page: Page): Promise<void> {
  await page.getByTestId("menu").click();
  await page.getByTestId("settings").getByLabel("Animation speed").selectOption("fast");
  await page.getByTestId("settings").getByRole("button", { name: "Close" }).click();
}

/** A click that tolerates the target disappearing under it: the board is
 * re-rendered by WebSocket pushes, so the exact element a move was about to
 * be made on can legitimately vanish because that move just got applied. */
async function tryClick(locator: Locator): Promise<boolean> {
  if (!(await locator.isVisible().catch(() => false))) return false;
  try {
    await locator.click({ timeout: 5_000 });
    return true;
  } catch {
    return false;
  }
}

/** One decision, made through the UI exactly as a person would: pick the
 * first thing that is offered, in whichever phase the table is in. */
async function takeATurn(page: Page): Promise<boolean> {
  // A forced mid-turn choice (progress token, Great Library, destroy,
  // Mausoleum, or who starts the next age) takes over the action tray.
  if (await tryClick(page.locator("[data-testid^='choice-']:not([disabled])").first())) return true;

  // The wonder draft.
  if (await tryClick(page.locator("[data-testid^='wonder-']:not([disabled])").first())) return true;

  // An ordinary turn: select an accessible card in the structure, then use
  // the action tray's own buttons.
  if (await tryClick(page.locator(".pyr .card.acc.clickable").first())) {
    const build = page.getByTestId("act-build");
    if (await build.isEnabled().catch(() => false)) return await tryClick(build);
    return await tryClick(page.getByTestId("act-discard"));
  }
  return false;
}

test("plays a full game against the random bot, from the home screen to the end screen", async ({ page }) => {
  test.setTimeout(180_000);

  await page.goto("/");
  await page.getByTestId("start-vs-bot").click();

  // The table (and its first server-pushed state) has loaded.
  await expect(page.getByTestId("log")).toBeVisible({ timeout: 20_000 });
  await useFastAnimations(page);

  // Everything the design promises is on screen at once, from the very first
  // phase: the shared board rail, both city strips, and the log.
  await expect(page.locator(".rail")).toBeVisible();
  await expect(page.locator(".strip")).toHaveCount(2);
  await expect(page.getByTestId("draft")).toBeVisible();

  for (let i = 0; i < 900; i++) {
    if (await page.getByTestId("end-screen").isVisible().catch(() => false)) break;
    if (await takeATurn(page)) continue;
    // Nothing to act on yet: the opponent is thinking, or a move is still
    // playing back.
    await page.waitForTimeout(120);
  }

  // The end screen, reached through real play: winner, condition and the
  // per-category breakdown.
  const end = page.getByTestId("end-screen");
  await expect(end).toBeVisible({ timeout: 20_000 });
  await expect(end.getByText(/wins? on|it's a draw/i)).toBeVisible();
  await expect(end.getByText("Civilian buildings")).toBeVisible();
  await expect(end.getByText("Total", { exact: true })).toBeVisible();

  // Post-game review renders an earlier position over the same table.
  await end.getByRole("button", { name: "Open review" }).click();
  await expect(page.locator(".reviewband")).toContainText("Reviewing move");
  await page.getByTestId("return-to-live").click();
  await expect(page.locator(".reviewband")).toHaveCount(0);

  await page.getByTestId("leave-game").click();
  await expect(page.getByTestId("start-vs-bot")).toBeVisible();
});

test("shows each player's own cost for the same card, and switches the lens", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("start-vs-bot").click();
  await expect(page.getByTestId("log")).toBeVisible({ timeout: 20_000 });

  await useFastAnimations(page);

  // Play past the draft so there is a structure with real costs on it.
  for (let i = 0; i < 200; i++) {
    if (await page.locator(".pyr .card.acc.clickable").first().isVisible().catch(() => false)) break;
    if (!(await takeATurn(page))) await page.waitForTimeout(120);
  }
  await expect(page.locator(".pyr .card").first()).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("tray")).toBeVisible();

  // Selecting a card fills the tray with *both* players' costs at once.
  await page.locator(".pyr .card.acc.clickable").first().click();
  const tray = page.getByTestId("tray");
  await expect(tray.locator(".costbox")).toHaveCount(2);
  await expect(tray.locator(".costbox.you h5")).toContainText("Cost for");
  await expect(tray.locator(".costbox.opp h5")).toContainText("Cost for");

  // The lens switches which player's numbers every card in the structure
  // shows. Both settings must render a net badge on every face-up card.
  const badges = page.locator(".pyr .card:not(.back) .cf .bd");
  const before = await badges.allInnerTexts();
  await page.getByTestId("lens-two").click();
  const after = await badges.allInnerTexts();
  expect(before.length).toBeGreaterThan(0);
  expect(after.length).toBe(before.length);
});

test("hot-seat keeps fixed seats and names both players in the log", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("start-hotseat").click();
  await expect(page.getByTestId("log")).toBeVisible({ timeout: 20_000 });

  await expect(page.getByTestId("lens-one")).toHaveText("Player 1");
  await expect(page.getByTestId("lens-two")).toHaveText("Player 2");

  // Player 1 is always the bottom strip and Player 2 the top one, whoever is
  // on move.
  await expect(page.locator(".strip").last()).toContainText("Player 1");
  await expect(page.locator(".strip").first()).toContainText("Player 2");

  // A few draft picks land in the log with the acting player's name.
  await useFastAnimations(page);
  for (let i = 0; i < 30; i++) {
    if (await page.getByTestId("log").textContent().then((t) => (t ?? "").includes("drafted"))) break;
    if (!(await takeATurn(page))) await page.waitForTimeout(150);
  }
  await expect(page.getByTestId("log")).toContainText("drafted");
});
