// Small grammar helpers, so seat names read naturally whether they are
// "You", "Opponent" or "Player 2".

export function possessive(name: string): string {
  if (name === "You") return "your";
  return `${name}'s`;
}

/** "You win" / "Opponent wins". */
export function verb(name: string, thirdPerson: string, secondPerson: string): string {
  return name === "You" ? secondPerson : thirdPerson;
}
