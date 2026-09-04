//! Victory conditions and final scoring.
//!
//! There are three ways a game of 7 Wonders Duel ends (see
//! `docs/rules-spec.md` R-080..R-092):
//!
//! * **Military supremacy** — a player pushes the conflict pawn all the way
//!   to their opponent's capital (distance 9). The game stops at once and
//!   nothing else is scored.
//! * **Scientific supremacy** — a player gathers 6 *distinct* scientific
//!   symbols. Again the game stops at once.
//! * **Civilian victory** — Age III's structure empties, and the players
//!   compare victory points.
//!
//! The victory-point categories are: blue (civilian) cards, green
//! (scientific) cards, yellow (commercial) cards, purple (guild) cards,
//! wonders, progress tokens, the military track, and `floor(coins / 3)`.
//!
//! **Green cards score only the victory points printed on them.** There is no
//! per-symbol or per-set bonus in Duel — that is a 7 Wonders (base game)
//! rule. Identical symbol pairs are rewarded during play, with progress
//! tokens, not at scoring time.
//!
//! Guild cards score `points_per_unit × max(my count, opponent's count)` from
//! the board *as it stands at the end of the game*, which may differ from the
//! count that determined their immediate coin payout when they were built.

use serde::{Deserialize, Serialize};

use crate::data::{self, CardType, CountTarget};
use crate::state::{iter_mask_u128, GameState};
use crate::Player;

/// How a game was won.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VictoryKind {
    /// The conflict pawn reached the loser's capital.
    MilitarySupremacy,
    /// Six distinct scientific symbols.
    ScientificSupremacy,
    /// Most victory points at the end of Age III.
    CivilianVictory,
    /// Tied on victory points, decided by civilian (blue) points.
    CivilianTiebreak,
}

/// The outcome of a finished game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameResult {
    /// One player won.
    Win {
        /// The winner.
        winner: Player,
        /// How they won.
        kind: VictoryKind,
    },
    /// Equal victory points *and* equal civilian points.
    Draw,
}

impl GameResult {
    /// The winner, or `None` for a draw.
    #[inline]
    pub fn winner(&self) -> Option<Player> {
        match self {
            GameResult::Win { winner, .. } => Some(*winner),
            GameResult::Draw => None,
        }
    }

    /// Whether this outcome short-circuited the game before Age III ended,
    /// in which case no victory-point breakdown was computed.
    #[inline]
    pub fn is_instant(&self) -> bool {
        matches!(
            self,
            GameResult::Win {
                kind: VictoryKind::MilitarySupremacy | VictoryKind::ScientificSupremacy,
                ..
            }
        )
    }
}

/// One player's victory points, per category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Breakdown {
    /// Blue cards.
    pub civilian: u16,
    /// Green cards (printed points only).
    pub scientific: u16,
    /// Yellow cards.
    pub commercial: u16,
    /// Purple cards, recomputed from the final board.
    pub guilds: u16,
    /// Constructed wonders.
    pub wonders: u16,
    /// Progress tokens, including Mathematics' per-token bonus.
    pub progress_tokens: u16,
    /// The military track.
    pub military: u16,
    /// `floor(coins / 3)`.
    pub coins: u16,
    /// Sum of every category.
    pub total: u16,
}

/// Victory points a player scores from cards of one colour.
fn card_points(state: &GameState, player: Player, kind: CardType) -> u16 {
    let s = data::statics();
    let mask = state.player(player).built_mask() & s.card_masks[kind.index()];
    iter_mask_u128(mask)
        .map(|c| u16::from(c.def().victory_points))
        .sum()
}

/// The count both players are compared on for a guild / majority effect.
///
/// The rule is "whoever has the most", and the guild's owner is paid on that
/// higher number whether or not it is their own.
pub fn majority_count(state: &GameState, target: CountTarget) -> u16 {
    state
        .player(Player::One)
        .count(target)
        .max(state.player(Player::Two).count(target))
}

/// The full victory-point breakdown for `player`, computed from the state as
/// it stands right now.
pub fn breakdown(state: &GameState, player: Player) -> Breakdown {
    let me = state.player(player);
    let s = data::statics();

    let mut b = Breakdown {
        civilian: card_points(state, player, CardType::Civilian),
        scientific: card_points(state, player, CardType::Scientific),
        commercial: card_points(state, player, CardType::Commercial),
        ..Default::default()
    };

    // Guild cards: printed points (none carry any) plus the majority effect,
    // recomputed from the final board.
    let guild_mask = me.built_mask() & s.guild_mask;
    for card in iter_mask_u128(guild_mask) {
        let def = card.def();
        b.guilds += u16::from(def.victory_points);
        if let Some((target, per)) = def.points_by_majority {
            b.guilds += majority_count(state, target) * u16::from(per);
        }
    }

    for w in me.wonders_built() {
        b.wonders += u16::from(w.def().victory_points);
    }

    let token_count = u16::from(me.token_count());
    for t in me.tokens() {
        let def = t.def();
        b.progress_tokens += u16::from(def.victory_points);
        b.progress_tokens += u16::from(def.vp_per_token) * token_count;
    }

    if let Some(leader) = state.military_leader() {
        if leader == player {
            let distance = state.conflict().unsigned_abs();
            b.military = u16::from(data::military().vp_for_distance(distance));
        }
    }

    b.coins = me.coins() / 3;

    b.total = b.civilian
        + b.scientific
        + b.commercial
        + b.guilds
        + b.wonders
        + b.progress_tokens
        + b.military
        + b.coins;
    b
}

/// Both players' breakdowns, indexed by [`Player::index`].
pub fn score(state: &GameState) -> [Breakdown; 2] {
    [breakdown(state, Player::One), breakdown(state, Player::Two)]
}

/// Decide a civilian victory from the current board.
///
/// Called when Age III's structure empties. Ties on total points are broken
/// on civilian (blue) points; if those are equal too the game is a true draw.
pub fn civilian_result(state: &GameState) -> GameResult {
    let [a, b] = score(state);
    match a.total.cmp(&b.total) {
        std::cmp::Ordering::Greater => GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::CivilianVictory,
        },
        std::cmp::Ordering::Less => GameResult::Win {
            winner: Player::Two,
            kind: VictoryKind::CivilianVictory,
        },
        std::cmp::Ordering::Equal => match a.civilian.cmp(&b.civilian) {
            std::cmp::Ordering::Greater => GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::CivilianTiebreak,
            },
            std::cmp::Ordering::Less => GameResult::Win {
                winner: Player::Two,
                kind: VictoryKind::CivilianTiebreak,
            },
            std::cmp::Ordering::Equal => GameResult::Draw,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CardId, TokenId, WonderId};
    use crate::testing::StateBuilder;

    #[test]
    fn coins_score_one_point_per_three_with_a_floor() {
        for (coins, want) in [(0, 0), (2, 0), (3, 1), (5, 1), (6, 2), (17, 5)] {
            let st = StateBuilder::new().coins(Player::One, coins).build();
            assert_eq!(breakdown(&st, Player::One).coins, want, "coins = {coins}");
        }
    }

    #[test]
    fn green_cards_score_only_their_printed_points() {
        // Four green cards worth 1 + 2 + 3 + 3 = 9, covering four distinct
        // symbols plus a doubled one. Duel has no set bonus, so the total is
        // exactly 9.
        let st = StateBuilder::new()
            .built(Player::One, &["workshop", "library", "academy", "study"])
            .build();
        let b = breakdown(&st, Player::One);
        assert_eq!(b.scientific, 9);
        assert_eq!(b.total, 9);
    }

    #[test]
    fn military_points_go_only_to_the_leading_player() {
        for (conflict, p1, p2) in [
            (0i8, 0u16, 0u16),
            (1, 2, 0),
            (-2, 0, 2),
            (4, 5, 0),
            (-6, 0, 10),
            (8, 10, 0),
        ] {
            let st = StateBuilder::new().conflict(conflict).build();
            assert_eq!(breakdown(&st, Player::One).military, p1, "at {conflict}");
            assert_eq!(breakdown(&st, Player::Two).military, p2, "at {conflict}");
        }
    }

    #[test]
    fn mathematics_counts_itself() {
        let st = StateBuilder::new()
            .tokens(Player::One, &["mathematics"])
            .build();
        assert_eq!(breakdown(&st, Player::One).progress_tokens, 3);

        let st = StateBuilder::new()
            .tokens(Player::One, &["mathematics", "philosophy", "law"])
            .build();
        // Philosophy 7 + Mathematics 3 x 3 tokens = 16.
        assert_eq!(breakdown(&st, Player::One).progress_tokens, 16);
    }

    #[test]
    fn agriculture_scores_four_flat_points() {
        let st = StateBuilder::new()
            .tokens(Player::One, &["agriculture"])
            .build();
        assert_eq!(breakdown(&st, Player::One).progress_tokens, 4);
    }

    #[test]
    fn guild_points_use_the_higher_of_the_two_counts() {
        // Player One owns the Merchants Guild but only 1 yellow card; Player
        // Two owns 3. The guild pays on 3.
        let st = StateBuilder::new()
            .built(Player::One, &["merchants-guild", "tavern"])
            .built(Player::Two, &["brewery", "forum", "caravansery"])
            .build();
        assert_eq!(breakdown(&st, Player::One).guilds, 3);
        // The guild is purple, so it does not count itself as commercial.
        assert_eq!(breakdown(&st, Player::Two).guilds, 0);
    }

    #[test]
    fn builders_guild_pays_two_per_wonder_of_the_leader() {
        let st = StateBuilder::new()
            .built(Player::One, &["builders-guild"])
            .wonders_built(Player::One, &["the-pyramids"])
            .wonders_built(Player::Two, &["the-colossus", "the-sphinx", "piraeus"])
            .build();
        assert_eq!(breakdown(&st, Player::One).guilds, 6);
    }

    #[test]
    fn moneylenders_guild_pays_on_the_richer_players_coins() {
        let st = StateBuilder::new()
            .built(Player::One, &["moneylenders-guild"])
            .coins(Player::One, 4)
            .coins(Player::Two, 11)
            .build();
        // max(floor(4/3), floor(11/3)) = 3.
        assert_eq!(breakdown(&st, Player::One).guilds, 3);
    }

    #[test]
    fn shipowners_guild_counts_brown_and_grey_together() {
        let st = StateBuilder::new()
            .built(
                Player::One,
                &["shipowners-guild", "lumber-yard", "glassworks"],
            )
            .built(Player::Two, &["quarry"])
            .build();
        assert_eq!(breakdown(&st, Player::One).guilds, 2);
    }

    #[test]
    fn a_fully_hand_computed_breakdown() {
        // Player One:
        //   blue   : Theater 3, Palace 7                      = 10
        //   green  : Academy 3                                =  3
        //   yellow : Port 3                                   =  3
        //   purple : Magistrate's Guild, blue count max(2, 1) =  2
        //   wonder : The Pyramids 9                           =  9
        //   token  : Philosophy 7                             =  7
        //   military: pawn at +4 in Player One's favour        =  5
        //   coins  : 10 coins                                 =  3
        //                                                total = 42
        let st = StateBuilder::new()
            .built(
                Player::One,
                &["theater", "palace", "academy", "port", "magistrate-s-guild"],
            )
            .wonders_built(Player::One, &["the-pyramids"])
            .tokens(Player::One, &["philosophy"])
            .built(Player::Two, &["altar"])
            .conflict(4)
            .coins(Player::One, 10)
            .build();
        let b = breakdown(&st, Player::One);
        assert_eq!(b.civilian, 10);
        assert_eq!(b.scientific, 3);
        assert_eq!(b.commercial, 3);
        assert_eq!(b.guilds, 2);
        assert_eq!(b.wonders, 9);
        assert_eq!(b.progress_tokens, 7);
        assert_eq!(b.military, 5);
        assert_eq!(b.coins, 3);
        assert_eq!(b.total, 42);
    }

    #[test]
    fn civilian_victory_is_decided_on_totals_then_blue_then_draw() {
        // Straight win on totals.
        let st = StateBuilder::new().built(Player::One, &["palace"]).build();
        assert_eq!(
            civilian_result(&st),
            GameResult::Win {
                winner: Player::One,
                kind: VictoryKind::CivilianVictory
            }
        );

        // Equal totals (7 each), but Player Two's come from a blue card.
        let st = StateBuilder::new()
            .tokens(Player::One, &["philosophy"])
            .built(Player::Two, &["palace"])
            .build();
        assert_eq!(
            civilian_result(&st),
            GameResult::Win {
                winner: Player::Two,
                kind: VictoryKind::CivilianTiebreak
            }
        );

        // Identical cities: a true draw.
        let st = StateBuilder::new()
            .built(Player::One, &["palace"])
            .built(Player::Two, &["town-hall"])
            .build();
        assert_eq!(civilian_result(&st), GameResult::Draw);
    }

    #[test]
    fn instant_wins_are_flagged_as_such() {
        assert!(GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::MilitarySupremacy
        }
        .is_instant());
        assert!(GameResult::Win {
            winner: Player::Two,
            kind: VictoryKind::ScientificSupremacy
        }
        .is_instant());
        assert!(!GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::CivilianVictory
        }
        .is_instant());
        assert!(!GameResult::Draw.is_instant());
    }

    #[test]
    fn ids_used_in_scoring_tests_exist() {
        // Guards the test helpers above against a data rename.
        for slug in ["palace", "merchants-guild", "moneylenders-guild"] {
            assert!(CardId::from_slug(slug).is_some(), "missing card {slug}");
        }
        assert!(WonderId::from_slug("the-pyramids").is_some());
        assert!(TokenId::from_slug("mathematics").is_some());
    }
}
