//! How many decisions each player still gets, and how many of them come in a
//! row.
//!
//! Every race in this game is a race against a *decision budget*, not against
//! the clock: a symbol you can reach in three of your own turns is worthless
//! if the age ends in two. [`Tempo`] is that budget, read once per position
//! for both players, and it is the shared input both the science and the
//! military threat magnitudes are built on.
//!
//! # Why extra turns get their own arithmetic
//!
//! Five wonders in the base game print "play again" (Piraeus, The Appian Way,
//! The Hanging Gardens, The Sphinx, The Temple of Artemis — resolved from
//! [`crate::masks::Masks::play_again_wonders`], not from slugs), and the
//! Theology progress token grants the same for *every* wonder its holder
//! builds. The engine sets its extra-turn flag on each such construction and
//! clears it when the turn is taken, so a second play-again wonder built on an
//! extra turn grants another one: `e` usable play-again wonders are `e + 1`
//! consecutive actions, not two. That is what [`Tempo::chain`] counts, and it
//! is the difference between "the card I need is behind an accessible card, so
//! the opponent will take it first" and "the card I need is behind an
//! accessible card, and I take both".
//!
//! Two hard caps apply, both read off the real state rather than assumed: at
//! most [`duels_core::state::MAX_WONDERS_BUILT`] wonders are ever built
//! between the two players, and an extra turn granted by the last card of an
//! age is simply lost (`engine::end_age` logs it and clears the flag), so a
//! chain cannot be longer than the cards left to take.

use duels_core::data::WonderId;
use duels_core::state::MAX_WONDERS_BUILT;
use duels_core::{GameState, Player};

use crate::board::Board;
use crate::masks::{masks, DECISIONS_PER_AGE};
use crate::prices::Prices;

/// Every tunable constant of the threat model, in one place.
///
/// None of these are derivable from the rules: they price *uncertainty* —
/// how likely an unbuilt wonder is to actually get built, how much a race
/// outcome is worth against a pile of victory points, how much a defender
/// discounts a threat that lands five turns from now. They are grouped here so
/// an arena sweep can fit them without touching any logic, and so every
/// judgement call in the model is visible in one struct.
///
/// The two headline numbers, [`ThreatWeights::game_swing_vp`] and the
/// [`ThreatWeights::stakes_scale`] band, are Kristian's calls: a race lost is
/// worth about twenty-four points of swing, and a player who is behind should
/// gamble on races rather than spend turns denying them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreatWeights {
    // --- tempo ------------------------------------------------------------
    /// How much of a play-again wonder's extra turn to count when the wonder
    /// is owned and unbuilt but *not* affordable right now. It may well
    /// become affordable before the age ends, so this is not zero.
    pub extra_turn_unaffordable: f64,

    // --- science ----------------------------------------------------------
    /// How much of a Mausoleum-recoverable copy to count when the Mausoleum is
    /// owned and unbuilt but not affordable right now.
    pub maus_unaffordable: f64,
    /// How much of the Great Library's three-token draw to count when the
    /// wonder is owned and unbuilt but not affordable right now.
    pub great_library_unaffordable: f64,
    /// How much of a Law token *on the board* to count for a player who has no
    /// live half-pair to claim it with. They can still build one, but it takes
    /// two more cards of the right kind.
    pub p_law_from_scratch: f64,
    /// How many of the set-aside tokens The Great Library draws. Read from the
    /// rules (three) rather than tuned; it lives here so the ratio
    /// `draw / pool` is written once.
    pub great_library_draw: f64,

    // --- military ---------------------------------------------------------
    /// How much of the defender's own counter-push actually converts into
    /// pushing the threat-holder's pawn back, and how much of the shared
    /// shield stream the defender takes out of the threat-holder's hands.
    pub counter_efficiency: f64,
    /// Per-round discount on a military close: a race that closes on the
    /// threat-holder's fourth turn is worth `turn_discount^3` of one that
    /// closes on their first.
    pub turn_discount: f64,

    // --- denial pricing ---------------------------------------------------
    /// Victory-point equivalent of a full race-outcome swing, so a `deltaM`
    /// of one is worth this much.
    pub game_swing_vp: f64,
    /// How fast the stakes multiplier moves with [`crate::VpRead::structural_edge`].
    pub stakes_scale: f64,
    /// Lower bound on the stakes multiplier: a player who is far behind on
    /// points should be gambling on a race of their own, not spending turns
    /// denying one.
    pub stakes_min: f64,
    /// Upper bound on the stakes multiplier: a player who is far ahead has
    /// more to lose from a race than from a card, so denial is worth more.
    pub stakes_max: f64,

    // --- diagnostic labels ------------------------------------------------
    /// Magnitude at or above which a race is labelled
    /// [`crate::MilitaryStatus::Live`] / [`crate::ScienceStatus::Live`].
    pub live_threshold: f64,
    /// Magnitude at or above which a science race is labelled
    /// [`crate::ScienceStatus::Pressure`] — not winnable against a denying
    /// opponent, but worth the denial it forces.
    pub pressure_threshold: f64,
}

impl Default for ThreatWeights {
    fn default() -> Self {
        Self {
            extra_turn_unaffordable: 0.5,
            maus_unaffordable: 0.6,
            great_library_unaffordable: 0.5,
            p_law_from_scratch: 0.25,
            great_library_draw: 3.0,
            counter_efficiency: 0.5,
            turn_discount: 0.7,
            game_swing_vp: 24.0,
            stakes_scale: 0.04,
            stakes_min: 0.6,
            stakes_max: 1.4,
            live_threshold: 0.25,
            pressure_threshold: 0.05,
        }
    }
}

/// One player's decision budget for the rest of the game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tempo {
    /// The player this tempo is for.
    pub player: Player,
    /// Cards still in the current age's structure.
    pub cards_left: u8,
    /// How many of them this player expects to take: the mover gets the odd
    /// one.
    pub picks_in_age: u8,
    /// `picks_in_age` plus [`DECISIONS_PER_AGE`] for every age not yet dealt.
    pub decisions_left: u8,
    /// Whether the engine is already holding an extra turn for this player.
    pub banked: bool,
    /// Extra turns this player could take *right now*, capped by the shared
    /// wonder limit and by the cards left to take, plus any banked one. This
    /// is the length of a chained sequence on their very next turn.
    pub chain: u8,
    /// Extra turns this player can expect over the rest of the game, counting
    /// an unaffordable play-again wonder at
    /// [`ThreatWeights::extra_turn_unaffordable`].
    pub extra_expected: f64,
    /// `decisions_left`, adjusted for the *difference* in expected extra
    /// turns: an extra turn is a decision taken from the other player as much
    /// as one gained.
    pub decisions_left_eff: f64,
    /// This player's share of the remaining decisions, clamped to
    /// `0.2..=0.8`. The probability that any one contested card falls to them.
    pub share: f64,
    /// Coin cost of the `d` cheapest play-again wonder builds, for
    /// `d` in `0..=chain`: what a chained sequence costs before the card at
    /// the end of it is paid for. Index 0 is always zero.
    pub chain_cost: [u16; MAX_CHAIN + 1],
}

/// The longest chained sequence [`Tempo`] will model. Five wonders print
/// "play again" and a player drafts four, so four is already unreachable in
/// practice; the array is sized one past that.
pub const MAX_CHAIN: usize = 5;

/// The lower and upper bound on [`Tempo::share`].
const SHARE_CLAMP: (f64, f64) = (0.2, 0.8);

/// Read both players' tempo, indexed by [`Player::index`].
pub fn tempo_pair(
    state: &GameState,
    board: &Board,
    prices: &[Prices; 2],
    w: &ThreatWeights,
) -> [Tempo; 2] {
    let raw = [
        raw_tempo(state, Player::One, board, &prices[0], w),
        raw_tempo(state, Player::Two, board, &prices[1], w),
    ];
    // `decisions_left_eff` and `share` are each defined against the *other*
    // player's expectation, so they can only be filled in once both raw
    // halves exist.
    let eff = |i: usize| -> f64 {
        let other = 1 - i;
        (f64::from(raw[i].decisions_left) + raw[i].extra_expected - raw[other].extra_expected)
            .max(0.0)
    };
    let (e0, e1) = (eff(0), eff(1));
    let share = |mine: f64, theirs: f64| -> f64 {
        let total = mine + theirs;
        if total <= 0.0 {
            0.5
        } else {
            (mine / total).clamp(SHARE_CLAMP.0, SHARE_CLAMP.1)
        }
    };
    [
        Tempo {
            decisions_left_eff: e0,
            share: share(e0, e1),
            ..raw[0]
        },
        Tempo {
            decisions_left_eff: e1,
            share: share(e1, e0),
            ..raw[1]
        },
    ]
}

/// Everything about one player's tempo that does not depend on the other's.
fn raw_tempo(
    state: &GameState,
    player: Player,
    board: &Board,
    prices: &Prices,
    w: &ThreatWeights,
) -> Tempo {
    let m = masks();
    let me = state.player(player);
    let cards_left = board.cards_left();
    let picks_in_age = if state.current_player() == player {
        cards_left.div_ceil(2)
    } else {
        cards_left / 2
    };
    let decisions_left =
        picks_in_age.saturating_add(board.undealt_age_count().saturating_mul(DECISIONS_PER_AGE));

    let theology = m
        .theology_token()
        .is_some_and(|t| me.tokens().any(|held| held == t));
    // A play-again wonder needs a card from the structure to spend and a free
    // wonder slot, exactly like any other wonder build.
    let cap = MAX_WONDERS_BUILT.saturating_sub(state.wonders_built_total());

    let mut affordable_costs = [0u16; MAX_CHAIN];
    let mut affordable_n = 0usize;
    let mut expected = 0.0f64;
    if prices.can_build_wonder {
        for wonder in me.wonders() {
            if me.has_built_wonder(wonder) || !grants_extra_turn(wonder, theology) {
                continue;
            }
            if prices.can_afford_wonder(wonder) {
                if affordable_n < MAX_CHAIN {
                    affordable_costs[affordable_n] = prices.wonder_cost[wonder.index()];
                    affordable_n += 1;
                }
                expected += 1.0;
            } else {
                expected += w.extra_turn_unaffordable;
            }
        }
    }
    affordable_costs[..affordable_n].sort_unstable();
    let affordable_costs = &affordable_costs[..affordable_n];

    let banked = state.extra_turn() && state.current_player() == player;
    let extra_now = u8::try_from(affordable_n)
        .unwrap_or(u8::MAX)
        .min(cap)
        .min(cards_left.saturating_sub(1))
        .saturating_add(u8::from(banked));
    let extra_expected = expected.min(f64::from(cap));

    // What a chained sequence costs before the card at the end of it is paid
    // for. A banked extra turn is already paid for, so when there is one it is
    // the free first step and the wonders come after it.
    let mut chain_cost = [0u16; MAX_CHAIN + 1];
    let mut running = 0u16;
    for (d, slot) in chain_cost.iter_mut().enumerate().skip(1) {
        let step = match (d - 1).checked_sub(usize::from(banked)) {
            None => 0,
            Some(i) => affordable_costs.get(i).copied().unwrap_or(0),
        };
        running = running.saturating_add(step);
        *slot = running;
    }

    Tempo {
        player,
        cards_left,
        picks_in_age,
        decisions_left,
        banked,
        chain: extra_now.min(u8::try_from(MAX_CHAIN).unwrap_or(u8::MAX)),
        extra_expected,
        // Filled in by `tempo_pair`, which can see both players.
        decisions_left_eff: f64::from(decisions_left),
        share: 0.5,
        chain_cost,
    }
}

/// Whether constructing `wonder` grants its builder another turn: the printed
/// effect, or the Theology token, which grants one for every wonder.
#[inline]
pub fn grants_extra_turn(wonder: WonderId, holds_theology: bool) -> bool {
    holds_theology || masks().play_again_wonders() & (1u16 << wonder.index()) != 0
}

/// Whether `player` holds Theology, which turns every wonder they build into a
/// play-again wonder.
#[inline]
pub fn holds_theology(state: &GameState, player: Player) -> bool {
    masks()
        .theology_token()
        .is_some_and(|t| state.player(player).tokens().any(|held| held == t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data::WonderId;
    use duels_core::testing::StateBuilder;

    fn prices(st: &GameState, board: &Board) -> [Prices; 2] {
        [
            Prices::of(st, Player::One, board),
            Prices::of(st, Player::Two, board),
        ]
    }

    #[test]
    fn exactly_the_five_printed_play_again_wonders_are_found() {
        let mut got: Vec<&str> = WonderId::all()
            .filter(|w| masks().play_again_wonders() & (1u16 << w.index()) != 0)
            .map(|w| w.slug())
            .collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![
                "piraeus",
                "the-appian-way",
                "the-hanging-gardens",
                "the-sphinx",
                "the-temple-of-artemis",
            ],
            "the play-again set changed; the Great Lighthouse is *not* one of them"
        );
        assert_eq!(masks().theology_token().map(|t| t.slug()), Some("theology"));
    }

    #[test]
    fn theology_makes_every_wonder_a_play_again_wonder() {
        let colossus = WonderId::from_slug("the-colossus").unwrap();
        assert!(!grants_extra_turn(colossus, false));
        assert!(grants_extra_turn(colossus, true));
    }

    #[test]
    fn a_chain_is_capped_by_the_cards_left_and_the_shared_wonder_limit() {
        let w = ThreatWeights::default();
        // One card left in the structure: building a wonder with it empties
        // the age, and an extra turn with nowhere to go is lost.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "palace")])
            .wonders(Player::One, &["the-sphinx"])
            .coins(Player::One, 40)
            .current(Player::One)
            .build();
        let board = Board::of(&st);
        let t = tempo_pair(&st, &board, &prices(&st, &board), &w);
        assert_eq!(
            t[Player::One.index()].chain,
            0,
            "no card left to chain into"
        );

        // Two cards left: the wonder eats one and the extra turn takes the
        // other.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .wonders(Player::One, &["the-sphinx"])
            .coins(Player::One, 40)
            .current(Player::One)
            .build();
        let board = Board::of(&st);
        let t = tempo_pair(&st, &board, &prices(&st, &board), &w);
        assert_eq!(t[Player::One.index()].chain, 1);
    }

    #[test]
    fn a_banked_extra_turn_counts_and_costs_nothing() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .extra_turn(true)
            .current(Player::One)
            .build();
        let board = Board::of(&st);
        let t = tempo_pair(&st, &board, &prices(&st, &board), &ThreatWeights::default());
        let me = &t[Player::One.index()];
        assert!(me.banked);
        assert_eq!(me.chain, 1);
        assert_eq!(me.chain_cost[1], 0, "a banked turn is already paid for");
        // ...and not for the player who is not on move.
        assert!(!t[Player::Two.index()].banked);
        assert_eq!(t[Player::Two.index()].chain, 0);
    }

    #[test]
    fn the_share_of_remaining_decisions_is_symmetric_and_clamped() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "town-hall")])
            .current(Player::One)
            .build();
        let board = Board::of(&st);
        let t = tempo_pair(&st, &board, &prices(&st, &board), &ThreatWeights::default());
        let (a, b) = (t[0].share, t[1].share);
        assert!((a + b - 1.0).abs() < 1e-9, "{a} + {b}");
        assert!(a >= SHARE_CLAMP.0 && a <= SHARE_CLAMP.1);
    }
}
