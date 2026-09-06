//! The individual evaluation terms, each read for **one** player.
//!
//! Every function here takes a player and returns that player's own value for
//! one idea; [`crate::evaluate`] differences the two players and applies the
//! root-fixed weights from [`crate::blend`]. Writing them per player rather
//! than as differences is what lets each side's term carry *its own* owner's
//! commitment multiplier — a science race is more important to the player who
//! is in it, whichever of the two that is.
//!
//! Everything here reads only public information and only `duels-core`'s own
//! accessors and cost/scoring engines. Nothing re-implements a rule.

use duels_core::cost;
use duels_core::data::NUM_RESOURCES;
use duels_core::data::{self, CardId, CardType, CountTarget, Resource, TokenId, WonderId};
use duels_core::scoring::{self, Breakdown};
use duels_core::state::{Phase, MAX_WONDERS_BUILT};
use duels_core::{GameState, Player};
use duels_strategy::board::Board;
use duels_strategy::masks::{iter_cards, masks, DECISIONS_PER_AGE};
use duels_strategy::military::signed_distance;
use duels_strategy::science::token_value;

use crate::{EvalWeights, ScienceWeights, SupplyModel};

/// The largest number of units of one resource the development term prices.
///
/// No card or wonder in the base game asks for more than three of anything,
/// so a fourth unit of the same resource is already worth nothing; the table
/// is sized one past that so the zero is visible rather than assumed.
pub const MAX_UNITS: usize = 4;

// ---------------------------------------------------------------------------
// Decision budget
// ---------------------------------------------------------------------------

/// Roughly how many more decisions `p` gets this game.
///
/// A deliberately cheap restatement of [`duels_strategy::Tempo::decisions_left`]
/// — the mover gets the odd card of the current structure, plus
/// [`DECISIONS_PER_AGE`] for every age not yet dealt. `Tempo` itself is
/// computed from a whole [`duels_strategy::Context`], which is far too
/// expensive to rebuild for every candidate action against every chance
/// outcome; this is a handful of integer operations and agrees with it
/// exactly on the parts it models (it omits `Tempo`'s extra-turn adjustment,
/// which is a fractional correction).
pub fn decisions_left(state: &GameState, p: Player) -> f64 {
    let cards_left = state.occupied_slots().count_ones();
    let picks = if state.current_player() == p {
        cards_left.div_ceil(2)
    } else {
        cards_left / 2
    };
    let age = u32::from(state.age().max(1));
    // During the wonder draft the current age's structure has not been laid
    // out yet, so the current age is itself still undealt.
    let first_undealt = if state.phase() == Phase::WonderDraft {
        age
    } else {
        age + 1
    };
    let undealt = 4u32.saturating_sub(first_undealt).min(3);
    f64::from(picks + undealt * u32::from(DECISIONS_PER_AGE))
}

// ---------------------------------------------------------------------------
// Development
// ---------------------------------------------------------------------------

/// How much of the card pool still to come wants each unit of each resource.
///
/// `f[k - 1][r]` is the fraction of the pool whose printed cost demands a
/// `k`-th unit of resource `r`. The pool is the current age's unknown cards
/// plus every age not yet dealt — all of it public information (it is
/// [`Board::unknown_pool`], which
/// `duels-strategy`'s own test pins to [`duels_core::engine::hidden_info`]'s
/// pool, plus whole undealt decks).
///
/// This is a *supply statistic*, not a position: taking one card out of a
/// sixty-card pool moves it by less than the rounding on any weight it feeds.
/// It is therefore computed once per decision at the root, like every weight,
/// while the parts a move genuinely changes — what the player produces, what
/// they would have to pay, which wonders they still owe resources for — are
/// measured on the post-action state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DevSupply {
    /// `f[k - 1][r]`: fraction of the pool needing a `k`-th unit of `r`.
    pub f: [[f64; NUM_RESOURCES]; MAX_UNITS],
    /// How many cards the pool holds. Zero means every fraction is zero.
    pub pool_size: u32,
    /// How locked-in each city's production already is: `0` while every brown
    /// and grey card in the game is still to come, rising to exactly `1` once
    /// none is.
    ///
    /// **Age III has no production cards at all** — the base game prints nine
    /// brown and four grey, six and two of them in Age I and three and two in
    /// Age II, and none in Age III (counted off `data/cards.json` here rather
    /// than asserted from memory; `crate::tests::production_is_completely_
    /// frozen_by_age_three` pins it). So an Age III position has a lock-in of
    /// exactly one: whatever a city produces at the end of Age II is what it
    /// will produce for the rest of the game, and the resource bill it faces
    /// from there is not an estimate of something still fixable but a bill it
    /// is definitely going to pay.
    ///
    /// Mid-Age-II it is partial, which is the interesting case and the reason
    /// this is a continuous factor rather than an `age == 3` branch.
    pub production_lock_in: f64,
    /// How many brown / grey cards producing each resource the pool still
    /// holds, **discounted by the chance each of them is ever dealt**: a card
    /// in the current age's unknown pool appears only if it is behind a
    /// face-down slot rather than in the box, and a card of an undealt age
    /// only if it is not one of the three that age returns to the box unseen.
    ///
    /// The one term that uses it is [`crate::Config::destroy_replace_discount`]
    /// — a destroyed production card is only permanently gone if the market
    /// cannot print another one — so it is deliberately an *expected dealt
    /// count*, not a raw pool count.
    ///
    /// Age III prints no brown or grey card at all
    /// (`crate::tests::production_is_completely_frozen_by_age_three` counts it
    /// off `data/cards.json`), so this goes to exactly zero once Ages I and II
    /// are gone, and an Age III destroy is priced as the permanent loss it is.
    pub sources: [f64; NUM_RESOURCES],
    /// What fraction of the same pool each card colour makes up, indexed by
    /// [`duels_core::data::CardType::index`].
    ///
    /// The colour-wise companion to [`DevSupply::f`], weighted exactly the same
    /// way, and read for exactly one reason: a guild scores on *how many cards
    /// of one colour end up in the leading city*, so projecting that count
    /// forward needs the rate at which the remaining pool prints that colour.
    /// See [`GuildTable`].
    pub kind_f: [f64; data::CardType::ALL.len()],
}

impl DevSupply {
    /// Read the supply statistics off a root position's board, with every pool
    /// entry weighted equally — the pre-existing behaviour.
    pub fn of(board: &Board) -> DevSupply {
        DevSupply::of_with(board, SupplyModel::Raw)
    }

    /// [`DevSupply::of`] with the pool weighting explicit.
    ///
    /// # Why the weighting is a question at all
    ///
    /// The pool is the current age's unknown cards *plus every whole deck not
    /// yet dealt*, and a whole deck is not what reaches the table. Setup deals
    /// [`duels_core::layout::SLOTS`] cards per age out of a larger pool: 20
    /// of Ages I and II's 23, and — Age III being the only age with guilds — 17
    /// of its 20 plain cards plus 3 of its 7 guilds, the rest going back in the
    /// box unseen (`duels_core::engine::new_game`, and
    /// `duels_core::state::GUILDS_IN_PLAY`).
    ///
    /// [`SupplyModel::Raw`] ignores that and counts every undealt card once,
    /// which mildly over-weights Age III's guilds — seven of them in the
    /// statistics where three will be dealt — and so mildly over-states what
    /// the remaining pool will charge for the resources guilds happen to ask
    /// for. [`SupplyModel::Dealt`] weights each undealt entry by its own age's
    /// dealt fraction instead.
    ///
    /// The current age's unknown pool is weighted `1` under both, unchanged:
    /// those cards are already on the table, face down, and
    /// [`duels_strategy::context::Expectations::p_hidden`] is the right
    /// discount for "is this one of them", which is a different question from
    /// "will this ever be dealt".
    pub fn of_with(board: &Board, model: SupplyModel) -> DevSupply {
        let s = data::statics();
        let m = masks();
        let mut pool = board.unknown_pool;
        for age in board.undealt_ages() {
            if age >= 1 && usize::from(age) <= s.age_masks.len() {
                pool |= s.age_masks[usize::from(age) - 1];
            }
        }

        let pool_size = pool.count_ones();
        // The weight one pool entry carries in the statistics below. `Raw`
        // makes every one exactly `1.0`, so the sums are exact integers and
        // reproduce the previous `u32` counts bit for bit.
        let entry_weight = |card: CardId| -> f64 {
            match model {
                SupplyModel::Raw => 1.0,
                SupplyModel::Dealt => {
                    if board.unknown_pool & (1u128 << card.index()) != 0 {
                        1.0
                    } else if s.guild_mask & (1u128 << card.index()) != 0 {
                        m.age_supply(card.def().age).guild_dealt_fraction()
                    } else {
                        m.age_supply(card.def().age).plain_dealt_fraction()
                    }
                }
            }
        };

        let mut counts = [[0.0f64; NUM_RESOURCES]; MAX_UNITS];
        let mut kind_counts = [0.0f64; data::CardType::ALL.len()];
        let mut total = 0.0f64;
        for card in iter_cards(pool) {
            let w = entry_weight(card);
            total += w;
            kind_counts[card.def().kind.index()] += w;
            let cost = card.def().resource_cost;
            for (r, &need) in cost.iter().enumerate() {
                for (k, row) in counts.iter_mut().enumerate() {
                    if u32::from(need) >= (k + 1) as u32 {
                        row[r] += w;
                    }
                }
            }
        }

        let scale = if pool_size == 0 || total <= 0.0 {
            0.0
        } else {
            1.0 / total
        };
        let production = production_mask();
        let total = production.count_ones();
        let still_coming = (pool & production).count_ones();

        // The chance a named card of the *current* age that is not publicly
        // placed is behind a face-down slot rather than in the box — the same
        // quantity `duels_strategy::Expectations::p_hidden` reports, derived
        // here from the board directly so `DevSupply` keeps its one-argument
        // constructor.
        let unknown = board.unknown_pool.count_ones();
        let p_hidden = if unknown == 0 {
            0.0
        } else {
            f64::from(u32::from(board.hidden_slot_count())) / f64::from(unknown)
        };
        let mut sources = [0.0f64; NUM_RESOURCES];
        for card in iter_cards(pool & production) {
            let def = card.def();
            let p_dealt = if board.unknown_pool & (1u128 << card.index()) != 0 {
                p_hidden
            } else {
                duels_strategy::masks()
                    .age_supply(def.age)
                    .plain_dealt_fraction()
            };
            for (r, &n) in def.produces.iter().enumerate() {
                if n > 0 {
                    sources[r] += p_dealt * f64::from(n);
                }
            }
        }

        DevSupply {
            f: std::array::from_fn(|k| std::array::from_fn(|r| counts[k][r] * scale)),
            kind_f: std::array::from_fn(|i| kind_counts[i] * scale),
            pool_size,
            production_lock_in: if total == 0 {
                1.0
            } else {
                1.0 - f64::from(still_coming) / f64::from(total)
            },
            sources,
        }
    }
}

impl DevSupply {
    /// What fraction of the remaining pool is of colour `kind`.
    #[inline]
    pub fn kind_fraction(&self, kind: CardType) -> f64 {
        self.kind_f[kind.index()]
    }

    /// What fraction of the remaining pool is brown or grey — the Shipowners
    /// Guild's category, which is two colours rather than one.
    #[inline]
    pub fn raw_and_manufactured_fraction(&self) -> f64 {
        self.kind_f[CardType::RawMaterial.index()] + self.kind_f[CardType::ManufacturedGood.index()]
    }
}

/// Every brown and grey card in the game, as a mask.
///
/// Resolved from [`duels_core::data::CardType`] rather than by slug, so a
/// change to `data/cards.json` cannot silently desynchronise it.
pub fn production_mask() -> u128 {
    let s = data::statics();
    s.card_masks[data::CardType::RawMaterial.index()]
        | s.card_masks[data::CardType::ManufacturedGood.index()]
}

/// What `p`'s own production is worth, in coins they will not have to spend.
///
/// For each unit of each resource the player actually produces, the term asks
/// how often that unit will be *wanted* — by the cards still to come
/// (`f_k(r)` of the pool, over the `take_rate × decisions_left` further cards
/// this player expects to build) and by their own drafted, unbuilt wonders
/// (a certain want, weight one) — and prices each want at the trade price
/// they would otherwise pay ([`duels_core::cost::trade_prices`]: `2 +` the
/// opponent's production, or a flat 1 behind a trading post).
///
/// A "produce one of your choice" source (Forum, Caravansery, Piraeus, The
/// Great Lighthouse) is valued as one more unit of whichever resource of its
/// group is currently worth the most, greedily and one at a time.
///
/// A trading post is valued separately, by the discount it gives on the units
/// the player does *not* produce — because for the units they do produce, the
/// post has already reduced `price_r` to 1 and so quietly deflated the main
/// term.
///
/// Nothing here says "grey is good": the value of a glass card falls out of
/// how many pool cards and wonders ask for glass, against what glass costs.
/// `examples/watch_blend.rs` prints the per-resource split so that claim can
/// be checked rather than believed.
pub fn development_value(state: &GameState, p: Player, supply: &DevSupply, take_rate: f64) -> f64 {
    development_value_with(state, p, supply, take_rate, true)
}

/// [`development_value`] with the trading-post credit switchable — see
/// [`development_by_resource_with`] for why it has to be.
pub fn development_value_with(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
    include_trading_post: bool,
) -> f64 {
    development_by_resource_with(state, p, supply, take_rate, include_trading_post)
        .iter()
        .sum()
}

/// [`development_value`] split by resource, for the diagnostics.
pub fn development_by_resource(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
) -> [f64; NUM_RESOURCES] {
    development_by_resource_with(state, p, supply, take_rate, true)
}

/// How many units of each resource `p`'s drafted-but-unbuilt wonders still
/// owe, as a `[k - 1][r]` table matching [`DevSupply::f`]'s shape.
///
/// A *certain* want, unlike the pool statistic: those wonders will be paid for
/// out of this city or not at all.
pub fn wonder_wants(state: &GameState, p: Player) -> [[f64; NUM_RESOURCES]; MAX_UNITS] {
    let me = state.player(p);
    let mut wonder = [[0.0f64; NUM_RESOURCES]; MAX_UNITS];
    for w in me.wonders() {
        if me.has_built_wonder(w) {
            continue;
        }
        let cost = w.def().resource_cost;
        for (r, &need) in cost.iter().enumerate() {
            for (k, row) in wonder.iter_mut().enumerate() {
                if u32::from(need) >= (k + 1) as u32 {
                    row[r] += 1.0;
                }
            }
        }
    }
    wonder
}

/// [`development_by_resource`], with the trading-post credit switchable.
///
/// # Why the post credit is optional
///
/// A trading post fixes `price_r` at 1, which quietly *deflates* the main
/// development term for every unit of `r` this city already makes — so the
/// post is credited back separately, by the gouging it prevents on the units
/// the city cannot make. That is correct as long as nothing else in the
/// evaluation prices those un-produced units. [`crate::EconomyModel::Bill`]
/// does exactly that, and it already sees the lower `price_r` the post
/// produces, so with `Bill` in force the separate credit is the same coins
/// counted twice and `include_trading_post` turns it off. See
/// [`resource_bill_by_resource`] for the arithmetic showing the two are
/// numerically the same quantity.
pub fn development_by_resource_with(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
    include_trading_post: bool,
) -> [f64; NUM_RESOURCES] {
    let me = state.player(p);
    let prices = cost::trade_prices(state, p);
    let n_take = (take_rate * decisions_left(state, p)).max(0.0);

    let wonder = wonder_wants(state, p);

    // The marginal worth of the `k`-th unit of `r`, zero past the table.
    let marginal = |k: usize, r: usize| -> f64 {
        if k == 0 || k > MAX_UNITS {
            return 0.0;
        }
        (supply.f[k - 1][r] * n_take + wonder[k - 1][r]) * f64::from(prices[r])
    };

    let production = me.production();
    let mut have: [usize; NUM_RESOURCES] = std::array::from_fn(|r| usize::from(production[r]));
    let mut out = [0.0f64; NUM_RESOURCES];
    for (r, &n) in have.iter().enumerate() {
        for k in 1..=n.min(MAX_UNITS) {
            out[r] += marginal(k, r);
        }
    }

    // Choice sources stand in for one more unit of the best-valued resource
    // in their group, taken greedily so two of them do not both claim the
    // same slot.
    let (choice_raw, choice_manufactured) = me.choice_sources();
    for (count, raw) in [(choice_raw, true), (choice_manufactured, false)] {
        for _ in 0..count {
            let mut best: Option<(usize, f64)> = None;
            for (r, resource) in Resource::ALL.iter().enumerate() {
                if resource.is_raw() != raw {
                    continue;
                }
                let v = marginal(have[r] + 1, r);
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((r, v));
                }
            }
            if let Some((r, v)) = best {
                out[r] += v;
                have[r] += 1;
            }
        }
    }

    // A trading post is worth the gouging it prevents on the units this city
    // still cannot make for itself.
    let fixed = me.fixed_trade();
    let opponent = state.player(p.other());
    for (r, &has_post) in fixed.iter().enumerate() {
        if !has_post || !include_trading_post {
            continue;
        }
        let raw_price = 2.0 + f64::from(opponent.trade_relevant_production(Resource::ALL[r]));
        let discount = (raw_price - 1.0).max(0.0);
        let mut deficit = 0.0;
        for k in (have[r] + 1)..=MAX_UNITS {
            deficit += supply.f[k - 1][r] * n_take + wonder[k - 1][r];
        }
        out[r] += discount * deficit;
    }

    out
}

/// How many units of each resource `p` effectively produces, after each
/// "produce one of your choice" source has been assigned greedily to whichever
/// resource of its group is currently worth the most.
///
/// The same allocation [`development_by_resource_with`] performs internally,
/// factored out so [`resource_bill_by_resource`] can start counting from the
/// same `have[]` rather than from raw production — a Forum really does cover a
/// glass payment, so the bill must not charge for one.
pub fn effective_production(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
) -> [usize; NUM_RESOURCES] {
    let me = state.player(p);
    let prices = cost::trade_prices(state, p);
    let n_take = (take_rate * decisions_left(state, p)).max(0.0);
    let wonder = wonder_wants(state, p);
    let marginal = |k: usize, r: usize| -> f64 {
        if k == 0 || k > MAX_UNITS {
            return 0.0;
        }
        (supply.f[k - 1][r] * n_take + wonder[k - 1][r]) * f64::from(prices[r])
    };

    let production = me.production();
    let mut have: [usize; NUM_RESOURCES] = std::array::from_fn(|r| usize::from(production[r]));
    let (choice_raw, choice_manufactured) = me.choice_sources();
    for (count, raw) in [(choice_raw, true), (choice_manufactured, false)] {
        for _ in 0..count {
            let mut best: Option<(usize, f64)> = None;
            for (r, resource) in Resource::ALL.iter().enumerate() {
                if resource.is_raw() != raw {
                    continue;
                }
                let v = marginal(have[r] + 1, r);
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((r, v));
                }
            }
            if let Some((r, _)) = best {
                have[r] += 1;
            }
        }
    }
    have
}

/// `B(p)`: the coins `p` still expects to hand over in trade payments, over
/// the rest of the game.
///
/// The mirror image of [`development_value`]. Development asks what the units
/// a city *does* produce save it; the bill asks what the units it does *not*
/// produce will cost it, at the price it currently faces — `f_k(r)` of the
/// remaining pool over the builds it expects to make, plus its own unbuilt
/// wonders' certain wants, times `price_r(p)`.
///
/// # Why this is where monopoly value comes from
///
/// [`crate::evaluate`] reads every term per player and differences them, so
/// this term enters as `−B(me)/3 + B(opp)/3`. `price_r(opp)` is `2 + my
/// production of r`. Producing a second unit of something the opponent cannot
/// make therefore *raises* `B(opp)` and so raises the score — and it does so
/// in proportion to how much of the remaining pool actually wants that
/// resource and how little the opponent already makes of it. Nothing anywhere
/// says "grey is good"; grey comes out ahead because Ages I and II carry two
/// grey cards each and Age III carries none, so a second glass source is much
/// more often a genuine monopoly than a second clay source is.
/// `examples/watch_blend.rs` prints the per-resource split so that can be
/// checked rather than believed.
pub fn resource_bill(state: &GameState, p: Player, supply: &DevSupply, take_rate: f64) -> f64 {
    resource_bill_by_resource(state, p, supply, take_rate)
        .iter()
        .sum()
}

/// [`resource_bill`] split by resource, for the diagnostics.
pub fn resource_bill_by_resource(
    state: &GameState,
    p: Player,
    supply: &DevSupply,
    take_rate: f64,
) -> [f64; NUM_RESOURCES] {
    let have = effective_production(state, p, supply, take_rate);
    let prices = cost::trade_prices(state, p);
    let n_take = (take_rate * decisions_left(state, p)).max(0.0);
    let wonder = wonder_wants(state, p);

    let mut out = [0.0f64; NUM_RESOURCES];
    for (r, slot) in out.iter_mut().enumerate() {
        for k in (have[r] + 1)..=MAX_UNITS {
            *slot += (supply.f[k - 1][r] * n_take + wonder[k - 1][r]) * f64::from(prices[r]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Next-age-start tempo
// ---------------------------------------------------------------------------

/// Who is projected to begin the next age.
///
/// Verified against `duels_core::engine::end_age`:
///
/// * if the conflict pawn is **not** centred, the militarily *weaker* player
///   chooses who starts — and will choose themselves;
/// * if it **is** centred, the player who took the age's last card simply
///   begins, with no choice at all;
/// * an extra turn still pending when the age ends is lost.
///
/// So while the age is running, only the player currently projected to take
/// its *last* card is the one whose "start the next age" claim a slow, low-
/// development card would actually forfeit. That projection is a parity
/// count on the cards left, flipped by a banked extra turn (the mover then
/// takes two in a row). Returns `None` when the question is not yet meaningful
/// — during the wonder draft, or once the game is over.
///
/// **Known approximation**: a play-again wonder built later in the age flips
/// the parity again and is not modelled, and the pawn can move before the age
/// ends, changing who the weaker player is. Both are accepted: the term is a
/// tie-breaker worth a few points, and modelling them exactly would need the
/// search this agent does not have.
pub fn projected_starter(state: &GameState) -> Option<Player> {
    if state.is_over() {
        return None;
    }
    match state.phase() {
        // The weaker player has been asked and is about to answer; they will
        // say themselves.
        Phase::ChooseFirstPlayer => Some(state.current_player()),
        Phase::WonderDraft | Phase::GameOver => None,
        Phase::Turn => match state.military_leader() {
            Some(leader) => Some(leader.other()),
            None => {
                let cards_left = state.occupied_slots().count_ones();
                if cards_left == 0 {
                    return Some(state.last_card_taker());
                }
                let n = cards_left + u32::from(state.extra_turn());
                Some(if n % 2 == 1 {
                    state.current_player()
                } else {
                    state.current_player().other()
                })
            }
        },
    }
}

/// What beginning the next age is worth to `p` in this position.
///
/// Not commitment-scaled: the value of moving first into a fresh structure is
/// about the cards, not about anybody's plan. Zero in Age III by default,
/// where there is no next age to begin.
pub fn next_age_start(state: &GameState, p: Player, w: &EvalWeights) -> f64 {
    if projected_starter(state) != Some(p) {
        return 0.0;
    }
    let age = usize::from(state.age().max(1));
    let i = age.saturating_sub(1).min(w.next_age_start.len() - 1);
    w.next_age_start[i]
}

// ---------------------------------------------------------------------------
// Science ladder
// ---------------------------------------------------------------------------

/// Whether the second copy of `symbol` is still physically obtainable.
///
/// A cheap, public-information restatement of the part of
/// [`duels_strategy::ScienceRead::second_copy_obtainable`] this term needs: a
/// copy is gone once it is in a city, spent under a wonder, or in the discard
/// pile, and a copy printed on a card of an *earlier* age than the one being
/// played was returned to the box if it never appeared. The full read is far
/// too expensive to rebuild per candidate action per chance outcome.
///
/// Slightly optimistic: a current-age copy may be one of the three cards
/// boxed unseen at setup.
pub fn second_copy_obtainable(state: &GameState, symbol: data::Science) -> bool {
    let gone = state.player(Player::One).built_mask()
        | state.player(Player::Two).built_mask()
        | state.wonder_fodder_mask()
        | state.discard_mask();
    let age = state.age().max(1);
    iter_cards(masks().symbol_mask(symbol))
        .any(|c| gone & (1u128 << c.index()) == 0 && c.def().age >= age)
}

/// How many of the three progress tokens that a science player most wants —
/// Law (a seventh symbol), Theology (an extra turn per wonder) and Strategy
/// (a shield per red card) — are still on the board to be claimed.
///
/// Resolved by *effect* via [`duels_strategy::masks`], not by slug, so a
/// rename in `data/tokens.json` cannot silently break it.
pub fn strong_board_tokens(state: &GameState) -> f64 {
    let m = masks();
    let strong: [Option<TokenId>; 3] = [m.law_token(), m.theology_token(), m.strategy_token()];
    state
        .board_tokens()
        .filter(|t| strong.contains(&Some(*t)))
        .count() as f64
}

/// The value of the half-pairs `p` is sitting on: each one is a progress
/// token the moment its twin is built, and a turn the opponent has to spend
/// if they would rather it were not.
fn pair_threat(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let m = masks();
    let me = state.player(p);
    let held = me.science();
    let mut awarded = 0u8;
    for sym in me.pairs_awarded() {
        awarded |= 1u8 << sym.index();
    }

    let mut candidates = 0.0f64;
    for sym in duels_strategy::masks::ALL_SCIENCE {
        let i = sym.index();
        // Balance only ever comes from the Law token, so it can never pair.
        if held[i] != 1 || Some(sym) == m.law_symbol() || awarded & (1u8 << i) != 0 {
            continue;
        }
        if second_copy_obtainable(state, sym) {
            candidates += 1.0;
        }
    }
    if candidates == 0.0 {
        return 0.0;
    }

    // What completing a pair would actually pay: the best token currently on
    // the board, priced by `duels-strategy`'s own calibration rather than by
    // a number invented here.
    let best_token = state
        .board_tokens()
        .map(|t| token_value(state, p, t))
        .fold(0.0f64, f64::max);
    candidates * (w.pair_token_share * best_token + w.pair_tempo_tax)
}

/// How much `p`'s scientific position is worth: a convex ladder in distinct
/// symbols, scaled by how good the tokens on offer are, plus the half-pairs
/// they are threatening to complete.
///
/// Convex on purpose. The fourth symbol is worth far more than the second:
/// after Age I's four symbols are dealt and taken, *every* Age II green card
/// completes a pair (Ages I and II each carry one copy of the same four
/// symbols), and the sixth symbol needs one of Age III's two exclusive
/// symbols or the Law token. A linear "points per symbol" term cannot see any
/// of that.
///
/// Six distinct symbols wins outright and is handled by the terminal check,
/// so the ladder's last entry is five.
pub fn science_ladder(state: &GameState, p: Player, w: &ScienceWeights) -> f64 {
    let distinct = usize::from(state.player(p).distinct_science()).min(w.ladder.len() - 1);
    let token_mult = 1.0 + w.strong_token_mult * strong_board_tokens(state);
    w.ladder[distinct] * token_mult + w.pair_threat_weight * pair_threat(state, p, w)
}

// ---------------------------------------------------------------------------
// Military, points, economy, tactics
// ---------------------------------------------------------------------------

/// The conflict pawn's position, signed so positive favours `p`.
pub fn military_position(state: &GameState, p: Player) -> f64 {
    f64::from(signed_distance(state, p))
}

// ---------------------------------------------------------------------------
// Military as a supply-smoothed step function
// ---------------------------------------------------------------------------

/// The smoothing width the band model uses, read once from the root position.
///
/// End-of-game military scoring is a *step* function of the pawn's distance
/// (0 / 2 / 5 / 10 victory points at distances 0 / 1-2 / 3-5 / 6-8, straight
/// out of `data/military.json`), and the loot tokens are two more steps. A
/// flat `0.3 × distance` — what [`military_position`] does — gets the shape
/// wrong in both directions: it pays for a shield that crosses nothing and
/// under-pays the one that crosses 2→3 or 5→6.
///
/// Reading the steps *sharply* would be wrong too, though, because the pawn is
/// still going to move: what a position is worth is the expectation of the
/// step function over where the pawn ends up. The width of that distribution
/// scales with how many shields are still in play, which is exactly
/// [`duels_strategy::MilitaryRead`]'s `visible + expected_hidden +
/// expected_future_ages`. So `σ = max(σ_min, κ·√S_rem)`, a random-walk
/// standard deviation, and each step is replaced by a logistic of width
/// `s = 0.55·σ` centred on the step's own boundary.
///
/// Root-fixed: `S_rem` is a supply statistic, and one card leaving a
/// twenty-shield pool moves it by less than the rounding on any weight it
/// feeds — the same argument [`DevSupply`] rests on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MilSmoothing {
    /// The logistic width.
    pub s: f64,
    /// `(entry distance, victory points gained on entry)` for each band
    /// boundary above zero, read off [`duels_core::data::MilitaryTrack`].
    pub steps: [(f64, f64); 3],
}

impl MilSmoothing {
    /// Build the smoothing for a position with `shields_remaining` shields
    /// still obtainable anywhere in the game.
    pub fn of(shields_remaining: f64, kappa: f64, sigma_min: f64, s_scale: f64) -> MilSmoothing {
        let sigma = (kappa * shields_remaining.max(0.0).sqrt()).max(sigma_min);
        MilSmoothing {
            s: (s_scale * sigma).max(f64::MIN_POSITIVE),
            steps: band_steps(),
        }
    }

    /// The logistic `1 / (1 + e^(−x/s))`.
    #[inline]
    pub fn phi(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x / self.s).exp())
    }
}

/// The real end-of-game military scoring table, as `(entry distance, victory
/// points gained on entering that band)`.
///
/// Derived from [`duels_core::data::MilitaryTrack::victory_points`] — which is
/// `(inclusive max distance, victory points)` ascending — rather than written
/// out, so a change to `data/military.json` cannot silently desynchronise the
/// evaluation from the rules. `duels-strategy`'s own
/// `MilitaryRead::bands` derives the entry distances the same way.
pub fn band_steps() -> [(f64, f64); 3] {
    let track = data::military();
    std::array::from_fn(|i| {
        let (prev_max, prev_vp) = track.victory_points[i];
        let (_, vp) = track.victory_points[i + 1];
        (
            f64::from(prev_max) + 1.0,
            f64::from(vp.saturating_sub(prev_vp)),
        )
    })
}

/// How wide a distribution the pawn's remaining travel should be smoothed
/// over, given the shields still in play and how many rounds the player
/// actually has left to use them.
///
/// # Why the supply alone is the wrong width
///
/// [`MilSmoothing::of`] takes `S_rem`, every shield still obtainable *anywhere
/// in the game*. Early on that is around twenty, which makes `σ ≈ 0.8·√20 ≈
/// 3.6` and the logistic width `s ≈ 2.0` — wide enough that the "step
/// function" the band model exists to represent is, for most of the game,
/// indistinguishable from a straight line. The whole point of the band model
/// is that the shield crossing 2→3 is worth more than the one crossing 1→2,
/// and at that width it barely is.
///
/// The pawn is not going to travel `√S_rem`, though: it is going to travel
/// however far the *next few rounds* carry it, and the rest of `S_rem` belongs
/// to a future in which the position will have been re-evaluated many times.
/// So the width is taken over a horizon of `h` rounds' worth of the shield
/// stream instead:
///
/// ```text
/// s̄   = S_rem / rounds_left        (shields per round of this player's)
/// S_h = min(S_rem, h · s̄)
/// ```
///
/// `horizon = None` restores `S_h = S_rem` exactly, bit for bit, which is what
/// [`crate::Config::v2`] uses.
pub fn horizon_supply(shields_remaining: f64, rounds_left: f64, horizon: Option<f64>) -> f64 {
    match horizon {
        None => shields_remaining,
        Some(h) if h > 0.0 && rounds_left > 0.0 => {
            shields_remaining.min(h * (shields_remaining / rounds_left))
        }
        Some(_) => shields_remaining,
    }
}

/// The smoothed scoring bands as a function of a pawn distance, rather than of
/// a position.
///
/// Factored out of [`military_band`] so [`military_shield_delta`] can evaluate
/// the same curve at `d + k` without inventing a hypothetical `GameState`.
#[inline]
pub fn band_at(distance: f64, sm: &MilSmoothing) -> f64 {
    sm.steps
        .iter()
        .map(|&(entry, gain)| gain * sm.phi(distance - entry + 0.5))
        .sum()
}

/// [`military_loot`] as a function of a pawn distance. Which tokens are still
/// on the board, and what the victim can actually pay, still come from
/// `state`.
pub fn loot_at(state: &GameState, p: Player, distance: f64, sm: &MilSmoothing) -> f64 {
    let track = data::military();
    let opp_coins = f64::from(state.player(p.other()).coins());
    let mut out = 0.0;
    for (i, &(at, coins)) in track.loot.iter().enumerate() {
        if !state.loot_available(p, i) {
            continue;
        }
        let take = f64::from(coins).min(opp_coins);
        out += (take / 3.0) * sm.phi(distance - f64::from(at) + 0.5);
    }
    out
}

/// Exactly what `k` more shields for `p` would move in the main evaluation.
///
/// [`crate::evaluate`] reads the military terms per player and differences
/// them, under each player's own root-fixed commitment multiplier, so the
/// swing `k` shields produce is
///
/// ```text
/// Δ(k) = band · [ w_p · (B(d+k) − B(d))  +  w_opp · (B(−d) − B(−d−k)) ]
///      + loot · [ (L_p(d+k) − L_p(d))    +  (L_opp(−d) − L_opp(−d−k)) ]
/// ```
///
/// A **finite difference**, not `k` times a one-shield slope: a three-shield
/// card that crosses a band boundary is not three separate one-shield steps,
/// and the whole reason the band model exists is that the steps are not
/// evenly spaced. It is also two-sided — the opponent's band falls as mine
/// rises, and that half was simply missing from the price
/// [`crate::menu::TakeValue`] used to put on a shield, which is why the menu
/// systematically under-valued red cards relative to the evaluation that
/// scored the position they produced.
pub fn military_shield_delta(
    state: &GameState,
    p: Player,
    shields: u8,
    sm: &MilSmoothing,
    band_weight: f64,
    loot_weight: f64,
    military_multiplier: (f64, f64),
) -> f64 {
    if shields == 0 {
        return 0.0;
    }
    let d = f64::from(signed_distance(state, p));
    let k = f64::from(shields);
    let (w_p, w_opp) = military_multiplier;
    let opp = p.other();
    let band = w_p * (band_at(d + k, sm) - band_at(d, sm))
        + w_opp * (band_at(-d, sm) - band_at(-d - k, sm));
    let loot = (loot_at(state, p, d + k, sm) - loot_at(state, p, d, sm))
        + (loot_at(state, opp, -d, sm) - loot_at(state, opp, -d - k, sm));
    band_weight * band + loot_weight * loot
}

/// `band(p)`: the expected end-of-game military victory points, as a smoothed
/// step function of the pawn's signed distance.
///
/// Read per player and then differenced by [`crate::evaluate`], exactly like
/// every other term. That is not a double count: `d_opp = −d_me`, so at a
/// centred pawn the two are equal and cancel, and at a decisive lead one
/// saturates at the full 10 points while the other goes to zero — the
/// difference spans `[−10, +10]`, which is the real range of the scoring
/// table. `tests::the_band_model_differences_antisymmetrically` pins it.
pub fn military_band(state: &GameState, p: Player, sm: &MilSmoothing) -> f64 {
    band_at(f64::from(signed_distance(state, p)), sm)
}

/// `loot(p)`: the coins `p` expects to strip off the opponent by pushing the
/// pawn across a loot token, in victory-point units.
///
/// Only counts tokens still on the board on *this* state — a token already
/// triggered has moved into the coin totals the point projection reads, and
/// charging for it again would double it. Capped at the coins the opponent
/// actually holds, matching [`duels_strategy::MilitaryRead::loot_damage`].
///
/// Not commitment-scaled: two coins off a rich opponent is worth the same
/// whether or not this player has a military plan.
pub fn military_loot(state: &GameState, p: Player, sm: &MilSmoothing) -> f64 {
    loot_at(state, p, f64::from(signed_distance(state, p)), sm)
}

/// The marginal value of one more shield to `p` at the root pawn position:
/// the band model's local slope. Used to price a red card's shields inside
/// [`crate::menu`]'s take-value function.
pub fn military_slope(state: &GameState, p: Player, sm: &MilSmoothing) -> f64 {
    let d = f64::from(signed_distance(state, p));
    sm.steps
        .iter()
        .map(|&(entry, gain)| gain * (sm.phi(d + 1.0 - entry + 0.5) - sm.phi(d - entry + 0.5)))
        .sum()
}

// ---------------------------------------------------------------------------
// Coins
// ---------------------------------------------------------------------------

/// The points channel of a coin pile: `c / 3` while the game still has moves
/// left in it, `floor(c / 3)` once the rounding is about to be real.
///
/// The real rule is `floor(c / 3)`, and [`crate::CoinModel::Legacy`] uses it
/// throughout. The trouble is that mid-game it makes the evaluation a *step*
/// function of coins: earning the second of three coins is worth exactly
/// nothing, so a move that leaves the player one coin short of a rounding
/// boundary scores identically to one that leaves them three short, and every
/// comparison in that band is decided by whatever tiny term happens to be
/// next in line. With a dozen more decisions to come the pile will be spent
/// and re-earned several times over before the floor is ever applied, so the
/// smooth rate is the honest expectation; near the end it is not.
pub fn coin_points(state: &GameState, p: Player, endgame_decisions: f64) -> f64 {
    let c = f64::from(state.player(p).coins());
    if decisions_left(state, p) <= endgame_decisions {
        (c / 3.0).floor()
    } else {
        c / 3.0
    }
}

/// The liquidity channel: a concave `β·c_ref·(1 − e^(−c/c_ref))`.
///
/// One function in place of the two ad hoc coin terms it replaces — a cap on
/// "cash for the contested card" and a penalty for falling under a safety
/// floor. Both were trying to say the same thing, which is that the *first*
/// few coins are worth much more than a coin's face value and further ones
/// quickly are not, and a saturating exponential says it once, continuously,
/// with no thresholds to fall off.
pub fn coin_liquidity(state: &GameState, p: Player, beta: f64, c_ref: f64) -> f64 {
    if c_ref <= 0.0 {
        return 0.0;
    }
    let c = f64::from(state.player(p).coins());
    beta * c_ref * (1.0 - (-c / c_ref).exp())
}

/// The marginal victory-point value of one more coin to `p`, for callers that
/// have to price a cost. `liquidity_weight` is `p`'s blended multiplier on the
/// points channel.
pub fn coin_marginal(
    state: &GameState,
    p: Player,
    smooth: bool,
    liquidity_weight: f64,
    coins_div3: f64,
    beta: f64,
    c_ref: f64,
) -> f64 {
    let linear = liquidity_weight * coins_div3 / 3.0;
    if !smooth || c_ref <= 0.0 {
        return linear;
    }
    let c = f64::from(state.player(p).coins());
    linear + beta * (-c / c_ref).exp()
}

/// The escalating "somebody is about to win outright" term: quadratic in how
/// far past the second loot token the pawn is, signed towards whoever it
/// favours.
///
/// Deliberately **not** commitment-scaled. It is the one military term that
/// is as much about the opponent's race as about this player's own plan, and
/// a player who is not committed to military still has to notice a pawn three
/// steps from their capital.
pub fn military_urgency(state: &GameState, p: Player) -> f64 {
    let signed = military_position(state, p);
    let second_loot = f64::from(data::military().loot[1].0);
    if signed.abs() > second_loot {
        let excess = signed.abs() - second_loot;
        signed.signum() * excess * excess
    } else {
        0.0
    }
}

/// The "as if the game ended now" points from cards, wonders and tokens.
///
/// Straight out of [`duels_core::scoring::breakdown`], excluding military
/// (which [`military_position`] reads as board position, not points) and
/// coins (which get their own liquidity terms).
pub fn card_and_token_vp(b: &Breakdown) -> f64 {
    f64::from(b.civilian + b.scientific + b.commercial + b.guilds + b.wonders + b.progress_tokens)
}

/// How many of the game's seven shared wonder slots are still open.
///
/// The base game stops at [`MAX_WONDERS_BUILT`] wonders **between both
/// players**: the eighth is never built, whoever drafted it. Read on the
/// post-action state, so a candidate that builds the seventh wonder is scored
/// against a board with no slots left.
pub fn wonder_slots_left(state: &GameState) -> f64 {
    f64::from(MAX_WONDERS_BUILT.saturating_sub(state.wonders_built_total()))
}

/// How many wonders `p` has drafted and not yet built.
pub fn unbuilt_wonders(state: &GameState, p: Player) -> f64 {
    let ps = state.player(p);
    ps.wonders().filter(|&w| !ps.has_built_wonder(w)).count() as f64
}

/// A rough, hand-tuned "how strong is this wonder" score, summed over `p`'s
/// drafted-but-unbuilt wonders. Without it the evaluation cannot tell two
/// wonder-draft picks apart, since the point projection only credits *built*
/// wonders.
///
/// # The seven-wonder cap
///
/// The sum is zero once [`wonder_slots_left`] is, and that is a **bug fix**
/// rather than a new model: the base game builds seven wonders between the two
/// players and no more, so a player still holding an unbuilt Pyramids after
/// the seventh wonder goes up is holding a card that can never be played. This
/// term used to keep paying half of [`wonder_power`] for it, in every position
/// for the rest of the game, for both players — which is not even symmetric,
/// since the two sides rarely hold the same number of dead wonders. It is
/// unconditional (not behind a [`crate::Config`] knob) for the same reason
/// `next_age_start`'s doubled magnitude was: it is arithmetic that was simply
/// wrong. `tests::an_unbuildable_wonder_is_worth_nothing` pins it.
pub fn wonder_potential(state: &GameState, p: Player) -> f64 {
    if wonder_slots_left(state) <= 0.0 {
        return 0.0;
    }
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(wonder_power)
        .sum()
}

/// The flat, effect-blind power score [`wonder_potential`] sums.
///
/// Every effect that is not points, coins or shields is priced at a flat `+3`
/// ("this wonder does something"), which is what
/// [`crate::WonderModel::Budget`] replaces with a per-effect price.
pub fn wonder_power(w: WonderId) -> f64 {
    let def = w.def();
    let mut v = f64::from(def.victory_points)
        + f64::from(def.coins) * 0.3
        + f64::from(def.shields)
        + f64::from(def.opponent_loses_coins) * 0.3;
    for flag in [
        def.play_again,
        def.destroy.is_some(),
        def.build_discarded_free,
        def.choose_progress_token,
    ] {
        if flag {
            v += 3.0;
        }
    }
    if def.produces_choice.is_some() {
        v += 2.0;
    }
    v
}

// ---------------------------------------------------------------------------
// The wonder budget model
// ---------------------------------------------------------------------------

/// Everything [`crate::WonderModel::Budget`] needs, read **once from the root
/// position** and held fixed for every candidate action.
///
/// Root-fixing is the same discipline every price in this crate follows (see
/// [`crate::blend`] and [`crate::menu`]): a candidate that builds a wonder
/// would otherwise be credited twice, once through the wonder leaving the
/// unbuilt set and again through `p_build` rising for everything left in it.
///
/// The one thing read on the post-action state is *which* wonders are still
/// unbuilt, which is exactly the part a move genuinely changes.
#[derive(Debug, Clone)]
pub struct WonderBudget {
    /// `p_build`, indexed by [`Player::index`].
    p_build: [f64; 2],
    /// `power_p(w)`, indexed by player and [`WonderId::index`].
    power: [[f64; data::NUM_WONDERS]; 2],
}

impl WonderBudget {
    /// An all-zero budget, for when the model is switched off. Never read.
    pub fn empty() -> WonderBudget {
        WonderBudget {
            p_build: [0.0; 2],
            power: [[0.0; data::NUM_WONDERS]; 2],
        }
    }

    /// Read the budget off the root position.
    ///
    /// ```text
    /// p_build(p)  = cap_share(p) · turn_factor(p)
    /// cap_share   = min(1, slots_left / (U_p + U_opp))
    /// turn_factor = min(1, decisions_left(p) / (U_p · turns_per_wonder))
    /// ```
    ///
    /// `cap_share` is the seven-wonder cap read as a *rationing* problem
    /// rather than a boolean: with three slots left and eight unbuilt wonders
    /// between the two cities, no unbuilt wonder is better than three-eighths
    /// likely to happen, and the flat model's "count them all at half price"
    /// is simply the wrong shape. `turn_factor` is the other half of the same
    /// question — a player with four unbuilt wonders and five decisions left
    /// is not going to build four wonders, whatever the cap says.
    ///
    /// At `slots_left == 0` the whole thing is zero, which reproduces the
    /// cap fix in [`wonder_potential`] under this model too; the fix is landed
    /// unconditionally there anyway, because it is a bug rather than a model.
    pub fn of(
        state: &GameState,
        take: &[crate::menu::TakeValue; 2],
        chain: &crate::menu::ChainTable,
        e: &EvalWeights,
    ) -> WonderBudget {
        let mut p_build = [0.0f64; 2];
        let mut power = [[0.0f64; data::NUM_WONDERS]; 2];
        for p in Player::ALL {
            p_build[p.index()] = wonder_p_build(state, p, e);

            let ps = state.player(p);
            for w in ps.wonders() {
                if ps.has_built_wonder(w) {
                    continue;
                }
                power[p.index()][w.index()] = wonder_power_for(state, p, w, take, chain, e);
            }
        }
        WonderBudget { p_build, power }
    }

    /// `p_build(p)`, for the diagnostics.
    #[inline]
    pub fn p_build(&self, p: Player) -> f64 {
        self.p_build[p.index()]
    }

    /// `power_p(w)`, for the diagnostics. Zero for a wonder `p` never drafted
    /// or has already built at the root.
    #[inline]
    pub fn power(&self, p: Player, w: WonderId) -> f64 {
        self.power[p.index()][w.index()]
    }
}

/// `p_build(p)`: the chance any one of `p`'s drafted-but-unbuilt wonders is
/// actually built before the game ends.
///
/// ```text
/// p_build(p)  = cap_share(p) · turn_factor(p)
/// cap_share   = min(1, slots_left / (U_p + U_opp))
/// turn_factor = min(1, decisions_left(p) / (U_p · turns_per_wonder))
/// ```
///
/// Factored out of [`WonderBudget::of`] because it is a **standalone
/// probability estimate** and two unrelated things want it: the wonder budget
/// itself, and [`GuildTable`], which has to project how many wonders the
/// Builders Guild will end up counting. Callable whatever
/// [`crate::WonderModel`] is in force — nothing about it depends on how the
/// evaluation happens to price a wonder's *effects*.
pub fn wonder_p_build(state: &GameState, p: Player, e: &EvalWeights) -> f64 {
    let slots = wonder_slots_left(state);
    let u = unbuilt_wonders(state, p);
    let total_unbuilt = u + unbuilt_wonders(state, p.other());
    let cap_share = if total_unbuilt <= 0.0 {
        0.0
    } else {
        (slots / total_unbuilt).min(1.0)
    };
    let turn_factor = if u <= 0.0 {
        0.0
    } else {
        (decisions_left(state, p) / (u * e.wonder_turns_per_wonder)).min(1.0)
    };
    cap_share * turn_factor
}

/// `Σ_unbuilt p_build(w) · power_p(w)` over the wonders `p` still holds in
/// **this** state, at the root-fixed prices in `budget`.
pub fn wonder_potential_budget(state: &GameState, p: Player, budget: &WonderBudget) -> f64 {
    let f = budget.p_build[p.index()];
    if f == 0.0 {
        return 0.0;
    }
    let ps = state.player(p);
    let table = &budget.power[p.index()];
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w))
        .map(|w| f * table[w.index()])
        .sum()
}

/// `power_p(w)`: what building wonder `w` would actually be worth to `p`,
/// priced effect by effect against the root position.
///
/// Every channel reuses a pricer that already exists rather than inventing a
/// second one — [`crate::menu::TakeValue::coin_marginal`],
/// [`crate::menu::TakeValue::shield_delta`] (which is
/// [`military_shield_delta`], both players' halves and both players'
/// multipliers included), [`crate::menu::TakeValue::produced_value`],
/// [`crate::menu::TakeValue::free_value`] and
/// [`duels_strategy::science::token_value`].
///
/// ```text
/// power_p(w) = VP
///            + coins·coin_marginal_p  +  opp_loses·coin_marginal_opp
///            + shield_delta_p[shields]
///            + play_again · extra_turn_vp
///            + produces_choice · marginal_p(best of the group)
///            + destroy       · max_{c ∈ opp's built cards of that colour}
///                                    production_value_opp(c)
///            + build_free    · max_{c ∈ discard pile} free_value_p(c)
///            + choose_token  · E[ max over the three drawn tokens ]
/// ```
///
/// `extra_turn_vp` defaults to `3.0`, which is deliberately the same number
/// the flat model paid for "this wonder has an effect": at `p_build = 1` and
/// no other effect firing, the two models agree on a play-again wonder, so the
/// new one is a refinement of the old rather than a rescaling of it.
///
/// The progress-token channel prices each token with
/// [`duels_strategy::science::token_value`], which is what this repository
/// already has. That function is a good read on the tokens whose value is
/// points or a symbol and a **flat constant** on Masonry, Mathematics,
/// Architecture, Urbanism and Economy, whose real worth depends on the city
/// they land in. Building those out is a follow-up; this deliberately uses
/// what exists rather than inventing a second, unmeasured token pricer.
pub fn wonder_power_for(
    state: &GameState,
    p: Player,
    w: WonderId,
    take: &[crate::menu::TakeValue; 2],
    chain: &crate::menu::ChainTable,
    e: &EvalWeights,
) -> f64 {
    let def = w.def();
    let opp = p.other();
    let mine = &take[p.index()];
    let theirs = &take[opp.index()];

    let mut v = f64::from(def.victory_points);
    v += f64::from(def.coins) * mine.coin_marginal;
    // Their loss is priced at *their* marginal coin, and read per player and
    // differenced by the evaluation, so it appears exactly once.
    v += f64::from(def.opponent_loses_coins) * theirs.coin_marginal;
    // A wonder's shields never get the Strategy token's bonus -- that is a
    // *military card* effect -- so the table is indexed by the printed count.
    v += mine.shield_delta[usize::from(def.shields).min(mine.shield_delta.len() - 1)];
    if def.play_again {
        v += e.wonder_extra_turn_vp;
    }
    if let Some(group) = def.produces_choice {
        v += mine.produced_value(&[0; NUM_RESOURCES], Some(group));
    }
    if let Some(kind) = def.destroy {
        let mask = state.player(opp).built_mask() & data::statics().card_masks[kind.index()];
        v += iter_cards(mask)
            .map(|c| theirs.production_value(c))
            .fold(0.0f64, f64::max);
    }
    if def.build_discarded_free {
        v += state
            .discard_pile()
            .map(|c| mine.free_value(c, chain))
            .fold(0.0f64, f64::max);
    }
    if def.choose_progress_token {
        v += expected_best_of_three(state, p);
    }
    v
}

/// `E[ max over the three tokens The Great Library draws ]`, averaged over
/// every `C(n, 3)` draw from the set-aside pile with equal probability —
/// exactly the distribution [`duels_core::engine::chance_outcomes`] enumerates
/// for the build itself.
///
/// Zero when fewer than three tokens are set aside, matching the engine: it
/// creates no pending choice at all in that case.
fn expected_best_of_three(state: &GameState, p: Player) -> f64 {
    let aside: Vec<TokenId> = state.set_aside_tokens().collect();
    if aside.len() < 3 {
        return 0.0;
    }
    let values: Vec<f64> = aside.iter().map(|&t| token_value(state, p, t)).collect();
    let n = values.len();
    let mut total = 0.0;
    let mut draws = 0u32;
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                total += values[i].max(values[j]).max(values[k]);
                draws += 1;
            }
        }
    }
    if draws == 0 {
        0.0
    } else {
        total / f64::from(draws)
    }
}

// ---------------------------------------------------------------------------
// Guilds
// ---------------------------------------------------------------------------

/// How many distinct [`CountTarget`]s exist: one per card colour, plus brown +
/// grey together, wonders, and `floor(coins / 3)`.
pub const NUM_COUNT_TARGETS: usize = data::CardType::ALL.len() + 3;

/// A dense index for a [`CountTarget`], so the projections can live in an
/// array rather than a map.
#[inline]
pub fn count_target_index(target: CountTarget) -> usize {
    let n = data::CardType::ALL.len();
    match target {
        CountTarget::Cards(kind) => kind.index(),
        CountTarget::RawAndManufactured => n,
        CountTarget::Wonders => n + 1,
        CountTarget::CoinsDiv3 => n + 2,
    }
}

/// Every [`CountTarget`], in index order — the inverse of
/// [`count_target_index`].
fn all_count_targets() -> [CountTarget; NUM_COUNT_TARGETS] {
    let n = data::CardType::ALL.len();
    std::array::from_fn(|i| {
        if i < n {
            CountTarget::Cards(data::CardType::ALL[i])
        } else if i == n {
            CountTarget::RawAndManufactured
        } else if i == n + 1 {
            CountTarget::Wonders
        } else {
            CountTarget::CoinsDiv3
        }
    })
}

/// What each majority count is projected to be *at the end of the game*, read
/// once from the root position.
///
/// # Why a guild needs a projection at all
///
/// [`duels_core::scoring::breakdown`] scores a built guild at `per_vp ×
/// max(c_me, c_opp)` on the board **as it stands**, which is correct at scoring
/// time and badly wrong as a forward estimate: a Scientists Guild taken in the
/// first half of Age III is bought for the green cards both cities are *going
/// to* have, not the ones they have when it goes up. Nothing in this crate
/// projected that, and [`crate::menu::TakeValue`] did not price a guild at all
/// — every guild card has `victory_points == 0` and `coins == 0`, because a
/// guild scores through `points_by_majority` / `coins_by_majority` instead, so
/// the menu priced every face-up guild at `−cost × coin_marginal`, a strictly
/// negative number. The agent therefore never fought for a guild and never
/// denied one.
///
/// ```text
/// Ĝ(t) = max( c_1 + Δ_1(t),  c_2 + Δ_2(t) )
/// Δ_p(t) = ρ_t · take_rate · decisions_left(p)      for a colour, or brown+grey
///        = U_p · p_build(p)                          for wonders
///        = 0                                         for coins / 3
/// ```
///
/// `ρ_t` is [`DevSupply::kind_fraction`] — the rate at which the *remaining*
/// pool prints that colour, from the same pool walk every other development
/// price uses. `p_build` is [`wonder_p_build`], the same probability
/// [`WonderModel::Budget`](crate::WonderModel::Budget) rations wonders with,
/// called directly rather than through the budget so it is available whatever
/// wonder model is in force.
///
/// **Root-fixed**, like every other price in this crate: computed once per
/// decision and held constant across every candidate, so a move that adds a
/// green card is credited once — through the card — rather than twice, through
/// the card and again through the basis its own guild scores on.
///
/// The `max` is what makes the whole thing a *race* with no special-casing.
/// Both players read the same `Ĝ`, because the rule pays the guild's owner on
/// the higher of the two counts whether or not it is their own, so a guild is
/// worth the same to whoever ends up with it, and denial falls out of the menu
/// differencing the two sides' menus rather than out of a denial rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuildTable {
    hat: [f64; NUM_COUNT_TARGETS],
    live: [f64; NUM_COUNT_TARGETS],
}

impl GuildTable {
    /// An all-zero table, for when guild pricing is switched off. Never read.
    pub fn empty() -> GuildTable {
        GuildTable {
            hat: [0.0; NUM_COUNT_TARGETS],
            live: [0.0; NUM_COUNT_TARGETS],
        }
    }

    /// Read the projections off the root position.
    pub fn of(state: &GameState, supply: &DevSupply, e: &EvalWeights) -> GuildTable {
        let n_take: [f64; 2] =
            Player::ALL.map(|p| (e.development_take_rate * decisions_left(state, p)).max(0.0));
        let wonder_add: [f64; 2] =
            Player::ALL.map(|p| unbuilt_wonders(state, p) * wonder_p_build(state, p, e));

        let mut hat = [0.0f64; NUM_COUNT_TARGETS];
        let mut live = [0.0f64; NUM_COUNT_TARGETS];
        for target in all_count_targets() {
            let i = count_target_index(target);
            // The engine's own majority accessor, not a second copy of the
            // rule.
            live[i] = f64::from(scoring::majority_count(state, target));
            let rho = match target {
                CountTarget::Cards(kind) => supply.kind_fraction(kind),
                CountTarget::RawAndManufactured => supply.raw_and_manufactured_fraction(),
                // Wonders are not dealt from the card pool, and coins have no
                // pool at all.
                CountTarget::Wonders | CountTarget::CoinsDiv3 => 0.0,
            };
            hat[i] = Player::ALL
                .map(|p| {
                    let now = f64::from(state.player(p).count(target));
                    let delta = match target {
                        CountTarget::Wonders => wonder_add[p.index()],
                        CountTarget::CoinsDiv3 => 0.0,
                        _ => rho * n_take[p.index()],
                    };
                    now + delta
                })
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
        }
        GuildTable { hat, live }
    }

    /// `Ĝ(target)`: the projected end-of-game majority count.
    #[inline]
    pub fn hat(&self, target: CountTarget) -> f64 {
        self.hat[count_target_index(target)]
    }

    /// `max(c_1, c_2)(target)` on the root board, which is what a guild's
    /// *immediate* coin payout is settled on and what
    /// [`duels_core::scoring::breakdown`] already credits a built guild's
    /// points at.
    #[inline]
    pub fn live(&self, target: CountTarget) -> f64 {
        self.live[count_target_index(target)]
    }

    /// What one guild card is worth in victory points to a player whose
    /// marginal coin is `coin_marginal`, projections included.
    ///
    /// Zero for every card that is not a guild, because only a guild carries a
    /// `by_majority` effect.
    pub fn card_value(&self, card: CardId, coin_marginal: f64) -> f64 {
        let def = card.def();
        let mut v = 0.0;
        if let Some((target, per)) = def.points_by_majority {
            v += f64::from(per) * self.hat(target);
        }
        if let Some((target, per)) = def.coins_by_majority {
            // Paid once, on the spot, out of the board as it stands — not a
            // projection, and so priced on `live` rather than on `hat`.
            v += f64::from(per) * self.live(target) * coin_marginal;
        }
        v
    }

    /// `Σ_{built guilds of p} per_vp · (Ĝ(t) − live(t))`: the forward
    /// *increment* on the guilds `p` has already built.
    ///
    /// [`duels_core::scoring::breakdown`] — which [`card_and_token_vp`] reads —
    /// already credits those guilds at `per_vp × live(t)`, so this term adds
    /// only the part the snapshot cannot see, and adding the whole projection
    /// would double the half that is already there.
    ///
    /// `built(p)` is read on the post-action state, so a candidate that takes a
    /// guild is credited for it; every price is root-fixed.
    pub fn projection(&self, state: &GameState, p: Player) -> f64 {
        let mask = state.player(p).built_mask() & data::statics().guild_mask;
        if mask == 0 {
            return 0.0;
        }
        let mut out = 0.0;
        for card in iter_cards(mask) {
            if let Some((target, per)) = card.def().points_by_majority {
                let i = count_target_index(target);
                out += f64::from(per) * (self.hat[i] - self.live[i]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Yellow density
// ---------------------------------------------------------------------------

/// How often a decision is actually spent on [`duels_core::Action::Discard`],
/// **measured** rather than guessed.
///
/// `examples/discard_rate.rs` plays whole self-play games and counts, over the
/// decisions at which a discard was legal at all (so the wonder draft, the
/// start-of-age first-player choice and the pending resolutions are excluded —
/// they would deflate the rate for a reason that has nothing to do with how
/// willing a player is to discard). Over 200 `phased` self-play games on each
/// of three disjoint seed ranges it reads **0.2489 / 0.2499 / 0.2467**, so the
/// constant is 0.249 and the third decimal is the only one worth writing down.
///
/// It is strongly age-dependent — 0.18 in Age I, 0.31 in Age II, 0.26 in Age
/// III — and that is *not* modelled here: a per-age rate is a strictly better
/// model and an untested one, and this term already carries enough new
/// modelling to be measured on its own.
pub const DISCARD_RATE_PER_DECISION: f64 = 0.249;

/// `yellow_equity(p)`: what `p`'s commercial cards are worth for the discards
/// they have not made yet.
///
/// [`duels_core::cost::discard_reward`] is `2 + the player's own commercial
/// cards`, so every yellow card in a city raises the payout of **every future
/// discard that city makes** by one coin. The post-action coin pile already
/// flows through [`coin_points`] and [`coin_liquidity`] exactly, so a discard
/// this player has just made is priced correctly; what was entirely unpriced is
/// the forward half — a yellow-heavy city should value discard-as-income and
/// discard-as-denial more highly than a yellow-poor one, and did not.
///
/// ```text
/// yellow_equity(p) = coin_marginal_p · yellows(p) · rate · decisions_left(p)
/// ```
///
/// `rate` is [`DISCARD_RATE_PER_DECISION`]. `coin_marginal` is root-fixed; the
/// yellow count and the decision budget are read on the post-action state,
/// which is exactly the part a move changes.
pub fn yellow_equity(state: &GameState, p: Player, coin_marginal: f64, rate: f64) -> f64 {
    let yellows = f64::from(
        state
            .player(p)
            .count(CountTarget::Cards(CardType::Commercial)),
    );
    if yellows == 0.0 {
        return 0.0;
    }
    coin_marginal * yellows * rate * decisions_left(state, p)
}

/// Cash on hand, capped: what it takes to actually *pay* for the race card
/// the moment it turns up, rather than watch it go.
///
/// Capped because past the price of one contested card, further coins are
/// points rather than liquidity, and the points are already counted.
pub fn race_liquidity(state: &GameState, p: Player, cap: f64) -> f64 {
    f64::from(state.player(p).coins()).min(cap)
}

/// How far below a safe coin cushion `p` is.
pub fn coin_shortfall(state: &GameState, p: Player, floor: f64) -> f64 {
    (floor - f64::from(state.player(p).coins())).max(0.0)
}

/// `p`'s average per-unit trade price: how exposed they are to being gouged.
pub fn average_trade_price(state: &GameState, p: Player) -> f64 {
    let prices = cost::trade_prices(state, p);
    prices.iter().map(|&c| f64::from(c)).sum::<f64>() / prices.len() as f64
}

/// The total value of the free chain builds `p` is about to hand their
/// opponent, if it is genuinely the opponent's turn next.
///
/// Deliberately shallow — no search of the opponent's actual best reply, just
/// a refusal to give away an obviously free, valuable card when an
/// alternative move avoids it.
///
/// # Why `root_age` is a parameter
///
/// This is the only term that reads a *card in the structure*, and that makes
/// it the only one that can leak hidden information. Within one age it is
/// safe: a move that turns a face-down slot over is a real chance event, the
/// engine enumerates every way it can resolve
/// ([`duels_core::engine::chance_outcomes`]), and averaging over them is
/// exactly what this agent does. But a move that empties the structure ends
/// the age, and the engine then deals the *whole next age* from a deck the
/// observation knows nothing about — randomness that is not enumerated as
/// chance outcomes and is instead invented wholesale by
/// [`duels_core::Observation::sample_state`]. Reading those cards would make
/// an age-ending discard score differently depending on which world the
/// throwaway sample happened to invent.
///
/// So the term is switched off the moment the age has moved on from the root
/// position's. `tests/determinization_invariance.rs::
/// resampling_the_same_observation_changes_nothing` is the test that found
/// this, and is the test that keeps it fixed.
pub fn chain_gift_exposure(state: &GameState, p: Player, root_age: u8) -> f64 {
    let taker = p.other();
    if state.age() != root_age || state.phase() != Phase::Turn || state.current_player() != taker {
        return 0.0;
    }
    let mut value = 0.0;
    let mut mask = state.accessible_slots();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        if let Some(card) = state.face_up_card(slot) {
            let def = card.def();
            if let Some(prereq) = def.chain_from {
                if state.player(taker).has_built(prereq) {
                    value += 2.0 + f64::from(def.victory_points);
                }
            }
        }
    }
    value
}

/// Every card carrying `symbol`, for the diagnostics.
pub fn symbol_cards(symbol: data::Science) -> impl Iterator<Item = CardId> {
    iter_cards(masks().symbol_mask(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::engine;
    use duels_core::testing::StateBuilder;

    #[test]
    fn the_decision_budget_agrees_with_duels_strategys_tempo() {
        for seed in 0..6u64 {
            let mut st = engine::new_game(seed);
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
            for _ in 0..40 {
                if st.is_over() {
                    break;
                }
                let ctx = duels_strategy::Context::of(&st);
                for p in Player::ALL {
                    let want = f64::from(ctx.tempo(p).decisions_left);
                    let got = decisions_left(&st, p);
                    assert_eq!(want, got, "seed {seed}: decisions_left for {p:?}");
                }
                let legal = engine::legal_actions(&st);
                if legal.is_empty() {
                    break;
                }
                let action = legal[st.turn() as usize % legal.len()];
                engine::apply_quiet(&mut st, action, &mut rng).unwrap();
            }
        }
    }

    #[test]
    fn the_projected_starter_matches_who_really_begins_the_next_age() {
        // Pawn centred, one card left: whoever takes it begins Age II.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::One));
        let mut after = st;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1);
        engine::apply_quiet(
            &mut after,
            duels_core::Action::Discard { slot: 19 },
            &mut rng,
        )
        .unwrap();
        assert_eq!(after.age(), 2);
        assert_eq!(after.current_player(), Player::One);

        // Two cards left: the mover takes the first, the opponent the last.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::Two));

        // ...unless the mover has a banked extra turn and takes both.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(18, "clay-pool"), (19, "quarry")])
            .conflict(0)
            .extra_turn(true)
            .current(Player::One)
            .build();
        assert_eq!(projected_starter(&st), Some(Player::One));
    }

    #[test]
    fn an_off_centre_pawn_hands_the_choice_to_the_weaker_player() {
        // Player One leads militarily, so Player Two chooses and picks
        // themselves — regardless of who would take the last card.
        for cards in 1..=3u8 {
            let slots: Vec<(u8, &str)> = (0..cards).map(|i| (19 - i, "clay-pool")).collect();
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&slots)
                .conflict(3)
                .current(Player::One)
                .build();
            assert_eq!(
                projected_starter(&st),
                Some(Player::Two),
                "{cards} cards left"
            );
        }
    }

    #[test]
    fn the_science_ladder_is_convex_and_rewards_a_strong_token_row() {
        let w = ScienceWeights::default();
        let mut last_step = 0.0;
        for i in 1..w.ladder.len() {
            let step = w.ladder[i] - w.ladder[i - 1];
            assert!(
                step >= last_step,
                "the ladder is not convex at {i}: {step} < {last_step}"
            );
            last_step = step;
        }

        // Two token rows with the *same* best token, so the only thing that
        // differs is how many of Law / Theology / Strategy are on offer.
        let four = ["workshop", "apothecary", "scriptorium", "pharmacist"];
        let row = |tokens: &[&str]| {
            StateBuilder::new()
                .age(2)
                .built(Player::One, &four)
                .board_tokens(tokens)
                .build()
        };
        let plain = row(&["philosophy"]);
        let strong = row(&["philosophy", "law", "theology"]);
        assert_eq!(plain.player(Player::One).distinct_science(), 4);
        assert_eq!(strong_board_tokens(&plain), 0.0);
        assert_eq!(strong_board_tokens(&strong), 2.0);
        let plain_v = science_ladder(&plain, Player::One, &w);
        let strong_v = science_ladder(&strong, Player::One, &w);
        assert!(
            strong_v > plain_v,
            "law/theology/strategy on the board should read higher: {strong_v} vs {plain_v}"
        );
    }

    #[test]
    fn the_band_steps_match_the_real_scoring_table() {
        // `data/military.json` scores 0 / 2 / 5 / 10 at distances 0 / 1-2 /
        // 3-5 / 6-8, so the boundaries are at 1, 3 and 6 and the gains are 2,
        // 3 and 5.
        assert_eq!(band_steps(), [(1.0, 2.0), (3.0, 3.0), (6.0, 5.0)]);
    }

    #[test]
    fn the_band_model_differences_antisymmetrically() {
        // `d_opp = -d_me` by construction, so `V(me) - V(opp)` must be an odd
        // function of the pawn's position: zero at the centre, equal and
        // opposite at mirrored distances. This is the property that makes
        // reading the term per player and differencing it *not* a double
        // count -- the difference spans the real table's [-10, +10], not
        // twice it.
        let sm = MilSmoothing::of(12.0, 0.8, 0.35, 0.55);
        let diff = |conflict: i8| -> f64 {
            let st = StateBuilder::new()
                .age(2)
                .conflict(conflict)
                .coins(Player::One, 7)
                .coins(Player::Two, 7)
                .build();
            military_band(&st, Player::One, &sm) - military_band(&st, Player::Two, &sm)
        };
        assert!(diff(0).abs() < 1e-12, "centred pawn reads {}", diff(0));
        for d in 1..=8i8 {
            assert!(
                (diff(d) + diff(-d)).abs() < 1e-12,
                "distance {d}: {} vs {}",
                diff(d),
                diff(-d)
            );
            assert!(diff(d) > diff(d - 1), "not monotone at {d}");
        }
        // ...and it saturates inside the real table's range rather than at
        // twice it.
        assert!(diff(8) < 10.5, "diff at the far band is {}", diff(8));
        assert!(diff(8) > 7.0, "diff at the far band is only {}", diff(8));
    }

    #[test]
    fn crossing_a_band_boundary_is_worth_more_than_a_shield_inside_one() {
        // The whole point of the step function: the shield that takes the
        // pawn from 2 to 3 (2 VP -> 5 VP) is worth more than the one that
        // takes it from 1 to 2 (2 VP -> 2 VP), which the flat legacy term
        // prices identically.
        let sm = MilSmoothing::of(8.0, 0.8, 0.35, 0.55);
        let at = |c: i8| {
            let st = StateBuilder::new().age(2).conflict(c).build();
            military_band(&st, Player::One, &sm)
        };
        let inside = at(2) - at(1);
        let crossing = at(3) - at(2);
        assert!(
            crossing > inside,
            "crossing 2->3 ({crossing}) should beat 1->2 ({inside})"
        );
        // The legacy term cannot tell them apart at all.
        let legacy = |c: i8| {
            let st = StateBuilder::new().age(2).conflict(c).build();
            military_position(&st, Player::One)
        };
        assert_eq!(legacy(2) - legacy(1), legacy(3) - legacy(2));
    }

    #[test]
    fn the_loot_term_only_prices_tokens_still_on_the_board() {
        let sm = MilSmoothing::of(8.0, 0.8, 0.35, 0.55);
        let live = StateBuilder::new()
            .age(2)
            .conflict(3)
            .coins(Player::Two, 10)
            .build();
        let spent = StateBuilder::new()
            .age(2)
            .conflict(3)
            .coins(Player::Two, 10)
            .loot_taken(Player::One, 0)
            .build();
        let a = military_loot(&live, Player::One, &sm);
        let b = military_loot(&spent, Player::One, &sm);
        assert!(a > b, "an untaken token should be worth more: {a} vs {b}");

        // ...and it is capped by what the opponent actually holds.
        let broke = StateBuilder::new()
            .age(2)
            .conflict(3)
            .coins(Player::Two, 0)
            .build();
        assert_eq!(military_loot(&broke, Player::One, &sm), 0.0);
    }

    #[test]
    fn the_smooth_coin_model_has_no_rounding_plateau_until_the_endgame() {
        // Mid-game, one more coin is always worth something; the floored
        // model pays nothing for two coins out of three.
        let mid = |coins: u16| {
            let st = StateBuilder::new()
                .age(1)
                .deal(&["clay-pool"; 20])
                .coins(Player::One, coins)
                .current(Player::One)
                .build();
            coin_points(&st, Player::One, 2.0)
        };
        assert!(mid(4) > mid(3), "{} vs {}", mid(4), mid(3));
        assert!(mid(5) > mid(4));

        // With two decisions left the real rounding is about to happen, and
        // the model says so.
        let late = |coins: u16| {
            let st = StateBuilder::new()
                .age(3)
                .open_slots(&[(18, "clay-pool"), (19, "quarry")])
                .coins(Player::One, coins)
                .current(Player::One)
                .build();
            (
                decisions_left(&st, Player::One),
                coin_points(&st, Player::One, 2.0),
            )
        };
        assert!(late(4).0 <= 2.0, "test setup: {} decisions left", late(4).0);
        assert_eq!(late(4).1, 1.0);
        assert_eq!(late(5).1, 1.0);
        assert_eq!(late(6).1, 2.0);
    }

    #[test]
    fn the_resource_bill_makes_a_second_copy_of_a_scarce_resource_pay() {
        // Two cities, identical except for how many glass sources one holds.
        // Nothing in the term names glass; what it reads is that the pool
        // still wants glass and the opponent cannot make any, so the price
        // they face is high.
        let build = |mine: &[&str]| -> f64 {
            let st = StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .built(Player::One, mine)
                .coins(Player::One, 20)
                .coins(Player::Two, 20)
                .current(Player::One)
                .build();
            let supply = DevSupply::of(&duels_strategy::Board::of(&st));
            resource_bill(&st, Player::Two, &supply, 0.6)
        };
        let none = build(&[]);
        let one = build(&["glassworks"]);
        let both = build(&["glassworks", "press"]);
        assert!(
            one > none,
            "one grey source should already raise their bill: {one} vs {none}"
        );
        assert!(
            both > one,
            "a second grey source raises it further: {both} vs {one}"
        );
    }

    #[test]
    fn a_second_copy_is_unobtainable_once_both_copies_are_spoken_for() {
        let mortar = data::Science::Mortar;
        let cards: Vec<CardId> = symbol_cards(mortar).collect();
        assert_eq!(cards.len(), 2, "every card symbol prints exactly twice");

        let fresh = StateBuilder::new().age(1).build();
        assert!(second_copy_obtainable(&fresh, mortar));

        let slugs: Vec<&str> = cards.iter().map(|c| c.def().id).collect();
        let both_built = StateBuilder::new()
            .age(1)
            .built(Player::Two, &slugs)
            .build();
        assert!(!second_copy_obtainable(&both_built, mortar));
    }

    /// The seven-wonder cap: a wonder nobody can ever build is worth nothing,
    /// however good it would have been.
    #[test]
    fn an_unbuildable_wonder_is_worth_nothing() {
        let with_a_slot = StateBuilder::new()
            .age(3)
            .wonders(Player::One, &["the-pyramids"])
            .wonders_built(
                Player::One,
                &["the-colossus", "the-sphinx", "the-hanging-gardens"],
            )
            .wonders_built(
                Player::Two,
                &["piraeus", "the-appian-way", "the-great-lighthouse"],
            )
            .build();
        assert_eq!(with_a_slot.wonders_built_total(), 6);
        assert_eq!(wonder_slots_left(&with_a_slot), 1.0);
        assert!(
            wonder_potential(&with_a_slot, Player::One) > 0.0,
            "an unbuilt Pyramids with a slot left is worth something"
        );

        let full = StateBuilder::new()
            .age(3)
            .wonders(Player::One, &["the-pyramids"])
            .wonders_built(
                Player::One,
                &["the-colossus", "the-sphinx", "the-hanging-gardens"],
            )
            .wonders_built(
                Player::Two,
                &[
                    "piraeus",
                    "the-appian-way",
                    "the-great-lighthouse",
                    "the-mausoleum",
                ],
            )
            .build();
        assert_eq!(full.wonders_built_total(), MAX_WONDERS_BUILT);
        assert_eq!(wonder_slots_left(&full), 0.0);
        assert_eq!(unbuilt_wonders(&full, Player::One), 1.0);
        assert_eq!(wonder_potential(&full, Player::One), 0.0);
        assert_eq!(wonder_potential(&full, Player::Two), 0.0);
    }

    /// Age III prints no brown or grey card, so nothing destroyed in Age III
    /// can ever be replaced — the fact
    /// [`crate::Config::destroy_replace_discount`] rests on, counted off the
    /// card data rather than taken on faith.
    #[test]
    fn no_production_source_survives_into_age_three() {
        let mut age_three_production = 0usize;
        for card in iter_cards(production_mask()) {
            assert_ne!(
                card.def().age,
                3,
                "{} is an Age III production card, which the destroy-replacement \
                 model assumes cannot exist",
                card.def().id
            );
            if card.def().age == 3 {
                age_three_production += 1;
            }
        }
        assert_eq!(age_three_production, 0);

        // ...and the supply statistic agrees on a real Age III position.
        let st = StateBuilder::new().age(3).build();
        let supply = DevSupply::of(&Board::of(&st));
        for (r, &n) in supply.sources.iter().enumerate() {
            assert_eq!(n, 0.0, "Age III still expects to print resource {r}");
        }
    }

    // -----------------------------------------------------------------
    // Guilds
    // -----------------------------------------------------------------

    /// Twenty of Age II's twenty-three cards, dealt into a real structure, so
    /// some slots are genuinely face down.
    const AGE_TWO_DEAL: [&str; 20] = [
        "sawmill",
        "brickyard",
        "shelf-quarry",
        "glassblower",
        "drying-room",
        "walls",
        "horse-breeders",
        "barracks",
        "archery-range",
        "parade-ground",
        "library",
        "dispensary",
        "school",
        "laboratory",
        "courthouse",
        "statue",
        "temple",
        "aqueduct",
        "rostrum",
        "forum",
    ];

    /// The `CardId` with this slug.
    fn card(slug: &str) -> CardId {
        CardId::from_slug(slug).unwrap_or_else(|| panic!("no card {slug:?}"))
    }

    /// **Why guild pricing was needed at all**, asserted rather than argued:
    /// every guild in the game prints zero victory points and zero coins, so
    /// the two fields [`crate::menu::TakeValue::free_value`] starts from say a
    /// guild is worth nothing, and the pricer's answer for a face-up guild was
    /// therefore always `-cost x coin_marginal` — strictly negative.
    #[test]
    fn every_guild_scores_through_a_majority_and_not_through_printed_points() {
        let guilds: Vec<CardId> = iter_cards(data::statics().guild_mask).collect();
        assert_eq!(guilds.len(), 7, "the base game prints seven guilds");
        for g in guilds {
            let def = g.def();
            assert_eq!(def.kind, CardType::Guild, "{}", def.id);
            assert_eq!(def.age, 3, "{} is not an Age III card", def.id);
            assert_eq!(
                def.victory_points, 0,
                "{} prints victory points, so the premise is wrong",
                def.id
            );
            assert_eq!(def.coins, 0, "{} prints coins", def.id);
            assert!(
                def.points_by_majority.is_some() || def.coins_by_majority.is_some(),
                "{} scores through neither majority effect",
                def.id
            );
        }
    }

    /// The category each of the seven keys off, and what its printed cost
    /// actually demands — read off `data/cards.json` through the loader rather
    /// than from anybody's memory of the rulebook, because getting the mapping
    /// wrong would price a guild against the wrong colour and nothing else in
    /// the crate would notice.
    #[test]
    fn the_seven_guilds_key_off_the_categories_the_card_data_prints() {
        /// `(slug, points_by_majority, coins_by_majority, needs glass, needs papyrus)`
        type Row = (
            &'static str,
            Option<(CountTarget, u8)>,
            Option<(CountTarget, u8)>,
            bool,
            bool,
        );
        let want: [Row; 7] = [
            (
                "merchants-guild",
                Some((CountTarget::Cards(CardType::Commercial), 1)),
                Some((CountTarget::Cards(CardType::Commercial), 1)),
                true,
                true,
            ),
            (
                "shipowners-guild",
                Some((CountTarget::RawAndManufactured, 1)),
                Some((CountTarget::RawAndManufactured, 1)),
                true,
                true,
            ),
            (
                "builders-guild",
                Some((CountTarget::Wonders, 2)),
                None,
                true,
                false,
            ),
            (
                "magistrate-s-guild",
                Some((CountTarget::Cards(CardType::Civilian), 1)),
                Some((CountTarget::Cards(CardType::Civilian), 1)),
                false,
                true,
            ),
            (
                "scientists-guild",
                Some((CountTarget::Cards(CardType::Scientific), 1)),
                Some((CountTarget::Cards(CardType::Scientific), 1)),
                false,
                false,
            ),
            (
                "moneylenders-guild",
                Some((CountTarget::CoinsDiv3, 1)),
                None,
                false,
                false,
            ),
            (
                "tacticians-guild",
                Some((CountTarget::Cards(CardType::Military), 1)),
                Some((CountTarget::Cards(CardType::Military), 1)),
                false,
                true,
            ),
        ];

        // Every guild in the data is covered, so a new one could not be added
        // without this test noticing.
        let mut covered = 0u128;
        for (slug, points, coins, glass, papyrus) in want {
            let c = card(slug);
            covered |= 1u128 << c.index();
            let def = c.def();
            assert_eq!(def.points_by_majority, points, "{slug}: points_by_majority");
            assert_eq!(def.coins_by_majority, coins, "{slug}: coins_by_majority");
            let cost = def.resource_cost;
            assert_eq!(
                cost[Resource::Glass.index()] > 0,
                glass,
                "{slug}: glass in the printed cost"
            );
            assert_eq!(
                cost[Resource::Papyrus.index()] > 0,
                papyrus,
                "{slug}: papyrus in the printed cost"
            );
        }
        assert_eq!(
            covered,
            data::statics().guild_mask,
            "the table above does not cover exactly the guilds in the data"
        );

        // Two of the seven need a manufactured good of *both* kinds, which is
        // why an Age III guild is often unaffordable rather than merely
        // expensive, and why denying one matters.
        for slug in ["merchants-guild", "shipowners-guild"] {
            let cost = card(slug).def().resource_cost;
            assert!(cost[Resource::Glass.index()] > 0 && cost[Resource::Papyrus.index()] > 0);
        }
    }

    /// `hat` really is a projection: it is never below the count the board
    /// shows today, and the categories with no pool to draw from do not move.
    #[test]
    fn the_projection_never_falls_below_the_live_count_and_coins_never_move() {
        let st = StateBuilder::new()
            .age(2)
            .built(Player::One, &["tavern", "brewery", "theater"])
            .built(Player::Two, &["altar", "baths"])
            .coins(Player::One, 11)
            .coins(Player::Two, 4)
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&Board::of(&st));
        let e = EvalWeights::default();
        let g = GuildTable::of(&st, &supply, &e);

        for target in all_count_targets() {
            assert!(
                g.hat(target) >= g.live(target),
                "{target:?}: hat {} below live {}",
                g.hat(target),
                g.live(target)
            );
        }

        // `floor(coins / 3)` has no supply and no projection: the Moneylenders
        // Guild is priced on the board exactly as it stands.
        assert_eq!(
            g.hat(CountTarget::CoinsDiv3).to_bits(),
            g.live(CountTarget::CoinsDiv3).to_bits()
        );
        assert_eq!(g.live(CountTarget::CoinsDiv3), 3.0, "max(11/3, 4/3)");

        // ...and a colour with a pool behind it does move.
        assert!(
            g.hat(CountTarget::Cards(CardType::Civilian))
                > g.live(CountTarget::Cards(CardType::Civilian)),
            "Age II still deals blue cards, so the blue count must project upwards"
        );
    }

    /// The Builders Guild's projection comes from [`wonder_p_build`], and goes
    /// to exactly the live count once the seven-wonder cap has closed — at
    /// which point no further wonder can be built by anybody.
    #[test]
    fn the_builders_guild_projects_off_the_wonder_build_probability() {
        // Age II with a full structure, so both players still have plenty of
        // decisions to spend on a wonder — `p_build`'s `turn_factor` is the
        // binding constraint late in Age III, and this test is about the other
        // half of it.
        let open = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .wonders(Player::One, &["the-colossus", "the-sphinx"])
            .wonders_built(Player::One, &["the-pyramids"])
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let e = EvalWeights::default();
        let supply = DevSupply::of(&Board::of(&open));
        let g = GuildTable::of(&open, &supply, &e);
        assert_eq!(g.live(CountTarget::Wonders), 1.0, "one wonder is up");
        assert!(wonder_p_build(&open, Player::One, &e) > 0.0);
        assert!(
            g.hat(CountTarget::Wonders) > g.live(CountTarget::Wonders),
            "two unbuilt wonders with slots and turns left must raise the \
             Builders Guild's basis: {} vs {}",
            g.hat(CountTarget::Wonders),
            g.live(CountTarget::Wonders)
        );

        let full = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "circus")])
            .wonders(Player::One, &["the-pyramids"])
            .wonders_built(
                Player::One,
                &["the-colossus", "the-sphinx", "the-hanging-gardens"],
            )
            .wonders_built(
                Player::Two,
                &[
                    "piraeus",
                    "the-appian-way",
                    "the-great-lighthouse",
                    "the-mausoleum",
                ],
            )
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&Board::of(&full));
        let g = GuildTable::of(&full, &supply, &e);
        assert_eq!(wonder_p_build(&full, Player::One, &e), 0.0);
        assert_eq!(
            g.hat(CountTarget::Wonders).to_bits(),
            g.live(CountTarget::Wonders).to_bits(),
            "with every slot gone the Builders Guild counts what is already up"
        );
    }

    /// The forward increment on a *built* guild adds only what the snapshot
    /// cannot see — `scoring::breakdown` already credits `per_vp x live`.
    #[test]
    fn the_built_guild_projection_is_only_the_part_the_snapshot_misses() {
        let st = StateBuilder::new()
            .age(3)
            .built(Player::One, &["scientists-guild", "workshop"])
            .built(Player::Two, &["apothecary", "dispensary"])
            .coins(Player::One, 9)
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&Board::of(&st));
        let e = EvalWeights::default();
        let g = GuildTable::of(&st, &supply, &e);

        let target = CountTarget::Cards(CardType::Scientific);
        let want = g.hat(target) - g.live(target);
        assert_eq!(
            g.projection(&st, Player::One).to_bits(),
            want.to_bits(),
            "one guild at one point per unit is exactly the increment"
        );
        // The opponent holds no guild, so they get nothing from the term.
        assert_eq!(g.projection(&st, Player::Two), 0.0);
        // And `breakdown` really is already paying the live half, so the two
        // together are the whole projection rather than a double count.
        assert_eq!(
            f64::from(scoring::breakdown(&st, Player::One).guilds),
            g.live(target),
        );
    }

    /// `card_value` is the two channels of the rule and nothing else: points on
    /// the *projected* count, coins on the *live* one.
    #[test]
    fn a_guild_card_is_priced_on_the_projection_for_points_and_the_board_for_coins() {
        let st = StateBuilder::new()
            .age(3)
            .built(Player::One, &["tavern", "brewery"])
            .built(Player::Two, &["theater"])
            .coins(Player::One, 12)
            .current(Player::One)
            .build();
        let supply = DevSupply::of(&Board::of(&st));
        let e = EvalWeights::default();
        let g = GuildTable::of(&st, &supply, &e);

        let yellow = CountTarget::Cards(CardType::Commercial);
        let coin_marginal = 0.25;
        let want = g.hat(yellow) + g.live(yellow) * coin_marginal;
        assert_eq!(
            g.card_value(card("merchants-guild"), coin_marginal)
                .to_bits(),
            want.to_bits()
        );

        // The Builders Guild pays two points per wonder and no coins at all.
        let builders = g.card_value(card("builders-guild"), coin_marginal);
        assert_eq!(
            builders.to_bits(),
            (2.0 * g.hat(CountTarget::Wonders)).to_bits()
        );

        // Nothing that is not a guild is priced here.
        for slug in ["palace", "lumber-yard", "theater"] {
            assert_eq!(g.card_value(card(slug), coin_marginal), 0.0, "{slug}");
        }
    }

    /// A yellow-heavy city really does value its future discards more.
    #[test]
    fn yellow_equity_rises_with_the_commercial_cards_in_the_city() {
        let build = |cards: &[&str]| {
            StateBuilder::new()
                .age(2)
                .deal(&AGE_TWO_DEAL)
                .built(Player::One, cards)
                .coins(Player::One, 10)
                .current(Player::One)
                .build()
        };
        let none = build(&["theater"]);
        let some = build(&["theater", "tavern", "brewery"]);
        let rate = DISCARD_RATE_PER_DECISION;
        assert_eq!(yellow_equity(&none, Player::One, 0.3, rate), 0.0);
        let two = yellow_equity(&some, Player::One, 0.3, rate);
        assert!(two > 0.0, "two yellow cards must be worth something: {two}");
        // Exactly linear in the count, which is what the rule is: every yellow
        // card adds one coin to every future discard.
        assert_eq!(
            two.to_bits(),
            (0.3 * 2.0 * rate * decisions_left(&some, Player::One)).to_bits()
        );
    }

    /// The dealt-fraction weighting really does move the supply statistics, and
    /// moves them in the direction the setup rule says: Age III's guilds are
    /// three of seven, so a pool that treats all seven as certain over-weights
    /// whatever they ask for.
    #[test]
    fn weighting_the_undealt_pool_by_its_dealt_fraction_changes_the_statistics() {
        // Age I, so Ages II and III are both wholly undealt.
        let st = StateBuilder::new().age(1).build();
        let board = Board::of(&st);
        let raw = DevSupply::of_with(&board, SupplyModel::Raw);
        let dealt = DevSupply::of_with(&board, SupplyModel::Dealt);
        assert_eq!(
            DevSupply::of(&board),
            raw,
            "the one-argument constructor must still be the raw one"
        );
        assert_ne!(raw.f, dealt.f, "the weighting changed nothing at all");
        assert!(
            dealt.kind_fraction(CardType::Guild) < raw.kind_fraction(CardType::Guild),
            "seven guilds counted where three are dealt: {} vs {}",
            raw.kind_fraction(CardType::Guild),
            dealt.kind_fraction(CardType::Guild)
        );
        // The colour fractions are a distribution, under either weighting.
        for s in [raw, dealt] {
            let total: f64 = CardType::ALL.iter().map(|&k| s.kind_fraction(k)).sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "colour fractions sum to {total}"
            );
        }
        // The exact rates the setup rule prints.
        let m = masks();
        assert_eq!(m.age_supply(3).guild_dealt_fraction(), 3.0 / 7.0);
        assert_eq!(m.age_supply(3).plain_dealt_fraction(), 17.0 / 20.0);
        assert_eq!(m.age_supply(1).plain_dealt_fraction(), 20.0 / 23.0);
        assert_eq!(m.age_supply(1).guild_dealt_fraction(), 0.0);
    }
}
