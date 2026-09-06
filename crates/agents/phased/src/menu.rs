//! Two forward-looking, root-priced terms: **what the next mover's best
//! available move is worth**, and **what a chain starter is worth for the
//! successor it unlocks**.
//!
//! Both are answers to the same complaint about a snapshot evaluation: it
//! scores the city a move produces and nothing about the position that move
//! leaves behind. Everything in [`crate::terms`] is a property of one
//! player's own tableau. Nothing there notices that the move just made left a
//! 7-point Palace face up and affordable for the opponent, and nothing there
//! notices that the Scriptorium in this city is a free Library later.
//!
//! # Pricing is root-fixed, exposure is not
//!
//! `v_q(card)` — what one card is worth to one player — is priced **once per
//! decision from the root position**, for the same reason every weight is (see
//! [`crate::blend`]): a candidate that changes what a card is worth would
//! otherwise be credited twice, once through the move and again through the
//! re-priced menu. What *is* read on the post-outcome state is which cards are
//! accessible, whose turn it is, and what is affordable — the parts a move
//! genuinely changes.
//!
//! # The stand-down rule
//!
//! [`menu_term`] reads cards in the structure, which makes it the second term
//! in this crate (after [`crate::terms::chain_gift_exposure`]) that could leak
//! hidden information. Within one age it cannot: a move that turns a slot over
//! is a real chance event, [`duels_core::engine::chance_outcomes`] enumerates
//! every way it resolves from the *public* unseen pool, and averaging over
//! them is exactly what this agent does. A move that empties the structure is
//! different — the engine deals a whole new age from a deck no observation can
//! see, and the identities that appear are the throwaway sample's invention.
//! So the term stands down the moment the age or the phase has moved on from
//! the root's.
//!
//! [`chain_equity`] needs no such rule: it reads only cities, the discard pile
//! and the wonder-fodder pile, all of which stay public across an age
//! boundary.
//!
//! # What `v_q` prices, and the one thing it used not to
//!
//! [`TakeValue::free_value`] is the whole of a card's worth to one player:
//! printed points and coins, shields (differenced across both players — see
//! [`MenuShieldPricing`]), a scientific symbol's place on the ladder,
//! production against the resources the remaining pool will ask for, chain
//! equity, and — from round five — the guild majority a purple card scores on
//! and the future discards a yellow card pays for.
//!
//! Those last two were **missing entirely**, and the guild half was not a
//! refinement but a sign error in effect: every guild in the game prints zero
//! victory points and zero coins, so a face-up guild priced out at `−cost ×
//! coin_marginal` — strictly negative, in every position, for both players. See
//! [`crate::GuildPricing`] and [`crate::terms::GuildTable`].
//!
//! # The floor
//!
//! [`menu_term`]'s softmax used to return a hard zero when nothing on the board
//! was affordable, which says an opponent one coin short of everything is in
//! the same position as one whose turn is genuinely worthless. [`crate::MenuFloor`]
//! adds the discard they can always take, and optionally the best wonder they
//! can already pay for, as further entries in the same softmax. Off by default;
//! the measurement is in the crate docs.

use duels_core::data::{
    CardId, CardType, Resource, Science, NUM_CARDS, NUM_RESOURCES, NUM_SCIENCE,
};
use duels_core::state::Phase;
use duels_core::{cost, GameState, Player};
use duels_strategy::board::Board;
use duels_strategy::context::Expectations;
use duels_strategy::masks::masks;
use duels_strategy::science::token_value;

use crate::terms::{self, DevSupply, GuildTable, MilSmoothing, WonderBudget, MAX_UNITS};
use crate::{
    CoinModel, Config, GuildPricing, MenuFloor, MenuShieldPricing, MenuWeights, MilitaryModel,
};

/// The largest shield gain one card can carry: the biggest printed red card
/// plus the Strategy token's bonus. The `shield_delta` table is sized to it.
const MAX_SHIELD_STEP: usize = 4;

/// Everything `v_q(card)` needs, read once from the root position for one
/// player.
#[derive(Debug, Clone)]
pub struct TakeValue {
    /// The player whose pricing context this is.
    pub player: Player,
    /// What one more shield is worth to them, right now: the local, one-sided
    /// slope of the military model at the root pawn position. Used only by
    /// [`MenuShieldPricing::OneSided`].
    pub military_slope: f64,
    /// `Δ(k)` for `k` shields: the exact finite difference of what the main
    /// evaluation's military terms would move, both players' halves included
    /// and both players' root-fixed multipliers applied. Used by
    /// [`MenuShieldPricing::Differenced`].
    pub shield_delta: [f64; MAX_SHIELD_STEP + 1],
    /// Whether this player holds the Strategy token, which adds a shield to
    /// every red card they build.
    pub strategy: bool,
    /// Which of the two prices above [`TakeValue::shields_value`] uses.
    pub pricing: MenuShieldPricing,
    /// What one more coin is worth to them, right now.
    pub coin_marginal: f64,
    /// What the next distinct scientific symbol is worth: `ladder[k + 1] −
    /// ladder[k]` at their current count.
    pub ladder_step: f64,
    /// What completing a half-pair is worth: a share of the best progress
    /// token currently on the board, plus the tempo tax.
    pub pair_bonus: f64,
    held: [u8; NUM_SCIENCE],
    awarded: u8,
    /// Their effective production after choice sources are assigned.
    pub have: [usize; NUM_RESOURCES],
    /// The per-unit trade price they face for each resource.
    pub prices: [u16; NUM_RESOURCES],
    marginal_want: [[f64; NUM_RESOURCES]; MAX_UNITS],
    /// The root-fixed majority projections a guild card is priced against.
    guild: GuildTable,
    /// Whether [`TakeValue::free_value`] consults it.
    guild_pricing: GuildPricing,
    /// What one more commercial card is worth to this player for the coins it
    /// adds to every future discard: `coin_marginal · rate · decisions_left`,
    /// already multiplied by the term's own weight so a weight of zero is an
    /// exact no-op. See [`terms::yellow_equity`], of which this is the
    /// per-card finite difference.
    yellow_step: f64,
}

/// The root-fixed tables [`TakeValue::of`] reads, bundled so its signature
/// stays one a reader can hold in their head.
#[derive(Debug, Clone, Copy)]
pub struct TakeContext<'a> {
    /// The development supply statistics.
    pub supply: &'a DevSupply,
    /// The military band smoothing.
    pub smoothing: &'a MilSmoothing,
    /// The guild majority projections.
    pub guild: &'a GuildTable,
}

impl TakeValue {
    /// Read one player's pricing context off the root position.
    ///
    /// `military_multiplier` is `(this player's, the opponent's)` root-fixed
    /// commitment multiplier on the military term — both halves, because
    /// [`MenuShieldPricing::Differenced`] prices a shield by what the
    /// *differenced* evaluation would actually move, not by what one side's
    /// band gains.
    pub fn of(
        state: &GameState,
        player: Player,
        tables: TakeContext<'_>,
        config: &Config,
        liquidity_weight: f64,
        military_multiplier: (f64, f64),
    ) -> TakeValue {
        let TakeContext {
            supply,
            smoothing: sm,
            guild,
        } = tables;
        let e = &config.eval;
        let smooth_coins = config.coin_model == CoinModel::Smooth;
        // Under the legacy military model the pawn is priced flat, so one more
        // shield is worth exactly that flat rate; under the band model it is
        // the local slope of the real scoring table.
        let legacy_military_step = match config.military_model {
            MilitaryModel::Legacy => Some(e.military_position),
            MilitaryModel::Band => None,
        };
        let me = state.player(player);
        let distinct = usize::from(me.distinct_science());
        let ladder = &e.science.ladder;
        let ladder_step = if distinct + 1 < ladder.len() {
            ladder[distinct + 1] - ladder[distinct]
        } else {
            // Already at five distinct symbols: the sixth wins outright, and
            // the terminal check prices that. Use the top rung's own step so
            // the last symbol is not read as worthless.
            ladder[ladder.len() - 1] - ladder[ladder.len() - 2]
        };
        let best_token = state
            .board_tokens()
            .map(|t| token_value(state, player, t))
            .fold(0.0f64, f64::max);
        let pair_bonus = e.science.pair_token_share * best_token + e.science.pair_tempo_tax;

        let mut awarded = 0u8;
        for sym in me.pairs_awarded() {
            awarded |= 1u8 << sym.index();
        }

        let take_rate = e.development_take_rate;
        let n_take = (take_rate * terms::decisions_left(state, player)).max(0.0);
        let wonder = terms::wonder_wants(state, player);
        let prices = cost::trade_prices(state, player);
        let marginal_want = std::array::from_fn(|k| {
            std::array::from_fn(|r| (supply.f[k][r] * n_take + wonder[k][r]) * f64::from(prices[r]))
        });

        let strategy = duels_strategy::masks()
            .strategy_token()
            .is_some_and(|t| me.tokens().any(|held| held == t));
        let (w_p, w_opp) = military_multiplier;
        let shield_delta: [f64; MAX_SHIELD_STEP + 1] = std::array::from_fn(|k| {
            let k = u8::try_from(k).unwrap_or(u8::MAX);
            match config.military_model {
                // The legacy model is flat in the pawn's position, so the
                // finite difference is exactly linear -- but it is still the
                // *differenced* one, with both players' multipliers.
                MilitaryModel::Legacy => e.military_position * f64::from(k) * (w_p + w_opp),
                MilitaryModel::Band => terms::military_shield_delta(
                    state,
                    player,
                    k,
                    sm,
                    e.military_band,
                    e.military_loot,
                    (w_p, w_opp),
                ),
            }
        });

        let coin_marginal = terms::coin_marginal(
            state,
            player,
            smooth_coins,
            liquidity_weight,
            e.coins_div3,
            e.coin_smooth_beta,
            e.coin_smooth_ref,
        );
        // The menu has to agree with the main evaluation about what a yellow
        // card is worth, or a card the evaluation likes reads as a card the
        // opponent would not bother taking. Zero weight, zero step, exactly.
        let yellow_step = if e.yellow_equity == 0.0 {
            0.0
        } else {
            e.yellow_equity
                * coin_marginal
                * e.yellow_discard_rate
                * terms::decisions_left(state, player)
        };

        TakeValue {
            player,
            shield_delta,
            strategy,
            guild: *guild,
            guild_pricing: config.guild_pricing,
            yellow_step,
            pricing: config.menu_shield_pricing,
            military_slope: legacy_military_step
                .unwrap_or_else(|| terms::military_slope(state, player, sm)),
            coin_marginal,
            ladder_step,
            pair_bonus,
            held: me.science(),
            awarded,
            have: terms::effective_production(state, player, supply, take_rate),
            prices,
            marginal_want,
        }
    }

    /// What a card's `printed` shields are worth to this player.
    ///
    /// Under [`MenuShieldPricing::OneSided`] this is `printed × slope`, the
    /// round-two behaviour: one player's band only, and linear in `k`. Under
    /// [`MenuShieldPricing::Differenced`] it is the table built at the root
    /// from [`terms::military_shield_delta`], with the Strategy token's extra
    /// shield folded in — which is the same quantity the main evaluation would
    /// actually credit for taking the card, rather than an approximation of
    /// half of it.
    #[inline]
    pub fn shields_value(&self, printed: u8) -> f64 {
        match self.pricing {
            MenuShieldPricing::OneSided => f64::from(printed) * self.military_slope,
            MenuShieldPricing::Differenced => {
                if printed == 0 {
                    return 0.0;
                }
                let k = usize::from(printed.saturating_add(u8::from(self.strategy)));
                self.shield_delta[k.min(MAX_SHIELD_STEP)]
            }
        }
    }

    /// What one scientific symbol on a card would do for this player: open a
    /// new rung of the ladder, complete a half-pair, or nothing.
    pub fn symbol_value(&self, symbol: Science) -> f64 {
        let i = symbol.index();
        if self.held[i] == 0 {
            self.ladder_step
        } else if self.held[i] == 1 && self.awarded & (1u8 << i) == 0 {
            self.pair_bonus
        } else {
            0.0
        }
    }

    /// The worth of the `k`-th unit of resource `r` to this player, zero past
    /// the table.
    fn marginal(&self, k: usize, r: usize) -> f64 {
        if k == 0 || k > MAX_UNITS {
            0.0
        } else {
            self.marginal_want[k - 1][r]
        }
    }

    /// What a card's production would save this player.
    ///
    /// Public because [`crate::terms::wonder_power_for`] prices what a
    /// destroy effect takes *off the opponent* with exactly this function,
    /// read against the opponent's own pricing context — the same quantity,
    /// asked from the other side, rather than a second implementation of it.
    pub fn production_value(&self, card: CardId) -> f64 {
        let def = card.def();
        self.produced_value(&def.produces, def.produces_choice)
    }

    /// [`TakeValue::production_value`] for a production profile that is not a
    /// card's: a wonder's "produce one of this group, your choice" source has
    /// no [`CardId`] to look it up from.
    pub fn produced_value(
        &self,
        produces: &[u8; NUM_RESOURCES],
        choice: Option<duels_core::data::ResourceGroup>,
    ) -> f64 {
        let mut have = self.have;
        let mut out = 0.0;
        for (r, &n) in produces.iter().enumerate() {
            for _ in 0..n {
                out += self.marginal(have[r] + 1, r);
                have[r] += 1;
            }
        }
        if let Some(group) = choice {
            let mut best: Option<(usize, f64)> = None;
            for (r, resource) in Resource::ALL.iter().enumerate() {
                if !group.members().contains(resource) {
                    continue;
                }
                let v = self.marginal(have[r] + 1, r);
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((r, v));
                }
            }
            if let Some((_, v)) = best {
                out += v;
            }
        }
        out
    }

    /// `v_q(card)`: what taking `card` would be worth to this player, in
    /// victory-point equivalents, priced against the root position.
    ///
    /// Deliberately excludes the race-magnitude channel
    /// ([`duels_strategy::delta_m`]): [`crate::Root::denial_term`] already
    /// prices what a move does to the opponent's races at the root, and
    /// counting it again here would double it.
    pub fn value(&self, state: &GameState, card: CardId, chain: &ChainTable) -> f64 {
        let price = f64::from(cost::card_cost(state, self.player, card).coins);
        self.free_value(card, chain) - price * self.coin_marginal
    }

    /// [`TakeValue::value`] with the cost term dropped: what the card is worth
    /// to a player who does **not** have to pay for it.
    ///
    /// The Mausoleum's retrieval is exactly that — one card out of the discard
    /// pile, constructed for free — so this is the price
    /// [`crate::terms::wonder_power_for`] puts on it, rather than a second
    /// pricing function that would drift from the one the menu uses.
    pub fn free_value(&self, card: CardId, chain: &ChainTable) -> f64 {
        let def = card.def();
        let mut v = f64::from(def.victory_points) + f64::from(def.coins) / 3.0;
        v += self.shields_value(def.shields);
        if let Some(sym) = def.science {
            v += self.symbol_value(sym);
        }
        v += self.production_value(card);
        v += chain.equity(self.player, card);
        v += self.guild_value(card);
        if def.kind == CardType::Commercial {
            v += self.yellow_step;
        }
        v
    }

    /// What a guild card's majority scoring is worth to this player.
    ///
    /// **Zero for every non-guild card**, and zero throughout under
    /// [`GuildPricing::Unpriced`], which is what makes the whole thing an exact
    /// no-op when it is switched off.
    ///
    /// Every guild in the base game prints `victory_points == 0` and `coins ==
    /// 0` — verified card by card in `tests::every_guild_scores_through_a_
    /// majority_and_not_through_printed_points` — so the two terms
    /// [`TakeValue::free_value`] starts from contribute nothing for a guild and
    /// this is the whole of its value. Without it a face-up guild priced out at
    /// `−cost × coin_marginal`: strictly negative, always, so the agent would
    /// never take one and never deny one.
    #[inline]
    pub fn guild_value(&self, card: CardId) -> f64 {
        match self.guild_pricing {
            GuildPricing::Unpriced => 0.0,
            GuildPricing::Projected => self.guild.card_value(card, self.coin_marginal),
        }
    }
}

// ---------------------------------------------------------------------------
// Chain equity
// ---------------------------------------------------------------------------

/// The probability that a chain successor is still going to show up, times
/// the chance the holder is the one who gets it. A flat constant for this
/// round; a real model would read the opponent's own interest in the card.
const CHAIN_MINE_SHARE: f64 = 0.7;

/// What every chain starter is worth to each player, priced once from the
/// root position.
///
/// ```text
/// equity(p, c) = P_app(chain_to(c)) · g · [ VP_t + shields_t·m_p
///                                          + symbol_value_t(p) + K_t(p) ]
/// ```
///
/// `P_app(t)` is the probability the successor appears at all: one if it is
/// already face up in the structure, [`Expectations::p_hidden`] if it is in
/// the current age's unknown pool, that age's
/// [`duels_strategy::AgeSupply::plain_dealt_fraction`] if it belongs to an age
/// not yet dealt, and zero once it is in a city, in the discard pile, spent
/// under a wonder, or from an age that has come and gone without it. `K_t(p)`
/// is the coin cost the chain avoids, which is the one quantity
/// [`duels_core::cost::card_cost`] cannot answer — by the time the starter is
/// built it reports zero — so it comes from
/// [`duels_core::cost::card_cost_ignoring_chain`].
///
/// The base game has seventeen chain links (verified against
/// `data/cards.json`, including the two that skip an age: Palisade →
/// Fortifications and Tavern → Lighthouse), so the whole table is a handful of
/// cost calls.
#[derive(Debug, Clone)]
pub struct ChainTable {
    equity: [Box<[f64; NUM_CARDS]>; 2],
    /// Cards that start a chain at all, as a mask.
    starters: u128,
}

impl ChainTable {
    /// An all-zero table, for when the term is switched off.
    pub fn empty() -> ChainTable {
        ChainTable {
            equity: [Box::new([0.0; NUM_CARDS]), Box::new([0.0; NUM_CARDS])],
            starters: 0,
        }
    }

    /// Price every chain link in the game from the root position.
    pub fn of(
        state: &GameState,
        board: &Board,
        expected: &Expectations,
        take: &[TakeValue; 2],
    ) -> ChainTable {
        let m = masks();
        let gone = board.in_city | board.discard | board.fodder;
        let mut out = ChainTable::empty();

        for i in 0..NUM_CARDS {
            let starter = CardId::from_index(i);
            let Some(target) = starter.def().chain_to else {
                continue;
            };
            out.starters |= 1u128 << i;
            let bit = 1u128 << target.index();
            let p_app = if gone & bit != 0 {
                0.0
            } else if board.face_up & bit != 0 {
                1.0
            } else if board.unknown_pool & bit != 0 {
                // Guilds never chain, so the plain-pool probability is the
                // right one for every successor in the game.
                expected.p_hidden
            } else if target.def().age >= board.first_undealt_age {
                m.age_supply(target.def().age).plain_dealt_fraction()
            } else {
                // An earlier age's card that never appeared went back in the
                // box at setup.
                0.0
            };
            if p_app <= 0.0 {
                continue;
            }
            let def = target.def();
            for p in Player::ALL {
                let tv = &take[p.index()];
                let mut value = f64::from(def.victory_points)
                    + f64::from(def.coins) / 3.0
                    + tv.shields_value(def.shields);
                if let Some(sym) = def.science {
                    value += tv.symbol_value(sym);
                }
                let avoided = f64::from(cost::card_cost_ignoring_chain(state, p, target).coins);
                value += avoided * tv.coin_marginal;
                out.equity[p.index()][i] = p_app * CHAIN_MINE_SHARE * value;
            }
        }
        out
    }

    /// The equity of holding `card` as a chain starter, for `p`. Zero for a
    /// card that starts no chain.
    #[inline]
    pub fn equity(&self, p: Player, card: CardId) -> f64 {
        self.equity[p.index()][card.index()]
    }

    /// Every card that starts a chain, as a mask.
    #[inline]
    pub fn starters(&self) -> u128 {
        self.starters
    }
}

/// `CE(p)`: the forward value of the chain starters in `p`'s city whose
/// successor is still in the game.
///
/// `built(p)` is read on the post-action state so a candidate that takes a
/// starter is credited for it; whether the successor is still available is
/// read there too, so taking the successor itself stops paying equity for the
/// starter rather than double-counting it. Every *price* in the table is
/// root-fixed.
pub fn chain_equity(state: &GameState, p: Player, table: &ChainTable) -> f64 {
    let mut mask = state.player(p).built_mask() & table.starters();
    if mask == 0 {
        return 0.0;
    }
    let gone = state.player(Player::One).built_mask()
        | state.player(Player::Two).built_mask()
        | state.discard_mask()
        | state.wonder_fodder_mask();
    let mut out = 0.0;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let starter = CardId::from_index(i);
        let Some(target) = starter.def().chain_to else {
            continue;
        };
        if gone & (1u128 << target.index()) != 0 {
            continue;
        }
        out += table.equity(p, starter);
    }
    out
}

// ---------------------------------------------------------------------------
// The opponent's menu
// ---------------------------------------------------------------------------

/// The parts of [`crate::Config`] [`menu_term`] reads, snapshotted at the root
/// so the term keeps its original signature.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MenuOptions {
    /// What the menu falls back on when nothing on the board is affordable.
    pub floor: MenuFloor,
    /// `c_soft` in the soft affordability weight. Zero is the hard cutoff.
    pub afford_soft: f64,
}

/// Everything [`menu_term`] needs that is fixed at the root.
#[derive(Debug, Clone)]
pub struct MenuTables {
    /// Root pricing context per player.
    pub take: [TakeValue; 2],
    /// The chain table `v_q` prices chain starters against.
    pub chain: ChainTable,
    /// Cached `v_q(card)` for every card face up at the root, per player, so
    /// the usual case costs no work at all per chance outcome.
    cached: [Box<[f64; NUM_CARDS]>; 2],
    priced: u128,
    /// The root state, for pricing a card that only appears later.
    root_state: GameState,
    options: MenuOptions,
    /// Per-effect wonder prices, for [`MenuFloor::DiscardAndWonder`]. All zero
    /// unless something asked for them.
    wonders: WonderBudget,
}

impl MenuTables {
    /// The tables with nothing priced, for when the menu term is switched
    /// off. The chain table is still carried, because
    /// [`crate::terms`]'s chain-equity term uses it independently.
    pub fn unpriced(state: &GameState, take: [TakeValue; 2], chain: ChainTable) -> MenuTables {
        MenuTables {
            cached: [Box::new([0.0; NUM_CARDS]), Box::new([0.0; NUM_CARDS])],
            priced: 0,
            take,
            chain,
            root_state: *state,
            options: MenuOptions::default(),
            wonders: WonderBudget::empty(),
        }
    }

    /// Build the tables, pricing every card face up at the root for both
    /// players (at most twenty cards, so at most forty cost calls).
    pub fn of(
        state: &GameState,
        board: &Board,
        take: [TakeValue; 2],
        chain: ChainTable,
    ) -> MenuTables {
        MenuTables::with(
            state,
            board,
            take,
            chain,
            MenuOptions::default(),
            WonderBudget::empty(),
        )
    }

    /// [`MenuTables::of`] with the menu's own options and the wonder prices the
    /// [`MenuFloor::DiscardAndWonder`] floor needs.
    pub fn with(
        state: &GameState,
        board: &Board,
        take: [TakeValue; 2],
        chain: ChainTable,
        options: MenuOptions,
        wonders: WonderBudget,
    ) -> MenuTables {
        let mut cached = [Box::new([0.0; NUM_CARDS]), Box::new([0.0; NUM_CARDS])];
        let mut mask = board.face_up;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let card = CardId::from_index(i);
            for p in Player::ALL {
                cached[p.index()][i] = take[p.index()].value(state, card, &chain);
            }
        }
        MenuTables {
            cached,
            priced: board.face_up,
            take,
            chain,
            root_state: *state,
            options,
            wonders,
        }
    }

    /// `w_q`: the best wonder `q` could afford to build out of the position
    /// `next`, net of what it costs them, at root-fixed prices.
    ///
    /// `None` when they hold no unbuilt wonder they can pay for, or when the
    /// seven-wonder cap has already closed
    /// ([`terms::wonder_slots_left`]) — in which case the menu simply does not
    /// carry the entry, rather than carrying a zero, because "no wonder is
    /// available" and "an available wonder is worth nothing" are different
    /// positions.
    fn best_affordable_wonder(&self, next: &GameState, q: Player) -> Option<f64> {
        if terms::wonder_slots_left(next) <= 0.0 {
            return None;
        }
        let ps = next.player(q);
        let coins = ps.coins();
        let coin_marginal = self.take[q.index()].coin_marginal;
        let mut best: Option<f64> = None;
        for w in ps.wonders() {
            if ps.has_built_wonder(w) {
                continue;
            }
            let price = cost::wonder_cost(next, q, w).coins;
            if price > coins {
                continue;
            }
            let v = self.wonders.power(q, w) - f64::from(price) * coin_marginal;
            if best.is_none_or(|b| v > b) {
                best = Some(v);
            }
        }
        best
    }

    /// `v_q(card)`, from the cache when the card was already face up at the
    /// root and priced fresh against the root position otherwise.
    pub fn value(&self, p: Player, card: CardId) -> f64 {
        if self.priced & (1u128 << card.index()) != 0 {
            self.cached[p.index()][card.index()]
        } else {
            self.take[p.index()].value(&self.root_state, card, &self.chain)
        }
    }

    /// The chain table, for the terms that share it.
    #[inline]
    pub fn chain(&self) -> &ChainTable {
        &self.chain
    }

    /// One player's root pricing context, for the diagnostics.
    #[inline]
    pub fn take(&self, p: Player) -> &TakeValue {
        &self.take[p.index()]
    }
}

/// What the position `next` hands the player about to move, signed towards
/// `me`.
///
/// ```text
/// menu_q = τ · ln( Σ_{j ∈ A} exp(v_q(card_j) / τ) )
/// term   = ±λ · menu_q        (+ if `me` moves next, − if the opponent does)
/// ```
///
/// `A` is the set of accessible, face-up slots whose card `q` can pay for
/// right now. A softmax rather than a plain maximum on purpose: denying the
/// opponent's *second*-best option should still earn partial credit, because
/// the best one may well be gone by the time they get to it. `τ` sets how
/// hard the max is; at `τ → 0` it becomes one.
///
/// Both signs matter. The negative case is denial: leave a Palace face up and
/// affordable and this scores the gift. The positive case is the parity flip
/// the backlog calls out — a move that grants an extra turn leaves *me* to
/// move next, which turns a reveal that would have been a gift into a private
/// draw, and it falls out of the same expression with no special case.
///
/// Not commitment-scaled: what the opponent's next move is worth does not
/// depend on whether this player has a race plan.
pub fn menu_term(
    next: &GameState,
    me: Player,
    root_age: u8,
    tables: &MenuTables,
    w: &MenuWeights,
) -> f64 {
    if w.lambda == 0.0 || w.tau <= 0.0 {
        return 0.0;
    }
    // The stand-down rule: see the module docs.
    if next.is_over() || next.age() != root_age || next.phase() != Phase::Turn {
        return 0.0;
    }
    let q = next.current_player();
    let coins = next.player(q).coins();
    let soft = tables.options.afford_soft;

    // One pass to find the largest value, a second to sum the exponentials
    // shifted by it — the usual log-sum-exp guard, which also keeps the answer
    // finite when the whole menu is worthless.
    //
    // Two past the slot count so the floor entries fit: at most one discard and
    // one wonder.
    const CAP: usize = duels_core::layout::SLOTS + 2;
    let mut values: [f64; CAP] = [0.0; CAP];
    let mut afford: [f64; CAP] = [0.0; CAP];
    let mut n = 0usize;
    let mut best = f64::NEG_INFINITY;
    let mut mask = next.accessible_slots();
    while mask != 0 {
        let slot = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        let Some(card) = next.face_up_card(slot) else {
            continue;
        };
        let price = cost::card_cost(next, q, card).coins;
        let weight = if soft > 0.0 {
            afford_weight(coins, price, soft)
        } else if price > coins {
            continue;
        } else {
            1.0
        };
        let v = tables.value(q, card);
        values[n] = v;
        afford[n] = weight;
        n += 1;
        if v > best {
            best = v;
        }
    }

    // The floor. A turn never actually degrades to nothing: at minimum it is
    // worth the discard it can always take, and possibly a wonder the player
    // can already pay for. Without those entries an opponent one coin short of
    // affording anything reads identically to an opponent staring at a Palace,
    // which is wrong in both directions — and, worse, it flattens the reward
    // for taking their *last* affordable card, since the position after reads
    // as a hard zero either way.
    if tables.options.floor != MenuFloor::None {
        let d = f64::from(cost::discard_reward(next, q)) * tables.take[q.index()].coin_marginal;
        values[n] = d;
        afford[n] = 1.0;
        n += 1;
        if d > best {
            best = d;
        }
    }
    if tables.options.floor == MenuFloor::DiscardAndWonder {
        if let Some(v) = tables.best_affordable_wonder(next, q) {
            values[n] = v;
            afford[n] = 1.0;
            n += 1;
            if v > best {
                best = v;
            }
        }
    }

    if n == 0 {
        // Nothing on the table they can pay for and no floor asked for: their
        // turn degrades to a discard. Worth nothing rather than minus infinity.
        return 0.0;
    }
    // The `soft == 0` branch is the pre-existing expression, character for
    // character, so the hard cutoff stays bit-identical rather than
    // approximately identical: `1.0 * x` would in fact reproduce it, but that
    // is a claim about IEEE-754 rather than about this code, and
    // `tests/v4_identity.rs` should not have to rest on it.
    let sum: f64 = if soft > 0.0 {
        (0..n)
            .map(|i| afford[i] * ((values[i] - best) / w.tau).exp())
            .sum::<f64>()
            .max(f64::MIN_POSITIVE)
    } else {
        values[..n].iter().map(|v| ((v - best) / w.tau).exp()).sum()
    };
    let menu = best + w.tau * sum.ln();
    if q == me {
        w.lambda * menu
    } else {
        -w.lambda * menu
    }
}

/// How much weight a card the player is `coins − price` coins away from
/// affording carries on the menu.
///
/// The hard cutoff `menu_term` uses by default says a card one coin out of
/// reach is worth exactly nothing to the next mover, which is not true: they
/// can take a cheap card now and it will very often still be there, or the
/// board can hand them the coin. `w_j = σ((coins − price) / c_soft)` says it
/// smoothly instead, and at `c_soft → 0` it *is* the hard cutoff.
///
/// Only ever called with `c_soft > 0`.
#[inline]
fn afford_weight(coins: u16, price: u16, c_soft: f64) -> f64 {
    let slack = f64::from(coins) - f64::from(price);
    1.0 / (1.0 + (-slack / c_soft).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_core::data;
    use duels_core::testing::StateBuilder;

    /// The `CardId` with this slug, for the tests.
    fn card(id: &str) -> CardId {
        data::statics()
            .cards
            .iter()
            .position(|c| c.id == id)
            .map(CardId::from_index)
            .unwrap_or_else(|| panic!("no card {id:?}"))
    }

    /// Every chain link the base game prints, read straight out of the card
    /// data rather than from anybody's memory of the rulebook — including the
    /// two that skip an age, which `docs/strategy-backlog.md` flags as
    /// uncertain and which are real.
    #[test]
    fn the_chain_map_matches_the_card_data_including_the_age_skips() {
        let mut links: Vec<(&str, u8, &str, u8)> = Vec::new();
        for i in 0..NUM_CARDS {
            let c = CardId::from_index(i);
            if let Some(t) = c.def().chain_to {
                links.push((c.def().id, c.def().age, t.def().id, t.def().age));
            }
        }
        assert_eq!(links.len(), 17, "expected seventeen chain links: {links:?}");
        // Every successor names its predecessor back.
        for &(from, _, to, _) in &links {
            let target = card(to);
            assert_eq!(
                target.def().chain_from.map(|c| c.def().id),
                Some(from),
                "{to} does not chain back from {from}"
            );
        }
        // The two age-skipping links the backlog flags as uncertain.
        assert!(links.contains(&("palisade", 1, "fortifications", 3)));
        assert!(links.contains(&("tavern", 1, "lighthouse", 3)));
    }

    fn table_for(state: &GameState) -> (ChainTable, [TakeValue; 2]) {
        let board = Board::of(state);
        let supply = DevSupply::of(&board);
        let expected = Expectations::of(&board);
        let sm = MilSmoothing::of(10.0, 0.8, 0.35, 0.55);
        let cfg = Config::default();
        let guilds = GuildTable::of(state, &supply, &cfg.eval);
        let ctx = TakeContext {
            supply: &supply,
            smoothing: &sm,
            guild: &guilds,
        };
        let take =
            [Player::One, Player::Two].map(|p| TakeValue::of(state, p, ctx, &cfg, 1.0, (1.0, 1.0)));
        let chain = ChainTable::of(state, &board, &expected, &take);
        (chain, take)
    }

    /// The `shield_delta` table has to be long enough for the biggest gain a
    /// single card can produce, or a three-shield card in the hands of a
    /// Strategy holder would silently read as something smaller.
    #[test]
    fn the_shield_delta_table_covers_the_biggest_gain_any_card_can_make() {
        assert_eq!(
            MAX_SHIELD_STEP,
            usize::from(duels_strategy::max_single_shield_gain()),
            "the card data's largest single shield gain no longer matches the \
             table this module sizes for it"
        );
    }

    #[test]
    fn holding_a_chain_starter_is_worth_something_and_taking_the_successor_stops_paying() {
        // Scriptorium chains into Library. Age II, Library face up.
        let st = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "library"), (19, "quarry")])
            .built(Player::One, &["scriptorium"])
            .coins(Player::One, 20)
            .coins(Player::Two, 20)
            .current(Player::One)
            .build();
        let (chain, _) = table_for(&st);
        let with_starter = chain_equity(&st, Player::One, &chain);
        assert!(with_starter > 0.0, "scriptorium should carry equity");
        assert_eq!(chain_equity(&st, Player::Two, &chain), 0.0);

        // Once the Library is in a city the equity is spent, not doubled.
        let after = StateBuilder::new()
            .age(2)
            .open_slots(&[(19, "quarry")])
            .built(Player::One, &["scriptorium", "library"])
            .coins(Player::One, 20)
            .current(Player::One)
            .build();
        assert_eq!(chain_equity(&after, Player::One, &chain), 0.0);
    }

    #[test]
    fn a_successor_already_in_the_discard_pile_carries_no_equity() {
        let live = StateBuilder::new()
            .age(2)
            .open_slots(&[(18, "library"), (19, "quarry")])
            .built(Player::One, &["scriptorium"])
            .build();
        let dead = StateBuilder::new()
            .age(2)
            .open_slots(&[(19, "quarry")])
            .discard(&["library"])
            .built(Player::One, &["scriptorium"])
            .build();
        let (a, _) = table_for(&live);
        let (b, _) = table_for(&dead);
        assert!(chain_equity(&live, Player::One, &a) > 0.0);
        assert_eq!(chain_equity(&dead, Player::One, &b), 0.0);
    }

    #[test]
    fn the_menu_falls_when_the_best_card_on_offer_is_taken_away() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace"), (19, "clay-pool")])
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let board = Board::of(&st);
        let supply = DevSupply::of(&board);
        let expected = Expectations::of(&board);
        let sm = MilSmoothing::of(4.0, 0.8, 0.35, 0.55);
        let cfg = Config::default();
        let guilds = GuildTable::of(&st, &supply, &cfg.eval);
        let ctx = TakeContext {
            supply: &supply,
            smoothing: &sm,
            guild: &guilds,
        };
        let take =
            [Player::One, Player::Two].map(|p| TakeValue::of(&st, p, ctx, &cfg, 1.0, (1.0, 1.0)));
        let chain = ChainTable::of(&st, &board, &expected, &take);
        let tables = MenuTables::of(&st, &board, take, chain);
        let w = MenuWeights {
            lambda: 0.6,
            tau: 1.5,
        };

        // The Palace is worth far more than the Clay Pool, so a menu holding
        // both reads higher than one holding only the Clay Pool.
        assert!(
            tables.value(Player::Two, card("palace"))
                > tables.value(Player::Two, card("clay-pool"))
        );

        // Player One is to move here, so the menu is *theirs*: positive for
        // Player One and the same magnitude negated for Player Two.
        let mine = menu_term(&st, Player::One, 3, &tables, &w);
        let theirs = menu_term(&st, Player::Two, 3, &tables, &w);
        assert!(mine > 0.0, "player one moves next here: {mine}");
        assert!((mine + theirs).abs() < 1e-12, "{mine} vs {theirs}");

        // Take the Palace away and the same menu is worth strictly less.
        let thinner = StateBuilder::new()
            .age(3)
            .open_slots(&[(19, "clay-pool")])
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let after = menu_term(&thinner, Player::One, 3, &tables, &w);
        assert!(after < mine, "palace still on offer: {after} vs {mine}");

        // ...and the term stands down once the age has turned over.
        assert_eq!(menu_term(&st, Player::One, 1, &tables, &w), 0.0);
    }

    // -----------------------------------------------------------------
    // Round five: guild pricing, the menu floor, soft affordability
    // -----------------------------------------------------------------

    /// Everything [`menu_term`] needs, under an explicit configuration.
    fn tables_for(state: &GameState, cfg: Config) -> MenuTables {
        let board = Board::of(state);
        let supply = DevSupply::of_with(&board, cfg.supply_model);
        let expected = Expectations::of(&board);
        let sm = MilSmoothing::of(10.0, 0.8, 0.35, 0.55);
        let guilds = GuildTable::of(state, &supply, &cfg.eval);
        let ctx = TakeContext {
            supply: &supply,
            smoothing: &sm,
            guild: &guilds,
        };
        let take =
            [Player::One, Player::Two].map(|p| TakeValue::of(state, p, ctx, &cfg, 1.0, (1.0, 1.0)));
        let chain = ChainTable::of(state, &board, &expected, &take);
        let wonders = terms::WonderBudget::of(state, &take, &chain, &cfg.eval);
        MenuTables::with(
            state,
            &board,
            take,
            chain,
            MenuOptions {
                floor: cfg.menu_floor,
                afford_soft: cfg.menu_afford_soft,
            },
            wonders,
        )
    }

    /// **The bug, in one position.** A face-up Scientists Guild, in an Age III
    /// where both players have been collecting green cards, priced out at a
    /// strictly negative number — every guild prints zero points and zero
    /// coins, so the pricer saw a costly card with no value at all and the
    /// agent would neither take it nor deny it.
    #[test]
    fn a_face_up_guild_used_to_price_out_negative_and_now_does_not() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "scientists-guild"), (19, "palace")])
            .built(Player::One, &["workshop", "apothecary", "library"])
            .built(Player::Two, &["dispensary", "school", "laboratory"])
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();

        let unpriced = tables_for(
            &st,
            Config {
                guild_pricing: GuildPricing::Unpriced,
                ..Config::default()
            },
        );
        let projected = tables_for(
            &st,
            Config {
                guild_pricing: GuildPricing::Projected,
                ..Config::default()
            },
        );
        let guild = card("scientists-guild");

        let before = unpriced.value(Player::One, guild);
        assert!(
            before < 0.0,
            "the whole premise of this round: a face-up guild was worth {before}"
        );
        let after = projected.value(Player::One, guild);
        assert!(
            after > before,
            "guild pricing did not raise the guild's value: {before} -> {after}"
        );
        assert!(
            after > 0.0,
            "three green cards each side and the guild is still worth {after}"
        );

        // ...and it is the *guild* that moved, not everything. A plain card in
        // the same structure is priced identically under both.
        assert_eq!(
            unpriced.value(Player::One, card("palace")).to_bits(),
            projected.value(Player::One, card("palace")).to_bits(),
        );
    }

    /// Both players read the same projected basis, because the rule pays the
    /// guild's owner on the higher of the two counts whether or not it is their
    /// own. The race falls out of that, with no denial rule anywhere.
    #[test]
    fn a_guild_is_worth_the_same_projected_basis_to_whoever_ends_up_with_it() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "magistrate-s-guild"), (19, "palace")])
            .built(Player::One, &["altar", "baths", "theater"])
            .built(Player::Two, &["temple"])
            .coins(Player::One, 30)
            .coins(Player::Two, 30)
            .current(Player::One)
            .build();
        let t = tables_for(&st, Config::default());
        let guild = card("magistrate-s-guild");
        // Player Two leads on nothing here and still values the guild, because
        // it would pay them on Player One's three blue cards.
        assert!(t.value(Player::Two, guild) > 0.0);
        // The two differ only through the coin channel and the cost each
        // player faces, both of which are per-player quantities; the points
        // channel is identical.
        let g = t.take(Player::One).guild_value(guild);
        let h = t.take(Player::Two).guild_value(guild);
        assert!(
            (g - h).abs() < 1.0,
            "the two sides read wildly different guild values: {g} vs {h}"
        );
    }

    /// **The floor.** A position in which the next mover can afford nothing at
    /// all reads as a flat zero without it — indistinguishable from a position
    /// in which their turn is genuinely worthless.
    #[test]
    fn the_menu_floor_replaces_the_hard_zero_with_the_discard_the_player_can_always_take() {
        // Player Two is to move with no coins, facing a Palace they cannot
        // begin to pay for.
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace")])
            .built(Player::Two, &["tavern", "brewery"])
            .coins(Player::One, 20)
            .coins(Player::Two, 0)
            .current(Player::Two)
            .build();
        let w = MenuWeights {
            lambda: 0.6,
            tau: 1.5,
        };

        let none = tables_for(&st, Config::default());
        assert_eq!(
            menu_term(&st, Player::One, 3, &none, &w),
            0.0,
            "test setup: nothing here is affordable, so the old menu is flat zero"
        );

        let floored = tables_for(
            &st,
            Config {
                menu_floor: MenuFloor::Discard,
                ..Config::default()
            },
        );
        let with_floor = menu_term(&st, Player::One, 3, &floored, &w);
        assert!(
            with_floor < 0.0,
            "Player Two is to move, so their discard is a cost to Player One: {with_floor}"
        );

        // ...and the floor is worth more to a yellow-heavy city, which is the
        // whole point of reading `discard_reward` rather than a constant.
        let poorer = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace")])
            .coins(Player::One, 20)
            .coins(Player::Two, 0)
            .current(Player::Two)
            .build();
        let poor_tables = tables_for(
            &poorer,
            Config {
                menu_floor: MenuFloor::Discard,
                ..Config::default()
            },
        );
        assert!(
            menu_term(&poorer, Player::One, 3, &poor_tables, &w) > with_floor,
            "two commercial cards must make the fallback discard worth more"
        );
    }

    /// The wonder half of the floor: a player who can afford a wonder is not
    /// starved even when the structure is out of reach.
    #[test]
    fn the_wonder_floor_notices_a_wonder_the_starved_player_can_still_afford() {
        let st = StateBuilder::new()
            .age(3)
            .open_slots(&[(18, "palace")])
            .wonders(Player::Two, &["the-pyramids"])
            .built(
                Player::Two,
                &["quarry", "stone-pit", "shelf-quarry", "press"],
            )
            .coins(Player::One, 20)
            .coins(Player::Two, 0)
            .current(Player::Two)
            .build();
        let w = MenuWeights {
            lambda: 0.6,
            tau: 1.5,
        };
        let cfg = |floor| Config {
            menu_floor: floor,
            ..Config::default()
        };
        let discard = tables_for(&st, cfg(MenuFloor::Discard));
        let both = tables_for(&st, cfg(MenuFloor::DiscardAndWonder));
        let a = menu_term(&st, Player::One, 3, &discard, &w);
        let b = menu_term(&st, Player::One, 3, &both, &w);
        assert!(
            b < a,
            "the Pyramids are free to this city and worth nine points, so the \
             wonder entry must make Player Two's turn look better (and so the \
             term, signed towards Player One, smaller): {a} vs {b}"
        );
    }

    /// Soft affordability lets a card the player is narrowly short on carry
    /// partial weight — which the hard cutoff cannot express at all: under it,
    /// an expensive card the next mover cannot *quite* pay for is exactly as
    /// good as no card.
    ///
    /// The test is that the Palace's presence in the structure changes what the
    /// position is worth. Under the hard cutoff it provably does not.
    #[test]
    fn soft_affordability_gives_a_narrowly_unaffordable_card_partial_weight() {
        let with_palace = |slots: &[(u8, &str)]| {
            StateBuilder::new()
                .age(3)
                .open_slots(slots)
                .coins(Player::One, 20)
                // Enough for the Clay Pool, which is free, and three coins
                // short of the Palace, which is not.
                .coins(Player::Two, 3)
                .current(Player::Two)
                .build()
        };
        let rich = with_palace(&[(18, "palace"), (19, "clay-pool")]);
        let thin = with_palace(&[(19, "clay-pool")]);
        let w = MenuWeights {
            lambda: 0.6,
            tau: 1.5,
        };
        let hard = Config::default();
        let soft = Config {
            menu_afford_soft: 3.0,
            ..Config::default()
        };
        // The setup: the Palace really is out of reach.
        assert!(cost::card_cost(&rich, Player::Two, card("palace")).coins > 3);

        // Under the hard cutoff the Palace is invisible: the two positions read
        // *identically*, bit for bit.
        let t = tables_for(&rich, hard);
        assert_eq!(
            menu_term(&rich, Player::One, 3, &t, &w).to_bits(),
            menu_term(&thin, Player::One, 3, &t, &w).to_bits(),
            "an unaffordable Palace must be worth exactly nothing under the \
             hard cutoff, or this test is not about the cutoff"
        );

        // Under the soft one it is not.
        let t = tables_for(&rich, soft);
        let a = menu_term(&rich, Player::One, 3, &t, &w);
        let b = menu_term(&thin, Player::One, 3, &t, &w);
        assert!(
            a < b,
            "a Palace three coins out of reach must still make Player Two's \
             menu better, and so the term (signed towards Player One) smaller: \
             {a} vs {b}"
        );

        // The weight itself: a card exactly affordable sits at one half, and
        // the function is monotone in the slack. Half rather than one is the
        // formula's own choice and it deflates every menu uniformly; what it
        // buys is that "one coin short" and "ten coins short" stop being the
        // same position.
        assert!((afford_weight(5, 5, 2.0) - 0.5).abs() < 1e-12);
        assert!(afford_weight(6, 5, 2.0) > 0.5);
        assert!(afford_weight(4, 5, 2.0) < 0.5);
        assert!(afford_weight(4, 5, 2.0) > afford_weight(0, 5, 2.0));
    }
}
