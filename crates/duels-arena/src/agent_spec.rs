//! Agent construction from a *specification string*: either a bare name
//! (`"mcts-uct"`, looked up unmodified via [`crate::agent_registry`]) or a
//! name plus `key=value` parameters (`"mcts-uct:exploration=1.2"`,
//! `"alphabeta:max_depth=10,rollouts=4"`) that build one specific agent
//! crate's own `Config`/`Weights` type explicitly.
//!
//! This generalizes the ad hoc parser `examples/ab_lab.rs` built for the
//! alphabeta tuning investigation into `duels-arena` proper, so any future
//! ablation (e.g. `"mcts-uct:rollout=uniform"` vs `"mcts-uct:rollout=biased"`)
//! can be benchmarked with the regular `duels-arena match` CLI — no new
//! registry code needed for a parameter sweep, only for a genuinely new agent
//! crate (see [`crate::agent_registry`]).
//!
//! # Syntax
//!
//! ```text
//! <name>                       -- bare name, identical to agent_registry::make_agent
//! <name>:<key>=<value>[,<key>=<value>...]
//! ```
//!
//! An empty parameter list (`"mcts-uct:"`) is accepted and equivalent to the
//! bare name; unknown agent names, unknown keys, and unparsable values are
//! all reported as `Err` rather than panicking (unlike `ab_lab`'s original
//! `parse_config`, which is fine for a one-off harness but not for library
//! code the CLI depends on).
//!
//! # Supported agents and keys
//!
//! * `alphabeta` -- every [`duels_agent_alphabeta::Config`] field: `base`
//!   (`v1`/`default`), `max_depth`/`depth`, `chance_cap`/`cap`, `tt_bits`,
//!   `tt`, `star1`, `order` (`static`/`none`/`lookahead`/`priors`), `rollouts`,
//!   `rollout_blend`/`blend`, `rollout_cap`/`cap-rollouts`,
//!   `rollout_common_seed`/`crn`, `policy` (`uniform`/`biased`), `greedy`,
//!   `metric` (`margin:<clamp>` or `outcome:<scale>`), `weights`
//!   (`v1`/`default`/`score-only`), the individual evaluation weights `card`,
//!   `coin`, `breadth`, `shield`, `threat`, and the root-ensembling pair
//!   `root_determinizations`/`dets` and `ensemble_exact_root`/`exact`.
//! * `mcts-uct` -- every [`duels_agent_mcts_uct::Config`] field:
//!   `exploration`/`c`, `rollout` (`uniform`/`biased`/`smart`), `race`
//!   (`neutral`/`mild`/`medium`/`strong`/`tier1_only`), `chance_widen_c`,
//!   `chance_widen_alpha`, `max_rollout_plies`, `time_check_interval`,
//!   `root_determinizations`/`dets` and `prior` (`none`, `expansion_order`,
//!   or `progressive_bias:<weight>`).
//! * `mcts-eval` -- the same search, keyed identically (`exploration`/`c`,
//!   `rollout`, `race`, `chance_widen_c`, `chance_widen_alpha`,
//!   `max_rollout_plies`, `time_check_interval`,
//!   `root_determinizations`/`dets`, `prior`), plus the keys that are its
//!   own: `leaf` (`rollout`, `static`, `truncated:<plies>`,
//!   `blend:<weight>` — `blend:0.5` by default — and the two opt-in
//!   `duels-value` learned leaves, `learned` and `learned_blend:<weight>`)
//!   and `base`
//!   (`default`, or `rollout` for [`duels_agent_mcts_eval::Config::rollout_base`],
//!   the pure-playout `c = 1.0` control that is `mcts-uct` move for move).
//!   By default this agent tracks `duels_eval::Config::default()` live
//!   rather than pinning a generation, and its own spec string records the
//!   whole evaluation configuration it used — see its crate docs for why
//!   that is the opposite choice from `mcts-uct`'s and must not be "fixed"
//!   into a permanent pin. `eval=vN` (e.g. `eval=v6`) is the one deliberate,
//!   A/B-testing-only exception: it pins this one agent instance to a frozen
//!   `duels_eval::Config::vN()` snapshot so it can be matched directly, in one
//!   binary, against a live (unpinned) `mcts-eval` — see
//!   [`duels_agent_mcts_eval::Config::eval_override`]. `value_sum=serial` is
//!   the analogous exception for the learned leaves' arithmetic: it selects
//!   `duels_value::Summation::Serial`, the accumulation order that predates
//!   the four-way unroll, so the two can be matched directly in one binary.
//!   `value_sum=axpy` is the same idea for
//!   `duels_value::Summation::TransposedAxpy`, the transposed-`w1` accumulator
//!   — see that type's docs for why it targets the memory-bound half of the
//!   cost the unroll did not reach. All three are read only by a learned leaf
//!   and so cannot move the default.
//! * `mcts-value` -- the same search again, keyed identically to `mcts-eval`
//!   (including `leaf`, `value_sum`, `eval=vN` and the `duels-eval` scalar
//!   fallthrough), with a different `base` set: `eval` selects
//!   [`duels_agent_mcts_value::Config::eval_base`], which is `mcts-eval` at
//!   its default, and `rollout` selects
//!   [`duels_agent_mcts_value::Config::rollout_base`], which is `mcts-uct`.
//!   Both are asserted move-for-move against verbatim frozen copies inside
//!   that agent crate, so the whole ablation chain runs in one binary:
//!   `mcts-value` vs `mcts-value:base=eval` vs `mcts-value:base=rollout`.
//!   Note the reversal against `mcts-eval`: here `value_sum` is on the
//!   default path and the `duels-eval` keys are not, because the default leaf
//!   reads no hand-crafted evaluation at all. Also unique to this agent:
//!   `objective`/`obj` (`win`/`science`/`military`/`civilian`), which selects
//!   [`duels_agent_mcts_value::Objective`] -- `win` (the default) rewards any
//!   victory, and the other three build a specialist that is only rewarded
//!   for one victory kind, reusing the same trained weights with no retrain.
//! * `phased` -- `base` (`v1`..`v8`/`default`), the
//!   science ladder's individual rungs (`ladder1`..`ladder5`) and the leaf
//!   temperature (`temp1`/`temp2`/`temp3`), guild pricing
//!   (`guild`, `unpriced`/`projected`) and the guild projection weight
//!   (`guildproj`), the menu floor (`menufloor`, `none`/`discard`/
//!   `discardwonder`), the menu's soft affordability width (`afford`), the
//!   supply weighting (`supply`, `raw`/`dealt`), the yellow-density term
//!   (`yellow`, with `discardrate`), the pending-effect model
//!   (`pending`, `unresolved`/`completed`), the wonder model (`wonder`,
//!   `flat`/`budget`/`rationed`, with `wturns`, `wextra` and the rationed
//!   model's reference build probability `wonder_p_build_ref`/`pref`), the
//!   science ladder's symbol-reachability model (`reach_model`/`reach`,
//!   `optimistic`/`structure`), the flat model's extra-turn
//!   premium (`wonder_extra_turn_premium`/`wprem`), the destroy-replacement
//!   discount (`destroy_replace`/`destroyrepl`, `on`/`off`), the terminal rails
//!   (`rails` `on`/`off`, `imminent`), the menu's shield price
//!   (`shield_price`/`shieldprice`, `onesided`/`diff`), the military
//!   smoothing horizon (`horizon`, a number of rounds or `supply` for the old
//!   supply-wide width), the production lock-in multiplier
//!   (`production_lock_in`/`lockin`), the three model switches
//!   `military_model`/`mil`, `coin_model`/`coin` and `economy_model`/`econ`,
//!   `blend` (`on`/`off`), the two forward-looking terms' weights
//!   `menu_lambda`/`lambda`, `menu_tau`/`tau` and `chain_equity`/`chaineq`,
//!   the band-model shape (`military_band`/`band`, `military_loot`/`loot`,
//!   `military_sigma_scale`/`kappa`, `military_sigma_min`,
//!   `military_logistic_scale`), the smooth coin model's `coin_smooth_beta`
//!   /`beta` and `coin_smooth_ref`/`cref`, `resource_bill`/`bill`, the
//!   `next_age_start` array as `start1`/`start2`/`start3`, the individual
//!   weights `military_position`, `vp_projection`, `development`,
//!   `science_ladder`, `deny`, `deny_chain_gift`, `wonder_potential`, the
//!   science pair-threat weight (`science_pair_threat`/`pairthreat`) and the
//!   dead-race gate (`dead_race_scale`/`dead`), the forward token-equity term
//!   (`token_equity`/`tokeneq`), the right-to-move term (`to_move`/`tomove`),
//!   the value-scale knob (`value_scale`/`scale`), and the count-priced menu
//!   switch (`count`/`count_pricing`, `unpriced`/`counted`).
//!
//! Every parameterisable agent on the roster is listed above: `random`,
//! `greedy` and `greedy-ev` were retired (see `docs/milestones.md`), so
//! their names — and the `greedy`/`greedy-ev` `EvalWeights` parsers that used
//! to live here — are gone rather than silently accepted.
//!
//! # Examples
//!
//! ```
//! use duels_arena::agent_spec::make_agent_from_spec;
//!
//! let a = make_agent_from_spec("mcts-uct:exploration=1.2", 1).unwrap();
//! let b = make_agent_from_spec("alphabeta:max_depth=10,rollouts=4", 1).unwrap();
//! let c = make_agent_from_spec("phased", 1).unwrap(); // bare name, unchanged
//! assert_eq!(a.spec().name, "mcts-uct");
//! assert_eq!(b.spec().name, "alphabeta");
//! assert_eq!(c.spec().name, "phased");
//! ```

use duels_agent_alphabeta::{eval, playout, AlphaBetaAgent, Config as AlphaBetaConfig};
use duels_agent_mcts_eval::{
    Config as MctsEvalConfig, LeafValue, MctsEvalAgent, PriorMode as EvalPriorMode,
    RaceWeights as EvalRaceWeights, RolloutWeights as EvalRolloutWeights,
};
use duels_agent_mcts_uct::{
    Config as MctsConfig, MctsAgent, PriorMode, RaceWeights, RolloutWeights,
};
use duels_agent_mcts_value::{
    Config as MctsValueConfig, LeafValue as ValueLeafValue, MctsValueAgent,
    Objective as ValueObjective, PriorMode as ValuePriorMode, RaceWeights as ValueRaceWeights,
    RolloutWeights as ValueRolloutWeights,
};
use duels_agent_phased::{
    Blend as PhasedBlend, CoinModel, Config as PhasedConfig, EconomyModel, GuildPricing, MenuFloor,
    MenuShieldPricing, MilitaryModel, PendingModel, PhasedAgent, RailModel, SupplyModel,
    WonderModel,
};
use duels_agents_api::Agent;
// `ReachModel` is round nine's addition and `duels-agent-phased` does not
// re-export it; taken straight from the library below the agents, exactly as
// `CountPricing` already is.
use duels_core::scoring::VictoryKind;
use duels_eval::{CountPricing, ReachModel, ScienceProgress};

use crate::agent_registry::{make_agent, KNOWN_AGENTS};

/// Construct the `Agent` a specification string names, seeded from `seed`.
/// See the module docs for the syntax and the per-agent keys supported.
pub fn make_agent_from_spec(spec: &str, seed: u64) -> Result<Box<dyn Agent + Send>, String> {
    let Some((name, params)) = spec.split_once(':') else {
        return make_agent(spec, seed);
    };

    match name {
        "alphabeta" => {
            let cfg = AlphaBetaConfig {
                seed,
                ..parse_alphabeta_config(params)?
            };
            Ok(Box::new(AlphaBetaAgent::with_config(cfg)))
        }
        "mcts-uct" => {
            let cfg = parse_mcts_config(params)?;
            Ok(Box::new(MctsAgent::with_config(seed, cfg)))
        }
        "mcts-eval" => {
            let cfg = parse_mcts_eval_config(params)?;
            Ok(Box::new(MctsEvalAgent::with_config(seed, cfg)))
        }
        "mcts-value" => {
            let cfg = parse_mcts_value_config(params)?;
            Ok(Box::new(MctsValueAgent::with_config(seed, cfg)))
        }
        "phased" => {
            let cfg = parse_phased_config(params)?;
            Ok(Box::new(PhasedAgent::with_config(seed, cfg)))
        }
        other => Err(format!(
            "unknown agent \"{other}\" in spec \"{spec}\" (known agents: {})",
            KNOWN_AGENTS.join(", ")
        )),
    }
}

/// Split a `key=value,key=value` parameter list into pairs, ignoring empty
/// segments so a trailing/empty parameter string (`"name:"`) is valid.
fn parse_params(params: &str) -> Result<Vec<(&str, &str)>, String> {
    params
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            kv.split_once('=')
                .ok_or_else(|| format!("expected \"key=value\", got \"{kv}\""))
        })
        .collect()
}

/// Parse one `value` into `T`, tagging a failure with which `key` it was for.
fn parse_field<T: std::str::FromStr>(key: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value \"{value}\" for key \"{key}\""))
}

/// Parse an `alphabeta:...` parameter list into a [`AlphaBetaConfig`] (the
/// `seed` field is left at its default; callers overwrite it). Generalizes
/// `examples/ab_lab.rs`'s original `parse_config`.
pub fn parse_alphabeta_config(params: &str) -> Result<AlphaBetaConfig, String> {
    let mut cfg = AlphaBetaConfig::default();
    let mut w = cfg.weights;
    for (k, v) in parse_params(params)? {
        match k {
            "base" => match v {
                "v1" => {
                    cfg = AlphaBetaConfig::v1();
                    w = cfg.weights;
                }
                "default" => {}
                other => return Err(format!("alphabeta: unknown base \"{other}\"")),
            },
            "max_depth" | "depth" => cfg.max_depth = parse_field(k, v)?,
            "chance_cap" | "cap" => cfg.chance_cap = parse_field(k, v)?,
            "tt_bits" => cfg.tt_bits = parse_field(k, v)?,
            "tt" => cfg.use_tt = parse_field(k, v)?,
            "star1" => cfg.star1 = parse_field(k, v)?,
            "rollouts" => cfg.rollouts = parse_field(k, v)?,
            "rollout_blend" | "blend" => cfg.rollout_blend = parse_field(k, v)?,
            "rollout_cap" | "cap-rollouts" => cfg.rollout_cap = parse_field(k, v)?,
            "rollout_common_seed" | "crn" => cfg.rollout_common_seed = parse_field(k, v)?,
            "root_determinizations" | "dets" => cfg.root_determinizations = parse_field(k, v)?,
            "ensemble_exact_root" | "exact" => cfg.ensemble_exact_root = parse_field(k, v)?,
            "policy" => {
                cfg.rollout_policy = match v {
                    "uniform" => playout::PolicyWeights::UNIFORM,
                    "biased" => playout::PolicyWeights::BIASED,
                    other => return Err(format!("alphabeta: unknown policy \"{other}\"")),
                };
            }
            "greedy" => cfg.rollout_policy.greedy = parse_field(k, v)?,
            "order" => {
                cfg.order_moves = v != "none";
                cfg.order_lookahead = v == "lookahead";
                cfg.order_priors = v == "priors";
            }
            "weights" => {
                w = match v {
                    "score-only" => eval::Weights::SCORE_ONLY,
                    "v1" => eval::Weights::V1,
                    "default" => eval::Weights::DEFAULT,
                    other => return Err(format!("alphabeta: unknown weights \"{other}\"")),
                };
            }
            "metric" => {
                cfg.rollout_metric = match v.split_once(':') {
                    Some(("margin", c)) => playout::Metric::Margin {
                        clamp: parse_field("metric", c)?,
                    },
                    Some(("outcome", c)) => playout::Metric::Outcome {
                        scale: parse_field("metric", c)?,
                    },
                    _ => {
                        return Err(format!(
                            "alphabeta: metric must be \"margin:<clamp>\" or \"outcome:<scale>\", got \"{v}\""
                        ))
                    }
                };
            }
            "card" => w.card_in_city = parse_field(k, v)?,
            "coin" => w.coin = parse_field(k, v)?,
            "breadth" => w.resource_breadth = parse_field(k, v)?,
            "shield" => w.shield = parse_field(k, v)?,
            "threat" => w.capital_threat = parse_field(k, v)?,
            other => return Err(format!("alphabeta: unknown key \"{other}\"")),
        }
    }
    cfg.weights = w;
    Ok(cfg)
}

/// Parse a `mcts-uct:...` parameter list into a [`MctsConfig`].
pub fn parse_mcts_config(params: &str) -> Result<MctsConfig, String> {
    let mut cfg = MctsConfig::default();
    for (k, v) in parse_params(params)? {
        match k {
            "exploration" | "c" => cfg.exploration = parse_field(k, v)?,
            "chance_widen_c" => cfg.chance_widen_c = parse_field(k, v)?,
            "chance_widen_alpha" => cfg.chance_widen_alpha = parse_field(k, v)?,
            "max_rollout_plies" => cfg.max_rollout_plies = parse_field(k, v)?,
            "time_check_interval" => cfg.time_check_interval = parse_field(k, v)?,
            "root_determinizations" | "dets" => cfg.root_determinizations = parse_field(k, v)?,
            "rollout" => {
                cfg.rollout = match v {
                    "uniform" => RolloutWeights::UNIFORM,
                    "biased" => RolloutWeights::BIASED,
                    "smart" => RolloutWeights::SMART,
                    other => return Err(format!("mcts-uct: unknown rollout \"{other}\"")),
                };
            }
            "race" => {
                cfg.race = match v {
                    "neutral" | "off" => RaceWeights::NEUTRAL,
                    "tier1" | "tier1_only" => RaceWeights::TIER1_ONLY,
                    "mild" => RaceWeights::mild(),
                    "medium" => RaceWeights::MEDIUM,
                    "strong" => RaceWeights::strong(),
                    other => {
                        return Err(format!(
                            "mcts-uct: unknown race \"{other}\" (expected \"neutral\", \
                             \"mild\", \"medium\", \"strong\", or \"tier1_only\")"
                        ))
                    }
                };
            }
            "prior" => {
                // `progressive_bias` carries a weight, spelled with a colon
                // (`prior=progressive_bias:1.5`) so it survives the `,`/`=`
                // splitting, exactly as `alphabeta`'s `metric` key does. A
                // bare `progressive_bias` takes the mode's default weight.
                cfg.prior = match v.split_once(':') {
                    Some(("progressive_bias" | "bias", w)) => PriorMode::ProgressiveBias {
                        weight: parse_field("prior", w)?,
                    },
                    None => match v {
                        "none" | "off" => PriorMode::None,
                        "expansion_order" | "order" => PriorMode::ExpansionOrder,
                        "progressive_bias" | "bias" => PriorMode::ProgressiveBias { weight: 1.0 },
                        other => {
                            return Err(format!(
                                "mcts-uct: unknown prior \"{other}\" (expected \"none\", \
                                 \"expansion_order\", or \"progressive_bias[:<weight>]\")"
                            ))
                        }
                    },
                    Some((other, _)) => {
                        return Err(format!(
                            "mcts-uct: prior \"{other}\" takes no weight (only \
                             \"progressive_bias:<weight>\" does)"
                        ))
                    }
                };
            }
            "leaf" | "evalgen" | "eval_generation" => {
                // Both used to be `mcts-uct` keys, while a `duels-eval` leaf
                // value was an opt-in option here. That machinery now lives
                // in its own agent; point a caller at it rather than silently
                // ignoring a key whose whole purpose was to change the leaf.
                return Err(format!(
                    "mcts-uct: \"{k}\" moved to the \"mcts-eval\" agent, which is this \
                     search with a duels-eval leaf value (try \"mcts-eval\", or \
                     \"mcts-eval:leaf={v}\" / \"mcts-eval:base=rollout\")"
                ));
            }
            other => return Err(format!("mcts-uct: unknown key \"{other}\"")),
        }
    }
    Ok(cfg)
}

/// Parse a `mcts-eval:...` parameter list into a [`MctsEvalConfig`].
///
/// The search keys are `mcts-uct`'s, spelled identically, because it is the
/// same search — `base=rollout` selects
/// [`duels_agent_mcts_eval::Config::rollout_base`], the pure-playout `c = 1.0`
/// control that agent proves is `mcts-uct` move for move, and is the arm every
/// strength claim about the leaf value is measured against.
///
/// By default `mcts-eval` scores against `duels_eval::Config::default()`
/// live and pins nothing — its own spec string records the whole evaluation
/// configuration in force, which is what makes a results file interpretable
/// after a later `duels-eval` round. `eval=vN` is the one deliberate
/// exception: it pins `duels_agent_mcts_eval::Config::eval_override` to a
/// frozen `duels_eval::Config::vN()` snapshot, purely so a `duels-eval`
/// change can be A/B tested directly in one binary — `mcts-eval` (live, new)
/// against `mcts-eval:eval=vN` (pinned, old) — rather than only indirectly,
/// through an unrelated anchor agent. Not a second production default; see
/// [`duels_agent_mcts_eval::Config::eval_override`]'s own docs.
pub fn parse_mcts_eval_config(params: &str) -> Result<MctsEvalConfig, String> {
    let mut cfg = MctsEvalConfig::default();
    for (k, v) in parse_params(params)? {
        match k {
            "base" => match v {
                // Deliberately first-listed and last-applied like
                // `alphabeta`'s and `phased`'s `base`: keys after it override.
                "rollout" | "mcts-uct" => cfg = MctsEvalConfig::rollout_base(),
                "default" => {}
                other => {
                    return Err(format!(
                        "mcts-eval: unknown base \"{other}\" (expected \"default\" or \
                         \"rollout\")"
                    ))
                }
            },
            // A/B-testing override only -- see this function's doc comment
            // and `duels_agent_mcts_eval::Config::eval_override`. Add a new
            // arm here the same way `phased`'s `base=` key gains one, each
            // time `duels-eval` freezes a new version.
            "eval" => {
                cfg.eval_override = Some(match v {
                    "default" | "live" => {
                        return Err(
                            "mcts-eval: eval=default/live is the same as omitting the key -- \
                             pass no eval key at all to track duels-eval live"
                                .to_string(),
                        )
                    }
                    "v1" => duels_eval::Config::v1(),
                    "v2" => duels_eval::Config::v2(),
                    "v3" => duels_eval::Config::v3(),
                    "v4" => duels_eval::Config::v4(),
                    "v5" => duels_eval::Config::v5(),
                    "v6" => duels_eval::Config::v6(),
                    "v7" => duels_eval::Config::v7(),
                    "v8" => duels_eval::Config::v8(),
                    "v9" => duels_eval::Config::v9(),
                    other => {
                        return Err(format!(
                            "mcts-eval: unknown eval generation \"{other}\" (expected v1-v9)"
                        ))
                    }
                });
            }
            // `duels-eval`'s round-ten option, which lives on
            // `duels_eval::Config::blend` rather than on `EvalWeights` and so
            // cannot ride the scalar fallthrough at the bottom of this match.
            //
            // Note that passing it *at all* pins `eval_override`, including
            // `sciprog=root`. That is deliberate and is what makes the A/B
            // symmetric: `mcts-eval:sciprog=leaf` against
            // `mcts-eval:sciprog=root` differs in exactly one field, where
            // against a bare `mcts-eval` it would also differ in whether the
            // evaluation is pinned at all (`tree`'s
            // `eval_override_none_is_bit_identical_to_pinning_todays_live_default`
            // says that pinning today's default is behaviourally free, so the
            // control arm is still today's agent).
            "sci_progress" | "sciprog" => {
                let eval = cfg
                    .eval_override
                    .get_or_insert_with(duels_eval::Config::default);
                eval.blend.science_progress = match v {
                    "root" | "off" => duels_eval::ScienceProgress::Root,
                    "leaf" | "on" => duels_eval::ScienceProgress::Leaf,
                    other => {
                        return Err(format!(
                            "mcts-eval: unknown sci_progress \"{other}\" (expected \"root\" \
                             or \"leaf\")"
                        ))
                    }
                };
            }
            "exploration" | "c" => cfg.exploration = parse_field(k, v)?,
            "chance_widen_c" => cfg.chance_widen_c = parse_field(k, v)?,
            "chance_widen_alpha" => cfg.chance_widen_alpha = parse_field(k, v)?,
            "max_rollout_plies" => cfg.max_rollout_plies = parse_field(k, v)?,
            "time_check_interval" => cfg.time_check_interval = parse_field(k, v)?,
            "root_determinizations" | "dets" => cfg.root_determinizations = parse_field(k, v)?,
            "rollout" => {
                cfg.rollout = match v {
                    "uniform" => EvalRolloutWeights::UNIFORM,
                    "biased" => EvalRolloutWeights::BIASED,
                    "smart" => EvalRolloutWeights::SMART,
                    other => return Err(format!("mcts-eval: unknown rollout \"{other}\"")),
                };
            }
            "race" => {
                cfg.race = match v {
                    "neutral" | "off" => EvalRaceWeights::NEUTRAL,
                    "tier1" | "tier1_only" => EvalRaceWeights::TIER1_ONLY,
                    "mild" => EvalRaceWeights::mild(),
                    "medium" => EvalRaceWeights::MEDIUM,
                    "strong" => EvalRaceWeights::strong(),
                    other => {
                        return Err(format!(
                            "mcts-eval: unknown race \"{other}\" (expected \"neutral\", \
                             \"mild\", \"medium\", \"strong\", or \"tier1_only\")"
                        ))
                    }
                };
            }
            "prior" => {
                cfg.prior = match v.split_once(':') {
                    Some(("progressive_bias" | "bias", w)) => EvalPriorMode::ProgressiveBias {
                        weight: parse_field("prior", w)?,
                    },
                    None => match v {
                        "none" | "off" => EvalPriorMode::None,
                        "expansion_order" | "order" => EvalPriorMode::ExpansionOrder,
                        "progressive_bias" | "bias" => {
                            EvalPriorMode::ProgressiveBias { weight: 1.0 }
                        }
                        other => {
                            return Err(format!(
                                "mcts-eval: unknown prior \"{other}\" (expected \"none\", \
                                 \"expansion_order\", or \"progressive_bias[:<weight>]\")"
                            ))
                        }
                    },
                    Some((other, _)) => {
                        return Err(format!(
                            "mcts-eval: prior \"{other}\" takes no weight (only \
                             \"progressive_bias:<weight>\" does)"
                        ))
                    }
                };
            }
            "leaf" => {
                // `truncated` and `blend` each carry a parameter, spelled with
                // a colon (`leaf=trunc:8`, `leaf=blend:0.3`) so it survives
                // the `,`/`=` splitting, exactly as `prior` and `alphabeta`'s
                // `metric` do. A bare `blend` is the default weight.
                //
                // Note that `leaf` alone does **not** rescale `exploration`:
                // the two move together in `Config::default` and a caller
                // changing one has to decide about the other, which is the
                // whole reason this configuration is its own agent (see
                // `duels_agent_mcts_eval::LeafValue::Blend`).
                cfg.leaf = match v.split_once(':') {
                    Some(("trunc" | "truncated", p)) => LeafValue::Truncated {
                        plies: parse_field("leaf", p)?,
                    },
                    Some(("blend", w)) => {
                        // Range-checked, unlike the other float keys in this
                        // module, and for a specific reason: a blend weight
                        // outside `[0, 1]` makes the leaf value stop being a
                        // probability, which silently violates the value
                        // convention every node in that tree accumulates
                        // (`duels_agent_mcts_eval::tree`). An out-of-range
                        // *evaluation* weight elsewhere is merely a strange
                        // agent; this one is a broken search that would still
                        // produce a plausible-looking win rate.
                        let weight: f64 = parse_field("leaf", w)?;
                        if !(0.0..=1.0).contains(&weight) {
                            return Err(format!(
                                "mcts-eval: leaf blend weight must be in [0, 1], got \"{w}\" \
                                 (outside it the leaf value is not a probability)"
                            ));
                        }
                        LeafValue::Blend { weight }
                    }
                    // The learned leaves (`duels-value`) mirror `static` and
                    // `blend`, and their blend weight is range-checked for
                    // exactly the same reason.
                    Some(("learned_blend" | "lblend", w)) => {
                        let weight: f64 = parse_field("leaf", w)?;
                        if !(0.0..=1.0).contains(&weight) {
                            return Err(format!(
                                "mcts-eval: leaf blend weight must be in [0, 1], got \"{w}\" \
                                 (outside it the leaf value is not a probability)"
                            ));
                        }
                        LeafValue::LearnedBlend { weight }
                    }
                    None => match v {
                        "rollout" | "off" => LeafValue::Rollout,
                        "static" => LeafValue::Static,
                        "trunc" | "truncated" => LeafValue::Truncated { plies: 8 },
                        "blend" => LeafValue::Blend { weight: 0.5 },
                        "learned" => LeafValue::Learned,
                        "learned_blend" | "lblend" => LeafValue::LearnedBlend { weight: 0.5 },
                        other => {
                            return Err(format!(
                                "mcts-eval: unknown leaf \"{other}\" (expected \"rollout\", \
                                 \"static\", \"truncated[:<plies>]\", \"blend[:<weight>]\", \
                                 \"learned\", or \"learned_blend[:<weight>]\")"
                            ))
                        }
                    },
                    Some((other, _)) => {
                        return Err(format!(
                            "mcts-eval: leaf \"{other}\" takes no parameter (only \
                             \"truncated:<plies>\", \"blend:<weight>\" and \
                             \"learned_blend:<weight>\" do)"
                        ))
                    }
                };
            }
            // Which accumulation order the learned leaves' forward pass uses.
            // Read only by a learned leaf, so it cannot move the default
            // configuration; it is here so the four-way accumulator unroll can
            // be A/B tested against the arithmetic the crate docs' Elo numbers
            // were taken with. See `duels_value::Summation`.
            "value_sum" | "value_summation" => {
                cfg.value_summation = match v {
                    "serial" => duels_value::Summation::Serial,
                    "unrolled4" | "unrolled" => duels_value::Summation::Unrolled4,
                    "axpy" | "transposed_axpy" => duels_value::Summation::TransposedAxpy,
                    other => {
                        return Err(format!(
                            "mcts-eval: unknown value summation \"{other}\" (expected \
                             \"serial\", \"unrolled4\" or \"axpy\")"
                        ))
                    }
                };
            }
            "evalgen" | "eval_generation" => {
                return Err("mcts-eval: no key by that name. The agent still tracks \
                     duels_eval::Config::default() live by default, on purpose, and records the \
                     whole configuration it used in its spec string — see its crate docs before \
                     changing that. For an A/B test against a frozen generation specifically, \
                     use \"eval=vN\" (e.g. \"eval=v6\"), not this key."
                    .to_string())
            }
            // Anything left is tried as a `duels-eval` scalar, by the same key
            // names `phased` already accepts, applied on top of whatever
            // generation `eval=` selected (or on top of today's live default
            // when no `eval=` key was given). This is what lets a candidate
            // *weight vector* -- as opposed to a frozen generation -- be A/B
            // tested inside the search that actually consumes it, instead of
            // only through `phased`'s 1-ply argmax.
            //
            // Note the asymmetry with omitting the key entirely: touching any
            // eval scalar necessarily pins `eval_override`, so the agent stops
            // tracking `duels-eval` live. That is correct for a measurement
            // -- a fitted vector *is* a frozen evaluation -- but it means
            // these keys are for experiments, never for a shipped default.
            other => {
                let eval = &mut cfg
                    .eval_override
                    .get_or_insert_with(duels_eval::Config::default)
                    .eval;
                if !apply_eval_config_key(eval, other, v)? {
                    return Err(format!("mcts-eval: unknown key \"{other}\""));
                }
            }
        }
    }
    Ok(cfg)
}

/// Parse a `mcts-value:...` parameter list into a [`MctsValueConfig`].
///
/// The search keys are `mcts-eval`'s, spelled identically, because it is the
/// same search — `mcts-value`'s whole content is a different
/// [`duels_agent_mcts_value::Config::leaf`] and the exploration constant that
/// was swept for it. What is new is the two `base` values, which are this
/// agent's ablation chain and the arms its measured Elo is quoted against:
///
/// * `base=eval` selects [`duels_agent_mcts_value::Config::eval_base`], which
///   is `mcts-eval` at its default — proven so, node for node, against a
///   verbatim frozen copy of that agent's search inside the agent crate.
/// * `base=rollout` selects
///   [`duels_agent_mcts_value::Config::rollout_base`], which is `mcts-uct`,
///   proven the same way.
///
/// So the whole chain is measurable in one binary and one process:
/// `mcts-value` against `mcts-value:base=eval` against
/// `mcts-value:base=rollout`.
///
/// Two keys have no effect on this agent's default configuration and are
/// accepted for its `base=eval`/`base=rollout` arms, where they do: `eval=vN`
/// and the `duels-eval` scalar fallthrough. That is the reverse of
/// `mcts-eval`, and it is not a quirk of the parser — the default leaf here
/// reads no hand-crafted evaluation at all, which
/// `duels_agent_mcts_value`'s `tree::tests::the_evaluation_cannot_reach_the_default_leaf`
/// asserts directly. `value_sum=serial`, conversely, *is* on this agent's
/// default path: it selects `duels_value::Summation::Serial`, the accumulation
/// order that predates the four-way unroll.
/// Read a `weights=file:<path>` candidate's bytes once per unique path and
/// leak them to `'static`, so [`duels_agent_mcts_value::Config::value_weights_override`]
/// (which requires `&'static [u8]`, the same convention every named
/// `WEIGHTS_*` constant already uses) can point at a not-yet-promoted weights
/// file without adding a new constant, `agent_spec` match arm and
/// `golden.rs` entry -- and a rebuild -- for every candidate.
///
/// A process that gates one candidate plays many thousands of games, and
/// this function is called once per game (agent specs are parsed fresh per
/// game, not cached by the caller) -- so leaking unconditionally would leak
/// megabytes per game. The cache below leaks each distinct path's bytes at
/// most once per process instead, which is the same bounded, one-time cost
/// every compiled-in `WEIGHTS_*` constant already pays (its bytes live for
/// the process's whole lifetime too, just baked in at compile time rather
/// than read at first use).
fn load_weights_file(path: &str) -> Result<&'static [u8], String> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, &'static [u8]>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&bytes) = guard.get(path) {
        return Ok(bytes);
    }
    let data = std::fs::read(path)
        .map_err(|e| format!("mcts-value: failed to read weights file \"{path}\": {e}"))?;
    let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
    guard.insert(path.to_string(), leaked);
    Ok(leaked)
}

pub fn parse_mcts_value_config(params: &str) -> Result<MctsValueConfig, String> {
    let mut cfg = MctsValueConfig::default();
    for (k, v) in parse_params(params)? {
        match k {
            "base" => match v {
                // Deliberately first-listed and last-applied like every other
                // agent's `base`: keys after it override.
                "eval" | "mcts-eval" | "blend" => cfg = MctsValueConfig::eval_base(),
                "rollout" | "mcts-uct" => cfg = MctsValueConfig::rollout_base(),
                "default" | "learned" => {}
                other => {
                    return Err(format!(
                        "mcts-value: unknown base \"{other}\" (expected \"default\", \
                         \"eval\" for the mcts-eval control, or \"rollout\" for the \
                         mcts-uct one)"
                    ))
                }
            },
            // Only reachable on the `base=eval` arm; see this function's docs.
            "eval" => {
                cfg.eval_override = Some(match v {
                    "default" | "live" => {
                        return Err(
                            "mcts-value: eval=default/live is the same as omitting the key -- \
                             pass no eval key at all to read duels-eval live"
                                .to_string(),
                        )
                    }
                    "v1" => duels_eval::Config::v1(),
                    "v2" => duels_eval::Config::v2(),
                    "v3" => duels_eval::Config::v3(),
                    "v4" => duels_eval::Config::v4(),
                    "v5" => duels_eval::Config::v5(),
                    "v6" => duels_eval::Config::v6(),
                    "v7" => duels_eval::Config::v7(),
                    "v8" => duels_eval::Config::v8(),
                    "v9" => duels_eval::Config::v9(),
                    other => {
                        return Err(format!(
                            "mcts-value: unknown eval generation \"{other}\" (expected v1-v9)"
                        ))
                    }
                });
            }
            "sci_progress" | "sciprog" => {
                let eval = cfg
                    .eval_override
                    .get_or_insert_with(duels_eval::Config::default);
                eval.blend.science_progress = match v {
                    "root" | "off" => duels_eval::ScienceProgress::Root,
                    "leaf" | "on" => duels_eval::ScienceProgress::Leaf,
                    other => {
                        return Err(format!(
                            "mcts-value: unknown sci_progress \"{other}\" (expected \"root\" \
                             or \"leaf\")"
                        ))
                    }
                };
            }
            "exploration" | "c" => cfg.exploration = parse_field(k, v)?,
            "chance_widen_c" => cfg.chance_widen_c = parse_field(k, v)?,
            "chance_widen_alpha" => cfg.chance_widen_alpha = parse_field(k, v)?,
            "max_rollout_plies" => cfg.max_rollout_plies = parse_field(k, v)?,
            "time_check_interval" => cfg.time_check_interval = parse_field(k, v)?,
            "root_determinizations" | "dets" => cfg.root_determinizations = parse_field(k, v)?,
            // Like `race` below, a playout-policy knob: inert on this agent's
            // default leaf, which runs no playout. Accepted rather than
            // rejected because it is live on both `base=` arms, and because
            // `LeafValue::LearnedBlend` brings a playout back.
            "rollout" => {
                cfg.rollout = match v {
                    "uniform" => ValueRolloutWeights::UNIFORM,
                    "biased" => ValueRolloutWeights::BIASED,
                    "smart" => ValueRolloutWeights::SMART,
                    other => return Err(format!("mcts-value: unknown rollout \"{other}\"")),
                };
            }
            "race" => {
                cfg.race = match v {
                    "neutral" | "off" => ValueRaceWeights::NEUTRAL,
                    "tier1" | "tier1_only" => ValueRaceWeights::TIER1_ONLY,
                    "mild" => ValueRaceWeights::mild(),
                    "medium" => ValueRaceWeights::MEDIUM,
                    "strong" => ValueRaceWeights::strong(),
                    other => {
                        return Err(format!(
                            "mcts-value: unknown race \"{other}\" (expected \"neutral\", \
                             \"mild\", \"medium\", \"strong\", or \"tier1_only\")"
                        ))
                    }
                };
            }
            "prior" => {
                cfg.prior = match v.split_once(':') {
                    Some(("progressive_bias" | "bias", w)) => ValuePriorMode::ProgressiveBias {
                        weight: parse_field("prior", w)?,
                    },
                    None => match v {
                        "none" | "off" => ValuePriorMode::None,
                        "expansion_order" | "order" => ValuePriorMode::ExpansionOrder,
                        "progressive_bias" | "bias" => {
                            ValuePriorMode::ProgressiveBias { weight: 1.0 }
                        }
                        other => {
                            return Err(format!(
                                "mcts-value: unknown prior \"{other}\" (expected \"none\", \
                                 \"expansion_order\", or \"progressive_bias[:<weight>]\")"
                            ))
                        }
                    },
                    Some((other, _)) => {
                        return Err(format!(
                            "mcts-value: prior \"{other}\" takes no weight (only \
                             \"progressive_bias:<weight>\" does)"
                        ))
                    }
                };
            }
            "leaf" => {
                // Note that `leaf` alone does **not** change `exploration`:
                // what a leaf backs up decides what `c` means, and the two
                // move together only in `Config::default`/`eval_base`. A
                // caller changing one has to decide about the other — see
                // `duels_agent_mcts_value::Config::default`'s doc comment,
                // where the sweep that produced `c = 0.15` is recorded.
                cfg.leaf = match v.split_once(':') {
                    Some(("trunc" | "truncated", p)) => ValueLeafValue::Truncated {
                        plies: parse_field("leaf", p)?,
                    },
                    Some(("blend", w)) => {
                        // Range-checked, unlike the other float keys here: a
                        // blend weight outside `[0, 1]` makes the leaf value
                        // stop being a probability, which silently violates
                        // the value convention every node in that tree
                        // accumulates.
                        let weight: f64 = parse_field("leaf", w)?;
                        if !(0.0..=1.0).contains(&weight) {
                            return Err(format!(
                                "mcts-value: leaf blend weight must be in [0, 1], got \"{w}\" \
                                 (outside it the leaf value is not a probability)"
                            ));
                        }
                        ValueLeafValue::Blend { weight }
                    }
                    Some(("learned_blend" | "lblend", w)) => {
                        let weight: f64 = parse_field("leaf", w)?;
                        if !(0.0..=1.0).contains(&weight) {
                            return Err(format!(
                                "mcts-value: leaf blend weight must be in [0, 1], got \"{w}\" \
                                 (outside it the leaf value is not a probability)"
                            ));
                        }
                        ValueLeafValue::LearnedBlend { weight }
                    }
                    None => match v {
                        "rollout" | "off" => ValueLeafValue::Rollout,
                        "static" => ValueLeafValue::Static,
                        "trunc" | "truncated" => ValueLeafValue::Truncated { plies: 8 },
                        "blend" => ValueLeafValue::Blend { weight: 0.5 },
                        "learned" => ValueLeafValue::Learned,
                        "learned_blend" | "lblend" => ValueLeafValue::LearnedBlend { weight: 0.5 },
                        "learned_symmetric" | "lsym" => ValueLeafValue::LearnedSymmetric,
                        other => {
                            return Err(format!(
                                "mcts-value: unknown leaf \"{other}\" (expected \"learned\", \
                                 \"learned_blend[:<weight>]\", \"learned_symmetric\", \
                                 \"rollout\", \"static\", \"truncated[:<plies>]\", or \
                                 \"blend[:<weight>]\")"
                            ))
                        }
                    },
                    Some((other, _)) => {
                        return Err(format!(
                            "mcts-value: leaf \"{other}\" takes no parameter (only \
                             \"truncated:<plies>\", \"blend:<weight>\" and \
                             \"learned_blend:<weight>\" do)"
                        ))
                    }
                };
            }
            // Pins the learned leaf to a frozen historical `duels-value`
            // weights generation instead of the live embedded default — see
            // `duels_agent_mcts_value::Config::value_weights_override`. The
            // whole point is measuring the current generation against a
            // prior one, in one process (`duels-arena match --agent-a
            // mcts-value --agent-b mcts-value:weights=v3`); the live default
            // is `v4` (see `duels_value`'s crate docs, "Generation 3", on
            // why `v4`/`gen3-l05` was promoted), and `v1`/`v2`/`v3` all stay
            // reachable as frozen generations -- `v2` and `v3` additionally
            // as frozen reference-panel members (`docs/roadmap.md` Tier 1-G).
            "weights" => {
                cfg.value_weights_override = if let Some(path) = v.strip_prefix("file:") {
                    // An ad hoc, not-yet-promoted weights file, read from disk
                    // at runtime rather than compiled in -- for the autonomous
                    // self-play loop's own gating (`docs/roadmap.md`'s
                    // "Autonomous self-play loop design"), which needs to
                    // measure a fresh candidate every generation without
                    // adding a new `WEIGHTS_*` constant, `agent_spec` match
                    // arm and `golden.rs` entry -- and a rebuild -- each time.
                    // Every other `weights=` value stays a compiled-in
                    // `&'static [u8]`; promoting a generation to the live
                    // default, or pinning it as a permanently reachable named
                    // generation, still goes through that same reviewed path.
                    Some(load_weights_file(path)?)
                } else {
                    match v {
                        "default" | "live" | "current" => None,
                        "v1" => Some(duels_agent_mcts_value::WEIGHTS_V1),
                        "v2" => Some(duels_agent_mcts_value::WEIGHTS_V2),
                        "v3" => Some(duels_agent_mcts_value::WEIGHTS_V3),
                        // Unpromoted Tier 1-D/E experiment candidates -- see
                        // `duels_agent_mcts_value::WEIGHTS_ARM_A`'s docs for what
                        // each arm actually is. None of these is the default.
                        "arm-a" => Some(duels_agent_mcts_value::WEIGHTS_ARM_A),
                        "arm-b" => Some(duels_agent_mcts_value::WEIGHTS_ARM_B),
                        "arm-c" => Some(duels_agent_mcts_value::WEIGHTS_ARM_C),
                        // Corrected retest of arm-c's idea (the blended-loss
                        // gradient bug fix) -- see
                        // `duels_agent_mcts_value::WEIGHTS_ARM_C_PRIME`'s docs.
                        "arm-c2" => Some(duels_agent_mcts_value::WEIGHTS_ARM_C_PRIME),
                        "arm-d2" => Some(duels_agent_mcts_value::WEIGHTS_ARM_D_PRIME),
                        // Generation 3 (held, not promoted -- v3 remains
                        // DEFAULT_WEIGHTS) and the recipe-calibration-day retrain
                        // from its identical corpus -- see
                        // `duels_agent_mcts_value::WEIGHTS_GEN3_L05`'s docs.
                        "gen3-l05" => Some(duels_agent_mcts_value::WEIGHTS_GEN3_L05),
                        "gen3-l10" => Some(duels_agent_mcts_value::WEIGHTS_GEN3_L10),
                        "gen3-l05-fixedrecipe" => {
                            Some(duels_agent_mcts_value::WEIGHTS_GEN3_L05_FIXEDRECIPE)
                        }
                        // Recipe-calibration-day node-budget ablation -- see
                        // `duels_agent_mcts_value::WEIGHTS_NB2000`'s docs.
                        "nb2000" => Some(duels_agent_mcts_value::WEIGHTS_NB2000),
                        "nb8000" => Some(duels_agent_mcts_value::WEIGHTS_NB8000),
                        other => {
                            return Err(format!(
                                "mcts-value: unknown weights generation \"{other}\" (expected \
                             \"default\", \"v1\", \"v2\", \"v3\", \"arm-a\"/\"arm-b\"/\"arm-c\" \
                             for the Tier 1-D/E experiment candidates, \"arm-c2\"/\"arm-d2\" for \
                             the corrected-gradient retest, \"gen3-l05\"/\"gen3-l10\"/\
                             \"gen3-l05-fixedrecipe\" for Generation 3 and its recipe-fix retest \
                             (v3 remains DEFAULT_WEIGHTS -- gen3-l05 was held, not promoted), \
                             \"nb2000\"/\"nb8000\" for the node-budget ablation, or \
                             \"file:<path>\" to load an ad hoc, not-yet-promoted weights file \
                             from disk at runtime)"
                            ))
                        }
                    }
                };
            }
            "value_sum" | "value_summation" => {
                cfg.value_summation = match v {
                    "serial" => duels_value::Summation::Serial,
                    "unrolled4" | "unrolled" => duels_value::Summation::Unrolled4,
                    "axpy" | "transposed_axpy" => duels_value::Summation::TransposedAxpy,
                    other => {
                        return Err(format!(
                            "mcts-value: unknown value summation \"{other}\" (expected \
                             \"serial\", \"unrolled4\" or \"axpy\")"
                        ))
                    }
                };
            }
            // What the search is rewarded for winning: any victory (the
            // default) or one specific `VictoryKind`, for the "how purely
            // does a specialist pursue one strategy" research direction --
            // see `duels_agent_mcts_value::Objective`'s docs. This changes
            // both the terminal reward the tree backs up and, for the
            // default learned leaf, which component of `duels_value`'s
            // four-way head a leaf reads; no weights are retrained for it.
            "objective" | "obj" => {
                cfg.objective = match v {
                    "win" | "win_probability" | "default" => ValueObjective::WinProbability,
                    "science" | "sci" => {
                        ValueObjective::TargetKind(VictoryKind::ScientificSupremacy)
                    }
                    "military" | "mil" => {
                        ValueObjective::TargetKind(VictoryKind::MilitarySupremacy)
                    }
                    "civilian" | "civ" => ValueObjective::TargetKind(VictoryKind::CivilianVictory),
                    other => {
                        return Err(format!(
                            "mcts-value: unknown objective \"{other}\" (expected \"win\", \
                             \"science\", \"military\", or \"civilian\")"
                        ))
                    }
                };
            }
            // As on `mcts-eval`: anything left is tried as a `duels-eval`
            // scalar, which necessarily pins `eval_override` and so is only
            // meaningful alongside `base=eval`.
            other => {
                let eval = &mut cfg
                    .eval_override
                    .get_or_insert_with(duels_eval::Config::default)
                    .eval;
                if !apply_eval_config_key(eval, other, v)? {
                    return Err(format!("mcts-value: unknown key \"{other}\""));
                }
            }
        }
    }
    Ok(cfg)
}

/// Applies one `key=value` pair to a [`duels_eval::Config`], returning whether
/// the key was recognised.
///
/// The key names are exactly [`parse_phased_config`]'s, which is the point:
/// a weight vector written for one agent has to mean the same thing in the
/// other, or an A/B across the two measures two different candidates. The
/// duplication against `parse_phased_config`'s own arms is an accepted cost
/// (it is the same one the now-deleted `eval_weights_parser!` macro carried
/// across `greedy` and `greedy-ev`), and
/// `mcts_eval_shares_phaseds_eval_key_names` holds the two in agreement.
fn apply_eval_config_key(
    eval: &mut duels_eval::EvalWeights,
    k: &str,
    v: &str,
) -> Result<bool, String> {
    match k {
        "menu_lambda" | "lambda" => eval.menu.lambda = parse_field(k, v)?,
        "menu_tau" | "tau" => eval.menu.tau = parse_field(k, v)?,
        "chain_equity" | "chaineq" => eval.chain_equity = parse_field(k, v)?,
        "resource_bill" | "bill" => eval.resource_bill = parse_field(k, v)?,
        "military_band" | "band" => eval.military_band = parse_field(k, v)?,
        "military_loot" | "loot" => eval.military_loot = parse_field(k, v)?,
        "military_sigma_scale" | "kappa" => eval.military_sigma_scale = parse_field(k, v)?,
        "military_sigma_min" => eval.military_sigma_min = parse_field(k, v)?,
        "military_logistic_scale" => eval.military_logistic_scale = parse_field(k, v)?,
        "coin_smooth_beta" | "beta" => eval.coin_smooth_beta = parse_field(k, v)?,
        "coin_smooth_ref" | "cref" => eval.coin_smooth_ref = parse_field(k, v)?,
        "military_position" => eval.military_position = parse_field(k, v)?,
        "military_endgame_urgency" | "urgency" => {
            eval.military_endgame_urgency = parse_field(k, v)?
        }
        "coins_div3" => eval.coins_div3 = parse_field(k, v)?,
        "vp_projection" => eval.vp_projection = parse_field(k, v)?,
        "development" => eval.development = parse_field(k, v)?,
        "science_ladder" => eval.science_ladder = parse_field(k, v)?,
        "science_pair_threat" | "pairthreat" => {
            eval.science.pair_threat_weight = parse_field(k, v)?
        }
        "dead_race_scale" | "dead" => eval.science.dead_race_scale = parse_field(k, v)?,
        "ladder1" => eval.science.ladder[1] = parse_field(k, v)?,
        "ladder2" => eval.science.ladder[2] = parse_field(k, v)?,
        "ladder3" => eval.science.ladder[3] = parse_field(k, v)?,
        "ladder4" => eval.science.ladder[4] = parse_field(k, v)?,
        "ladder5" => eval.science.ladder[5] = parse_field(k, v)?,
        "temp1" => eval.win_probability_temperature[0] = parse_field(k, v)?,
        "temp2" => eval.win_probability_temperature[1] = parse_field(k, v)?,
        "temp3" => eval.win_probability_temperature[2] = parse_field(k, v)?,
        "token_equity" | "tokeneq" => eval.token_equity = parse_field(k, v)?,
        "to_move" | "tomove" => eval.to_move = parse_field(k, v)?,
        "value_scale" | "scale" => eval.value_scale = parse_field(k, v)?,
        "deny" => eval.deny = parse_field(k, v)?,
        "deny_chain_gift" => eval.deny_chain_gift = parse_field(k, v)?,
        "wonder_potential" => eval.wonder_potential = parse_field(k, v)?,
        "guild_projection" | "guildproj" => eval.guild_projection = parse_field(k, v)?,
        "yellow_equity" | "yellow" => eval.yellow_equity = parse_field(k, v)?,
        "yellow_discard_rate" | "discardrate" => eval.yellow_discard_rate = parse_field(k, v)?,
        "wonder_turns_per_wonder" | "wturns" => eval.wonder_turns_per_wonder = parse_field(k, v)?,
        "wonder_p_build_ref" | "pref" => eval.wonder_p_build_ref = parse_field(k, v)?,
        "wonder_extra_turn_vp" | "wextra" => eval.wonder_extra_turn_vp = parse_field(k, v)?,
        "wonder_extra_turn_premium" | "wprem" => {
            eval.wonder_extra_turn_premium = parse_field(k, v)?
        }
        "imminent" => eval.imminent = parse_field(k, v)?,
        "production_lock_in" | "lockin" => eval.production_lock_in = parse_field(k, v)?,
        "start1" => eval.next_age_start[0] = parse_field(k, v)?,
        "start2" => eval.next_age_start[1] = parse_field(k, v)?,
        "start3" => eval.next_age_start[2] = parse_field(k, v)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Parse a `phased:...` parameter list into a [`PhasedConfig`].
///
/// `base=v1` restores the configuration the crate first shipped with — the
/// legacy military / coin / economy models, no chain equity, no opponent menu
/// and the original `next_age_start` magnitudes — which is what makes
/// "the new agent against the old one" a single-binary measurement rather than
/// a build-two-checkouts exercise. Keys after `base` override it, exactly as
/// `alphabeta`'s `base` and `weights` keys do.
pub fn parse_phased_config(params: &str) -> Result<PhasedConfig, String> {
    let mut cfg = PhasedConfig::default();
    for (k, v) in parse_params(params)? {
        match k {
            "base" => match v {
                "v1" => cfg = PhasedConfig::v1(),
                "v2" => cfg = PhasedConfig::v2(),
                "v3" => cfg = PhasedConfig::v3(),
                "v4" => cfg = PhasedConfig::v4(),
                "v5" => cfg = PhasedConfig::v5(),
                "v6" => cfg = PhasedConfig::v6(),
                "v7" => cfg = PhasedConfig::v7(),
                "v8" => cfg = PhasedConfig::v8(),
                "v9" => cfg = PhasedConfig::v9(),
                "default" => cfg = PhasedConfig::default(),
                other => return Err(format!("phased: unknown base \"{other}\"")),
            },
            "guild" | "guild_pricing" => {
                cfg.guild_pricing = match v {
                    "unpriced" | "off" => GuildPricing::Unpriced,
                    "projected" | "on" => GuildPricing::Projected,
                    other => return Err(format!("phased: unknown guild \"{other}\"")),
                }
            }
            "guild_projection" | "guildproj" => cfg.eval.guild_projection = parse_field(k, v)?,
            "menu_floor" | "menufloor" => {
                cfg.menu_floor = match v {
                    "none" | "off" => MenuFloor::None,
                    "discard" => MenuFloor::Discard,
                    "discardwonder" | "discard_and_wonder" => MenuFloor::DiscardAndWonder,
                    other => return Err(format!("phased: unknown menu_floor \"{other}\"")),
                }
            }
            "menu_afford_soft" | "afford" => cfg.menu_afford_soft = parse_field(k, v)?,
            "supply" | "supply_model" => {
                cfg.supply_model = match v {
                    "raw" => SupplyModel::Raw,
                    "dealt" => SupplyModel::Dealt,
                    other => return Err(format!("phased: unknown supply_model \"{other}\"")),
                }
            }
            "yellow_equity" | "yellow" => cfg.eval.yellow_equity = parse_field(k, v)?,
            "yellow_discard_rate" | "discardrate" => {
                cfg.eval.yellow_discard_rate = parse_field(k, v)?
            }
            "pending" | "pending_model" => {
                cfg.pending_model = match v {
                    "unresolved" | "off" => PendingModel::Unresolved,
                    "completed" | "on" => PendingModel::Completed,
                    other => return Err(format!("phased: unknown pending \"{other}\"")),
                }
            }
            "wonder" | "wonder_model" => {
                cfg.wonder_model = match v {
                    "flat" => WonderModel::Flat,
                    "budget" => WonderModel::Budget,
                    "rationed" => WonderModel::Rationed,
                    other => return Err(format!("phased: unknown wonder_model \"{other}\"")),
                }
            }
            "reach_model" | "reach" => {
                cfg.eval.science.reach_model = match v {
                    "optimistic" | "off" => ReachModel::Optimistic,
                    "structure" | "on" => ReachModel::Structure,
                    other => return Err(format!("phased: unknown reach_model \"{other}\"")),
                }
            }
            "destroy_replace" | "destroyrepl" => {
                cfg.destroy_replace_discount = match v {
                    "on" | "true" => true,
                    "off" | "false" => false,
                    other => return Err(format!("phased: unknown destroy_replace \"{other}\"")),
                }
            }
            "wonder_turns_per_wonder" | "wturns" => {
                cfg.eval.wonder_turns_per_wonder = parse_field(k, v)?
            }
            "wonder_p_build_ref" | "pref" => cfg.eval.wonder_p_build_ref = parse_field(k, v)?,
            "wonder_extra_turn_vp" | "wextra" => cfg.eval.wonder_extra_turn_vp = parse_field(k, v)?,
            "wonder_extra_turn_premium" | "wprem" => {
                cfg.eval.wonder_extra_turn_premium = parse_field(k, v)?
            }
            "rails" => {
                cfg.rails = match v {
                    "on" | "true" => RailModel::On,
                    "off" | "false" => RailModel::Off,
                    other => return Err(format!("phased: unknown rails \"{other}\"")),
                }
            }
            "imminent" => cfg.eval.imminent = parse_field(k, v)?,
            "shield_price" | "shieldprice" => {
                cfg.menu_shield_pricing = match v {
                    "onesided" | "one_sided" => MenuShieldPricing::OneSided,
                    "diff" | "differenced" => MenuShieldPricing::Differenced,
                    other => return Err(format!("phased: unknown shield_price \"{other}\"")),
                }
            }
            "horizon" => {
                cfg.military_horizon = match v {
                    "supply" | "none" | "off" => None,
                    other => Some(other.parse::<f64>().map_err(|e| {
                        format!("phased: horizon \"{other}\" is not a number: {e}")
                    })?),
                }
            }
            "production_lock_in" | "lockin" => cfg.eval.production_lock_in = parse_field(k, v)?,
            "military_model" | "mil" => {
                cfg.military_model = match v {
                    "legacy" => MilitaryModel::Legacy,
                    "band" => MilitaryModel::Band,
                    other => return Err(format!("phased: unknown military_model \"{other}\"")),
                }
            }
            "coin_model" | "coin" => {
                cfg.coin_model = match v {
                    "legacy" => CoinModel::Legacy,
                    "smooth" => CoinModel::Smooth,
                    other => return Err(format!("phased: unknown coin_model \"{other}\"")),
                }
            }
            "economy_model" | "econ" => {
                cfg.economy_model = match v {
                    "legacy" => EconomyModel::Legacy,
                    "bill" => EconomyModel::Bill,
                    other => return Err(format!("phased: unknown economy_model \"{other}\"")),
                }
            }
            "blend" => {
                cfg.blend = match v {
                    "on" | "true" => PhasedBlend::default(),
                    "off" | "false" => PhasedBlend::off(),
                    other => return Err(format!("phased: unknown blend \"{other}\"")),
                }
            }
            // Round ten's option. Listed after `blend` so `blend=off,
            // sciprog=leaf` means what it reads like, the same ordering
            // convention `base` follows.
            "sci_progress" | "sciprog" => {
                cfg.blend.science_progress = match v {
                    "root" | "off" => ScienceProgress::Root,
                    "leaf" | "on" => ScienceProgress::Leaf,
                    other => return Err(format!("phased: unknown sci_progress \"{other}\"")),
                }
            }
            "menu_lambda" | "lambda" => cfg.eval.menu.lambda = parse_field(k, v)?,
            "menu_tau" | "tau" => cfg.eval.menu.tau = parse_field(k, v)?,
            "chain_equity" | "chaineq" => cfg.eval.chain_equity = parse_field(k, v)?,
            "resource_bill" | "bill" => cfg.eval.resource_bill = parse_field(k, v)?,
            "military_band" | "band" => cfg.eval.military_band = parse_field(k, v)?,
            "military_loot" | "loot" => cfg.eval.military_loot = parse_field(k, v)?,
            "military_sigma_scale" | "kappa" => cfg.eval.military_sigma_scale = parse_field(k, v)?,
            "military_sigma_min" => cfg.eval.military_sigma_min = parse_field(k, v)?,
            "military_logistic_scale" => cfg.eval.military_logistic_scale = parse_field(k, v)?,
            "coin_smooth_beta" | "beta" => cfg.eval.coin_smooth_beta = parse_field(k, v)?,
            "coin_smooth_ref" | "cref" => cfg.eval.coin_smooth_ref = parse_field(k, v)?,
            "military_position" => cfg.eval.military_position = parse_field(k, v)?,
            "military_endgame_urgency" | "urgency" => {
                cfg.eval.military_endgame_urgency = parse_field(k, v)?
            }
            "coins_div3" => cfg.eval.coins_div3 = parse_field(k, v)?,
            "vp_projection" => cfg.eval.vp_projection = parse_field(k, v)?,
            "development" => cfg.eval.development = parse_field(k, v)?,
            "science_ladder" => cfg.eval.science_ladder = parse_field(k, v)?,
            "science_pair_threat" | "pairthreat" => {
                cfg.eval.science.pair_threat_weight = parse_field(k, v)?
            }
            "dead_race_scale" | "dead" => cfg.eval.science.dead_race_scale = parse_field(k, v)?,
            // The individual ladder rungs, and the leaf temperature, which
            // round eight moved. `duels-eval`'s own `examples/head_to_head.rs`
            // uses the same `ladderN` names.
            "ladder1" => cfg.eval.science.ladder[1] = parse_field(k, v)?,
            "ladder2" => cfg.eval.science.ladder[2] = parse_field(k, v)?,
            "ladder3" => cfg.eval.science.ladder[3] = parse_field(k, v)?,
            "ladder4" => cfg.eval.science.ladder[4] = parse_field(k, v)?,
            "ladder5" => cfg.eval.science.ladder[5] = parse_field(k, v)?,
            "temp1" => cfg.eval.win_probability_temperature[0] = parse_field(k, v)?,
            "temp2" => cfg.eval.win_probability_temperature[1] = parse_field(k, v)?,
            "temp3" => cfg.eval.win_probability_temperature[2] = parse_field(k, v)?,
            "token_equity" | "tokeneq" => cfg.eval.token_equity = parse_field(k, v)?,
            "to_move" | "tomove" => cfg.eval.to_move = parse_field(k, v)?,
            "value_scale" | "scale" => cfg.eval.value_scale = parse_field(k, v)?,
            "count" | "count_pricing" => {
                cfg.count_pricing = match v {
                    "unpriced" | "off" => CountPricing::Unpriced,
                    "counted" | "on" => CountPricing::Counted,
                    other => return Err(format!("phased: unknown count \"{other}\"")),
                }
            }
            "deny" => cfg.eval.deny = parse_field(k, v)?,
            "deny_chain_gift" => cfg.eval.deny_chain_gift = parse_field(k, v)?,
            "wonder_potential" => cfg.eval.wonder_potential = parse_field(k, v)?,
            // The `next_age_start` array, one age at a time, because a comma
            // would collide with the parameter separator.
            "start1" => cfg.eval.next_age_start[0] = parse_field(k, v)?,
            "start2" => cfg.eval.next_age_start[1] = parse_field(k, v)?,
            "start3" => cfg.eval.next_age_start[2] = parse_field(k, v)?,
            other => return Err(format!("phased: unknown key \"{other}\"")),
        }
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_name_still_works_exactly_as_before() {
        let agent = make_agent_from_spec("phased", 1).unwrap();
        assert_eq!(agent.spec().name, "phased");
        let agent = make_agent_from_spec("mcts-uct", 1).unwrap();
        assert_eq!(agent.spec().name, "mcts-uct");
    }

    /// A retired agent's name is rejected whether or not it carries
    /// parameters — the `greedy`/`greedy-ev` weight parsers went with the
    /// crates, so there is no arm left to accept one. See
    /// `docs/milestones.md`.
    #[test]
    fn retired_agent_names_are_rejected_bare_and_parameterised() {
        for retired in ["random", "greedy", "greedy-ev", "strategist"] {
            assert!(make_agent_from_spec(retired, 1).is_err(), "{retired}");
            assert!(
                make_agent_from_spec(&format!("{retired}:vp_projection=2.5"), 1).is_err(),
                "{retired} with parameters"
            );
        }
    }

    #[test]
    fn bare_name_with_no_colon_matches_agent_registry_directly() {
        for name in KNOWN_AGENTS {
            let from_spec = make_agent_from_spec(name, 7).unwrap();
            let from_registry = make_agent(name, 7).unwrap();
            assert_eq!(from_spec.spec(), from_registry.spec());
        }
    }

    #[test]
    fn alphabeta_spec_sets_the_named_fields() {
        let agent = make_agent_from_spec("alphabeta:max_depth=10,rollouts=2", 3).unwrap();
        assert_eq!(agent.spec().name, "alphabeta");
        assert!(agent.spec().params.contains("max_depth=10"));
    }

    #[test]
    fn alphabeta_config_parser_reads_every_supported_key() {
        let cfg = parse_alphabeta_config(
            "max_depth=5,cap=2,tt_bits=10,tt=false,star1=false,rollouts=3,blend=0.5,\
             cap-rollouts=16,crn=true,policy=uniform,greedy=0.5,order=none,weights=v1,\
             metric=outcome:2.0,card=1.0,coin=2.0,breadth=3.0,shield=4.0,threat=5.0",
        )
        .unwrap();
        assert_eq!(cfg.max_depth, 5);
        assert_eq!(cfg.chance_cap, 2);
        assert_eq!(cfg.tt_bits, 10);
        assert!(!cfg.use_tt);
        assert!(!cfg.star1);
        assert_eq!(cfg.rollouts, 3);
        assert_eq!(cfg.rollout_blend, 0.5);
        assert_eq!(cfg.rollout_cap, 16);
        assert!(cfg.rollout_common_seed);
        // `policy=uniform` sets the base policy, then `greedy=0.5` overrides
        // just that one field on top of it (order in the spec string
        // matters, matching `ab_lab`'s original behavior).
        assert_eq!(
            cfg.rollout_policy.build,
            playout::PolicyWeights::UNIFORM.build
        );
        assert_eq!(
            cfg.rollout_policy.wonder,
            playout::PolicyWeights::UNIFORM.wonder
        );
        assert_eq!(cfg.rollout_policy.greedy, 0.5);
        assert!(!cfg.order_moves);
        assert!(!cfg.order_lookahead);
        assert!(!cfg.order_priors);
        assert_eq!(cfg.rollout_metric, playout::Metric::Outcome { scale: 2.0 });
        // `weights=v1` sets the base weights, then the individual `card`,
        // `coin`, `breadth`, `shield`, `threat` keys override just those
        // fields on top of it (order in the spec string matters). A field
        // the spec string never named stays at `v1`'s own value.
        assert_eq!(cfg.weights.card_in_city, 1.0);
        assert_eq!(cfg.weights.coin, 2.0);
        assert_eq!(cfg.weights.resource_breadth, 3.0);
        assert_eq!(cfg.weights.shield, 4.0);
        assert_eq!(cfg.weights.capital_threat, 5.0);
        assert_eq!(cfg.weights.science_single, eval::Weights::V1.science_single);
    }

    #[test]
    fn alphabeta_order_priors_sets_order_moves_and_priors_together() {
        let cfg = parse_alphabeta_config("order=priors").unwrap();
        assert!(cfg.order_moves);
        assert!(cfg.order_priors);
        assert!(!cfg.order_lookahead);
    }

    #[test]
    fn alphabeta_base_v1_sets_the_pre_rework_defaults() {
        let cfg = parse_alphabeta_config("base=v1").unwrap();
        assert_eq!(cfg, AlphaBetaConfig::v1());
    }

    #[test]
    fn empty_parameter_list_is_equivalent_to_the_default_config() {
        let cfg = parse_alphabeta_config("").unwrap();
        assert_eq!(cfg, AlphaBetaConfig::default());
    }

    #[test]
    fn mcts_spec_sets_the_named_fields() {
        let agent = make_agent_from_spec("mcts-uct:exploration=1.2", 1).unwrap();
        assert_eq!(agent.spec().name, "mcts-uct");
        assert!(agent.spec().params.contains("1.2"));
    }

    #[test]
    fn mcts_config_parser_reads_every_supported_key() {
        let cfg = parse_mcts_config(
            "exploration=2.0,chance_widen_c=0.5,chance_widen_alpha=0.25,\
             max_rollout_plies=100,time_check_interval=32,rollout=uniform,race=medium,dets=4",
        )
        .unwrap();
        assert_eq!(cfg.exploration, 2.0);
        assert_eq!(cfg.chance_widen_c, 0.5);
        assert_eq!(cfg.chance_widen_alpha, 0.25);
        assert_eq!(cfg.max_rollout_plies, 100);
        assert_eq!(cfg.time_check_interval, 32);
        assert_eq!(cfg.rollout, RolloutWeights::UNIFORM);
        assert_eq!(cfg.race, RaceWeights::MEDIUM);
        assert_eq!(cfg.root_determinizations, 4);
    }

    /// The eval-scalar keys are strictly opt-in: a spec that names none of
    /// them must leave `eval_override` `None`, which is what keeps `mcts-eval`
    /// tracking `duels-eval` live exactly as its crate docs promise. This is
    /// the "off value is bit-identical" obligation for this feature.
    #[test]
    fn eval_scalar_keys_are_opt_in_and_do_not_pin_anything_by_themselves() {
        for spec in ["", "c=0.5", "base=rollout", "leaf=blend:0.3,race=mild"] {
            assert_eq!(
                parse_mcts_eval_config(spec).unwrap().eval_override,
                None,
                "spec {spec:?} must not pin an eval generation"
            );
        }
        // ...and naming one pins today's live default with just that field moved.
        let cfg = parse_mcts_eval_config("menu_lambda=0.408").unwrap();
        let mut want = duels_eval::Config::default();
        want.eval.menu.lambda = 0.408;
        assert_eq!(cfg.eval_override, Some(want));
        // An eval scalar layers on top of an explicit generation, like
        // `phased`'s keys layer on top of its `base=`.
        let cfg = parse_mcts_eval_config("eval=v6,to_move=0.0").unwrap();
        let mut want = duels_eval::Config::v6();
        want.eval.to_move = 0.0;
        assert_eq!(cfg.eval_override, Some(want));
        // A genuinely unknown key still fails rather than being swallowed.
        assert!(parse_mcts_eval_config("no_such_key=1.0").is_err());
    }

    /// Round ten's option has to be reachable from both agents' specs, under
    /// the same key name, and has to show up in the recorded spec — otherwise
    /// the two arms of its A/B are indistinguishable in a results file after
    /// the fact.
    #[test]
    fn the_science_progress_key_reaches_both_agents_and_shows_up_in_the_spec() {
        // Default is the root reading, for both.
        assert_eq!(
            parse_phased_config("").unwrap().blend.science_progress,
            ScienceProgress::Root
        );
        assert_eq!(parse_mcts_eval_config("").unwrap().eval_override, None);

        for value in ["leaf", "on"] {
            assert_eq!(
                parse_phased_config(&format!("sciprog={value}"))
                    .unwrap()
                    .blend
                    .science_progress,
                ScienceProgress::Leaf
            );
            let pinned = parse_mcts_eval_config(&format!("sci_progress={value}"))
                .unwrap()
                .eval_override
                .expect("naming the key pins the evaluation");
            assert_eq!(pinned.blend.science_progress, ScienceProgress::Leaf);
        }
        // `root` is spellable too, and pins the control arm so the A/B differs
        // in exactly one field.
        let control = parse_mcts_eval_config("sciprog=root")
            .unwrap()
            .eval_override
            .expect("the control arm pins today's default too");
        assert_eq!(control, duels_eval::Config::default());

        // It layers on top of a frozen generation, like every other eval key.
        let cfg = parse_mcts_eval_config("eval=v6,sciprog=leaf")
            .unwrap()
            .eval_override
            .unwrap();
        let mut want = duels_eval::Config::v6();
        want.blend.science_progress = ScienceProgress::Leaf;
        assert_eq!(cfg, want);

        // And the recorded spec says which reading was in force.
        let leaf = make_agent_from_spec("mcts-eval:sciprog=leaf", 1).unwrap();
        assert!(leaf.spec().params.contains("sciprog=leaf"));
        let root = make_agent_from_spec("mcts-eval:sciprog=root", 1).unwrap();
        assert!(root.spec().params.contains("sciprog=root"));

        assert!(parse_phased_config("sciprog=nonsense").is_err());
        assert!(parse_mcts_eval_config("sciprog=nonsense").is_err());
    }

    /// A weight vector has to mean the same thing to both consumers of
    /// `duels-eval`, or an A/B run across the two agents silently measures two
    /// different candidates. Every key here is one of the 18 scalars the
    /// regression fit reports.
    #[test]
    fn mcts_eval_shares_phaseds_eval_key_names() {
        let spec = "vp_projection=2.121753,coins_div3=2.851599,development=0.069698,\
                    chain_equity=0.249180,resource_bill=2.009706,science_ladder=0.731634,\
                    military_band=6.167215,military_loot=-6.316501,\
                    military_endgame_urgency=0.979549,start1=2.461020,start2=5.139748,\
                    start3=4.272387,wonder_potential=0.304082,guild_projection=2.693124,\
                    yellow_equity=2.320637,token_equity=0.655919,to_move=1.589933,\
                    menu_lambda=0.407872";
        let from_mcts = parse_mcts_eval_config(spec)
            .unwrap()
            .eval_override
            .unwrap()
            .eval;
        let from_phased = parse_phased_config(spec).unwrap().eval;
        assert_eq!(from_mcts, from_phased);
        // And the vector really did move off the shipped weights.
        assert_ne!(from_mcts, duels_eval::Config::default().eval);
    }

    /// The race key has to reach every shipped variant *and* show up in the
    /// spec a results file records, or an arena run cannot be told apart from
    /// its baseline after the fact.
    #[test]
    fn the_race_key_reaches_every_variant_and_shows_up_in_the_spec() {
        assert_eq!(parse_mcts_config("").unwrap().race, RaceWeights::NEUTRAL);
        for (value, want) in [
            ("neutral", RaceWeights::NEUTRAL),
            ("off", RaceWeights::NEUTRAL),
            ("tier1", RaceWeights::TIER1_ONLY),
            ("tier1_only", RaceWeights::TIER1_ONLY),
            ("mild", RaceWeights::mild()),
            ("medium", RaceWeights::MEDIUM),
            ("strong", RaceWeights::strong()),
        ] {
            let cfg = parse_mcts_config(&format!("race={value}")).unwrap();
            assert_eq!(cfg.race, want, "race={value}");
        }
        assert!(parse_mcts_config("race=sideways").is_err());

        let agent = make_agent_from_spec("mcts-uct:race=medium", 1).unwrap();
        assert!(
            agent.spec().params.contains("race=medium"),
            "{}",
            agent.spec().params
        );
    }

    #[test]
    fn the_prior_key_reaches_every_mode_and_shows_up_in_the_spec() {
        assert_eq!(
            parse_mcts_config("prior=none").unwrap().prior,
            PriorMode::None
        );
        assert_eq!(
            parse_mcts_config("prior=expansion_order").unwrap().prior,
            PriorMode::ExpansionOrder
        );
        assert_eq!(
            parse_mcts_config("prior=order").unwrap().prior,
            PriorMode::ExpansionOrder
        );
        assert_eq!(
            parse_mcts_config("prior=progressive_bias:2.5")
                .unwrap()
                .prior,
            PriorMode::ProgressiveBias { weight: 2.5 }
        );
        assert_eq!(
            parse_mcts_config("prior=bias:0.5").unwrap().prior,
            PriorMode::ProgressiveBias { weight: 0.5 }
        );
        assert_eq!(
            parse_mcts_config("prior=progressive_bias").unwrap().prior,
            PriorMode::ProgressiveBias { weight: 1.0 }
        );
        // Unknown modes, and a weight on a mode that takes none, are errors
        // rather than a silently-wrong benchmark.
        assert!(parse_mcts_config("prior=sideways").is_err());
        assert!(parse_mcts_config("prior=expansion_order:2").is_err());
        assert!(parse_mcts_config("prior=bias:not_a_number").is_err());

        // ... and the mode reaches the spec a results file records, so a run
        // can be told apart from its baseline after the fact.
        let agent = make_agent_from_spec("mcts-uct:prior=expansion_order", 1).unwrap();
        assert!(
            agent.spec().params.contains("prior=expansion_order"),
            "{}",
            agent.spec().params
        );
    }

    /// `mcts-eval`'s leaf-value family, round-tripped: every variant parses,
    /// the parameterised ones carry their parameter, and each one reaches the
    /// spec string a results file records.
    #[test]
    fn the_leaf_key_reaches_every_variant_and_shows_up_in_the_spec() {
        // The default is the configuration that was measured, and the crate
        // exists to be it.
        let cfg = parse_mcts_eval_config("").unwrap();
        assert_eq!(cfg.leaf, LeafValue::Blend { weight: 0.5 });
        assert_eq!(cfg.exploration, 0.5);
        for (value, want) in [
            ("rollout", LeafValue::Rollout),
            ("off", LeafValue::Rollout),
            ("static", LeafValue::Static),
            ("truncated:4", LeafValue::Truncated { plies: 4 }),
            ("trunc:16", LeafValue::Truncated { plies: 16 }),
            ("truncated", LeafValue::Truncated { plies: 8 }),
            ("blend:0.3", LeafValue::Blend { weight: 0.3 }),
            ("blend", LeafValue::Blend { weight: 0.5 }),
            // The two opt-in `duels-value` learned leaves.
            ("learned", LeafValue::Learned),
            ("learned_blend", LeafValue::LearnedBlend { weight: 0.5 }),
            ("learned_blend:0.3", LeafValue::LearnedBlend { weight: 0.3 }),
            ("lblend:0.7", LeafValue::LearnedBlend { weight: 0.7 }),
        ] {
            let cfg = parse_mcts_eval_config(&format!("leaf={value}")).unwrap();
            assert_eq!(cfg.leaf, want, "leaf={value}");
        }
        // Unknown variants, a parameter on one that takes none, and an
        // unparsable parameter are all errors rather than a silently-wrong
        // benchmark.
        assert!(parse_mcts_eval_config("leaf=sideways").is_err());
        assert!(parse_mcts_eval_config("leaf=static:3").is_err());
        assert!(parse_mcts_eval_config("leaf=trunc:not_a_number").is_err());
        assert!(parse_mcts_eval_config("leaf=blend:not_a_number").is_err());
        // A blend weight outside [0, 1] is rejected rather than accepted into
        // a search whose leaf values are then not probabilities.
        assert!(parse_mcts_eval_config("leaf=blend:1.5").is_err());
        assert!(parse_mcts_eval_config("leaf=blend:-0.5").is_err());
        assert!(parse_mcts_eval_config("leaf=blend:0.0").is_ok());
        assert!(parse_mcts_eval_config("leaf=blend:1.0").is_ok());
        // ...and the learned blend is range-checked for exactly the same
        // reason, so the check cannot be added to one and forgotten on the
        // other.
        assert!(parse_mcts_eval_config("leaf=learned_blend:1.5").is_err());
        assert!(parse_mcts_eval_config("leaf=learned_blend:-0.5").is_err());
        assert!(parse_mcts_eval_config("leaf=learned:3").is_err());

        for (spec, want) in [
            ("mcts-eval:leaf=static", "leaf=static"),
            ("mcts-eval:leaf=learned", "leaf=learned"),
            ("mcts-eval:leaf=lblend:0.25", "leaf=learned_blend(0.250)"),
            ("mcts-eval:leaf=trunc:8", "leaf=truncated(8)"),
            ("mcts-eval:leaf=blend:0.3", "leaf=blend(0.300)"),
        ] {
            let agent = make_agent_from_spec(spec, 1).unwrap();
            assert!(
                agent.spec().params.contains(want),
                "{spec}: {}",
                agent.spec().params
            );
        }
    }

    /// `base=rollout` is the ablation control every `mcts-eval` strength claim
    /// is measured against, and it has to be the *whole* control — a playout
    /// leaf **and** the unrescaled exploration constant, since the two move
    /// together.
    #[test]
    fn the_mcts_eval_rollout_base_is_the_whole_control() {
        let cfg = parse_mcts_eval_config("base=rollout").unwrap();
        assert_eq!(cfg, MctsEvalConfig::rollout_base());
        assert_eq!(cfg.leaf, LeafValue::Rollout);
        assert_eq!(cfg.exploration, 1.0);
        // Keys after `base` override it, exactly as `alphabeta`'s do.
        assert_eq!(
            parse_mcts_eval_config("base=rollout,c=0.7")
                .unwrap()
                .exploration,
            0.7
        );
        assert_eq!(
            parse_mcts_eval_config("base=default").unwrap(),
            MctsEvalConfig::default()
        );
        assert!(parse_mcts_eval_config("base=sideways").is_err());

        // The bare name and the search keys behave like `mcts-uct`'s.
        let agent = make_agent_from_spec("mcts-eval", 1).unwrap();
        assert_eq!(agent.spec().name, "mcts-eval");
        assert!(agent.spec().params.contains("c=0.500"));
        let agent = make_agent_from_spec("mcts-eval:base=rollout", 1).unwrap();
        assert!(agent.spec().params.contains("leaf=rollout"));
        assert!(agent.spec().params.contains("c=1.000"));
        assert!(agent.spec().params.contains("eval=unused"));
        assert!(make_agent_from_spec("mcts-eval:race=tier1,dets=2,prior=order", 1).is_ok());
        assert!(make_agent_from_spec("mcts-eval:nonsense=1", 1).is_err());
    }

    /// **`mcts-value`'s whole ablation chain is addressable from one binary**,
    /// and each link is the agent it claims to be.
    ///
    /// The chain is `mcts-value` (learned leaf, swept `c = 0.15`) against
    /// `base=eval` (`mcts-eval`) against `base=rollout` (`mcts-uct`). The
    /// *identity* of the two controls is asserted move-for-move inside
    /// `duels-agent-mcts-value` against verbatim frozen copies of those
    /// searches; what is checked here is that these spec strings really select
    /// them, since a typo in this parser would silently benchmark the wrong
    /// arm and still produce a plausible win rate.
    #[test]
    fn the_mcts_value_ablation_chain_is_addressable() {
        // The bare name is the measured configuration.
        let cfg = parse_mcts_value_config("").unwrap();
        assert_eq!(cfg, MctsValueConfig::default());
        assert_eq!(cfg.leaf, ValueLeafValue::Learned);
        assert_eq!(cfg.exploration, 0.15);

        // The two controls, whole: leaf *and* exploration constant, since the
        // two move together in each.
        let eval = parse_mcts_value_config("base=eval").unwrap();
        assert_eq!(eval, MctsValueConfig::eval_base());
        assert_eq!(eval.leaf, ValueLeafValue::Blend { weight: 0.5 });
        assert_eq!(eval.exploration, 0.5);
        // `eval_base` must not pin the evaluation: the control has to be
        // `mcts-eval` as shipped, tracking `duels-eval` live.
        assert_eq!(eval.eval_override, None);

        let rollout = parse_mcts_value_config("base=rollout").unwrap();
        assert_eq!(rollout, MctsValueConfig::rollout_base());
        assert_eq!(rollout.leaf, ValueLeafValue::Rollout);
        assert_eq!(rollout.exploration, 1.0);

        // Keys after `base` override it, as everywhere else in this module.
        assert_eq!(
            parse_mcts_value_config("base=eval,c=0.7")
                .unwrap()
                .exploration,
            0.7
        );
        assert_eq!(
            parse_mcts_value_config("base=default").unwrap(),
            MctsValueConfig::default()
        );
        assert!(parse_mcts_value_config("base=sideways").is_err());

        // ...and the three arms are distinguishable in the spec string a
        // results file records, which is what makes a run interpretable after
        // the fact.
        for (spec, wants, rejects) in [
            (
                "mcts-value",
                vec!["leaf=learned;", "c=0.150", "eval=unused", "value="],
                vec!["leaf=blend"],
            ),
            (
                "mcts-value:base=eval",
                vec!["leaf=blend(0.500)", "c=0.500"],
                vec!["value="],
            ),
            (
                "mcts-value:base=rollout",
                vec!["leaf=rollout", "c=1.000", "eval=unused"],
                vec!["value="],
            ),
        ] {
            let agent = make_agent_from_spec(spec, 1).unwrap();
            assert_eq!(agent.spec().name, "mcts-value");
            let params = agent.spec().params;
            for want in wants {
                assert!(params.contains(want), "{spec} lacks {want}: {params}");
            }
            for reject in rejects {
                assert!(!params.contains(reject), "{spec} has {reject}: {params}");
            }
        }
    }

    /// `weights=file:<path>` loads a not-yet-promoted candidate's bytes from
    /// disk at runtime instead of requiring a new compiled-in `WEIGHTS_*`
    /// constant and `agent_spec` match arm per candidate -- the autonomous
    /// self-play loop's own gating needs exactly this (`docs/roadmap.md`'s
    /// "Autonomous self-play loop design").
    #[test]
    fn weights_file_loads_a_candidate_from_disk_by_path() {
        let dir = std::env::temp_dir().join(format!(
            "duels-agent-spec-weights-file-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("candidate.bin");
        std::fs::write(&path, duels_agent_mcts_value::WEIGHTS_V2).unwrap();
        let path_str = path.to_str().unwrap();

        let cfg = parse_mcts_value_config(&format!("weights=file:{path_str}")).unwrap();
        assert_eq!(
            cfg.value_weights_override,
            Some(duels_agent_mcts_value::WEIGHTS_V2)
        );

        // A second load of the same path is served from the cache (and
        // returns the identical bytes) rather than leaking a fresh
        // allocation per call -- this function is called once per game, so
        // an experiment of any real size calls it thousands of times.
        let cfg_again = parse_mcts_value_config(&format!("weights=file:{path_str}")).unwrap();
        assert_eq!(
            cfg_again.value_weights_override.unwrap().as_ptr(),
            cfg.value_weights_override.unwrap().as_ptr(),
            "the second load of the same path should reuse the cached allocation"
        );

        // A missing path is a normal, catchable error, not a panic.
        let err =
            parse_mcts_value_config("weights=file:/definitely/not/a/real/path.bin").unwrap_err();
        assert!(err.contains("failed to read weights file"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `leaf=learned_symmetric` (and its `lsym` alias) reach
    /// [`ValueLeafValue::LearnedSymmetric`] — the exploratory, opt-in
    /// averaged-perspective leaf `docs/roadmap.md`'s Tier 0-B added. Opt-in
    /// means nothing here changes what `leaf=learned` (the default) parses
    /// to; that is `the_mcts_value_ablation_chain_is_addressable`'s job.
    #[test]
    fn the_learned_symmetric_leaf_is_addressable_by_spec_string() {
        for value in ["learned_symmetric", "lsym"] {
            let cfg = parse_mcts_value_config(&format!("leaf={value}")).unwrap();
            assert_eq!(cfg.leaf, ValueLeafValue::LearnedSymmetric, "leaf={value}");
        }
        let agent = make_agent_from_spec("mcts-value:leaf=learned_symmetric", 1).unwrap();
        assert!(
            agent.spec().params.contains("leaf=learned_symmetric"),
            "{}",
            agent.spec().params
        );
    }

    /// `mcts-value` keeps `mcts-eval`'s keys, and the two agents' defaults
    /// disagree about which of them is on the default path — asserted here
    /// because it is the kind of asymmetry a reader will assume away.
    ///
    /// `value_sum` is live for `mcts-value` and opt-in-only for `mcts-eval`;
    /// the `duels-eval` keys are the other way round. Both parsers accept both
    /// sets, so the difference is in what reaches a *default* search, not in
    /// what parses.
    #[test]
    fn the_mcts_value_keys_are_the_mcts_eval_keys_with_the_defaults_reversed() {
        // The learned summation order is on `mcts-value`'s default path...
        let serial = parse_mcts_value_config("value_sum=serial").unwrap();
        assert_eq!(serial.value_summation, duels_value::Summation::Serial);
        assert!(serial.leaf.needs_learned_net());
        let params = make_agent_from_spec("mcts-value:value_sum=serial", 1)
            .unwrap()
            .spec()
            .params;
        assert!(params.contains("/serial"), "{params}");
        assert!(parse_mcts_value_config("value_sum=sideways").is_err());

        // ...and the hand-crafted evaluation is not: pinning a generation
        // parses, but the default leaf reads no evaluation, so the spec string
        // still says `eval=unused`.
        let pinned = make_agent_from_spec("mcts-value:eval=v6", 1)
            .unwrap()
            .spec()
            .params;
        assert!(pinned.contains("eval=unused"), "{pinned}");
        // On the `base=eval` control the same key does reach the search.
        let pinned = make_agent_from_spec("mcts-value:base=eval,eval=v6", 1)
            .unwrap()
            .spec()
            .params;
        assert!(
            pinned.contains(&duels_eval::Config::v6().params_string()),
            "{pinned}"
        );
        assert!(parse_mcts_value_config("eval=live").is_err());
        assert!(parse_mcts_value_config("eval=v99").is_err());

        // The search keys are `mcts-eval`'s, spelled identically.
        assert!(make_agent_from_spec("mcts-value:race=tier1,dets=2,prior=order", 1).is_ok());
        assert!(make_agent_from_spec("mcts-value:leaf=lblend:0.5,c=0.5", 1).is_ok());
        assert!(make_agent_from_spec("mcts-value:nonsense=1", 1).is_err());
        // And the leaf blend weights are range-checked here too, so the check
        // cannot be added to one parser and forgotten in the other.
        assert!(parse_mcts_value_config("leaf=blend:1.5").is_err());
        assert!(parse_mcts_value_config("leaf=lblend:-0.5").is_err());
        assert!(parse_mcts_value_config("leaf=learned:3").is_err());
        assert!(parse_mcts_value_config("leaf=sideways").is_err());
    }

    /// `objective` is the key the specialist research direction hangs off
    /// of: `win` (the default) reproduces every existing search, and
    /// `science`/`military`/`civilian` build a specialist that reuses the
    /// same trained weights under a different reward. Parsed here the same
    /// way `leaf` is above: every accepted value round-trips to the right
    /// [`ValueObjective`], and it shows up in the recorded spec string so a
    /// results file says which objective a game was played under.
    #[test]
    fn the_objective_key_reaches_every_specialist_and_shows_up_in_the_spec() {
        // The default parses explicitly to the same thing an empty parameter
        // list already gives you, and leaves no trace in the spec string --
        // this is the "changes nothing at its default" property, at the
        // spec-string layer rather than the `Config` layer.
        for value in ["win", "win_probability", "default"] {
            let cfg = parse_mcts_value_config(&format!("objective={value}")).unwrap();
            assert_eq!(
                cfg.objective,
                ValueObjective::WinProbability,
                "objective={value}"
            );
        }
        assert_eq!(
            parse_mcts_value_config("").unwrap().objective,
            ValueObjective::WinProbability
        );
        let default_params = make_agent_from_spec("mcts-value", 1).unwrap().spec().params;
        assert!(
            !default_params.contains("objective="),
            "the default objective must not appear in the spec string: {default_params}"
        );

        for (value, kind, name) in [
            ("science", VictoryKind::ScientificSupremacy, "science"),
            ("sci", VictoryKind::ScientificSupremacy, "science"),
            ("military", VictoryKind::MilitarySupremacy, "military"),
            ("mil", VictoryKind::MilitarySupremacy, "military"),
            ("civilian", VictoryKind::CivilianVictory, "civilian"),
            ("civ", VictoryKind::CivilianVictory, "civilian"),
        ] {
            let cfg = parse_mcts_value_config(&format!("objective={value}")).unwrap();
            assert_eq!(
                cfg.objective,
                ValueObjective::TargetKind(kind),
                "objective={value}"
            );
            let params = make_agent_from_spec(&format!("mcts-value:objective={value}"), 1)
                .unwrap()
                .spec()
                .params;
            assert!(
                params.contains(&format!("objective=target({name})")),
                "objective={value} did not show up in the spec: {params}"
            );
        }

        assert!(parse_mcts_value_config("objective=sideways").is_err());
        assert!(parse_mcts_value_config("obj=science").is_ok());
    }

    /// **`mcts-eval` tracks `duels-eval` live by default, on purpose**, so
    /// the old `evalgen`/`eval_generation` key names are rejected with an
    /// error that explains itself and points at the real mechanism
    /// (`eval=vN`, tested separately below) rather than being silently
    /// ignored or mistaken for a permanent pin. See that crate's docs; this
    /// is the opposite default from `mcts-uct`'s old pin and must not be
    /// "fixed" back.
    #[test]
    fn the_old_evalgen_key_names_point_at_the_real_mechanism() {
        let err = parse_mcts_eval_config("evalgen=v6").unwrap_err();
        assert!(err.contains("tracks"), "{err}");
        assert!(err.contains("eval=vN"), "{err}");
        assert!(parse_mcts_eval_config("eval_generation=v6").is_err());

        // ...and by default the spec string carries the whole live
        // configuration in its place, which is what makes a results file
        // interpretable later.
        let params = make_agent_from_spec("mcts-eval", 1).unwrap().spec().params;
        assert!(
            params.ends_with(&format!(
                "eval={}",
                duels_eval::Config::default().params_string()
            )),
            "{params}"
        );
        assert!(!params.contains("evalgen="), "{params}");
    }

    /// `eval=vN` pins this one agent instance to a frozen `duels-eval`
    /// generation, for A/B testing a change directly against a live (default)
    /// `mcts-eval` in one binary — see
    /// `duels_agent_mcts_eval::Config::eval_override`. Not a second
    /// production default: `eval=default`/`eval=live` is rejected rather than
    /// silently accepted as a no-op spelling of the same thing.
    #[test]
    fn eval_v_n_pins_a_frozen_generation_for_ab_testing() {
        for (v, want) in [
            ("v1", duels_eval::Config::v1()),
            ("v2", duels_eval::Config::v2()),
            ("v3", duels_eval::Config::v3()),
            ("v4", duels_eval::Config::v4()),
            ("v5", duels_eval::Config::v5()),
            ("v6", duels_eval::Config::v6()),
            ("v7", duels_eval::Config::v7()),
            ("v8", duels_eval::Config::v8()),
            ("v9", duels_eval::Config::v9()),
        ] {
            let cfg = parse_mcts_eval_config(&format!("eval={v}")).unwrap();
            assert_eq!(cfg.eval_override, Some(want), "eval={v}");
        }
        // `v9` is round ten's control — the generation whose `menu.lambda` the
        // round moved — so it has to parse, and it has to be distinguishable
        // from the live default it was measured against.
        assert_ne!(
            parse_mcts_eval_config("eval=v9").unwrap().eval_override,
            Some(duels_eval::Config::default())
        );
        assert!(parse_mcts_eval_config("eval=v0").is_err());
        assert!(parse_mcts_eval_config("eval=v10").is_err());
        assert!(parse_mcts_eval_config("eval=default").is_err());
        assert!(parse_mcts_eval_config("eval=live").is_err());

        // The pin reaches the recorded spec, not just the parsed `Config` --
        // a results file has to say which generation an override actually
        // used, the same discipline the live default already gets.
        let agent = make_agent_from_spec("mcts-eval:eval=v1", 1).unwrap();
        assert!(
            agent.spec().params.ends_with(&format!(
                "eval={}",
                duels_eval::Config::v1().params_string()
            )),
            "{}",
            agent.spec().params
        );

        // And a plain `mcts-eval` is untouched: still live, still no override.
        assert_eq!(parse_mcts_eval_config("").unwrap().eval_override, None);
    }

    /// The two keys the leaf-value work introduced on `mcts-uct` are gone from
    /// it, and the error says where they went rather than reading as a typo.
    /// A stale `mcts-uct:leaf=blend:0.5,c=0.5` command line is exactly the
    /// thing most likely to be re-run out of a doc comment or a shell history.
    #[test]
    fn the_leaf_keys_are_gone_from_mcts_uct_and_say_where_they_went() {
        for spec in ["leaf=blend:0.5", "leaf=rollout", "evalgen=v6"] {
            let err = parse_mcts_config(spec).unwrap_err();
            assert!(err.contains("mcts-eval"), "{spec}: {err}");
        }
        assert!(make_agent_from_spec("mcts-uct:leaf=blend:0.5,c=0.5", 1).is_err());
        // The search keys it does still own are untouched.
        assert!(make_agent_from_spec("mcts-uct:c=0.5,race=tier1", 1).is_ok());
    }

    /// Root ensembling is the one knob whose *default* has to keep reading
    /// `1`, since that is what makes the shipped agents the pre-ensembling
    /// ones; the sweep behind those crates' docs is driven by these keys.
    #[test]
    fn root_ensembling_keys_reach_both_search_agents() {
        assert_eq!(parse_mcts_config("").unwrap().root_determinizations, 1);
        assert_eq!(
            parse_mcts_config("root_determinizations=8")
                .unwrap()
                .root_determinizations,
            8
        );
        let cfg = parse_alphabeta_config("").unwrap();
        assert_eq!(cfg.root_determinizations, 1);
        assert!(cfg.ensemble_exact_root);
        let cfg = parse_alphabeta_config("dets=4,exact=false").unwrap();
        assert_eq!(cfg.root_determinizations, 4);
        assert!(!cfg.ensemble_exact_root);
        // ... and they show up in the spec the results file records.
        let agent = make_agent_from_spec("mcts-uct:dets=2", 1).unwrap();
        assert!(
            agent.spec().params.contains("dets=2"),
            "{}",
            agent.spec().params
        );
        let agent = make_agent_from_spec("alphabeta:dets=2", 1).unwrap();
        assert!(
            agent.spec().params.contains("dets=2"),
            "{}",
            agent.spec().params
        );
    }

    #[test]
    fn phased_base_v1_is_the_configuration_the_crate_shipped_with() {
        assert_eq!(parse_phased_config("base=v1").unwrap(), PhasedConfig::v1());
        assert_eq!(parse_phased_config("base=v2").unwrap(), PhasedConfig::v2());
        assert_eq!(parse_phased_config("").unwrap(), PhasedConfig::default());
        // The round-three keys, and their "off" values reproducing v2's --
        // round four's `pending=unresolved` and round five's six included,
        // since `v2()` is built on `v3()` on `v4()` and so carries every later
        // option at its own off value too. Round ten's `lambda=0.6` is in every
        // one of these strings for the same reason.
        let off = parse_phased_config(
            "rails=off,imminent=0,shieldprice=onesided,horizon=supply,lockin=0,band=2.0,\
             pending=unresolved,guild=unpriced,guildproj=0,menufloor=none,afford=0,\
             supply=raw,yellow=0,wprem=0,science_ladder=1,chaineq=1,bill=3,\
             development=0.3333333333333333,pairthreat=1,dead=1,\
             ladder4=12,ladder5=18,temp1=47.57,temp2=43.75,temp3=25.18,lambda=0.6",
        )
        .unwrap();
        assert_eq!(off, PhasedConfig::v2());
        assert_eq!(
            parse_phased_config("horizon=5").unwrap().military_horizon,
            Some(5.0)
        );
        assert!(parse_phased_config("rails=maybe").is_err());
        assert!(parse_phased_config("shieldprice=sideways").is_err());
        assert!(parse_phased_config("horizon=wide").is_err());
        assert_ne!(PhasedConfig::v1(), PhasedConfig::default());

        // The round-four keys, and their "off" values reproducing v3's.
        assert_eq!(parse_phased_config("base=v3").unwrap(), PhasedConfig::v3());
        let off =
            parse_phased_config("pending=unresolved,wonder=flat,destroyrepl=off,base=v3").unwrap();
        assert_eq!(off, PhasedConfig::v3());
        assert_eq!(
            parse_phased_config(
                "pending=unresolved,wonder=flat,destroyrepl=off,wprem=0,guild=unpriced,guildproj=0,\
                 menufloor=none,afford=0,supply=raw,yellow=0,science_ladder=1,chaineq=1,bill=3,\
                 development=0.3333333333333333,pairthreat=1,dead=1,\
                 ladder4=12,ladder5=18,temp1=47.57,temp2=43.75,temp3=25.18,lambda=0.6"
            )
            .unwrap(),
            PhasedConfig::v3()
        );
        let on = parse_phased_config("wonder=budget,destroyrepl=on,wturns=3,wextra=2").unwrap();
        assert_eq!(on.wonder_model, WonderModel::Budget);
        assert!(on.destroy_replace_discount);
        assert_eq!(on.eval.wonder_turns_per_wonder, 3.0);
        assert_eq!(on.eval.wonder_extra_turn_vp, 2.0);
        assert!(parse_phased_config("pending=sometimes").is_err());
        assert!(parse_phased_config("wonder=lavish").is_err());
        assert!(parse_phased_config("destroyrepl=perhaps").is_err());
        assert_ne!(PhasedConfig::v3(), PhasedConfig::default());

        // The round-five keys, and their "off" values reproducing v4's.
        assert_eq!(parse_phased_config("base=v4").unwrap(), PhasedConfig::v4());
        assert_eq!(
            parse_phased_config(
                "guild=unpriced,guildproj=0,menufloor=none,afford=0,supply=raw,yellow=0,wprem=0,\
                 science_ladder=1,chaineq=1,bill=3,development=0.3333333333333333,\
                 pairthreat=1,dead=1,ladder4=12,ladder5=18,\
                 temp1=47.57,temp2=43.75,temp3=25.18,lambda=0.6"
            )
            .unwrap(),
            PhasedConfig::v4()
        );
        let on = parse_phased_config(
            "guild=projected,guildproj=0.5,menufloor=discardwonder,afford=2.5,supply=dealt,\
             yellow=0.75,discardrate=0.3",
        )
        .unwrap();
        assert_eq!(on.guild_pricing, GuildPricing::Projected);
        assert_eq!(on.eval.guild_projection, 0.5);
        assert_eq!(on.menu_floor, MenuFloor::DiscardAndWonder);
        assert_eq!(on.menu_afford_soft, 2.5);
        assert_eq!(on.supply_model, SupplyModel::Dealt);
        assert_eq!(on.eval.yellow_equity, 0.75);
        assert_eq!(on.eval.yellow_discard_rate, 0.3);
        assert_eq!(
            parse_phased_config("menufloor=discard").unwrap().menu_floor,
            MenuFloor::Discard
        );
        assert!(parse_phased_config("guild=freehand").is_err());
        assert!(parse_phased_config("menufloor=basement").is_err());
        assert!(parse_phased_config("supply=plentiful").is_err());

        // The round-six key, and its "off" value reproducing v5's.
        assert_eq!(parse_phased_config("base=v5").unwrap(), PhasedConfig::v5());
        assert_eq!(
            parse_phased_config(
                "wprem=0,science_ladder=1,chaineq=1,bill=3,development=0.3333333333333333,\
                 pairthreat=1,dead=1,ladder4=12,ladder5=18,\
                 temp1=47.57,temp2=43.75,temp3=25.18,lambda=0.6"
            )
            .unwrap(),
            PhasedConfig::v5(),
            "the extra-turn premium is the only thing round six changed"
        );
        // The round-eight keys, and their "off" values reproducing v7's: the
        // top two science ladder rungs and the three leaf temperatures.
        assert_eq!(parse_phased_config("base=v7").unwrap(), PhasedConfig::v7());
        assert_eq!(
            parse_phased_config(
                "ladder4=12,ladder5=18,temp1=47.57,temp2=43.75,temp3=25.18,lambda=0.6"
            )
            .unwrap(),
            PhasedConfig::v7(),
            "the ladder's top rungs and the leaf temperature are the only \
             things round eight changed"
        );
        // Round nine's two options. **It did not move the default** -- it
        // measured both and left them off -- so unlike every other block here
        // there is no "off" string to reproduce a previous generation with,
        // and `base=v9` is `base=v8`. What the keys are for is reaching the
        // options from a spec string at all, which is how the round's own
        // transfer checks against `alphabeta` and `mcts-uct` were run. `pref`
        // is read only under `wonder=rationed`.
        assert_eq!(parse_phased_config("base=v8").unwrap(), PhasedConfig::v8());
        assert_eq!(parse_phased_config("base=v9").unwrap(), PhasedConfig::v9());
        assert_eq!(PhasedConfig::v8(), PhasedConfig::v9());
        assert_eq!(
            parse_phased_config("wonder=flat,wonder_potential=0.5,reach=optimistic,pref=1")
                .unwrap(),
            PhasedConfig::default(),
            "round nine's options are off in the default, so naming their off \
             values has to be a no-op"
        );
        assert_eq!(
            parse_phased_config(
                "wonder=flat,wonder_potential=0.5,reach=optimistic,pref=1,lambda=0.6"
            )
            .unwrap(),
            PhasedConfig::v9(),
            "...and naming them alongside round ten's off value has to reach \
             round nine's configuration"
        );
        // Round ten's one key, and its "off" value reproducing v9's: the
        // opponent-menu weight, 0.6 -> the fitted 0.408.
        assert_eq!(
            parse_phased_config("lambda=0.6").unwrap(),
            PhasedConfig::v9(),
            "the opponent-menu weight is the only thing round ten changed"
        );
        assert_ne!(PhasedConfig::v9(), PhasedConfig::default());
        let on = parse_phased_config("wonder=rationed,reach=structure,pref=0.875").unwrap();
        assert_eq!(on.wonder_model, WonderModel::Rationed);
        assert_eq!(on.eval.science.reach_model, ReachModel::Structure);
        assert_eq!(on.eval.wonder_p_build_ref, 0.875);
        assert!(parse_phased_config("reach=hopeful").is_err());
        let on =
            parse_phased_config("ladder1=0.5,ladder2=2,ladder3=7,ladder4=25,temp1=30").unwrap();
        assert_eq!(on.eval.science.ladder, [0.0, 0.5, 2.0, 7.0, 25.0, 54.0]);
        assert_eq!(on.eval.win_probability_temperature[0], 30.0);
        assert_ne!(PhasedConfig::v7(), PhasedConfig::default());
        assert_eq!(
            parse_phased_config("wonder_extra_turn_premium=4.5")
                .unwrap()
                .eval
                .wonder_extra_turn_premium,
            4.5
        );
        assert_ne!(PhasedConfig::v5(), PhasedConfig::default());
    }

    #[test]
    fn phased_config_parser_reads_every_supported_key() {
        let cfg = parse_phased_config(
            "mil=legacy,coin=smooth,econ=bill,blend=off,lambda=0.4,tau=2.0,chaineq=0.5,\
             bill=0.9,band=0.8,loot=0.7,kappa=0.9,military_sigma_min=0.4,\
             military_logistic_scale=0.6,beta=0.5,cref=6,military_position=0.2,\
             vp_projection=1.1,development=0.4,science_ladder=1.2,deny=0.9,\
             deny_chain_gift=0.1,wonder_potential=0.6,start1=1.5,start2=1,start3=0",
        )
        .unwrap();
        assert_eq!(cfg.military_model, MilitaryModel::Legacy);
        assert_eq!(cfg.coin_model, CoinModel::Smooth);
        assert_eq!(cfg.economy_model, EconomyModel::Bill);
        assert!(!cfg.blend.enabled);
        assert_eq!(cfg.eval.menu.lambda, 0.4);
        assert_eq!(cfg.eval.menu.tau, 2.0);
        assert_eq!(cfg.eval.chain_equity, 0.5);
        assert_eq!(cfg.eval.resource_bill, 0.9);
        assert_eq!(cfg.eval.military_band, 0.8);
        assert_eq!(cfg.eval.military_loot, 0.7);
        assert_eq!(cfg.eval.military_sigma_scale, 0.9);
        assert_eq!(cfg.eval.military_sigma_min, 0.4);
        assert_eq!(cfg.eval.military_logistic_scale, 0.6);
        assert_eq!(cfg.eval.coin_smooth_beta, 0.5);
        assert_eq!(cfg.eval.coin_smooth_ref, 6.0);
        assert_eq!(cfg.eval.next_age_start, [1.5, 1.0, 0.0]);
        assert!(parse_phased_config("mil=sideways").is_err());
        assert!(parse_phased_config("not_a_real_key=1").is_err());
    }

    /// The model switches have to reach the spec string a results file
    /// records, or two runs of the same ablation campaign are
    /// indistinguishable after the fact.
    #[test]
    fn the_phased_models_show_up_in_the_recorded_spec() {
        let agent = make_agent_from_spec("phased:mil=band,coin=smooth,econ=bill", 1).unwrap();
        assert_eq!(agent.spec().name, "phased");
        assert!(
            agent.spec().params.contains("models=band/smooth/bill"),
            "{}",
            agent.spec().params
        );
        let old = make_agent_from_spec("phased:base=v1", 1).unwrap();
        assert!(
            old.spec().params.contains("models=legacy/legacy/legacy"),
            "{}",
            old.spec().params
        );
        assert_ne!(agent.spec().params, old.spec().params);
    }

    #[test]
    fn random_rejects_any_parameters() {
        assert!(make_agent_from_spec("random:seed=5", 1).is_err());
    }

    #[test]
    fn unknown_agent_name_is_rejected() {
        // `Box<dyn Agent>` doesn't implement `Debug`, so `unwrap_err` (which
        // requires `T: Debug` for its panic message) doesn't type-check here;
        // match it out by hand instead, as `agent_registry`'s equivalent test
        // does.
        let err = match make_agent_from_spec("nonexistent:foo=1", 1) {
            Ok(_) => panic!("expected an error for an unknown agent name"),
            Err(e) => e,
        };
        assert!(err.contains("nonexistent"));
    }

    #[test]
    fn unknown_key_is_rejected_not_panicked() {
        assert!(parse_alphabeta_config("not_a_real_key=1").is_err());
        assert!(parse_mcts_config("not_a_real_key=1").is_err());
        assert!(parse_mcts_eval_config("not_a_real_key=1").is_err());
        assert!(parse_phased_config("not_a_real_key=1").is_err());
    }

    #[test]
    fn malformed_key_value_pair_is_rejected() {
        // No "=" at all.
        assert!(parse_alphabeta_config("max_depth").is_err());
        // An empty value fails to parse as the field's numeric type.
        assert!(parse_alphabeta_config("max_depth=").is_err());
        // An empty key is accepted as a split but rejected as an unknown key.
        assert!(parse_alphabeta_config("=5").is_err());
    }

    #[test]
    fn invalid_value_is_rejected_with_the_key_named() {
        let err = parse_alphabeta_config("max_depth=not_a_number").unwrap_err();
        assert!(err.contains("max_depth"));
    }
}
