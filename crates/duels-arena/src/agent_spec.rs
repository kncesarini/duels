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
//!   `root_determinizations`/`dets`, and `prior` (`none`, `expansion_order`,
//!   or `progressive_bias:<weight>`).
//! * `greedy` -- every [`duels_agent_greedy::EvalWeights`] field, by its own
//!   name (`military_position`, `military_endgame_urgency`,
//!   `science_distinct_symbol`, `science_near_supremacy`,
//!   `science_pair_setup`, `vp_projection`, `coins_div3`,
//!   `coin_safety_floor`, `coin_safety_penalty`, `resource_vulnerability`,
//!   `deny_chain_gift`, `wonder_potential`, `instant_result`).
//! * `greedy-ev` -- the same field names, against
//!   [`duels_agent_greedy_ev::EvalWeights`] (an identically-shaped struct in
//!   its own crate).
//! * `phased` -- `base` (`v1`/`v2`/`default`), the terminal rails
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
//!   `next_age_start` array as `start1`/`start2`/`start3`, and the individual
//!   weights `military_position`, `vp_projection`, `development`,
//!   `science_ladder`, `deny`, `deny_chain_gift`, `wonder_potential`.
//! * `random` -- bare name only; it has no parameters.
//!
//! # Examples
//!
//! ```
//! use duels_arena::agent_spec::make_agent_from_spec;
//!
//! let a = make_agent_from_spec("mcts-uct:exploration=1.2", 1).unwrap();
//! let b = make_agent_from_spec("alphabeta:max_depth=10,rollouts=4", 1).unwrap();
//! let c = make_agent_from_spec("random", 1).unwrap(); // bare name, unchanged
//! assert_eq!(a.spec().name, "mcts-uct");
//! assert_eq!(b.spec().name, "alphabeta");
//! assert_eq!(c.spec().name, "random");
//! ```

use duels_agent_alphabeta::{eval, playout, AlphaBetaAgent, Config as AlphaBetaConfig};
use duels_agent_greedy::{EvalWeights as GreedyWeights, GreedyAgent};
use duels_agent_greedy_ev::{EvalWeights as GreedyEvWeights, GreedyEvAgent};
use duels_agent_mcts_uct::{
    Config as MctsConfig, MctsAgent, PriorMode, RaceWeights, RolloutWeights,
};
use duels_agent_phased::{
    Blend as PhasedBlend, CoinModel, Config as PhasedConfig, EconomyModel, MenuShieldPricing,
    MilitaryModel, PhasedAgent, RailModel,
};
use duels_agents_api::Agent;

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
        "greedy" => {
            let w = parse_greedy_weights(params)?;
            Ok(Box::new(GreedyAgent::with_weights(seed, w)))
        }
        "greedy-ev" => {
            let w = parse_greedy_ev_weights(params)?;
            Ok(Box::new(GreedyEvAgent::with_weights(seed, w)))
        }
        "phased" => {
            let cfg = parse_phased_config(params)?;
            Ok(Box::new(PhasedAgent::with_config(seed, cfg)))
        }
        "random" => Err(format!(
            "\"random\" takes no parameters; use the bare name \"random\", not \"{spec}\""
        )),
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
            other => return Err(format!("mcts-uct: unknown key \"{other}\"")),
        }
    }
    Ok(cfg)
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
                "default" => cfg = PhasedConfig::default(),
                other => return Err(format!("phased: unknown base \"{other}\"")),
            },
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

/// Generates a `key=value` parser for one of the (identically-shaped, but
/// distinctly-typed) per-crate `EvalWeights` structs shared by `greedy` and
/// `greedy-ev`.
macro_rules! eval_weights_parser {
    ($(#[$meta:meta])* $fn_name:ident, $ty:ty) => {
        $(#[$meta])*
        pub fn $fn_name(params: &str) -> Result<$ty, String> {
            let mut w = <$ty>::default();
            for (k, v) in parse_params(params)? {
                match k {
                    "military_position" => w.military_position = parse_field(k, v)?,
                    "military_endgame_urgency" => w.military_endgame_urgency = parse_field(k, v)?,
                    "science_distinct_symbol" => w.science_distinct_symbol = parse_field(k, v)?,
                    "science_near_supremacy" => w.science_near_supremacy = parse_field(k, v)?,
                    "science_pair_setup" => w.science_pair_setup = parse_field(k, v)?,
                    "vp_projection" => w.vp_projection = parse_field(k, v)?,
                    "coins_div3" => w.coins_div3 = parse_field(k, v)?,
                    "coin_safety_floor" => w.coin_safety_floor = parse_field(k, v)?,
                    "coin_safety_penalty" => w.coin_safety_penalty = parse_field(k, v)?,
                    "resource_vulnerability" => w.resource_vulnerability = parse_field(k, v)?,
                    "deny_chain_gift" => w.deny_chain_gift = parse_field(k, v)?,
                    "wonder_potential" => w.wonder_potential = parse_field(k, v)?,
                    "instant_result" => w.instant_result = parse_field(k, v)?,
                    other => return Err(format!("unknown eval-weight key \"{other}\"")),
                }
            }
            Ok(w)
        }
    };
}

eval_weights_parser!(
    /// Parse a `greedy:...` parameter list into a [`GreedyWeights`].
    parse_greedy_weights,
    GreedyWeights
);
eval_weights_parser!(
    /// Parse a `greedy-ev:...` parameter list into a [`GreedyEvWeights`].
    parse_greedy_ev_weights,
    GreedyEvWeights
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_name_still_works_exactly_as_before() {
        let agent = make_agent_from_spec("random", 1).unwrap();
        assert_eq!(agent.spec().name, "random");
        let agent = make_agent_from_spec("greedy", 1).unwrap();
        assert_eq!(agent.spec().name, "greedy");
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
    fn greedy_spec_sets_named_weights() {
        let agent = make_agent_from_spec("greedy:vp_projection=2.5", 1).unwrap();
        assert_eq!(agent.spec().name, "greedy");
        let w = parse_greedy_weights("vp_projection=2.5").unwrap();
        assert_eq!(w.vp_projection, 2.5);
    }

    #[test]
    fn greedy_ev_spec_sets_named_weights() {
        let agent = make_agent_from_spec("greedy-ev:instant_result=500", 1).unwrap();
        assert_eq!(agent.spec().name, "greedy-ev");
        let w = parse_greedy_ev_weights("instant_result=500").unwrap();
        assert_eq!(w.instant_result, 500.0);
    }

    #[test]
    fn phased_base_v1_is_the_configuration_the_crate_shipped_with() {
        assert_eq!(parse_phased_config("base=v1").unwrap(), PhasedConfig::v1());
        assert_eq!(parse_phased_config("base=v2").unwrap(), PhasedConfig::v2());
        assert_eq!(parse_phased_config("").unwrap(), PhasedConfig::default());
        // The round-three keys, and their "off" values reproducing v2's.
        let off = parse_phased_config(
            "rails=off,imminent=0,shieldprice=onesided,horizon=supply,lockin=0,band=2.0",
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
        assert!(parse_greedy_weights("not_a_real_key=1").is_err());
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
