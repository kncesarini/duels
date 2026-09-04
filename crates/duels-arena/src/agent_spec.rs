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
//!   `exploration`/`c`, `rollout` (`uniform`/`biased`/`smart`),
//!   `chance_widen_c`, `chance_widen_alpha`, `max_rollout_plies`,
//!   `time_check_interval`, `root_determinizations`/`dets`, and `prior`
//!   (`none`, `expansion_order`, or `progressive_bias:<weight>`).
//! * `greedy` -- every [`duels_agent_greedy::EvalWeights`] field, by its own
//!   name (`military_position`, `military_endgame_urgency`,
//!   `science_distinct_symbol`, `science_near_supremacy`,
//!   `science_pair_setup`, `vp_projection`, `coins_div3`,
//!   `coin_safety_floor`, `coin_safety_penalty`, `resource_vulnerability`,
//!   `deny_chain_gift`, `wonder_potential`, `instant_result`).
//! * `greedy-ev` -- the same field names, against
//!   [`duels_agent_greedy_ev::EvalWeights`] (an identically-shaped struct in
//!   its own crate).
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
use duels_agent_mcts_uct::{Config as MctsConfig, MctsAgent, PriorMode, RolloutWeights};
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
             max_rollout_plies=100,time_check_interval=32,rollout=uniform,dets=4",
        )
        .unwrap();
        assert_eq!(cfg.exploration, 2.0);
        assert_eq!(cfg.chance_widen_c, 0.5);
        assert_eq!(cfg.chance_widen_alpha, 0.25);
        assert_eq!(cfg.max_rollout_plies, 100);
        assert_eq!(cfg.time_check_interval, 32);
        assert_eq!(cfg.rollout, RolloutWeights::UNIFORM);
        assert_eq!(cfg.root_determinizations, 4);
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
