//! A measurement harness for one narrow question: at the age-boundary
//! `Phase::ChooseFirstPlayer` decision, does it actually matter who goes
//! first?
//!
//! `phased` (`crates/agents/phased`) currently *always* chooses to go first
//! at this decision. That policy was never independently verified — it falls
//! out of `next_age_start` (`crates/agents/phased/src/terms.rs`) reading
//! identically for both candidate actions except for `menu.rs`'s `λ ×
//! (menu_me + menu_opp)` term, which is positive whenever both menus are
//! positive, so "choose first" wins by construction rather than by having
//! been shown to be correct.
//!
//! [`AgeStartPolicyAgent`] answers the question empirically without touching
//! `phased` (or any other agent) at all: it wraps any `Agent` and overrides
//! *only* its `Phase::ChooseFirstPlayer` decisions with a fixed policy,
//! passing every other decision straight through to the wrapped agent
//! unmodified. That turns "should an agent choose to go first here" from a
//! question about one agent's internal evaluation weights into an ordinary
//! `duels-arena` matchup: `AlwaysFirst` vs `AlwaysSecond` wrapping the same
//! inner agent, played as a normal paired-seed match (see
//! `examples/age_start_lab.rs`).
//!
//! # Why wrap instead of edit `phased`
//!
//! Editing `phased`'s evaluation to test a hypothesis about `phased`'s
//! evaluation would conflate "did the hypothesis change the answer" with
//! "did the code change introduce some other effect". Wrapping at the
//! `Agent` boundary isolates exactly the one decision point, for *any* agent
//! (see `examples/age_start_lab.rs`'s `mcts-uct` runs, which reuse the same
//! wrapper around a completely different agent to see whether a full
//! searcher agrees with whatever `phased` shows).

use duels_agents_api::{Agent, AgentSpec, Budget};
use duels_core::state::Phase;
use duels_core::{Action, Observation};

/// Which side of the `Phase::ChooseFirstPlayer` decision an
/// [`AgeStartPolicyAgent`] is forced to take, relative to *itself* (the seat
/// whose `Agent::choose` this decision was routed to — see
/// `Observation::current_player`, which is always the militarily weaker
/// player being asked, i.e. exactly the wrapped agent's own seat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgeStartChoice {
    /// Always choose to start the new age itself.
    AlwaysFirst,
    /// Always hand the new age to the opponent.
    AlwaysSecond,
    /// Different answers for the Age II and Age III boundaries specifically
    /// (`true` = itself starts that age, `false` = the opponent does).
    /// `Phase::ChooseFirstPlayer` only ever occurs transitioning into Age II
    /// or Age III (the pawn is recentred, or already fully resolved, before
    /// any other age boundary), so this is exhaustive: see
    /// `duels_core::engine::end_age`, which sets `Observation::age` to the
    /// *new* age before raising the phase, so `obs.age` at the decision
    /// point already reads 2 or 3.
    Split {
        age2_self_first: bool,
        age3_self_first: bool,
    },
}

impl AgeStartChoice {
    /// Whether the wrapped agent's own seat should start the age, given the
    /// `age` a `Phase::ChooseFirstPlayer` observation reports (2 or 3, per
    /// [`AgeStartChoice::Split`]'s doc comment).
    fn self_first(self, age: u8) -> bool {
        match self {
            AgeStartChoice::AlwaysFirst => true,
            AgeStartChoice::AlwaysSecond => false,
            AgeStartChoice::Split {
                age2_self_first,
                age3_self_first,
            } => {
                if age <= 2 {
                    age2_self_first
                } else {
                    age3_self_first
                }
            }
        }
    }

    /// A short, spec-string-style suffix identifying this policy, for
    /// `AgentSpec::params`.
    fn tag(self) -> String {
        match self {
            AgeStartChoice::AlwaysFirst => "age_start=first".to_string(),
            AgeStartChoice::AlwaysSecond => "age_start=second".to_string(),
            AgeStartChoice::Split {
                age2_self_first,
                age3_self_first,
            } => format!(
                "age_start=split(age2={},age3={})",
                if age2_self_first { "first" } else { "second" },
                if age3_self_first { "first" } else { "second" },
            ),
        }
    }
}

/// Wraps any `Agent`, forcing every `Phase::ChooseFirstPlayer` decision to a
/// fixed [`AgeStartChoice`] and delegating every other decision to the inner
/// agent completely unmodified. See the module docs for why this is the
/// right boundary at which to test the "always choose first" hypothesis.
pub struct AgeStartPolicyAgent {
    inner: Box<dyn Agent + Send>,
    choice: AgeStartChoice,
}

impl AgeStartPolicyAgent {
    pub fn new(inner: Box<dyn Agent + Send>, choice: AgeStartChoice) -> Self {
        Self { inner, choice }
    }
}

impl Agent for AgeStartPolicyAgent {
    fn spec(&self) -> AgentSpec {
        let inner = self.inner.spec();
        AgentSpec {
            name: inner.name,
            version: inner.version,
            params: if inner.params.is_empty() {
                self.choice.tag()
            } else {
                format!("{},{}", inner.params, self.choice.tag())
            },
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action {
        if obs.phase != Phase::ChooseFirstPlayer {
            return self.inner.choose(obs, legal, budget);
        }

        // `Observation::current_player` is always the seat being asked (the
        // militarily weaker player — see `duels_core::engine::end_age`),
        // which is exactly the seat this `choose` call was routed to. So
        // "self" always means `obs.current_player` here, regardless of
        // which physical seat the wrapped agent occupies in the match.
        let want_self_first = self.choice.self_first(obs.age);
        let wanted = if want_self_first {
            obs.current_player
        } else {
            obs.current_player.other()
        };
        let action = Action::ChooseFirstPlayer { player: wanted };
        debug_assert!(
            legal.contains(&action),
            "AgeStartPolicyAgent produced an action outside `legal`: {action:?} not in {legal:?}"
        );
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duels_agent_random::RandomAgent;
    use duels_core::engine;
    use duels_core::Player;
    use rand::{rngs::StdRng, SeedableRng};

    /// A deterministic, stateless inner agent: always the first legal
    /// action, purely as a function of `legal` — no RNG, no memory across
    /// calls. Used for the "wrapper only changes `ChooseFirstPlayer`" test
    /// below, where the point is to isolate *the wrapper's* behaviour: an
    /// inner agent that consumes its own RNG on every call (like
    /// `RandomAgent`) would legitimately diverge on every decision *after*
    /// the one the wrapper intercepts, simply because the wrapper's whole
    /// job is to answer `ChooseFirstPlayer` *without* calling into the inner
    /// agent — that's a fact about stateful RNG-consuming agents in general,
    /// not a bug in the wrapper. The "bit identical end to end" test below
    /// covers the RNG-consuming case the honest way instead: a seed where
    /// the wrapper never fires at all, so there is nothing for it to skip.
    struct FirstLegalAgent;

    impl Agent for FirstLegalAgent {
        fn spec(&self) -> AgentSpec {
            AgentSpec {
                name: "first-legal".to_string(),
                version: "0.0.1".to_string(),
                params: String::new(),
            }
        }

        fn choose(&mut self, _obs: &Observation, legal: &[Action], _budget: Budget) -> Action {
            legal.first().copied().expect("no legal actions available")
        }
    }

    /// Drive one seeded game to completion through `agent`, recording every
    /// `(phase, action)` pair it produced. Used to compare a plain agent
    /// against the same agent wrapped in an `AgeStartPolicyAgent` — see the
    /// tests below for how the two recordings are compared.
    fn play_and_record(mut agent: Box<dyn Agent>, setup_seed: u64) -> (Vec<Phase>, Vec<Action>) {
        let mut state = engine::new_game(setup_seed);
        let mut rng = StdRng::seed_from_u64(setup_seed ^ 0xE5CA_9E37_79B9_0001);
        let mut phases = Vec::new();
        let mut actions = Vec::new();
        loop {
            if state.is_over() {
                break;
            }
            let legal = engine::legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let obs = state.observation();
            let action = agent.choose(&obs, &legal, Budget::Nodes(1));
            phases.push(obs.phase);
            actions.push(action);
            engine::apply(&mut state, action, &mut rng).expect("agent returned a legal action");
        }
        (phases, actions)
    }

    #[test]
    fn wrapper_only_changes_choose_first_player_decisions() {
        // A handful of seeds so at least some of these games actually reach
        // an age boundary with the pawn off-centre (not every game does).
        //
        // Note on scope: once a `ChooseFirstPlayer` decision actually picks
        // a *different* player to start the age than the unwrapped agent
        // would have, the two runs are legitimately playing different games
        // from that point on (a different player takes the very next turn,
        // builds a different card into a different city, and everything
        // downstream of that is free to differ) — that is the wrapper
        // working as designed, not a leak. So this test only asserts
        // equality for every decision *up to and including* the first
        // `ChooseFirstPlayer` the game reaches, which is exactly the claim
        // "the wrapper is a no-op for every decision it doesn't intercept
        // and doesn't otherwise perturb the game": everything strictly
        // before that decision is provably identical.
        let mut checked_a_boundary = false;
        for seed in 0..40u64 {
            let plain = Box::new(FirstLegalAgent) as Box<dyn Agent>;
            let (plain_phases, plain_actions) = play_and_record(plain, seed);

            let wrapped = Box::new(AgeStartPolicyAgent::new(
                Box::new(FirstLegalAgent),
                AgeStartChoice::AlwaysFirst,
            )) as Box<dyn Agent>;
            let (wrapped_phases, wrapped_actions) = play_and_record(wrapped, seed);

            let boundary = plain_phases
                .iter()
                .position(|p| *p == Phase::ChooseFirstPlayer);
            let prefix_len = match boundary {
                Some(i) => {
                    checked_a_boundary = true;
                    i + 1
                }
                None => plain_phases.len(),
            };

            assert_eq!(
                plain_phases[..prefix_len],
                wrapped_phases[..prefix_len],
                "seed {seed}: wrapping changed the sequence of decision phases before/at \
                 the first age-boundary choice"
            );
            for i in 0..prefix_len {
                if plain_phases[i] == Phase::ChooseFirstPlayer {
                    // This is exactly the decision the wrapper is allowed
                    // (and, for `AlwaysFirst`, likely) to change.
                    continue;
                }
                assert_eq!(
                    plain_actions[i], wrapped_actions[i],
                    "seed {seed}, decision {i} (phase {:?}): wrapper changed a \
                     non-ChooseFirstPlayer decision",
                    plain_phases[i]
                );
            }
        }
        assert!(
            checked_a_boundary,
            "none of the 40 seeds tried reached a ChooseFirstPlayer decision; \
             widen the seed range so this test actually exercises the wrapper"
        );
    }

    #[test]
    fn a_game_with_no_age_boundary_choice_is_bit_identical_end_to_end() {
        // Find a seed whose game never actually reaches `ChooseFirstPlayer`
        // (the pawn ends up centred, or already resolved, at every age
        // boundary) and confirm the wrapped and unwrapped runs are
        // completely identical, not just equal outside one phase.
        for seed in 0..200u64 {
            let plain = Box::new(RandomAgent::new(seed)) as Box<dyn Agent>;
            let (phases, plain_actions) = play_and_record(plain, seed);
            if phases.contains(&Phase::ChooseFirstPlayer) {
                continue;
            }

            let wrapped = Box::new(AgeStartPolicyAgent::new(
                Box::new(RandomAgent::new(seed)),
                AgeStartChoice::AlwaysSecond,
            )) as Box<dyn Agent>;
            let (_, wrapped_actions) = play_and_record(wrapped, seed);

            assert_eq!(
                plain_actions, wrapped_actions,
                "seed {seed}: expected a bit-identical game with no age-boundary choice"
            );
            return;
        }
        panic!("no seed in 0..200 produced a game without a ChooseFirstPlayer decision");
    }

    #[test]
    fn always_first_always_picks_the_asking_players_own_seat() {
        // Directly exercise `choose` at a hand-built `ChooseFirstPlayer`
        // decision for each seat, independent of any particular game.
        for asking in [Player::One, Player::Two] {
            let mut agent = AgeStartPolicyAgent::new(
                Box::new(RandomAgent::new(1)),
                AgeStartChoice::AlwaysFirst,
            );
            let obs = make_choose_first_player_observation(asking, 2);
            let legal = [
                Action::ChooseFirstPlayer {
                    player: Player::One,
                },
                Action::ChooseFirstPlayer {
                    player: Player::Two,
                },
            ];
            let action = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert_eq!(action, Action::ChooseFirstPlayer { player: asking });
        }
    }

    #[test]
    fn always_second_always_picks_the_other_seat() {
        for asking in [Player::One, Player::Two] {
            let mut agent = AgeStartPolicyAgent::new(
                Box::new(RandomAgent::new(1)),
                AgeStartChoice::AlwaysSecond,
            );
            let obs = make_choose_first_player_observation(asking, 2);
            let legal = [
                Action::ChooseFirstPlayer {
                    player: Player::One,
                },
                Action::ChooseFirstPlayer {
                    player: Player::Two,
                },
            ];
            let action = agent.choose(&obs, &legal, Budget::Nodes(1));
            assert_eq!(
                action,
                Action::ChooseFirstPlayer {
                    player: asking.other()
                }
            );
        }
    }

    #[test]
    fn split_policy_reads_the_age_the_decision_is_about() {
        let mut age2_first = AgeStartPolicyAgent::new(
            Box::new(RandomAgent::new(1)),
            AgeStartChoice::Split {
                age2_self_first: true,
                age3_self_first: false,
            },
        );
        let legal = [
            Action::ChooseFirstPlayer {
                player: Player::One,
            },
            Action::ChooseFirstPlayer {
                player: Player::Two,
            },
        ];

        let obs_age2 = make_choose_first_player_observation(Player::One, 2);
        assert_eq!(
            age2_first.choose(&obs_age2, &legal, Budget::Nodes(1)),
            Action::ChooseFirstPlayer {
                player: Player::One
            }
        );

        let obs_age3 = make_choose_first_player_observation(Player::One, 3);
        assert_eq!(
            age2_first.choose(&obs_age3, &legal, Budget::Nodes(1)),
            Action::ChooseFirstPlayer {
                player: Player::Two
            }
        );
    }

    /// Build a minimal, standalone `ChooseFirstPlayer` observation for
    /// `asking` at `age`, by driving a real game to that phase and then
    /// overwriting just `current_player`/`age`. There is no public
    /// `Observation` constructor (by design — see `duels-core`'s docs), so
    /// this reuses a real one from an actual game rather than constructing
    /// one field-by-field.
    fn make_choose_first_player_observation(asking: Player, age: u8) -> Observation {
        // Drive real games until one reaches `ChooseFirstPlayer`, then patch
        // the two fields this test cares about. Every other field is left as
        // whatever a real game produced, which is fine: `choose` only reads
        // `phase`, `age` and `current_player` at this decision point.
        for seed in 0..500u64 {
            let mut state = engine::new_game(seed);
            let mut rng = StdRng::seed_from_u64(seed);
            let mut agent = RandomAgent::new(seed);
            loop {
                if state.is_over() {
                    break;
                }
                let legal = engine::legal_actions(&state);
                if legal.is_empty() {
                    break;
                }
                if state.phase() == Phase::ChooseFirstPlayer {
                    let mut obs = state.observation();
                    obs.current_player = asking;
                    obs.age = age;
                    return obs;
                }
                let obs = state.observation();
                let action = agent.choose(&obs, &legal, Budget::Nodes(1));
                engine::apply(&mut state, action, &mut rng).expect("legal action");
            }
        }
        panic!("no seed in 0..500 reached a ChooseFirstPlayer decision");
    }
}
