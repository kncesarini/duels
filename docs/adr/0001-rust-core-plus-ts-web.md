# 0001. Rust core engine + PyO3 bindings + TypeScript/React web client

## Context

We are building a from-scratch 7 Wonders Duel implementation whose end goals
include: a fast, correct rules engine; AI agents trained via reinforcement
learning; and a web UI humans can play against those agents (or each other)
in. These goals have different natural languages/ecosystems: performance and
correctness favor a systems language for the engine; the RL training loop is
overwhelmingly a Python ecosystem (numpy/torch/rllib-style tooling); and the
web client needs to run in a browser.

## Decision

- The rules engine (`duels-core`) is written in Rust: state/action/
  observation types, legal-move generation, effect resolution, and scoring.
- A PyO3 binding crate will expose `duels-core` to Python for RL training
  (self-play, vectorized environments) without reimplementing the rules.
- The web client is TypeScript/React, talking to a Rust game server
  (`duels-server`) over WebSocket rather than embedding rules logic in the
  browser.
- An `Agent` trait (`duels-agents-api`) is the stable contract between the
  engine and any bot player (hand-written heuristics or RL-trained), so
  agents can be swapped without touching the engine.

## Consequences

- One authoritative rules implementation (Rust) — no risk of the web client
  and training environment disagreeing about legality or scoring.
- Two language boundaries to maintain (Rust<->Python via PyO3, Rust<->
  TypeScript via a WebSocket protocol), each adding some integration
  overhead and a contract that must be versioned deliberately (see
  `docs/agent-contract.md`).
- Contributors need Rust toolchain familiarity for core engine work; this is
  acceptable given engine correctness/performance is the highest-leverage
  risk area.
- PyO3 bindings and the WebSocket protocol are not built in M0; this ADR
  commits to the shape but M0 only lays out the workspace and crate
  boundaries so later milestones can fill them in without restructuring.
