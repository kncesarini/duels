# 0004. CI-only gating for now; no LLM review-agent workflow yet

## Context

Automated PR gating can range from plain CI (formatting/linting/tests) to
LLM-based review agents that comment on or block PRs. An LLM review-agent
workflow was considered for this repo but explicitly declined for now — it
adds operational surface area (API keys/secrets in CI, prompt/response
review, cost) that isn't justified while the project is a single-maintainer
scaffold with no active outside contributors yet.

## Decision

PR gating on `main` is CI-only: `cargo fmt --check`, `cargo clippy -D
warnings`, and `cargo test --workspace`, aggregated into a single required
`gate` job (see `.github/workflows/ci.yml`). No Claude/LLM-based automated
review workflow is configured.

## Consequences

- Simple, auditable, free-of-API-key CI. Branch protection can require the
  single `gate` check by name without needing to reason about a second,
  non-deterministic gate.
- Review quality (design/architecture feedback, not just fmt/lint/test)
  depends entirely on human review via `CODEOWNERS` for now.
- This decision is revisitable: if/when outside contribution volume or
  review load justifies it, an LLM review-agent workflow can be added as a
  new CI job without restructuring the existing gate — it was declined for
  now, not permanently ruled out.
