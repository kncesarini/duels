//! The forward pass: a hand-rolled `f32` multilayer perceptron with one
//! hidden layer and a four-way softmax head.
//!
//! # Why hand-rolled
//!
//! The inference budget is the whole design constraint. `duels_eval::evaluate`
//! against a cached `Root` costs about **0.42 µs** and one full `mcts-eval`
//! simulation about **18.8 µs** (`mcts-eval`'s `examples/leaf_bench.rs`), so a
//! leaf value has single-digit microseconds to work with before it stops being
//! free next to the playout it sits beside. The shipped `211 → 128 → 4`
//! network is `211 × 128 + 128 × 4 = 27,520` multiply-adds — a couple of
//! microseconds of plain scalar `f32` work, no allocation, no threading, and
//! no library.
//!
//! Bringing in an ML runtime (`tract`, `ort`, `candle`) to run 27,520 FLOPs
//! would add a dependency tree, a model-file format, and a startup cost, to
//! save about eighty lines of arithmetic. It would also break this crate's
//! one-dependency rule, which is what keeps it usable from every agent.
//!
//! # The weights are data, and the header is checked
//!
//! [`Net::from_bytes`] refuses a buffer whose magic, feature count, or length
//! disagrees with what this build expects. That check is the thing standing
//! between a [`crate::features`] layout change and a search silently scoring
//! positions with a weight matrix whose columns have all shifted by one.

use crate::features::NUM_FEATURES;

/// Magic bytes at the head of a weights file: `duels-value weights, v1`.
const MAGIC: [u8; 4] = *b"DVW1";

/// How many outcome classes the head predicts. See [`crate::Outcome`].
pub const NUM_OUTCOMES: usize = 4;

/// How a layer's dot products are accumulated — the *only* thing that
/// distinguishes two otherwise identical networks, and it is here because
/// floating-point addition is not associative.
///
/// # Why this is a choice and not an implementation detail
///
/// `examples/value_bench.rs` measured the original forward pass at **45% of a
/// full random playout**, which is far more than 27,520 multiply-adds should
/// cost. The reason is not the FLOP count, it is the **dependency chain**:
/// each hidden unit was a single serial `acc += w * x` reduction over 211
/// terms, and because `f32` addition is not associative LLVM may neither
/// reorder nor vectorise it. The loop therefore ran at the *latency* of one
/// `f32` add per element (about two cycles each) rather than at throughput —
/// roughly ten times off.
///
/// Splitting the accumulator into four independent partial sums breaks that
/// chain. It also changes the summation order, and so changes the network's
/// output in the last couple of `f32` digits. That is why the choice is
/// explicit and why [`Summation::Serial`] is kept: the Elo the crate docs
/// report was measured with the serial order, and an agent must be able to
/// reproduce it exactly rather than approximately.
///
/// The difference is tiny but it is not zero, so it is measured on both axes
/// rather than argued about: `tests/summation_equivalence.rs` bounds it
/// numerically (`|Δ| < 1e-5` over 1,000 sampled positions) and the crate docs
/// record the before/after arena run that bounds it in Elo.
///
/// # A third order: transpose the loop nest, not just the reduction
///
/// `examples/value_bench.rs` found that [`Summation::Unrolled4`] only
/// recovered `1.41x` of the ten-fold gap the module docs above describe, and
/// diagnosed why: `w1` is 108 KiB, which does not fit in L1, so the loop is
/// substantially **memory**-bound rather than purely latency-bound, and a
/// wider unroll of the same per-unit reduction would not address that.
///
/// [`Summation::TransposedAxpy`] addresses the other axis instead. Every
/// existing order visits `w1` **row by row** (one hidden unit, all 211
/// inputs) and reduces each row to a scalar with a dot product. This order
/// visits it **column by column**: for each of the 211 inputs in turn, scale
/// that input's whole row of `w1ᵀ` (128 contiguous weights, one per hidden
/// unit) and accumulate it into a 128-wide running output — an axpy
/// (`h += x_i · w1ᵀ[i]`) rather than 128 separate dot products. The 128 `h[j]`
/// accumulators are mutually independent, so there is no horizontal reduction
/// at all until every input has been folded in, and the compiler is free to
/// vectorise straight across the hidden dimension with no cross-lane
/// dependency to break.
///
/// This does not reduce how many bytes of `w1` are read — every element is
/// still touched exactly once either way — so it does not sidestep the
/// memory-bound diagnosis above by shrinking traffic. What it changes is the
/// *shape* of the reduction: no per-unit horizontal sum, and a stream of
/// elementwise fused multiply-adds that is about as friendly a target for
/// auto-vectorisation as this loop can be handed.
///
/// It is also, perhaps surprisingly, **not a reassociation at all**. For a
/// fixed hidden unit `j`, this order adds exactly the same terms in exactly
/// the same sequence — `b1[j]`, then `x[0]·w1[j][0]`, then `x[1]·w1[j][1]`,
/// … — as [`Summation::Serial`] does; only the *loop nesting* (which `j`s are
/// visited between successive `i`s) changed, not the order of additions any
/// one accumulator sees. `tests/summation_equivalence.rs` checks this
/// directly (bit-for-bit, in both debug and release builds), alongside the
/// numerical bound the other pair is held to.
///
/// **Measured, and promoted to the default.** `examples/value_bench.rs`
/// records `4.2x`-`4.5x` on the forward pass alone against
/// [`Summation::Serial`] on the machine it was benchmarked on (`2.6x`-`3.3x`
/// against [`Summation::Unrolled4`], the default this replaced) — smaller
/// than a since-superseded planning estimate of `5.8x` from a different,
/// throwaway benchmark, but a large, real recovery of the memory-bound half
/// of the cost the unroll did not reach. At production's
/// `Budget::TimeMs(1_000)`, `arena/results/experiments/` holds three
/// paired-seed, seat-swapped batches against the unrolled default (2,000
/// games total): `+40.0 [+5.8, +74.3]` (400 games, shared machine),
/// `+23.5 [-0.7, +47.6]` (800 games, shared machine) and, on a verified-quiet
/// machine, `+32.2 [+8.0, +56.4]` (800 games) — consistently positive, and
/// the mechanism gate (`civilian_share`, `science_share`, `military_share`)
/// passes on every batch, so the gain is not a distorted victory-kind mix.
/// `arena/results/experiments/axpy-vs-default-nodes32000` confirms no
/// behaviour change at a **fixed** node count (`+1.2 [-26.6, +28.9]` over 600
/// games), exactly as a change that only makes each simulation cheaper has
/// to show. See `crates/agents/mcts-value/src/lib.rs`'s crate docs for the
/// full write-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Summation {
    /// One serial `f32` accumulator per unit — the original order, and the
    /// one every Elo number predating the unroll was measured with.
    Serial,
    /// Four independent partial sums, combined pairwise at the end. Computes
    /// the same function as [`Summation::Serial`] to within `1e-5` and breaks
    /// the latency chain that made the forward pass cost half a playout. Was
    /// the default; superseded by [`Summation::TransposedAxpy`], which goes
    /// further at the same numerical honesty (in fact more: it is
    /// bit-identical to `Serial`, not merely close), and stays reachable
    /// rather than being deleted.
    Unrolled4,
    /// The hidden layer as a sequence of 211 axpy updates into a 128-wide
    /// accumulator, over a pre-transposed `w1`, rather than 128 dot products
    /// over the row-major one. **The default**, because it is measurably
    /// faster than [`Summation::Unrolled4`] *and* matches
    /// [`Summation::Serial`] bit for bit rather than merely to within
    /// tolerance — see this enum's docs for the measurements.
    #[default]
    TransposedAxpy,
}

impl Summation {
    /// A short stable name, for an agent's params string.
    pub const fn name(self) -> &'static str {
        match self {
            Summation::Serial => "serial",
            Summation::Unrolled4 => "unrolled4",
            Summation::TransposedAxpy => "axpy",
        }
    }
}

/// A one-hidden-layer perceptron: `softmax(W2 · relu(W1 · x + b1) + b2)`.
///
/// Holds its parameters as flat `Vec<f32>`s in row-major order, which is the
/// layout the forward pass walks contiguously.
#[derive(Debug, Clone, PartialEq)]
pub struct Net {
    hidden: usize,
    /// `hidden × NUM_FEATURES`, row-major.
    w1: Vec<f32>,
    /// `w1`, transposed to `NUM_FEATURES × hidden`, row-major — so row `i` is
    /// the `hidden` weights input feature `i` feeds into, contiguous. Only
    /// [`Net::forward_transposed_axpy`] reads this; it is derived once from
    /// `w1` in [`Net::from_bytes`] (a few thousand `f32` moves, not on any hot
    /// path) rather than recomputed per call, so a handle can be built once
    /// per search tree and used at any [`Summation`] without a per-call cost
    /// for the ones that do not need it.
    w1_t: Vec<f32>,
    /// `hidden`.
    b1: Vec<f32>,
    /// `NUM_OUTCOMES × hidden`, row-major.
    w2: Vec<f32>,
    /// `NUM_OUTCOMES`.
    b2: Vec<f32>,
    /// Which accumulation order [`Net::forward`] uses. Not part of the
    /// weights file — it is a property of *this* handle on them, so one
    /// parsed buffer can be read both ways for an A/B.
    summation: Summation,
}

/// Why a weights buffer was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightsError {
    /// The buffer does not start with this crate's magic bytes.
    BadMagic,
    /// The buffer is shorter than its own header claims, or shorter than a
    /// header.
    Truncated,
    /// The weights were trained against a different feature layout. Almost
    /// always means [`NUM_FEATURES`] moved without the model being retrained.
    FeatureCountMismatch {
        /// What the file says it was trained on.
        file: usize,
        /// What this build produces.
        build: usize,
    },
    /// The head is not the four-way one this crate defines.
    OutcomeCountMismatch {
        /// What the file says.
        file: usize,
        /// What this build expects.
        build: usize,
    },
    /// A hidden width of zero, or one large enough to be an accident.
    BadHiddenWidth(usize),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightsError::BadMagic => write!(f, "not a duels-value weights file (bad magic)"),
            WeightsError::Truncated => write!(f, "the weights buffer is truncated"),
            WeightsError::FeatureCountMismatch { file, build } => write!(
                f,
                "weights were trained on {file} features but this build produces {build}"
            ),
            WeightsError::OutcomeCountMismatch { file, build } => {
                write!(f, "weights have {file} outputs, expected {build}")
            }
            WeightsError::BadHiddenWidth(h) => write!(f, "implausible hidden width {h}"),
        }
    }
}

impl std::error::Error for WeightsError {}

/// The largest hidden width this crate will load, which is also the width of
/// the stack buffer [`Net::forward`] keeps the hidden layer in.
///
/// Making the loader's bound and the forward pass's buffer *the same*
/// constant is the point: a wider network is rejected at load time with a
/// clear error rather than loaded and then silently truncated to the first
/// this-many units. 128 is the shipped width, so there is ample headroom for a
/// retrain; going past it means raising this one number.
const MAX_HIDDEN: usize = 512;

impl Net {
    /// Parse a weights buffer produced by `tools/train_value.py`.
    ///
    /// Layout, all little-endian: the four magic bytes, then `u32`
    /// `num_features`, `u32` `hidden`, `u32` `num_outcomes`, then `f32`
    /// `w1` (`hidden × num_features`, row-major), `b1` (`hidden`), `w2`
    /// (`num_outcomes × hidden`, row-major), `b2` (`num_outcomes`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, WeightsError> {
        if bytes.len() < 16 {
            return Err(WeightsError::Truncated);
        }
        if bytes[..4] != MAGIC {
            return Err(WeightsError::BadMagic);
        }
        let u32_at = |off: usize| -> usize {
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
                as usize
        };
        let n_in = u32_at(4);
        let hidden = u32_at(8);
        let n_out = u32_at(12);
        if n_in != NUM_FEATURES {
            return Err(WeightsError::FeatureCountMismatch {
                file: n_in,
                build: NUM_FEATURES,
            });
        }
        if n_out != NUM_OUTCOMES {
            return Err(WeightsError::OutcomeCountMismatch {
                file: n_out,
                build: NUM_OUTCOMES,
            });
        }
        if hidden == 0 || hidden > MAX_HIDDEN {
            return Err(WeightsError::BadHiddenWidth(hidden));
        }
        let counts = [hidden * n_in, hidden, n_out * hidden, n_out];
        let total: usize = counts.iter().sum();
        if bytes.len() < 16 + total * 4 {
            return Err(WeightsError::Truncated);
        }
        let mut off = 16;
        let mut take = |n: usize| -> Vec<f32> {
            let v = (0..n)
                .map(|i| {
                    let b = off + i * 4;
                    f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
                })
                .collect();
            off += n * 4;
            v
        };
        let w1 = take(counts[0]);
        let b1 = take(counts[1]);
        let w2 = take(counts[2]);
        let b2 = take(counts[3]);
        let w1_t = transpose(&w1, hidden, n_in);
        Ok(Net {
            hidden,
            w1,
            w1_t,
            b1,
            w2,
            b2,
            summation: Summation::default(),
        })
    }

    /// The same weights, read with a different accumulation order.
    ///
    /// Consumes and returns the net so it reads as a builder at a call site
    /// that parses once and keeps the result, which is how a search uses it.
    #[must_use]
    pub fn with_summation(mut self, summation: Summation) -> Net {
        self.summation = summation;
        self
    }

    /// Which accumulation order [`Net::forward`] will use.
    #[inline]
    pub fn summation(&self) -> Summation {
        self.summation
    }

    /// The hidden layer's width.
    #[inline]
    pub fn hidden_width(&self) -> usize {
        self.hidden
    }

    /// How many parameters this network holds, which is what "small" means
    /// concretely and what the inference cost is proportional to.
    #[inline]
    pub fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// The four-way outcome distribution for a feature vector.
    ///
    /// Deterministic, allocation-free, and reads no clock — the only thing a
    /// search leaf is allowed to be.
    ///
    /// Dispatches on [`Net::summation`]. All three arms compute the same
    /// function: [`Summation::Unrolled4`] agrees with [`Summation::Serial`]
    /// to within `1e-5` (a genuine reassociation), while
    /// [`Summation::TransposedAxpy`] is expected to agree with it bit for bit
    /// (a reordered loop nest, not a reassociation) — see [`Summation`] for
    /// why, and `tests/summation_equivalence.rs` for where both claims are
    /// checked rather than assumed.
    pub fn forward(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_OUTCOMES] {
        match self.summation {
            Summation::Serial => self.forward_serial(x),
            Summation::Unrolled4 => self.forward_unrolled4(x),
            Summation::TransposedAxpy => self.forward_transposed_axpy(x),
        }
    }

    /// The original forward pass, kept **verbatim** as the reference order.
    ///
    /// Not `#[cfg(test)]`: it is reachable from an agent's configuration on
    /// purpose, so the Elo numbers taken before the unroll can be reproduced
    /// exactly rather than approximately. The first rule of agent development
    /// in `docs/conventions.md` is that the old behaviour stays available as an
    /// explicit, proven-identical option.
    pub fn forward_serial(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_OUTCOMES] {
        // Hidden layer. Walked as `hidden` contiguous rows of `NUM_FEATURES`,
        // which is why `w1` is stored row-major. `MAX_HIDDEN` bounds both this
        // buffer and what `from_bytes` will accept, so `hidden` always fits.
        let mut h = [0.0f32; MAX_HIDDEN];
        let hidden = self.hidden;
        for (j, hj) in h.iter_mut().enumerate().take(hidden) {
            let row = &self.w1[j * NUM_FEATURES..(j + 1) * NUM_FEATURES];
            let mut acc = self.b1[j];
            for (w, xi) in row.iter().zip(x.iter()) {
                acc += w * xi;
            }
            // ReLU.
            *hj = if acc > 0.0 { acc } else { 0.0 };
        }

        // Output layer, then a numerically stable softmax.
        let mut logits = [0.0f32; NUM_OUTCOMES];
        for (k, lk) in logits.iter_mut().enumerate() {
            let row = &self.w2[k * self.hidden..k * self.hidden + hidden];
            let mut acc = self.b2[k];
            for (w, hj) in row.iter().zip(h.iter().take(hidden)) {
                acc += w * hj;
            }
            *lk = acc;
        }
        softmax(logits)
    }

    /// The same arithmetic with the reduction split four ways, so the
    /// multiply-adds run at throughput instead of at add latency.
    ///
    /// Identical in structure to [`Net::forward_serial`] — same rows, same
    /// ReLU, same softmax — differing only in the order the products are
    /// summed, which the private `dot4` localises to one function.
    pub fn forward_unrolled4(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_OUTCOMES] {
        let mut h = [0.0f32; MAX_HIDDEN];
        let hidden = self.hidden;
        for (j, hj) in h.iter_mut().enumerate().take(hidden) {
            let row = &self.w1[j * NUM_FEATURES..(j + 1) * NUM_FEATURES];
            let acc = dot4(row, x, self.b1[j]);
            *hj = if acc > 0.0 { acc } else { 0.0 };
        }

        let mut logits = [0.0f32; NUM_OUTCOMES];
        for (k, lk) in logits.iter_mut().enumerate() {
            let row = &self.w2[k * hidden..(k + 1) * hidden];
            *lk = dot4(row, &h[..hidden], self.b2[k]);
        }
        softmax(logits)
    }

    /// The hidden layer as 211 axpy updates over a pre-transposed `w1`,
    /// instead of 128 dot products over the row-major one.
    ///
    /// Identical to [`Net::forward_serial`] in every respect except *how* the
    /// hidden layer's sums are accumulated — same rows of meaning, same bias,
    /// same ReLU, same output layer verbatim. See [`Summation`] for why this
    /// reorders the loop nest rather than the arithmetic, and is expected to
    /// match [`Net::forward_serial`] bit for bit rather than merely to within
    /// tolerance; `tests/summation_equivalence.rs` checks that directly.
    pub fn forward_transposed_axpy(&self, x: &[f32; NUM_FEATURES]) -> [f32; NUM_OUTCOMES] {
        let hidden = self.hidden;

        // `h[j] = b1[j] + sum_i x[i] * w1[j][i]`, but visited input-by-input:
        // for each `i`, scale that input's whole (contiguous) row of `w1ᵀ` —
        // the `hidden` weights it feeds into — and fold it into every `h[j]`
        // at once. Each `h[j]` still sees its terms in exactly `i = 0, 1, ...`
        // order, so this is the same sum as `forward_serial`'s, computed with
        // a different loop nesting.
        let mut h = [0.0f32; MAX_HIDDEN];
        h[..hidden].copy_from_slice(&self.b1[..hidden]);
        for (&xi, row) in x.iter().zip(self.w1_t.chunks_exact(hidden)) {
            for (hj, w) in h[..hidden].iter_mut().zip(row) {
                *hj += xi * w;
            }
        }
        for hj in h[..hidden].iter_mut() {
            // ReLU.
            *hj = if *hj > 0.0 { *hj } else { 0.0 };
        }

        // Output layer, verbatim from `forward_serial` — this variant only
        // changes how the hidden layer is computed.
        let mut logits = [0.0f32; NUM_OUTCOMES];
        for (k, lk) in logits.iter_mut().enumerate() {
            let row = &self.w2[k * hidden..k * hidden + hidden];
            let mut acc = self.b2[k];
            for (w, hj) in row.iter().zip(h.iter().take(hidden)) {
                acc += w * hj;
            }
            *lk = acc;
        }
        softmax(logits)
    }
}

/// `w1` (`hidden × n_in`, row-major) transposed to `n_in × hidden`,
/// row-major — row `i` becomes the `hidden` weights input feature `i` feeds
/// into, contiguous. [`Net::forward_transposed_axpy`]'s whole point is
/// reading `w1` in this order instead of the original one.
fn transpose(w1: &[f32], hidden: usize, n_in: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; hidden * n_in];
    for j in 0..hidden {
        let row = &w1[j * n_in..(j + 1) * n_in];
        for (i, &w) in row.iter().enumerate() {
            t[i * hidden + j] = w;
        }
    }
    t
}

/// `bias + a · b`, accumulated into four independent partial sums.
///
/// The four lanes are combined pairwise (`(a0 + a1) + (a2 + a3)`) rather than
/// left-to-right, and the ragged tail — `NUM_FEATURES` is 211, which is
/// `4 × 52 + 3` — is added last. Both choices are arbitrary but *fixed*: the
/// function has to be deterministic to the bit for a search's results to be
/// reproducible, which is a stronger requirement than being accurate.
///
/// Sums the shorter of the two slices, which is only ever reached with equal
/// lengths from [`Net::forward_unrolled4`].
#[inline]
fn dot4(a: &[f32], b: &[f32], bias: f32) -> f32 {
    let mut acc = [0.0f32; 4];
    // `as_chunks` yields `&[[f32; 4]]`, so the four indexings below are
    // statically in bounds and compile without checks.
    let (a4, a_tail) = a.as_chunks::<4>();
    let (b4, b_tail) = b.as_chunks::<4>();
    for (aw, bw) in a4.iter().zip(b4.iter()) {
        acc[0] += aw[0] * bw[0];
        acc[1] += aw[1] * bw[1];
        acc[2] += aw[2] * bw[2];
        acc[3] += aw[3] * bw[3];
    }
    let mut tail = 0.0f32;
    for (aw, bw) in a_tail.iter().zip(b_tail.iter()) {
        tail += aw * bw;
    }
    bias + ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// A numerically stable softmax over the four logits.
fn softmax(logits: [f32; NUM_OUTCOMES]) -> [f32; NUM_OUTCOMES] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0f32; NUM_OUTCOMES];
    let mut sum = 0.0f32;
    for (o, l) in out.iter_mut().zip(logits.iter()) {
        *o = (l - max).exp();
        sum += *o;
    }
    // `sum >= 1` always, since the largest logit contributes `exp(0)`, so
    // this cannot divide by zero.
    for o in out.iter_mut() {
        *o /= sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weights buffer with the given shape and every parameter set from a
    /// simple deterministic pattern, for exercising the parser and the shapes.
    fn synthetic(n_in: usize, hidden: usize, n_out: usize) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        for n in [n_in, hidden, n_out] {
            v.extend_from_slice(&(n as u32).to_le_bytes());
        }
        let total = hidden * n_in + hidden + n_out * hidden + n_out;
        for i in 0..total {
            let x = ((i % 17) as f32 - 8.0) / 64.0;
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    #[test]
    fn a_well_formed_buffer_round_trips_into_a_distribution() {
        let bytes = synthetic(NUM_FEATURES, 96, NUM_OUTCOMES);
        let net = Net::from_bytes(&bytes).expect("a well-formed buffer loads");
        assert_eq!(net.hidden_width(), 96);
        assert_eq!(net.parameters(), 96 * NUM_FEATURES + 96 + 4 * 96 + 4);
        let x = [0.25f32; NUM_FEATURES];
        let p = net.forward(&x);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "the softmax sums to {sum}");
        for v in p {
            assert!((0.0..=1.0).contains(&v), "{v} is not a probability");
        }
    }

    #[test]
    fn the_header_rejects_a_layout_mismatch() {
        assert_eq!(
            Net::from_bytes(b"nope").unwrap_err(),
            WeightsError::Truncated
        );
        let mut bad = synthetic(NUM_FEATURES, 8, NUM_OUTCOMES);
        bad[0] = b'X';
        assert_eq!(Net::from_bytes(&bad).unwrap_err(), WeightsError::BadMagic);
        assert_eq!(
            Net::from_bytes(&synthetic(NUM_FEATURES - 1, 8, NUM_OUTCOMES)).unwrap_err(),
            WeightsError::FeatureCountMismatch {
                file: NUM_FEATURES - 1,
                build: NUM_FEATURES
            }
        );
        assert_eq!(
            Net::from_bytes(&synthetic(NUM_FEATURES, 8, 3)).unwrap_err(),
            WeightsError::OutcomeCountMismatch { file: 3, build: 4 }
        );
        assert_eq!(
            Net::from_bytes(&synthetic(NUM_FEATURES, 0, NUM_OUTCOMES)).unwrap_err(),
            WeightsError::BadHiddenWidth(0)
        );
        // A header that promises more than the buffer holds.
        let short = synthetic(NUM_FEATURES, 8, NUM_OUTCOMES);
        assert_eq!(
            Net::from_bytes(&short[..short.len() - 4]).unwrap_err(),
            WeightsError::Truncated
        );
    }

    /// A network wider than [`MAX_HIDDEN`] is refused at load time rather
    /// than truncated by the forward pass's stack buffer. This is the pairing
    /// of the two that makes the truncation unreachable.
    #[test]
    fn a_network_wider_than_the_stack_buffer_is_refused_not_truncated() {
        let too_wide = MAX_HIDDEN + 1;
        assert_eq!(
            Net::from_bytes(&synthetic(NUM_FEATURES, too_wide, NUM_OUTCOMES)).unwrap_err(),
            WeightsError::BadHiddenWidth(too_wide)
        );
        // ...and the shipped weights are comfortably inside it.
        assert!(crate::default_net().hidden_width() <= MAX_HIDDEN);
    }

    /// A softmax is shift-invariant, which is what makes the stable form
    /// legitimate rather than merely convenient.
    #[test]
    fn the_softmax_is_shift_invariant_and_saturates_cleanly() {
        let a = softmax([1.0, 2.0, 3.0, 4.0]);
        let b = softmax([101.0, 102.0, 103.0, 104.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
        // An extreme logit must not produce a NaN.
        let c = softmax([1000.0, -1000.0, 0.0, 0.0]);
        assert!(c.iter().all(|v| v.is_finite()));
        assert!(c[0] > 0.999);
    }
}
