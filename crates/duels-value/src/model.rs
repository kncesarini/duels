//! Hand-rolled inference for a small two-layer network, and the file format
//! its trained weights arrive in.
//!
//! # Shape
//!
//! `NUM_FEATURES -> hidden (ReLU) -> 4 (softmax)`. The four outputs are the
//! mutually exclusive ways a game ends **for the player being evaluated**:
//! a win by military supremacy, a win by scientific supremacy, a win on
//! civilian points (tiebreak included), or a loss of any kind. The scalar a
//! search backs up is the sum of the first three — see
//! [`Distribution::win`] — so a consumer that only wants a win probability is
//! no more complicated than one reading a single sigmoid. The decomposition is
//! about what the network is *trained to represent*, not about its interface.
//!
//! # Why the first layer is walked sparsely
//!
//! Most of a feature vector is zero: a city holds fifteen of seventy-three
//! cards, a structure shows a handful face up. Storing the first layer
//! **input-major** (`w1[i * hidden ..]` is input `i`'s row) lets the forward
//! pass skip every zero input and add a contiguous, vectorisable row per
//! non-zero one, which is what keeps a 500-wide input inside the few-µs budget
//! a leaf value has. `tests::sparse_and_dense_forward_agree` pins that the
//! shortcut is arithmetic-neutral.
//!
//! # The weights file
//!
//! Little-endian, written by `tools/train.py`:
//!
//! ```text
//! b"DVAL"  u32 version=1  u32 n_in  u32 n_hidden  u32 n_out  u32 desc_len
//! desc (UTF-8, desc_len bytes)
//! f32 w1[n_in * n_hidden]   input-major
//! f32 b1[n_hidden]
//! f32 w2[n_hidden * n_out]  hidden-major
//! f32 b2[n_out]
//! ```
//!
//! Input normalisation is **folded into `w1`/`b1` by the trainer**, so
//! inference reads the raw integer features directly. `desc` is a free-text
//! provenance line (training set, validation metrics) that
//! [`Model::describe`] surfaces so an `AgentSpec` can record exactly which
//! weights a result was measured with.

use std::sync::OnceLock;

use crate::features::NUM_FEATURES;

/// The number of outcome classes.
pub const NUM_OUTCOMES: usize = 4;

/// A predicted outcome distribution for the evaluated player. Sums to one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distribution {
    /// The evaluated player wins by military supremacy.
    pub military_win: f32,
    /// The evaluated player wins by scientific supremacy.
    pub science_win: f32,
    /// The evaluated player wins on points (civilian victory or tiebreak).
    pub civilian_win: f32,
    /// The evaluated player loses, by any of the three.
    pub loss: f32,
}

impl Distribution {
    /// `P(win)`, the scalar a search backs up: the three winning heads summed.
    #[inline]
    pub fn win(&self) -> f32 {
        self.military_win + self.science_win + self.civilian_win
    }

    /// The four heads in class order (`military_win`, `science_win`,
    /// `civilian_win`, `loss`) — the order the trainer's labels use.
    #[inline]
    pub fn as_array(&self) -> [f32; NUM_OUTCOMES] {
        [
            self.military_win,
            self.science_win,
            self.civilian_win,
            self.loss,
        ]
    }
}

/// A trained network, ready to run.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    n_in: usize,
    n_hidden: usize,
    /// Input-major: `w1[i * n_hidden + h]`.
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// Hidden-major: `w2[h * NUM_OUTCOMES + o]`.
    w2: Vec<f32>,
    b2: [f32; NUM_OUTCOMES],
    desc: String,
}

const MAGIC: &[u8; 4] = b"DVAL";

impl Model {
    /// Parse a weights file (see the module docs for the format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8], String> {
            let end = pos
                .checked_add(n)
                .filter(|&e| e <= bytes.len())
                .ok_or_else(|| format!("weights file truncated at byte {pos}"))?;
            let s = &bytes[*pos..end];
            *pos = end;
            Ok(s)
        };
        let u32_at = |pos: &mut usize| -> Result<u32, String> {
            let b = take(pos, 4)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        if take(&mut pos, 4)? != MAGIC {
            return Err("not a duels-value weights file (bad magic)".into());
        }
        let version = u32_at(&mut pos)?;
        if version != 1 {
            return Err(format!("unsupported weights version {version}"));
        }
        let n_in = u32_at(&mut pos)? as usize;
        let n_hidden = u32_at(&mut pos)? as usize;
        let n_out = u32_at(&mut pos)? as usize;
        let desc_len = u32_at(&mut pos)? as usize;
        if n_in != NUM_FEATURES {
            return Err(format!(
                "weights expect {n_in} inputs but this build has {NUM_FEATURES} features"
            ));
        }
        if n_out != NUM_OUTCOMES {
            return Err(format!(
                "weights have {n_out} outputs, expected {NUM_OUTCOMES}"
            ));
        }
        if n_hidden == 0 {
            return Err("weights have no hidden units".into());
        }
        let desc = String::from_utf8(take(&mut pos, desc_len)?.to_vec())
            .map_err(|e| format!("weights description is not UTF-8: {e}"))?;
        let floats = |pos: &mut usize, n: usize| -> Result<Vec<f32>, String> {
            let b = take(pos, n * 4)?;
            Ok(b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let w1 = floats(&mut pos, n_in * n_hidden)?;
        let b1 = floats(&mut pos, n_hidden)?;
        let w2 = floats(&mut pos, n_hidden * n_out)?;
        let b2v = floats(&mut pos, n_out)?;
        if pos != bytes.len() {
            return Err(format!(
                "weights file has {} trailing bytes",
                bytes.len() - pos
            ));
        }
        if w1
            .iter()
            .chain(&b1)
            .chain(&w2)
            .chain(&b2v)
            .any(|v| !v.is_finite())
        {
            return Err("weights contain a non-finite value".into());
        }
        let mut b2 = [0.0f32; NUM_OUTCOMES];
        b2.copy_from_slice(&b2v);
        Ok(Model {
            n_in,
            n_hidden,
            w1,
            b1,
            w2,
            b2,
            desc,
        })
    }

    /// Hidden width.
    pub fn hidden(&self) -> usize {
        self.n_hidden
    }

    /// The provenance line the trainer wrote into the file.
    pub fn describe(&self) -> &str {
        &self.desc
    }

    /// Run the network on one feature vector.
    ///
    /// Panics if `x.len() != NUM_FEATURES`.
    pub fn predict(&self, x: &[f32]) -> Distribution {
        assert_eq!(x.len(), self.n_in, "feature vector has the wrong width");
        let h = self.n_hidden;
        // Stack scratch sized for any plausible hidden width; falls back to
        // the heap only for a wider network than this crate ships.
        let mut stack = [0.0f32; 256];
        let mut heap: Vec<f32>;
        let hidden: &mut [f32] = if h <= stack.len() {
            &mut stack[..h]
        } else {
            heap = vec![0.0; h];
            &mut heap
        };
        hidden.copy_from_slice(&self.b1);
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w1[i * h..(i + 1) * h];
            for (acc, &w) in hidden.iter_mut().zip(row) {
                *acc += xi * w;
            }
        }
        let mut logits = self.b2;
        for (j, &a) in hidden.iter().enumerate() {
            if a <= 0.0 {
                continue; // ReLU
            }
            let row = &self.w2[j * NUM_OUTCOMES..(j + 1) * NUM_OUTCOMES];
            for (l, &w) in logits.iter_mut().zip(row) {
                *l += a * w;
            }
        }
        softmax(logits)
    }
}

fn softmax(mut z: [f32; NUM_OUTCOMES]) -> Distribution {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    Distribution {
        military_win: z[0] * inv,
        science_win: z[1] * inv,
        civilian_win: z[2] * inv,
        loss: z[3] * inv,
    }
}

/// The trained weights this crate ships, parsed once.
///
/// `weights/value.bin` is produced by `tools/train.py`; its provenance line is
/// [`Model::describe`]. Panics only if the embedded file is malformed, which a
/// test catches before it could reach a search.
pub fn embedded() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        Model::from_bytes(include_bytes!("../weights/value.bin"))
            .expect("the embedded duels-value weights file is well-formed")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic model with every weight non-zero, so the sparse
    /// shortcut has something to skip *and* something to miss if wrong.
    fn toy(n_hidden: usize) -> Model {
        let mut w1 = Vec::with_capacity(NUM_FEATURES * n_hidden);
        for i in 0..NUM_FEATURES * n_hidden {
            w1.push(((i * 7919) % 1000) as f32 / 1000.0 - 0.5);
        }
        let b1 = (0..n_hidden).map(|i| i as f32 * 0.01 - 0.1).collect();
        let w2 = (0..n_hidden * NUM_OUTCOMES)
            .map(|i| ((i * 104_729) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        Model {
            n_in: NUM_FEATURES,
            n_hidden,
            w1,
            b1,
            w2,
            b2: [0.1, -0.2, 0.3, -0.4],
            desc: "toy".into(),
        }
    }

    fn dense_forward(m: &Model, x: &[f32]) -> [f32; NUM_OUTCOMES] {
        let mut hidden = m.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            for (h, acc) in hidden.iter_mut().enumerate() {
                *acc += xi * m.w1[i * m.n_hidden + h];
            }
        }
        let mut logits = m.b2;
        for (j, &a) in hidden.iter().enumerate() {
            let a = a.max(0.0);
            for (o, l) in logits.iter_mut().enumerate() {
                *l += a * m.w2[j * NUM_OUTCOMES + o];
            }
        }
        let d = softmax(logits);
        d.as_array()
    }

    #[test]
    fn sparse_and_dense_forward_agree() {
        let m = toy(32);
        let mut x = [0.0f32; NUM_FEATURES];
        for i in (0..NUM_FEATURES).step_by(5) {
            x[i] = ((i % 11) as f32) - 3.0;
        }
        let sparse = m.predict(&x).as_array();
        let dense = dense_forward(&m, &x);
        for (a, b) in sparse.iter().zip(&dense) {
            assert!((a - b).abs() < 1e-5, "{sparse:?} vs {dense:?}");
        }
    }

    #[test]
    fn the_distribution_sums_to_one_and_is_a_probability() {
        let m = toy(64);
        for k in 0..20 {
            let mut x = [0.0f32; NUM_FEATURES];
            for (i, v) in x.iter_mut().enumerate() {
                if (i * 31 + k) % 7 == 0 {
                    *v = ((i + k) % 5) as f32;
                }
            }
            let d = m.predict(&x);
            let a = d.as_array();
            assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(a.iter().all(|&p| (0.0..=1.0).contains(&p)));
            assert!((0.0..=1.0 + 1e-6).contains(&d.win()));
        }
    }

    #[test]
    fn the_file_format_round_trips() {
        let m = toy(16);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        for v in [1u32, NUM_FEATURES as u32, 16, NUM_OUTCOMES as u32, 3] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(b"toy");
        for v in m.w1.iter().chain(&m.b1).chain(&m.w2).chain(&m.b2) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let back = Model::from_bytes(&bytes).expect("parses");
        assert_eq!(back, m);
        assert_eq!(back.describe(), "toy");

        // Truncation and trailing bytes are both refused.
        assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(Model::from_bytes(&longer).is_err());
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(Model::from_bytes(&bad_magic).is_err());
    }

    #[test]
    fn the_embedded_weights_parse_and_match_this_build() {
        let m = embedded();
        assert_eq!(m.n_in, NUM_FEATURES);
        assert!(m.hidden() > 0);
        assert!(!m.describe().is_empty(), "the weights carry no provenance");
    }
}
