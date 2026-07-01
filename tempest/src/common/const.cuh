#ifndef CONST_H
#define CONST_H

#include <cstdint>

constexpr double DEFAULT_TIMESCALE_BOUND = -1;
constexpr double DEFAULT_NODE2VEC_P = 1.0;
constexpr double DEFAULT_NODE2VEC_Q = 1.0;

constexpr int EMPTY_NODE_VALUE = -1;
constexpr int64_t EMPTY_TIMESTAMP_VALUE = -1;
constexpr int64_t EMPTY_EDGE_ID = -1;
constexpr uint64_t EMPTY_GLOBAL_SEED = UINT64_MAX;

// Per-walk start-time cutoff sentinel: a walk whose cutoff equals this is
// unbounded (samples from the whole graph). A real cutoff t is an EXCLUSIVE
// upper bound — the walk may only traverse edges with timestamp < t ("walk
// this node as of time t"). -1 is reused for parity with the existing
// "no timestamp constraint" convention and because it is memset-friendly
// (0xFF bytes), so default-filling a per-walk cutoff buffer is a plain fill.
constexpr int64_t NO_WALK_CUTOFF = -1;

constexpr bool DEFAULT_SHUFFLE_WALK_ORDER = true;

// Temporal Node2Vec rejection-sampling retry cap.  Per-hop proposal is
// drawn from the static-exp distribution and accepted with prob β/β_max;
// expected retries ≤ β_max / E[β] (typically 1–2 for benign p, q).  On
// overflow the last proposal is accepted defensively, so degenerate
// (p ≪ 1 or q ≫ 1) configurations stay bounded.
//
// 64 covers the full node2vec literature sweep {0.25, 0.5, 1, 2, 4}
// without measurable bias: at the worst-case corner (p=0.25, q=4) the
// per-attempt acceptance is 1/16, so P(overflow at 64) ≤ (15/16)^64
// ≈ 2 × 10⁻².  Expected retries on benign p, q is still 1–2; the cap
// only matters for outliers, and raising it past what they actually
// need has no cost on the common path.
constexpr int NODE2VEC_MAX_RETRIES = 64;

#endif // CONST_H
