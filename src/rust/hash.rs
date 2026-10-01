//! In-tree fast, non-cryptographic hasher for the emulator's internal maps.
//!
//! The keys are page numbers, RIPs, table indices and labels derived from guest
//! state, not attacker-supplied strings, so a cryptographic hash is unnecessary.
//! The design follows rustc-hash's FxHash: one add-and-multiply per word, a
//! rotating finalizer (multiplicative hashes concentrate entropy in the high
//! bits, while hashbrown indexes buckets with the low bits), and word-at-a-time
//! handling of byte slices instead of FNV's byte loop. Keep the standard-library
//! hasher (SipHash) only where collision resistance is a real concern.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

// FxHash's 64-bit multiplier, from "Computationally Easy, Spectrally Good
// Multipliers for Congruential Pseudorandom Number Generators".
const K: u64 = 0xf135_7aea_2e62_a9c5;

#[derive(Default)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = self.0.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // Move the high-entropy bits down into the range hashbrown indexes with.
        self.0.rotate_left(26)
    }

    // Mix eight bytes per step rather than one.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }

    // Cover the integer keys directly: the default impls would fall back to the
    // byte path. Narrower types are widened, matching FxHash.
    #[inline]
    fn write_u8(&mut self, n: u8) { self.add(n as u64); }
    #[inline]
    fn write_u16(&mut self, n: u16) { self.add(n as u64); }
    #[inline]
    fn write_u32(&mut self, n: u32) { self.add(n as u64); }
    #[inline]
    fn write_u64(&mut self, n: u64) { self.add(n); }
    #[inline]
    fn write_usize(&mut self, n: usize) { self.add(n as u64); }
    #[inline]
    fn write_i8(&mut self, n: i8) { self.add(n as u64); }
    #[inline]
    fn write_i16(&mut self, n: i16) { self.add(n as u64); }
    #[inline]
    fn write_i32(&mut self, n: i32) { self.add(n as u64); }
    #[inline]
    fn write_i64(&mut self, n: i64) { self.add(n as u64); }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub type FastSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;
