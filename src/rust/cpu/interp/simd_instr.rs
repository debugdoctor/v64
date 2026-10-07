//! Mode-agnostic SIMD compute, shared by the interpreters and JITs.
//!
//! Every function here is pure: it takes and returns `reg128` (or scalars) and
//! never touches memory, decoding state or the register file. The per-engine
//! dispatch keeps the operand/address model and calls into this layer, so the
//! 32-bit and 64-bit paths have a single source of truth for SSE/SSE2/SSSE3/
//! SSE4/SSE4.2 and AVX lane semantics.

use crate::cpu::core::reg128;
use crate::cpu::global_pointers::mxcsr;

// SSE2 packed integer, lane-wise (0F D0-0xFF and part of 0x60-0x7F).
pub unsafe fn int_apply(opcode: u8, mut dst: reg128, src: reg128) -> Option<reg128> {
    match opcode {
        // 8-bit lanes
        0xFC => for i in 0..16 { dst.u8[i] = dst.u8[i].wrapping_add(src.u8[i]); },
        0xF8 => for i in 0..16 { dst.u8[i] = dst.u8[i].wrapping_sub(src.u8[i]); },
        0xEC => for i in 0..16 { dst.i8[i] = dst.i8[i].saturating_add(src.i8[i]); },
        0xE8 => for i in 0..16 { dst.i8[i] = dst.i8[i].saturating_sub(src.i8[i]); },
        0xDC => for i in 0..16 { dst.u8[i] = dst.u8[i].saturating_add(src.u8[i]); },
        0xD8 => for i in 0..16 { dst.u8[i] = dst.u8[i].saturating_sub(src.u8[i]); },
        0x74 => for i in 0..16 { dst.u8[i] = if dst.u8[i] == src.u8[i] { 0xFF } else { 0 }; },
        0x64 => for i in 0..16 { dst.u8[i] = if dst.i8[i] > src.i8[i] { 0xFF } else { 0 }; },
        0xDA => for i in 0..16 { dst.u8[i] = dst.u8[i].min(src.u8[i]); },
        0xDE => for i in 0..16 { dst.u8[i] = dst.u8[i].max(src.u8[i]); },

        // 16-bit lanes
        0xFD => for i in 0..8 { dst.u16[i] = dst.u16[i].wrapping_add(src.u16[i]); },
        0xF9 => for i in 0..8 { dst.u16[i] = dst.u16[i].wrapping_sub(src.u16[i]); },
        0xED => for i in 0..8 { dst.i16[i] = dst.i16[i].saturating_add(src.i16[i]); },
        0xE9 => for i in 0..8 { dst.i16[i] = dst.i16[i].saturating_sub(src.i16[i]); },
        0xDD => for i in 0..8 { dst.u16[i] = dst.u16[i].saturating_add(src.u16[i]); },
        0xD9 => for i in 0..8 { dst.u16[i] = dst.u16[i].saturating_sub(src.u16[i]); },
        0x75 => for i in 0..8 { dst.u16[i] = if dst.u16[i] == src.u16[i] { 0xFFFF } else { 0 }; },
        0x65 => for i in 0..8 { dst.u16[i] = if dst.i16[i] > src.i16[i] { 0xFFFF } else { 0 }; },
        0xEA => for i in 0..8 { dst.i16[i] = dst.i16[i].min(src.i16[i]); },
        0xEE => for i in 0..8 { dst.i16[i] = dst.i16[i].max(src.i16[i]); },
        0xD5 => for i in 0..8 { dst.u16[i] = dst.u16[i].wrapping_mul(src.u16[i]); },
        0xE5 => for i in 0..8 { dst.u16[i] = ((dst.i16[i] as i32 * src.i16[i] as i32) >> 16) as u16; },
        0xE4 => for i in 0..8 { dst.u16[i] = ((dst.u16[i] as u32 * src.u16[i] as u32) >> 16) as u16; },
        0xF5 => for i in 0..4 {
            dst.i32[i] = (dst.i16[i * 2] as i32 * src.i16[i * 2] as i32)
                .wrapping_add(dst.i16[i * 2 + 1] as i32 * src.i16[i * 2 + 1] as i32);
        },

        // 32-bit lanes
        0xFE => for i in 0..4 { dst.u32[i] = dst.u32[i].wrapping_add(src.u32[i]); },
        0xFA => for i in 0..4 { dst.u32[i] = dst.u32[i].wrapping_sub(src.u32[i]); },
        0x76 => for i in 0..4 { dst.u32[i] = if dst.u32[i] == src.u32[i] { !0 } else { 0 }; },
        0x66 => for i in 0..4 { dst.u32[i] = if dst.i32[i] > src.i32[i] { !0 } else { 0 }; },

        // 64-bit lanes and bitwise
        0xD4 => for i in 0..2 { dst.u64[i] = dst.u64[i].wrapping_add(src.u64[i]); },
        0xFB => for i in 0..2 { dst.u64[i] = dst.u64[i].wrapping_sub(src.u64[i]); },
        0xDB => for i in 0..2 { dst.u64[i] &= src.u64[i]; },
        0xDF => for i in 0..2 { dst.u64[i] = !dst.u64[i] & src.u64[i]; },
        0xEB => for i in 0..2 { dst.u64[i] |= src.u64[i]; },
        0xEF => for i in 0..2 { dst.u64[i] ^= src.u64[i]; },

        // unpack low/high
        0x60 => { let mut r = dst; for i in 0..8 { r.u16[i] = dst.u8[i] as u16 | (src.u8[i] as u16) << 8; } dst = r; },
        0x68 => { let mut r = dst; for i in 0..8 { r.u16[i] = dst.u8[i + 8] as u16 | (src.u8[i + 8] as u16) << 8; } dst = r; },
        0x61 => { let mut r = dst; for i in 0..4 { r.u32[i] = dst.u16[i] as u32 | (src.u16[i] as u32) << 16; } dst = r; },
        0x69 => { let mut r = dst; for i in 0..4 { r.u32[i] = dst.u16[i + 4] as u32 | (src.u16[i + 4] as u32) << 16; } dst = r; },
        0x62 => { let mut r = dst; for i in 0..2 { r.u64[i] = dst.u32[i] as u64 | (src.u32[i] as u64) << 32; } dst = r; },
        0x6A => { let mut r = dst; for i in 0..2 { r.u64[i] = dst.u32[i + 2] as u64 | (src.u32[i + 2] as u64) << 32; } dst = r; },
        0x6C => { dst.u64[1] = src.u64[0]; },
        0x6D => { dst.u64[0] = dst.u64[1]; dst.u64[1] = src.u64[1]; },

        // pack with signed/unsigned saturation
        0x63 => { let mut r = dst; for i in 0..8 {
            r.i8[i] = dst.i16[i].clamp(-128, 127) as i8;
            r.i8[i + 8] = src.i16[i].clamp(-128, 127) as i8;
        } dst = r; },
        0x6B => { let mut r = dst; for i in 0..4 {
            r.i16[i] = dst.i32[i].clamp(-32768, 32767) as i16;
            r.i16[i + 4] = src.i32[i].clamp(-32768, 32767) as i16;
        } dst = r; },
        0x67 => { let mut r = dst; for i in 0..8 {
            r.u8[i] = dst.i16[i].clamp(0, 255) as u8;
            r.u8[i + 8] = src.i16[i].clamp(0, 255) as u8;
        } dst = r; },

        // shifts by the count in the other operand
        0xD1 => for i in 0..8 { dst.u16[i] = shift(dst.u16[i] as u64, src.u64[0], 16, 0) as u16; },
        0xD2 => for i in 0..4 { dst.u32[i] = shift(dst.u32[i] as u64, src.u64[0], 32, 0) as u32; },
        0xD3 => for i in 0..2 { dst.u64[i] = shift(dst.u64[i], src.u64[0], 64, 0); },
        0xE1 => for i in 0..8 { dst.u16[i] = shift(dst.i16[i] as i64 as u64, src.u64[0], 16, 1) as u16; },
        0xE2 => for i in 0..4 { dst.u32[i] = shift(dst.i32[i] as i64 as u64, src.u64[0], 32, 1) as u32; },
        0xF1 => for i in 0..8 { dst.u16[i] = shift(dst.u16[i] as u64, src.u64[0], 16, 2) as u16; },
        0xF2 => for i in 0..4 { dst.u32[i] = shift(dst.u32[i] as u64, src.u64[0], 32, 2) as u32; },
        0xF3 => for i in 0..2 { dst.u64[i] = shift(dst.u64[i], src.u64[0], 64, 2); },

        // PMULUDQ: unsigned multiplication of the even dwords.
        0xF4 => {
            let mut r = reg128 { u64: [0, 0] };
            r.u64[0] = (dst.u32[0] as u64) * (src.u32[0] as u64);
            r.u64[1] = (dst.u32[2] as u64) * (src.u32[2] as u64);
            dst = r;
        },

        // sum of absolute differences
        0xF6 => {
            let mut total = 0u64;
            for i in 0..8 {
                total += (dst.u8[i].abs_diff(src.u8[i]) as u64)
                    + (dst.u8[i + 8].abs_diff(src.u8[i + 8]) as u64);
            }
            dst.u64[0] = total;
            dst.u64[1] = 0;
        },

        _ => return None,
    }
    Some(dst)
}

// kind: 0 logical right, 1 arithmetic right, 2 left; the count is per lane.
pub unsafe fn shift(value: u64, count: u64, width: u32, kind: u8) -> u64 {
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    if count >= width as u64 {
        return match kind {
            1 => if value >> (width - 1) & 1 != 0 { mask } else { 0 },
            2 => 0,
            _ => 0,
        };
    }
    let count = count as u32;
    match kind {
        1 => {
            // Sign-extend the lane from bit `width - 1` first: a plain 64-bit
            // shift would shift in zeros.
            let align = 64 - width;
            (((value << align) as i64 >> align) >> count) as u64 & mask
        },
        2 => value << count & mask,
        _ => value >> count,
    }
}

// PSLL/PSRL/PSRA/PSRLDQ/PSLLDQ with an immediate count (0F 71/72/73).
// Apply to `dst`; None if the group is unsupported.
pub unsafe fn shift_imm_apply(opcode: u8, group: u8, count: u64, mut dst: reg128) -> Option<reg128> {
    let (width, kind) = match (opcode, group) {
        (0x71, 2) => (16, 0), // PSRLW
        (0x71, 4) => (16, 1), // PSRAW
        (0x71, 6) => (16, 2), // PSLLW
        (0x72, 2) => (32, 0), // PSRLD
        (0x72, 4) => (32, 1), // PSRAD
        (0x72, 6) => (32, 2), // PSLLD
        (0x73, 2) => (64, 0), // PSRLQ
        (0x73, 6) => (64, 2), // PSLLQ
        // PSRLDQ (/3) / PSLLDQ (/7) shift the whole 128-bit operand by imm8
        // bytes (>= 16 clears it). cf. https://www.felixcloutier.com/x86/psrldq
        (0x73, 3) | (0x73, 7) => {
            let bits = (count.min(16) * 8) as u32;
            let v = dst.u64[0] as u128 | (dst.u64[1] as u128) << 64;
            let r = if bits >= 128 {
                0
            }
            else if group == 3 {
                v >> bits
            }
            else {
                v << bits
            };
            dst.u64[0] = r as u64;
            dst.u64[1] = (r >> 64) as u64;
            return Some(dst);
        },
        _ => return None,
    };
    let lanes = 128 / width;
    for i in 0..lanes {
        let bit = (i * width) as u32;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let value = dst.u64[(bit / 64) as usize] >> (bit % 64) & mask;
        let result = shift(value, count, width, kind);
        let index = (bit / 64) as usize;
        dst.u64[index] = dst.u64[index] & !(mask << (bit % 64)) | (result << (bit % 64));
    }
    Some(dst)
}

// PSHUFD/PSHUFHW/PSHUFLW (0F 70); variant 0/1/2.
pub unsafe fn pshuf_apply(variant: u8, control: u8, src: reg128) -> reg128 {
    let mut result = src;
    if variant == 1 {
        for i in 0..4 {
            result.u16[i] = src.u16[((control >> (i * 2)) & 3) as usize];
        }
    }
    else if variant == 2 {
        for i in 0..4 {
            result.u16[i + 4] = src.u16[4 + ((control >> (i * 2)) & 3) as usize];
        }
    }
    else {
        for i in 0..4 {
            result.u32[i] = src.u32[((control >> (i * 2)) & 3) as usize];
        }
    }
    result
}
pub unsafe fn compare_f32(predicate: u8, a: f32, b: f32) -> bool {
    match predicate {
        0 => a == b,                   // EQ
        1 => a < b,                    // LT
        2 => a <= b,                   // LE
        3 => a.is_nan() || b.is_nan(), // UNORD
        4 => !(a == b),                // NEQ
        5 => !(a < b),                 // NLT
        6 => !(a <= b),                // NLE
        _ => !(a.is_nan() || b.is_nan()), // ORD
    }
}
pub unsafe fn compare_f64(predicate: u8, a: f64, b: f64) -> bool {
    match predicate {
        0 => a == b,
        1 => a < b,
        2 => a <= b,
        3 => a.is_nan() || b.is_nan(),
        4 => !(a == b),
        5 => !(a < b),
        6 => !(a <= b),
        _ => !(a.is_nan() || b.is_nan()),
    }
}
pub unsafe fn arith_f32(opcode: u8, a: f32, b: f32) -> f32 {
    match opcode {
        0x51 => a.sqrt(),
        0x58 => a + b,
        0x59 => a * b,
        0x5C => a - b,
        0x5D => a.min(b),
        0x5E => a / b,
        _ => a.max(b),
    }
}
pub unsafe fn arith_f64(opcode: u8, a: f64, b: f64) -> f64 {
    match opcode {
        0x51 => a.sqrt(),
        0x58 => a + b,
        0x59 => a * b,
        0x5C => a - b,
        0x5D => a.min(b),
        0x5E => a / b,
        _ => a.max(b),
    }
}

// ---- SSSE3 (66 0F 38 00-0B, 1C-1E) ----------------------------------------
pub unsafe fn ssse3_apply(opcode: u8, dst: reg128, src: reg128) -> Option<reg128> {
    let mut result = dst;
    match opcode {
        // PSHUFB
        0x00 => for i in 0..16 {
            result.u8[i] = if src.u8[i] & 0x80 != 0 { 0 } else { dst.u8[src.u8[i] as usize & 15] };
        },
        // PHADDW / PHADDD / PHADDSW
        0x01 => for i in 0..4 {
            result.u16[i] = dst.u16[i * 2].wrapping_add(dst.u16[i * 2 + 1]);
            result.u16[i + 4] = src.u16[i * 2].wrapping_add(src.u16[i * 2 + 1]);
        },
        0x02 => for i in 0..2 {
            result.u32[i] = dst.u32[i * 2].wrapping_add(dst.u32[i * 2 + 1]);
            result.u32[i + 2] = src.u32[i * 2].wrapping_add(src.u32[i * 2 + 1]);
        },
        0x03 => for i in 0..4 {
            result.i16[i] = dst.i16[i * 2].saturating_add(dst.i16[i * 2 + 1]);
            result.i16[i + 4] = src.i16[i * 2].saturating_add(src.i16[i * 2 + 1]);
        },
        // PMADDUBSW
        0x04 => for i in 0..8 {
            let low = dst.u8[i * 2] as i32 * src.i8[i * 2] as i32;
            let high = dst.u8[i * 2 + 1] as i32 * src.i8[i * 2 + 1] as i32;
            result.i16[i] = (low + high).clamp(-32768, 32767) as i16;
        },
        // PHSUBW / PHSUBD / PHSUBSW
        0x05 => for i in 0..4 {
            result.u16[i] = dst.u16[i * 2].wrapping_sub(dst.u16[i * 2 + 1]);
            result.u16[i + 4] = src.u16[i * 2].wrapping_sub(src.u16[i * 2 + 1]);
        },
        0x06 => for i in 0..2 {
            result.u32[i] = dst.u32[i * 2].wrapping_sub(dst.u32[i * 2 + 1]);
            result.u32[i + 2] = src.u32[i * 2].wrapping_sub(src.u32[i * 2 + 1]);
        },
        0x07 => for i in 0..4 {
            result.i16[i] = dst.i16[i * 2].saturating_sub(dst.i16[i * 2 + 1]);
            result.i16[i + 4] = src.i16[i * 2].saturating_sub(src.i16[i * 2 + 1]);
        },
        // PSIGNB / PSIGNW / PSIGND
        0x08 => for i in 0..16 {
            result.i8[i] = if src.i8[i] == 0 { 0 } else if src.i8[i] < 0 { dst.i8[i].wrapping_neg() } else { dst.i8[i] };
        },
        0x09 => for i in 0..8 {
            result.i16[i] = if src.i16[i] == 0 { 0 } else if src.i16[i] < 0 { dst.i16[i].wrapping_neg() } else { dst.i16[i] };
        },
        0x0A => for i in 0..4 {
            result.i32[i] = if src.i32[i] == 0 { 0 } else if src.i32[i] < 0 { dst.i32[i].wrapping_neg() } else { dst.i32[i] };
        },
        // PMULHRSW
        0x0B => for i in 0..8 {
            let product = dst.i16[i] as i32 * src.i16[i] as i32;
            result.i16[i] = ((product + 0x4000) >> 15).clamp(-32768, 32767) as i16;
        },
        // PABSB / PABSW / PABSD
        0x1C => for i in 0..16 { result.u8[i] = dst.i8[i].unsigned_abs(); },
        0x1D => for i in 0..8 { result.u16[i] = dst.i16[i].unsigned_abs(); },
        0x1E => for i in 0..4 { result.u32[i] = dst.i32[i].unsigned_abs(); },
        _ => return None,
    }
    Some(result)
}

// ---- SSE4.1 / SSE4.2 (66 0F 38) -------------------------------------------
pub unsafe fn sse4_38_apply(opcode: u8, dst: reg128, src: reg128) -> Option<reg128> {
    let mut result = dst;
    match opcode {
        // PCMPEQQ / PCMPGTQ
        0x29 => for i in 0..2 { result.u64[i] = if dst.u64[i] == src.u64[i] { !0 } else { 0 }; },
        0x37 => for i in 0..2 { result.u64[i] = if dst.i64[i] > src.i64[i] { !0 } else { 0 }; },

        // PMULDQ: the even i32 lanes, as i64 products
        0x28 => {
            result.u64[0] = (dst.i32[0] as i64 * src.i32[0] as i64) as u64;
            result.u64[1] = (dst.i32[2] as i64 * src.i32[2] as i64) as u64;
        },
        // PMULLD: the low 32 bits of each u32 product
        0x40 => for i in 0..4 { result.u32[i] = dst.u32[i].wrapping_mul(src.u32[i]); },

        // signed/unsigned min/max
        0x38 => for i in 0..16 { result.i8[i] = dst.i8[i].min(src.i8[i]); },
        0x39 => for i in 0..4 { result.i32[i] = dst.i32[i].min(src.i32[i]); },
        0x3A => for i in 0..8 { result.u16[i] = dst.u16[i].min(src.u16[i]); },
        0x3B => for i in 0..4 { result.u32[i] = dst.u32[i].min(src.u32[i]); },
        0x3C => for i in 0..16 { result.i8[i] = dst.i8[i].max(src.i8[i]); },
        0x3D => for i in 0..4 { result.i32[i] = dst.i32[i].max(src.i32[i]); },
        0x3E => for i in 0..8 { result.u16[i] = dst.u16[i].max(src.u16[i]); },
        0x3F => for i in 0..4 { result.u32[i] = dst.u32[i].max(src.u32[i]); },

        // PMOVSX* / PMOVZX*: the source is the low lanes of `src`
        0x20 => for i in 0..8 { result.i16[i] = src.i8[i] as i16; },
        0x21 => for i in 0..4 { result.i32[i] = src.i8[i] as i32; },
        0x22 => for i in 0..2 { result.i64[i] = src.i8[i] as i64; },
        0x23 => for i in 0..4 { result.i32[i] = src.i16[i] as i32; },
        0x24 => for i in 0..2 { result.i64[i] = src.i16[i] as i64; },
        0x25 => for i in 0..2 { result.i64[i] = src.i32[i] as i64; },
        0x30 => for i in 0..8 { result.u16[i] = src.u8[i] as u16; },
        0x31 => for i in 0..4 { result.u32[i] = src.u8[i] as u32; },
        0x32 => for i in 0..2 { result.u64[i] = src.u8[i] as u64; },
        0x33 => for i in 0..4 { result.u32[i] = src.u16[i] as u32; },
        0x34 => for i in 0..2 { result.u64[i] = src.u16[i] as u64; },
        0x35 => for i in 0..2 { result.u64[i] = src.u32[i] as u64; },

        // PACKUSDW: pack four i32 into eight u16 with unsigned saturation
        0x2B => {
            let mut r = result;
            for i in 0..4 { r.u16[i] = dst.i32[i].clamp(0, 0xFFFF) as u16; }
            for i in 0..4 { r.u16[i + 4] = src.i32[i].clamp(0, 0xFFFF) as u16; }
            result = r;
        },

        // PHMINPOSUW: minimum u16 and its index in the low two lanes
        0x41 => {
            let mut best = u16::MAX;
            let mut index = 0u16;
            for i in 0..8 {
                if src.u16[i] < best {
                    best = src.u16[i];
                    index = i as u16;
                }
            }
            let mut r = reg128 { u64: [0, 0] };
            r.u16[0] = best;
            r.u16[1] = index;
            result = r;
        },

        _ => return None,
    }
    Some(result)
}

// PBLENDVB / BLENDVPS / BLENDVPD use XMM0 as the implicit mask.
pub unsafe fn blendv_apply(dst: reg128, src: reg128, mask: reg128, element_bytes: usize) -> reg128 {
    let mut result = reg128 { u64: [0, 0] };
    for i in 0..16 {
        let sign_byte = i / element_bytes * element_bytes + element_bytes - 1;
        result.u8[i] = if mask.u8[sign_byte] & 0x80 != 0 { src.u8[i] } else { dst.u8[i] };
    }
    result
}

// PTEST: returns (and, and-not) of dst and src; the dispatcher sets ZF/CF.
pub unsafe fn ptest(dst: reg128, src: reg128) -> (u128, u128) {
    let a = dst.u64[0] as u128 | (dst.u64[1] as u128) << 64;
    let b = src.u64[0] as u128 | (src.u64[1] as u128) << 64;
    (a & b, !a & b)
}

// ROUNDPS/PD/SS/SD use the MXCSR rounding control.
pub unsafe fn round_apply(value: f64, imm: u8) -> f64 {
    let mode = if imm & 4 != 0 { imm & 3 } else { (unsafe { *mxcsr } >> 13 & 3) as u8 };
    let v = match mode {
        0 => {
            // round to nearest, ties to even
            let r = value.round();
            let d = r - value;
            if d == 0.5 || d == -0.5 { 2.0 * (value * 0.5).round() } else { r }
        },
        1 => value.floor(),
        2 => value.ceil(),
        _ => value.trunc(),
    };
    // Suppress precision/inexact exceptions (MXCSR bits are not modelled).
    v
}

// PCMPESTRI/PCMPESTRM and PCMPISTRI/PCMPISTRM. `a` is the first operand
// (xmm1/ModRM:reg), `b` the second (xmm2/m128); `la`/`lb` are the explicit
// lengths (negative = NUL-terminated with |len| as the cap).
#[derive(Copy, Clone)]
pub struct StrCmp {
    pub intres: u32,
    pub index: u32,
    pub mask: reg128,
    pub zf: bool,
    pub sf: bool,
    pub of: bool,
}
pub unsafe fn strcmp_apply(a: &[u8; 16], b: &[u8; 16], la: i32, lb: i32, imm: u8) -> StrCmp {
    let words = imm & 1 != 0;
    let signed = imm & 2 != 0;
    let agg = (imm >> 2) & 3;
    let neg = imm & 0x10 != 0;
    let masked = imm & 0x20 != 0;
    let msb = imm & 0x40 != 0;
    let unit = if words { 2 } else { 1 };
    let n = 16 / unit;

    let elem = |buf: &[u8; 16], i: usize| -> i32 {
        if words {
            (buf[i * 2] as u16 | (buf[i * 2 + 1] as u16) << 8) as i16 as i32
        }
        else {
            buf[i] as i8 as i32
        }
    };
    let is_null = |buf: &[u8; 16], i: usize| -> bool {
        if words { buf[i * 2] == 0 && buf[i * 2 + 1] == 0 } else { buf[i] == 0 }
    };
    // Explicit lengths (PCMPESTRI) are `len`; implicit ones (PCMPISTRI) are
    // negative: the string ends at the first null (capped by |len|).
    let valid = |buf: &[u8; 16], i: usize, len: i32| -> bool {
        if i >= n {
            return false;
        }
        if len >= 0 {
            i < len as usize
        }
        else {
            let cap = ((-(len as i64)) as usize).min(n);
            i < cap && !(0..=i).any(|k| is_null(buf, k))
        }
    };
    let eq = |x: i32, y: i32| -> bool {
        if words {
            if signed { x as i16 == y as i16 } else { x as u16 == y as u16 }
        }
        else if signed { x as i8 == y as i8 } else { x as u8 == y as u8 }
    };
    // SDM Table 4-7: comparison result for an element pair, overridden when
    // either element is past the end of its string (invalid). `a` is xmm1 and
    // `b` is xmm2/m128, so the first validity bit is `a`'s and the second
    // `b`'s. `None` means the real comparison result is used.
    let force = |va: bool, vb: bool| -> Option<bool> {
        if va && vb {
            None
        }
        else if !va && !vb {
            Some(agg == 2 || agg == 3)
        }
        else if !va {
            Some(agg == 3)
        }
        else {
            Some(false)
        }
    };

    let mut intres = 0u32;
    match agg {
        // Equal any: bit m (over b) if any valid element of a equals b[m].
        0 => {
            for m in 0..n {
                for r in 0..n {
                    let bit = match force(valid(a, r, la), valid(b, m, lb)) {
                        Some(v) => v,
                        None => eq(elem(a, r), elem(b, m)),
                    };
                    if bit {
                        intres |= 1 << m;
                        break;
                    }
                }
            }
        },
        // Ranges: bit m (over b) if b[m] falls in an [a[2k], a[2k+1]] pair.
        1 => {
            for m in 0..n {
                let mut r = 0;
                while r + 1 < n {
                    let vb = valid(b, m, lb);
                    let lo = match force(valid(a, r, la), vb) {
                        Some(v) => v,
                        None => elem(b, m) >= elem(a, r),
                    };
                    let hi = match force(valid(a, r + 1, la), vb) {
                        Some(v) => v,
                        None => elem(b, m) <= elem(a, r + 1),
                    };
                    if lo && hi {
                        intres |= 1 << m;
                        break;
                    }
                    r += 2;
                }
            }
        },
        // Equal each: element-wise, bit i over both operands.
        2 => {
            for i in 0..n {
                let bit = match force(valid(a, i, la), valid(b, i, lb)) {
                    Some(v) => v,
                    None => eq(elem(a, i), elem(b, i)),
                };
                if bit {
                    intres |= 1 << i;
                }
            }
        },
        // Equal ordered: bit m (over b) if a matches b starting at m.
        _ => {
            for m in 0..n {
                let mut ok = true;
                for r in 0..n - m {
                    let bit = match force(valid(a, r, la), valid(b, m + r, lb)) {
                        Some(v) => v,
                        None => eq(elem(a, r), elem(b, m + r)),
                    };
                    if !bit {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    intres |= 1 << m;
                }
            }
        },
    }

    // SDM 4.1.4: masked negative inverts only the bits whose xmm2/m128 element
    // is valid; the others keep IntRes1. Plain negative inverts every bit.
    if neg {
        if masked {
            let mut r = 0u32;
            for i in 0..n {
                let bit = if valid(b, i, lb) { !(intres >> i) } else { intres >> i };
                r |= (bit & 1) << i;
            }
            intres = r;
        }
        else {
            intres = !intres & ((1u32 << n) - 1);
        }
    }
    intres &= (1u32 << n) - 1;

    let index = if intres == 0 {
        n as u32
    }
    else if msb {
        31 - intres.leading_zeros()
    }
    else {
        intres.trailing_zeros()
    };

    let mut mask = reg128 { u64: [0, 0] };
    if msb {
        for k in 0..n {
            if intres >> k & 1 != 0 {
                if words { mask.u16[k] = 0xFFFF; } else { mask.u8[k] = 0xFF; }
            }
        }
    }
    else {
        mask.u32[0] = intres;
    }

    let mut zf = false;
    let mut sf = false;
    for k in 0..n {
        if is_null(b, k) { zf = true; }
        if is_null(a, k) { sf = true; }
    }
    StrCmp { intres, index, mask, zf, sf, of: intres & 1 != 0 }
}

// BLENDPS/BLENDPD/PBLENDW with an immediate control.
pub unsafe fn blend_imm_apply(dst: reg128, src: reg128, imm: u8, lanes: usize, bits: u32) -> reg128 {
    let mut result = dst;
    for i in 0..lanes {
        if imm >> i & 1 != 0 {
            match bits {
                16 => result.u16[i] = src.u16[i],
                32 => result.u32[i] = src.u32[i],
                _ => result.u64[i] = src.u64[i],
            }
        }
    }
    result
}

// DPPS/DPPD: multiply selected lanes, horizontally add, broadcast.
pub unsafe fn dpps(dst: reg128, src: reg128, imm: u8) -> reg128 {
    let mut result = reg128 { u64: [0, 0] };
    let mut products = [0f32; 4];
    for i in 0..4 {
        products[i] = if imm >> (4 + i) & 1 != 0 { dst.f32[i] * src.f32[i] } else { 0.0 };
    }
    let sum = products[0] + products[1] + products[2] + products[3];
    for i in 0..4 {
        if imm >> i & 1 != 0 { result.f32[i] = sum; }
    }
    result
}

pub unsafe fn dppd(dst: reg128, src: reg128, imm: u8) -> reg128 {
    let mut result = reg128 { u64: [0, 0] };
    let p0 = if imm >> 4 & 1 != 0 { dst.f64[0] * src.f64[0] } else { 0.0 };
    let p1 = if imm >> 5 & 1 != 0 { dst.f64[1] * src.f64[1] } else { 0.0 };
    let sum = p0 + p1;
    if imm & 1 != 0 { result.f64[0] = sum; }
    if imm >> 1 & 1 != 0 { result.f64[1] = sum; }
    result
}

// MPSADBW: four 4-byte groups, each the sum of absolute differences of a
// 4-byte window of dst against a 4-byte window of src.
pub unsafe fn mpsadbw(dst: reg128, src: reg128, imm: u8) -> reg128 {
    let mut result = reg128 { u64: [0, 0] };
    for i in 0..4usize {
        let d_off = (if imm >> 2 & 1 != 0 { 4 } else { 0 }) + i * 4;
        let s_off = (if imm & 1 != 0 { 4 } else { 0 }) + i * 4;
        let mut total = 0u16;
        for k in 0..4usize {
            total += dst.u8[(d_off + k) & 15].abs_diff(src.u8[(s_off + k) & 15]) as u16;
        }
        result.u16[i] = total;
    }
    result
}

// INSERTPS: copy one source lane into the destination, then zero the masked
// destination lanes.
pub unsafe fn insertps(dst: reg128, src: reg128, imm: u8) -> reg128 {
    let mut result = dst;
    let src_index = (imm >> 6) & 3;
    let dst_index = (imm >> 4) & 3;
    result.u32[dst_index as usize] = src.u32[src_index as usize];
    for i in 0..4 {
        if imm >> i & 1 != 0 {
            result.u32[i] = 0;
        }
    }
    result
}

pub unsafe fn extractps(src: reg128, imm: u8) -> u32 {
    src.u32[imm as usize & 3]
}

// SSE4.2 CRC32: reflected CRC-32C (Castagnoli), byte at a time.
pub unsafe fn crc32(mut crc: u32, data: u64, bytes: usize) -> u32 {
    for i in 0..bytes {
        crc ^= (data >> (i * 8)) as u8 as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 };
        }
    }
    crc
}

// ---- AES-NI and PCLMULQDQ --------------------------------------------------

fn aes_sbox(x: u8) -> u8 {
    // Affine transform of the multiplicative inverse in GF(2^8).
    let inv = if x == 0 { 0 } else { gf8_pow(x, 254) };
    inv ^ inv.rotate_left(1) ^ inv.rotate_left(2) ^ inv.rotate_left(3) ^ inv.rotate_left(4) ^ 0x63
}

fn gf8_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 { p ^= a; }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 { a ^= 0x1B; }
        b >>= 1;
    }
    p
}

fn gf8_pow(a: u8, mut e: u32) -> u8 {
    let mut r = 1u8;
    let mut base = a;
    while e > 0 {
        if e & 1 != 0 { r = gf8_mul(r, base); }
        base = gf8_mul(base, base);
        e >>= 1;
    }
    r
}

fn aes_shift_rows(s: &[u8; 16]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = s[((c + row) % 4) * 4 + row];
        }
    }
    r
}

fn aes_mix_columns(s: &[u8; 16]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for c in 0..4 {
        let a = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
        r[c * 4] = gf8_mul(a[0], 2) ^ gf8_mul(a[1], 3) ^ a[2] ^ a[3];
        r[c * 4 + 1] = a[0] ^ gf8_mul(a[1], 2) ^ gf8_mul(a[2], 3) ^ a[3];
        r[c * 4 + 2] = a[0] ^ a[1] ^ gf8_mul(a[2], 2) ^ gf8_mul(a[3], 3);
        r[c * 4 + 3] = gf8_mul(a[0], 3) ^ a[1] ^ a[2] ^ gf8_mul(a[3], 2);
    }
    r
}

fn aes_sub_bytes(s: &[u8; 16]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for i in 0..16 { r[i] = aes_sbox(s[i]); }
    r
}
// AESENC/AESENCLAST; the final round omits MixColumns.
pub unsafe fn aesenc(state: reg128, key: reg128, last: bool) -> reg128 {
    let mut s = aes_shift_rows(&state.u8);
    s = aes_sub_bytes(&s);
    if !last { s = aes_mix_columns(&s); }
    let mut out = reg128 { u64: [0, 0] };
    out.u64[0] = u64::from_le_bytes(s[0..8].try_into().unwrap()) ^ key.u64[0];
    out.u64[1] = u64::from_le_bytes(s[8..16].try_into().unwrap()) ^ key.u64[1];
    out
}
pub unsafe fn aesdec(state: reg128, key: reg128, last: bool) -> reg128 {
    // Inverse cipher: InvShiftRows, InvSubBytes, InvMixColumns (except last).
    let inv_sbox = |y: u8| -> u8 {
        let z = y ^ 0x63;
        let t = z.rotate_left(1) ^ z.rotate_left(3) ^ z.rotate_left(6);
        if t == 0 { 0 } else { gf8_pow(t, 254) }
    };
    let s = state.u8;
    let mut shifted = [0u8; 16];
    for c in 0..4 {
        for row in 0..4 {
            shifted[c * 4 + row] = s[((c + 4 - row) % 4) * 4 + row];
        }
    }
    let mut subbed = [0u8; 16];
    for i in 0..16 { subbed[i] = inv_sbox(shifted[i]); }
    let mixed = if !last {
        let mut r = [0u8; 16];
        for c in 0..4 {
            let a = [subbed[c * 4], subbed[c * 4 + 1], subbed[c * 4 + 2], subbed[c * 4 + 3]];
            r[c * 4] = gf8_mul(a[0], 14) ^ gf8_mul(a[1], 11) ^ gf8_mul(a[2], 13) ^ gf8_mul(a[3], 9);
            r[c * 4 + 1] = gf8_mul(a[0], 9) ^ gf8_mul(a[1], 14) ^ gf8_mul(a[2], 11) ^ gf8_mul(a[3], 13);
            r[c * 4 + 2] = gf8_mul(a[0], 13) ^ gf8_mul(a[1], 9) ^ gf8_mul(a[2], 14) ^ gf8_mul(a[3], 11);
            r[c * 4 + 3] = gf8_mul(a[0], 11) ^ gf8_mul(a[1], 13) ^ gf8_mul(a[2], 9) ^ gf8_mul(a[3], 14);
        }
        r
    }
    else {
        subbed
    };
    let mut out = reg128 { u64: [0, 0] };
    out.u64[0] = u64::from_le_bytes(mixed[0..8].try_into().unwrap()) ^ key.u64[0];
    out.u64[1] = u64::from_le_bytes(mixed[8..16].try_into().unwrap()) ^ key.u64[1];
    out
}
pub unsafe fn aesimc(state: reg128) -> reg128 {
    let s = state.u8;
    let mut r = [0u8; 16];
    for c in 0..4 {
        let a = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
        r[c * 4] = gf8_mul(a[0], 14) ^ gf8_mul(a[1], 11) ^ gf8_mul(a[2], 13) ^ gf8_mul(a[3], 9);
        r[c * 4 + 1] = gf8_mul(a[0], 9) ^ gf8_mul(a[1], 14) ^ gf8_mul(a[2], 11) ^ gf8_mul(a[3], 13);
        r[c * 4 + 2] = gf8_mul(a[0], 13) ^ gf8_mul(a[1], 9) ^ gf8_mul(a[2], 14) ^ gf8_mul(a[3], 11);
        r[c * 4 + 3] = gf8_mul(a[0], 11) ^ gf8_mul(a[1], 13) ^ gf8_mul(a[2], 9) ^ gf8_mul(a[3], 14);
    }
    let mut out = reg128 { u64: [0, 0] };
    out.u64[0] = u64::from_le_bytes(r[0..8].try_into().unwrap());
    out.u64[1] = u64::from_le_bytes(r[8..16].try_into().unwrap());
    out
}
// RotWord rotates a little-endian dword right by eight bits.
pub unsafe fn aeskeygenassist(src: reg128, imm: u8) -> reg128 {
    let subword = |w: u32| -> u32 {
        (aes_sbox(w as u8) as u32)
            | ((aes_sbox((w >> 8) as u8) as u32) << 8)
            | ((aes_sbox((w >> 16) as u8) as u32) << 16)
            | ((aes_sbox((w >> 24) as u8) as u32) << 24)
    };
    let rotword = |w: u32| -> u32 { w.rotate_right(8) };
    let x1 = src.u32[1];
    let x3 = src.u32[3];
    let rcon = imm as u32;
    let mut out = reg128 { u64: [0, 0] };
    out.u32[0] = subword(x1);
    out.u32[1] = subword(rotword(x1)) ^ rcon;
    out.u32[2] = subword(x3);
    out.u32[3] = subword(rotword(x3)) ^ rcon;
    out
}
pub unsafe fn pclmulqdq(a: reg128, b: reg128, imm: u8) -> reg128 {
    let sel = |v: reg128, which: u8| -> u64 { if which & 1 != 0 { v.u64[1] } else { v.u64[0] } };
    let x = sel(a, imm & 1);
    let y = sel(b, (imm >> 4) & 1);
    // Carry-less multiply of two 64-bit values, 128-bit result.
    let mut result = 0u128;
    for i in 0..64 {
        if (y >> i) & 1 != 0 {
            result ^= (x as u128) << i;
        }
    }
    reg128 { u64: [result as u64, (result >> 64) as u64] }
}

// AVX2 variable shifts use the full unsigned count of each lane.
pub unsafe fn variable_shift32(src: reg128, counts: reg128, kind: u8) -> reg128 {
    let mut r = src;
    for i in 0..4 {
        r.u32[i] = shift(src.u32[i] as u64, counts.u32[i] as u64, 32, kind) as u32;
    }
    r
}

// VEX.W1 selects 64-bit lanes.
pub unsafe fn variable_shift64(src: reg128, counts: reg128, kind: u8) -> reg128 {
    let mut r = src;
    for i in 0..2 {
        r.u64[i] = shift(src.u64[i], counts.u64[i], 64, kind);
    }
    r
}

// VPBLENDD: per-dword blend with an immediate.
pub unsafe fn blend_d(dst: reg128, src: reg128, imm: u8) -> reg128 {
    let mut r = dst;
    for i in 0..4 {
        if imm >> i & 1 != 0 {
            r.u32[i] = src.u32[i];
        }
    }
    r
}
