//! Shared address model. Both the 32-bit (`modrm.rs`) and long-mode
//! (`interp64_core::modrm_effective_address`) decoders build the base/index/
//! displacement operands and call `effective_address`, so the linear-address
//! computation lives in one place.

/// `base + index * 2^scale + disp`, wrapped to `addr_size` bits (16, 32 or 64),
/// then the segment base added. `seg_base` is zero outside the 32-bit paths
/// (long mode is flat except for FS/GS, which the caller passes as `seg_base`).
#[inline]
pub fn effective_address(
    base: u64,
    index: u64,
    scale: u8,
    disp: u64,
    seg_base: u64,
    addr_size: u32,
) -> u64 {
    let mut address = base
        .wrapping_add(index.wrapping_mul(1u64 << scale))
        .wrapping_add(disp);
    match addr_size {
        16 => address &= 0xFFFF,
        32 => address &= 0xFFFF_FFFF,
        _ => {},
    }
    address.wrapping_add(seg_base)
}
