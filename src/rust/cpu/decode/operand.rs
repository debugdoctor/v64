//! Shared operand model. `Prefixes` (decoded prefix state), `Modrm` (the
//! decoded ModRM fields) and `Operand` (what a dispatch acts on) are produced
//! and consumed by both the 32-bit (`modrm.rs`, `instructions*.rs`) and
//! long-mode (`interp64_core`) engines.

use crate::cpu::decode::op_size::OpSize;
use crate::cpu::decode::prefix;

#[derive(Copy, Clone, Default)]
pub struct Prefixes {
    pub opsize_16: bool,
    pub addrsize_32: bool,
    pub segment: u8, // 0 = none, else a segment register index (FS/GS)
    pub rex: u8,
    pub f3: bool, // REP/SSE/FSGSBASE prefix
    pub f2: bool, // SSE prefix
    pub p66: bool, // 0x66 prefix (SSE2 for 0F opcodes)
}

impl Prefixes {
    pub fn has_rex(&self) -> bool { self.rex != 0 }

    pub fn has_rex_w(&self) -> bool { self.rex & prefix::REX_W != 0 }

    pub fn operand_size(&self) -> OpSize {
        if self.rex & prefix::REX_W != 0 {
            OpSize::S64
        } else if self.opsize_16 {
            OpSize::S16
        } else {
            OpSize::S32
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Modrm {
    pub mod_bits: u8,
    pub reg: u8,
    pub rm: u8,
}

impl Modrm {
    /// Split a raw ModRM byte into its fields (the 32-bit decoder reads the
    /// byte itself; the long-mode decoder reads it from the instruction stream).
    #[inline]
    pub fn from_byte(byte: i32) -> Modrm {
        Modrm {
            mod_bits: (byte >> 6) as u8,
            reg: (byte >> 3 & 7) as u8,
            rm: (byte & 7) as u8,
        }
    }
}

#[derive(Copy, Clone)]
pub enum Operand {
    Reg(u8),
    Mem(u64),
}

impl Operand {
    /// Linear address of a memory operand. The 32-bit engines narrow it to
    /// `i32` for their paging; long mode uses it as-is.
    #[inline]
    pub fn addr(&self) -> u64 {
        match self {
            Operand::Mem(a) => *a,
            Operand::Reg(_) => unreachable!(),
        }
    }
}
