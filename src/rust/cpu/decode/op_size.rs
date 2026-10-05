// Operand size for the shared instruction helpers. Defined here rather than in
// interp64 so that mode-independent code (misc_instr, arith, ...) can use it
// without depending on a specific execution engine.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OpSize {
    S8,
    S16,
    S32,
    S64,
}

impl OpSize {
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            OpSize::S8 => 8,
            OpSize::S16 => 16,
            OpSize::S32 => 32,
            OpSize::S64 => 64,
        }
    }

    #[inline]
    pub fn mask(self) -> u64 {
        match self {
            OpSize::S8 => 0xFF,
            OpSize::S16 => 0xFFFF,
            OpSize::S32 => 0xFFFF_FFFF,
            OpSize::S64 => u64::MAX,
        }
    }

    #[inline]
    pub fn sign_bit(self) -> u64 {
        match self {
            OpSize::S8 => 0x80,
            OpSize::S16 => 0x8000,
            OpSize::S32 => 0x8000_0000,
            OpSize::S64 => 0x8000_0000_0000_0000,
        }
    }
}
