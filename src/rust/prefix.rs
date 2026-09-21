pub const PREFIX_REPZ: u8 = 0b01000;
pub const PREFIX_REPNZ: u8 = 0b10000;
pub const PREFIX_MASK_REP: u8 = PREFIX_REPZ | PREFIX_REPNZ;

pub const PREFIX_MASK_OPSIZE: u8 = 0b100000;
pub const PREFIX_MASK_ADDRSIZE: u8 = 0b1000000;

pub const PREFIX_66: u8 = PREFIX_MASK_OPSIZE;
pub const PREFIX_67: u8 = PREFIX_MASK_ADDRSIZE;
pub const PREFIX_F2: u8 = PREFIX_REPNZ;
pub const PREFIX_F3: u8 = PREFIX_REPZ;

pub const SEG_PREFIX_ZERO: u8 = 7;

pub const PREFIX_MASK_SEGMENT: u8 = 0b111;

// REX prefix (0x40-0x4F), long mode only. Stored as the full opcode byte so that
// 0x40 (no bits set) is still distinguishable from "no REX prefix" (0).
pub const REX_W: u8 = 0b1000; // 64-bit operand size
pub const REX_R: u8 = 0b0100; // extension of the ModRM reg field
pub const REX_X: u8 = 0b0010; // extension of the SIB index field
pub const REX_B: u8 = 0b0001; // extension of the ModRM r/m, SIB base or opcode reg field
