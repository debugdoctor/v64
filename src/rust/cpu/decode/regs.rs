#![allow(dead_code)]

pub const ES: u32 = 0;
pub const CS: u32 = 1;
pub const SS: u32 = 2;
pub const DS: u32 = 3;
pub const FS: u32 = 4;
pub const GS: u32 = 5;

pub const EAX: u32 = 0;
pub const ECX: u32 = 1;
pub const EDX: u32 = 2;
pub const EBX: u32 = 3;
pub const ESP: u32 = 4;
pub const EBP: u32 = 5;
pub const ESI: u32 = 6;
pub const EDI: u32 = 7;

pub const AX: u32 = 0;
pub const CX: u32 = 1;
pub const DX: u32 = 2;
pub const BX: u32 = 3;
pub const SP: u32 = 4;
pub const BP: u32 = 5;
pub const SI: u32 = 6;
pub const DI: u32 = 7;

pub const AL: u32 = 0;
pub const CL: u32 = 1;
pub const DL: u32 = 2;
pub const BL: u32 = 3;
pub const AH: u32 = 4;
pub const CH: u32 = 5;
pub const DH: u32 = 6;
pub const BH: u32 = 7;

// x86-64 registers r8-r15. Indexed like the legacy registers and selected through
// the REX prefix (see src/rust/prefix.rs). Not all are referenced yet, as the
// 64-bit instruction tables are still being ported.
pub const R8: u32 = 8;
pub const R9: u32 = 9;
pub const R10: u32 = 10;
pub const R11: u32 = 11;
pub const R12: u32 = 12;
pub const R13: u32 = 13;
pub const R14: u32 = 14;
pub const R15: u32 = 15;

pub const CR0_EM: u32 = 1 << 2;
pub const CR0_TS: u32 = 1 << 3;

pub const CR4_TSD: u32 = 1 << 2;
