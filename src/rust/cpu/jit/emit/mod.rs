use crate::cpu::core::{
    tlb_data, FLAG_CARRY, FLAG_OVERFLOW, FLAG_SIGN, FLAG_ZERO, OPSIZE_16, OPSIZE_32, OPSIZE_8,
    TLB_GLOBAL, TLB_HAS_CODE, TLB_NO_USER, TLB_READONLY, TLB_VALID,
};
use crate::cpu::global_pointers;
use crate::memory;
use crate::cpu::jit::{Instruction, InstructionOperand, InstructionOperandDest, JitContext};
use crate::cpu::jit::modrm;
use crate::cpu::jit::modrm::ModrmByte;
use crate::opstats;
use crate::profiler;
use crate::cpu::decode::regs;
use crate::wasmgen::wasm_builder::{WasmBuilder, WasmLocal, WasmLocalI64};

pub mod control;
pub mod flags;
pub mod mem;
pub mod misc;
pub mod registers;
pub mod stack;

pub use control::*;
pub use flags::*;
pub use mem::*;
pub use misc::*;
pub use registers::*;
pub use stack::*;
