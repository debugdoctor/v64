#![allow(non_snake_case)]

use crate::cpu::decode::context::CpuContext;
use crate::cpu::jit::modrm;
use crate::cpu::decode::prefix::{PREFIX_66, PREFIX_67, PREFIX_F2, PREFIX_F3, PREFIX_MASK_SEGMENT};
use crate::cpu::decode::regs::{CS, DS, ES, FS, GS, SS};

#[derive(PartialEq, Eq)]
pub enum AnalysisType {
    Normal,
    BlockBoundary,
    Jump {
        offset: i32,
        is_32: bool,
        condition: Option<u8>,
    },
    STI,
    // An instruction the 32-bit JIT cannot generate (e.g. VEX/AVX): the whole
    // page is left to the interpreter.
    Unsupported,
}

pub struct Analysis {
    pub no_next_instruction: bool,
    pub absolute_jump: bool,
    pub ty: AnalysisType,
}

pub fn analyze_step(mut cpu: &mut CpuContext) -> Analysis {
    let mut analysis = Analysis {
        no_next_instruction: false,
        absolute_jump: false,
        ty: AnalysisType::Normal,
    };
    cpu.prefixes = 0;
    let opcode = cpu.read_imm8() as u32 | (cpu.osize_32() as u32) << 8;
    crate::cpu::jit::analyzer::analyzer(opcode, &mut cpu, &mut analysis);
    analysis
}

pub fn analyze_step_handle_prefix(cpu: &mut CpuContext, analysis: &mut Analysis) {
    crate::cpu::jit::analyzer::analyzer(
        cpu.read_imm8() as u32 | (cpu.osize_32() as u32) << 8,
        cpu,
        analysis,
    )
}
pub fn analyze_step_handle_segment_prefix(
    segment: u32,
    cpu: &mut CpuContext,
    analysis: &mut Analysis,
) {
    dbg_assert!(segment <= 5);
    cpu.prefixes = cpu.prefixes & !PREFIX_MASK_SEGMENT | (segment as u8 + 1);
    analyze_step_handle_prefix(cpu, analysis)
}

pub fn instr16_0F_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    crate::cpu::jit::analyzer0f::analyzer(cpu.read_imm8() as u32, cpu, analysis)
}
pub fn instr32_0F_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    crate::cpu::jit::analyzer0f::analyzer(cpu.read_imm8() as u32 | 0x100, cpu, analysis)
}
pub fn instr_26_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(ES, cpu, analysis)
}
pub fn instr_2E_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(CS, cpu, analysis)
}
pub fn instr_36_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(SS, cpu, analysis)
}
pub fn instr_3E_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(DS, cpu, analysis)
}
pub fn instr_64_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(FS, cpu, analysis)
}
pub fn instr_65_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    analyze_step_handle_segment_prefix(GS, cpu, analysis)
}
pub fn instr_66_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_66;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_67_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_67;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F0_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    // lock: Ignored
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F2_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_F2;
    analyze_step_handle_prefix(cpu, analysis)
}
pub fn instr_F3_analyze(cpu: &mut CpuContext, analysis: &mut Analysis) {
    cpu.prefixes |= PREFIX_F3;
    analyze_step_handle_prefix(cpu, analysis)
}

pub fn modrm_analyze(ctx: &mut CpuContext, modrm_byte: u8) { modrm::skip(ctx, modrm_byte); }

// Skip a VEX-encoded instruction (C4/C5 with a second byte of bits[7:6] == 11)
// so the analyzer computes the right length.
pub fn vex_analyze(cpu: &mut CpuContext, first: u8, byte2: u8) {
    let map = if first == 0xC5 {
        1u8
    }
    else {
        cpu.read_imm8();
        byte2 & 0x1F
    };
    let opcode = cpu.read_imm8();
    let modrm_byte = cpu.read_imm8();
    modrm::skip(cpu, modrm_byte);
    let has_imm = match map {
        3 => true,
        1 => matches!(opcode, 0x70 | 0x71 | 0x72 | 0x73 | 0xC2 | 0xC6),
        _ => false,
    };
    if has_imm {
        cpu.read_imm8();
    }
}
