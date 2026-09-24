//! Hand-written x86-64 long mode interpreter, independent of `gen/interpreter*.rs`.
//! Operand size: 32, 64 with REX.W, 16 with 0x66. Address size: 64, 32 with 0x67.
//! Virtual addresses are 64-bit; physical addresses stay below 4 GiB (wasm32).
//! http://www.sandpile.org/x86/opra.htm

#![allow(dead_code)]

use crate::cpu::cpu::{
    io_port_read8, io_port_write8, read_reg64, read_tsc, test_privileges_for_io, translate_address_64,
    write_reg64, CR4_TSD, CS, FLAG_CARRY, FLAG_INTERRUPT, SS,
};
use crate::cpu::global_pointers::*;
use crate::cpu::memory;
use crate::paging::OrPageFault;
use crate::prefix;

// General-purpose register indices
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R11: u8 = 11;

// long mode MSRs
const MSR_EFER: i32 = 0xC0000080u32 as i32;
const MSR_STAR: i32 = 0xC0000081u32 as i32;
const MSR_LSTAR: i32 = 0xC0000082u32 as i32;
const MSR_SFMASK: i32 = 0xC0000084u32 as i32;
const MSR_FS_BASE: i32 = 0xC0000100u32 as i32;
const MSR_GS_BASE: i32 = 0xC0000101u32 as i32;
const MSR_KERNEL_GS_BASE: i32 = 0xC0000102u32 as i32;

const EFER_SCE: u64 = 1;
const EFER_LME: u64 = 1 << 8;
const EFER_LMA: u64 = 1 << 10;

// RFLAGS bits
const FLAG_CF: u32 = 1;
const FLAG_PF: u32 = 1 << 2;
const FLAG_AF: u32 = 1 << 4;
const FLAG_ZF: u32 = 1 << 6;
const FLAG_SF: u32 = 1 << 7;
const FLAG_OF: u32 = 1 << 11;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OpSize {
    S8,
    S16,
    S32,
    S64,
}

impl OpSize {
    fn bits(self) -> u32 {
        match self {
            OpSize::S8 => 8,
            OpSize::S16 => 16,
            OpSize::S32 => 32,
            OpSize::S64 => 64,
        }
    }

    fn mask(self) -> u64 {
        match self {
            OpSize::S8 => 0xFF,
            OpSize::S16 => 0xFFFF,
            OpSize::S32 => 0xFFFF_FFFF,
            OpSize::S64 => u64::MAX,
        }
    }

    fn sign_bit(self) -> u64 {
        match self {
            OpSize::S8 => 0x80,
            OpSize::S16 => 0x8000,
            OpSize::S32 => 0x8000_0000,
            OpSize::S64 => 0x8000_0000_0000_0000,
        }
    }
}

#[derive(Copy, Clone, Default)]
struct Prefixes {
    opsize_16: bool,
    addrsize_32: bool,
    segment: u8, // 0 = none, else a segment register index (FS/GS)
    rex: u8,
}

impl Prefixes {
    fn has_rex(&self) -> bool { self.rex != 0 }

    fn operand_size(&self) -> OpSize {
        if self.rex & prefix::REX_W != 0 {
            OpSize::S64
        }
        else if self.opsize_16 {
            OpSize::S16
        }
        else {
            OpSize::S32
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Modrm {
    mod_bits: u8,
    reg: u8,
    rm: u8,
}

unsafe fn fetch8() -> OrPageFault<u8> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read8(phys) as u8;
    *rip = (*rip).wrapping_add(1);
    Ok(value)
}

unsafe fn fetch16() -> OrPageFault<u16> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read16(phys) as u16;
    *rip = (*rip).wrapping_add(2);
    Ok(value)
}

unsafe fn fetch32() -> OrPageFault<u32> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read32s(phys) as u32;
    *rip = (*rip).wrapping_add(4);
    Ok(value)
}

unsafe fn fetch64() -> OrPageFault<u64> {
    let low = fetch32()? as u64;
    let high = fetch32()? as u64;
    Ok(low | (high << 32))
}

unsafe fn decode_prefixes() -> OrPageFault<(Prefixes, u8)> {
    let mut pfx = Prefixes::default();

    loop {
        let byte = fetch8()?;
        match byte {
            0x66 => pfx.opsize_16 = true,
            0x67 => pfx.addrsize_32 = true,
            0xF0 => {}, // LOCK: ignored
            0xF2 | 0xF3 => {}, // REP: handled by the instructions that need it
            0x26 | 0x2E | 0x36 | 0x3E => {}, // null prefix in long mode
            0x64 => pfx.segment = 4, // FS
            0x65 => pfx.segment = 5, // GS
            0x40..=0x4F => pfx.rex = byte,
            opcode => return Ok((pfx, opcode)),
        }
    }
}

unsafe fn decode_modrm(pfx: &Prefixes) -> OrPageFault<Modrm> {
    let byte = fetch8()?;
    Ok(Modrm {
        mod_bits: byte >> 6,
        // REX.R/REX.B extend the 3-bit field by 8
        reg: (byte >> 3 & 7) | (pfx.rex & prefix::REX_R) << 1,
        rm: (byte & 7) | (pfx.rex & prefix::REX_B) << 3,
    })
}

// mod != 3
unsafe fn modrm_effective_address(modrm: &Modrm, pfx: &Prefixes) -> OrPageFault<u64> {
    dbg_assert!(modrm.mod_bits != 3);

    let mut address: u64;
    let rm_low = modrm.rm & 7;

    if rm_low == 4 {
        // SIB byte
        let sib = fetch8()?;
        let scale = 1u64 << (sib >> 6);
        let index_low = sib >> 3 & 7;
        let base_low = sib & 7;

        // index == 4 with no REX.X means "no index"
        let index_value = if index_low == 4 && pfx.rex & prefix::REX_X == 0 {
            0
        }
        else {
            let index = (index_low | (pfx.rex & prefix::REX_X) << 2) as i32;
            read_reg64(index)
        };

        if base_low == 5 && modrm.mod_bits == 0 {
            // no base, disp32
            address = fetch32()? as i32 as i64 as u64;
        }
        else {
            let base = (base_low | (pfx.rex & prefix::REX_B) << 3) as i32;
            address = read_reg64(base);
        }
        address = address.wrapping_add(index_value.wrapping_mul(scale));
    }
    else if rm_low == 5 && modrm.mod_bits == 0 {
        // RIP-relative
        let displacement = fetch32()? as i32 as i64 as u64;
        address = (*rip).wrapping_add(displacement);
    }
    else {
        address = read_reg64(modrm.rm as i32);
    }

    match modrm.mod_bits {
        0 => {},
        1 => address = address.wrapping_add(fetch8()? as i8 as i64 as u64),
        2 => address = address.wrapping_add(fetch32()? as i32 as i64 as u64),
        _ => unreachable!(),
    }

    if pfx.segment == 4 {
        address = address.wrapping_add(*fs_base);
    }
    else if pfx.segment == 5 {
        address = address.wrapping_add(*gs_base);
    }

    Ok(address)
}

unsafe fn set_cpl_segments(cs: u16, new_cpl: u8) {
    // long mode: code/data bases are 0, SS = CS + 8
    *sreg.offset(CS as isize) = cs;
    *sreg.offset(SS as isize) = cs.wrapping_add(8);
    *segment_offsets.offset(CS as isize) = 0;
    *segment_offsets.offset(SS as isize) = 0;
    *segment_is_null.offset(CS as isize) = false;
    *segment_is_null.offset(SS as isize) = false;
    *cpl = new_cpl;
}

unsafe fn msr_read(index: i32) -> Option<u64> {
    match index {
        MSR_EFER => Some(*efer),
        MSR_STAR => Some(*star),
        MSR_LSTAR => Some(*lstar),
        MSR_SFMASK => Some(*sfmask),
        MSR_FS_BASE => Some(*fs_base),
        MSR_GS_BASE => Some(*gs_base),
        MSR_KERNEL_GS_BASE => Some(*kernel_gs_base),
        _ => None,
    }
}

unsafe fn msr_write(index: i32, value: u64) -> bool {
    match index {
        MSR_EFER => *efer = value | EFER_LMA,
        MSR_STAR => *star = value,
        MSR_LSTAR => *lstar = value,
        MSR_SFMASK => *sfmask = value,
        MSR_FS_BASE => *fs_base = value,
        MSR_GS_BASE => *gs_base = value,
        MSR_KERNEL_GS_BASE => *kernel_gs_base = value,
        _ => return false,
    }
    true
}

unsafe fn read_reg8(r: u8, has_rex: bool) -> u64 {
    if !has_rex && r >= 4 && r < 8 {
        // AH/CH/DH/BH
        read_reg64((r - 4) as i32) >> 8 & 0xFF
    }
    else {
        read_reg64(r as i32) & 0xFF
    }
}

unsafe fn write_reg8(r: u8, has_rex: bool, value: u64) {
    if !has_rex && r >= 4 && r < 8 {
        // AH/CH/DH/BH: high byte of r0-r3
        let old = read_reg64((r - 4) as i32);
        write_reg64((r - 4) as i32, old & !0xFF00 | (value & 0xFF) << 8);
    }
    else {
        let old = read_reg64(r as i32);
        write_reg64(r as i32, old & !0xFF | value & 0xFF);
    }
}

unsafe fn read_reg(r: u8, size: OpSize, has_rex: bool) -> u64 {
    match size {
        OpSize::S8 => read_reg8(r, has_rex),
        OpSize::S16 => read_reg64(r as i32) & 0xFFFF,
        OpSize::S32 => read_reg64(r as i32) & 0xFFFF_FFFF,
        OpSize::S64 => read_reg64(r as i32),
    }
}

unsafe fn write_reg(r: u8, size: OpSize, has_rex: bool, value: u64) {
    match size {
        OpSize::S8 => write_reg8(r, has_rex, value),
        // 32-bit writes zero-extend
        OpSize::S32 => write_reg64(r as i32, value & 0xFFFF_FFFF),
        // 16-bit writes preserve the upper bits
        OpSize::S16 => {
            let old = read_reg64(r as i32);
            write_reg64(r as i32, old & !0xFFFF | value & 0xFFFF);
        },
        OpSize::S64 => write_reg64(r as i32, value),
    }
}

unsafe fn mem_read(address: u64, size: OpSize) -> OrPageFault<u64> {
    let bytes = (size.bits() / 8) as usize;
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = translate_address_64(address + offset as u64, false, *cpl == 3)?;
    }
    if (address as usize & 0xFFF) + bytes <= 0x1000 {
        return Ok(match size {
            OpSize::S8 => memory::read8(physical[0]) as u8 as u64,
            OpSize::S16 => memory::read16(physical[0]) as u16 as u64,
            OpSize::S32 => memory::read32s(physical[0]) as u32 as u64,
            OpSize::S64 => memory::read64s(physical[0]) as u64,
        });
    }
    let mut value = 0;
    for offset in 0..bytes {
        value |= (memory::read8(physical[offset]) as u8 as u64) << (offset * 8);
    }
    Ok(value)
}

unsafe fn mem_write(address: u64, size: OpSize, value: u64) -> OrPageFault<()> {
    let bytes = (size.bits() / 8) as usize;
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    if (address as usize & 0xFFF) + bytes <= 0x1000 {
        match size {
            OpSize::S8 => memory::write8(physical[0], value as i32),
            OpSize::S16 => memory::write16(physical[0], value as i32),
            OpSize::S32 => memory::write32(physical[0], value as i32),
            OpSize::S64 => {
                memory::write32(physical[0], value as i32);
                memory::write32(physical[0] + 4, (value >> 32) as i32);
            },
        }
        return Ok(());
    }
    for offset in 0..bytes {
        memory::write8(physical[offset], (value >> (offset * 8)) as u8 as i32);
    }
    Ok(())
}

unsafe fn mem_probe_write(address: u64, size: OpSize) -> OrPageFault<()> {
    for offset in 0..size.bits() / 8 {
        translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    Ok(())
}

// The ModRM/SIB/displacement bytes are consumed here, so decoding has to happen
// before any immediate is fetched.
#[derive(Copy, Clone)]
enum Operand {
    Reg(u8),
    Mem(u64),
}

unsafe fn probe_operand_write(operand: Operand, size: OpSize) -> OrPageFault<()> {
    if let Operand::Mem(address) = operand {
        mem_probe_write(address, size)?;
    }
    Ok(())
}

unsafe fn decode_operand_after_modrm(
    pfx: &Prefixes,
    modrm: Modrm,
    trailing_bytes: u64,
) -> OrPageFault<(Modrm, Operand)> {
    let operand = if modrm.mod_bits == 3 {
        Operand::Reg(modrm.rm)
    }
    else {
        let mut address = modrm_effective_address(&modrm, pfx)?;
        if modrm.mod_bits == 0 && modrm.rm & 7 == 5 {
            address = address.wrapping_add(trailing_bytes);
        }
        Operand::Mem(address)
    };
    Ok((modrm, operand))
}

unsafe fn decode_operand_with_trailing(
    pfx: &Prefixes,
    trailing_bytes: u64,
) -> OrPageFault<(Modrm, Operand)> {
    let modrm = decode_modrm(pfx)?;
    decode_operand_after_modrm(pfx, modrm, trailing_bytes)
}

unsafe fn decode_operand(pfx: &Prefixes) -> OrPageFault<(Modrm, Operand)> {
    decode_operand_with_trailing(pfx, 0)
}

unsafe fn read_operand(operand: Operand, size: OpSize, has_rex: bool) -> OrPageFault<u64> {
    match operand {
        Operand::Reg(r) => Ok(read_reg(r, size, has_rex)),
        Operand::Mem(address) => mem_read(address, size),
    }
}

unsafe fn write_operand(
    operand: Operand,
    size: OpSize,
    has_rex: bool,
    value: u64,
) -> OrPageFault<()> {
    match operand {
        Operand::Reg(r) => {
            write_reg(r, size, has_rex, value);
            Ok(())
        },
        Operand::Mem(address) => mem_write(address, size, value),
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum AluOp {
    Add,
    Adc,
    Or,
    And,
    Sub,
    Sbb,
    Xor,
    Cmp,
}

fn alu_op_of(opcode: u8) -> AluOp {
    match opcode & 0x38 {
        0x00 => AluOp::Add,
        0x08 => AluOp::Or,
        0x10 => AluOp::Adc,
        0x18 => AluOp::Sbb,
        0x20 => AluOp::And,
        0x28 => AluOp::Sub,
        0x30 => AluOp::Xor,
        0x38 => AluOp::Cmp,
        _ => unreachable!(),
    }
}

fn parity_even(value: u64) -> bool { (value as u8).count_ones() % 2 == 0 }

unsafe fn set_szp_flags(result: u64, size: OpSize) {
    let mut f = *flags as u32;
    f &= !(FLAG_ZF | FLAG_SF | FLAG_PF);

    if result & size.mask() == 0 {
        f |= FLAG_ZF;
    }
    if result & size.sign_bit() != 0 {
        f |= FLAG_SF;
    }
    if parity_even(result) {
        f |= FLAG_PF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_add_flags(result: u64, dst: u64, src: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF | FLAG_AF);

    if (dst as u128 + src as u128) >> size.bits() & 1 != 0 {
        f |= FLAG_CF;
    }
    if (dst ^ src ^ result) & 0x10 != 0 {
        f |= FLAG_AF;
    }
    if (dst ^ result) & (src ^ result) & size.sign_bit() != 0 {
        f |= FLAG_OF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_sub_flags(result: u64, dst: u64, src: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF | FLAG_AF);

    if dst & size.mask() < src & size.mask() {
        f |= FLAG_CF;
    }
    if (dst ^ src ^ result) & 0x10 != 0 {
        f |= FLAG_AF;
    }
    if (dst ^ src) & (dst ^ result) & size.sign_bit() != 0 {
        f |= FLAG_OF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_logic_flags(result: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF);
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn alu(op: AluOp, dst: u64, src: u64, size: OpSize) -> u64 {
    let mask = size.mask();
    match op {
        AluOp::Add => {
            let result = dst.wrapping_add(src) & mask;
            set_add_flags(result, dst, src, size);
            result
        },
        AluOp::Adc | AluOp::Sbb => adc_sbb_value(dst, src, size, op == AluOp::Sbb),
        AluOp::Sub | AluOp::Cmp => {
            let result = dst.wrapping_sub(src) & mask;
            set_sub_flags(result, dst, src, size);
            result
        },
        AluOp::And => {
            let result = dst & src & mask;
            set_logic_flags(result, size);
            result
        },
        AluOp::Or => {
            let result = (dst | src) & mask;
            set_logic_flags(result, size);
            result
        },
        AluOp::Xor => {
            let result = (dst ^ src) & mask;
            set_logic_flags(result, size);
            result
        },
    }
}

fn sign_extend(value: u64, size: OpSize) -> i128 {
    match size {
        OpSize::S8 => value as u8 as i8 as i128,
        OpSize::S16 => value as u16 as i16 as i128,
        OpSize::S32 => value as u32 as i32 as i128,
        OpSize::S64 => value as i64 as i128,
    }
}

unsafe fn adc_sbb_value(dst: u64, src: u64, size: OpSize, subtract: bool) -> u64 {
    let mask = size.mask();
    let dst = dst & mask;
    let src = src & mask;
    let signed_dst = sign_extend(dst, size);
    let signed_src = sign_extend(src, size);
    let carry = (*flags as u32 & FLAG_CF) as u64;
    let (result, carry_out, overflow, adjust) = if subtract {
        let rhs = src as u128 + carry as u128;
        let result = dst.wrapping_sub(src).wrapping_sub(carry) & mask;
        let signed = signed_dst - signed_src - carry as i128;
        let min = -(1i128 << (size.bits() - 1));
        let max = (1i128 << (size.bits() - 1)) - 1;
        (result, (dst as u128) < rhs, signed < min || signed > max, (dst & 0xF) < (src & 0xF) + carry)
    }
    else {
        let sum = dst as u128 + src as u128 + carry as u128;
        let result = sum as u64 & mask;
        let signed = signed_dst + signed_src + carry as i128;
        let min = -(1i128 << (size.bits() - 1));
        let max = (1i128 << (size.bits() - 1)) - 1;
        (result, sum > mask as u128, signed < min || signed > max, (dst & 0xF) + (src & 0xF) + carry > 0xF)
    };
    set_szp_flags(result, size);
    let mut f = *flags as u32 & !(FLAG_CF | FLAG_AF | FLAG_OF);
    if carry_out { f |= FLAG_CF; }
    if adjust { f |= FLAG_AF; }
    if overflow { f |= FLAG_OF; }
    *flags = f as i32;
    result
}

unsafe fn imul_value(lhs: u64, rhs: u64, size: OpSize) -> u64 {
    let product = sign_extend(lhs, size) * sign_extend(rhs, size);
    let result = product as u64 & size.mask();
    let extended = sign_extend(result, size);
    *flags &= !(FLAG_CF | FLAG_OF) as i32;
    if product != extended {
        *flags |= (FLAG_CF | FLAG_OF) as i32;
    }
    result
}

// accumulator (AL/AX/EAX/RAX) with an immediate
unsafe fn alu_acc_imm(op: AluOp, size: OpSize, imm: u64) -> OrPageFault<()> {
    let has_rex = false; // no REX on the accumulator immediate forms
    let dst = read_reg(RAX, size, has_rex);
    let result = alu(op, dst, imm, size);
    if op != AluOp::Cmp {
        write_reg(RAX, size, has_rex, result);
    }
    Ok(())
}

unsafe fn alu_rm_r_rm(op: AluOp, opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let has_rex = pfx.has_rex();
    if opcode & 0x02 == 0 {
        // r/m, r
        let src = read_reg(modrm.reg, size, has_rex);
        let dst = read_operand(operand, size, has_rex)?;
        if op != AluOp::Cmp {
            probe_operand_write(operand, size)?;
        }
        let result = alu(op, dst, src, size);
        if op != AluOp::Cmp {
            write_operand(operand, size, has_rex, result)?;
        }
    }
    else {
        // r, r/m
        let src = read_operand(operand, size, has_rex)?;
        let dst = read_reg(modrm.reg, size, has_rex);
        let result = alu(op, dst, src, size);
        if op != AluOp::Cmp {
            write_reg(modrm.reg, size, has_rex, result);
        }
    }
    Ok(())
}

unsafe fn fetch_imm(size: OpSize) -> OrPageFault<u64> {
    Ok(match size {
        OpSize::S8 => fetch8()? as u64,
        OpSize::S16 => fetch16()? as u64,
        // imm32 is sign-extended to 64 bits in long mode
        OpSize::S32 => fetch32()? as u64,
        OpSize::S64 => fetch32()? as i32 as i64 as u64,
    })
}

unsafe fn fetch_imm8_signed(size: OpSize) -> OrPageFault<u64> {
    Ok(fetch8()? as i8 as i64 as u64 & size.mask())
}

unsafe fn alu_group1(opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let trailing = if opcode == 0x80 || opcode == 0x83 || size == OpSize::S8 {
        1
    }
    else if size == OpSize::S16 {
        2
    }
    else {
        4
    };
    let (modrm, operand) = decode_operand_with_trailing(pfx, trailing)?;
    let op = match modrm.reg & 7 {
        0 => AluOp::Add,
        1 => AluOp::Or,
        2 => AluOp::Adc,
        3 => AluOp::Sbb,
        4 => AluOp::And,
        5 => AluOp::Sub,
        6 => AluOp::Xor,
        7 => AluOp::Cmp,
        _ => {
            crate::cpu::cpu::trigger_ud();
            return Ok(());
        },
    };
    let has_rex = pfx.has_rex();
    let dst = read_operand(operand, size, has_rex)?;
    // The immediate follows the ModRM/SIB/displacement bytes
    let imm = if opcode == 0x83 {
        fetch_imm8_signed(size)?
    }
    else {
        fetch_imm(size)?
    };
    if op != AluOp::Cmp {
        probe_operand_write(operand, size)?;
    }
    let result = alu(op, dst, imm, size);
    if op != AluOp::Cmp {
        write_operand(operand, size, has_rex, result)?;
    }
    Ok(())
}

unsafe fn test_rm_r(pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let a = read_operand(operand, size, pfx.has_rex())?;
    let b = read_reg(modrm.reg, size, pfx.has_rex());
    set_logic_flags(a & b & size.mask(), size);
    Ok(())
}

unsafe fn push64(value: u64) -> OrPageFault<()> {
    let rsp = read_reg64(RSP as i32).wrapping_sub(8);
    mem_write(rsp, OpSize::S64, value)?;
    write_reg64(RSP as i32, rsp);
    Ok(())
}

unsafe fn pop64() -> OrPageFault<u64> {
    let rsp = read_reg64(RSP as i32);
    let value = mem_read(rsp, OpSize::S64)?;
    write_reg64(RSP as i32, rsp.wrapping_add(8));
    Ok(value)
}

unsafe fn condition(code: u8) -> bool {
    let f = *flags as u32;
    let cf = f & FLAG_CF != 0;
    let zf = f & FLAG_ZF != 0;
    let sf = f & FLAG_SF != 0;
    let of = f & FLAG_OF != 0;
    let pf = f & FLAG_PF != 0;

    match code {
        0x0 => of,              // O
        0x1 => !of,             // NO
        0x2 => cf,              // B/C
        0x3 => !cf,             // AE/NC
        0x4 => zf,              // E/Z
        0x5 => !zf,             // NE/NZ
        0x6 => cf || zf,        // BE
        0x7 => !cf && !zf,      // A
        0x8 => sf,              // S
        0x9 => !sf,             // NS
        0xA => pf,              // P
        0xB => !pf,             // NP
        0xC => sf != of,        // L
        0xD => sf == of,        // GE
        0xE => zf || sf != of,  // LE
        0xF => !zf && sf == of, // G
        _ => unreachable!(),
    }
}

unsafe fn jcc(code: u8, displacement: i64) {
    if condition(code) {
        *rip = (*rip).wrapping_add(displacement as u64);
    }
}

pub unsafe fn run_one() {
    *previous_rip = *rip;
    let _ = run_one_inner();
    // keep the 32-bit view in sync
    *instruction_pointer = *rip as u32 as i32;
}

// Test entry point: seed rip from instruction_pointer so single-step tests work
// without entering long mode.
#[no_mangle]
pub unsafe fn interp64_run_one() {
    *rip = *instruction_pointer as u32 as u64;
    run_one();
}

// Point CR3 at a PML4, enable PAE and paging, and switch the main loop over to
// the 64-bit interpreter.
#[no_mangle]
pub unsafe fn enter_long_mode(cr3: u32) {
    // The 16/32-bit interpreter keeps arithmetic flags lazy. Long mode reads the
    // flags word directly, so resolve them once on entry.
    *flags = crate::cpu::cpu::get_eflags();
    *flags_changed = 0;
    *rip = *instruction_pointer as u32 as u64;
    *previous_rip = *rip;
    *cr.offset(3) = cr3 as i32; // CR3
    *cr.offset(4) |= crate::cpu::cpu::CR4_PAE; // CR4.PAE
    *cr.offset(4) |= crate::cpu::cpu::CR4_PSE; // CR4.PSE (2 MiB pages)
    *long_mode = true;
    *efer |= EFER_LME | EFER_LMA;
    *cr |= crate::cpu::cpu::CR0_PG; // CR0.PG
    crate::cpu::cpu::full_clear_tlb();
}

/// Enter long mode at a kernel entry point, as a 64-bit boot loader would:
/// identity-mapped lower memory, CS = 0x10, SS = 0x18, RSI = boot_params.
#[no_mangle]
pub unsafe fn boot64(cr3: u32, entry: u64, boot_params: u64) {
    enter_long_mode(cr3);
    *rip = entry;
    *previous_rip = entry;
    *instruction_pointer = entry as u32 as i32;
    write_reg64(RSI as i32, boot_params);
    write_reg64(RSP as i32, 0x9F000);
    set_cpl_segments(0x10, 0);
}

unsafe fn run_one_inner() -> OrPageFault<()> {
    let (pfx, opcode) = decode_prefixes()?;

    if opcode == 0x0F {
        return run_0f(&pfx);
    }

    let size = pfx.operand_size();

    match opcode {
        // NOP
        0x90 => {},

        // HLT
        0xF4 => {
            *in_hlt = true;
        },

        // MOV r/m8, r8  /  MOV r/m, r
        0x88 | 0x89 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x88 { OpSize::S8 } else { size };
            let value = read_reg(modrm.reg, size, pfx.has_rex());
            write_operand(operand, size, pfx.has_rex(), value)?;
        },

        // MOV r8, r/m8  /  MOV r, r/m
        0x8A | 0x8B => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x8A { OpSize::S8 } else { size };
            let value = read_operand(operand, size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // XCHG r/m, r
        0x86 | 0x87 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x86 { OpSize::S8 } else { size };
            let left = read_operand(operand, size, pfx.has_rex())?;
            let right = read_reg(modrm.reg, size, pfx.has_rex());
            write_operand(operand, size, pfx.has_rex(), right)?;
            write_reg(modrm.reg, size, pfx.has_rex(), left);
        },

        // MOV r/m, imm (0xC6: r/m8, imm8; 0xC7: r/m, imm16/32)
        0xC6 | 0xC7 => {
            let size = if opcode == 0xC6 { OpSize::S8 } else { size };
            let trailing = if size == OpSize::S8 { 1 } else if size == OpSize::S16 { 2 } else { 4 };
            let (_, operand) = decode_operand_with_trailing(&pfx, trailing)?;
            let value = fetch_imm(size)?;
            write_operand(operand, size, pfx.has_rex(), value)?;
        },

        // MOV r, imm (0xB0-0xB7: r8, imm8; 0xB8-0xBF: r, imm)
        0xB0..=0xB7 => {
            let r = (opcode - 0xB0) | (pfx.rex & prefix::REX_B) << 3;
            let value = fetch8()? as u64;
            write_reg(r, OpSize::S8, pfx.has_rex(), value);
        },
        0xB8..=0xBF => {
            let r = (opcode - 0xB8) | (pfx.rex & prefix::REX_B) << 3;
            let value = if size == OpSize::S64 {
                fetch64()?
            }
            else {
                fetch32()? as u64
            };
            write_reg(r, size, pfx.has_rex(), value);
        },

        // LEA r, m (0x8D)
        0x8D => {
            let modrm = decode_modrm(&pfx)?;
            if modrm.mod_bits == 3 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            let address = modrm_effective_address(&modrm, &pfx)?;
            write_reg(modrm.reg, size, pfx.has_rex(), address);
        },

        // ALU r/m, r  and  r, r/m
        0x00 | 0x01 | 0x02 | 0x03 | 0x08 | 0x09 | 0x0A | 0x0B | 0x10 | 0x11 | 0x12 | 0x13
        | 0x18 | 0x19 | 0x1A | 0x1B | 0x20 | 0x21 | 0x22 | 0x23 | 0x28 | 0x29 | 0x2A
        | 0x2B | 0x30 | 0x31 | 0x32 | 0x33 | 0x38 | 0x39 | 0x3A | 0x3B => {
            let size = if opcode & 1 == 0 { OpSize::S8 } else { size };
            alu_rm_r_rm(alu_op_of(opcode), opcode, &pfx, size)?;
        },

        // ALU accumulator, imm (0x04/0x0C/...: AL, imm8; 0x05/0x0D/...: rAX, imm)
        0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
            let imm = fetch8()? as u64;
            alu_acc_imm(alu_op_of(opcode), OpSize::S8, imm)?;
        },
        0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
            let imm = fetch_imm(size)?;
            alu_acc_imm(alu_op_of(opcode), size, imm)?;
        },

        // ALU r/m, imm
        0x80 => alu_group1(opcode, &pfx, OpSize::S8)?,
        0x81 => alu_group1(opcode, &pfx, size)?,
        0x83 => alu_group1(opcode, &pfx, size)?,

        // TEST
        0x84 => test_rm_r(&pfx, OpSize::S8)?,
        0x85 => test_rm_r(&pfx, size)?,
        0xA8 => {
            let a = read_reg(RAX, OpSize::S8, false);
            let imm = fetch8()? as u64;
            set_logic_flags(a & imm, OpSize::S8);
        },
        0xA9 => {
            let a = read_reg(RAX, size, false);
            let imm = fetch_imm(size)?;
            set_logic_flags(a & imm & size.mask(), size);
        },

        // NOT/NEG r/m
        0xF7 => {
            let modrm = decode_modrm(&pfx)?;
            let trailing = if modrm.reg & 7 == 0 { if size == OpSize::S16 { 2 } else { 4 } } else { 0 };
            let (modrm, operand) = decode_operand_after_modrm(&pfx, modrm, trailing)?;
            let value = read_operand(operand, size, pfx.has_rex())?;
            match modrm.reg & 7 {
                0 => {
                    let immediate = fetch_imm(size)?;
                    set_logic_flags(value & immediate & size.mask(), size);
                },
                2 => {
                    probe_operand_write(operand, size)?;
                    write_operand(operand, size, pfx.has_rex(), !value & size.mask())?;
                },
                3 => {
                    probe_operand_write(operand, size)?;
                    let result = alu(AluOp::Sub, 0, value, size);
                    write_operand(operand, size, pfx.has_rex(), result)?;
                },
                _ => crate::cpu::cpu::trigger_ud(),
            }
        },

        // SHL/SHR/SAR r/m, 1/imm8
        0xD1 | 0xC1 | 0xD3 => {
            let (modrm, operand) = decode_operand_with_trailing(
                &pfx,
                if opcode == 0xC1 { 1 } else { 0 },
            )?;
            let count = match opcode {
                0xD1 => 1,
                0xC1 => fetch8()? as u32,
                _ => read_reg64(RCX as i32) as u32 & 0xFF,
            }
                & if size == OpSize::S64 { 63 } else { 31 };
            if count == 0 {
                return Ok(());
            }
            let old = read_operand(operand, size, pfx.has_rex())? & size.mask();
            probe_operand_write(operand, size)?;
            let sign = size.sign_bit();
            let (result, cf, of) = match modrm.reg & 7 {
                4 => {
                    let result = old.wrapping_shl(count) & size.mask();
                    let cf = old >> (size.bits() - count) & 1 != 0;
                    (result, cf, count == 1 && (result & sign != 0) ^ cf)
                },
                5 => (old >> count, old >> (count - 1) & 1 != 0, count == 1 && old & sign != 0),
                7 => {
                    let signed = (old ^ sign).wrapping_sub(sign);
                    let result = ((signed as i64) >> count) as u64 & size.mask();
                    (result, old >> (count - 1) & 1 != 0, false)
                },
                _ => {
                    crate::cpu::cpu::trigger_ud();
                    return Ok(());
                },
            };
            set_logic_flags(result, size);
            *flags |= if cf { FLAG_CF as i32 } else { 0 };
            *flags |= if of { FLAG_OF as i32 } else { 0 };
            write_operand(operand, size, pfx.has_rex(), result)?;
        },

        // PUSH/POP r64 (0x50-0x5F)
        0x50..=0x57 => {
            let r = (opcode - 0x50) | (pfx.rex & prefix::REX_B) << 3;
            push64(read_reg64(r as i32))?;
        },
        0x58..=0x5F => {
            let r = (opcode - 0x58) | (pfx.rex & prefix::REX_B) << 3;
            let value = pop64()?;
            write_reg64(r as i32, value);
        },

        // MOVSXD r64, r/m32
        0x63 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            if size != OpSize::S64 {
                crate::cpu::cpu::trigger_ud();
            }
            else {
                let value = read_operand(operand, OpSize::S32, pfx.has_rex())? as u32 as i32 as i64;
                write_reg64(modrm.reg as i32, value as u64);
            }
        },

        // IMUL r, r/m, imm32/imm8
        0x69 | 0x6B => {
            let trailing = if opcode == 0x6B { 1 } else if size == OpSize::S16 { 2 } else { 4 };
            let (modrm, operand) = decode_operand_with_trailing(&pfx, trailing)?;
            let source = read_operand(operand, size, pfx.has_rex())?;
            let immediate = if opcode == 0x6B {
                fetch_imm8_signed(size)?
            }
            else {
                fetch_imm(size)?
            };
            write_reg(
                modrm.reg,
                size,
                pfx.has_rex(),
                imul_value(source, immediate, size),
            );
        },
        // PUSH imm (0x68: imm32, 0x6A: imm8)
        0x68 => {
            let value = fetch32()? as i32 as i64 as u64;
            push64(value)?;
        },
        0x6A => {
            let value = fetch8()? as i8 as i64 as u64;
            push64(value)?;
        },

        // Jcc rel8 (0x70-0x7F)
        0x70..=0x7F => {
            let displacement = fetch8()? as i8 as i64;
            jcc(opcode - 0x70, displacement);
        },

        // IN AL, imm8 (E4) / IN AX|EAX, imm8 (E5)
        0xE4 | 0xE5 => {
            let port = fetch8()? as i32;
            let (op, width) = if opcode == 0xE4 {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                let value = match width {
                    1 => io_port_read8(port) as u64,
                    2 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
                };
                write_reg(RAX, op, false, value);
            }
        },

        // OUT imm8, AL (E6) / OUT imm8, AX|EAX (E7)
        0xE6 | 0xE7 => {
            let port = fetch8()? as i32;
            let (op, width) = if opcode == 0xE6 {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                match width {
                    1 => io_port_write8(port, read_reg(RAX, op, false) as i32),
                    2 => crate::cpu::cpu::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::cpu::io_port_write32(port, read_reg(RAX, op, false) as i32),
                }
            }
        },

        // IN AL, DX (EC) / IN AX|EAX, DX (ED)
        0xEC | 0xED => {
            let port = read_reg(RDX, OpSize::S16, false) as i32;
            let (op, width) = if opcode == 0xEC {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                let value = match width {
                    1 => io_port_read8(port) as u64,
                    2 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
                };
                write_reg(RAX, op, false, value);
            }
        },

        // OUT DX, AL (EE) / OUT DX, AX|EAX (EF)
        0xEE | 0xEF => {
            let port = read_reg(RDX, OpSize::S16, false) as i32;
            let (op, width) = if opcode == 0xEE {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                match width {
                    1 => io_port_write8(port, read_reg(RAX, op, false) as i32),
                    2 => crate::cpu::cpu::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::cpu::io_port_write32(port, read_reg(RAX, op, false) as i32),
                }
            }
        },

        // JMP rel8/rel32
        0xEB => {
            let displacement = fetch8()? as i8 as i64;
            *rip = (*rip).wrapping_add(displacement as u64);
        },
        0xE9 => {
            let displacement = fetch32()? as i32 as i64;
            *rip = (*rip).wrapping_add(displacement as u64);
        },

        // CALL/JMP r/m64 (FF /2 and /4)
        0xFF => {
            let (modrm, operand) = decode_operand(&pfx)?;
            match modrm.reg & 7 {
                group @ (0 | 1) => {
                    let carry = *flags & FLAG_CARRY;
                    let value = read_operand(operand, size, pfx.has_rex())?;
                    probe_operand_write(operand, size)?;
                    let result = alu(
                        if group == 0 { AluOp::Add } else { AluOp::Sub },
                        value,
                        1,
                        size,
                    );
                    *flags = *flags & !FLAG_CARRY | carry;
                    write_operand(operand, size, pfx.has_rex(), result)?;
                },
                2 => {
                    let target = read_operand(operand, OpSize::S64, pfx.has_rex())?;
                    let return_address = *rip;
                    push64(return_address)?;
                    *rip = target;
                },
                4 => *rip = read_operand(operand, OpSize::S64, pfx.has_rex())?,
                // jmp m16:64. The indirect far jump firmware uses to land in a
                // 64-bit code segment; long mode has no direct form.
                5 => {
                    let Operand::Mem(address) = operand else {
                        crate::cpu::cpu::trigger_ud();
                        return Ok(());
                    };
                    let target = mem_read(address, OpSize::S64)?;
                    let selector = mem_read(address.wrapping_add(8), OpSize::S16)? as u16;
                    set_cpl_segments(selector, selector as u8 & 3);
                    *rip = target;
                },
                _ => crate::cpu::cpu::trigger_ud(),
            }
        },

        // CALL rel32 (0xE8)
        0xE8 => {
            let displacement = fetch32()? as i32 as i64;
            let return_address = *rip;
            push64(return_address)?;
            *rip = (*rip).wrapping_add(displacement as u64);
        },

        // RET (0xC3) / RET imm16 (0xC2)
        0xC3 => {
            *rip = pop64()?;
        },
        0xC2 => {
            let adjustment = fetch16()? as u64;
            *rip = pop64()?;
            let rsp = read_reg64(RSP as i32).wrapping_add(adjustment);
            write_reg64(RSP as i32, rsp);
        },

        // LEAVE: mov rsp, rbp; pop rbp
        0xC9 => {
            write_reg64(RSP as i32, read_reg64(RBP as i32));
            let value = pop64()?;
            write_reg64(RBP as i32, value);
        },

        // IRETQ (0xCF)
        0xCF => {
            let new_rip = pop64()?;
            let cs = pop64()?;
            let rflags = pop64()?;
            let rsp = pop64()?;
            let ss = pop64()?;
            *sreg.offset(1) = cs as u16; // CS
            *sreg.offset(2) = ss as u16; // SS
            *flags = rflags as u32 as i32;
            write_reg64(RSP as i32, rsp);
            *rip = new_rip;
        },

        // INT3 (0xCC) / INT imm8 (0xCD)
        0xCC => crate::cpu::cpu::call_interrupt_vector64(3, None),
        0xCD => {
            let vector = fetch8()? as i32;
            crate::cpu::cpu::call_interrupt_vector64(vector, None);
        },

        // CLI (0xFA) / STI (0xFB)
        0xFA => *flags &= !FLAG_INTERRUPT,
        0xFB => *flags |= FLAG_INTERRUPT,

        _ => {
            dbg_log!("#ud interp64: opcode {:02x}", opcode);
            crate::cpu::cpu::trigger_ud();
        },
    }

    Ok(())
}

unsafe fn run_0f(pfx: &Prefixes) -> OrPageFault<()> {
    let opcode = fetch8()?;
    let size = pfx.operand_size();

    match opcode {
        // Jcc rel32 (0F 80-0F 8F)
        0x80..=0x8F => {
            let displacement = fetch32()? as i32 as i64;
            jcc(opcode - 0x80, displacement);
        },

        // CMOVcc r, r/m
        0x40..=0x4F => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, size, pfx.has_rex())?;
            if condition(opcode - 0x40) {
                write_reg(modrm.reg, size, pfx.has_rex(), value);
            }
        },

        // SETcc r/m8
        0x90..=0x9F => {
            let (_, operand) = decode_operand(pfx)?;
            write_operand(
                operand,
                OpSize::S8,
                pfx.has_rex(),
                condition(opcode - 0x90) as u64,
            )?;
        },

        // MOVZX r, r/m8 (0F B6) / r/m16 (0F B7)
        0xB6 | 0xB7 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src_size = if opcode == 0xB6 { OpSize::S8 } else { OpSize::S16 };
            let value = read_operand(operand, src_size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // MOVSX r, r/m8 (0F BE) / r/m16 (0F BF)
        0xBE | 0xBF => {
            let (modrm, operand) = decode_operand(pfx)?;
            let (src_size, sign) = if opcode == 0xBE {
                (OpSize::S8, 0x80u64)
            }
            else {
                (OpSize::S16, 0x8000u64)
            };
            let value = read_operand(operand, src_size, pfx.has_rex())?;
            let value = if value & sign != 0 {
                value | !src_size.mask()
            }
            else {
                value
            } & size.mask();
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // IMUL r, r/m (0F AF)
        0xAF => {
            let (modrm, operand) = decode_operand(pfx)?;
            let lhs = read_reg(modrm.reg, size, pfx.has_rex());
            let rhs = read_operand(operand, size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), imul_value(lhs, rhs, size));
        },

        // NOP r/m (0F 1F)
        0x1F => {
            let _ = decode_operand(pfx)?;
        },

        // ENDBR64/ENDBR32
        0x1E => {
            match fetch8()? {
                0xFA | 0xFB => {},
                _ => crate::cpu::cpu::trigger_ud(),
            }
        },

        // MOV r64, CRn (0F 20) / MOV CRn, r64 (0F 22)
        0x20 | 0x22 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits != 3 || modrm.reg > 7 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            if opcode == 0x20 {
                let value = match modrm.reg {
                    0 => *cr as u32 as u64,
                    2 => *cr2,
                    3 => *cr.offset(3) as u32 as u64,
                    4 => *cr.offset(4) as u32 as u64,
                    _ => {
                        crate::cpu::cpu::trigger_ud();
                        return Ok(());
                    },
                };
                write_reg(modrm.rm, OpSize::S64, true, value);
            }
            else if modrm.reg == 2 {
                let value = read_reg(modrm.rm, OpSize::S64, true);
                *cr2 = value;
                *cr.offset(2) = value as i32;
            }
            else {
                crate::cpu::instructions_0f::instr_0F22(modrm.rm as i32, modrm.reg as i32);
            }
        },

        // CPUID (0F A2)
        0xA2 => {
            crate::cpu::instructions_0f::instr_0FA2();
        },

        // BSWAP r32/r64 (0F C8-0F CF)
        0xC8..=0xCF => {
            let r = opcode - 0xC8 | (pfx.rex & prefix::REX_B) << 3;
            let value = read_reg(r, size, pfx.has_rex());
            write_reg(
                r,
                size,
                pfx.has_rex(),
                if size == OpSize::S64 { value.swap_bytes() } else { (value as u32).swap_bytes() as u64 },
            );
        },

        // SWAPGS (0F 01 F8)
        0x01 => {
            if fetch8()? == 0xF8 {
                let tmp = *gs_base;
                *gs_base = *kernel_gs_base;
                *kernel_gs_base = tmp;
            }
            else {
                crate::cpu::cpu::trigger_ud();
            }
        },

        // SYSCALL (0F 05)
        0x05 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            write_reg64(RCX as i32, *rip);
            write_reg64(R11 as i32, *flags as u32 as u64);
            *flags &= !(*sfmask as u32) as i32;
            *rip = *lstar;
            let cs = (*star >> 32 & 0xFFFF) as u16;
            set_cpl_segments(cs, 0);
        },

        // SYSRET (0F 07)
        0x07 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            let target = read_reg64(RCX as i32);
            *flags = read_reg64(R11 as i32) as u32 as i32;
            let cs = (*star >> 48 & 0xFFFF) as u16 | 3;
            set_cpl_segments(cs, 3);
            *rip = target;
        },

        // WRMSR (0F 30)
        0x30 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            let value = read_reg64(RAX as i32) as u32 as u64
                | (read_reg64(RDX as i32) as u32 as u64) << 32;
            if !msr_write(index, value) {
                crate::cpu::instructions_0f::instr_0F30();
            }
        },

        // RDTSC (0F 31)
        0x31 => {
            if *cpl == 0 || *cr.offset(4) & CR4_TSD == 0 {
                let tsc = read_tsc();
                write_reg(RAX, OpSize::S32, false, tsc & 0xFFFF_FFFF);
                write_reg(RDX, OpSize::S32, false, tsc >> 32);
            }
            else {
                crate::cpu::cpu::trigger_gp(0);
            }
        },

        // RDMSR (0F 32)
        0x32 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            match msr_read(index) {
                Some(value) => {
                    write_reg(RAX, OpSize::S32, false, value & 0xFFFF_FFFF);
                    write_reg(RDX, OpSize::S32, false, value >> 32);
                },
                None => crate::cpu::instructions_0f::instr_0F32(),
            }
        },

        _ => {
            dbg_log!("#ud interp64: 0f opcode {:02x}", opcode);
            crate::cpu::cpu::trigger_ud();
        },
    }

    Ok(())
}
