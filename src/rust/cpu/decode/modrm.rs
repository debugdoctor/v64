use crate::cpu::decode::address::effective_address;
use crate::cpu::core::*;
use crate::cpu::decode::operand::{Modrm, Operand};
use crate::paging::OrPageFault;

// 16/32-bit ModRM effective address, computed by the shared model
// (`address::effective_address`): base + index*2^scale + disp, wrapped to the
// address size, plus the segment base.
#[inline]
unsafe fn seg_ea(
    seg: i32,
    base: u64,
    index: u64,
    scale: u8,
    disp: u64,
    addr_size: u32,
) -> OrPageFault<Operand> {
    Ok(Operand::Mem(effective_address(base, index, scale, disp, get_seg_prefix(seg)? as u64, addr_size)))
}

pub unsafe fn resolve_modrm16(modrm_byte: i32) -> OrPageFault<Operand> {
    let bx = read_reg16(BX) as u64;
    let bp = read_reg16(BP) as u64;
    let si = read_reg16(SI) as u64;
    let di = read_reg16(DI) as u64;
    match modrm_byte & !0o070 {
        0o000 => seg_ea(DS, bx, si, 0, 0, 16),
        0o100 => seg_ea(DS, bx, si, 0, read_imm8s()? as u64, 16),
        0o200 => seg_ea(DS, bx, si, 0, read_imm16()? as u64, 16),
        0o001 => seg_ea(DS, bx, di, 0, 0, 16),
        0o101 => seg_ea(DS, bx, di, 0, read_imm8s()? as u64, 16),
        0o201 => seg_ea(DS, bx, di, 0, read_imm16()? as u64, 16),
        0o002 => seg_ea(SS, bp, si, 0, 0, 16),
        0o102 => seg_ea(SS, bp, si, 0, read_imm8s()? as u64, 16),
        0o202 => seg_ea(SS, bp, si, 0, read_imm16()? as u64, 16),
        0o003 => seg_ea(SS, bp, di, 0, 0, 16),
        0o103 => seg_ea(SS, bp, di, 0, read_imm8s()? as u64, 16),
        0o203 => seg_ea(SS, bp, di, 0, read_imm16()? as u64, 16),
        0o004 => seg_ea(DS, si, 0, 0, 0, 16),
        0o104 => seg_ea(DS, si, 0, 0, read_imm8s()? as u64, 16),
        0o204 => seg_ea(DS, si, 0, 0, read_imm16()? as u64, 16),
        0o005 => seg_ea(DS, di, 0, 0, 0, 16),
        0o105 => seg_ea(DS, di, 0, 0, read_imm8s()? as u64, 16),
        0o205 => seg_ea(DS, di, 0, 0, read_imm16()? as u64, 16),
        0o006 => seg_ea(DS, 0, 0, 0, read_imm16()? as u64, 16),
        0o106 => seg_ea(SS, bp, 0, 0, read_imm8s()? as u64, 16),
        0o206 => seg_ea(SS, bp, 0, 0, read_imm16()? as u64, 16),
        0o007 => seg_ea(DS, bx, 0, 0, 0, 16),
        0o107 => seg_ea(DS, bx, 0, 0, read_imm8s()? as u64, 16),
        0o207 => seg_ea(DS, bx, 0, 0, read_imm16()? as u64, 16),
        _ => {
            dbg_assert!(false);
            std::hint::unreachable_unchecked()
        },
    }
}

// SIB byte -> (base, index, scale, segment). `with_imm` selects the mod=00
// base==5 case (disp32 instead of EBP).
unsafe fn sib_parts(with_imm: bool) -> OrPageFault<(u64, u64, u8, i32)> {
    let sib_byte = read_imm8()?;
    let r = sib_byte & 7;
    let m = sib_byte >> 3 & 7;
    let (base, seg) = if r == 4 {
        (read_reg32(ESP) as u64, SS)
    }
    else if r == 5 {
        if with_imm {
            (read_reg32(EBP) as u64, SS)
        }
        else {
            (read_imm32s()? as u64, DS)
        }
    }
    else {
        (read_reg32(r) as u64, DS)
    };
    let (index, scale) = if m == 4 {
        (0, 0)
    }
    else {
        (read_reg32(m) as u64, (sib_byte >> 6 & 3) as u8)
    };
    Ok((base, index, scale, seg))
}

pub unsafe fn resolve_modrm32_(modrm_byte: i32) -> OrPageFault<Operand> {
    let m = Modrm::from_byte(modrm_byte);
    dbg_assert!(m.mod_bits != 3);
    Ok(if m.rm as i32 == 4 {
        let (base, index, scale, seg) = sib_parts(m.mod_bits > 0)?;
        let disp = if m.mod_bits == 0 {
            0
        }
        else if m.mod_bits == 1 {
            read_imm8s()? as u64
        }
        else {
            read_imm32s()? as u64
        };
        seg_ea(seg, base, index, scale, disp, 32)?
    }
    else if m.rm as i32 == 5 {
        if m.mod_bits == 0 {
            seg_ea(DS, read_imm32s()? as u64, 0, 0, 0, 32)?
        }
        else {
            seg_ea(
                SS,
                read_reg32(EBP) as u64,
                0,
                0,
                if m.mod_bits == 1 { read_imm8s()? as u64 } else { read_imm32s()? as u64 },
                32,
            )?
        }
    }
    else if m.mod_bits == 0 {
        seg_ea(DS, read_reg32(m.rm as i32) as u64, 0, 0, 0, 32)?
    }
    else {
        seg_ea(
            DS,
            read_reg32(m.rm as i32) as u64,
            0,
            0,
            if m.mod_bits == 1 { read_imm8s()? as u64 } else { read_imm32s()? as u64 },
            32,
        )?
    })
}

pub unsafe fn resolve_modrm32(modrm_byte: i32) -> OrPageFault<Operand> {
    let eax = read_reg32(EAX) as u64;
    let ecx = read_reg32(ECX) as u64;
    let edx = read_reg32(EDX) as u64;
    let ebx = read_reg32(EBX) as u64;
    let esi = read_reg32(ESI) as u64;
    let edi = read_reg32(EDI) as u64;
    match modrm_byte & !0o070 {
        0o000 => seg_ea(DS, eax, 0, 0, 0, 32),
        0o100 => seg_ea(DS, eax, 0, 0, read_imm8s()? as u64, 32),
        0o200 => seg_ea(DS, eax, 0, 0, read_imm32s()? as u64, 32),
        0o001 => seg_ea(DS, ecx, 0, 0, 0, 32),
        0o101 => seg_ea(DS, ecx, 0, 0, read_imm8s()? as u64, 32),
        0o201 => seg_ea(DS, ecx, 0, 0, read_imm32s()? as u64, 32),
        0o002 => seg_ea(DS, edx, 0, 0, 0, 32),
        0o102 => seg_ea(DS, edx, 0, 0, read_imm8s()? as u64, 32),
        0o202 => seg_ea(DS, edx, 0, 0, read_imm32s()? as u64, 32),
        0o003 => seg_ea(DS, ebx, 0, 0, 0, 32),
        0o103 => seg_ea(DS, ebx, 0, 0, read_imm8s()? as u64, 32),
        0o203 => seg_ea(DS, ebx, 0, 0, read_imm32s()? as u64, 32),
        0o004 => {
            let (base, index, scale, seg) = sib_parts(false)?;
            seg_ea(seg, base, index, scale, 0, 32)
        },
        0o104 => {
            let (base, index, scale, seg) = sib_parts(true)?;
            seg_ea(seg, base, index, scale, read_imm8s()? as u64, 32)
        },
        0o204 => {
            let (base, index, scale, seg) = sib_parts(true)?;
            seg_ea(seg, base, index, scale, read_imm32s()? as u64, 32)
        },
        0o005 => seg_ea(DS, read_imm32s()? as u64, 0, 0, 0, 32),
        0o105 => seg_ea(SS, read_reg32(EBP) as u64, 0, 0, read_imm8s()? as u64, 32),
        0o205 => seg_ea(SS, read_reg32(EBP) as u64, 0, 0, read_imm32s()? as u64, 32),
        0o006 => seg_ea(DS, esi, 0, 0, 0, 32),
        0o106 => seg_ea(DS, esi, 0, 0, read_imm8s()? as u64, 32),
        0o206 => seg_ea(DS, esi, 0, 0, read_imm32s()? as u64, 32),
        0o007 => seg_ea(DS, edi, 0, 0, 0, 32),
        0o107 => seg_ea(DS, edi, 0, 0, read_imm8s()? as u64, 32),
        0o207 => seg_ea(DS, edi, 0, 0, read_imm32s()? as u64, 32),
        _ => {
            dbg_assert!(false);
            std::hint::unreachable_unchecked()
        },
    }
}
