//! Shared decode-cache machinery for the interpreters.
//!
//! A basic block is decoded once into engine-specific instructions (`I`),
//! cached by virtual RIP, and executed directly. Self-modifying code is caught
//! with a per-physical-page write counter: a block records its page's counter
//! at decode time and only re-checks the bytes when the counter moved. The
//! JIT's inline stores bypass the counter, so an engine must drop its cache if a
//! JIT that can write those pages is (re-)enabled.

use crate::paging::OrPageFault;

// --- Instruction representation ----------------------------------------------
//
// Mode-agnostic: operand widths 8/16/32/64, registers 0-15, and a `Mem.addr_size`
// of 64 (long mode default), 32 (`67` in long mode, protected mode default) or 16
// (`67` in protected mode). Both the long-mode decoder (jit64.rs) and the
// protected-mode decoder (jit32.rs) produce these, and either engine can execute
// them, so this lives next to the cache that stores them.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8, // index << scale

    pub disp: i64,
    // Effective-address width: 64 by default, 32 with a `67` prefix in long
    // mode, 16 with a `67` prefix in protected mode.
    pub addr_size: u8,
    // FS (0x64) / GS (0x65) override, whose base is added to the address.
    pub segment: Option<u8>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Instruction {
    MovRegImm { r: u8, value: u64, width: u8 },
    MovRegReg { dst: u8, src: u8, width: u8 },
    MovRegMem { dst: u8, mem: Mem, width: u8 },
    MovMemReg { mem: Mem, src: u8, width: u8 },
    MovMemImm { mem: Mem, value: u64, width: u8 },
    MovExtendReg { dst: u8, src: u8, src_width: u8, dst_width: u8, signed: bool, high8: bool },
    MovExtendMem { dst: u8, mem: Mem, src_width: u8, dst_width: u8, signed: bool },
    XchgRegReg { a: u8, b: u8, width: u8 },
    XchgMemReg { mem: Mem, r: u8, width: u8 },
    NotReg { r: u8, width: u8 },
    NotMem { mem: Mem, width: u8 },
    NegReg { r: u8, width: u8 },
    NegMem { mem: Mem, width: u8 },
    IncDecReg { r: u8, width: u8, decrement: bool },
    IncDecMem { mem: Mem, width: u8, decrement: bool },
    CmovRegReg { code: u8, dst: u8, src: u8, width: u8 },
    CmovRegMem { code: u8, dst: u8, mem: Mem, width: u8 },
    SetccReg { code: u8, dst: u8, high8: bool },
    SetccMem { code: u8, mem: Mem },
    ShiftReg { kind: ShiftKind, r: u8, width: u8, count: u8 },
    ShiftRegCl { kind: ShiftKind, r: u8, width: u8 },
    ShiftMem { kind: ShiftKind, mem: Mem, width: u8, count: u8 },
    ShiftMemCl { kind: ShiftKind, mem: Mem, width: u8 },
    Cpuid,
    Rdtsc,
    // A VEX/AVX instruction: executed by the interpreter at `rip` (the JIT and
    // the decode cache call the shared implementation).
    Avx { rip: u64 },
    VectorReg { op: u8, dst: u8, src1: u8, src2: u8, wide: bool },
    Rorx { dst: u8, src: u8, width: u8, count: u8 },
    BmiShift { dst: u8, src: u8, count: u8, width: u8, kind: u8 },
    SignHigh { width: u8 },

    // SSE2. The XMM file lives in emulated memory, so these are pairs of
    // 64-bit accesses rather than 128-bit values.
    XmmCopy { dst: u8, src: u8 },
    XmmLoad { dst: u8, mem: Mem },
    XmmStore { mem: Mem, src: u8 },
    XmmXor { dst: u8, src: u8 },
    XmmXorMem { dst: u8, mem: Mem },
    // Integer SSE2 ops (66 0F xx), applied by a helper shared with interp64.
    XmmInt { op: u8, dst: u8, src: u8 },
    XmmIntMem { op: u8, dst: u8, mem: Mem },
    // Packed shift by immediate (66 0F 71/72/73), register operand only.
    XmmShiftImm { op: u8, group: u8, count: u8, dst: u8 },
    // PSHUFD (66 0F 70), register or memory source.
    XmmShuf { control: u8, dst: u8, src: u8 },
    XmmShufMem { control: u8, dst: u8, mem: Mem },

    In { imm: Option<u16>, width: u8 },
    Out { imm: Option<u16>, width: u8 },
    Cli,
    PushFlags,
    CmpxchgReg { dst: u8, src: u8, width: u8 },
    CmpxchgMem { mem: Mem, src: u8, width: u8 },
    BitTestReg { r: u8, index_r: u8, op: u8, width: u8 },
    BitTestImm { r: u8, index: u64, op: u8, width: u8 },
    ImulRegReg { dst: u8, lhs: u8, rhs: u8, width: u8 },
    ImulRegImm { dst: u8, src: u8, value: u64, width: u8 },
    ImulRegMem { dst: u8, mem: Mem, value: Option<u64>, width: u8 },
    Bswap { r: u8, width: u8 },
    Lea { dst: u8, mem: Mem, width: u8 },
    AddRegReg { dst: u8, src: u8, width: u8 },
    AddRegImm { r: u8, value: u64, width: u8 },
    // cmp/test update flags without writing dst
    ArithRegReg { op: ArithOp, dst: u8, src: u8, width: u8 },
    ArithRegImm { op: ArithOp, r: u8, value: u64, width: u8 },
    ArithRegMem { op: ArithOp, dst: u8, mem: Mem, width: u8 },
    ArithMemReg { op: ArithOp, mem: Mem, src: u8, width: u8 },
    ArithMemImm { op: ArithOp, mem: Mem, value: u64, width: u8 },
    PushReg { r: u8 },
    PopReg { r: u8 },
    PushImm { value: u64 },
    Call { target: u64, return_address: u64 },
    CallReg { r: u8, return_address: u64 },
    CallMem { mem: Mem, return_address: u64 },
    JmpReg { r: u8 },
    JmpMem { mem: Mem },
    Ret { adjustment: u16 },
    Leave,
    Nop,
    Jmp { target: u64 },
    // fallthrough is the next instruction
    Jcc { code: u8, target: u64, fallthrough: u64 },
    Hlt,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ArithOp {
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    Test,
    And,
    Or,
    Xor,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ShiftKind { Shl, Shr, Sar, Rol, Ror }


/// One decoded instruction.
pub trait DecodedInstr: Copy {
    /// Whether the interpreter can execute this instruction directly.
    fn is_supported(&self) -> bool;
    /// `Some((condition code, target rip, fallthrough rip))` for a conditional
    /// branch, else `None`.
    fn jcc(&self) -> Option<(u8, u64, u64)>;
    /// Execute the instruction.
    fn exec(&self) -> OrPageFault<JStep>;
}

pub enum JStep {
    Continue,
    Stop,
}

/// Per-engine hooks used by the shared loop.
pub trait DecodeHost {
    type Instr: DecodedInstr;
    unsafe fn set_previous_ip(rip: u64);
    unsafe fn set_ip(rip: u64);
    unsafe fn retire(n: u32);
    unsafe fn page_write_count(phys_page: u32) -> u32;
    unsafe fn validate_block(start_rip: u64, bytes: &[u8]) -> bool;
    unsafe fn decode_block_at(
        start_rip: u64,
    ) -> Option<(Vec<Self::Instr>, Vec<u64>, u64, Vec<u8>, u32)>;
    /// Called after a freshly decoded block is stored, with the block's physical
    /// page. The engine uses this to keep the JIT from writing that page inline:
    /// such a write would bypass the page write counter this cache validates
    /// against.
    unsafe fn on_block_cached(_phys_page: u32) {}
}

pub struct DecEntry<I> {
    pub valid: bool,
    pub tag: u64,
    pub gen: u32,
    pub user: bool,
    pub end_rip: u64,
    pub limit: usize,
    // Physical page of the block and its write counter when it was decoded.
    pub phys_page: u32,
    pub page_version: u32,
    pub instrs: Vec<I>,
    pub rips: Vec<u64>,
    pub bytes: Vec<u8>,
}

impl<I> Default for DecEntry<I> {
    fn default() -> Self {
        DecEntry {
            valid: false,
            tag: 0,
            gen: 0,
            user: false,
            end_rip: 0,
            limit: 0,
            phys_page: 0,
            page_version: 0,
            instrs: Vec::new(),
            rips: Vec::new(),
            bytes: Vec::new(),
        }
    }
}

impl<I> DecEntry<I> {
    fn negative(tag: u64, gen: u32, user: bool) -> Self {
        DecEntry { valid: true, tag, gen, user, ..Default::default() }
    }
}

/// Execute a decoded block. Each instruction is read once and the next rip is
/// carried across iterations instead of re-indexing `instrs`/`rips`.
pub unsafe fn exec_decoded<H: DecodeHost>(entry: &DecEntry<H::Instr>) -> u32 {
    let instrs = entry.instrs.as_slice();
    let rips = entry.rips.as_slice();
    let len = instrs.len();
    let limit = entry.limit;
    let mut i = 0;
    let mut retired = 0u32;
    while i < limit {
        let instr = &instrs[i];
        H::set_previous_ip(rips[i]);
        let resume = if i + 1 < len { rips[i + 1] } else { entry.end_rip };
        H::set_ip(resume);
        // A non-final conditional branch only exits when taken; otherwise it
        // falls through to the next instruction in the same block.
        if let Some((code, target, fallthrough)) = instr.jcc() {
            retired += 1;
            if crate::cpu::core::condition(code) {
                H::retire(retired);
                H::set_ip(target);
                return (i + 1) as u32;
            }
            if i + 1 == len {
                H::retire(retired);
                H::set_ip(fallthrough);
                return (i + 1) as u32;
            }
            i += 1;
            continue;
        }
        let step = match instr.exec() {
            Ok(s) => s,
            Err(()) => {
                H::retire(retired + 1);
                return (i + 1) as u32;
            },
        };
        retired += 1;
        if let JStep::Stop = step {
            H::retire(retired);
            return (i + 1) as u32;
        }
        i += 1;
    }
    if i == limit {
        H::set_ip(if limit < len { rips[limit] } else { entry.end_rip });
    }
    H::retire(retired);
    i as u32
}

/// Look up, validate, decode and execute one block. `cache.len()` must be a
/// power of two.
pub unsafe fn cache_run<H: DecodeHost>(
    cache: &mut [DecEntry<H::Instr>],
    start_rip: u64,
    user: bool,
    gen: u32,
) -> Option<u32> {
    let slot = (start_rip.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & (cache.len() - 1);
    let entry = &mut cache[slot];
    if entry.valid && entry.tag == start_rip && entry.gen == gen && entry.user == user {
        // Empty instrs = negative entry: this RIP is not cacheable, so do not
        // retry or validate it.
        if entry.instrs.is_empty() {
            return None;
        }
        // Fast path: nothing wrote the block's physical page since it decoded.
        if entry.page_version == H::page_write_count(entry.phys_page) {
            let used = exec_decoded::<H>(entry);
            return if used > 0 { Some(used) } else { None };
        }
        // The page was written: a write may have missed the block, so re-check
        // the bytes before trusting the cached instructions.
        if H::validate_block(start_rip, &entry.bytes) {
            entry.page_version = H::page_write_count(entry.phys_page);
            let used = exec_decoded::<H>(entry);
            return if used > 0 { Some(used) } else { None };
        }
        entry.valid = false;
    }
    match H::decode_block_at(start_rip) {
        Some((instrs, rips, end_rip, bytes, phys_page)) =>
        {
            let limit = instrs
                .iter()
                .position(|i| !i.is_supported())
                .unwrap_or(instrs.len());
            if limit == 0 {
                *entry = DecEntry::negative(start_rip, gen, user);
                return None;
            }
            *entry = DecEntry {
                valid: true,
                tag: start_rip,
                gen,
                user,
                end_rip,
                limit,
                phys_page,
                page_version: H::page_write_count(phys_page),
                instrs,
                rips,
                bytes,
            };
            H::on_block_cached(phys_page);
            let used = exec_decoded::<H>(entry);
            if used > 0 { Some(used) } else { None }
        },
        _ =>
        {
            *entry = DecEntry::negative(start_rip, gen, user);
            None
        },
    }
}
