// x86-64 opcode-space coverage data.
//
// Source (fetched verbatim, then reduced to one category per opcode value):
//   https://raw.githubusercontent.com/torvalds/linux/master/arch/x86/lib/x86-opcode-map.txt
// That file is the Linux kernel's opcode map, which is generated from the
// opcode maps in Appendix A of the Intel SDM, Vol. 2 (Instruction Set
// Reference). See also the AMD64 Architecture Programmer's Manual, Vol. 3.
//
// Encoding of every table below: a string of 256 characters, one per opcode
// value, where
//   'V' - a defined instruction
//   'G' - defined, but the instruction depends on the ModRM.reg group escape
//   '.' - undefined / reserved / invalid in 64-bit mode
//
// The group escapes ('G') are resolved by the GrpTable entries in the same
// kernel file (not included here yet).

/* eslint-disable */
export const ONE_BYTE = "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGGGVVVVVVVVVVVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVGGVVVVVVVVGGGGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVVVGG";
export const TWO_BYTE = "GGVVVVVVVVVVVGVVVVVVVVVVGVVVGVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVVVGVVVVVVVVVVGGVVVVVVVVVVVVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV";
export const THREE_BYTE_38 = "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV.V.VVVVVVVVVVV..VVVVV.V...VVVVV.V...V...VVVV.VVVVVVVVVVVVVVV....VVVVVVVVVVVVVVVVVVVVVVVVVVVV..VVVVVVVVVVVV..VVVVVVVVVVVV....V.GGVVVVVV.V..VV....V.VVVVVVVVVVVVVVVVVVVVVVVVVG.VVVVVVVV...";
export const THREE_BYTE_3A = "VVVVVVVVVVVVVVVV....VVVVVVVV.VVVVVVV.VVV........VVVV....VVVV..VVVVVVV.V...VVV...VV..VVVV........VVVV..VV........VVVV..............................................................................V.........V.VV..............VV................V...............";

// Number of group tables (GrpTable) in the kernel map. Their per-reg
// sub-opcodes still have to be folded in; see the coverage script.
export const GROUP_TABLE_COUNT = 31;

// Iterate the defined opcodes of a table, calling fn(opcode, category).
export function forEachOpcode(table, fn)
{
    for(let i = 0; i < table.length; i++)
    {
        if(table[i] !== ".")
        {
            fn(i, table[i]);
        }
    }
}

export const TABLES = [
    { name: "one byte", prefix: [], table: ONE_BYTE },
    { name: "0f", prefix: [0x0F], table: TWO_BYTE },
    { name: "0f 38", prefix: [0x0F, 0x38], table: THREE_BYTE_38 },
    { name: "0f 3a", prefix: [0x0F, 0x3A], table: THREE_BYTE_3A },
];
