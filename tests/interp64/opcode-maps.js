// x86-64 opcode-space coverage data.
//
// Source: https://raw.githubusercontent.com/torvalds/linux/master/arch/x86/lib/x86-opcode-map.txt
//   the Linux kernel's opcode map (thanks to its authors), generated from the
//   opcode maps in Intel SDM Vol. 2 Appendix A
//
// Each table is 256 characters, one per opcode value:
//   V  defined instruction      G  ModRM.reg group escape      .  undefined
//
// The two VEX/EVEX-encoded sets of the 0F/0F38/0F3A maps are annotated inline
// in that file (mnemonics such as VMOVUPS) and are not listed separately here.

export const ONE_BYTE = "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGGGVVVVVVVVVVVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVGGVVVVVVVVGGGGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVVVGG";
export const TWO_BYTE = "GGVVVVVVVVVVVGVVVVVVVVVVGVVVGVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVGGVVVVVVGVVVVVVVVVVGGVVVVVVVVVVVVGVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV";
export const THREE_BYTE_38 = "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV.V.VVVVVVVVVVV..VVVVV.V...VVVVV.V...V...VVVV.VVVVVVVVVVVVVVV....VVVVVVVVVVVVVVVVVVVVVVVVVVVV..VVVVVVVVVVVV..VVVVVVVVVVVV....V.GGVVVVVV.V..VV....V.VVVVVVVVVVVVVVVVVVVVVVVVVG.VVVVVVVV...";
export const THREE_BYTE_3A = "VVVVVVVVVVVVVVVV....VVVVVVVV.VVVVVVV.VVV........VVVV....VVVV..VVVVVVV.V...VVV...VV..VVVV........VVVV..VV........VVVV..............................................................................V.........V.VV..............VV................V...............";
export const EVEX_MAP4 = "VVVV....VVVV....VVVV....VVVV....VVVVV...VVVVV...VVVV....VVVV....VVVVVVVVVVVVVVVV................VV...VV..V.V....................GG.GVV..V......V.....................V.......V.V................GG..............GGGG............................VVV.VVGGVV....GG";
export const EVEX_MAP5 = "................VV...........V............V.VVVV.................................V......VVVVVVVV..............V.........VVVVVVV.................................................................................................................................";
export const EVEX_MAP6 = "...................V........................VV....................VV........VVVV......VV..............................................................VVVVVVVVVV......VVVVVVVVVV......VVVVVVVVVV......................VV........................................";
export const XOP_MAP8 = ".....................................................................................................................................VVV......VV.....VVV......VV..VV..V...............V.........VVVV........VVVV............................VVVV................";
export const XOP_MAP9 = ".GG...............G.............................................................................................................VVVV............VVVVVVVVVVVV.....................................VVV..VV...V.....VVV..VV...V.....VVV............................";
export const XOP_MAPA = "................V.G.............................................................................................................................................................................................................................................";

// prefix: the bytes before the 256-entry table's opcode byte. The EVEX prefix
// selects the map in P0's low bits, the XOP prefix in byte 2.
export const TABLES = [
    { name: "one byte", prefix: [], table: ONE_BYTE },
    { name: "0f", prefix: [0x0F], table: TWO_BYTE },
    { name: "0f 38", prefix: [0x0F, 0x38], table: THREE_BYTE_38 },
    { name: "0f 3a", prefix: [0x0F, 0x3A], table: THREE_BYTE_3A },
    { name: "evex map 4", prefix: [0x62, 0xF0, 0x7C, 0x48], table: EVEX_MAP4 },
    { name: "evex map 5", prefix: [0x62, 0xF1, 0x7C, 0x48], table: EVEX_MAP5 },
    { name: "evex map 6", prefix: [0x62, 0xF2, 0x7C, 0x48], table: EVEX_MAP6 },
    { name: "xop map 8", prefix: [0x8F, 0xE8, 0x78], table: XOP_MAP8 },
    { name: "xop map 9", prefix: [0x8F, 0xE9, 0x78], table: XOP_MAP9 },
    { name: "xop map a", prefix: [0x8F, 0xEA, 0x78], table: XOP_MAPA },
];

export function for_each_opcode(table, fn)
{
    for(let i = 0; i < table.length; i++)
    {
        if(table[i] !== ".")
        {
            fn(i, table[i]);
        }
    }
}
