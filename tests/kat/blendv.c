// Intel SDM Vol. 2: PBLENDVB, BLENDVPS, BLENDVPD.
// Each element is selected by its mask's sign bit; lower mask bits are ignored.
#include "kat.h"

static const u8 a[32] __attribute__((aligned(32))) = {
    0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,
    0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,
    0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55,
    0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55
};
static const u8 b[32] __attribute__((aligned(32))) = {
    0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,
    0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,
    0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,
    0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa,0xaa
};
static const u8 mask[32] __attribute__((aligned(32))) = {
    0,1,0x7f,0x80,4,5,0xff,7,8,0x80,10,0x7f,12,13,14,0xff,
    0x80,1,2,0x7f,4,5,6,0xff,8,9,10,0x80,12,13,14,0x7f
};
static u8 output[32];
static u32 failures;

static void check(const char *name, unsigned width, unsigned size) {
    unsigned bad = 0;
    for(unsigned i = 0; i < size; i++) {
        unsigned last = i / width * width + width - 1;
        u8 expected = mask[last] & 0x80 ? 0xaa : 0x55;
        bad += output[i] != expected;
    }
    emit("  "); emit(name); emit(bad ? "=MISMATCH\n" : "=ok\n");
    failures += !!bad;
}

#define LEGACY(insn_, width_)                                              \
    do {                                                                  \
        for(int repeat = 0; repeat < 600; repeat++)                        \
        __asm__ volatile("movdqu %1, %%xmm1\n\t"                          \
                         "movdqu %2, %%xmm2\n\t"                          \
                         "movdqu %3, %%xmm0\n\t"                          \
                         insn_ " %%xmm2, %%xmm1\n\t"                    \
                         "movdqu %%xmm1, %0"                             \
                         : "=m"(output) : "m"(a), "m"(b), "m"(mask)       \
                         : "xmm0", "xmm1", "xmm2", "memory");            \
        check(insn_, width_, 16);                                          \
    } while(0)

__attribute__((target("no-avx")))
static void legacy(void) {
    LEGACY("pblendvb", 1);
    LEGACY("blendvps", 4);
    LEGACY("blendvpd", 8);
}

#define VEX(insn_, width_)                                                 \
    do {                                                                  \
        for(int repeat = 0; repeat < 600; repeat++)                        \
        __asm__ volatile("vmovdqu %1, %%ymm1\n\t"                         \
                         "vmovdqu %2, %%ymm2\n\t"                         \
                         "vmovdqu %3, %%ymm3\n\t"                         \
                         insn_ " %%ymm3, %%ymm2, %%ymm1, %%ymm4\n\t"     \
                         "vmovdqu %%ymm4, %0"                            \
                         : "=m"(output) : "m"(a), "m"(b), "m"(mask)       \
                         : "ymm1", "ymm2", "ymm3", "ymm4", "memory");   \
        check(insn_, width_, 32);                                          \
    } while(0)

__attribute__((force_align_arg_pointer))
void _start(void) {
    kat_setup(kat_rdx(), kat_rsi());
    kat_emit_selfcheck();
    legacy();
    VEX("vpblendvb", 1);
    VEX("vblendvps", 4);
    VEX("vblendvpd", 8);
    emit("  failures="); emit_u64(failures); emit("\n== done ==\n");
    kat_output[kat_len] = 0;
    kat_finish(0);
    kat_exit(0);
}
