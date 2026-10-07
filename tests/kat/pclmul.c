// Legacy PCLMULQDQ register, pointer and RIP-relative reference vectors.
// Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pclmulqdq

#include "kat.h"

#define OP0 (KAT_SCRATCH_ADDR + 0x000)
#define OP1 (KAT_SCRATCH_ADDR + 0x100)
#define RES (KAT_SCRATCH_ADDR + 0x300)

static u8 pbuf[64];

static void read_bytes(u64 addr, u8 *out, int n) {
  for (int i = 0; i < n; i++) out[i] = *(volatile u8 *)((u64)addr + (u64)i);
}

static void emit128(const u8 *p) {
  for (int i = 0; i < 16; i++) emit_hex(p[i], 2);
}

static void put4(u64 addr, u64 a, u64 b, u64 c, u64 d) {
  volatile u64 *q = (volatile u64 *)addr;
  q[0] = a; q[1] = b; q[2] = c; q[3] = d;
}

// Literal immediates and no-avx keep these on the legacy encoding.
#define CLMUL_CASE(imm_)                                                     \
  __attribute__((target("no-avx")))                                          \
  static void clmul_##imm_(u64 out, u64 in1, u64 in2) {                      \
    __asm__ volatile("movdqu (%1), %%xmm0\n\t"                               \
                     "movdqu (%2), %%xmm1\n\t"                               \
                     "pclmulqdq $" #imm_ ", %%xmm1, %%xmm0\n\t"              \
                     "movdqu %%xmm0, (%0)"                                   \
                     : : "r"(out), "r"(in1), "r"(in2)                        \
                     : "xmm0", "xmm1", "memory");                            \
  }

CLMUL_CASE(0x00)
CLMUL_CASE(0x01)
CLMUL_CASE(0x10)
CLMUL_CASE(0x11)

// Match the register cases: OP0 is the destination, OP1 the memory source.
#define CLMUL_MEM_CASE(imm_)                                                 \
  __attribute__((target("no-avx")))                                          \
  static void clmul_mem_##imm_(u64 out, u64 in1, u64 in2) {                  \
    __asm__ volatile("movdqu (%1), %%xmm0\n\t"                              \
                     "pclmulqdq $" #imm_ ", (%2), %%xmm0\n\t"              \
                     "movdqu %%xmm0, (%0)"                                   \
                     : : "r"(out), "r"(in1), "r"(in2)                        \
                     : "xmm0", "memory");                                    \
  }

CLMUL_MEM_CASE(0x00)
CLMUL_MEM_CASE(0x01)
CLMUL_MEM_CASE(0x10)
CLMUL_MEM_CASE(0x11)

struct clmul_fn { int imm; void (*run)(u64, u64, u64); const char *name; };

static const struct clmul_fn CASES[] = {
    {0x00, clmul_0x00, "imm00"},
    {0x01, clmul_0x01, "imm01"},
    {0x10, clmul_0x10, "imm10"},
    {0x11, clmul_0x11, "imm11"},
};

// The same four, with the second operand read from memory.
static const struct clmul_fn MEM_CASES[] = {
    {0x00, clmul_mem_0x00, "mem00"},
    {0x01, clmul_mem_0x01, "mem01"},
    {0x10, clmul_mem_0x10, "mem10"},
    {0x11, clmul_mem_0x11, "mem11"},
};

// Every qword distinct, so the four selections cannot agree by accident.
static const u64 A[][4] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL},
    {0xffffffffffffffffULL, 0x0000000000000000ULL, 0x8000000000000000ULL, 0x0000000000000001ULL},
    {0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x0f1e2d3c4b5a6978ULL, 0x8796a5b4c3d2e1f0ULL},
    {0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL},
    {0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL},
    {0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
};

static void test_pclmul(void) {
  emit("== pclmulqdq ==\n");
  for (unsigned v = 0; v < sizeof(A) / sizeof(A[0]); v++) {
    const u64 *b = A[(v * 3 + 1) % 8];
    put4(OP0, A[v][0], A[v][1], A[v][2], A[v][3]);
    put4(OP1, b[0], b[1], b[2], b[3]);
    for (unsigned k = 0; k < 4; k++) {
      CASES[k].run(RES, OP0, OP1);
      kat_case(100 + v * 4 + k);
      emit("  ");
      emit(CASES[k].name);
      emit("=");
      read_bytes(RES, pbuf, 16);
      emit128(pbuf);
      emit("\n");
    }
    for (unsigned k = 0; k < 4; k++) {
      // OP1 is the memory operand; the instruction reads it itself.
      MEM_CASES[k].run(RES, OP0, OP1);
      kat_case(300 + v * 4 + k);
      emit("  ");
      emit(MEM_CASES[k].name);
      emit("=");
      read_bytes(RES, pbuf, 16);
      emit128(pbuf);
      emit("\n");
    }
  }
}

__asm__(".section .rodata\n"
        ".balign 16\n"
        ".fill 16,1,0x7f\n"
        "rip_source:\n"
        ".quad 4,8\n"
        ".text\n");

// Monomial products x^a * x^b = x^(a+b); the sentinel detects an address off by one.
#define RIP_CASE(imm_, expected_)                                          \
  do {                                                                    \
    u64 input[2] = {1, 2}, output[2];                                      \
    for(int repeat = 0; repeat < 600; repeat++)                             \
    __asm__ volatile("movdqu %1, %%xmm8\n\t"                             \
                     "pclmulqdq $" #imm_ ", rip_source(%%rip), %%xmm8\n\t" \
                     "movdqu %%xmm8, %0"                                  \
                     : "=m"(output) : "m"(input) : "xmm8", "memory");     \
    const int ok = output[0] == expected_ && output[1] == 0;                \
    emit("  rip" #imm_ "=");                                              \
    emit(ok ? "ok" : "MISMATCH");                                         \
    emit(" got="); emit_hex(output[1], 16); emit_hex(output[0], 16);        \
    emit("\n");                                                            \
  } while(0)

__attribute__((target("no-avx")))
static void test_rip(void) {
  RIP_CASE(0x00, 4);
  RIP_CASE(0x01, 8);
  RIP_CASE(0x10, 8);
  RIP_CASE(0x11, 16);
}

__attribute__((force_align_arg_pointer))
void _start(void) {
  kat_setup(kat_rdx(), kat_rsi());
  kat_emit_selfcheck();
  test_pclmul();
  test_rip();
  emit("== done ==\n");
  kat_output[kat_len] = 0;
  kat_finish(0);
  kat_exit(0);
}
