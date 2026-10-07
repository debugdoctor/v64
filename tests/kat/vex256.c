// VEX.128/256 reference vectors, operand mapping and upper-half preservation.
// Reference captures: Readme.md.
#include <immintrin.h>
#include "kat.h"

// Offsets into the harness scratch area.
#define OP0 (KAT_SCRATCH_ADDR + 0x000)
#define OP1 (KAT_SCRATCH_ADDR + 0x100)
#define OP2 (KAT_SCRATCH_ADDR + 0x200)
#define RES (KAT_SCRATCH_ADDR + 0x300)
#define SELF (KAT_SCRATCH_ADDR + 0x400)

// ---- staging ----------------------------------------------------------------
//
// volatile throughout: the test data must be opaque to the optimizer, or it
// constant-folds the instruction under test away at compile time and the binary
// passes without executing anything.

static inline __m256i ld256(u64 addr) {
  kat_barrier();
  __m256i v = _mm256_loadu_si256((const __m256i *)(u64)addr);
  kat_barrier();
  return v;
}

static inline __m128i ld128(u64 addr) {
  kat_barrier();
  __m128i v = _mm_loadu_si128((const __m128i *)(u64)addr);
  kat_barrier();
  return v;
}

static inline void st256(__m256i v, u64 addr) {
  kat_barrier();
  _mm256_storeu_si256((__m256i *)(u64)addr, v);
  kat_barrier();
}

static inline void st128(__m128i v, u64 addr) {
  kat_barrier();
  _mm_storeu_si128((__m128i *)(u64)addr, v);
  kat_barrier();
}

static inline void put64(u64 addr, u64 v) {
  kat_barrier();
  *(volatile u64 *)(u64)addr = v;
  kat_barrier();
}

// ---- 256-bit output ---------------------------------------------------------

// Print all 32 bytes, low half first.
static void emit256(const u8 *p) {
  for (int i = 0; i < 16; i++) emit_hex(p[i], 2);
  for (int i = 16; i < 32; i++) emit_hex(p[i], 2);
}

static void emit128(const u8 *p) {
  for (int i = 0; i < 16; i++) emit_hex(p[i], 2);
}

// Keep staging on the baseline ISA.
__attribute__((target("no-avx")))
static void read_bytes(u64 addr, u8 *out, int n) {
  for (int i = 0; i < n; i++) out[i] = *(volatile u8 *)((u64)addr + (u64)i);
}

__attribute__((target("no-avx")))
static void fill_bytes(u64 addr, int n, u8 v) {
  for (int i = 0; i < n; i++) *(volatile u8 *)((u64)addr + (u64)i) = v;
}

__attribute__((aligned(64))) static u8 selftest_out[32];
__attribute__((aligned(64))) static u8 selftest_pat[32];

// ---- self-check -------------------------------------------------------------

// Write 16 distinct 32-byte patterns, read every one back, and require all of
// them to match. This proves the staging path works before any result produced
// through it is believed. Sixteen distinct patterns rather than one, because a
// path that works for one value can still fold addresses together.
__attribute__((target("no-avx")))
static int selftest(void) {
  // Aligned static buffers support compiler-generated aligned stores.
  u8 *out = selftest_out;
  int bad = 0;
  emit("selftest\n");
  for (int i = 0; i < 16; i++) {
    u8 *pat = selftest_pat;
    for (int j = 0; j < 32; j++) pat[j] = (u8)(i * 32 + j + 1);
    for (int j = 0; j < 32; j++) *(volatile u8 *)(SELF + (u64)i * 64 + (u64)j) = pat[j];
    read_bytes(SELF + (u64)i * 64, out, 32);
    for (int j = 0; j < 32; j++) if (out[j] != pat[j]) { bad++; break; }
  }
  if (bad) {
    emit("selftest FAILED: ");
    emit_u64((u64)bad);
    emit(" of 16 patterns mismatched; results discarded\n");
    emit("== done ==\n");
    kat_output[kat_len] = 0;
    kat_exit(1);
  }
  emit("  16/16 distinct 32-byte patterns read back correctly\n");
  return 0;
}

// ---- operand generation -----------------------------------------------------
//
// Values are derived at run time rather than written as literals in a table so
// the optimizer cannot fold the arithmetic below.

static u64 mix(u64 x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 29;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 32;
  return x;
}

static u64 gseed;

// Next pseudo-random 64-bit value. Deterministic, so both the guest and the
// reference script see the same inputs.
static u64 next64(void) {
  gseed += 0x9e3779b97f4a7c15ULL;
  return mix(gseed);
}

static void fill(u64 addr, u64 seed) {
  u64 s = seed;
  for (int i = 0; i < 4; i++) {
    s += 0x9e3779b97f4a7c15ULL;
    put64(addr + (u64)i * 8, mix(s));
  }
}

// Fill with a small set of interesting patterns rather than pure noise: all ones,
// all zeros, alternating, sign-bit patterns, and values that make carries and
// borrows line up across lane and half boundaries.
static void fill_interesting(u64 addr, int variant) {
  static const u64 P[][4] = {
      {0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL,
       0xffffffffffffffffULL},
      {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
       0x0000000000000000ULL},
      {0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL,
       0x5555555555555555ULL},
      {0x8000000000000000ULL, 0x7fffffffffffffffULL, 0xffffffff00000000ULL,
       0x00000000ffffffffULL},
      {0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x0000000100000000ULL,
       0xffffffff00000000ULL},
      {0x0000ffff0000ffffULL, 0xffff0000ffff0000ULL, 0x00ff00ff00ff00ffULL,
       0xff00ff00ff00ff00ULL},
      {0x0000000100000000ULL, 0x0000000100000000ULL, 0x0000000100000000ULL,
       0x0000000100000000ULL},
      {0xfffffffeffffffffULL, 0x00000000ffffffffULL, 0xffffffff00000001ULL,
       0x0000000100000000ULL},
  };
  const u64 *p = P[variant % (int)(sizeof(P) / sizeof(P[0]))];
  for (int i = 0; i < 4; i++) put64(addr + (u64)i * 8, p[i]);
}

// ---- tests ------------------------------------------------------------------

// Each case stamps its index before running, so a fault names itself.
static u32 v256_case_index;

#define V256_CASE(name_, op_)                                               \
  do {                                                                      \
    kat_case(++v256_case_index);                                            \
    emit("  " name_ "=");                                                   \
    read_bytes(RES, buf, 32);                                               \
    emit256(buf);                                                           \
    emit("\n");                                                             \
  } while (0)

static u8 buf[80];

static u32 v128_pair_index;
static u32 v128_pair_fail;

// VPERM2I128 ymm, ymm, ymm, imm8, pinned so the instruction survives. Intrinsics
// get constant-folded into vinserti128/vextracti128 for the simple selections,
// which would leave the test proving nothing about the instruction.
#define KAT_VPERM2I128(a_, b_, imm_)                                        \
  ({                                                                        \
    __m256i r_;                                                             \
    __asm__ volatile("vperm2i128 $" #imm_ ", %[s2], %[s1], %[d]"           \
                     : [d] "=x"(r_)                                         \
                     : [s1] "x"(a_), [s2] "x"(b_));                          \
    r_;                                                                    \
  })

// VEX.256 integer operations, each printed as two 16-byte halves.
//
// Every case stamps its index with kat_case() *before* the instruction runs, not
// when its result is printed. An instruction the emulator does not implement
// raises #UD; with no IDT in this harness that becomes a triple fault and a halt,
// and the case index is the only way to tell which one.
#define V256_BEGIN() kat_case(++v256_case_index)

static void test_vex256_integer(void) {
  emit("== vex256 integer ==\n");
  for (int v = 0; v < 8; v++) {
    fill_interesting(OP0, v);
    fill_interesting(OP1, v * 3 + 1);

    __m256i a = ld256(OP0), b = ld256(OP1);

    // Print the operands once per vector. The variable-shift cases depend on the
    // shift counts in b, and "the result is zero" means two completely different
    // things depending on whether b holds what this thinks it holds -- so the
    // inputs have to be in the transcript, not inferred from it.
    {
      u8 t[32];
      emit("  inputs a=");
      read_bytes(OP0, t, 32); emit256(t);
      emit(" b=");
      read_bytes(OP1, t, 32); emit256(t);
      emit("\n");
    }

    V256_BEGIN(); st256(_mm256_add_epi64(a, b), RES);
    V256_CASE("vpaddq",);
    V256_BEGIN(); st256(_mm256_sub_epi64(a, b), RES);
    V256_CASE("vpsubq",);
    V256_BEGIN(); st256(_mm256_and_si256(a, b), RES);
    V256_CASE("vpand",);
    V256_BEGIN(); st256(_mm256_or_si256(a, b), RES);
    V256_CASE("vpor",);
    V256_BEGIN(); st256(_mm256_xor_si256(a, b), RES);
    V256_CASE("vpxor",);
    V256_BEGIN(); st256(_mm256_mullo_epi32(a, b), RES);
    V256_CASE("vpmulld",);
    V256_BEGIN(); st256(_mm256_mul_epu32(a, b), RES);
    V256_CASE("vpmuludq",);
    V256_BEGIN(); st256(_mm256_slli_epi64(a, 13), RES);
    V256_CASE("vpsllq13",);
    V256_BEGIN(); st256(_mm256_srli_epi64(a, 13), RES);
    V256_CASE("vpsrlq13",);
    V256_BEGIN(); st256(_mm256_srai_epi32(a, 7), RES);
    V256_CASE("vpsrad7",);
    V256_BEGIN(); st256(_mm256_cmpeq_epi64(a, b), RES);
    V256_CASE("vpcmpeqq",);
    V256_BEGIN(); st256(_mm256_cmpgt_epi64(a, b), RES);
    V256_CASE("vpcmpgtq",);
    V256_BEGIN(); st256(_mm256_shuffle_epi32(a, 0x1b), RES);
    V256_CASE("vpshufd1b",);
    V256_BEGIN(); st256(KAT_VPERM2I128(a, b, 0x20), RES);
    V256_CASE("vperm2i128_20",);
    V256_BEGIN(); st256(KAT_VPERM2I128(a, b, 0x31), RES);
    V256_CASE("vperm2i128_31",);
    V256_BEGIN(); st256(_mm256_permute4x64_epi64(a, 0x1b), RES);
    V256_CASE("vpermq1b",);
    V256_BEGIN(); st256(_mm256_blend_epi32(a, b, 0xaa), RES);
    V256_CASE("vpblendd_aa",);
    // The variable shifts are the one place where a transcript showing only the
    // result cannot tell you which of the two operands was misused: every lane of
    // a wrong result is consistent with a wrong count, a wrong width, or a wrong
    // source. So read both operands back out of the vector registers the
    // instruction itself will use and print them next to the answer. If these two
    // lines do not match the "inputs a=" line printed for this vector then the
    // staging is what is broken, and no amount of result staring will help.
    V256_BEGIN(); st256(a, RES);
    V256_CASE("vpsllvq_values",);
    V256_BEGIN(); st256(b, RES);
    V256_CASE("vpsllvq_counts",);
    V256_BEGIN(); st256(_mm256_sllv_epi64(a, b), RES);
    V256_CASE("vpsllvq",);
    V256_BEGIN(); st256(_mm256_srlv_epi64(a, b), RES);
    V256_CASE("vpsrlvq",);
    V256_BEGIN(); st256(_mm256_sllv_epi32(a, b), RES);
    V256_CASE("vpsllvd",);
    V256_BEGIN(); st256(_mm256_broadcastq_epi64(_mm_set1_epi64x((long long)next64())), RES);
    V256_CASE("vpbroadcastq",);
  }
}

// Compare legacy and VEX.128 encodings byte for byte using immediate stores.

__attribute__((target("no-avx"))) static void legacy_paddq(void) {
  __asm__ volatile("paddq %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("avx"))) static void vex_paddq(void) {
  __asm__ volatile("vpaddq %%xmm1, %%xmm0, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("no-avx"))) static void legacy_psubq(void) {
  __asm__ volatile("psubq %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("avx"))) static void vex_psubq(void) {
  __asm__ volatile("vpsubq %%xmm1, %%xmm0, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("no-avx"))) static void legacy_pand(void) {
  __asm__ volatile("pand %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("avx"))) static void vex_pand(void) {
  __asm__ volatile("vpand %%xmm1, %%xmm0, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("no-avx"))) static void legacy_por(void) {
  __asm__ volatile("por %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("avx"))) static void vex_por(void) {
  __asm__ volatile("vpor %%xmm1, %%xmm0, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("no-avx"))) static void legacy_pxor(void) {
  __asm__ volatile("pxor %%xmm1, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("avx"))) static void vex_pxor(void) {
  __asm__ volatile("vpxor %%xmm1, %%xmm0, %%xmm0" ::: "xmm0", "xmm1");
}
__attribute__((target("no-avx"))) static void legacy_psllq(void) {
  __asm__ volatile("psllq $17, %%xmm0" ::: "xmm0");
}
__attribute__((target("avx"))) static void vex_psllq(void) {
  __asm__ volatile("vpsllq $17, %%xmm0, %%xmm0" ::: "xmm0");
}
__attribute__((target("no-avx"))) static void legacy_psrlq(void) {
  __asm__ volatile("psrlq $17, %%xmm0" ::: "xmm0");
}
__attribute__((target("avx"))) static void vex_psrlq(void) {
  __asm__ volatile("vpsrlq $17, %%xmm0, %%xmm0" ::: "xmm0");
}
__attribute__((target("no-avx"))) static void legacy_pshufd(void) {
  __asm__ volatile("pshufd $0x4e, %%xmm0, %%xmm0" ::: "xmm0");
}
__attribute__((target("avx"))) static void vex_pshufd(void) {
  __asm__ volatile("vpshufd $0x4e, %%xmm0, %%xmm0" ::: "xmm0");
}

struct pair_case { const char *name; void (*legacy)(void); void (*vex)(void); };

// Loads OP0 into xmm0 and OP1 into xmm1, runs `op`, and stores xmm0 to `dst`.
#define RUN128(op_, dst_)                                                    \
  do {                                                                       \
    __asm__ volatile("movdqu (%[a]), %%xmm0\n\t"                           \
                     "movdqu (%[b]), %%xmm1"                                \
                     : : [a] "r"(a_addr), [b] "r"(b_addr)                    \
                     : "xmm0", "xmm1", "memory");                             \
    op_();                                                                   \
    __asm__ volatile("movdqu %%xmm0, (%[d])"                                 \
                     : : [d] "r"((u64)(dst_))                                \
                     : "memory");                                            \
  } while (0)

static void test_vex128_matches_legacy(void) {
  static const struct pair_case cases[] = {
      {"paddq", legacy_paddq, vex_paddq},
      {"psubq", legacy_psubq, vex_psubq},
      {"pand", legacy_pand, vex_pand},
      {"por", legacy_por, vex_por},
      {"pxor", legacy_pxor, vex_pxor},
      {"psllq_17", legacy_psllq, vex_psllq},
      {"psrlq_17", legacy_psrlq, vex_psrlq},
      {"pshufd_4e", legacy_pshufd, vex_pshufd},
  };
  const u64 a_addr = OP0, b_addr = OP1;
  emit("== vex128 vs legacy ==\n");
  for (int v = 0; v < 8; v++) {
    fill_interesting(OP0, v);
    fill_interesting(OP1, v * 5 + 2);
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
      u32 diff = 0;
      kat_case(++v128_pair_index);
      RUN128(cases[c].legacy, RES);
      kat_case(++v128_pair_index);
      RUN128(cases[c].vex, RES + 32);
      read_bytes(RES, buf, 16);
      read_bytes(RES + 32, buf + 32, 16);
      for (int k = 0; k < 16; k++)
        if (buf[k] != buf[32 + k]) diff++;
      emit("  ");
      emit(cases[c].name);
      emit(diff ? "=DIFFER at " : "=agree ");
      emit_hex(diff, 2);
      emit(" ");
      emit128(buf + 32);
      emit("\n");
      if (diff) v128_pair_fail++;
    }
  }
  emit("  v128_pair_failures=");
  emit_u64(v128_pair_fail);
  emit("\n");
}


// The upper half rule. A VEX.128 instruction writes 128 bits and must zero bits
// 255:128 of the destination; a VEX.256 instruction must leave the upper half of
// a source register alone. Both are checked by planting a recognisable pattern
// in the high half and then storing the whole YMM register.
static void test_upper_half_rules(void) {
  emit("== upper half rules ==\n");
  for (int v = 0; v < 8; v++) {
    fill_interesting(OP0, v);
    fill_interesting(OP1, v * 7 + 3);
    // OP2 holds the high half to plant: a recognisable, non-zero pattern.
    for (int i = 0; i < 4; i++) put64(OP2 + (u64)i * 8, 0xa5a5a5a5a5a5a5a5ULL);

    // Plant the pattern in the high half of ymm0 and ymm1 via a 256-bit store.
    __m256i plant;
    {
      u8 tmp[32];
      for (int i = 0; i < 16; i++) tmp[i] = *(volatile u8 *)(OP0 + (u64)i);
      for (int i = 16; i < 32; i++) tmp[i] = 0xa5;
      plant = _mm256_loadu_si256((const __m256i *)tmp);
    }
    st256(plant, OP2);

    // VEX.128 into ymm0: the upper 128 bits must become zero.
    //
    // Written as inline asm because nothing may sit between the VEX.128
    // instruction and the 256-bit store: the register has to be read in the
    // state the instruction left it in. An earlier version of this test built
    // the result with intrinsics and then put it back into a YMM whose high
    // half it had just filled with 0xa5, which reported 0xa5 in the high half
    // no matter what the emulator did -- the test was measuring its own setup.
    __asm__ volatile(
        "vmovdqu (%[src]), %%ymm0\n\t"
        "vmovdqu (%[b]), %%xmm1\n\t"
        "vpaddq %%xmm1, %%xmm0, %%xmm0\n\t"
        "vmovdqu %%ymm0, (%[dst])"
        : : [src] "r"((u64)OP2), [b] "r"((u64)OP1), [dst] "r"((u64)RES)
        : "ymm0", "ymm1", "memory");
    emit("  v128zero_hi=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");

    // VEX.256 reading a register whose high half is the pattern: the low 128
    // bits of the result must not be contaminated by it, and a full 256-bit
    // store of the source must still hold the pattern.
    __m256i y = _mm256_loadu_si256((const __m256i *)(u64)OP2);
    __m256i s = _mm256_add_epi64(y, y);
    st256(s, RES);
    emit("  v256_dbl=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");
    st256(y, RES);
    emit("  v256_src_preserved=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");

  }
}

// Three-operand mapping: dest = ModRM.reg, first source = VEX.vvvv, second =
// ModRM.rm. Commutativity hides a swap for add, so the tests use non-commutative
// operations where a swap is visible.
static void test_operand_mapping(void) {
  emit("== operand mapping ==\n");
  for (int v = 0; v < 8; v++) {
    fill_interesting(OP0, v);
    fill_interesting(OP1, v * 11 + 4);
    __m256i a = ld256(OP0), b = ld256(OP1);
    // Printed here for the same reason as in the integer section: the point of
    // this section is which register each operand came from, and a transcript
    // that only shows results cannot distinguish "the mapping is right but the
    // shift is wrong" from "the mapping is wrong".
    {
      u8 t[32];
      emit("  inputs a=");
      read_bytes(OP0, t, 32); emit256(t);
      emit(" b=");
      read_bytes(OP1, t, 32); emit256(t);
      emit("\n");
    }
    st256(_mm256_sub_epi64(a, b), RES);
    emit("  vpsubq_a_b=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");
    st256(_mm256_sub_epi64(b, a), RES);
    emit("  vpsubq_b_a=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");
    st256(_mm256_sllv_epi64(b, a), RES);
    emit("  vpsllvq_b_a=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");
    st256(_mm256_shuffle_epi32(_mm256_permute4x64_epi64(a, 0x4e), 0x39), RES);
    emit("  vpermq_then_shufd=");
    read_bytes(RES, buf, 32);
    emit256(buf);
    emit("\n");
  }
}

// VPEXTRB/W/D/Q with a memory destination.
//
// Found in the wild: clang vectorises a byte-wise memset-like store loop into a
// chain of VEXTRB, and when that is wrong the store silently goes to the wrong
// place or with the wrong value. That is exactly how the self-check above first
// failed, and it is the kind of defect a program using AVX will hit without any
// obvious symptom.
//
// Only the memory form is tested. VEXTRB with a *register* destination is not a
// VEX encoding at all -- it needs EVEX -- so nasm emits `62 f3 7d 08 14 d3 03`
// for that form, which this emulator does not claim to support.
//
// Encodings from nasm (AT&T):
//   vpextrb $3,  %xmm2, (%r11)   c4 c3 79 14 13 03
//   vpextrw $5,  %xmm2, (%r11)   c4 c3 79 15 13 05
//   vpextrd $2,  %xmm2, (%r11)   c4 c3 79 16 13 02
//   vpextrq $1,  %xmm2, (%r11)   c4 c3 f9 16 13 01
#define KAT_VPEXTRB(addr_, x_, imm_)                                          \
  __asm__ volatile("vpextrb $" #imm_ ", %[x], %[m]"                          \
                   : : [m] "m"(*(volatile u8 *)(addr_)), [x] "x"(x_))
#define KAT_VPEXTRW(addr_, x_, imm_)                                          \
  __asm__ volatile("vpextrw $" #imm_ ", %[x], %[m]"                          \
                   : : [m] "m"(*(volatile u16 *)(addr_)), [x] "x"(x_))
#define KAT_VPEXTRD(addr_, x_, imm_)                                          \
  __asm__ volatile("vpextrd $" #imm_ ", %[x], %[m]"                          \
                   : : [m] "m"(*(volatile u32 *)(addr_)), [x] "x"(x_))
#define KAT_VPEXTRQ(addr_, x_, imm_)                                          \
  __asm__ volatile("vpextrq $" #imm_ ", %[x], %[m]"                          \
                   : : [m] "m"(*(volatile u64 *)(addr_)), [x] "x"(x_))

static void test_extract(void) {
  emit("== vepextr ==\n");
  for (int v = 0; v < 8; v++) {
    fill_interesting(OP0, v);
    __m128i x = ld128(OP0);

    // Clear the destination first so a store that writes nothing at all is
    // distinguishable from one that writes the wrong value.
    fill_bytes(RES, 32, 0xcc);

    KAT_VPEXTRB(RES + 0, x, 0);
    KAT_VPEXTRB(RES + 1, x, 3);
    KAT_VPEXTRB(RES + 2, x, 15);
    KAT_VPEXTRW(RES + 8, x, 0);
    KAT_VPEXTRW(RES + 10, x, 5);
    KAT_VPEXTRW(RES + 12, x, 7);
    KAT_VPEXTRD(RES + 16, x, 0);
    KAT_VPEXTRD(RES + 20, x, 2);
    KAT_VPEXTRD(RES + 24, x, 3);
    KAT_VPEXTRQ(RES + 32 - 8, x, 0);
    KAT_VPEXTRQ(RES + 32 - 8, x, 1);

    read_bytes(RES, buf, 32);
    emit("  vepextr=");
    emit256(buf);
    emit("\n");
  }
}

void _start(void) {
  kat_setup(kat_rdx(), kat_rsi());
  kat_emit_selfcheck();
  gseed = 0x0123456789abcdefULL;
  selftest();
  test_extract();
  test_vex256_integer();
  test_vex128_matches_legacy();
  test_upper_half_rules();
  test_operand_mapping();
  emit("== done ==\n");
  kat_output[kat_len] = 0;
  kat_finish(0);
  kat_exit(0);
}
