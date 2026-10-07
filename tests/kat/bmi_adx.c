// ADX/BMI2 QEMU reference vectors, including independent CF and OF carry chains.
#include "kat.h"

// ---- ADCX -------------------------------------------------------------------

// A multi-limb add built from ADCX. The carry-in to each limb comes from the
// previous limb's ADCX, so this walks CF as the chain state and re-establishes it
// with STC before folding in a propagated 1. A dropped or double-counted carry
// shows up immediately in the low limb.
static u64 addchain(const volatile u64 *a, const volatile u64 *b, u64 *out,
                   int limbs) {
  u64 carry = 0;
  for (int i = 0; i < limbs; i++) {
    // Only disturb CF when there is a carry to feed in. Issuing both STC and CLC
    // and letting the optimizer pick left CF permanently clear, which is what
    // the compiler actually did.
    if (carry) __asm__ volatile("stc" ::: "cc");
    else      __asm__ volatile("clc" ::: "cc");
    out[i] = kat_adcx(a[i], b[i]);
    carry = kat_get_flags_cf_of() & 1;
  }
  return carry;
}

// ---- runners ----------------------------------------------------------------

static volatile u64 PAIRS[][2] = {
    {0, 0},
    {1, 1},
    {0xffffffffffffffffULL, 1},
    {0xffffffffffffffffULL, 0xffffffffffffffffULL},
    {0x8000000000000000ULL, 0x8000000000000000ULL},
    {0x0123456789abcdefULL, 0xfedcba9876543210ULL},
    {0x7fffffffffffffffULL, 1},
    {0xfffffffffffffffeULL, 2},
    {0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL},
    {0x7fffffffffffffffULL, 0x7fffffffffffffffULL},
};

#define NPR (sizeof(PAIRS) / sizeof(PAIRS[0]))

// Read an operand through a barrier. Without this the optimizer can propagate the
// literal values from the table and constant-fold the instruction under test away
// at compile time, leaving a binary that passes while executing none of it.
static u64 op_a(unsigned i) {
  u64 v;
  kat_barrier();
  v = PAIRS[i][0];
  kat_barrier();
  return v;
}

static u64 op_b(unsigned i) {
  u64 v;
  kat_barrier();
  v = PAIRS[i][1];
  kat_barrier();
  return v;
}

static void test_addchain(void) {
  emit("== addchain ==\n");
  // The all-ones case must carry out of the top limb; the mixed cases must carry
  // at some limbs and not others.
  static volatile u64 cases[][6] = {
      {0xffffffffffffffffULL, 0xffffffffffffffffULL, 0x0000000000000000ULL,
       0x0000000000000000ULL, 0, 0},
      {0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL,
       1, 0, 0},
      {0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x0000000000000001ULL,
       0x0000000000000000ULL, 0, 0},
      {0, 0, 0, 0, 0, 0},
      {0x8000000000000000ULL, 0x8000000000000000ULL, 0xffffffffffffffffULL,
       0x5555555555555555ULL, 0, 0},
  };
  for (unsigned c = 0; c < 5; c++) {
    u64 a[3], b[3];
    kat_barrier();
    for (int k = 0; k < 3; k++) {
      a[k] = cases[c][k];
      b[k] = cases[c][k + 3];
    }
    kat_barrier();
    u64 out[3] = {0, 0, 0};
    u64 co = addchain(a, b, out, 3);
    emit("  addchain=");
    for (int i = 2; i >= 0; i--) emit_hex(out[i], 16);
    emit(" co=");
    emit_u64(co);
    emit("\n");
  }
}

// Establish CF and OF to known values. STC and CLC set CF without touching OF, so
// they are the right tool for CF. For OF, ADD on 0x7fffffff + 1 overflows signed
// but not unsigned, giving OF=1; 0 + 0 gives both clear. Done in that order so
// the STC/CLC is not overwritten by the ADD.
//
// An earlier version tried to set both from a single ADD. That cannot work: no
// single addition sets CF and OF independently in all four combinations, and the
// closest variant (adding 1 to zero when CF=1 was wanted) leaves CF clear,
// because 0 + 1 does not carry. Every ADCX case with cf=1 then disagreed with the
// reference, which looks exactly like a broken ADCX in the emulator.
#define KAT_SET_FLAGS(cf_, of_)                                             \
  do {                                                                      \
    if (of_) {                                                              \
      __asm__ volatile("movl $0x7fffffff, %%eax\n\t"                         \
                       "addl $1, %%eax"                                     \
                       ::: "eax", "cc");                                    \
    } else {                                                                \
      __asm__ volatile("xorl %%eax, %%eax\n\t"                              \
                       "addl %%eax, %%eax"                                 \
                       ::: "eax", "cc");                                    \
    }                                                                       \
    if (cf_) __asm__ volatile("stc" ::: "cc");                              \
    else    __asm__ volatile("clc" ::: "cc");                              \
  } while (0)

static void test_adcx(void) {
  emit("== adcx ==\n");
  // With CF clear, then with CF set: ADCX must add the incoming carry.
  for (int cf = 0; cf < 2; cf++) {
    emit(cf ? "  cf=1" : "  cf=0");
    emit("\n");
    for (unsigned i = 0; i < NPR; i++) {
      u64 a = op_a(i), b = op_b(i);
      KAT_SET_FLAGS(cf, 0);
      u64 r = kat_adcx(a, b);
      int co = (int)(kat_get_flags_cf_of() & 1);
      emit("    adcx[");
      emit_hex(a, 16);
      emit(",");
      emit_hex(b, 16);
      emit("]=");
      emit_hex(r, 16);
      emit(" co=");
      emit_u64((u64)co);
      emit("\n");
    }
  }
}

static void test_adox(void) {
  emit("== adox ==\n");
  for (int of = 0; of < 2; of++) {
    emit(of ? "  of=1" : "  of=0");
    emit("\n");
    for (unsigned i = 0; i < NPR; i++) {
      u64 a = op_a(i), b = op_b(i);
      KAT_SET_FLAGS(0, of);
      u64 r = kat_adox(a, b);
      int oo = (int)((kat_get_flags_cf_of() >> 1) & 1);
      emit("    adox[");
      emit_hex(a, 16);
      emit(",");
      emit_hex(b, 16);
      emit("]=");
      emit_hex(r, 16);
      emit(" oo=");
      emit_u64((u64)oo);
      emit("\n");
    }
  }
}

// The decisive test: CF and OF must be independent. If the emulator conflated
// them, or if one clobbered the other's input, every of=1 or cf=1 case would come
// out one too small.
//
// The flag setup, the instruction and the flag read all sit inside a *single* asm
// block on purpose. Written as separate statements, clang is free to hoist the
// setup out of the loop or reorder it against the ADOX, since it has no model of
// what those instructions do to the flags -- and it duly did, silently turning
// every of=1 case into an of=0 case, which looked exactly like a broken ADOX.
//
// The flags are set by editing EFLAGS directly on the stack rather than by
// arithmetic. Every arithmetic trick to set CF or OF has a trap: `addl %eax,%eax`
// with eax=1 computes 1+1 and sets no carry, so an ADCX case that wanted CF=1
// quietly ran with CF=0; that cost several rounds of false diagnosis before it
// was spotted. Reading and writing the flag word has no such edge cases, and it
// leaves the other flag untouched by construction -- which is exactly the
// property under test.
// Read the flag word, OR in the requested bits, write it back. Because this
// edits the word rather than deriving it from an addition, the other flag keeps
// whatever value it had -- which is the property under test.
#define KAT_PUSHF_SET(k_)                                                   \
  "pushfq\n\t"                                                            \
  "movl (%%rsp), %%eax\n\t"                                                \
  "orl " k_ ", %%eax\n\t"                                                   \
  "movl %%eax, (%%rsp)\n\t"                                                 \
  "popfq\n\t"

#define KAT_INDEP_CASE(a_, b_, cf_, of_)                                    \
  do {                                                                      \
    u64 r1_, r2_, d3_;                                                      \
    /* Declared u32 so clang emits a 32-bit register name: with a 64-bit        \
       constraint `orl %[k], %%eax` becomes `orl %rcx, %eax`, which is not a    \
       valid encoding. */                                                       \
    u32 k1_ = (u32)(cf_), k2_ = (u32)((u32)(of_) << 11);                     \
    u32 cf1_, of2_;                                                         \
    __asm__ volatile(                                                       \
        "movq %[av], %[d1]\n\t"                                              \
        "movq %[bv], %[d3]\n\t"                                              \
        /* OF := of, CF untouched */                                          \
        KAT_PUSHF_SET("%[k2]")                                              \
        /* CF := cf, OF untouched */                                          \
        KAT_PUSHF_SET("%[k1]")                                             \
        "adcx %[d3], %[d1]\n\t"   /* d1 += b + CF; CF := carry out */       \
        /* setcc writes one byte. Without the movzbl the C code reads the    \
           whole 32-bit output and picks up whatever the compiler next puts  \
           in that register -- which is how ten of these twenty lines read    \
           oo=3072 before the widen was added here. */                         \
        "setc %b[c1]\n\t"                                                    \
        "movzbl %b[c1], %[c1]\n\t"                                           \
        /* ADCX must have left OF alone; no re-establishment here, that is    \
           the point */                                                       \
        "movq %[av], %[d2]\n\t"                                              \
        "adox %[d3], %[d2]\n\t"                                              \
        "seto %b[o1]\n\t"                                                    \
        "movzbl %b[o1], %[o1]\n\t"                                           \
        : [d1] "=&r"(r1_), [d2] "=&r"(r2_), [d3] "=&r"(d3_),              \
          [c1] "=&q"(cf1_), [o1] "=&q"(of2_)                               \
        : [av] "r"((u64)(a_)), [bv] "r"((u64)(b_)),                        \
          [k1] "r"(k1_), [k2] "r"(k2_)                                    \
        : "cc", "rax", "rsp");                                               \
    emit("  pair[");                                                        \
    emit_hex((u64)(a_), 16);                                                 \
    emit(",");                                                              \
    emit_hex((u64)(b_), 16);                                                 \
    emit(",cf=");                                                           \
    emit_u64((u64)(cf_));                                                   \
    emit(",of=");                                                           \
    emit_u64((u64)(of_));                                                   \
    emit("] adcx=");                                                        \
    emit_hex(r1_, 16);                                                      \
    emit(" co=");                                                           \
    emit_u64((u64)cf1_);                                                    \
    emit(" adox=");                                                         \
    emit_hex(r2_, 16);                                                      \
    emit(" oo=");                                                           \
    emit_u64((u64)of2_);                                                    \
    emit("\n");                                                             \
  } while (0)

static void test_flag_independence(void) {
  emit("== adcx/adox flag independence ==\n");
  // Operands where an extra 1 in the incoming carry is visible in the result.
  static volatile u64 A[] = {0, 1, 0xffffffffffffffffULL, 0x0123456789abcdefULL,
                             0x7fffffffffffffffULL};
  static volatile u64 B[] = {0, 1, 1, 0xfedcba9876543210ULL, 1};
  for (unsigned i = 0; i < 5; i++) {
    kat_barrier();
    u64 a = A[i], b = B[i];
    kat_barrier();
    for (int cf = 0; cf < 2; cf++) {
      for (int of = 0; of < 2; of++) {
        KAT_INDEP_CASE(a, b, cf, of);
      }
    }
  }
}

static volatile u64 MUL_A[] = {
    0xffffffffffffffffULL, 0x8000000000000000ULL, 0x0123456789abcdefULL,
    0x0000000000000000ULL, 0xfedcba9876543210ULL, 0x7fffffffffffffffULL,
    0x00000000deadbeefULL, 0xffffffff00000000ULL,
};
static volatile u64 MUL_B[] = {
    0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xfedcba9876543210ULL,
    0xffffffffffffffffULL, 0x0000000000000001ULL, 0x8000000000000000ULL,
    0x0000000100000000ULL, 0x00000000ffffffffULL,
};

static u64 mul_a(unsigned i) {
  u64 v;
  kat_barrier();
  v = MUL_A[i];
  kat_barrier();
  return v;
}

static u64 mul_b(unsigned i) {
  u64 v;
  kat_barrier();
  v = MUL_B[i];
  kat_barrier();
  return v;
}

static void test_mulx(void) {
  emit("== mulx ==\n");
  for (unsigned i = 0; i < 8; i++) {
    u64 hi, lo, a = mul_a(i), b = mul_b(i);
    kat_mulx(a, b, &hi, &lo);
    emit("  mulx[");
    emit_hex(a, 16);
    emit(",");
    emit_hex(b, 16);
    emit("]=");
    emit_hex(hi, 16);
    emit(":");
    emit_hex(lo, 16);
    emit("\n");
  }
}

// MULX writes the high half to an explicit register rather than implicitly to RDX,
// and leaves the flags alone. Both matter: bignum code names the destination and
// relies on flags surviving.
static void test_mulx_flags(void) {
  emit("== mulx flags and dest ==\n");
  for (unsigned i = 0; i < 8; i++) {
    u64 hi, lo, a = mul_a(i), b = mul_b(i);
    KAT_SET_FLAGS(1, 1);   // both flags set beforehand
    kat_mulx(a, b, &hi, &lo);
    u64 f = kat_get_flags_cf_of();
    emit("  mulx[");
    emit_hex(a, 16);
    emit(",");
    emit_hex(b, 16);
    emit("] flags=");
    emit_u64(f);
    emit(" hi=");
    emit_hex(hi, 16);
    emit(" lo=");
    emit_hex(lo, 16);
    emit("\n");
  }
}

// The rotate count is an immediate, so it has to be written out literally. RSZ is
// only 6 bits wide, so 64, 65 and 127 must all behave as no rotation -- the case
// that catches an implementation which masks with 0x3F at the wrong end or
// forgets the mask.
//
// The operand comes from a volatile array rather than a literal, because a literal
// would let the optimizer evaluate RORX at compile time and the instruction would
// never run.
static volatile u64 RORX_V[5] = {
    0x0123456789abcdefULL, 0xffffffffffffffffULL, 0x8000000000000000ULL,
    1, 0x00000000ffffffffULL,
};

#define RORX_CASE(i, n)                                                     \
  do {                                                                      \
    kat_barrier();                                                          \
    u64 v_ = RORX_V[i];                                                     \
    kat_barrier();                                                          \
    emit("  rorx[");                                                        \
    emit_hex(v_, 16);                                                       \
    emit(",");                                                              \
    emit_u64(n);                                                            \
    emit("]=");                                                             \
    emit_hex(KAT_RORX_LIT(v_, n), 16);                                      \
    emit("\n");                                                             \
  } while (0)

#define RORX_ALL(i)                                                         \
  RORX_CASE(i, 0);                                                          \
  RORX_CASE(i, 1);                                                          \
  RORX_CASE(i, 7);                                                          \
  RORX_CASE(i, 8);                                                          \
  RORX_CASE(i, 31);                                                         \
  RORX_CASE(i, 32);                                                         \
  RORX_CASE(i, 63);                                                         \
  RORX_CASE(i, 64);                                                         \
  RORX_CASE(i, 65);                                                         \
  RORX_CASE(i, 127)

static void test_rorx(void) {
  emit("== rorx ==\n");
  RORX_ALL(0);
  RORX_ALL(1);
  RORX_ALL(2);
  RORX_ALL(3);
  RORX_ALL(4);
}

static void test_bmi2_misc(void) {
  emit("== bzhi/sarx/pext/pdep ==\n");
  static volatile u64 a[] = {0xffffffffffffffffULL, 0x0123456789abcdefULL, 0,
                             0x8000000000000000ULL};
  // BZHI's index is 6 bits wide, so a value above 63 leaves the destination
  // unchanged. SARX selects 32- or 64-bit operation from bit 5 of the second
  // operand, so both small and large masks have to be covered.
  static volatile u64 m[] = {32, 64, 0xffffffffffffffffULL, 17, 63, 0, 8};
  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 7; j++) {
      kat_barrier();
      u64 av = a[i], mv = m[j];
      kat_barrier();
      emit("  bzhi[");
      emit_hex(av, 16);
      emit(",");
      emit_hex(mv, 16);
      emit("]=");
      emit_hex(kat_bzhi(av, mv), 16);
      emit(" sarx[");
      emit_hex(av, 16);
      emit(",");
      emit_hex(mv, 16);
      emit("]=");
      emit_hex(kat_sarx((i64)av, mv), 16);
      emit("\n");
      emit("  pext[");
      emit_hex(av, 16);
      emit(",");
      emit_hex(mv, 16);
      emit("]=");
      emit_hex(kat_pext(av, mv), 16);
      emit(" pdep[");
      emit_hex(av, 16);
      emit(",");
      emit_hex(mv, 16);
      emit("]=");
      emit_hex(kat_pdep(av, mv), 16);
      emit("\n");
    }
  }
}

void _start(void) {
  kat_setup(kat_rdx(), kat_rsi());
  kat_emit_selfcheck();
  test_addchain();
  test_adcx();
  test_adox();
  test_flag_independence();
  test_mulx();
  test_mulx_flags();
  test_rorx();
  test_bmi2_misc();
  emit("== done ==\n");
  kat_output[kat_len] = 0;
  kat_finish(0);
  kat_exit(0);
}
