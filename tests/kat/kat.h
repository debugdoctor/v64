// Shared output buffer and instruction helpers for freestanding tests.
#ifndef KAT_H
#define KAT_H

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long long i64;
typedef int i32;

// Linux output and exit syscalls.

static inline long syscall3(long n, long a, long b, long c) {
  long r;
  __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c)
                   : "rcx", "r11", "memory");
  return r;
}

// Bare-metal passes a buffer address; Linux uses the mapped .bss buffer.
#define KAT_BARE_METAL_MAGIC 0x4B41543234364D47ULL   // "KAT64MG"
#define KAT_OUTPUT_SIZE 65536
#define KAT_WORK_SIZE   0x10000ULL
#define KAT_SCRATCH_OFF 0x10000ULL       // scratch area starts past the output
#define KAT_PROGRESS_OFF 0x10800ULL

// The buffer the guest uses when it did not get one handed to it. A plain .bss
// array is mapped by whatever loaded the program -- the ELF loader under Linux,
// the harness under bare metal -- so this works in both cases with no syscalls.
static volatile char kat_buf[KAT_OUTPUT_SIZE + KAT_WORK_SIZE] __attribute__((aligned(64)));

static volatile char *kat_base;
static int kat_bare_metal;

static inline u64 kat_rdx(void) {
  u64 v;
  __asm__ volatile("movq %%rdx, %0" : "=r"(v));
  return v;
}

static inline u64 kat_rsi(void) {
  u64 v;
  __asm__ volatile("movq %%rsi, %0" : "=r"(v));
  return v;
}

// Bare-metal: the harness passes a buffer address in %rsi together with the
// magic in %rdx, and reads the results back from the same place afterwards.
// Anything else means a real kernel loaded us, so the .bss array will do.
static void kat_setup(u64 magic, u64 given) {
  if (magic == KAT_BARE_METAL_MAGIC && given != 0) {
    kat_base = (volatile char *)given;
    kat_bare_metal = 1;
  } else {
    kat_base = kat_buf;
    kat_bare_metal = 0;
  }
}

#define kat_output ((volatile char *)kat_base)
// As an integer address, for the load/store helpers which take u64.
#define KAT_SCRATCH_ADDR ((u64)kat_base + KAT_SCRATCH_OFF)
#define KAT_SCRATCH (kat_base + KAT_SCRATCH_OFF)
#define KAT_PROGRESS_ADDR (kat_base + KAT_PROGRESS_OFF)

// The guest records the index of the case it is about to execute here. The
// harness has no IDT, so an unsupported instruction becomes a triple fault and
// halts with rip pointing at an arbitrary instruction; without this marker there
// is no way to tell which case died.
static inline void kat_case(u32 index) {
  *(volatile u32 *)KAT_PROGRESS_ADDR = index;
}

// `volatile` is load-bearing, not decoration. Nothing in this translation unit
// ever reads the buffer -- results are collected from guest memory after the
// guest halts -- so a plain array lets the optimizer delete every store as dead,
// which makes the computed results dead in turn, which lets it drop the inline
// asm that produced them. The binary then passes while executing none of the
// instructions under test.
static u32 kat_len;

// ---- output -----------------------------------------------------------------

// Under a real kernel the harness has to hand the buffer to stdout itself; under
// the bare-metal runner it must not, since there is no kernel to ask. Without
// this a KAT that halts early -- which is exactly what a failed self-check does
// -- prints nothing at all under qemu, and the failure is invisible.
static void kat_flush(void) {
  if(kat_bare_metal) return;
  syscall3(1L, 1L, (long)kat_base, (long)kat_len);
}

static void emit(const char *s) {
  while (*s) kat_output[kat_len++] = *s++;
}

static void emit_u64(u64 v) {
  char tmp[24];
  int n = 0;
  if (!v) tmp[n++] = '0';
  while (v) { tmp[n++] = '0' + (int)(v % 10); v /= 10; }
  while (n) kat_output[kat_len++] = tmp[--n];
}

static void emit_hex(u64 v, int digits) {
  static const char d[] = "0123456789abcdef";
  for (int i = (digits - 1) * 4; i >= 0; i -= 4) {
    kat_output[kat_len++] = d[(v >> i) & 0xf];
  }
}

// ---- harness self-check -----------------------------------------------------
//
// The output path is the one thing every result in this file travels through, so
// it gets checked before anything else runs. Each of 16 distinct values goes out
// through emit_u64 with a separator, and what lands in the buffer is read back
// and compared against an expectation built here, digit by digit -- not against
// a copy of emit_u64's own logic, and not against the host.
//
// A mismatch writes a marker and halts without a trailing "done" line, so the
// runner discards the whole run rather than reporting results that travelled
// through a broken buffer.
static void kat_emit_selfcheck(void) {
// Compare output with an independent literal.
  static const char want[] =
      "0 1 9 10 99 100 12345 4294967295 "
      "4294967296 9223372036854775807 18446744073709551615 "
      "255 256 1000000007 65536 42\n";
  static volatile u64 vals[16] = {
      0ULL, 1ULL, 9ULL, 10ULL, 99ULL, 100ULL, 12345ULL, 0xFFFFFFFFULL,
      0x100000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 255ULL,
      256ULL, 1000000007ULL, 65536ULL, 42ULL,
  };
  static volatile u32 kat_bad_at;
  u32 start = kat_len;
  for (int i = 0; i < 16; i++) {
    emit_u64(vals[i]);
    kat_output[kat_len++] = (i == 15) ? '\n' : ' ';
  }
  for (u32 i = 0; i < (u32)(sizeof(want) - 1); i++)
    if (kat_output[start + i] != want[i]) { kat_bad_at = i; goto bad; }
  if (kat_len != start + (u32)(sizeof(want) - 1)) { kat_bad_at = 1000; goto bad; }

  // emit_hex at the width the tests use, also against a literal.
  emit_hex(0x0123456789abcdefULL, 16);
  kat_output[kat_len++] = '\n';
  if (kat_len != start + (u32)sizeof(want) - 1 + 17) { kat_bad_at = 1001; goto bad; }
  {
    static const char w[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++)
      if (kat_output[start + (u32)sizeof(want) - 1 + i] != w[i]) { kat_bad_at = 1100 + i; goto bad; }
  }
  return;

bad:
  // Omit the completion marker so the runner rejects this output.
  emit("\nKAT-EMIT-SELFCHECK-FAILED at=");
  emit_u64((u64)kat_bad_at);
  emit("\n");
  kat_output[kat_len] = 0;
  kat_flush();
  __asm__ volatile("hlt");
}



#define kat_exit(code)                                                     \
  do {                                                                     \
    __asm__ volatile("movq %0, %%rdi; movl $60, %%eax; syscall" :: "r"((long)(code)) : "memory"); \
    __builtin_unreachable();                                              \
  } while (0)

// End of run. Under a kernel the buffer has already been handed to stdout, and
// HLT is privileged, so exiting is the only way to finish with a status the
// shell can read; the bare-metal runner has no kernel to ask and wants the halt
// so it can read the buffer back itself.
static void kat_finish(int code) {
  kat_flush();
  if(!kat_bare_metal) kat_exit(code);
  __asm__ volatile("hlt");
}

// ---- instruction access -----------------------------------------------------
//
// These are written as inline asm rather than compiler intrinsics on purpose.
// clang's *_u64 / *_di intrinsics are only expanded when the enclosing function
// carries a matching __attribute__((target(...))), and they are free to
// substitute an equivalent instruction sequence -- which is exactly what a known
// answer test must not allow. Inline asm pins the encoding, and the Makefile
// disassembles the result and fails if the expected instruction is absent.

// Keep the compiler from folding a constant result back through the asm. Every
// inline asm in this file that only writes flags or writes to a discarded
// register must be volatile for the same reason: without `volatile` clang may
// delete it, and a deleted flag-setting instruction turns the of=1 cases into
// of=0 cases that look exactly like a broken ADOX.
#define kat_barrier() __asm__ __volatile__("" ::: "memory")

// ADCX: dst += src + CF, and only CF is updated.
static inline u64 kat_adcx(u64 dst, u64 src) {
  __asm__ volatile("adcx %[s], %[d]" : [d] "+r"(dst) : [s] "r"(src) : "cc");
  return dst;
}

// ADOX: dst += src + OF, and only OF is updated.
static inline u64 kat_adox(u64 dst, u64 src) {
  __asm__ volatile("adox %[s], %[d]" : [d] "+r"(dst) : [s] "r"(src) : "cc");
  return dst;
}

// MULX uses implicit RDX; AT&T names the low destination before the high one.
static inline void kat_mulx(u64 a, u64 b, u64 *hi, u64 *lo) {
  u64 low, high;
  __asm__ volatile("mulx %[b], %[low], %[high]"
                   : [low] "=r"(low), [high] "=r"(high)
                   : [b] "r"(b), "d"(a));
  *hi = high;
  *lo = low;
}

// RORX encodes its rotate count as an immediate, so the count has to be baked
// into the template as a literal rather than passed as an operand. Callers pass
// a literal, which is also what makes the count genuinely an immediate in the
// encoding.
#define KAT_RORX_LIT(v, lit)                                                \
  ({                                                                        \
    u64 kat_rorx_tmp_ = (v);                                                \
    __asm__ volatile("rorx $" #lit ", %[a], %[a]"                          \
                     : [a] "+r"(kat_rorx_tmp_));                            \
    kat_rorx_tmp_;                                                          \
  })

static inline u64 kat_bzhi(u64 a, u64 b) {
  __asm__ volatile("bzhi %[b], %[a], %[a]" : [a] "+r"(a) : [b] "r"(b));
  return a;
}

static inline u64 kat_sarx(i64 a, u64 b) {
  u64 r;
  __asm__ volatile("sarx %[b], %[a], %[r]" : [r] "=r"(r) : [a] "r"(a), [b] "r"(b));
  return r;
}

static inline u64 kat_pext(u64 a, u64 b) {
  u64 r;
  __asm__ volatile("pext %[b], %[a], %[r]" : [r] "=r"(r) : [a] "r"(a), [b] "r"(b));
  return r;
}

// Read CF and OF into an ordinary value: bit 0 is CF, bit 1 is OF. SETcc needs a
// byte operand, so each flag goes to its own byte register.
static inline u64 kat_get_flags_cf_of(void) {
  u64 cf, of;
  __asm__ volatile("setc %b[cf]\n\t"
                   "seto %b[of]"
                   : [cf] "=&q"(cf), [of] "=&q"(of)
                   :
                   : "cc");
  return (cf & 1) | ((of & 1) << 1);
}

static inline u64 kat_pdep(u64 a, u64 b) {
  u64 r;
  __asm__ volatile("pdep %[b], %[a], %[r]" : [r] "=r"(r) : [a] "r"(a), [b] "r"(b));
  return r;
}

#endif
