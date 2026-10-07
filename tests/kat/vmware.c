// VMware backdoor protocol constants for port 0x5658; see src/vmware.ts.

#include "kat.h"

// From src/vmware.ts. Duplicated rather than included because that file is
// TypeScript driving a JS bus, not something a freestanding ELF can link.
#define VMWARE_PORT 0x5658
#define VMWARE_MAGIC 0x564D5868

#define CMD_GETSELLENGTH 6
#define CMD_GETVERSION 10
#define CMD_GETTIME 23

// Reads the backdoor. The command goes in ECX and the magic in EAX, both of
// which the device inspects; the reply comes back in EAX, and GETVERSION also
// writes EBX.
static u32 backdoor(u32 magic, u32 command) {
  u32 reply;
  __asm__ volatile("inl %%dx, %0" : "=a"(reply) : "d"((u16)VMWARE_PORT), "a"(magic), "c"(command));
  return reply;
}

struct reply { u32 eax, ebx, ecx; };

static struct reply backdoor_all(u32 magic, u32 command) {
  struct reply r;
  __asm__ volatile(
      "inl %%dx, %0"
      : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx)
      : "d"((u16)VMWARE_PORT), "a"(magic), "c"(command)
      : "memory");
  return r;
}

// A register the caller can check and the harness can predict. EBX is chosen
// because GETVERSION writes it, so it doubles as proof that the device ran.
#define CHK(name_, got_, want_)                                            \
  do {                                                                      \
    u32 g_ = (u32)(got_), w_ = (u32)(want_);                               \
    emit("  " name_ "=");                                                   \
    emit(g_ == w_ ? "ok" : "MISMATCH");                                     \
    emit(" got=");                                                          \
    emit_hex(g_, 8);                                                        \
    emit(" want=");                                                         \
    emit_hex(w_, 8);                                                        \
    emit("\n");                                                             \
    if (g_ != w_) bad++;                                                    \
  } while (0)

static u32 bad;

static void test_vmware(void) {
  struct reply r;

  emit("== vmware backdoor ==\n");
  bad = 0;

  // GETVERSION is the one command whose reply is fixed: six bytes, with the
  // magic echoed back in EBX so the caller can confirm it reached the hypervisor.
  kat_case(1);
  r = backdoor_all(VMWARE_MAGIC, CMD_GETVERSION);
  CHK("getversion_reply", r.eax, 6);
  kat_case(2);
  CHK("getversion_ebx", r.ebx, VMWARE_MAGIC);

  // A wrong magic word means "not a hypervisor port", which the device reports
  // as all ones. Without this case the magic check itself would be untested --
  // a device that answered 6 for any input would pass everything above.
  kat_case(3);
  CHK("bad_magic", backdoor(0, CMD_GETVERSION), 0xFFFFFFFF);
  kat_case(4);
  CHK("bad_magic_off_by_one", backdoor(VMWARE_MAGIC - 1, CMD_GETVERSION), 0xFFFFFFFF);
  kat_case(5);
  CHK("magic_high_bit_set", backdoor(VMWARE_MAGIC | 0x80000000u, CMD_GETVERSION),
      0xFFFFFFFF);

  // The magic must be compared as the full 32-bit word, so truncating the
  // command to 16 bits is not enough to select a command: ECX & 0xFFFF is what
  // the device switches on, and GETVERSION is 10.
  kat_case(6);
  CHK("getversion_upper_ecx_ignored",
      backdoor(VMWARE_MAGIC, CMD_GETVERSION | 0x10000u), 6);

  // GETTIME returns the host clock: EAX seconds since the epoch, EBX the
  // millisecond remainder, ECX the maximum lag in microseconds, EDX the host's
  // offset from UTC. The clock values move, so they are range-checked rather
  // than compared; ECX is a fixed constant in the protocol and is compared.
  kat_case(7);
  r = backdoor_all(VMWARE_MAGIC, CMD_GETTIME);
  CHK("gettime_is_not_an_error", r.eax != 0xFFFFFFFFu, 1);
  kat_case(8);
  CHK("gettime_max_lag_us", r.ecx, 1000000);
  kat_case(10);
  // EBX is the millisecond part, so it must be below 1000000. A device that
  // returned the whole microsecond count here, or the epoch seconds, would fail
  // this without needing the clock to be pinned.
  CHK("gettime_millis_in_range", r.ebx < 1000000u, 1);

  // An unknown command must not be answered as GETVERSION. 0xFFFF is not one of
  // the defined command numbers.
  kat_case(9);
  CHK("unknown_command", backdoor(VMWARE_MAGIC, 0xFFFF), 0xFFFFFFFF);

  emit("  failures=");
  emit_u64(bad);
  emit("\n");
}

void _start(void) {
  kat_setup(kat_rdx(), kat_rsi());
  kat_emit_selfcheck();
  test_vmware();
  emit("== done ==\n");
  kat_output[kat_len] = 0;
  kat_finish(0);
  kat_exit(0);
}
