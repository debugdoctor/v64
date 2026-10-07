# Freestanding instruction tests; see Readme.md for reference provenance.
RUST_SYSROOT := $(shell rustup run stable rustc --print sysroot)
RUST_HOST := $(shell rustup run stable rustc -vV | sed -n 's/^host: //p')
RUSTLD ?= $(RUST_SYSROOT)/lib/rustlib/$(RUST_HOST)/bin/rust-lld

CC = clang
CFLAGS = -target x86_64-unknown-linux-gnu -nostdlib -static -O2 -ffreestanding \
         -mavx2 -mbmi2 -madx -maes -mpclmul -msse4.2 \
         -fno-stack-protector -fno-builtin -Wall -Wno-unused-function
LDFLAGS = -e _start --no-dynamic-linker -z noexecstack

# Tests compared with QEMU captures.
TESTS = bmi_adx vex256 pclmul
# Tests with expectations checked by the guest.
SELFTESTS = vmware blendv
BINS  = $(TESTS:%=%.elf) $(SELFTESTS:%=%.elf)

.PHONY: all check clean
all: $(BINS)

%.o: %.c kat.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Verify that the assembler emitted the required instructions.
bmi_adx.must = adcxq adoxq mulxq rorxq bzhiq sarxq pextq pdepq
pclmul.must  = pclmulqdq
blendv.must = pblendvb blendvps blendvpd vpblendvb vblendvps vblendvpd
vmware.must  = inl
# Require both legacy and VEX encodings.
vex256.must  = vpaddq vpsubq vpand vpor vpxor vpmulld vpmuludq vpsllq vpsrlq \
               vpsrad vpcmpeqq vpcmpgtq vpshufd vperm2i128 vpermq vpblendd \
               vpsllvq vpsrlvq vpsllvd vpbroadcastq \
               paddq psubq pand por pxor psllq psrlq pshufd

%.elf: %.o
	$(RUSTLD) -flavor gnu -o $@ $< $(LDFLAGS)
	@echo "  [verify] $@"
	@for m in $($*.must); do \
	  objdump -d $@ | grep -qw "$$m" \
	    || { echo "  $@ does not contain $$m" >&2; exit 1; }; \
	done

check: all
	@for t in $(TESTS); do \
	  WASM_PATH=../../build/v64.wasm \
	  NODE_OPTIONS="--import ../../tools/ts-register.mjs" \
	  node ./run_kat.mjs $$t.elf > $$t.out || exit 1; \
	  if diff -u $$t.expected $$t.out > $$t.diff; then \
	    echo "ok   $$t ($$(wc -l < $$t.out | tr -d ' ') lines)"; \
	  else \
	    echo "FAIL $$t"; cat $$t.diff; exit 1; \
	  fi; \
	done
	@for t in $(SELFTESTS); do \
	  WASM_PATH=../../build/v64.wasm \
	  NODE_OPTIONS="--import ../../tools/ts-register.mjs" \
	  node ./run_kat.mjs $$t.elf > $$t.out || exit 1; \
	  n=$$(grep -c '=MISMATCH' $$t.out); \
	  f=$$(sed -n 's/^  failures=//p' $$t.out); \
	  if [ "$$n" != "0" ] || [ "$$f" != "0" ]; then \
	    echo "FAIL $$t: $$n mismatched case(s), failures=$$f"; \
	    grep -E '=MISMATCH|failures=' $$t.out; exit 1; \
	  fi; \
	  echo "ok   $$t ($$f failures, $$(($$(wc -l < $$t.out) )) lines)"; \
	done

# Preserve reference captures.
clean:
	rm -f *.o *.elf *.out *.diff
