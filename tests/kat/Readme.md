# Instruction reference tests

These freestanding x86-64 ELFs run under v64 or a Linux guest. Each test checks
its output buffer before executing instruction cases. The runner rejects an
incomplete transcript.

```sh
make -C tests/kat check
JIT=0 make -C tests/kat check
```

Building requires clang's x86-64 Linux target and an ELF linker. Override
`RUSTLD` to select a linker outside the stable Rust toolchain.

`bmi_adx.expected`, `vex256.expected`, and `pclmul.expected` are independent
QEMU captures. To regenerate them, run the same ELF in a Linux guest using
`qemu-system-x86_64 -cpu max` and capture its stdout, including the self-check.

`blendv` and the RIP-relative cases in `pclmul` check Intel SDM element-selection and carry-less
multiplication rules; their complete transcripts have also been compared with
QEMU. `vmware` checks fixed constants from the VMware backdoor protocol.
The Makefile verifies required instructions in the assembled binaries.

`make clean` removes generated files and preserves reference captures.
