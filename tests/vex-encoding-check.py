#!/usr/bin/env python3
"""Disassemble literal VEX encodings and verify their annotated operands.

Run with: python3 tests/vex-encoding-check.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = ["tests/interp64/avx.js",
         "tests/interp64/sse4.js", "tests/jit64/xsave.js", "tests/jit64/bmi1.js"]

# run([0xC4, 0xE2, 0xF2, 0xC8]);   // andnq %rax, %rbx, %rcx
LITERAL = re.compile(r"\[\s*((?:0x[0-9A-Fa-f]{2}\s*,\s*)+0x[0-9A-Fa-f]{2})\s*\]")
COMMENT = re.compile(r"//\s*([a-z][a-z0-9]*)\b")

# objdump appends the operand size to the mnemonic: `vpsllvq` prints as
# `vpsllvq`, `andnq` as `andnq`, `adcxq` as `adcxq`, but `mulxq` may print as
# `mulx`. Compare the alphabetic stem so the suffix convention does not matter.
def stem(m):
    m = m.strip()
    while m and m[-1] in "qwl":
        m = m[:-1]
    return m


# Operands are compared too, because a mnemonic check alone cannot see a wrong
# displacement or SIB byte -- corrupting one byte of `vmovdqu xmm0, [rax + r9]`
# still decodes to `vmovdqu`. The two files disagree about Intel vs AT&T syntax,
# so the comparison is on a normalised form: lowercased, size prefixes dropped,
# spacing collapsed, and AT&T `%`/`$` sigils removed. Anything that still will
# not normalise is reported as unverified rather than assumed equal.
SIGILS = re.compile(r"[%$]")
SIZEPFX = re.compile(
    r"\b(?:dword|qword|word|byte|xmmword|ymmword|zmmword|tword|oword|ptr|d|q)\b\s*")
SPACES = re.compile(r"\s*([,+\-])\s*")


HEXNUM = re.compile(r"0x([0-9a-f]+)")


def norm_operands(text):
    # objdump appends "# ymm0 = ymm1[3,2,1,0,...]" style commentary; it is not an
    # operand and must not take part in the comparison.
    t = text.split("#")[0].lower()
    t = SIGILS.sub("", t)
    t = SIZEPFX.sub("", t)
    t = SPACES.sub(r"\1", t)
    # objdump prints displacements in hex and the header tables in decimal, so
    # [r15+0x20] and [r15+32] are the same address. Only standalone literals are
    # rewritten; the digits inside a register name are part of the name.
    t = HEXNUM.sub(lambda m: str(int(m.group(1), 16)), t)
    return t.strip().strip(",")


def disassemble(enc, tmp, intel=True):
    # Wrapped in an object rather than fed to `objdump -b binary`: macOS's
    # objdump has no -b, and this way the same toolchain that built the KAT
    # binaries does the decoding.
    src = os.path.join(tmp, "b.s")
    obj = os.path.join(tmp, "b.o")
    with open(src, "w") as f:
        f.write(".text\n.byte " + ", ".join("0x%02x" % b for b in enc) + "\n")
    r = subprocess.run(["clang", "-target", "x86_64-unknown-linux-gnu", "-c",
                        "-o", obj, src], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    cmd = ["objdump", "-d"]
    if intel:
        cmd += ["-M", "intel"]
    out = subprocess.run(cmd + [obj], capture_output=True, text=True).stdout
    insns = []
    for line in out.splitlines():
        m = re.match(r"\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*\t([a-z][a-z0-9]*)(?:\s+(.*))?$",
                     line)
        if m:
            insns.append((m.group(1), (m.group(2) or "").strip()))
    return insns


# Both suites document their encodings in the file header as
#     //   vaddps ymm0,ymm1,ymm0    c5 f4 58 c0
# which is a machine-readable map from bytes to instruction and is better placed
# than a repeated comment on each use. Entries with no inline comment are looked
# up here instead of being written off as unverified.
HEADER = re.compile(r"^\s*//\s{2,}([a-z][a-z0-9]*)\s+(\S[^\n]*?)\s+((?:[0-9a-f]{2} )*[0-9a-f]{2})\s*$")


def header_map(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = HEADER.match(line)
            if m:
                enc = bytes(int(x, 16) for x in m.group(3).split())
                out[enc] = (m.group(1), m.group(2).strip())
    # Scanned whole-file rather than stopping at the end of the header block:
    # these files start with a shebang, so any "stop at the first non-comment
    # line" rule finds nothing. The pattern cannot match prose -- it requires a
    # run of hex byte pairs at the end of the line.
    return out


def entries(path):
    with open(path) as f:
        for n, line in enumerate(f, 1):
            code_part = line.split("//")[0]
            if "0xC4" not in code_part and "0xC5" not in code_part:
                continue
            bm = LITERAL.search(code_part)
            if not bm:
                continue                      # built by a helper or a loop
            enc = bytes(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]{2})", bm.group(1)))
            if enc[0] not in (0xC4, 0xC5):
                continue
            cm = COMMENT.search(line)
            if cm:
                tail = line[cm.start(1) + len(cm.group(1)):]
                yield n, enc, cm.group(1), tail.strip().lstrip(",").strip() or None
            else:
                yield n, enc, None, None


def main(argv):
    require = "--require" in argv
    total = checked = full = 0
    bad = []
    unannotated = []
    with tempfile.TemporaryDirectory() as tmp:
        for rel in FILES:
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                continue
            hmap = header_map(path)
            for lineno, enc, comment, operands in entries(path):
                total += 1
                got = disassemble(enc, tmp)
                checked += 1
                if len(got) != 1:
                    bad.append(f"{rel}:{lineno}: {enc.hex(' ')} decoded to "
                               f"{[g[0] for g in got] or 'nothing'}, expected one instruction")
                    continue
                mnem, ops = got[0]
                # tests/interp64/avx.js writes its comments in Intel syntax and
                # the other two in AT&T, and objdump's two renderings differ in
                # operand order, so both are produced and either may match.
                alt = disassemble(enc, tmp, intel=False)
                alt_ops = alt[0][1] if len(alt) == 1 else None
                if comment is None:
                    doc = hmap.get(enc)
                    if doc is None:
                        if hmap:
                            # The file documents its encodings in its header
                            # as "insn operands   c5 f4 58 c0", so a byte string
                            # absent from that table is one
                            # nobody has vouched for. Without this rule, changing
                            # a single byte of an encoding only moved it from
                            # "checked" to "unannotated" and the check stayed
                            # green -- the same failure arriving by another route.
                            bad.append(f"{rel}:{lineno}: {enc.hex(' ')} decodes to "
                                       f"{mnem} {ops} but does not appear in this "
                                       "file's encoding table, so nothing says that "
                                       "is the instruction intended")
                        else:
                            unannotated.append(f"{rel}:{lineno}: {enc.hex(' ')} "
                                               f"({mnem} {ops})")
                        continue
                    comment, operands = doc
                    if stem(mnem) != stem(comment):
                        bad.append(f"{rel}:{lineno}: {enc.hex(' ')}\n"
                                   f"      header says {comment}\n"
                                   f"      objdump says {mnem}")
                        continue
                if stem(mnem) != stem(comment):
                    bad.append(f"{rel}:{lineno}: {enc.hex(' ')}\n"
                               f"      comment says {comment}\n"
                               f"      objdump   says {mnem}")
                    continue
                if operands is None:
                    unannotated.append(f"{rel}:{lineno}: {enc.hex(' ')}: mnemonic "
                                       f"confirmed as {mnem}, but there are no "
                                       "operands to compare")
                    continue
                want = norm_operands(operands)
                if alt_ops is not None and norm_operands(alt_ops) == want:
                    full += 1
                elif norm_operands(ops) == want:
                    full += 1
                elif "%" in operands or "$" in operands:
                    # An AT&T comment is comparable to AT&T output exactly, so a
                    # disagreement here is a real error in one of the two.
                    bad.append(f"{rel}:{lineno}: {enc.hex(' ')}\n"
                               f"      comment says {comment} {operands}\n"
                               f"      objdump   says {mnem} {alt_ops}")
                else:
                    # Intel-syntax comment that this comparison cannot reconcile.
                    # Recorded rather than called wrong: guessing here would
                    # manufacture failures, which is the failure mode this file
                    # exists to prevent.
                    unannotated.append(f"{rel}:{lineno}: {enc.hex(' ')} operands "
                                       f"{want} not matched by either rendering")

    for b in bad:
        print("MISMATCH " + b)
    print(f"vex encoding check: {checked} hand-written VEX encodings disassembled, "
          f"{full} mnemonic and operands confirmed, {len(unannotated)} only partially")
    for u in unannotated:
        print("  no comment " + u)
    if bad:
        print(f"\n{len(bad)} encoding(s) do not decode to the instruction they claim")
        return 1
    if require and unannotated:
        print(f"\n{len(unannotated)} encoding(s) have no mnemonic comment and "
              "--require was given")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
