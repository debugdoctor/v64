; End-to-end program for the long-mode JIT, assembled with nasm.
;
; Exercises the instruction coverage added in Phase 2 stage 5: 16-bit operands
; (66), 32-bit addressing (67), LOCK, memory-form shifts, CPUID and a 16-bit
; shift whose count exceeds the operand width. The expected results are derived
; independently in the test, not from this code's own output.

bits 64

        mov r12, 600            ; loop counter; the first 500 runs are
                                ; interpreted, the rest run compiled
.loop:
        ; 16-bit arithmetic must keep the upper 48 bits of the register.
        mov rax, -1
        mov ax, 0x1234
        add ax, 1
        mov r13, rax

        ; 16-bit memory add.
        mov word [rsp+0x20], 0xABCD
        add word [rsp+0x20], 1

        ; 32-bit addressing override: load through ebx, not rbx.
        mov ebx, 0x60000
        mov eax, [ebx]
        mov r15d, eax

        ; LOCK add plus a shift/unshift through memory.
        lock add qword [rsp+0x28], 1
        shl qword [rsp+0x28], 1
        shr qword [rsp+0x28], 1

        ; CPUID leaf 0 returns the vendor string in ebx/edx/ecx.
        xor eax, eax
        cpuid
        mov r8d, ebx
        mov r9d, ecx
        mov r10d, edx

        ; A 16-bit shift by more than the operand width clears the low half
        ; and must leave the upper bits intact.
        mov rax, -1
        mov ax, 0x1234
        mov ecx, 20
        shl ax, cl
        mov r14, rax

        dec r12d
        jnz .loop
        hlt
