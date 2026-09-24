; Long-mode 9p: a 64-bit kernel speaks 9P2000.L over the virtio-9p device.
;
; It brings up the device like any virtio PCI device (find it, walk its
; capability list, negotiate VIRTIO_F_VERSION_1, set up queue 0), then runs a
; small 9P client: Tversion, Tattach, Twalk, Tlopen, Tread, Tclunk. Each request
; is one descriptor chain (device-readable message + device-writable reply).
;
; Results are left in memory for the harness to check.

bits 64
org 0x100200

; --- PCI scan for virtio-9p (device id 0x1049) ---
xor ebx, ebx
.scan:
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
shr eax, 16
cmp ax, 0x1049
je .found
inc ebx
cmp ebx, 32
jb .scan
mov dword [0xE000], 0xFFFFFFFF
hlt

.found:
mov [0xE000], ebx

; --- capability list: common (1), notify (2) ---
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
or eax, 0x34
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
and eax, 0xFF
mov esi, eax
.walk:
test esi, esi
jz .setup
mov edi, esi
and edi, 0xFC
call read_cfg
mov ecx, eax
and ecx, 0xFF
cmp ecx, 0x09
jne .next
mov ecx, eax
shr ecx, 24
and ecx, 0xFF
cmp ecx, 1
je .common
cmp ecx, 2
je .notify
jmp .next
.common:
call cap_port
mov r12d, eax
jmp .next
.notify:
call cap_port
mov r13d, eax
jmp .next
.next:
mov edi, esi
and edi, 0xFC
call read_cfg
shr eax, 8
and eax, 0xFF
mov esi, eax
jmp .walk

.setup:
; --- common configuration: reset, negotiate, set up queue 0 ---
mov edx, r12d
add edx, 0x14
xor al, al
out dx, al
mov al, 3
out dx, al
; driver features = VIRTIO_F_VERSION_1 (bit 32)
mov edx, r12d
add edx, 0x08
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x0C
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x08
mov eax, 1
out dx, eax
mov edx, r12d
add edx, 0x0C
mov eax, 1
out dx, eax
mov edx, r12d
add edx, 0x14
mov al, 0x0B
out dx, al
; queue 0
mov edx, r12d
add edx, 0x16
xor eax, eax
out dx, ax
mov edx, r12d
add edx, 0x18
mov eax, 8
out dx, ax
mov edx, r12d
add edx, 0x20
mov eax, 0x20000
out dx, eax
mov edx, r12d
add edx, 0x24
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x28
mov eax, 0x30000
out dx, eax
mov edx, r12d
add edx, 0x2C
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x30
mov eax, 0x31000
out dx, eax
mov edx, r12d
add edx, 0x34
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x1C
mov eax, 1
out dx, ax
mov edx, r12d
add edx, 0x14
mov al, 0x0F
out dx, al

; --- descriptor chain: request (out), reply (in) ---
mov rdi, 0x20000
mov qword [rdi + 0], 0x40000
mov dword [rdi + 8], 0
mov word [rdi + 12], 1
mov word [rdi + 14], 1
mov qword [rdi + 16], 0x41000
mov dword [rdi + 24], 8192
mov word [rdi + 28], 2
mov word [rdi + 30], 0

; --- rings ---
mov word [0x30000], 0
mov word [0x30002], 0
mov word [0x31000], 0
mov word [0x31002], 0

; ================= 1. Tversion =================
mov dword [0x3F000], 21
mov dword [0x40000], 21
mov byte [0x40004], 100
mov word [0x40005], 0
mov dword [0x40007], 8192
mov word [0x4000B], 8
mov dword [0x4000D], 0x30325039
mov dword [0x40011], 0x4C2E3030
call p9_submit
mov al, [0x41004]
mov [0xE010], al

; ================= 2. Tattach =================
mov dword [0x3F000], 27
mov dword [0x40000], 27
mov byte [0x40004], 104
mov word [0x40005], 0
mov dword [0x40007], 0
mov dword [0x4000B], 0xFFFFFFFF
mov word [0x4000F], 4
mov dword [0x40011], 0x746F6F72
mov word [0x40015], 0
mov dword [0x40017], 0
call p9_submit
mov al, [0x41004]
mov [0xE011], al

; ================= 3. Twalk "hello.txt" =================
mov dword [0x3F000], 28
mov dword [0x40000], 28
mov byte [0x40004], 110
mov word [0x40005], 0
mov dword [0x40007], 0
mov dword [0x4000B], 1
mov word [0x4000F], 1
mov word [0x40011], 9
mov dword [0x40013], 0x6C6C6568
mov dword [0x40017], 0x78742E6F
mov byte [0x4001B], 0x74
call p9_submit
mov al, [0x41004]
mov [0xE012], al

; ================= 4. Tlopen =================
mov dword [0x3F000], 15
mov dword [0x40000], 15
mov byte [0x40004], 12
mov word [0x40005], 0
mov dword [0x40007], 1
mov dword [0x4000B], 0
call p9_submit
mov al, [0x41004]
mov [0xE013], al

; ================= 5. Tread =================
mov dword [0x3F000], 23
mov dword [0x40000], 23
mov byte [0x40004], 116
mov word [0x40005], 0
mov dword [0x40007], 1
mov qword [0x4000B], 0
mov dword [0x40013], 512
call p9_submit
mov al, [0x41004]
mov [0xE014], al
mov eax, [0x41007]
mov [0xE020], eax

; ================= 6. Tclunk =================
mov dword [0x3F000], 11
mov dword [0x40000], 11
mov byte [0x40004], 120
mov word [0x40005], 0
mov dword [0x40007], 1
call p9_submit
mov al, [0x41004]
mov [0xE015], al

hlt

; Submits the request built in [0x40000] and waits for the used ring.
p9_submit:
mov eax, [0x3F000]
mov [0x20008], eax          ; desc0.len = request size
movzx eax, word [0x30002]
mov ecx, eax
and ecx, 7
mov word [0x30004 + ecx * 2], 0
inc eax
mov [0x30002], ax
; notify
mov edx, r12d
add edx, 0x1E
in ax, dx
movzx eax, ax
shl eax, 1
add eax, r13d
mov edx, eax
xor eax, eax
out dx, ax
; wait for the used ring to catch up
movzx ecx, word [0x30002]
.wait:
mov ax, [0x31002]
cmp ax, cx
jne .wait
ret

; Reads the config dword at offset edi of device ebx.
read_cfg:
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
or eax, edi
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
ret

; port = BAR[bar] + offset, for the capability at esi.
cap_port:
mov edi, esi
add edi, 4
and edi, 0xFC
call read_cfg
and eax, 0xFF
mov r9d, eax
mov edi, esi
add edi, 8
and edi, 0xFC
call read_cfg
mov r10d, eax
mov edi, r9d
shl edi, 2
add edi, 0x10
call read_cfg
and eax, 0xFFFFFFFC
add eax, r10d
ret
