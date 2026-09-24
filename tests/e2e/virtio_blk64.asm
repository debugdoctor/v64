; Long-mode virtio-blk: a 64-bit kernel drives the virtio-blk device the way
; the virtio 1.x specification describes.
;
;  1. It finds the virtio-blk PCI device (device id 0x1042) and walks the
;     vendor-specific capability list to locate the common, notification and
;     device configuration regions.
;  2. It reads the 64-bit capacity from the device configuration.
;  3. It resets the device, negotiates VIRTIO_F_VERSION_1, sets up virtqueue 0
;     (descriptor table, available ring, used ring), enables the queue and sets
;     DRIVER_OK.
;  4. It submits a single VIRTIO_BLK_T_IN request for sector 0: a device-readable
;     header, a device-writable data buffer and a device-writable status byte,
;     chained through the descriptor table.
;  5. It notifies the device and waits for the used ring.
;
; Results are left in memory for the harness to check.

bits 64
org 0x100200

; --- PCI scan for virtio-blk (device id 0x1042) ---
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
cmp ax, 0x1042
je .found
inc ebx
cmp ebx, 32
jb .scan
mov dword [0xE000], 0xFFFFFFFF
hlt

.found:
mov [0xE000], ebx

; --- capability list: common (1), notify (2), device (4) ---
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
cmp ecx, 4
je .device
jmp .next
.common:
call cap_port
mov r12d, eax
jmp .next
.notify:
call cap_port
mov r13d, eax
jmp .next
.device:
call cap_port
mov r14d, eax
; capacity: 64-bit little-endian sector count
mov edx, r14d
in eax, dx
mov [0xE004], eax
add edx, 4
in eax, dx
mov [0xE008], eax
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
out dx, al                  ; device_status = 0 (reset)
mov al, 3
out dx, al                  ; ACKNOWLEDGE | DRIVER
; driver features = VIRTIO_F_VERSION_1 (bit 32)
mov edx, r12d
add edx, 0x08
xor eax, eax
out dx, eax                 ; driver_feature_select = 0
mov edx, r12d
add edx, 0x0C
xor eax, eax
out dx, eax                 ; driver_feature (low) = 0
mov edx, r12d
add edx, 0x08
mov eax, 1
out dx, eax                 ; driver_feature_select = 1
mov edx, r12d
add edx, 0x0C
mov eax, 1
out dx, eax                 ; driver_feature (high) = 1
mov edx, r12d
add edx, 0x14
mov al, 0x0B
out dx, al                  ; FEATURES_OK
; queue 0
mov edx, r12d
add edx, 0x16
xor eax, eax
out dx, ax                  ; queue_select = 0
mov edx, r12d
add edx, 0x18
mov eax, 16
out dx, ax                  ; queue_size = 16
mov edx, r12d
add edx, 0x20
mov eax, 0x20000
out dx, eax                 ; queue_desc = 0x20000
mov edx, r12d
add edx, 0x24
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x28
mov eax, 0x30000
out dx, eax                 ; queue_avail = 0x30000
mov edx, r12d
add edx, 0x2C
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x30
mov eax, 0x31000
out dx, eax                 ; queue_used = 0x31000
mov edx, r12d
add edx, 0x34
xor eax, eax
out dx, eax
mov edx, r12d
add edx, 0x1C
mov eax, 1
out dx, ax                  ; queue_enable = 1
mov edx, r12d
add edx, 0x14
mov al, 0x0F
out dx, al                  ; DRIVER_OK

; --- descriptor chain: header (out), data (in), status (in) ---
mov rdi, 0x20000
mov qword [rdi + 0], 0x32000
mov dword [rdi + 8], 16
mov word [rdi + 12], 1      ; NEXT
mov word [rdi + 14], 1
mov qword [rdi + 16], 0x32100
mov dword [rdi + 24], 512
mov word [rdi + 28], 3      ; NEXT | WRITE
mov word [rdi + 30], 2
mov qword [rdi + 32], 0x32300
mov dword [rdi + 40], 1
mov word [rdi + 44], 2      ; WRITE
mov word [rdi + 46], 0

; --- request header: VIRTIO_BLK_T_IN, sector 0 ---
mov dword [0x32000], 0
mov dword [0x32004], 0
mov qword [0x32008], 0

; --- available ring: one head, descriptor 0 ---
mov word [0x30000], 0
mov word [0x30002], 1
mov word [0x30004], 0
; --- used ring ---
mov word [0x31000], 0
mov word [0x31002], 0

; --- notify queue 0: notify port + queue_notify_off * multiplier (2) ---
mov edx, r12d
add edx, 0x1E
in ax, dx
movzx eax, ax
shl eax, 1
add eax, r13d
mov edx, eax
xor eax, eax
out dx, ax

; --- wait for the device to consume the request ---
.wait:
mov ax, [0x31002]
cmp ax, 1
jne .wait
hlt

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
