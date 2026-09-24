; Long-mode disk I/O: a 64-bit kernel reads one sector from the primary IDE
; channel using the ATA PIO protocol (28-bit LBA), as documented in ATA-6.

bits 64
org 0x100200

; select drive 0, LBA mode
mov dx, 0x1F6
mov al, 0xE0
out dx, al
mov dx, 0x1F7
.wait_bsy:
in al, dx
test al, 0x80
jnz .wait_bsy
; one sector, LBA 0
mov dx, 0x1F2
mov al, 1
out dx, al
mov dx, 0x1F3
xor al, al
out dx, al
mov dx, 0x1F4
xor al, al
out dx, al
mov dx, 0x1F5
xor al, al
out dx, al
; READ SECTORS
mov dx, 0x1F7
mov al, 0x20
out dx, al
; wait for DRQ
mov dx, 0x1F7
.wait_drq:
in al, dx
test al, 0x80
jnz .wait_drq
test al, 0x08
jz .wait_drq
; read 256 words into guest memory
mov dx, 0x1F0
mov rdi, 0xE000
mov ecx, 256
.read:
in ax, dx
mov [rdi], ax
add rdi, 2
dec ecx
jnz .read
hlt
