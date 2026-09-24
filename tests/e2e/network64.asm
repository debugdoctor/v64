; Long-mode network bring-up: a 64-bit kernel enumerates PCI (mechanism #1)
; for a network-class device, then drives the DP8390/NE2000 through its
; documented register file and remote DMA to transmit one frame.

bits 64
org 0x100200

; Scan PCI bus 0 for a device whose class code is 0x02 (network controller).
xor ebx, ebx
.scan:
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
or eax, 0x08
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
shr eax, 24
cmp al, 0x02
je .found
inc ebx
cmp ebx, 32
jb .scan
mov dword [0xE000], 0xFFFFFFFF
hlt

.found:
mov [0xE000], ebx
; vendor and device id (config offset 0x00)
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
mov [0xE004], eax
; BAR0 (config offset 0x10)
mov eax, 0x80000000
mov ecx, ebx
shl ecx, 11
or eax, ecx
or eax, 0x10
mov dx, 0xCF8
out dx, eax
mov dx, 0xCFC
in eax, dx
and eax, 0xFFFFFFFC
mov [0xE008], eax
mov esi, eax

; DP8390: reset, then a transmit-only init.
mov dx, si
add dx, 0x1F
in al, dx
mov dx, si
mov al, 0x21
out dx, al
mov dx, si
add dx, 0x0E
mov al, 0x49
out dx, al
mov dx, si
add dx, 0x0C
xor al, al
out dx, al
mov dx, si
add dx, 0x0D
xor al, al
out dx, al
mov dx, si
add dx, 0x04
mov al, 0x40
out dx, al
mov dx, si
add dx, 0x05
mov al, 60
out dx, al
mov dx, si
add dx, 0x06
xor al, al
out dx, al
mov dx, si
add dx, 0x08
xor al, al
out dx, al
mov dx, si
add dx, 0x09
mov al, 0x40
out dx, al
mov dx, si
add dx, 0x0A
mov al, 60
out dx, al
mov dx, si
add dx, 0x0B
xor al, al
out dx, al
; remote DMA write of the frame into the transmit page
mov dx, si
mov al, 0x12
out dx, al
mov dx, si
add dx, 0x10
mov rdi, frame
mov ecx, 30
.write:
mov ax, [rdi]
out dx, ax
add rdi, 2
dec ecx
jnz .write
; transmit
mov dx, si
mov al, 0x04
out dx, al
hlt

frame:
db 0x02, 0x00, 0x00, 0x00, 0x00, 0x01 ; destination MAC
db 0x02, 0x00, 0x00, 0x00, 0x00, 0x02 ; source MAC
db 0x08, 0x00                         ; ethertype IPv4
db "v64!"
times 60 - ($ - frame) db 0
