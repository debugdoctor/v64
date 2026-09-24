import { h } from "./lib.js";
import { dbg_assert, dbg_log } from "./log.js";


// https://www.kernel.org/doc/Documentation/x86/boot.txt

const LINUX_BOOT_HDR_SETUP_SECTS = 0x1F1;
const LINUX_BOOT_HDR_SYSSIZE = 0x1F4;
const LINUX_BOOT_HDR_VIDMODE = 0x1FA;
const LINUX_BOOT_HDR_BOOT_FLAG = 0x1FE;
const LINUX_BOOT_HDR_HEADER = 0x202;
const LINUX_BOOT_HDR_VERSION = 0x206;
const LINUX_BOOT_HDR_TYPE_OF_LOADER = 0x210;
const LINUX_BOOT_HDR_LOADFLAGS = 0x211;
const LINUX_BOOT_HDR_CODE32_START = 0x214;
const LINUX_BOOT_HDR_RAMDISK_IMAGE = 0x218;
const LINUX_BOOT_HDR_RAMDISK_SIZE = 0x21C;
const LINUX_BOOT_HDR_HEAP_END_PTR = 0x224;
const LINUX_BOOT_HDR_CMD_LINE_PTR = 0x228;
const LINUX_BOOT_HDR_INITRD_ADDR_MAX = 0x22C;
const LINUX_BOOT_HDR_KERNEL_ALIGNMENT = 0x230;
const LINUX_BOOT_HDR_RELOCATABLE_KERNEL = 0x234;
const LINUX_BOOT_HDR_MIN_ALIGNMENT = 0x235;
const LINUX_BOOT_HDR_XLOADFLAGS = 0x236;
const LINUX_BOOT_HDR_CMDLINE_SIZE = 0x238;
const LINUX_BOOT_HDR_PAYLOAD_OFFSET = 0x248;
const LINUX_BOOT_HDR_PAYLOAD_LENGTH = 0x24C;
const LINUX_BOOT_HDR_PREF_ADDRESS = 0x258;
const LINUX_BOOT_HDR_INIT_SIZE = 0x260;

const LINUX_BOOT_HDR_CHECKSUM1 = 0xAA55;
const LINUX_BOOT_HDR_CHECKSUM2 = 0x53726448;

// struct boot_params: acpi_rsdp_addr
const BOOT_PARAMS_ACPI_RSDP_ADDR = 0x070;
// struct boot_params: e820_entries and e820_table
const BOOT_PARAMS_E820_ENTRIES = 0x1E8;
const BOOT_PARAMS_E820_TABLE = 0x2D0;

// ACPI tables we synthesize for direct boot (no firmware): placed in the
// 0xE0000..0xFFFFF region Linux also scans for the RSDP.
const ACPI_RSDP_ADDRESS = 0xF0000;
const ACPI_RSDT_ADDRESS = 0xF0040;
const ACPI_XSDT_ADDRESS = 0xF0080;
const ACPI_MADT_ADDRESS = 0xF0100;
const ACPI_HPET_ADDRESS = 0xF0200;
const ACPI_DSDT_ADDRESS = 0xF0300;
const ACPI_FADT_ADDRESS = 0xF0400;
const APIC_MEM_ADDRESS = 0xFEE00000;
const IOAPIC_MEM_ADDRESS = 0xFEC00000;

const LINUX_BOOT_HDR_TYPE_OF_LOADER_NOT_ASSIGNED = 0xFF;

const LINUX_BOOT_HDR_LOADFLAGS_LOADED_HIGH = 1 << 0;
const LINUX_BOOT_HDR_LOADFLAGS_QUIET_FLAG = 1 << 5;
const LINUX_BOOT_HDR_LOADFLAGS_KEEP_SEGMENTS = 1 << 6;
const LINUX_BOOT_HDR_LOADFLAGS_CAN_USE_HEAPS = 1 << 7;


// Synthesize the ACPI tables a 64-bit kernel needs to enumerate the CPU and
// the interrupt controllers. Direct boot has no firmware, so RSDP, RSDT, XSDT
// and MADT are built here and the RSDP address is published in boot_params.
function build_acpi_tables(mem8)
{
    const write8 = (address, value) => { mem8[address] = value & 0xFF; };
    const write32 = (address, value) => {
        mem8[address] = value & 0xFF;
        mem8[address + 1] = value >> 8 & 0xFF;
        mem8[address + 2] = value >> 16 & 0xFF;
        mem8[address + 3] = value >> 24 & 0xFF;
    };
    /** @param {number} address @param {bigint} value */
    const write64 = (address, value) => {
        for(let i = 0; i < 8; i++) mem8[address + i] = Number(value >> BigInt(i * 8) & 0xFFn);
    };
    const write_string = (address, text) => {
        for(let i = 0; i < text.length; i++) mem8[address + i] = text.charCodeAt(i);
    };
    const checksum = (address, length) => {
        let sum = 0;
        for(let i = 0; i < length; i++) sum = (sum + mem8[address + i]) & 0xFF;
        return -sum & 0xFF;
    };

    // A table header followed by its body, with the checksum filled in.
    const table = (address, signature, revision, body) =>
    {
        const length = 36 + body.length;
        write_string(address, signature);
        write32(address + 4, length);
        write8(address + 8, revision);
        write8(address + 9, 0);
        write_string(address + 10, "v64   ");   // OEM ID
        write_string(address + 16, "v64ACPI "); // OEM table ID
        write32(address + 24, 1);               // OEM revision
        write_string(address + 28, "v64 ");     // creator ID
        write32(address + 32, 1);               // creator revision
        for(let i = 0; i < body.length; i++) mem8[address + 36 + i] = body[i];
        write8(address + 9, checksum(address, length));
    };

    // MADT: one local APIC and one I/O APIC. ISA IRQs are identity-mapped, so
    // no interrupt source overrides are needed.
    const madt = [];
    const push32 = value => madt.push(value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF);
    push32(APIC_MEM_ADDRESS); // local APIC address
    push32(1);                // flags: PCAT_COMPAT
    madt.push(0, 8, 0, 0, 1, 0, 0, 0); // Processor Local APIC, APIC ID 0, enabled
    madt.push(1, 12, 0, 0);            // I/O APIC
    push32(IOAPIC_MEM_ADDRESS);
    push32(0);                         // GSI base
    table(ACPI_MADT_ADDRESS, "APIC", 5, madt);

    // HPET table, pointing at the emulated timer at 0xFED00000.
    const hpet = [];
    const push32h = value => hpet.push(value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF);
    push32h(0x8086A201);    // event timer block ID
    hpet.push(0, 64, 0, 0); // generic address: memory, 64-bit, offset 0, access size 0
    push32h(0xFED00000);    // base address (low)
    push32h(0);             // base address (high)
    hpet.push(0);           // HPET number
    hpet.push(0, 0);        // minimum tick
    hpet.push(0);           // page protection
    table(ACPI_HPET_ADDRESS, "HPET", 1, hpet);

    // A minimal DSDT (empty AML) and a FADT that declares hardware-reduced
    // ACPI, which needs no SMI/SCI/PM blocks. Linux requires both to use ACPI.
    table(ACPI_DSDT_ADDRESS, "DSDT", 2, []);

    const fadt = new Array(208).fill(0);
    const set16 = (offset, value) => { fadt[offset - 36] = value & 0xFF; fadt[offset - 36 + 1] = value >> 8 & 0xFF; };
    const set32 = (offset, value) => {
        fadt[offset - 36] = value & 0xFF;
        fadt[offset - 36 + 1] = value >> 8 & 0xFF;
        fadt[offset - 36 + 2] = value >> 16 & 0xFF;
        fadt[offset - 36 + 3] = value >> 24 & 0xFF;
    };
    /** @param {number} offset @param {bigint} value */
    const set64 = (offset, value) => {
        for(let i = 0; i < 8; i++) fadt[offset - 36 + i] = Number(value >> BigInt(i * 8) & 0xFFn);
    };
    set32(0x28, ACPI_DSDT_ADDRESS);          // DSDT
    set16(0x2E, 9);                          // SCI interrupt
    set32(0x70, 1 << 20);                    // flags: HW_REDUCED_ACPI
    set64(0x8C, BigInt(ACPI_DSDT_ADDRESS));  // X_DSDT
    table(ACPI_FADT_ADDRESS, "FACP", 6, fadt);

    // RSDT (32-bit pointers) and XSDT (64-bit pointers) list the tables.
    const pointers = [ACPI_MADT_ADDRESS, ACPI_FADT_ADDRESS, ACPI_HPET_ADDRESS];
    const rsdt = [];
    for(const pointer of pointers)
    {
        rsdt.push(pointer & 0xFF, pointer >> 8 & 0xFF, pointer >> 16 & 0xFF, pointer >> 24 & 0xFF);
    }
    table(ACPI_RSDT_ADDRESS, "RSDT", 1, rsdt);

    const xsdt = [];
    /** @param {bigint} value */
    const push64 = value => { for(let i = 0; i < 8; i++) xsdt.push(Number(value >> BigInt(i * 8) & 0xFFn)); };
    for(const pointer of pointers) push64(BigInt(pointer));
    table(ACPI_XSDT_ADDRESS, "XSDT", 1, xsdt);

    // RSDP, revision 2 (with an XSDT), 36 bytes.
    write_string(ACPI_RSDP_ADDRESS, "RSD PTR ");
    write8(ACPI_RSDP_ADDRESS + 8, 0);
    write_string(ACPI_RSDP_ADDRESS + 9, "v64   ");
    write8(ACPI_RSDP_ADDRESS + 15, 2);
    write32(ACPI_RSDP_ADDRESS + 16, ACPI_RSDT_ADDRESS);
    write32(ACPI_RSDP_ADDRESS + 20, 36);
    write64(ACPI_RSDP_ADDRESS + 24, BigInt(ACPI_XSDT_ADDRESS));
    for(let i = 32; i < 36; i++) write8(ACPI_RSDP_ADDRESS + i, 0);
    write8(ACPI_RSDP_ADDRESS + 8, checksum(ACPI_RSDP_ADDRESS, 20));
    write8(ACPI_RSDP_ADDRESS + 32, checksum(ACPI_RSDP_ADDRESS, 36));

    return ACPI_RSDP_ADDRESS;
}

export function load_kernel(mem8, bzimage, initrd, cmdline)
{
    dbg_log("Trying to load kernel of size " + bzimage.byteLength);

    const KERNEL_HIGH_ADDRESS = 0x100000;

    // Put the initrd at the 64 MB boundary. This means the minimum memory size
    // is 64 MB plus the size of the initrd.
    // Note: If set too low, kernel may fail to load the initrd with "invalid magic at start of compressed archive"
    const INITRD_ADDRESS = 64 << 20;

    const quiet = false;

    const bzimage8 = new Uint8Array(bzimage);
    const bzimage16 = new Uint16Array(bzimage);
    const bzimage32 = new Uint32Array(bzimage);

    const setup_sects = bzimage8[LINUX_BOOT_HDR_SETUP_SECTS] || 4;
    const syssize = bzimage32[LINUX_BOOT_HDR_SYSSIZE >> 2] << 4;

    const vidmode = bzimage16[LINUX_BOOT_HDR_VIDMODE >> 1];

    const checksum1 = bzimage16[LINUX_BOOT_HDR_BOOT_FLAG >> 1];
    if(checksum1 !== LINUX_BOOT_HDR_CHECKSUM1)
    {
        dbg_log("Bad checksum1: " + h(checksum1));
        return;
    }

    // Not aligned, so split into two 16-bit reads
    const checksum2 =
        bzimage16[LINUX_BOOT_HDR_HEADER >> 1] |
        bzimage16[LINUX_BOOT_HDR_HEADER + 2 >> 1] << 16;
    if(checksum2 !== LINUX_BOOT_HDR_CHECKSUM2)
    {
        dbg_log("Bad checksum2: " + h(checksum2));
        return;
    }

    const protocol = bzimage16[LINUX_BOOT_HDR_VERSION >> 1];
    dbg_assert(protocol >= 0x202); // older not supported by us

    const flags = bzimage8[LINUX_BOOT_HDR_LOADFLAGS];
    dbg_assert(flags & LINUX_BOOT_HDR_LOADFLAGS_LOADED_HIGH); // low kernels not supported by us

    // we don't relocate the kernel, so we don't care much about most of these

    const flags2 = bzimage16[LINUX_BOOT_HDR_XLOADFLAGS >> 1];
    const initrd_addr_max = bzimage32[LINUX_BOOT_HDR_INITRD_ADDR_MAX >> 2];
    const kernel_alignment = bzimage32[LINUX_BOOT_HDR_KERNEL_ALIGNMENT >> 2];
    const relocatable_kernel = bzimage8[LINUX_BOOT_HDR_RELOCATABLE_KERNEL];
    const min_alignment = bzimage8[LINUX_BOOT_HDR_MIN_ALIGNMENT];
    const cmdline_size = protocol >= 0x206 ? bzimage32[LINUX_BOOT_HDR_CMDLINE_SIZE >> 2] : 255;
    const payload_offset = bzimage32[LINUX_BOOT_HDR_PAYLOAD_OFFSET >> 2];
    const payload_length = bzimage32[LINUX_BOOT_HDR_PAYLOAD_LENGTH >> 2];
    const pref_address = bzimage32[LINUX_BOOT_HDR_PREF_ADDRESS >> 2];
    const pref_address_high = bzimage32[LINUX_BOOT_HDR_PREF_ADDRESS + 4 >> 2];
    const init_size = bzimage32[LINUX_BOOT_HDR_INIT_SIZE >> 2];

    dbg_log("kernel boot protocol version: " + h(protocol));
    dbg_log("flags=" + h(flags) + " xflags=" + h(flags2));
    dbg_log("code32_start=" + h(bzimage32[LINUX_BOOT_HDR_CODE32_START >> 2]));
    dbg_log("initrd_addr_max=" + h(initrd_addr_max));
    dbg_log("kernel_alignment=" + h(kernel_alignment));
    dbg_log("relocatable=" + relocatable_kernel);
    dbg_log("min_alignment=" + h(min_alignment));
    dbg_log("cmdline max=" + h(cmdline_size));
    dbg_log("payload offset=" + h(payload_offset) + " size=" + h(payload_length));
    dbg_log("pref_address=" + h(pref_address_high) + ":" + h(pref_address));
    dbg_log("init_size=" + h(init_size));

    const real_mode_segment = 0x8000;
    const base_ptr = real_mode_segment << 4;

    const heap_end = 0xE000;
    const heap_end_ptr = heap_end - 0x200;

    // fill in the kernel boot header with infos the kernel needs to know

    bzimage8[LINUX_BOOT_HDR_TYPE_OF_LOADER] = LINUX_BOOT_HDR_TYPE_OF_LOADER_NOT_ASSIGNED;

    const new_flags =
        (quiet ? flags | LINUX_BOOT_HDR_LOADFLAGS_QUIET_FLAG : flags & ~LINUX_BOOT_HDR_LOADFLAGS_QUIET_FLAG)
        & ~LINUX_BOOT_HDR_LOADFLAGS_KEEP_SEGMENTS
        | LINUX_BOOT_HDR_LOADFLAGS_CAN_USE_HEAPS;
    bzimage8[LINUX_BOOT_HDR_LOADFLAGS] = new_flags;

    bzimage16[LINUX_BOOT_HDR_HEAP_END_PTR >> 1] = heap_end_ptr;

    // should parse the vga=... paramter from cmdline here, but we don't really care
    bzimage16[LINUX_BOOT_HDR_VIDMODE >> 1] = 0xFFFF; // normal

    dbg_log("heap_end_ptr=" + h(heap_end_ptr));

    cmdline += "\x00";
    dbg_assert(cmdline.length < cmdline_size);

    const cmd_line_ptr = base_ptr + heap_end;
    dbg_log("cmd_line_ptr=" + h(cmd_line_ptr));

    bzimage32[LINUX_BOOT_HDR_CMD_LINE_PTR >> 2] = cmd_line_ptr;
    for(let i = 0; i < cmdline.length; i++)
    {
        mem8[cmd_line_ptr + i] = cmdline.charCodeAt(i);
    }

    const prot_mode_kernel_start = (setup_sects + 1) * 512;
    dbg_log("prot_mode_kernel_start=" + h(prot_mode_kernel_start));

    const real_mode_kernel = new Uint8Array(bzimage, 0, prot_mode_kernel_start);
    const protected_mode_kernel = new Uint8Array(bzimage, prot_mode_kernel_start);

    let ramdisk_address = 0;
    let ramdisk_size = 0;

    if(initrd)
    {
        ramdisk_address = INITRD_ADDRESS;
        ramdisk_size = initrd.byteLength;

        dbg_assert(KERNEL_HIGH_ADDRESS + protected_mode_kernel.length < ramdisk_address);

        mem8.set(new Uint8Array(initrd), ramdisk_address);
    }

    bzimage32[LINUX_BOOT_HDR_RAMDISK_IMAGE >> 2] = ramdisk_address;
    bzimage32[LINUX_BOOT_HDR_RAMDISK_SIZE >> 2] = ramdisk_size;

    dbg_assert(base_ptr + real_mode_kernel.length < 0xA0000);

    mem8.set(real_mode_kernel, base_ptr);
    mem8.set(protected_mode_kernel, KERNEL_HIGH_ADDRESS);

    return {
        name: "genroms/kernel.bin",
        data: make_linux_boot_rom(real_mode_segment, heap_end),
    };
}

// Direct x86-64 boot (Linux 64-bit boot protocol). Sets up boot_params, an
// identity page table, a GDT and the kernel/initrd, then jumps to the kernel
// entry in long mode. See Documentation/x86/boot.rst §64-bit BOOT PROTOCOL.
export function load_kernel64(mem8, bzimage, initrd, cmdline)
{
    const KERNEL_ADDRESS = 0x100000;
    const INITRD_ADDRESS = 64 << 20;
    const ZERO_PAGE = 0x10000;
    const PML4 = 0x1000;
    const PDPT = 0x2000;
    const PD = 0x3000;
    const GDT = 0x4000;
    const CMDLINE_ADDRESS = 0x80000;

    const bzimage8 = new Uint8Array(bzimage);
    const bzimage16 = new Uint16Array(bzimage);
    const bzimage32 = new Uint32Array(bzimage);

    const checksum1 = bzimage16[LINUX_BOOT_HDR_BOOT_FLAG >> 1];
    dbg_assert(checksum1 === LINUX_BOOT_HDR_CHECKSUM1, "load_kernel64: bad boot flag");

    const checksum2 =
        bzimage16[LINUX_BOOT_HDR_HEADER >> 1] |
        bzimage16[LINUX_BOOT_HDR_HEADER + 2 >> 1] << 16;
    dbg_assert(checksum2 === LINUX_BOOT_HDR_CHECKSUM2, "load_kernel64: bad HdrS");

    const protocol = bzimage16[LINUX_BOOT_HDR_VERSION >> 1];
    dbg_assert(protocol >= 0x20C, "load_kernel64: kernel too old for 64-bit boot");

    const setup_sects = bzimage8[LINUX_BOOT_HDR_SETUP_SECTS] || 4;
    const flags = bzimage8[LINUX_BOOT_HDR_LOADFLAGS] & ~LINUX_BOOT_HDR_LOADFLAGS_KEEP_SEGMENTS
        | LINUX_BOOT_HDR_LOADFLAGS_CAN_USE_HEAPS;
    const cmdline_size = bzimage32[LINUX_BOOT_HDR_CMDLINE_SIZE >> 2] || 255;

    const write16 = (address, value) => {
        mem8[address] = value & 0xFF;
        mem8[address + 1] = value >> 8 & 0xFF;
    };
    const write32 = (address, value) => {
        mem8[address] = value & 0xFF;
        mem8[address + 1] = value >> 8 & 0xFF;
        mem8[address + 2] = value >> 16 & 0xFF;
        mem8[address + 3] = value >> 24 & 0xFF;
    };
    /** @param {number} address @param {bigint} value */
    const write64 = (address, value) => {
        for(let i = 0; i < 8; i++) mem8[address + i] = Number(value >> BigInt(i * 8) & 0xFFn);
    };

    // Identity-map the first 1 GiB with 2 MiB pages. The user bit lets a kernel
    // return to ring 3; ring 0 is not restricted by it.
    write64(PML4, BigInt(PDPT) | 7n);
    write64(PDPT, BigInt(PD) | 7n);
    for(let i = 0; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x87n);
    }

    // GDT: null, 64-bit code (0x10, L=1), 64-bit data (0x18)
    write64(GDT, 0n);
    write64(GDT + 8, 0x00AF9A000000FFFFn);
    write64(GDT + 16, 0x00CF92000000FFFFn);

    // boot_params: copy the setup header, then fill the fields a boot loader owns.
    const header_start = LINUX_BOOT_HDR_SETUP_SECTS;
    const header_end = 0x202 + bzimage8[0x201];
    for(let i = header_start; i < header_end; i++)
    {
        mem8[ZERO_PAGE + i] = bzimage8[i];
    }
    mem8[ZERO_PAGE + LINUX_BOOT_HDR_TYPE_OF_LOADER] = LINUX_BOOT_HDR_TYPE_OF_LOADER_NOT_ASSIGNED;
    mem8[ZERO_PAGE + LINUX_BOOT_HDR_LOADFLAGS] = flags;
    write16(ZERO_PAGE + LINUX_BOOT_HDR_VIDMODE, 0xFFFF); // normal
    write32(ZERO_PAGE + LINUX_BOOT_HDR_CODE32_START, KERNEL_ADDRESS);

    cmdline += "\x00";
    dbg_assert(cmdline.length < cmdline_size, "load_kernel64: command line too long");
    for(let i = 0; i < cmdline.length; i++)
    {
        mem8[CMDLINE_ADDRESS + i] = cmdline.charCodeAt(i);
    }
    write32(ZERO_PAGE + LINUX_BOOT_HDR_CMD_LINE_PTR, CMDLINE_ADDRESS);

    let ramdisk_address = 0;
    let ramdisk_size = 0;
    if(initrd)
    {
        ramdisk_address = INITRD_ADDRESS;
        ramdisk_size = initrd.byteLength;
        mem8.set(new Uint8Array(initrd), ramdisk_address);
    }
    write32(ZERO_PAGE + LINUX_BOOT_HDR_RAMDISK_IMAGE, ramdisk_address);
    write32(ZERO_PAGE + LINUX_BOOT_HDR_RAMDISK_SIZE, ramdisk_size);

    const prot_mode_kernel_start = (setup_sects + 1) * 512;
    const protected_mode_kernel = new Uint8Array(bzimage, prot_mode_kernel_start);
    dbg_assert(KERNEL_ADDRESS + protected_mode_kernel.length < ramdisk_address || !initrd);
    mem8.set(protected_mode_kernel, KERNEL_ADDRESS);

    // ACPI tables so the kernel can enumerate the CPU and interrupt controllers.
    const rsdp = build_acpi_tables(mem8);
    write64(ZERO_PAGE + BOOT_PARAMS_ACPI_RSDP_ADDR, BigInt(rsdp));

    // e820 memory map. The ACPI tables live in the 0xF0000..0x100000 reserved
    // window, so that range is not reported as usable RAM.
    const e820 = [
        [0n, 0x9FC00n, 1],                               // low memory
        [0xF0000n, 0x10000n, 2],                         // reserved (ACPI/BIOS)
        [0x100000n, BigInt(mem8.length - 0x100000), 1],  // high memory
    ];
    mem8[ZERO_PAGE + BOOT_PARAMS_E820_ENTRIES] = e820.length;
    for(let i = 0; i < e820.length; i++)
    {
        const entry = ZERO_PAGE + BOOT_PARAMS_E820_TABLE + i * 20;
        write64(entry, e820[i][0]);
        write64(entry + 8, e820[i][1]);
        write32(entry + 16, e820[i][2]);
    }

    return {
        entry: KERNEL_ADDRESS + 0x200,
        boot_params: ZERO_PAGE,
        pml4: PML4,
        gdt: GDT,
    };
}

function make_linux_boot_rom(real_mode_segment, heap_end)
{
    // This rom will be executed by seabios after its initialisation
    // It sets up segment registers, the stack and calls the kernel real mode entry point

    const SIZE = 0x200;

    const data8 = new Uint8Array(SIZE);
    const data16 = new Uint16Array(data8.buffer);

    data16[0] = 0xAA55;
    data8[2] = SIZE / 0x200;

    let i = 3;

    data8[i++] = 0xFA; // cli
    data8[i++] = 0xB8; // mov ax, real_mode_segment
    data8[i++] = real_mode_segment >> 0;
    data8[i++] = real_mode_segment >> 8;
    data8[i++] = 0x8E; // mov es, ax
    data8[i++] = 0xC0;
    data8[i++] = 0x8E; // mov ds, ax
    data8[i++] = 0xD8;
    data8[i++] = 0x8E; // mov fs, ax
    data8[i++] = 0xE0;
    data8[i++] = 0x8E; // mov gs, ax
    data8[i++] = 0xE8;
    data8[i++] = 0x8E; // mov ss, ax
    data8[i++] = 0xD0;
    data8[i++] = 0xBC; // mov sp, heap_end
    data8[i++] = heap_end >> 0;
    data8[i++] = heap_end >> 8;
    data8[i++] = 0xEA; // jmp (real_mode_segment+0x20):0x0
    data8[i++] = 0x00;
    data8[i++] = 0x00;
    data8[i++] = real_mode_segment + 0x20 >> 0;
    data8[i++] = real_mode_segment + 0x20 >> 8;

    dbg_assert(i < SIZE);

    const checksum_index = i;
    data8[checksum_index] = 0;

    let checksum = 0;

    for(let i = 0; i < data8.length; i++)
    {
        checksum += data8[i];
    }

    data8[checksum_index] = -checksum;

    return data8;
}
