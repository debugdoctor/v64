// virtio-blk: a virtio block device backed by an in-memory disk image.
// https://docs.oasis-open.org/virtio/virtio/v1.2/csd01/virtio-v1.2-csd01.html

import { LOG_PCI } from "./const.js";
import { dbg_log } from "./log.js";
import { VirtIO, VIRTIO_F_VERSION_1 } from "./virtio.js";

// Request types (virtio_blk_req.type)
const VIRTIO_BLK_T_IN = 0;
const VIRTIO_BLK_T_OUT = 1;
const VIRTIO_BLK_T_FLUSH = 4;
const VIRTIO_BLK_T_GET_ID = 8;

const VIRTIO_BLK_S_OK = 0;
const VIRTIO_BLK_S_UNSUPP = 2;

const SECTOR_SIZE = 512;

// For Types Only
import { CPU } from "./cpu.js";

/**
 * @constructor
 * @param {CPU} cpu
 * @param {Uint8Array} disk
 */
export function VirtioBlk(cpu, disk)
{
    /** @const @type {Uint8Array} */
    this.disk = disk instanceof Uint8Array ? disk : new Uint8Array(disk.buffer || disk);
    /** @const */
    this.capacity = Math.floor(this.disk.length / SECTOR_SIZE);

    const queues = [
        {size_supported: 128, notify_offset: 0},
    ];

    /** @type {VirtIO} */
    this.virtio = new VirtIO(cpu,
    {
        name: "virtio-blk",
        pci_id: 0x02 << 3,
        device_id: 0x1042,
        subsystem_device_id: 2,
        isr_status:
        {
            initial_port: 0xE700,
        },
        common:
        {
            initial_port: 0xE800,
            queues: queues,
            features:
            [
                VIRTIO_F_VERSION_1,
            ],
            on_driver_ok: () => {
                dbg_log("Virtio-blk setup", LOG_PCI);
            },
        },
        notification:
        {
            initial_port: 0xE900,
            single_handler: false,
            handlers:
            [
                () =>
                {
                    const queue = this.virtio.queues[0];
                    while(queue.has_request())
                    {
                        const bufchain = queue.pop_request();
                        this.HandleRequest(bufchain);
                        queue.push_reply(bufchain);
                    }
                    queue.flush_replies();
                },
            ],
        },
        device_specific:
        {
            initial_port: 0xE600,
            struct:
            [
                {
                    bytes: 4,
                    name: "capacity_low",
                    read: () => this.capacity & 0xFFFF_FFFF,
                    write: () => { /* read only */ },
                },
                {
                    bytes: 4,
                    name: "capacity_high",
                    read: () => Math.floor(this.capacity / 0x1_0000_0000),
                    write: () => { /* read only */ },
                },
                {
                    bytes: 4,
                    name: "size_max",
                    read: () => 0,
                    write: () => { /* read only */ },
                },
                {
                    bytes: 4,
                    name: "seg_max",
                    read: () => 0,
                    write: () => { /* read only */ },
                },
            ],
        },
    });
}

VirtioBlk.prototype.get_state = function()
{
    return [this.virtio, this.disk];
};

VirtioBlk.prototype.set_state = function(state)
{
    this.virtio.set_state(state[0]);
    this.disk = state[1];
};

VirtioBlk.prototype.reset = function()
{
    this.virtio.reset();
};

VirtioBlk.prototype.HandleRequest = function(bufchain)
{
    // struct virtio_blk_req { u32 type; u32 reserved; u64 sector; ... }
    const header = new Uint8Array(16);
    bufchain.get_next_blob(header);
    const type = (header[0] | header[1] << 8 | header[2] << 16 | header[3] << 24) >>> 0;
    const sector = (header[8] | header[9] << 8 | header[10] << 16 | header[11] << 24) >>> 0
        | (header[12] | header[13] << 8 | header[14] << 16 | header[15] << 24) * 0x1_0000_0000;
    const offset = sector * SECTOR_SIZE;

    let status = VIRTIO_BLK_S_OK;

    if(type === VIRTIO_BLK_T_IN)
    {
        // device-writable: the data buffer, then the status byte
        const length = bufchain.length_writable - 1;
        const data = new Uint8Array(length);
        for(let i = 0; i < length; i++)
        {
            const address = offset + i;
            data[i] = address < this.disk.length ? this.disk[address] : 0;
        }
        bufchain.set_next_blob(data);
    }
    else if(type === VIRTIO_BLK_T_OUT)
    {
        // device-readable: the header, then the data buffer
        const length = bufchain.length_readable - 16;
        const data = new Uint8Array(length);
        bufchain.get_next_blob(data);
        for(let i = 0; i < length; i++)
        {
            const address = offset + i;
            if(address < this.disk.length)
            {
                this.disk[address] = data[i];
            }
        }
    }
    else if(type === VIRTIO_BLK_T_GET_ID)
    {
        const id = new Uint8Array(20);
        const text = "v64-virtio-blk";
        for(let i = 0; i < text.length; i++) id[i] = text.charCodeAt(i);
        bufchain.set_next_blob(id);
    }
    else if(type === VIRTIO_BLK_T_FLUSH)
    {
        // backed by memory, nothing to flush
    }
    else
    {
        status = VIRTIO_BLK_S_UNSUPP;
    }

    bufchain.set_next_blob(new Uint8Array([status]));
};
