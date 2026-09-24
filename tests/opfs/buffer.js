#!/usr/bin/env node

// Unit test for AsyncOPFSBuffer (src/buffer.js). OPFS is not available in Node,
// so navigator.storage is backed by an in-memory file system.

import assert from "assert/strict";
import { AsyncOPFSBuffer } from "../../src/buffer.js";

process.on("unhandledRejection", exn => { throw exn; });

const BLOCK_SIZE = 256;

function create_mock_opfs()
{
    const files = new Map();

    const file_handle = name => ({
        async getFile()
        {
            const data = files.get(name);
            return {
                size: data.length,
                slice(offset, len)
                {
                    return {
                        arrayBuffer: async () => data.slice(offset, offset + len).buffer,
                    };
                },
            };
        },
        async createWritable()
        {
            let data = new Uint8Array(files.get(name));
            return {
                async write(chunk)
                {
                    if(chunk.position + chunk.data.length > data.length)
                    {
                        const grown = new Uint8Array(chunk.position + chunk.data.length);
                        grown.set(data);
                        data = grown;
                    }
                    data.set(chunk.data, chunk.position);
                },
                async truncate(size)
                {
                    const resized = new Uint8Array(size);
                    resized.set(data.subarray(0, Math.min(size, data.length)));
                    data = resized;
                },
                async close()
                {
                    files.set(name, data);
                },
            };
        },
    });

    const storage = {
        async getDirectory()
        {
            return {
                async getFileHandle(name, options)
                {
                    if(!files.has(name))
                    {
                        if(!options || !options.create)
                        {
                            throw new Error("NotFoundError: " + name);
                        }
                        files.set(name, new Uint8Array(0));
                    }
                    return file_handle(name);
                },
            };
        },
        async persist()
        {
            return true;
        },
    };

    return { files, storage };
}

function load(buffer)
{
    return new Promise((resolve, reject) =>
    {
        buffer.onload = resolve;
        buffer.load().catch(reject);
    });
}

function get(buffer, offset, len)
{
    return new Promise(resolve => buffer.get(offset, len, resolve));
}

function set(buffer, offset, data)
{
    return new Promise(resolve => buffer.set(offset, data, resolve));
}

const mock = create_mock_opfs();
Object.defineProperty(globalThis, "navigator", { value: { storage: mock.storage }, configurable: true });

const NAME = "test.img";
const SIZE = 8192;

async function test_load_creates_file()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);

    assert.equal(buffer.byteLength, SIZE);
    assert.equal(mock.files.get(NAME).length, SIZE, "file is created with the requested size");
    assert.deepEqual(await get(buffer, 0, 512), new Uint8Array(512), "fresh disk reads as zero");
}

async function test_write_and_reload()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);

    const data = new Uint8Array(BLOCK_SIZE).fill(0xAB);
    await set(buffer, 256, data);
    assert.equal(buffer.get_from_cache(256, BLOCK_SIZE)[0], 0xAB, "write is visible in the cache");
    await buffer.flush();

    const on_disk = mock.files.get(NAME);
    assert.equal(on_disk[256], 0xAB, "write reaches the backing file");
    assert.equal(on_disk[511], 0xAB);
    assert.equal(on_disk[512], 0, "neighbouring bytes are untouched");

    // A new instance reads the persisted data.
    const reopened = new AsyncOPFSBuffer(NAME, SIZE);
    await load(reopened);
    const read_back = await get(reopened, 0, 512);
    assert.equal(read_back[256], 0xAB, "persisted data is read back after reload");
    assert.equal(read_back[0], 0, "the rest of the block is still zero");
}

async function test_coalescing()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);

    // Blocks 4 and 5 are contiguous, block 8 is separate.
    await set(buffer, 4 * BLOCK_SIZE, new Uint8Array(BLOCK_SIZE).fill(1));
    await set(buffer, 5 * BLOCK_SIZE, new Uint8Array(BLOCK_SIZE).fill(2));
    await set(buffer, 8 * BLOCK_SIZE, new Uint8Array(BLOCK_SIZE).fill(3));
    await buffer.flush();

    const on_disk = mock.files.get(NAME);
    assert.equal(on_disk[4 * BLOCK_SIZE], 1);
    assert.equal(on_disk[5 * BLOCK_SIZE], 2);
    assert.equal(on_disk[6 * BLOCK_SIZE], 0, "the gap between the runs is not written");
    assert.equal(on_disk[8 * BLOCK_SIZE], 3);
    assert.equal(buffer.block_cache_is_write.size, 0, "dirty marks are cleared after a flush");
}

async function test_state_roundtrip()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);
    await set(buffer, 0, new Uint8Array(BLOCK_SIZE).fill(0x5A));

    const state = buffer.get_state();
    assert.equal(state[0], NAME);
    assert.equal(state[1], SIZE);

    const restored = new AsyncOPFSBuffer("other.img", SIZE);
    restored.set_state(state);
    assert.equal(restored.name, NAME);
    assert.equal(restored.byteLength, SIZE);
    assert.equal(restored.get_from_cache(0, BLOCK_SIZE)[0], 0x5A, "unflushed blocks survive a restore");
}

async function test_reopen_does_not_truncate()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);
    assert.equal(mock.files.get(NAME).length, SIZE, "reopening keeps the existing size");
}

async function test_write_during_flush()
{
    const buffer = new AsyncOPFSBuffer(NAME, SIZE);
    await load(buffer);

    await set(buffer, 0, new Uint8Array(BLOCK_SIZE).fill(0x11));

    const flush = buffer.flush();
    // A write that arrives while the flush is running re-marks its block.
    await set(buffer, 0, new Uint8Array(BLOCK_SIZE).fill(0x22));
    await flush;
    await buffer.flush();

    assert.equal(mock.files.get(NAME)[0], 0x22, "the newer write wins over the in-flight flush");
}

const tests = [
    test_load_creates_file,
    test_write_and_reload,
    test_coalescing,
    test_state_roundtrip,
    test_reopen_does_not_truncate,
    test_write_during_flush,
];

for(const test of tests)
{
    await test();
    console.log("ok - " + test.name);
}

console.log("All AsyncOPFSBuffer tests passed");
