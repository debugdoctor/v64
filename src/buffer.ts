import { CPU } from "./cpu.js";
import { load_file, get_file_size } from "./lib.js";
import { dbg_assert, dbg_log } from "./log.js";

// The smallest size the emulated hardware can emit
const BLOCK_SIZE = 256;

const ASYNC_SAFE = false;

/**
 * Synchronous access to ArrayBuffer
 * @constructor
 */
export function SyncBuffer(buffer)
{
    dbg_assert(buffer instanceof ArrayBuffer);

    this.buffer = buffer;
    this.byteLength = buffer.byteLength;
    this.onload = undefined;
    this.onprogress = undefined;
}

SyncBuffer.prototype.load = function()
{
    this.onload && this.onload({ buffer: this.buffer });
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 * @param {number} start
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
SyncBuffer.prototype.get = function(start, len, fn)
{
    dbg_assert(start + len <= this.byteLength);
    fn(new Uint8Array(this.buffer, start, len));
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 * @param {number} start
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
SyncBuffer.prototype.get_and_cache = function(start, len, fn)
{
    this.get(start, len, fn);
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 * @param {number} start
 * @param {number} len
 * @return {Uint8Array|undefined}
 */
SyncBuffer.prototype.get_from_cache = function(start, len)
{
    dbg_assert(start + len <= this.byteLength);
    return new Uint8Array(this.buffer, start, len);
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 * @param {number} start
 * @param {!Uint8Array} slice
 * @param {function()} fn
 */
SyncBuffer.prototype.set = function(start, slice, fn)
{
    dbg_assert(start + slice.byteLength <= this.byteLength);

    new Uint8Array(this.buffer, start, slice.byteLength).set(slice);
    fn();
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 * @param {function(!ArrayBuffer)} fn
 */
SyncBuffer.prototype.get_buffer = function(fn)
{
    fn(this.buffer);
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 */
SyncBuffer.prototype.get_state = function()
{
    const state = [];
    state[0] = this.byteLength;
    state[1] = new Uint8Array(this.buffer);
    return state;
};

/**
 * @this {SyncBuffer|SyncFileBuffer}
 */
SyncBuffer.prototype.set_state = function(state)
{
    this.byteLength = state[0];
    this.buffer = state[1].slice().buffer;
};

/**
 * Asynchronous access to ArrayBuffer, loading blocks lazily as needed,
 * using the `Range: bytes=...` header
 *
 * @constructor
 * @param {string} filename Name of the file to download
 * @param {number|undefined} size
 * @param {number|undefined} fixed_chunk_size
 */
function AsyncXHRBuffer(filename, size, fixed_chunk_size)
{
    this.filename = filename;

    this.byteLength = size;

    this.block_cache = new Map();
    this.block_cache_is_write = new Set();

    this.fixed_chunk_size = fixed_chunk_size;
    this.cache_reads = !!fixed_chunk_size; // TODO: could also be useful in other cases (needs testing)

    this.onload = undefined;
    this.onprogress = undefined;
}

AsyncXHRBuffer.prototype.load = async function()
{
    if(this.byteLength !== undefined)
    {
        this.onload && this.onload(Object.create(null));
        return;
    }

    const size = await get_file_size(this.filename);
    this.byteLength = size;
    this.onload && this.onload(Object.create(null));
};

/**
 * @param {number} offset
 * @param {number} len
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 */
AsyncXHRBuffer.prototype.get_from_cache = function(offset, len)
{
    var number_of_blocks = len / BLOCK_SIZE;
    var block_index = offset / BLOCK_SIZE;

    for(var i = 0; i < number_of_blocks; i++)
    {
        var block = this.block_cache.get(block_index + i);

        if(!block)
        {
            return;
        }
    }

    if(number_of_blocks === 1)
    {
        return this.block_cache.get(block_index);
    }
    else
    {
        var result = new Uint8Array(len);
        for(var i = 0; i < number_of_blocks; i++)
        {
            result.set(this.block_cache.get(block_index + i), i * BLOCK_SIZE);
        }
        return result;
    }
};

/**
 * @param {number} offset
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
AsyncXHRBuffer.prototype.get = function(offset, len, fn, options)
{
    dbg_assert(offset + len <= this.byteLength);
    dbg_assert(offset % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    var block = this.get_from_cache(offset, len);
    if(block)
    {
        if(ASYNC_SAFE)
        {
            setTimeout(fn.bind(this, block), 0);
        }
        else
        {
            fn(block);
        }
        return;
    }

    var requested_start = offset;
    var requested_length = len;
    if(this.fixed_chunk_size)
    {
        requested_start = offset - (offset % this.fixed_chunk_size);
        requested_length = Math.ceil((offset - requested_start + len) / this.fixed_chunk_size) * this.fixed_chunk_size;
    }

    load_file(this.filename, {
        done: function done(buffer)
        {
            var block = new Uint8Array(buffer);
            this.handle_read(requested_start, requested_length, block);
            if(requested_start === offset && requested_length === len)
            {
                fn(block);
            }
            else
            {
                fn(block.subarray(offset - requested_start, offset - requested_start + len));
            }
        }.bind(this),
        range: { start: requested_start, length: requested_length },
        signal: options?.signal,
    });
};

/**
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 * @param {number} offset
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
AsyncXHRBuffer.prototype.get_and_cache = function(offset, len, fn, options)
{
    this.get(offset, len, function(block)
    {
        const start_block = offset / BLOCK_SIZE;
        const block_count = len / BLOCK_SIZE;

        for(let i = 0; i < block_count; i++)
        {
            if(!this.block_cache.has(start_block + i))
            {
                this.block_cache.set(start_block + i, block.slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE));
            }
        }

        fn(block);
    }.bind(this), options);
};

/**
 * Relies on this.byteLength and this.block_cache
 *
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 *
 * @param {number} start
 * @param {!Uint8Array} data
 * @param {function()} fn
 */
AsyncXHRBuffer.prototype.set = function(start, data, fn)
{
    var len = data.length;
    dbg_assert(start + data.byteLength <= this.byteLength);
    dbg_assert(start % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    var start_block = start / BLOCK_SIZE;
    var block_count = len / BLOCK_SIZE;

    for(var i = 0; i < block_count; i++)
    {
        var block = this.block_cache.get(start_block + i);

        if(block === undefined)
        {
            const data_slice = data.slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE);
            this.block_cache.set(start_block + i, data_slice);
        }
        else
        {
            const data_slice = data.subarray(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE);
            dbg_assert(block.byteLength === data_slice.length);
            block.set(data_slice);
        }

        this.block_cache_is_write.add(start_block + i);
    }

    fn();
};

/**
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 * @param {number} offset
 * @param {number} len
 * @param {!Uint8Array} block
 */
AsyncXHRBuffer.prototype.handle_read = function(offset, len, block)
{
    // Used by AsyncXHRBuffer, AsyncXHRPartfileBuffer and AsyncFileBuffer
    // Overwrites blocks from the original source that have been written since

    var start_block = offset / BLOCK_SIZE;
    var block_count = len / BLOCK_SIZE;

    for(var i = 0; i < block_count; i++)
    {
        const cached_block = this.block_cache.get(start_block + i);

        if(cached_block)
        {
            block.set(cached_block, i * BLOCK_SIZE);
        }
        else if(this.cache_reads)
        {
            this.block_cache.set(start_block + i, block.slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE));
        }
    }
};

AsyncXHRBuffer.prototype.get_buffer = function(fn)
{
    // We must download all parts, unlikely a good idea for big files
    fn();
};

///**
// * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
// */
//AsyncXHRBuffer.prototype.get_block_cache = function()
//{
//    var count = Object.keys(this.block_cache).length;

//    var buffer = new Uint8Array(count * BLOCK_SIZE);
//    var indices = [];

//    var i = 0;
//    for(var index of Object.keys(this.block_cache))
//    {
//        var block = this.block_cache.get(index);
//        dbg_assert(block.length === BLOCK_SIZE);
//        index = +index;
//        indices.push(index);
//        buffer.set(
//            block,
//            i * BLOCK_SIZE
//        );
//        i++;
//    }

//    return {
//        buffer,
//        indices,
//        block_size: BLOCK_SIZE,
//    };
//};

/**
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 */
AsyncXHRBuffer.prototype.get_state = function()
{
    const state = [];
    const block_cache = [];

    for(const [index, block] of this.block_cache)
    {
        dbg_assert(isFinite(index));
        if(this.block_cache_is_write.has(index))
        {
            block_cache.push([index, block]);
        }
    }

    state[0] = block_cache;
    return state;
};

/**
 * @this {AsyncXHRBuffer|AsyncXHRPartfileBuffer|AsyncFileBuffer|AsyncOPFSBuffer}
 */
AsyncXHRBuffer.prototype.set_state = function(state)
{
    const block_cache = state[0];
    this.block_cache.clear();
    this.block_cache_is_write.clear();

    for(const [index, block] of block_cache)
    {
        dbg_assert(isFinite(index));
        this.block_cache.set(index, block);
        this.block_cache_is_write.add(index);
    }
};

/**
 * Asynchronous access to ArrayBuffer, loading blocks lazily as needed,
 * downloading files named filename-%d-%d.ext (where the %d are start and end offset).
 * Or, if partfile_alt_format is set, filename-%08d.ext (where %d is the part number, compatible with gnu split).
 *
 * @constructor
 * @param {string} filename Name of the file to download
 * @param {number|undefined} size
 * @param {number|undefined} fixed_chunk_size
 * @param {boolean|undefined} partfile_alt_format
 */
export function AsyncXHRPartfileBuffer(filename, size, fixed_chunk_size, partfile_alt_format, zstd_decompress)
{
    const parts = filename.match(/\.[^\.]+(\.zst)?$/);

    this.extension = parts ? parts[0] : "";
    this.basename = filename.substring(0, filename.length - this.extension.length);

    this.is_zstd = this.extension.endsWith(".zst");

    if(!this.basename.endsWith("/"))
    {
        this.basename += "-";
    }

    this.block_cache = new Map();
    this.block_cache_is_write = new Set();

    this.byteLength = size;
    this.fixed_chunk_size = fixed_chunk_size;
    this.partfile_alt_format = !!partfile_alt_format;
    this.zstd_decompress = zstd_decompress;

    this.cache_reads = !!fixed_chunk_size; // TODO: could also be useful in other cases (needs testing)

    this.onload = undefined;
    this.onprogress = undefined;
}

AsyncXHRPartfileBuffer.prototype.load = function()
{
    if(this.byteLength !== undefined)
    {
        this.onload && this.onload(Object.create(null));
        return;
    }
    dbg_assert(false);
    this.onload && this.onload(Object.create(null));
};

/**
 * @param {number} offset
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
AsyncXHRPartfileBuffer.prototype.get = function(offset, len, fn, options)
{
    dbg_assert(offset + len <= this.byteLength);
    dbg_assert(offset % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    const block = this.get_from_cache(offset, len);

    if(block)
    {
        if(ASYNC_SAFE)
        {
            setTimeout(fn.bind(this, block), 0);
        }
        else
        {
            fn(block);
        }
        return;
    }

    if(this.fixed_chunk_size)
    {
        const start_index = Math.floor(offset / this.fixed_chunk_size);
        const m_offset = offset - start_index * this.fixed_chunk_size;
        dbg_assert(m_offset >= 0);
        const total_count = Math.ceil((m_offset + len) / this.fixed_chunk_size);
        const blocks = new Uint8Array(total_count * this.fixed_chunk_size);
        let finished = 0;

        for(let i = 0; i < total_count; i++)
        {
            const offset = (start_index + i) * this.fixed_chunk_size;

            const part_filename =
                this.partfile_alt_format ?
                    // matches output of gnu split:
                    //   split -b 512 -a8 -d --additional-suffix .img w95.img w95-
                    this.basename + (start_index + i + "").padStart(8, "0") + this.extension
                :
                    this.basename + offset + "-" + (offset + this.fixed_chunk_size) + this.extension;

            // XXX: unnecessary allocation
            const block = this.get_from_cache(offset, this.fixed_chunk_size);

            if(block)
            {
                blocks.set(block, i * this.fixed_chunk_size);
                finished++;
                if(finished === total_count)
                {
                    fn(blocks.subarray(m_offset, m_offset + len));
                }
            }
            else
            {
                load_file(part_filename, {
                    done: async function done(buffer)
                    {
                        let block = new Uint8Array(buffer);

                        if(this.is_zstd)
                        {
                            const decompressed = await this.zstd_decompress(this.fixed_chunk_size, block);
                            block = new Uint8Array(decompressed);
                        }

                        blocks.set(block, i * this.fixed_chunk_size);
                        this.handle_read((start_index + i) * this.fixed_chunk_size, this.fixed_chunk_size|0, block);

                        finished++;
                        if(finished === total_count)
                        {
                            fn(blocks.subarray(m_offset, m_offset + len));
                        }
                    }.bind(this),
                    signal: options?.signal,
                });
            }
        }
    }
    else
    {
        const part_filename = this.basename + offset + "-" + (offset + len) + this.extension;

        load_file(part_filename, {
            done: function done(buffer)
            {
                dbg_assert(buffer.byteLength === len);
                var block = new Uint8Array(buffer);
                this.handle_read(offset, len, block);
                fn(block);
            }.bind(this),
            signal: options?.signal,
        });
    }
};

AsyncXHRPartfileBuffer.prototype.get_from_cache = AsyncXHRBuffer.prototype.get_from_cache;
AsyncXHRPartfileBuffer.prototype.get_and_cache = AsyncXHRBuffer.prototype.get_and_cache;
AsyncXHRPartfileBuffer.prototype.set = AsyncXHRBuffer.prototype.set;
AsyncXHRPartfileBuffer.prototype.handle_read = AsyncXHRBuffer.prototype.handle_read;
//AsyncXHRPartfileBuffer.prototype.get_block_cache = AsyncXHRBuffer.prototype.get_block_cache;
AsyncXHRPartfileBuffer.prototype.get_state = AsyncXHRBuffer.prototype.get_state;
AsyncXHRPartfileBuffer.prototype.set_state = AsyncXHRBuffer.prototype.set_state;

/**
 * Synchronous access to File, loading blocks from the input type=file
 * The whole file is loaded into memory during initialisation
 *
 * @constructor
 */
export function SyncFileBuffer(file)
{
    this.file = file;
    this.byteLength = file.size;

    if(file.size > (1 << 30))
    {
        console.warn("SyncFileBuffer: Allocating buffer of " + (file.size >> 20) + " MB ...");
    }

    this.buffer = new ArrayBuffer(file.size);

    this.onload = undefined;
    this.onprogress = undefined;
}

SyncFileBuffer.prototype.load = function()
{
    this.load_next(0);
};

/**
 * @param {number} start
 */
SyncFileBuffer.prototype.load_next = function(start)
{
    const PART_SIZE = 4 << 20;

    var filereader = new FileReader();

    filereader.onload = function(e)
    {
        var buffer = new Uint8Array(e.target.result);
        new Uint8Array(this.buffer, start).set(buffer);
        this.load_next(start + PART_SIZE);
    }.bind(this);

    if(this.onprogress)
    {
        this.onprogress({
            loaded: start,
            total: this.byteLength,
            lengthComputable: true,
        });
    }

    if(start < this.byteLength)
    {
        var end = Math.min(start + PART_SIZE, this.byteLength);
        var slice = this.file.slice(start, end);
        filereader.readAsArrayBuffer(slice);
    }
    else
    {
        this.file = undefined;
        this.onload && this.onload({ buffer: this.buffer });
    }
};

SyncFileBuffer.prototype.get = SyncBuffer.prototype.get;
SyncFileBuffer.prototype.get_and_cache = SyncBuffer.prototype.get_and_cache;
SyncFileBuffer.prototype.get_from_cache = SyncBuffer.prototype.get_from_cache;
SyncFileBuffer.prototype.set = SyncBuffer.prototype.set;
SyncFileBuffer.prototype.get_buffer = SyncBuffer.prototype.get_buffer;
SyncFileBuffer.prototype.get_state = SyncBuffer.prototype.get_state;
SyncFileBuffer.prototype.set_state = SyncBuffer.prototype.set_state;

/**
 * Asynchronous access to File, loading blocks from the input type=file
 *
 * @constructor
 */
export function AsyncFileBuffer(file)
{
    this.file = file;
    this.byteLength = file.size;

    this.block_cache = new Map();
    this.block_cache_is_write = new Set();

    this.onload = undefined;
    this.onprogress = undefined;
}

AsyncFileBuffer.prototype.load = function()
{
    this.onload && this.onload(Object.create(null));
};

/**
 * @param {number} offset
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 */
AsyncFileBuffer.prototype.get = function(offset, len, fn)
{
    dbg_assert(offset % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    var block = this.get_from_cache(offset, len);
    if(block)
    {
        fn(block);
        return;
    }

    var fr = new FileReader();

    fr.onload = function(e)
    {
        var buffer = e.target.result;
        var block = new Uint8Array(buffer);

        this.handle_read(offset, len, block);
        fn(block);
    }.bind(this);

    fr.readAsArrayBuffer(this.file.slice(offset, offset + len));
};
AsyncFileBuffer.prototype.get_from_cache = AsyncXHRBuffer.prototype.get_from_cache;
AsyncFileBuffer.prototype.get_and_cache = AsyncXHRBuffer.prototype.get_and_cache;
AsyncFileBuffer.prototype.set = AsyncXHRBuffer.prototype.set;
AsyncFileBuffer.prototype.handle_read = AsyncXHRBuffer.prototype.handle_read;
AsyncFileBuffer.prototype.get_state = AsyncXHRBuffer.prototype.get_state;
AsyncFileBuffer.prototype.set_state = AsyncXHRBuffer.prototype.set_state;

AsyncFileBuffer.prototype.get_buffer = function(fn)
{
    // We must load all parts, unlikely a good idea for big files
    fn();
};

AsyncFileBuffer.prototype.get_as_file = function(name)
{
    var parts = [];
    var existing_blocks: any[] = Array.from(this.block_cache.keys() as any).sort(function(x: number, y: number) { return x - y; });

    var current_offset = 0;

    for(var i = 0; i < existing_blocks.length; i++)
    {
        var block_index = existing_blocks[i];
        var block = this.block_cache.get(block_index);
        var start = block_index * BLOCK_SIZE;
        dbg_assert(start >= current_offset);

        if(start !== current_offset)
        {
            parts.push(this.file.slice(current_offset, start));
            current_offset = start;
        }

        parts.push(block);
        current_offset += block.length;
    }

    if(current_offset !== this.file.size)
    {
        parts.push(this.file.slice(current_offset));
    }

    var file = new File(parts, name);
    dbg_assert(file.size === this.file.size);

    return file;
};

/**
 * Persistent disk image in the Origin Private File System (OPFS). Blocks are
 * read and written asynchronously, cached in RAM, and dirty blocks are
 * committed in batches.
 *
 * @constructor
 * @param {string} name File name in the OPFS root directory
 * @param {number} size Size of the disk image in bytes
 */
export function AsyncOPFSBuffer(name, size)
{
    dbg_assert(typeof name === "string" && name.length > 0);
    dbg_assert(size > 0 && size % BLOCK_SIZE === 0);

    this.name = name;
    this.byteLength = size;

    this.block_cache = new Map();
    this.block_cache_is_write = new Set();

    // Read blocks are cached up to this many; dirty blocks are never evicted.
    this.max_cached_blocks = 16 * 1024 * 1024 / BLOCK_SIZE;

    /** @type {?FileSystemFileHandle} */
    this.file = null;

    this.flush_in_progress = null;
    this.flush_again = false;
    this.flush_timer = undefined;

    this.onload = undefined;
    this.onprogress = undefined;
}

AsyncOPFSBuffer.prototype.load = async function()
{
    const dir = await navigator.storage.getDirectory();
    this.file = await dir.getFileHandle(this.name, { create: true });

    const existing = await this.file.getFile();
    if(existing.size !== this.byteLength)
    {
        const writable = await this.file.createWritable();
        await writable.truncate(this.byteLength);
        await writable.close();
    }

    request_persistent_storage();
    this.onload && this.onload({});
};

/**
 * @param {number} offset
 * @param {number} len
 * @param {function(!Uint8Array)} fn
 * @param {{signal: AbortSignal}=} options
 */
AsyncOPFSBuffer.prototype.get = function(offset, len, fn, options)
{
    dbg_assert(offset + len <= this.byteLength);
    dbg_assert(offset % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    const block = this.get_from_cache(offset, len);
    if(block)
    {
        fn(block);
        return;
    }

    this.read_from_storage(offset, len).then(block =>
    {
        if(options && options.signal && options.signal.aborted)
        {
            return;
        }
        fn(block);
    }, e => dbg_log("AsyncOPFSBuffer: read failed: " + e));
};

/**
 * @param {number} offset
 * @param {number} len
 * @return {!Promise<!Uint8Array>}
 */
AsyncOPFSBuffer.prototype.read_from_storage = async function(offset, len)
{
    const file = /** @type {!FileSystemFileHandle} */ (this.file);
    dbg_assert(file);

    const blob = await file.getFile();
    const block = new Uint8Array(await blob.slice(offset, offset + len).arrayBuffer());
    this.handle_read(offset, len, block);
    return block;
};

/**
 * Apply cached writes and cache the freshly read blocks.
 *
 * @param {number} offset
 * @param {number} len
 * @param {!Uint8Array} block
 */
AsyncOPFSBuffer.prototype.handle_read = function(offset, len, block)
{
    const start_block = offset / BLOCK_SIZE;
    const block_count = len / BLOCK_SIZE;

    for(let i = 0; i < block_count; i++)
    {
        const index = start_block + i;
        const cached = this.block_cache.get(index);

        if(cached)
        {
            block.set(cached, i * BLOCK_SIZE);
        }
        else
        {
            this.block_cache.set(index, block.slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE));
        }
    }

    this.evict();
};

AsyncOPFSBuffer.prototype.evict = function()
{
    if(this.block_cache.size <= this.max_cached_blocks)
    {
        return;
    }

    for(const index of this.block_cache.keys())
    {
        if(this.block_cache.size <= this.max_cached_blocks)
        {
            break;
        }
        if(!this.block_cache_is_write.has(index))
        {
            this.block_cache.delete(index);
        }
    }
};

/**
 * @param {number} start
 * @param {!Uint8Array} data
 * @param {function()} fn
 */
AsyncOPFSBuffer.prototype.set = function(start, data, fn)
{
    const len = data.length;
    dbg_assert(start + len <= this.byteLength);
    dbg_assert(start % BLOCK_SIZE === 0);
    dbg_assert(len % BLOCK_SIZE === 0);
    dbg_assert(len);

    const start_block = start / BLOCK_SIZE;
    const block_count = len / BLOCK_SIZE;

    for(let i = 0; i < block_count; i++)
    {
        const index = start_block + i;
        const slice = data.slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE);
        const cached = this.block_cache.get(index);

        if(cached)
        {
            cached.set(slice);
        }
        else
        {
            this.block_cache.set(index, slice);
        }

        this.block_cache_is_write.add(index);
    }

    this.schedule_flush();
    fn();
};

AsyncOPFSBuffer.prototype.schedule_flush = function()
{
    if(this.flush_timer !== undefined)
    {
        return;
    }

    this.flush_timer = setTimeout(() =>
    {
        this.flush_timer = undefined;
        this.flush();
    }, 50);
};

/**
 * Commit all dirty blocks. Concurrent calls are coalesced.
 * @return {!Promise}
 */
AsyncOPFSBuffer.prototype.flush = function()
{
    if(this.flush_in_progress)
    {
        this.flush_again = true;
        return this.flush_in_progress;
    }

    this.flush_in_progress = this.do_flush().catch(e =>
    {
        dbg_log("AsyncOPFSBuffer: flush failed: " + e);
    }).then(() =>
    {
        this.flush_in_progress = null;
        if(this.flush_again)
        {
            this.flush_again = false;
            this.flush();
        }
    });

    return this.flush_in_progress;
};

AsyncOPFSBuffer.prototype.do_flush = async function()
{
    if(!this.file || this.block_cache_is_write.size === 0)
    {
        return;
    }

    // Snapshot the dirty blocks and clear the marks. A write that arrives while
    // the flush is running re-marks its block; the next flush commits it.
    const indices: any[] = Array.from(this.block_cache_is_write as any).sort((a: number, b: number) => a - b);
    this.block_cache_is_write.clear();

    const runs = [];
    let first = indices[0];
    let last = indices[0];

    const push_run = () =>
    {
        const data = new Uint8Array((last - first + 1) * BLOCK_SIZE);
        for(let index = first; index <= last; index++)
        {
            data.set(this.block_cache.get(index), (index - first) * BLOCK_SIZE);
        }
        runs.push({ offset: first * BLOCK_SIZE, data });
    };

    for(let i = 1; i < indices.length; i++)
    {
        if(indices[i] === last + 1)
        {
            last = indices[i];
        }
        else
        {
            push_run();
            first = last = indices[i];
        }
    }
    push_run();

    const file = /** @type {!FileSystemFileHandle} */ (this.file);
    const writable = await file.createWritable({ keepExistingData: true });
    for(const run of runs)
    {
        await writable.write({ type: "write", position: run.offset, data: run.data });
    }
    await writable.close();
};

AsyncOPFSBuffer.prototype.get_buffer = function(fn)
{
    // The backing file is not materialised in memory.
    fn();
};

AsyncOPFSBuffer.prototype.get_state = function()
{
    const state = [];
    state[0] = this.name;
    state[1] = this.byteLength;
    state[2] = AsyncXHRBuffer.prototype.get_state.call(this);
    return state;
};

AsyncOPFSBuffer.prototype.set_state = function(state)
{
    this.name = state[0];
    this.byteLength = state[1];
    AsyncXHRBuffer.prototype.set_state.call(this, state[2]);
};

AsyncOPFSBuffer.prototype.get_from_cache = AsyncXHRBuffer.prototype.get_from_cache;
AsyncOPFSBuffer.prototype.get_and_cache = AsyncXHRBuffer.prototype.get_and_cache;

/**
 * Ask the browser to keep this origin's storage. Best effort; may be denied.
 * @return {!Promise<boolean>}
 */
export async function request_persistent_storage()
{
    if(navigator.storage && navigator.storage.persist)
    {
        try
        {
            return await navigator.storage.persist();
        }
        catch(e)
        {
        }
    }
    return false;
}

export function buffer_from_object(obj, zstd_decompress_worker)
{
    // TODO: accept Uint8Array, ArrayBuffer, File, url rather than { url }

    if(obj.buffer instanceof ArrayBuffer)
    {
        return new SyncBuffer(obj.buffer);
    }
    else if(typeof File !== "undefined" && obj.buffer instanceof File)
    {
        // SyncFileBuffer:
        // - loads the whole disk image into memory, impossible for large files (more than 1GB)
        // - can later serve get/set operations fast and synchronously
        // - takes some time for first load, neglectable for small files (up to 100Mb)
        //
        // AsyncFileBuffer:
        // - loads slices of the file asynchronously as requested
        // - slower get/set

        // Heuristics: If file is larger than or equal to 256M, use AsyncFileBuffer
        let is_async = obj.async;
        if(is_async === undefined)
        {
            is_async = obj.buffer.size >= 256 * 1024 * 1024;
        }

        if(is_async)
        {
            return new AsyncFileBuffer(obj.buffer);
        }
        else
        {
            return new SyncFileBuffer(obj.buffer);
        }
    }
    else if(obj.opfs)
    {
        // Persistent disk in the origin private file system
        dbg_assert(obj.size, "AsyncOPFSBuffer: a size is required");
        return new AsyncOPFSBuffer(obj.opfs, obj.size);
    }
    else if(obj.url)
    {
        // Note: Only async for now

        if(obj.use_parts)
        {
            return new AsyncXHRPartfileBuffer(obj.url, obj.size, obj.fixed_chunk_size, false, zstd_decompress_worker);
        }
        else
        {
            return new AsyncXHRBuffer(obj.url, obj.size, obj.fixed_chunk_size);
        }
    }
    else
    {
        dbg_log("Ignored file: url=" + obj.url + " buffer=" + obj.buffer);
    }
}
