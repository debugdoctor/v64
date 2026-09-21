use std::alloc;
use std::ffi::c_void;

extern "C" {
    fn ZSTD_createDStream() -> *mut c_void;
    fn ZSTD_freeDStream(ctx: *mut c_void) -> usize;
    fn ZSTD_decompressStream_simpleArgs(
        ctx: *mut c_void,
        dst: *mut u8,
        dst_capacity: usize,
        dst_pos: *mut usize,
        src: *const u8,
        src_size: usize,
        src_pos: *mut usize,
    ) -> usize;

    fn ZSTD_isError(err: usize) -> u32;
}

const MALLOC_ALIGN: usize = 16;

// malloc and free are needed by the zstd library. `usize` is 4 bytes on wasm32
// and 8 bytes on wasm64, matching the C `size_t`.
#[no_mangle]
pub unsafe fn v86_malloc(size: usize) -> *mut u8 {
    let layout = alloc::Layout::from_size_align(size + 4, MALLOC_ALIGN).unwrap();
    let addr = alloc::alloc(layout);
    *(addr as *mut u32) = size as u32;
    addr.add(4)
}
#[no_mangle]
pub unsafe fn v86_free(addr: *mut u8) {
    let size = *((addr.sub(4)) as *mut u32);
    let layout = alloc::Layout::from_size_align(size as usize + 4, MALLOC_ALIGN).unwrap();
    alloc::dealloc(addr.sub(4), layout)
}

pub struct ZstdContext {
    ctx: *mut c_void,
    src: *mut u8,
    src_size: u32,
    src_pos: usize,
}

// The zstd entry points are called from JavaScript. Sizes stay `u32` (guest
// memory is below 4 GiB), but pointers are real pointers so that they are 32-bit
// on wasm32 and 64-bit on wasm64.
#[no_mangle]
pub unsafe fn zstd_create_ctx(src_size: u32) -> *mut ZstdContext {
    let src = alloc::alloc(alloc::Layout::from_size_align(src_size as usize, 1).unwrap());
    let ctx = ZSTD_createDStream();
    let result = alloc::alloc(alloc::Layout::new::<ZstdContext>()) as *mut ZstdContext;
    *result = ZstdContext {
        ctx,
        src,
        src_size,
        src_pos: 0,
    };
    result
}

#[no_mangle]
pub unsafe fn zstd_get_src_ptr(ctx: *mut ZstdContext) -> *mut u8 { (*ctx).src }

#[no_mangle]
pub unsafe fn zstd_free_ctx(ctx: *mut ZstdContext) {
    alloc::dealloc(
        (*ctx).src,
        alloc::Layout::from_size_align((*ctx).src_size as usize, 1).unwrap(),
    );
    ZSTD_freeDStream((*ctx).ctx);
    std::ptr::drop_in_place(ctx);
}

#[no_mangle]
pub unsafe fn zstd_read(ctx: *mut ZstdContext, length: u32) -> *mut u8 {
    let dst = alloc::alloc(alloc::Layout::from_size_align(length as usize, 1).unwrap());
    let mut dst_pos = 0;
    let result = ZSTD_decompressStream_simpleArgs(
        (*ctx).ctx,
        dst,
        length as usize,
        &mut dst_pos,
        (*ctx).src,
        (*ctx).src_size as usize,
        &mut (*ctx).src_pos,
    );
    if ZSTD_isError(result) != 0 {
        dbg_log!(
            "ZSTD_decompressStream_simpleArgs returned error: {}",
            result
        );
        dbg_assert!(false);
        zstd_read_free(dst, length);
        return std::ptr::null_mut::<u8>();
    }
    if dst_pos != length as usize {
        dbg_assert!(false, "ZSTD: Partial read");
        zstd_read_free(dst, length);
        return std::ptr::null_mut::<u8>();
    }
    dst
}
#[no_mangle]
pub unsafe fn zstd_read_free(ptr: *mut u8, length: u32) {
    alloc::dealloc(
        ptr,
        alloc::Layout::from_size_align(length as usize, 1).unwrap(),
    );
}
