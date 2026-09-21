// On wasm64 indirect calls are dispatched through JavaScript instead (the wasm
// table would be a 64-bit table, which engines don't support yet).
#[cfg(not(target_arch = "wasm64"))]
#[no_mangle]
pub fn call_indirect1(f: fn(u16), x: u16) { f(x); }
