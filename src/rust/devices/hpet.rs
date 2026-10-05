//! Minimal HPET (High Precision Event Timer) at 0xFED00000.
//!
//! Enough for a kernel to use it as a clocksource and a clockevent: the
//! capabilities, a free-running 64-bit main counter, and three comparators
//! that can raise an I/O APIC interrupt. The counter advances from the host
//! millisecond clock at a fixed 100 MHz.

use std::sync::{Mutex, MutexGuard};

use crate::cpu::core::js;
use crate::devices::ioapic;

pub const HPET_MEM_ADDRESS: u32 = 0xFED00000;
pub const HPET_MEM_SIZE: u32 = 0x400;

// 100 MHz: 10 ns period in femtoseconds
const HPET_PERIOD_FS: u64 = 10_000_000;
const HPET_FREQ_PER_MS: f64 = 100_000.0;
const HPET_TIMER_COUNT: usize = 3;

const GEN_CONF_ENABLE: u64 = 1;
const TIMER_CONF_INT_ENABLE: u64 = 1 << 2;
const TIMER_CONF_PERIODIC: u64 = 1 << 3;
const TIMER_CONF_PERIODIC_CAPABLE: u64 = 1 << 4;
const TIMER_CONF_64BIT_CAPABLE: u64 = 1 << 5;
const TIMER_ROUTE_SHIFT: u64 = 9;
const TIMER_ROUTE_MASK: u64 = 0x1F;

#[derive(Copy, Clone)]
struct HpetTimer {
    config: u64,
    comparator: u64,
}

impl HpetTimer {
    const fn new() -> Self { HpetTimer { config: 0, comparator: 0 } }
}

struct Hpet {
    enabled: bool,
    counter: u64,
    last_tick: f64,
    config: u64,
    int_status: u64,
    timers: [HpetTimer; HPET_TIMER_COUNT],
}

static HPET: Mutex<Hpet> = Mutex::new(Hpet {
    enabled: false,
    counter: 0,
    last_tick: 0.0,
    config: 0,
    int_status: 0,
    timers: [HpetTimer::new(); HPET_TIMER_COUNT],
});

fn get_hpet() -> MutexGuard<'static, Hpet> { HPET.try_lock().unwrap() }

fn capabilities() -> u64 {
    // period (fs) | vendor id | timers-1 | revision, 64-bit counter, legacy route
    HPET_PERIOD_FS << 32
        | 0x8086u64 << 16
        | ((HPET_TIMER_COUNT as u64 - 1) << 8)
        | 1
        | 1 << 13
        | 1 << 15
}

fn timer_capabilities() -> u64 {
    TIMER_CONF_PERIODIC_CAPABLE | TIMER_CONF_64BIT_CAPABLE | (1 << 15) | (0x1F << 32)
}

impl Hpet {
    fn advance(&mut self, now: f64) {
        if !self.enabled {
            return;
        }
        let diff = if now > self.last_tick { now - self.last_tick } else { 0.0 };
        self.counter = self.counter.wrapping_add((diff * HPET_FREQ_PER_MS) as u64);
        self.last_tick = now;
    }

    // Fire any comparator that has been reached.
    fn check_timers(&mut self) {
        for i in 0..HPET_TIMER_COUNT {
            let timer = self.timers[i];
            if timer.config & TIMER_CONF_INT_ENABLE == 0 {
                continue;
            }
            let reached = if timer.config & TIMER_CONF_PERIODIC != 0 {
                timer.comparator != 0 && self.counter >= timer.comparator
            }
            else {
                timer.comparator != 0 && self.counter >= timer.comparator
            };
            if !reached {
                continue;
            }
            self.int_status |= 1 << i;
            let route = (timer.config >> TIMER_ROUTE_SHIFT) & TIMER_ROUTE_MASK;
            ioapic::set_irq(route as u8);
            if timer.config & TIMER_CONF_PERIODIC != 0 {
                self.timers[i].comparator = timer.comparator.wrapping_add(
                    // rearm for the next period; the period is whatever the
                    // kernel asked for, so just keep the same delta
                    self.counter.wrapping_sub(timer.comparator).max(1),
                );
            }
            else {
                self.timers[i].config &= !TIMER_CONF_INT_ENABLE;
            }
        }
    }

    fn read64(&self, offset: u32) -> u64 {
        match offset {
            0x000 => capabilities(),
            0x010 => self.config,
            0x020 => self.int_status,
            0x0F0 => self.counter,
            _ if offset >= 0x100 && offset < 0x100 + HPET_TIMER_COUNT as u32 * 0x20 => {
                let index = ((offset - 0x100) >> 5) as usize;
                match (offset - 0x100) & 0x1F {
                    0x00 => timer_capabilities() | self.timers[index].config,
                    0x08 => self.timers[index].comparator,
                    _ => 0,
                }
            },
            _ => 0,
        }
    }

    fn write64(&mut self, offset: u32, value: u64) {
        match offset {
            0x010 => {
                self.config = value;
                let enabled = value & GEN_CONF_ENABLE != 0;
                if enabled && !self.enabled {
                    self.last_tick = unsafe { js::microtick() };
                }
                self.enabled = enabled;
            },
            0x020 => self.int_status &= !value,
            0x0F0 => self.counter = value,
            _ if offset >= 0x100 && offset < 0x100 + HPET_TIMER_COUNT as u32 * 0x20 => {
                let index = ((offset - 0x100) >> 5) as usize;
                match (offset - 0x100) & 0x1F {
                    0x00 => {
                        self.timers[index].config = value
                            & !(TIMER_CONF_PERIODIC_CAPABLE | TIMER_CONF_64BIT_CAPABLE);
                    },
                    0x08 => self.timers[index].comparator = value,
                    _ => {},
                }
            },
            _ => {},
        }
    }
}

// 32-bit register access. 64-bit registers are two halves.
pub fn read32(addr: u32) -> u32 {
    let hpet = get_hpet();
    let offset = addr & !7;
    let value = hpet.read64(offset);
    if addr & 4 != 0 {
        (value >> 32) as u32
    }
    else {
        value as u32
    }
}

pub fn write32(addr: u32, value: u32) {
    let mut hpet = get_hpet();
    let offset = addr & !7;
    let old = hpet.read64(offset);
    let new = if addr & 4 != 0 {
        old & 0xFFFF_FFFF | (value as u64) << 32
    }
    else {
        old & 0xFFFF_FFFF_0000_0000 | value as u64
    };
    hpet.write64(offset, new);
}

#[no_mangle]
pub fn hpet_timer(now: f64) -> f64 {
    let mut hpet = get_hpet();
    hpet.advance(now);
    hpet.check_timers();
    1.0
}
