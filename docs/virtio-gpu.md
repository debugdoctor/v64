# Proposal: modern display device (virtio-gpu)

- Status: proposal, pending review
- Target version: TBD
- Related code: `src/virtio.ts`, `src/virtio_blk.ts`, `src/vga.ts`, `src/browser/screen.ts`, `src/cpu.ts`, `v64.d.ts`

## 1. Background

The only display device in v64 today is Bochs VGA + VBE (`1234:1111`,
`src/vga.ts`), a plain 2D linear framebuffer:

- the guest sets a mode through VBE/dispi and v64 composites the LFB onto a
  Canvas 2D surface;
- there is no GPU, no 3D and no virtio-gpu;
- modern Linux can drive it through `bochs-drm` / `simpledrm` / `vesafb`, but
  it is not a modern paravirtualized display path.

VirtualBox's VMSVGA/VBoxSVGA, VMware's SVGA II + SVGA3D and QEMU's
virtio-gpu + virgl/venus are the three shapes of a modern virtual GPU. This
proposal starts with the **deliverable virtio-gpu 2D** path and assesses 3D
separately.

## 2. Goals and non-goals

**Goals**

1. Implement the **2D path** of `virtio-gpu` (virtio device ID 16, modern PCI
   `1af4:1040`): mode setting, 2D resources, scanout and cursor.
2. Let a modern Linux guest (with the in-tree `virtio_gpu` driver) recognize
   and display without an extra driver.
3. Reuse the existing render pipeline (`screen.ts`); do not rewrite VGA.
4. Stay configurable, save/restore-able and fallback-safe (VGA remains the
   default).

**Non-goals (this phase)**

- 3D acceleration (see section 6; assessed separately).
- Multiple displays, hotplug and full EDID support.
- Replacing VGA: VGA stays the default and virtio-gpu is an optional device.

## 3. Current state and reusable pieces

- `VirtIO(cpu, options)` (`src/virtio.ts`) already wraps the PCI capability,
  virtqueues (desc/avail/used), MMIO/IO access, MSI-X, config space, notify
  and state save/restore.
- `virtio_blk` / `virtio_net` / `virtio_console` / `virtio_balloon` use device
  IDs `0x1041` / `0x1042` / `0x1043` / `0x1045`; **GPU uses `0x1040`** (ID 16)
  with subsystem device `0x10`.
- Devices are instantiated in `src/cpu.ts` (near `settings.net_device` /
  `settings.fs9p` / `settings.virtio_console`).
- Rendering: Canvas 2D + OffscreenCanvas in `src/browser/screen.ts`; `vga.ts`
  reports a resolution through `screen.set_size_graphical()`.
- Config types live in `v64.d.ts`; state uses `get_state` / `set_state`.

## 4. Phase 1: virtio-gpu 2D design

### 4.1 Device model

- PCI: vendor `0x1af4`, device `0x1040`, subsystem device `0x10`; BAR0 carries
  the config + virtqueues (modern virtio 1.0 common cfg / notify / ISR / device
  cfg layout, reusing `VirtIO`).
- Virtqueues:
  - **controlq (queue 0)**: commands and responses
    (`virtio_gpu_ctrl_hdr`).
  - **cursorq (queue 1)**: cursor updates (optional; start with a no-op or a
    minimal implementation).
- Features: start without `VIRTIO_GPU_F_VIRGL`, with `EDID` optional/off; the
  basic feature set first.

### 4.2 Command set (controlq)

| Command | Purpose | Phase |
|---------|---------|-------|
| `GET_DISPLAY_INFO` | report scanout count and size | P1 |
| `RESOURCE_CREATE_2D` | create a 2D resource (width/height/format) | P1 |
| `RESOURCE_UNREF` | release it | P1 |
| `RESOURCE_ATTACH_BACKING` / `DETACH_BACKING` | bind guest physical pages (scatter-gather) | P1 |
| `SET_SCANOUT` | bind a resource to a scanout | P1 |
| `TRANSFER_TO_HOST_2D` | copy guest resource contents into the host resource | P1 |
| `RESOURCE_FLUSH` | tell the host to redraw a region | P1 |
| `UPDATE_CURSOR` / `MOVE_CURSOR` | hardware cursor | P2 (minimal first) |
| `GET_EDID` | return EDID | P2 |
| `RESOURCE_ASSIGN_UUID` | compatibility | P2 |

Start with `B8G8R8A8_UNORM` / `B8G8R8X8_UNORM` / `R8G8B8A8_UNORM` and add the
rest of the pixman enum over time.

### 4.3 Resource and scanout model

- **Resource**: `{ id, width, height, format, guest_pages[] }`; the host holds a
  `Uint8Array` (can reuse a wasm memory view).
- **Scanout**: `{ resource_id, x, y, width, height }`.
- **Composition**: `TRANSFER_TO_HOST_2D` copies guest pages into the host
  resource; `RESOURCE_FLUSH` / `SET_SCANOUT` triggers drawing that region to the
  canvas.
- **Integration with the existing pipeline**: add a "gpu scanout -> screen"
  adapter that reuses `screen.set_size_graphical()` and
  `putImageData` / `drawImage`; VGA and virtio-gpu are mutually exclusive (only
  one is the primary display at a time).

### 4.4 Config and API

- New options (`v64.d.ts` + the `main.ts` setup UI):
  - `gpu?: "vga" | "virtio"` (default `"vga"`, preserving current behavior);
  - `gpu_memory_size?: number` (host-side resource cap).
- Default behavior is unchanged: without the option, VGA is used.

### 4.5 State save/restore

- The resource table, scanouts, control queue cursors and host resource contents
  (or a marker that they are not serializable, with a graceful fallback) must be
  part of `get_state` / `set_state`, following `virtio_blk`.

### 4.6 Guest support and verification

- **Linux**: the in-tree `virtio_gpu` + `drm` / `simpledrm`
  (`CONFIG_DRM_VIRTIO_GPU=y`). Validate with the existing Alpine / Tiny Core
  Pure64 kernels (or a custom kernel if the driver is not built in).
- **Windows**: needs a virtio-gpu driver (not promised in this phase).
- **Test cases**:
  1. the guest enumerates the PCI device and `/sys/class/drm/*/status` is
     `connected`;
  2. after mode setting the picture is correct and the resolution matches the
     guest;
  3. 2D blit / scrolling / cursor are correct;
  4. save + restore keeps picture and state consistent;
  5. compare `make browser` bundle size against the current baseline
     (gzip ~95 KB) and record the delta.

## 5. Phase 1 milestones

| Milestone | Content | Verification |
|-----------|---------|--------------|
| M1 | device skeleton + PCI/virtqueue + `GET_DISPLAY_INFO` | guest enumerates the device |
| M2 | `RESOURCE_CREATE_2D` + `ATTACH_BACKING` + `SET_SCANOUT` | a scanout appears |
| M3 | `TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH` + composite to canvas | guest picture is correct |
| M4 | config option + save/restore + tests | regression passes |
| M5 | docs + `v64.d.ts` + setup UI | releasable |

## 6. Phase 2: 3D feasibility (assessment only, not implemented here)

Three candidate routes:

| Route | Guest driver | Host mapping | Compatibility | Effort |
|-------|--------------|--------------|---------------|--------|
| virtio-gpu + **virgl** | Mesa `virgl` (Linux) | Gallium commands -> WebGL/WebGPU | standard, good ecosystem | very large |
| VMware **SVGA II + SVGA3D** | `vmwgfx` (Linux) / VMware Tools | SVGA3D -> WebGL | used by VBox/VMware/QEMU | very large |
| virtio-gpu + **venus** | Vulkan (Mesa) | Vulkan -> WebGPU | future-facing | huge |

**Assessment notes**

- Prefer **WebGPU** on the host: its resource/pipeline/command-buffer semantics
  map more directly to virgl/venus than WebGL does.
- It needs a real guest driver path (Mesa virgl or `vmwgfx`), which is expensive
  to test.
- Performance is limited by JS/WASM command translation; simple 3D is viable,
  complex scenes need benchmarking.
- Deliverable: an interface design plus a minimal demo (e.g. a virgl clear /
  triangle) before deciding to invest.

## 7. Risks

- Guest kernels/drivers may not include virtio-gpu, so validation depends on the
  chosen image.
- The modern virtio 1.0 config layout differs slightly from the existing
  devices; compare against what `VirtIO` already implements.
- Serializing host resource contents for save/restore can be large; decide on a
  strategy (compress, or mark as non-serializable).
- Coexisting with VGA requires an explicit "primary display" switch to avoid
  double rendering.

## 8. References

- virtio-gpu spec: <https://docs.oasis-open.org/virtio/virtio/v1.2/virtio-v1.2.html>
- Linux `virtio_gpu` driver: `drivers/gpu/drm/virtio`
- QEMU `hw/display/virtio-gpu.c`
- virgl: <https://virgil3d.github.io/>
- Bochs VBE (current state): <https://wiki.osdev.org/Bochs_VBE_Extensions>
