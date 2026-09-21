# V64

V64 让你在浏览器里运行一台完整的 x86 PC。它模拟 x86 CPU、内存和常见的 PC 硬件，并在运行时把 guest 机器码翻译成 WebAssembly 以保证性能。

V64 是 [v86](https://github.com/copy/v86) 的分支，保留了它的硬件支持、JavaScript API 和许可证。上游文档见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

- 许可证：[BSD-2-Clause](./LICENSE)
- English: [`README.md`](./README.md)

## 功能

模拟的硬件：

- x86 兼容 CPU（Pentium 4 级别，含 SSE3）
- x87 FPU，使用 Berkeley SoftFloat 精确模拟 80 位浮点
- VGA/SVGA 显卡，支持 Bochs VBE 扩展
- IDE 磁盘控制器，内置 ISO 9660 CD-ROM
- 软盘控制器，8042 PS/2 键盘鼠标控制器
- 8254 PIT、8259 PIC、部分 APIC、CMOS RTC
- PCI 总线，NE2000 网卡
- virtio 文件系统 / 网络 / balloon
- SoundBlaster 16 声卡，Hayes 兼容调制解调器

V64 能真正启动操作系统，包括 Linux（32 位）、FreeDOS/MS-DOS、Windows 1.x 到 2000、ReactOS、KolibriOS、Haiku 以及大量 hobby 系统。完整兼容列表见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

> **暂不支持 64 位 guest。** 这正是本分支要做的事，计划见 [`ROADMAP.md`](./ROADMAP.md)。

## 环境要求

- `make`
- Rust，并安装 `wasm32-unknown-unknown` target
- 与 Rust 兼容的 `clang`
- Node.js（较新版本，上游验证 v24.x）
- `java`（用于 Closure Compiler；只构建 debug 版时不需要）
- 运行测试额外需要：`nasm`、`gdb`、`qemu-system`、`gcc`、`libc-i386`、`rustfmt`

```sh
rustup target add wasm32-unknown-unknown
```

完整的 Debian / WSL 环境见 [`tools/docker/test-image/Dockerfile`](./tools/docker/test-image/Dockerfile)。

## 构建

```sh
# 调试构建（产物：debug.html，不需要 java）
make

# 优化构建（产物：index.html）
make all
```

首次构建会先生成 `src/rust/gen/*.rs`，再用 wasm target 编译 Rust，最后打包 JavaScript。

## 运行

ROM 和磁盘镜像通过 XHR 加载，必须用 HTTP 提供服务（`file://` 不行）：

```sh
make run
```

然后打开提示的地址，例如 `http://localhost:8000/`。

### Docker

```sh
docker build -f tools/docker/exec/Dockerfile -t v64:alpine .
docker run -it -p 8000:8000 v64:alpine
```

### Dev Container

用支持 Dev Container 的 IDE（VS Code、Codespaces、IntelliJ IDEA 等）打开仓库，运行 “Fetch images” 任务。

## 嵌入用法

JavaScript API 与 v86 一致：

```javascript
var emulator = new V86({
    screen_container: document.getElementById("screen_container"),
    bios: { url: "./bios/seabios.bin" },
    vga_bios: { url: "./bios/vgabios.bin" },
    cdrom: { url: "./images/linux.iso" },
    autostart: true,
});
```

更多示例见 [`examples/`](./examples)（basic、串口终端、save/restore、网络等）。类型定义见 [`v86.d.ts`](./v86.d.ts)。Bundler（Vite/React/Next/Webpack）场景可用官方 npm 包 `v86`。

## 测试

测试镜像不随仓库分发：

```sh
mkdir -p images && curl --compressed --output-dir images/ --remote-name-all \
  https://i.copy.sh/{linux.iso,linux3.iso,linux4.iso,buildroot-bzimage68.bin,TinyCore-11.0.iso,oberon.img,msdos.img,openbsd-floppy.img,kolibri.img,windows101.img,os8.img,freedos722.img,mobius-fd-release5.img,msdos622.img}

make tests
```

## 目录结构

```
src/            模拟器主体（JS）+ Rust JIT（src/rust/）
gen/            指令表生成器（Node 脚本 -> src/rust/gen/*.rs）
bios/           SeaBIOS / VGA BIOS 二进制（需自行准备）
tools/docker/   各 guest 的镜像构建脚本（含 alpine/）
examples/       嵌入用法示例
docs/           文档（how-it-works、filesystem、networking 等）
tests/          集成与单元测试
lib/            内置第三方库（softfloat、zstd）
```

## 许可证

BSD-2-Clause，见 [`LICENSE`](./LICENSE)。这是宽松许可证：允许闭源使用，下游不强制开源。

第三方组件见 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。

## 致谢

- [v86](https://github.com/copy/v86) —— 本项目所基于的上游项目
- [QEMU](https://wiki.qemu.org/) —— CPU 测试用例与参考实现
- [Berkeley SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) —— 精确的 80 位浮点
- [zstd](https://github.com/facebook/zstd) —— 状态镜像压缩
- [jor1k](https://github.com/s-macke/jor1k) —— 9p、文件系统与 UART 驱动

## 贡献

上游约定见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。注意：**上游 v86 不接受由生成式 AI 全量或部分撰写的 PR/issue**；若要向上游回贡献，请遵守其规则。
