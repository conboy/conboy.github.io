---
title: "Module 1 — The Fast Pool and the Toolchain"
description: "Installing Linux and ROCm 10, building llama.cpp twice (HIP and Vulkan), and measuring VRAM bandwidth to get your real, non-spec-sheet ratio between the two pools."
tags:
  - local-ai
  - rocm
  - vram-bandwidth
  - llama-cpp
  - toolchain
---

# Module 1 — The Fast Pool and the Toolchain

**6 hours** · Part of [[index|Two Pools]] · Prev: [[module-0-two-pools]] · Next: [[module-2-calibration]]

> [!info] What this module is for
> [[module-0-two-pools]] gave you an honest, measured number for the slow, large pool — system DDR5. This module does the same for the fast, small pool: 16GB of GDDR6 on the RX 9070 XT, theoretical 640 GB/s. But before you can measure anything on the GPU, you need a toolchain that can talk to it, so this module is really two things stapled together: a compressed, checklist-style Linux + ROCm install (deliberately terse — install guides age fast and this is the least differentiated content in the entire course), and the actual measurement work, building llama.cpp against the GPU twice, once through HIP and once through Vulkan, so you have two independently-verified backends before you trust either one’s numbers. The module ends with the number that matters most for the rest of the course: not the spec sheet’s 640 GB/s, and not the spec sheet’s 640/57.6 ≈ 11.1x ratio between the two pools, but **your own measured R** — the actual ratio between your Module 0 DDR number and this module’s VRAM number. Every later prediction in this course uses that measured R, never 11.1.

---

## 1.1 Linux and ROCm 10 — the checklist

> [!abstract] Goal
> Get a Linux install with ROCm 10 running and talking to the RX 9070 XT, as a checklist, not an essay.

**What’s going on:** ROCm 10.0 is the first ROCm release with official, non-workaround support for gfx1201 (the RX 9070 XT’s GPU architecture ID) on Linux, on Ubuntu 26.04, 24.04.4, or 22.04.5. This is worth stating plainly because a lot of 2025-era guides for this GPU predate that official support window and tell you to set `HSA_OVERRIDE_GFX_VERSION` to trick ROCm into treating gfx1201 as a similar, already-supported architecture. **You do not need that override on ROCm 10 — if a guide tells you to set it, the guide is stale.** Everything below is the minimum sequence; consult AMD’s current ROCm install docs for your exact distro’s package names, since these commands are the part of any AI/GPU tooling guide that goes stale fastest.

**Steps:**
1. Install one of the officially supported Ubuntu releases for ROCm 10 (26.04, 24.04.4, or 22.04.5) as your dual-boot Linux partition.
2. Add AMD’s ROCm apt repository and GPG key per the current ROCm 10 install docs for your distro version.
3. Install the ROCm meta-package (the exact package name varies by ROCm release — check current docs — but it typically pulls in the kernel driver, HIP runtime, and core libraries in one shot):
   ```bash
   sudo apt update
   sudo apt install rocm
   ```
4. Add your user to the `render` and `video` groups so you don’t need root to access the GPU device nodes:
   ```bash
   sudo usermod -aG render,video $LOGNAME
   ```
5. Reboot (a group membership change and a freshly installed kernel driver both need it).
6. Confirm the kernel driver loaded and the device node exists:
   ```bash
   ls /dev/dri/
   dmesg | grep -i amdgpu | tail -30
   ```

> [!success] Done when
> The system boots cleanly post-ROCm-install, `/dev/dri/` shows render and card device nodes, and `dmesg` shows the `amdgpu` driver attaching without repeated errors.

> [!question]- It’s not working
> 1. Did you pick a ROCm-10-supported Ubuntu point release specifically (26.04 / 24.04.4 / 22.04.5), not just "the latest Ubuntu"? A non-matching point release is the most common cause of package dependency failures.
> 2. Did you reboot after adding yourself to `render`/`video`? Group membership doesn’t apply to already-open shell sessions.
> 3. Is Secure Boot enabled in UEFI and blocking the out-of-tree kernel module from loading? Either disable Secure Boot or go through your distro’s MOK (Machine Owner Key) enrollment for the amdgpu/ROCm kernel module signing.
> 4. Does `dmesg` show the amdgpu driver repeatedly resetting or failing to initialize the ring? That usually means a version mismatch between the installed kernel and the ROCm package’s supported kernel range — check ROCm’s current compatibility matrix.

## 1.2 Verify the GPU shows up correctly

> [!abstract] Goal
> Confirm ROCm’s own tooling identifies the RX 9070 XT as gfx1201, before building anything against it.

**What’s going on:** `rocminfo` is ROCm’s runtime introspection tool — it enumerates every HSA agent (CPU and GPU) the runtime can see and prints their properties, including the GPU’s ISA/architecture string. This is the cheapest possible check that the entire driver-to-userspace stack is wired up correctly, and it’s the thing to run before you sink 20+ minutes into a llama.cpp build that might fail for a completely unrelated toolchain reason.

**Steps:**
1. Run:
   ```bash
   rocminfo | grep -i gfx
   ```
2. Confirm the output includes `gfx1201` associated with your GPU agent (not just a CPU agent entry).
3. Also check `rocm-smi` reports the card, its VRAM total, and current utilization:
   ```bash
   rocm-smi --showproductname --showmeminfo vram
   ```

> [!success] Done when
> `rocminfo` reports a GPU agent with `gfx1201`, and `rocm-smi` reports ~16GB of VRAM for that device.

> [!question]- It’s not working
> 1. Does `rocminfo` only list a CPU agent, no GPU agent at all? Recheck 1.1 — this points back to the driver/device-node layer, not anything ROCm-userspace-specific.
> 2. Does it report a different gfx ID than expected? Confirm you actually have the RX 9070 XT and not a different card in the system, and that you haven’t accidentally set `HSA_OVERRIDE_GFX_VERSION` somewhere in your shell profile from an old guide — unset it if so.
> 3. `rocm-smi` reports the card but with 0MB or wrong VRAM? Reboot once more — this can be a transient enumeration glitch right after a fresh driver install.

> [!bug] Gotcha
> If any guide you’re reading tells you to export `HSA_OVERRIDE_GFX_VERSION` for this card, check its date. gfx1201 was **not** supported when the RX 9070 XT launched in February 2025; official compatibility-matrix listing arrived around ROCm 7.0. A lot of "how to get ROCm working on RDNA4" content was written for that gap between launch and official support, and it uses the override as a workaround. On ROCm 10 the workaround is unnecessary and can actively cause `rocminfo` and build tooling to misidentify the architecture.

## 1.3 Build llama.cpp — HIP backend

> [!abstract] Goal
> Build llama.cpp with the HIP (ROCm-native) backend targeting gfx1201.

**What’s going on:** llama.cpp’s HIP backend compiles GPU kernels directly for your card’s architecture using AMD’s HIP toolchain (ROCm’s CUDA-alike). Targeting the architecture explicitly with `AMDGPU_TARGETS=gfx1201` avoids both an overly generic build (slower, or missing architecture-specific optimizations) and a mismatched one.

**Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   ```
2. Configure a HIP-backed build into its own directory:
   ```bash
   cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
       -DCMAKE_BUILD_TYPE=Release
   ```
3. Build:
   ```bash
   cmake --build build-hip --config Release -j $(nproc)
   ```
4. Confirm the HIP-specific binaries exist afterward, e.g. `build-hip/bin/llama-cli` and `build-hip/bin/llama-bench`.

> [!success] Done when
> The build completes with no errors, and `build-hip/bin/llama-cli --version` runs and prints a version string.

> [!question]- It’s not working
> 1. Does cmake configure fail immediately looking for HIP? Confirm `hipcc` is on your `PATH` (`which hipcc`) — if ROCm installed to a non-standard prefix, you may need to add its `bin/` directory to `PATH` or pass `-DCMAKE_HIP_COMPILER=...` explicitly.
> 2. Does the build fail deep into compiling `.hip`/`.cu`-equivalent kernel files with an "unsupported architecture" style error? Double check `-DAMDGPU_TARGETS=gfx1201` is spelled exactly right and matches what `rocminfo` reported in 1.2.
> 3. Does the build succeed but `llama-cli` immediately errors at runtime with a HIP initialization failure? Recheck the `render`/`video` group membership and device node permissions from 1.1 — this is a runtime permissions issue, not a build issue.

## 1.4 Build llama.cpp — Vulkan backend

> [!abstract] Goal
> Build a second, independent llama.cpp binary using the Vulkan backend, so you have a non-ROCm-dependent path to the same GPU.

**What’s going on:** The Vulkan backend talks to the GPU through the Vulkan graphics/compute API instead of HIP — it uses your Mesa (or AMD proprietary) Vulkan driver rather than ROCm’s runtime. Having both built matters for two reasons: it’s a cross-check (if HIP gives you a suspicious number, does Vulkan roughly agree?), and Vulkan is frequently the backend that still works when a ROCm point release temporarily regresses something on a brand-new architecture — RDNA4 support across the ROCm/HIP stack is newer and less battle-tested than Vulkan’s much longer-lived AMD GPU support.

**Steps:**
1. Ensure Vulkan development headers and a shader compiler are installed (package names vary by distro — you need the Vulkan SDK or your distro’s `vulkan-headers`/`libvulkan-dev` plus `glslc` or `shaderc`):
   ```bash
   sudo apt install libvulkan-dev glslc vulkan-tools
   ```
2. Confirm Vulkan itself sees the GPU before building anything:
   ```bash
   vulkaninfo --summary
   ```
3. Configure a separate build directory for the Vulkan backend (kept separate from `build-hip` so the two never contaminate each other’s cached CMake config):
   ```bash
   cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
   ```
4. Build:
   ```bash
   cmake --build build-vulkan --config Release -j $(nproc)
   ```

> [!success] Done when
> `build-vulkan/bin/llama-cli --version` runs successfully, and `vulkaninfo --summary` lists the RX 9070 XT as a discrete GPU device.

> [!question]- It’s not working
> 1. Does `vulkaninfo` fail to find any devices at all? That’s a driver-layer problem (Mesa RADV or the proprietary AMD Vulkan driver), not a llama.cpp build problem — fix that first.
> 2. Does cmake configure fail looking for `glslc`? It needs to be on `PATH` to compile the Vulkan backend’s shaders at build time.
> 3. Does the Vulkan build succeed but run dramatically slower than expected later, in Module 2? That’s expected to some degree — Vulkan compute paths in llama.cpp are not guaranteed to match HIP’s performance on every operation; the comparison itself is one of the things you’re here to measure, not assume.

## 1.5 Measure VRAM bandwidth

> [!abstract] Goal
> Run ROCm’s own bandwidth microbenchmark and get a real, measured GB/s number for the RX 9070 XT’s memory subsystem.

**What’s going on:** `rocm-bandwidth-test` is ROCm’s equivalent of STREAM for the GPU side — it moves large buffers between host and device, and device-to-device, and reports achieved GB/s for each direction. The number you care about here is the **device-to-device** figure — that’s the one that reflects the GPU’s own GDDR6 memory subsystem, analogous to what STREAM Triad measured for DDR5 in Module 0. 640 GB/s is the theoretical peak from the card’s 256-bit bus at 20 Gbps effective — a real, sustained figure will be some fraction of that, for the same reasons DDR fell short of its own theoretical ceiling in Module 0: refresh overhead, imperfect access-pattern efficiency, and the practical gap between a marketing peak and a measured one.

**Steps:**
1. Run the tool (it typically ships as part of the ROCm package set, or as a separate `rocm-bandwidth-test` package depending on your distro):
   ```bash
   rocm-bandwidth-test
   ```
2. Locate the device-to-device (D2D) bandwidth section in the output — it will report a matrix or list of GB/s figures for transfers local to the GPU’s own memory.
3. Record the largest/steady-state D2D figure — that’s your `measured_vram` number for the rest of this course.

> [!success] Done when
> You have a measured device-to-device GB/s figure from `rocm-bandwidth-test`, written down alongside the theoretical 640 GB/s it’s being compared against.

> [!question]- It’s not working
> 1. Does the tool report only host-to-device and device-to-host figures, no device-to-device section? Check you’re not running it with flags that restrict the transfer types tested — consult `rocm-bandwidth-test --help` (or `-h`) for the flag that enables all transfer types.
> 2. Is the reported number wildly low (well under half of 640 GB/s)? Check nothing else is contending for the GPU (a leftover llama.cpp process, a desktop compositor doing GPU work) and rerun.
> 3. Does the tool fail to find the device at all? Recheck 1.1/1.2 — this is the same driver/permissions layer those steps verified.

## 1.6 Ten-minute smoke test

> [!abstract] Goal
> Confirm both backends actually emit tokens end-to-end, on a tiny model, before you trust either one with anything resembling a real benchmark.

**What’s going on:** This is deliberately not a benchmark — Module 2 is where real, comparable numbers happen. This is just "does the plumbing work," the same spirit as Phase 0’s Hello World in the sibling NPU course: prove the whole path (weights load, GPU offload happens, tokens come out) before you build anything more complex on top of it.

**Steps:**
1. Grab any small GGUF model you already have or can fetch quickly (even a 1-2B model is fine for this check).
2. Run it through the HIP build with GPU offload forced on:
   ```bash
   ./build-hip/bin/llama-cli -m /path/to/small-model.gguf -ngl 99 -p "Say hello in one sentence." -n 32
   ```
3. Repeat with the Vulkan build:
   ```bash
   ./build-vulkan/bin/llama-cli -m /path/to/small-model.gguf -ngl 99 -p "Say hello in one sentence." -n 32
   ```
4. Confirm both produce coherent text output, and neither silently falls back to CPU-only (watch the startup log lines for GPU layer offload counts — it should say something like all layers assigned to the GPU device, not 0).

> [!success] Done when
> Both the HIP build and the Vulkan build produce token output from the same small model, with startup logs confirming GPU layers were actually offloaded (not silently running on CPU).

> [!question]- It’s not working
> 1. Does the startup log show 0 layers offloaded to GPU despite `-ngl 99`? The binary built but isn’t actually using the backend you think — recheck the build logs from 1.3/1.4 for whether HIP or Vulkan support actually compiled in (look for the backend registration line early in `llama-cli`’s own startup output).
> 2. Does one backend work and the other doesn’t? That’s useful, real data — note which one failed and revisit the corresponding build section (1.3 or 1.4) rather than assuming both should behave identically.
> 3. Does it run but produce garbage tokens? Check the GGUF file itself isn’t corrupted or truncated (a partial download) — try a known-good small model to isolate build vs. model issues.

## 1.7 Put the two pools side by side

> [!abstract] Goal
> Combine this module’s measured VRAM number with Module 0’s measured DDR number into the ratio the rest of the course actually uses.

**What’s going on:** The spec-sheet ratio between VRAM and DDR bandwidth here is 640/57.6 ≈ 11.1x. That number is real in the sense that both of its inputs are real theoretical peaks, but it is not what you should reason with, because neither pool sustains its theoretical peak in practice, and there’s no guarantee the two pools fall short by the same percentage. **R, your measured ratio, is measured_vram / measured_ddr** — using the actual numbers you produced in 1.5 and Module 0’s 0.5, not the spec sheet.

**Steps:**
1. Take your `measured_vram` GB/s from 1.5.
2. Take your `measured_ddr` GB/s from Module 0, section 0.5.
3. Compute `η_vram = measured_vram / 640`.
4. Compute `R = measured_vram / measured_ddr`.
5. Compare R against the naive spec-sheet ratio of ~11.1 — if they differ meaningfully, that gap is itself the finding: it means one pool is closer to (or further from) its theoretical ceiling than the other, and any later prediction that assumes an 11.1x ratio would be systematically wrong in your specific system.

> [!success] Done when
> You have measured VRAM GB/s, η_vram (measured/640), and R — your own measured ratio between the two pools, computed from two real numbers, sitting next to the naive 11.1x for comparison.

> [!bug] Gotcha
> Resist the urge to just use 11.1 going forward because it’s "close enough." The entire point of this course is that placement decisions are sensitive to the *real* ratio, and the whole reason Modules 0 and 1 exist separately, each ending in a measurement, is so that R is something you produced rather than something you assumed.

---

## Check your understanding

> [!question]- 1. Why build llama.cpp against both HIP and Vulkan instead of just picking one?
> > [!success]- Answer
> > They’re independent code paths to the same hardware — HIP goes through ROCm’s runtime, Vulkan goes through the Vulkan driver stack. Having both gives you a cross-check: if one backend produces a suspicious number later, you can compare against the other rather than wondering whether the number reflects the hardware or a bug/regression in one specific software stack. This matters especially on a newer architecture like RDNA4, where the ROCm/HIP support is less battle-tested than the much longer-lived Vulkan compute path.

> [!question]- 2. Your `rocm-bandwidth-test` device-to-device figure comes back at roughly 55% of the 640 GB/s theoretical peak. Is that a sign something is broken?
> > [!success]- Answer
> > Not necessarily — 640 GB/s is a theoretical peak computed from bus width and transfer rate, and real sustained bandwidth on any memory subsystem, GDDR6 included, falls short of that for the same categories of reasons DDR5 does in Module 0 (refresh overhead, access-pattern efficiency, and other real-world losses). Whether 55% specifically is normal or low is exactly what η_vram is for — record it, and treat unusually low efficiency as a prompt to recheck for GPU contention (another process using the card) rather than assuming the hardware itself is faulty.

> [!question]- 3. Why does this course insist on R = measured_vram / measured_ddr instead of the 11.1x you can compute from spec sheets alone?
> > [!success]- Answer
> > Because the two pools don’t necessarily fall short of their theoretical peaks by the same amount — DDR5 with a 4-DIMM population tax and GDDR6 on a GPU memory controller are different subsystems with different real-world efficiencies. If DDR is at 75% of its ceiling and VRAM is at 55% of its ceiling (for example), the real ratio between them is meaningfully different from 11.1x, and any later prediction in this course (Module 2’s `t_token` estimates especially) that used the naive spec-sheet ratio would be systematically off by however much those two efficiencies actually diverge.

## What’s next

You now have both halves of the roofline equation measured on real hardware: DDR5 GB/s and its saturation thread count from Module 0, and VRAM GB/s, η_vram, and R from this module. [[module-2-calibration]] is where these numbers stop being abstract and start predicting real token throughput — you’ll load actual coding models fully into VRAM, predict their tokens-per-second from your measured bandwidth, and then find out how wrong that prediction is.

<script src="/tutor.js" defer></script>
