# Phase 2B.6.1-R rendering audit

## Scope and immutable inputs

This audit used source commit `4f96a0f42dbaf0adb58c4c3aad79fdd7bccfde84` and preserved
Phase 2B.6 as immutable `RESULT_C` and Phase 2B.6.1 as immutable `RESULT_D`. All compact prior
artifacts, the directly relevant frozen production files, and the official StackCube archive
rehashed successfully before the rendering probe. No replay, writer, export, archive, restore,
policy, optimizer, or training process was active.

The server runtime was Python 3.12.13 in
`/root/autodl-tmp/conda-envs/langmani`, with ManiSkill 3.0.1, SAPIEN 3.0.3,
PyTorch 2.11.0+cu128, NVIDIA driver 580.76.05, and one NVIDIA GeForce RTX 5090.

## Vulkan and EGL inventory

The relevant process-selectable identities were:

| Identity | JSON path | Referenced library | SHA-256 |
| --- | --- | --- | --- |
| Explicit EGL-associated NVIDIA ICD | `/etc/vulkan/icd.d/my_nvidia_icd.json` | `/lib/x86_64-linux-gnu/libEGL_nvidia.so.0` | `sha256:faa5543269860bb750bab4c3a9cb08a8c749f6d27500780b773dfcd334c188f8` |
| System GLX-associated NVIDIA ICD | `/etc/vulkan/icd.d/nvidia_icd.json` | `/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0` | `sha256:7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b` |
| NVIDIA EGL vendor file | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | GLVND-selected NVIDIA EGL library | `sha256:9e6f14afaa7523370ad96e6675d3357441ee3985e60405cb3e0ced30b2ddaf76` |
| SAPIEN packaged automatic ICD | `/root/autodl-tmp/conda-envs/langmani/lib/python3.12/site-packages/sapien/vulkan_library/nvidia_icd.json` | `libGLX_nvidia.so.0` | recorded in `rendering_stack_inventory.json` |

The complete `/etc/vulkan/icd.d/`, `/usr/share/vulkan/icd.d/`, EGL-vendor, GLVND, loader,
NVIDIA-library, Python-package, and inherited-variable inventories are preserved in
`artifacts/langmani_v2/phase_2b6_1r/rendering_stack_inventory.json`. No system ICD or `/etc` file
was modified.

## Frozen candidate protocol

The three candidates were frozen before execution:

1. Candidate A, the primary recovery contract, explicitly selected
   `/etc/vulkan/icd.d/my_nvidia_icd.json`. This was the ICD that had already passed the
   Phase 2B.6.1 standalone `vulkaninfo` diagnosis.
2. Candidate B separately audited the unambiguous system GLX-associated NVIDIA ICD with
   `vulkaninfo` only. It was prohibited from SAPIEN and StackCube construction.
3. Candidate C retained the automatic/default loader route as a safe negative control. It was
   prohibited from StackCube construction.

Candidate A used the complete process-scoped contract:

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
VK_DRIVER_FILES=<unset>
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
CUDA_VISIBLE_DEVICES=0
DISPLAY=<unset>
WAYLAND_DISPLAY=<unset>
XDG_RUNTIME_DIR=<fresh phase-owned directory>
```

The committed launcher activates the pinned environment and applies this contract itself; no
interactive export is required.

## Probe results

Candidate A passed `vulkaninfo --summary` in every one of three fresh processes. It selected
`NVIDIA GeForce RTX 5090`, not a software renderer. The minimal SAPIEN device probe also passed
3/3, selecting CUDA device 0 at PCI address `0000:27:00.0`, with `can_render=true`,
`is_cuda=true`, and clean process exit.

Candidate B failed its standalone `vulkaninfo` audit because `libGLX_nvidia.so.0` did not provide a
usable `vkCreateInstance`, ending in `ERROR_INCOMPATIBLE_DRIVER`. No SAPIEN or StackCube operation
was attempted for Candidate B.

The Candidate C negative control reproduced the previous automatic-route failure. Importing the
pinned SAPIEN runtime selected its packaged GLX-associated ICD; `vulkaninfo --summary` then failed
at Vulkan instance creation with `ERROR_INCOMPATIBLE_DRIVER`. The host was not manipulated to
force that result, and no SAPIEN probe or StackCube construction followed the failed Vulkan gate.

## Device binding

All three accepted runs agreed on:

```text
GPU name: NVIDIA GeForce RTX 5090
GPU UUID: GPU-27f8a10c-1b18-1109-85da-247ee1927024
NVIDIA PCI bus ID: 00000000:27:00.0
SAPIEN PCI address: 0000:27:00.0
CUDA device ID: 0
driver: 580.76.05
```

No llvmpipe, integrated GPU, stale GLX rendering device, or packaged incompatible ICD was accepted.
The exact inventories, command outputs, device records, and hashes are independently verifiable
from the compact evidence package.
