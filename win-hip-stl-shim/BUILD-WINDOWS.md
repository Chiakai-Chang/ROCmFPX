# Building the gfx1151 HIP lane on Windows

The only working Windows build path is **Ninja + ROCm clang**. Do not use the
Visual Studio generator: CMake hands the HIP `.cu` sources to MSVC `cl.exe`,
which cannot compile HIP (`error C2062` floods from the HIP headers).

## Prerequisites

- AMD HIP SDK (ROCm 7.x) — `HIP_PATH` set, e.g. `C:\Program Files\AMD\ROCm\7.1`
- Visual Studio 2026 (MSVC toolset + bundled Ninja/CMake). With VS 17.11+ /
  VS 18 STL you also need the shim headers in this directory (see README below).
- Runtime DLL note: launchers must prepend `%HIP_PATH%\bin` to `PATH`
  (amdhip64 comes from the driver; hipblas/rocblas from the SDK).

## Build script

Save as `build-win.bat` (git-ignored) in the repo root, or run the blocks by hand:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "PATH=C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"

cmake -S %~dp0.. -B %~dp0..\build-win-hip-ninja -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  "-DCMAKE_CXX_FLAGS=-isystem \"%~dp0.\"" ^
  -DGGML_HIP=ON ^
  -DGGML_VULKAN=OFF ^
  -DGGML_CUDA=OFF ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 ^
  -DGPU_TARGETS=gfx1151 ^
  -DGGML_HIP_FORCE_MMQ=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_WEBUI=OFF ^
  -DLLAMA_USE_PREBUILT_WEBUI=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DGGML_BUILD_TESTS=OFF
if errorlevel 1 exit /b 1

cmake --build %~dp0..\build-win-hip-ninja --target llama-server -j 16
if errorlevel 1 exit /b 1
echo BUILD_OK
```

Output: `build-win-hip-ninja\bin\llama-server.exe`.

## Why the `-isystem` shim (headers in this directory)

MSVC STL 17.11+ (GH-4609) declares the `isgreater` family as constexpr clang
builtins in `<cmath>`. Under HIP, constexpr functions are implicitly
host+device, which collides with the explicit `__device__` declarations in
ROCm's `__clang_hip_cmath.h` / `__clang_cuda_math_forward_declares.h`
("cannot overload __host__ __device__ function"). The patched copies here
guard those declarations behind an `_MSVC_STL_UPDATE` check; `-isystem` makes
them shadow the resource-dir originals.

Do **not** work around it with `-fno-cuda-host-device-constexpr` instead —
that breaks `initializer_list` range-for loops in device code (fattn kernels).

## Relinking while a server runs

`llama-server.exe` cannot be relinked while an instance is running (link lock).
Stop all llama-server processes before rebuilding.

## Known Windows runtime notes

- On Strix Halo UMA, `--no-mmap` is required in launchers (mmap loader stalls).
- After laptop sleep/resume, ROCm state can be corrupted (MTP kernels fail
  with "unspecified launch failure"); reboot before loading models.
