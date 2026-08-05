# Publishing macOS wheels

Build with `build_scripts/build_wheel_macos.sh`, then `twine upload wheelhouse/*.whl`.

## Why the raw wheel is not publishable

`_tempest.cpython-*-darwin.so` links Homebrew LLVM's OpenMP runtime and Homebrew
TBB by absolute path:

```
/opt/homebrew/opt/tbb/lib/libtbb.12.dylib
/opt/homebrew/opt/llvm/lib/libomp.dylib
```

On a machine without those exact Homebrew kegs, `import tempest` fails in dyld.
`delocate-wheel` (the macOS counterpart to `auditwheel`) copies both dylibs into
`tempest/.dylibs/` inside the wheel and rewrites the install names to
`@loader_path/...`, leaving only OS-guaranteed libraries:

```
/usr/lib/libSystem.B.dylib
/usr/lib/libc++.1.dylib
tempest/.dylibs/libomp.dylib
tempest/.dylibs/libtbb.12.15.dylib
```

Homebrew LLVM and TBB stay **build-time** requirements. Users need neither.

## The platform tag is a hard floor

Two independent things set the minimum macOS version, and the stricter wins:

1. **The building Python.** The wheel tag comes from that interpreter's
   `MACOSX_DEPLOYMENT_TARGET`. Homebrew's `python@3.13` is built with `15`, so it
   can only produce `macosx_15_0_*` wheels — exporting a lower
   `MACOSX_DEPLOYMENT_TARGET` is silently ignored. Use python.org CPython (or the
   builds cibuildwheel downloads, target 11.0 on arm64) to go lower.
2. **The vendored dylibs.** Homebrew bottles are compiled for the host OS, so on
   macOS 15 `libomp.dylib` and `libtbb` are both `minos 15.0`. delocate verifies
   this and refuses to mislabel the wheel:

   ```
   DelocationError: Library dependencies do not satisfy target MacOS version 11.0:
     .../libomp.dylib has a minimum target of 15.0
     .../libtbb.12.15.dylib has a minimum target of 15.0
   ```

To support older macOS, either **build on the oldest OS you want to support** (the
Homebrew bottles then match that version automatically — the cheap fix), or build
libomp and TBB from source with `MACOSX_DEPLOYMENT_TARGET` set before building the
extension.

## Coverage

One run of the script produces one wheel: current arch, current interpreter. Full
coverage needs a matrix — `cibuildwheel` on GitHub Actions is the standard way,
and it invokes delocate for you:

| Runner | Arch | Notes |
|---|---|---|
| `macos-13` | x86_64 | Intel Macs; bottles are `minos 13` |
| `macos-14` | arm64 | Apple Silicon; bottles are `minos 14` |

Set `CIBW_BEFORE_ALL_MACOS="brew install llvm tbb"`. `universal2` wheels are not
practical here: Homebrew ships single-arch dylibs, so there is nothing to vendor
for the non-native slice.

## Known caveat: duplicate OpenMP runtime

Shipping `libomp.dylib` inside a wheel is standard practice (scikit-learn,
LightGBM, XGBoost all do it) but has a known hazard. If the user's process also
loads another OpenMP runtime — PyTorch and scikit-learn each bundle their own —
they may hit:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

There is no clean fix from inside a wheel; a static libomp is not available from
Homebrew, and Apple's `libc++` toolchain offers no `-static-openmp`. If this turns
out to affect users in practice, the options are to replace the 78 `#pragma omp`
regions with `std::thread`/`std::async`, or to document `KMP_DUPLICATE_LIB_OK=TRUE`
as an escape hatch (it suppresses the abort but is not officially safe).

## Note on `-D_GLIBCXX_USE_CXX11_ABI=0`

This flag exists for Linux/libstdc++ ABI pinning and is inert on macOS, which uses
libc++. Homebrew clang links the system `/usr/lib/libc++.1.dylib`, not its own, so
there is no C++ standard library to vendor.
