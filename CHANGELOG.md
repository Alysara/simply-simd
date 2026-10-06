All changes in `simply-simd` are documented here.

## Unreleased

## 0.2.0 - 2026-10-06

### Added
- `exp`, `exp2`, `log2`, and `pow`, along with unchecked variants.

### Changed
- **Breaking change**: `simd_neq` renamed to `simd_ne`.
- Added `StaticArch` as the default `Arch` for the `Simd` struct.

### Fixed
- Incorrect results for `all_false()` in the scalar fallback.

## 0.1.0 - 2026-07-25

Initial release.

