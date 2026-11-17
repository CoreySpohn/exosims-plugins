# Changelog

## [2.0.0](https://github.com/CoreySpohn/exosims-plugins/compare/v1.1.0...v2.0.0) (2026-11-17)


### ⚠ BREAKING CHANGES

* scale exozodi from the EXOSIMS 3.6.6 one-AU reference and require EXOSIMS 3.6.6
* scale exozodi with luminosity, look up exposure bins in days, treat the AYO noise floor as a point-source contrast, and key dMag0 caches on full axes

### Features

* absorb dMag0 grid from orbix with x64 and OWA-edge fixes ([7f08b96](https://github.com/CoreySpohn/exosims-plugins/commit/7f08b963bbc1875ded1c2e2deda0dd48e3e2045e))
* absorb Planets from orbix; de-jit class, filter_jit methods, fix save/load fields ([5705668](https://github.com/CoreySpohn/exosims-plugins/commit/5705668c5985a51d1fbddd2aee67169aa7e1ce37))
* Create ExoversesUniverse, add SNR margins and observation export, and fix Completeness cache issue ([b1c21a4](https://github.com/CoreySpohn/exosims-plugins/commit/b1c21a4adf3c38d28e408d69573f4d5e6aabe74d))


### Bug Fixes

* read survey detector and exozodi settings by their mapped names instead of always using defaults ([53b673e](https://github.com/CoreySpohn/exosims-plugins/commit/53b673ee6ca4548f0d91f0ce2c40945c7f6cf100))
* scale exozodi from the EXOSIMS 3.6.6 one-AU reference and require EXOSIMS 3.6.6 ([927943a](https://github.com/CoreySpohn/exosims-plugins/commit/927943a8b6bfd2d03ca0209c2cf49072aa947d71))
* scale exozodi with luminosity, look up exposure bins in days, treat the AYO noise floor as a point-source contrast, and key dMag0 caches on full axes ([109538b](https://github.com/CoreySpohn/exosims-plugins/commit/109538b31a98592593bf70e5998215109fccd3c7))
* use the orbix Grid solver in lazily imported simulation paths ([584291e](https://github.com/CoreySpohn/exosims-plugins/commit/584291e8ae3efe42ec887842b0201e5016cdac9e))

## [1.1.0](https://github.com/CoreySpohn/exosims-plugins/compare/v1.0.2...v1.1.0) (2025-12-20)


### Features

* Moving to completeness/observing time instead of completeness/int time. Adding additional logging information ([f1f28ba](https://github.com/CoreySpohn/exosims-plugins/commit/f1f28ba30be7c4d448e9cb1b62e9b45a7b763266))

## [1.0.2](https://github.com/CoreySpohn/exosims-plugins/compare/v1.0.1...v1.0.2) (2025-12-16)


### Bug Fixes

* Update trig_solver to use the proper level with recent orbix updates ([3d13aa3](https://github.com/CoreySpohn/exosims-plugins/commit/3d13aa3a16ee089110d9762ddc4e5ac1ad328b6e))

## [1.0.1](https://github.com/CoreySpohn/exosims-plugins/compare/v1.0.0...v1.0.1) (2025-12-15)


### Bug Fixes

* Add intervaltree dependency to pyproject.toml ([09225f5](https://github.com/CoreySpohn/exosims-plugins/commit/09225f547749689eee094e20c1a0e84e354604ef))

## 1.0.0 (2025-10-07)


### Features

* Full implementation of orbix modules ([18450cb](https://github.com/CoreySpohn/exosims-plugins/commit/18450cb083a4d8d93f267d7f15923f3ec433efce))
