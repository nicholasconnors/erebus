# Changelog

## 0.8.0

- Added helper methods to run Eureka stage 1 and 2 when starting from uncal files before using Erebus.
- Added setting to fit no eclipse
- Added settings for gaussian/uniform prior on eclipse timing
- Record log likelihood and BIC
- Allow setting custom systematic model for Erebus to detrend with
- Numerous bug fixes
- Version used in Connors et al. 2026

## 0.7.1

- Now uses ecosw and esinw instead of e and w
- Add optional setting to fix eclipse timing
- Improved convergence checking
- Version used in Connors et al. 2025

## 0.7.0

- Pre-processing of `calints` data for NaN and outlier rejection.
- Light curve fitting using FN-PCA, exponential ramp, or custom systematic model
- Fits for orbital parameters within provided errors. Support for `t0` predictions from lookup file.
