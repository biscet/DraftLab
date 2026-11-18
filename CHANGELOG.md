# Changelog

## 2.0.0

- Added D-PACE as an optional DFlash2 training objective.
- Added offline Whittle EAGLE-3 capture, training and export for V100, with patches for SpecForge and llama.cpp.
- Updated the Lite dataset profile and guides to DFlash ChatGPT TraceMix Lite.
- Grouped CLI commands and application code by domain around the coach.py entry point.
- Split preparation, preflight, mentor reviews, training and benchmark into focused stages.
- Separated native capture orchestration, feature extraction and metadata I/O.
- Separated portable configuration, storage, planning, metrics and mentor evidence.
- Split training sample loading, objective computation, model I/O and weight snapshots.
- Extracted DFlash layers and sampling while preserving model state keys and entry points.
- Centralized recipe constants and unified Python/C++ formatting.
- Included nested Python modules in frozen plans.

## 1.0.0

- Added a portable configuration for capture, training and comparison.
- Added DeepSeek validation review and learning-rate decisions.
- Added shard planning, checkpoint continuation and inference service restoration.
- Added a standalone CUDA trainer and GGUF capture sources.
- Added CPU tests, an offline demo and a publication content check.
- Added English and Russian instructions.
- Added pinned examples for two public Hugging Face datasets.

This release contains source code and examples. It contains no private run
results, datasets, model files or credentials. Full GPU acceptance of the
portable package remains to be done.
