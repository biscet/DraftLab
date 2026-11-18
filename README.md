# DFlash Coach

Version 2.0.0. [Russian instructions](README.ru.md).

> [!NOTE]
> **Experimental community release.** This release is intended for research
> and hands-on experiments with draft-model training and speculative decoding.

DFlash Coach captures features from a GGUF target model, fine-tunes a DFlash2
draft and compares the result with its parent. DeepSeek Flash reviews validation
metrics after each shard. It can change the learning rate or stop training.

The included training backend is specific to a Qwen3.8 35B A3B-compatible recipe.
It is not a general trainer for every DFlash model. GPU training requires Linux
and CUDA. The offline demo, status command and most tests also work on macOS.

## What is included

- Capture and token-count tools for a compatible llama.cpp fork.
- A full draft trainer with checkpoint, optimizer, scaler and RNG recovery.
- Uniform and D-PACE training objectives for DFlash2.
- Offline Whittle EAGLE-3 capture, training and GGUF export for a 32 GB V100.
- A controller that processes feature files in bounded shards.
- DeepSeek review, decision history and learning-rate changes.
- A benchmark for no-draft, parent draft and candidate draft.
- Tests, synthetic demo inputs and public dataset examples.

No model weights, private datasets, training logs or credentials are included.
The browser dashboard is not implemented in this version.

## Try the offline demo

Python 3.10 or newer is enough. No packages or API key are needed.

```bash
python3 coach.py demo --run-dir ../dflash-coach-demo
python3 coach.py status --run-dir ../dflash-coach-demo
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
```

The demo writes synthetic files outside the source folder. It does not train a
model or call DeepSeek. Tests that need PyTorch are skipped when it is absent.
Choose a new demo directory for another run.

## Train a DFlash2 draft

Follow the [English guide](docs/guide.en.md). It covers the runtime build, CUDA
environment, data format, API key, preparation, launch, resume and evaluation.
The [Russian guide](docs/guide.ru.md) covers the same steps.

Start with `configs/example.json`. Copy it to `configs/training.local.json`
before changing paths. Paths in JSON are resolved relative to that JSON file.
The small dataset example uses `configs/lite.example.json` instead.

Set `"training_objective": "dpace"` before preparing a new run to use
D-PACE. The default is `uniform`. The DFlash2 guide explains how D-PACE
weights token and selector losses.

## Train a Whittle EAGLE-3 draft

`python3 coach.py eagle3 --help` lists the capture, training and export stages.
The [EAGLE-3 guide](docs/eagle3.en.md) covers setup, preparation of frozen target
tables, and resuming an offline run with SpecForge and llama.cpp.

The research notes in ML-Docs cover the original D-PACE experiment and the
Whittle EAGLE-3 feature format. [ML-Docs](https://github.com/biscet/ML-Docs)

## Public dataset examples

- [DFlash DeepSeek TraceMix](https://huggingface.co/datasets/biscet/DFlash-DeepSeek-TraceMix).
  Use `deepseek` for the main task splits or `stage2` for the separate 272-context
  training supplement with the main validation and test splits.
- [DFlash ChatGPT TraceMix Lite](https://huggingface.co/datasets/biscet/DFlash-ChatGPT-TraceMix-Lite).
  The `lite` example uses OpenMix. It reserves part of the source validation
  families for testing because that subset has no separate test file.

Only links, pinned revisions and download settings are stored here. Dataset
records are downloaded only when the user runs `download-example`. Existing
completions are discarded. Labels and features for training come from the
configured target model.

## Limits of this version

The portable controller has CPU and mocked integration tests. Its training
backend comes from an existing CUDA implementation. A complete GPU run of this
portable package is still needed before treating a new setup as verified.

The recipe fixes the feature layers, vocabulary and draft architecture in code.
Changing `target_model` alone does not make another architecture compatible.
The example stops after at most one full pass; DeepSeek can stop earlier. Loss
and prefix metrics are offline proxies. They do not establish serving speed or
response quality. The candidate is not installed into an inference service.

## Versions and licenses

The architecture is documented in the
[Russian architecture guide](docs/architecture.ru.md).

`main` contains v2.0.0. The `v1.0.0` branch preserves the first release.
See the [changelog](CHANGELOG.md) and [release steps](docs/releases.md).

Original utilities use Apache-2.0. The upstream DFlash model implementation uses
MIT. See [NOTICE](NOTICE), [LICENSE](LICENSE) and [DFlash MIT](licenses/DFlash-MIT.txt).
