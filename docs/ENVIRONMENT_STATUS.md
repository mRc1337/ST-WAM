# Local reproduction environment status

Last checked: 2026-09-03 UTC.

## Checkouts

- ST-WAM: `/home/pai/zxw/ST-WAM`
  - commit: `f20d1de0d334fda76e60118f3da5bb908a0bc1e9`
- LIBERO-Plus: `/home/pai/zxw/LIBERO-plus`
  - commit: `4976dc30028e805ff8094b55501d532c48fec182`

## Configured local environment

- Virtual environment: `/home/pai/zxw/ST-WAM/.venv`
- Activation: `source scripts/activate_libero_plus_env.sh`
- LIBERO config: `.runtime/libero/config.yaml`
- ST-WAM and LIBERO-Plus source packages are installed editable.
- `PYTHONPATH`, `LIBERO_ROOT`, `LIBERO_PLUS_ROOT`, `LIBERO_DATA_ROOT`, cache paths, and headless EGL variables are configured by the activation script.

The current `.venv` uses Python 3.10, PyTorch `2.7.1+cu128`, torchvision
`0.22.1+cu128`, MuJoCo `3.3.2`, and robosuite `1.4.0`. Core ST-WAM imports
work. Four A800 80GB GPUs, CUDA BF16 DINO inference, NCCL, and model
construction have been verified.

## Remaining setup gate

`future==1.0.0` and `matplotlib==3.10.9` are installed. Python `wand`, the
LIBERO-Plus benchmark, and the ImageMagick/MagickWand system dependency are
available. EGL reset/render has passed for both 256×256 cameras. Formal
training has not started; verify that all four GPUs satisfy the free-memory
gate immediately before launching it.

## Data and artifact gate

Local LeRobot v3 training data is configured at
`/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero` via
`configs/data/libero_2cam_v3.yaml`. The real-data validator passes with 2,000
selected episodes and 338,575 frames. `libero_100` task indices 0-9 provide the
500-episode LIBERO-Long subset.

The official `assets.zip` was integrity-checked and merged into
`/home/pai/zxw/LIBERO-plus/libero/libero/assets`. The resulting tree is about
11GB with 449,909 files, including 432,482 files under `new_objects/`. All four
benchmark suites import and enumerate to 10,030 cases (2,402 Spatial, 2,518
Object, 2,591 Goal, and 2,519 Long).

The Wan2.2 DiT/VAE/T5/tokenizer payloads, both generated 1B initialization
checkpoints, and the mandatory DINOv3 ViT-S/16 backbone are present and have
passed GPU construction/forward checks. The optional official LIBERO DINO
feature cache and released Fast-WAM/DINO-only trained checkpoints remain
absent.

`scripts/precompute_text_embeds.py` now reads both LeRobot v2 `tasks.jsonl` and
v3 `tasks.parquet`; all 40 selected prompts are cached. Common normalization
statistics for 2,000 episodes and 338,575 transitions were generated at
`artifacts/libero_v3/dataset_stats.json`. A real processed sample passed with
video `(3,9,224,448)`, action `(32,7)`, proprio `(32,8)`, context
`(128,4096)`, and context mask `(128,)`.

Follow `docs/LIBERO_PLUS_ABLATION_REPRODUCTION.md` for sources, expected paths, training commands, and evaluation gates.
