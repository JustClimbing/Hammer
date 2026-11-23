# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## What this repo is

HAMMER (CVPR 2026): 3D affordance grounding where a Qwen2.5-VL-3B MLLM predicts an affordance token (`[CONT]`) whose hidden state is lifted into a PointNet++ point branch to produce a per-point affordance mask. Benchmarks: PIADv1 / PIADv2 (and PIADv1-C, not wired into train/eval).

## Commands

Environment (CUDA 11.8 path, as in README):
```bash
conda create -n hammer python==3.10 && conda activate hammer
pip install -r requirements.txt
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
`pointnet2_ops` is a hard import inside `src/models/point_model/point_utils.py`; nothing imports without it.

Data preprocessing (PIADv1 train split point affordances → `.npy` + `Point_Extracted_{split}.txt`):
```bash
python tools/preprocess_piadv1.py --data_root ./data --split train --setting Seen
```

Training (POSIX shell, deepspeed launcher). `-d` dataset, `-p` setting, `-g` #GPUs, `-b` global batch, `-l` lr, `-e` epochs, `-n` experiment name:
```bash
bash scripts/train.sh -d PIADv1 -p Seen -g 4 -b 64 -l 0.0001 -e 30 -n exp1
```
Direct equivalent (no code snapshotting): `deepspeed train.py --dataset=PIADv1 --setting=Seen --dataset_dir=./data --exp_name=... --lr=... --epochs=... --batch_size=1 --steps_per_epoch=... --grad_accumulation_steps=... --ce_loss_weight=1 --mask_loss_weight=2 --print_freq=20`

Evaluation on a merged checkpoint (`model.bin`):
```bash
python tools/eval.py --weight ckpt/PIADv1_Seen/model.bin --dataset PIADv1 --setting Seen
# add --limit N for a quick sanity run, --init_from_config --device_map auto to split across GPU/CPU
```

Export merge (deepspeed shards → HF-style `model.bin`):
```bash
bash scripts/merge_weights.sh   # edits EXP / SAVE_PATH inside the script first
# step 1: cd exp/$EXP/model && python zero_to_fp32.py . ../pytorch_model.bin
# step 2: python tools/merge_weights.py --weight=exp/$EXP/pytorch_model.bin --save_path=ckpt/$SAVE_PATH
```

Tests (unittest, no pytest dependency):
```bash
python -m unittest tests.test_afford_dataset -v
python -m unittest tests.test_afford_dataset.PIADv1AffordPathTest.test_windows_paths_load_a_test_sample
```

## Architecture

Forward pass lives in `src/models/hammer.py::Hammer.model_forward`; everything else is support code.

1. **Batch assembly** (`src/utils/afford_dataset.py::collate_fn`). Each dataset item is `(image, points[C,N], afford[N,k], conversation_str, inference_flag)`. The conversation already contains the full `<|im_start|>user ... <|im_start|>assistant ...` chat text (built via `src/utils/conversation.py::get_default_conv_template`). `collate_fn` expands one `<|image_pad|>` placeholder into `num_placeholders = T*H*W / merge_size**2` tokens per sample, tokenizes, builds `labels` by keeping only tokens between `<|im_start|>assistant\n` and the next `<|im_end|>` (everything else `-100`), and emits `offset` (cumulative conversation counts, so per-sample token slices can be recovered later). `offset` must have length `B+1`.

2. **VLM pass**: `self.vlm.forward(..., output_hidden_states=True)`; last hidden state goes through `seg_model.text_hidden_fcs` (Linear→ReLU→Linear to `vlm_out_dim=256`).

3. **[CONT] token extraction**: `[CONT]` is added to the tokenizer before model construction (`train.py`, `tools/eval.py`, `tools/merge_weights.py` all do `tokenizer.add_tokens("[CONT]")` and record `cont_token_idx`), and the embedding is resized accordingly. Its embedding from the last hidden state is the *only* thing passed to the geometric branch — there is no separate text encoder. `cont_token_mask` + `offset` reshape the flat `[#cont_tokens, C]` tensor into `[B, 1, C]`, then `projection` maps 256→512.

4. **Point branch** (`SegModel` inside `hammer.py`):
   - `PointBackbone`: PointNet++ MSG SA1(512 with radii 0.1/0.2/0.4) → SA2(128) → SA3(64). At SA3 output a `CrossAttnBlock` lets each point query the VLM hidden states, then `feat_proj` fuses a softmax-attention-pooled sentence vector into `feat_0`. FP3/FP2/FP1 propagate back to the full 2048-point cloud. Returns `[feat_3(64), feat_2(128), feat_1(512), feat_0(2048)]`, each 512-d, **coarse→fine**.
   - `LiftTo3D`: 4 stacked `CrossAttnBlock`s; the `[CONT]` embedding is refined successively against `point_embedding[0..3]` (coarse→fine).
   - `AffordDecoder`: cross-attends the refined token onto `point_embeddings[-1]` (the finest, full-resolution features, so output index order matches the input `.npy`/txt point order), then a 1-channel head + sigmoid → per-point affinity in `[0,1]`.

5. **Loss** = `ce_loss_weight * output.loss` (LM loss on the answer span) + `mask_loss_weight * ca_loss(pred_afford, gt_affords)` (`src/models/losses.py`: focal + symmetric dice). Defaults 1.0 / 2.0, set on the CLI and passed into the model kwargs.

## Training / evaluation pipeline

`train.py` parses args → builds processor/tokenizer (`min_pixels`/`max_pixels` control the vision token budget) → constructs `Hammer` → freezes `vlm.visual` → wraps `vlm` in LoRA (`q_proj,v_proj`, r16) → sets `requires_grad` manually for `lm_head`/`embed_tokens`/`text_hidden_fcs`/`projection`/`point_backbone`/`lift_to_3d`/`afford_decoder`. Optimizer/scheduler are Deepspeed ZeRO-2 configs built inline in `main()`.

- Training is **step-based**: the train dataset `__len__` returns `samples_per_epoch = batch*grad_accum*steps_per_epoch*world_size` and `__getitem__` samples randomly, so one epoch = `steps_per_epoch` optimizer steps. `scripts/train.sh` hardcodes the epoch sizes: `12565/global_batch` for PIADv1, `37120/global_batch` for PIADv2. Changing a dataset's size means changing those numbers.
- `scripts/train.sh` copies `scripts/ src/ tools/ train.py` into `exp/<EXP_NAME>/code` and runs deepspeed from there (`PYTHONPATH` points at the copy), so edits inside a running experiment directory are not picked up — rerun the script.
- Validation runs every epoch via `tools/trainer.py::validate`, returns aIoU; the best checkpoint is saved to `exp/<EXP_NAME>/model` (deepspeed format) with a sibling `meta_log_iou_<score>.pth`. `--auto_resume` / `--resume` read the `latest` file inside that dir to derive `start_epoch`.
- Logging: TensorBoard + wandb into `exp/<EXP_NAME>`; `--debug` disables wandb, and non-master ranks always disable it.
- `tools/eval.py` reuses `tools/trainer.py::evaluate`, computing AUC / aIoU / SIM / MAE. aIoU is averaged over 20 thresholds from `np.linspace(0, 1, 20)`; samples whose GT mask is all-0 or all-1 are dropped from AUC/aIoU (NaN) but still counted in SIM/MAE. It loads a **merged, non-LoRA** state dict with `strict=True`, so the weights must come from `tools/merge_weights.py`.

## Dataset specifics (they are not interchangeable)

- `PIADv1Afford` (dataset=`PIADv1`, settings `Seen` / `Unseen`): train points come from preprocessed `Point_Extracted_train.txt` `.npy` files holding `xyz + 1` affordance column; test points are raw `.txt` with `xyz + 17` columns and are sliced to the label index parsed from the path (`get_affordance_index`). Expects `Point_Extracted_train.txt` (from `tools/preprocess_piadv1.py`), `Point_test.txt`, `Img_train.txt`, `Img_test.txt`.
- `PIADAfford` (dataset=`PIADv2`, settings `Seen` / `Unseen_afford` / `Unseen_obj`): both splits use `.npy` files, and every `(obj, affordance)` path segment index differs by one from v1 (`[-4]` vs `[-3]`) — README explicitly says the logic diverges from GREAT. Expects `Img_{split}.txt` and `Point_{split}.txt` with val variants.
- Both readers strip a leading `Data/` prefix in the list files and rebuild absolute paths as `data_root/<Dataset>/<setting>/...`; `PIADv1Afford` also normalizes `\` to `/` for Windows-produced list files (see `tests/test_afford_dataset.py`).
- `question_type` (`simple` / `afford` / `afford_obj`) selects the Q&A template lists in `src/utils/utils.py`; answers must contain exactly one `[CONT]` (that is what the collate step and embedding extraction assume).

## Leftovers worth knowing

- `src/models/decoder/afford_decoder.py::MaskDecoder` and parts of `src/models/decoder/transformer.py` (`Transformer`, `TwoWayAttentionBlock`) are unused; `hammer.py` defines its own `AffordDecoder`/`CrossAttnBlock`.
- `src/utils/geal_dataset.py` (`GEAL`) is not referenced by `train.py` or `tools/eval.py`.
- `scripts/*.sh` are bash scripts with hardcoded `~/local/cuda-11.8` paths and `RANDOM`-based `MASTER_PORT`; on Windows run the `python`/`deepspeed` commands directly.
