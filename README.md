# FLUX.2 [dev] (32B) QLoRA on a single 24GB AMD RX 7900 XTX — native ROCm, Windows

Working recipe + patches for QLoRA-training the **32-billion-parameter FLUX.2 dev** transformer on one
**24GB AMD card** under **native ROCm on Windows** (no ZLUDA, no CUDA shim). Community consensus is that
>~13B QLoRA is past the 24GB ceiling; this is 32B, so every wall below is a "past the ceiling" problem.

**Status:** trains resident on the GPU, ~20 s/step @ 512 res, loss falling.

## If you've been stuck on one of these, this doc is for you

- **"QLoRA past ~13B doesn't fit in 24GB"** — the standing consensus this recipe gets past (32B, resident).
- **"You need 64GB+ of system RAM just to load the base"** — no: this whole recipe runs on **32GB of RAM**, paging the one-time 64GB bf16 load (and the 48GB TE on top of it) through a ~100GB pagefile. Walls 1–4 are the load-order surgery that makes that survivable.
- **ComfyUI/ROCm generation deadlocks** (VRAM fragmentation — can't-allocate hangs, unrecoverable via interrupt): the fix the community circulated ([ComfyUI#12672](https://github.com/comfyanonymous/ComfyUI/issues/12672), `expandable_segments`) is load-bearing for training too — wall 7.
- **The "GPU 100% busy, ~1/8 throughput, driver looks healthy" family** ([ROCm#6012](https://github.com/ROCm/ROCm/issues/6012), [#6165](https://github.com/ROCm/ROCm/issues/6165), [#2689](https://github.com/ROCm/ROCm/issues/2689)): Part 2 is a working detection + auto-rescue system for the training-time version — and we've since pinned the trigger itself to a deterministic two-process collision that anyone with an RDNA3 card (AMD engineers included) can confirm in ~15 minutes. (Formal upstream report with the full repro + negative controls: [TheRock#6824](https://github.com/ROCm/TheRock/issues/6824). The full root-cause story: [companion post](https://www.reddit.com/r/ROCm/comments/1v4r0mh/).)
- **Waiting for fused int4 training kernels on RDNA3?** Don't — torchao/tinygemm gates RDNA3 out at the arch check, sglang compiles only gfx942/950, GemLite and Marlin are NVIDIA-only, and the working AMD int4 stacks (AWQ/vLLM, llama.cpp) are inference-only with no backward pass. For training on gfx1100 there is currently nothing fused — this recipe accepts quanto's dequant tax instead of waiting.
- **"My trainer is secretly running on CPU"** — maybe it isn't; cross-process VRAM reads lie on ROCm/Windows (wall 10). Check board wattage before you kill a healthy run.
- **`hf download` stalling at 0 bytes on the 64GB base** — HuggingFace's Xet storage backend does this on big blobs (hit it on both `black-forest-labs/FLUX.2-dev` and `Comfy-Org/flux2-dev`; small LFS files come down fine). Fix: `curl -L -C -` the `https://huggingface.co/<repo>/resolve/main/<file>` URL directly (add `-H "Authorization: Bearer <token>"` for gated repos) — it follows the redirect to the CAS bridge and resumes properly.
- **Hunting for flash-attention builds on ROCm/Windows** — stop: this stack has no flash-attn ROCm kernels, attention runs on the Triton fallback path, and forcing `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` tested as a no-op-or-worse for Flux.2 training. Budget for the Triton path; it works.

## Environment
| | |
|---|---|
| GPU | RX 7900 XTX 24GB (gfx1100) — no FP8/FP4 hw → **uint4** (optimum-quanto) |
| Host | 32GB RAM + ~100GB pagefile (needed for one-time bf16 loads) |
| torch | 2.12.0+rocm7.15 |
| trainer | ai-toolkit, AMD ROCm fork (`cupertinomiranda/ai-toolkit-amd-rocm-support`) |
| base | official FLUX.2-dev, **repo-root single-file** `flux2-dev.safetensors` (64GB), NOT the diffusers `transformer/` subdir |
| text encoder | Mistral-Small-3.1-24B (FLUX.2 uses a 24B LLM as its TE) |
| eval | ComfyUI + ComfyUI-GGUF (Q3 GGUF unet + Mistral Q5 GGUF, `type=flux2`) |

## The walls, in order (each blocks the next)

**Load & quantize 64GB without dying**
1. `safetensors` mmap `load_file` crashes natively on the 64GB file (no traceback). → manual non-mmap loader (below).
2. Transformer OOMs at ~38GB before quantizing (trainer moves full bf16 to GPU first). → **quantize on CPU**.
3. `0xC0000005` loading the TE (64GB bf16 still referenced). → `del transformer_state_dict; gc.collect()`.
4. Mistral OOMs the GPU (c10 abort). → quantize Mistral on CPU first, then `.to(device)`.

**Make it train on the GPU, not the CPU**
5. Block-swap (`layer_offloading`) **deadlocks the HIP driver**. → `layer_offloading: false`.
6. In-training sampling deadlocks + uint4 previews are black frames. → `disable_sampling: true`, eval in ComfyUI.
7. uint4→GPU move fragments/OOMs. → `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
8. Re-quantizing every launch = ~8 min. → cache the quanto state-dict once; `torch.load` in seconds.
9. Base won't stay resident (looks like CPU training) — `low_vram` parks it and never moves it back. → move it back after the TE unloads (below).

**The two that cost me a whole night**
10. **"It's training on CPU" — except it wasn't.** A *separate process* reading `torch.cuda.mem_get_info()` **lies** on ROCm/Windows (reported 0.2GB while the process held 20GB). "1 busy CPU core" is normal for GPU training. I killed several *working* runs over this. → Trust an **in-process** VRAM print, the Windows `\GPU Engine(*engtype_compute)\Utilization` perf counter, and the **drop in system RAM** when the base leaves CPU.
11. Resident but crawling at 200 s/step — the 20GB base leaves no headroom, activations spill to host RAM over PCIe. → cut resolution until the spill is small.

## Resolution is the speed knob (batch 1, grad-checkpointing on)
| Max res | Host spill (PCIe) | Step time |
|--------:|------------------:|----------:|
| 1024    | 2.27 GB           | ~204 s    |
| 768     | 0.82 GB           | ~79 s     |
| 512     | 0.83 GB           | ~20–40 s  |

768 & 512 spill the **same** ~0.8GB — fixed overhead the allocator won't reclaim (leaves ~0.6GB VRAM unused), not activations. The 768→512 gain is purely less compute. Identity trains fine at 512.

## config.yaml (the knobs that matter)
```yaml
model:
  name_or_path: "/path/to/flux2/dir"   # holds flux2-dev-uint4.pt (the cache) + ae.safetensors
  arch: "flux2"
  quantize: true
  qtype: "uint4"          # quanto; also drives the TE quant in this fork
  quantize_te: true
  qtype_te: "uint4"
  low_vram: true          # park base during TE-cache, then move it back resident to train
  layer_offloading: false # block-swap DEADLOCKS on ROCm — do not enable
  model_kwargs:
    use_uint4_cache: true # load the pre-quantized flux2-dev-uint4.pt in seconds
datasets:
  - folder_path: "/path/to/images"
    caption_ext: "txt"
    resolution: [ 512 ]                 # 512 ≈ 20 s/step; 768/1024 spill and crawl
    cache_latents_to_disk: true
    cache_text_embeddings: true         # precompute + unload the 24B TE
train:
  batch_size: 1
  gradient_checkpointing: true
  disable_sampling: true                # judge in ComfyUI, not in-training
  optimizer: "adamw8bit"
  lr: 1e-4
  dtype: bf16
  steps: 2000
```

**Launch** — set both allocator env vars and run python **directly** (a detached
`Start-Process -RedirectStandardOutput` silently loses early output if the child dies during import;
vcvars64 is NOT required — torch+ROCm import fine without it):
```bat
set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u run.py config\your_config.yaml
```

## The patches (to the fork's `extensions_built_in/diffusion_models/flux2/flux2_model.py`)

**Wall 1 — manual non-mmap loader** (replaces `load_file` for the 64GB base):
```python
import struct as _struct, json as _json
_DT = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
       'F8_E4M3': torch.float8_e4m3fn, 'U8': torch.uint8, 'I8': torch.int8}  # extend as needed
transformer_state_dict = {}
with open(transformer_path, 'rb') as fh:
    n = _struct.unpack('<Q', fh.read(8))[0]
    hdr = _json.loads(fh.read(n)); base = 8 + n
    for k, meta in hdr.items():
        if k == '__metadata__':
            continue
        o0, o1 = meta['data_offsets']
        fh.seek(base + o0)
        raw = fh.read(o1 - o0)
        transformer_state_dict[k] = torch.frombuffer(bytearray(raw), dtype=_DT[meta['dtype']]).reshape(meta['shape'])
transformer.load_state_dict(transformer_state_dict, assign=True)
```

**Walls 2 + 3 — quantize on CPU, then free the bf16:**
```python
del transformer_state_dict            # wall 3: free the 64GB bf16 so the TE fits
import gc; gc.collect()
if self.model_config.quantize:
    patch_dequantization_on_save(transformer)
    quantize_model(self, transformer) # wall 2: runs IN PLACE ON CPU (do NOT .to(gpu) first)
```

**Wall 4 — quantize Mistral on CPU** (in `load_te`):
```python
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(MISTRAL_PATH, torch_dtype=dtype)
if self.model_config.quantize_te:
    quantize(text_encoder, weights=get_qtype(self.model_config.qtype))  # on CPU
    freeze(text_encoder)
    text_encoder.to(self.device_torch)   # only the ~12GB uint4 result moves
```

**Wall 8 — the uint4 cache branch** (skip re-quant if the cache exists):
```python
_cache = os.path.join(model_path, "flux2-dev-uint4.pt")
if self.model_config.model_kwargs.get("use_uint4_cache", False) and os.path.exists(_cache):
    sd = torch.load(_cache, weights_only=False, map_location="cpu")
    transformer.load_state_dict(sd, assign=True, strict=False); del sd
    patch_dequantization_on_save(transformer)
    transformer.to(self.quantize_device)
# else: run the manual-loader + quantize path above
```
(Build the cache once by saving `transformer.state_dict()` after quantizing.)

**Wall 9 — keep the base parked during TE-cache, don't collide.** In `load_model`, gate the
transformer→GPU move so it doesn't fight the resident 12GB TE mid-caching:
```python
if not self.model_config.low_vram:
    pipe.transformer = pipe.transformer.to(self.device_torch)
```
Then in `SDTrainer.py`, right after the TE is cached + unloaded, bring the base back for resident training:
```python
if self.model_config.low_vram and not self.model_config.layer_offloading:
    self.sd.model.to(self.sd.device_torch)   # the quanto move works; the framework just never did it
    flush()
```

## Evaluating a checkpoint (ComfyUI, since in-training previews are disabled)
ComfyUI + ComfyUI-GGUF, driven headless via the `/prompt` API. Node graph:
`UnetLoaderGGUF(flux2-dev-Q3_K_M.gguf)` → `LoraLoaderModelOnly(your_lora)` →
`CLIPLoaderGGUF(Mistral-Q5, type=flux2)` + `VAELoader(flux2-vae)` →
`CLIPTextEncode` → `FluxGuidance(4.0)` → `BasicGuider` + `Flux2Scheduler(20)` +
`SamplerCustomAdvanced` + `EmptyFlux2LatentImage` → `VAEDecode` → `SaveImage`.
The ai-toolkit LoRA keys (`diffusion_model…lora_A/lora_B`) load with **zero conversion**.

## Part 2 — the stall (added a week in; the biggest finding here)

**The symptom:** step rate swings 6.9 ↔ 118 s/it with clock/temp/spill flat. **The diagnostic is POWER:**

| state | core clock | power | mem-controller load |
|---|---|---|---|
| stalled (~1/8 speed) | ~3100 MHz (high!) | ~220–260 W | 2–7 % |
| saturated | ~2500 MHz | ~385 W | 23–52 % |

High clock + low watts = spinning-not-working. `device_ms ≈ wall_ms` in both states — device timers can't see it; only wattage separates them. Cause lives below anything visible from Windows (driver/scheduler).

**The fix (reproducible, with a negative control):** run a second, FRESH process doing sustained large GEMMs (6144×6144 @ 6144×18432 bf16, sync every iter) for ~25 s. The trainer flips to saturated and STAYS there after the rescuer exits. Pure alloc/free does nothing (tested); 512-dim keep-warm does nothing; it's the heavy foreign *compute* that flips it. Two traps: a long-lived rescuer context degrades alongside the trainer (in-process bursts ran at ~7 TFLOP/s on a card that does 77.5 idle — useless), and even fresh processes spawn degraded ~50% of the time — measure the burst's own TFLOP/s (<~11 = dud) and respawn immediately.

**Operationalized:** a telemetry logger (10 s cadence: clock/power/load/mem-load + the trainer's step) + a watchdog daemon that fires bursts when the stall signature holds for 2 samples, with prophylactic bursts ~60 s after process start and after checkpoint saves (stalls cluster at exactly those moments), dud-retry, a cooldown, and self-disable if fast bursts stop working. A full 2000-step run now holds ~9–10 s/it with zero human attention.

**Also coupled: batch size.** bs2 ≈ 1.45×/sample saturated (dequant amortizes) but LOSES ~1.47× while stalled — fix the stall first. bs2 fits 24 GB at 448 with ~150 MB spare; bs3 does not. Sqrt-scale LR.

**Lossless mid-run previews:** ai-toolkit's resume (checkpoint + optimizer.pt) is solid, so add flag-file checks in the train loop — `SAVE_NOW` = checkpoint now and continue; `STOP_NOW` = checkpoint now and exit cleanly. Pause → render the checkpoint in ComfyUI → relaunch (resumes at the exact step). Renders stall the same way (200 ↔ 800 s for 20 steps) — same power tell, same burst fix.

**Convergence note (n=3 subjects):** identity markers land by ~1000 steps; face/body GEOMETRY keeps truing up to 2000. "Looks like them" at 1200 is not done. Budget ~6.5 h per person all-in on this card.

## Credits — where the load-bearing pieces came from

- **[cupertinomiranda/ai-toolkit-amd-rocm-support](https://github.com/cupertinomiranda/ai-toolkit-amd-rocm-support)** — the AMD/ROCm ai-toolkit fork this whole recipe stands on (quanto uint4 path, Mistral TE wiring, ROCm patches). The single most load-bearing community contribution here.
- **[Ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)** — the upstream trainer the fork patches.
- **[TheRock (AMD)](https://github.com/ROCm/TheRock)** — nightly torch-on-Windows/ROCm wheels; without these there is no torch on Windows/ROCm at all.
- **[woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)** — the one-person Triton-on-Windows port the well-known Windows AI stacks quietly build on.
- **[st1vms/bitsandbytes-rocm-gfx1100](https://github.com/st1vms/bitsandbytes-rocm-gfx1100)** — community gfx1100 int4 groundwork (we stayed on quanto, but the map mattered).
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) + [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — the entire eval/render path.

The patches above (manual 64GB loader, CPU-quant ordering, uint4 cache, `low_vram` resident-return, the stall watchdog) are mine; use them freely.

---
Shared in the spirit of open information — corrections and reproductions welcome.
