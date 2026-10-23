# The LoRA/LoKR Creation Guide for ACE-Step 1.5

**Scope.** How an annotated, preprocessed dataset becomes a working LoRA or LoKR
adapter: what the adapter is allowed to learn, what a timestep means, what the loss
measures, how the optimiser behaves, and where the artefact lands. Every claim below
was read out of the ACE-Step source rather than inferred, and each carries a
`file:line` reference so you can re-check it against your own install.

**Not in scope.** What to write in a caption or a lyric file. That is
[`Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md`](Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md).
This guide assumes annotation and preprocessing are already done. The annotation
guide's section 8 keeps the handful of trainer facts that change *how you annotate*
and points back here for the mechanism behind each one.

**Two stacks.** `acestep/training/` is the legacy stack behind the Gradio tabs and the
HTTP API; `acestep/training_v2/` ("Side-Step") is the CLI stack. They are not
interchangeable: they differ in timestep sampling, checkpoint layout, where CFG dropout
is applied, and which flags exist. Where behaviour differs, both are named.

**On citations.** These are `path:line` references into ACE-Step 1.5, a codebase this
repository does not own. `AGENT_GUARDRAILS.md` G1 asks for *symbol* citations when a
document asserts what a file here does; a foreign dependency is cited by the function,
literal or log string the claim rests on (`apply_cfg_dropout`, `save_lokr_weights`,
`resolve_target_modules`, `[Side-Step] Pass 2 FAIL`) **plus** a line range for
reproducibility, and §13 gives the greps that re-anchor the line numbers when the
upstream release moves.

## 1. The pipeline, and which stage owns which decision

| Stage | What happens | What you decide | Section |
|---|---|---|---|
| Preprocess | caption and lyrics are encoded once; latents are computed; the result is frozen into `.pt` | model variant, `--max-duration`, `preprocess_mode` (legacy) | §2 |
| Inject | adapter modules are inserted into the DiT decoder; everything else is frozen | adapter type, rank/dim, target modules, DoRA | §3, §4 |
| Train | timestep sampled, forward pass run, MSE taken against the flow target | optimiser, LR, schedule, warmup, batch size, epochs | §5, §6, §9 |
| Save | adapter weights plus a config are written | output dir, save interval | §10 |
| Inference | the artefact is loaded by the generation service and applied to the decoder | nothing — but a repacked file may not load | §12 |

Two decisions are irreversible once stage one has run: the **model variant** (it
selects the text encoder, the VAE and the silence-latent subfolder) and, on the legacy
path, **`preprocess_mode`** (it selects `context_latents`, see the annotation guide
§7). Everything downstream can be re-run; those two cannot be repaired by training
differently.

### 1.1 Which subcommand you are actually running

The v2 CLI has no bare training invocation: the root parser requires a subcommand
(`acestep/training_v2/cli/args.py:100`, `required=True`), and `train.py` routes each one
to a different trainer (`train.py:96-105`).

| Subcommand | Reaches | Timesteps / CFG dropout | Adapters | Flags it does *not* have |
|---|---|---|---|---|
| `fixed` | `acestep/training_v2/trainer_fixed.py` | continuous logit-normal, `--cfg-ratio` dropout | LoRA and LoKR | — |
| `vanilla` | legacy `acestep/training/trainer.py` via `train_vanilla.py:66` | 8 discrete turbo values, no CFG dropout | LoRA and LoKR (`train_vanilla.py:150-160`) | `--cfg-ratio` (`args.py:325-328` adds it to `fixed` only) |
| `estimate` | `acestep/training_v2/estimate.py`, no training | n/a | n/a | everything training-related |

Two consequences worth carrying into every command you copy from a forum:

- A recipe that says "train for N epochs" without naming the subcommand is
  underspecified, because `vanilla` reproduces the discrete-timestep, no-dropout
  behaviour and prints a warning saying so (`train_vanilla.py:126-132`). `fixed` is the
  one that matches the mechanism described in §5 and §7.
- `vanilla` is not a LoRA-only path: `--adapter-type lokr` is honoured and builds a
  `LoKRConfig` against the legacy `LoKRTrainer` (`train_vanilla.py:150-160`). What it
  cannot do is change the legacy CFG ratio, which is hard-pinned at 0.15 in the model's
  `forward` (§7).

## 2. Preprocessing: the failure modes that cost you an evening

### 2.1 The conditioning is frozen here

Preprocessing encodes the caption and the lyric once and stores both as tensors in the
`.pt` file (`acestep/training_v2/preprocess.py:231-245, 265, 435-445`). No training
step ever re-reads your `.caption.txt`, your `.lyrics.txt` or your dataset JSON.

Consequence: a caption or lyric edit made *after* preprocessing has **zero** effect
until the preprocessor is re-run — not a reduced effect, zero. "Tweak the caption and
resume training" silently trains on the stale tensor. Settle the text, then preprocess,
then train.

### 2.2 A failed pass 2 leaves `.tmp.pt` files that the loader will pick up

Preprocessing runs in two passes. Pass 1 writes one intermediate per track named
`<stem>.tmp.pt` (`acestep/training_v2/preprocess.py:279`); pass 2 turns each into
`<stem>.pt` and deletes the intermediate **only on success** (`:431-432`, `:453`). The
`except` branch logs `Pass 2 FAIL` and continues to the next track (`:458-460`), so a
pass-2 failure (typically out-of-memory on a long file) leaves the `.tmp.pt` in the
tensor directory with nothing cleaning it up.

That file is not inert. The tensor loader reads `manifest.json` if it exists
(`acestep/training/data_module.py:64-70`) — nothing in the preprocessing code ever
writes one, so the fallback scan is what actually runs — and that scan keeps any file
matching `f.endswith('.pt')` (`:76`). `song.tmp.pt` ends with `.pt`, so it is indexed as a
track named `song.tmp` and then loaded without the fused `encoder_hidden_states` key
that pass 2 exists to add — a failure that surfaces during training, not during
preprocessing, and reads nothing like "preprocessing failed".

Operational rules:

- After every preprocess run, check the log for `[Side-Step] Pass 2 FAIL` and count it
  against your file count. A short run that exits 0 is still a broken dataset.
- Remove stragglers before training:
  `find <tensor-dir> -name '*.tmp.pt' -delete`.
- Half-finished preprocessing means re-running it for the missing tracks; nothing
  resumes it automatically.

### 2.3 `--max-duration` truncates from the end, not the middle

Audio longer than `--max-duration` is cut at the tail, keeping the head (default
240.0 s, `acestep/training_v2/cli/args.py:318-322`; applied in
`acestep/training/dataset_builder_modules/preprocess_audio.py:17-19`). A six-minute
track trains on its first four minutes. Either trim deliberately so the musically
complete portion survives, or raise the flag and pay the VRAM.

### 2.4 What the caption is embedded in

Your caption is never encoded on its own. Both paths wrap it in the same fixed template
— instruction line, caption, then a four-line `# Metas` block, then the end-of-text
token (`acestep/constants.py:160-168`, filled by
`acestep/training_v2/preprocess_prompt.py:63-74`) — and the whole string is tokenised
with `max_length=256, truncation=True` (`acestep/training/dataset_builder_modules/preprocess_text.py:15-24`).

Measured with the shipped tokenizer (`tokenizer.json` at the repo root, `tokenizers`
0.23.1), that skeleton costs **57 tokens** before a single word of caption: 34 of them
are the `# Metas` block itself at ordinary values (`bpm: 120`, `timesignature: 4`,
`keyscale: C major`, `duration: 180.0 seconds`). The caption therefore has about **199
tokens** before the 256-token ceiling starts cutting, and what gets cut is the tail — the
Metas block and the terminator, not the front of your caption.

Two consequences: a caption written in the `bpm: …` style duplicates a field the
template already emits (see the annotation guide §7), and a caption long enough to
displace the Metas block teaches the adapter a truncated metadata line. The exact
template, the token budgets and the per-field defaults are in
[`Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md`](Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md)
§7, where they change what you write rather than how you train.

## 3. What actually becomes trainable

### 3.1 The four attention projections are the default, not a hard ceiling

Adapter targets default to `q_proj` / `k_proj` / `v_proj` / `o_proj` on both adapter
configs (`acestep/training/configs.py:25-27` for LoRA, `:52-54` for LoKR), and the
discovery helper that enumerates candidates is hard-coded to those four names
(`acestep/training/lora_injection.py:127-131`). Injection itself is PEFT with
`target_modules=lora_config.target_modules` (`:204-213`), and every parameter whose
name does not contain `lora_` is frozen immediately afterwards (`:216-218`).

So the allowlist is a **default, not a wall**:

- `--target-modules` (v2 only) replaces the list, default
  `["q_proj", "k_proj", "v_proj", "o_proj"]` (`acestep/training_v2/cli/args.py:289`).
- `attention_type` rewrites each entry to `self_attn.<name>` or `cross_attn.<name>`
  (`acestep/training_v2/cli/validation.py:60-98`, called from
  `acestep/training_v2/cli/config_builder.py:98`); the decoder really does expose
  `self_attn` / `cross_attn` attributes
  (`acestep/models/turbo/modeling_acestep_v15_turbo.py:391, 462`), so the rewrite
  matches rather than silently missing.
- The hard-coded helper `get_dit_target_modules` is used for VRAM estimation
  (`acestep/training_v2/estimate.py:92`), not to override your config.

The invariant that survives all of this is the one that matters for annotation: the
text encoder, lyric encoder, `text_projector` and VAE never receive an adapter
(`lora_injection.py:143-154` keeps every `encoder`/`text_encoder` prefix frozen). An
annotation convention whose vocabulary lives only in the text encoder cannot be taught.

### 3.2 The legacy LoKR target filter fails open, not closed

`acestep/training/lokr_utils.py:154-157`: if the target filter enabled nothing, the
code does not raise — it flips `requires_grad = True` on **every** LyCORIS parameter
and continues. A typo in `target_modules`, or a pattern that matches no module in this
checkpoint, therefore produces the largest and slowest possible LoKR instead of an
error. Check the log line `LoKr target filter: enabled N LyCORIS modules` before
committing to a long run; N should be a plausible fraction of the network, not the
whole thing.

### 3.3 Scale is pinned at 2.0 everywhere; capacity is not

Both adapter families ship with `alpha = 2 × rank`, so their default effective scale is
2.0 on every surface:

| Adapter | Rank / dim | alpha | Effective scale (alpha/rank) | Source |
|---|---|---|---|---|
| LoRA | `r = 8` | 16 | 2.0 | `acestep/training/configs.py:22-23` |
| LoKR | `linear_dim = 64` | 128 | 2.0 | `acestep/training/configs.py:45-46` |

That `r = 8` is the **legacy dataclass** default, and it is the one number here that no
shipped interface actually launches:

| Surface | rank | alpha | dropout | Source |
|---|---|---|---|---|
| `LoRAConfig` dataclass (only when a caller omits the field) | 8 | 16 | 0.1 | `acestep/training/configs.py:22-24` |
| Gradio LoRA tab | 64 | 128 | 0.1 | `acestep/ui/gradio/interfaces/training_lora_tab_dataset.py:37-63` |
| v2 CLI (`train.py fixed`, `train.py vanilla`) | 64 | 128 | 0.1 | `acestep/training_v2/cli/args.py:286-288` |
| wizard preset `quick_test`, `vram_8gb` | 16 | 32 | 0.1 | `acestep/training_v2/presets/quick_test.json` |
| wizard preset `vram_12gb` | 32 | 64 | 0.1 | `acestep/training_v2/presets/vram_12gb.json` |
| wizard preset `recommended`, `vram_16gb` | 64 | 128 | 0.1 | `acestep/training_v2/presets/recommended.json` |
| wizard preset `high_quality`, `vram_24gb_plus` | 128 | 256 | 0.1 | `acestep/training_v2/presets/high_quality.json` |

LoRA dropout is 0.1 on every one of those surfaces, so dropout is not a source of
divergence; rank is. Two things follow.

First, "LoRA `r = 8` versus LoKR `linear_dim = 64`" — the two dataclass defaults, both at
scale 2.0 — is not a fair fight: the LoRA has an 8x smaller rank, so the comparison is of
adapter sizes as much as of algorithms, and a conclusion drawn from it does not transfer
to matched capacities. State the rank (or dim) with the result.

Second, the dataclass default is not what any interface launches: the CLI and the Gradio
tab both start at 64/128, and the presets span 16 to 128. A recipe transcribed from the
dataclass is therefore 8x smaller than the same recipe clicked through the UI, and the
two will not reproduce each other.

The presets are reachable from the interactive wizard only: nothing in
`acestep/training_v2/cli/` reads a preset, and the `vram_profile` field they set
(`acestep/training_v2/configs.py:138`) is stored but never consulted — the preset's
effect is entirely the numbers it copies into the answers
(`acestep/training_v2/ui/flows_train.py:69-76`). A scripted run has to spell those
numbers out.

### 3.4 `--bias` is inert: PEFT enables the biases and the next loop freezes them

`--bias {none,all,lora_only}` (`acestep/training_v2/cli/args.py:290`) is passed straight
into the PEFT `LoraConfig` (`acestep/training/lora_injection.py:204-211`), so
`get_peft_model` really does flip `requires_grad = True` on the bias tensors it was
asked to train. A few lines later, the injection function runs:

```python
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False
```

(`acestep/training/lora_injection.py:216-218`.) A bias parameter is named
`…q_proj.bias`, which contains no `lora_` substring, so it is re-frozen immediately
after PEFT enabled it. The trainable-parameter count reported in the info block
(`:220-221`) is therefore the adapter alone for all three choices.

Consequence: `--bias all` and `--bias lora_only` train exactly what `--bias none`
trains. If an adapter cannot hold a constant — an unusual instrument, a room tone — the
bias route is closed, and the only levers left are the data (§3.1) and the rank.

### 3.5 `estimate` measures module sensitivity, but nothing consumes the measurement

`python train.py estimate …` runs forward/backward passes over a handful of batches with
gradients enabled only on candidate modules, accumulates the mean gradient norm per
module and prints a ranked list (`acestep/training_v2/estimate.py:26-46, 118-220`); the
same list is written as JSON to `--output` or `./estimate_results.json`
(`train.py:206-260`).

What it does **not** do is close the loop. There is no flag that reads that JSON back:
`--module-config` and `--auto-estimate` do not exist in the argument parser at all, and
the two config fields that would hold them (`acestep/training_v2/configs.py:209-212`)
are only ever set from a `getattr` default in `config_builder.py:180-181` and copied
into `to_dict` (`configs.py:280-281`). Nothing in the training path reads either. The
target list is whatever `--target-modules` says, verbatim (`config_builder.py:98`).

So the workflow is manual, and worth doing once per model variant:

```bash
python train.py estimate --checkpoint-dir ./checkpoints --model-variant sft \
    --dataset-dir <tensor-dir> --top-k 16 --output estimate_results.json
# read the ranking, then name a subset with plain module names; --attention-type
# decides whether they become self_attn.* or cross_attn.*
python train.py fixed --target-modules q_proj v_proj --attention-type both ...
```

A caveat on interpretation: the ranking is average gradient magnitude over a small
sample (5 batches unless `--estimate-batches` says otherwise, `train.py:211`), which is
a measure of where the loss is *sensitive*, not of where extra capacity would improve
it. It narrows a search; it does not answer it.

## 4. Adapter choice: LoRA or LoKR, and DoRA's per-surface default

LoRA is a PEFT adapter on the attention projections. LoKR is a LyCORIS network
(`acestep/training/lokr_utils.py`), saved in LyCORIS format rather than PEFT format —
so the two artefacts are not interchangeable even when the trained behaviour is
similar (see §10).

**DoRA (`weight_decompose`) is on or off depending on how you launched the run**, and
nothing in the artefact name tells you which:

| Surface | DoRA default | Source |
|---|---|---|
| LoKR config dataclass | off | `acestep/training/configs.py:51` |
| CLI (`--lokr-weight-decompose`) | off (`store_true`) | `acestep/training_v2/cli/args.py:301` |
| Interactive wizard | off | `acestep/training_v2/ui/flows_train_steps.py:137-139` |
| Gradio LoKR tab | **on** | `acestep/ui/gradio/interfaces/training_lokr_tab_dataset.py:80-85` (`value=True`) |
| HTTP API (`/train/lokr/start`) | **on** | `acestep/api/train_api_models.py:42` (`default=True`) |

Consequence: a LoKR trained from the UI or the API is a DoRA LoKR, a LoKR trained from
the CLI is not, and the two have different trainable parameter counts and different
weight layouts (`dora_wd=True` at `acestep/training/lokr_utils.py:111`). When you
compare or reproduce runs, state the surface you used, or pass the flag explicitly.

## 5. Timesteps: what `t` means here, and why the target looks inverted

### 5.1 The flow convention

```python
x1 = torch.randn_like(target_latents)   # noise
x0 = target_latents                    # data
xt = t_ * x1 + (1.0 - t_) * x0         # interpolation
flow = x1 - x0                         # regression target
```

`acestep/training_v2/fixed_lora_module.py:321-322, 337, 353-354`; identical in the
legacy trainer at `acestep/training/trainer.py:512, 528` (and `:1395, 1409` on the
LoKR path).

Read the interpolation: **`t = 1` is pure noise and `t = 0` is clean data.** This is
the opposite of the convention most diffusion write-ups use, where `t = 0` is noise and
`t = T` is data. The target is `noise − data`, not `data − noise`, which goes with it:
the model learns the direction that takes a sample *toward* noise.

Why it matters in practice, beyond notation:

- A timestep distribution described as "centred low" means *clean*, and "centred high"
  means *noisy*, in ACE-Step terms. Reading an ACE-Step sampling curve with the
  diffusers convention inverts the conclusion about what the adapter saw most.
- The inference loop has to apply the model output in the matching direction; a
  hand-written sampler or a re-implementation that assumes the other convention
  produces noise instead of audio while the loss curve looks perfectly healthy.

### 5.2 Two sampling regimes, and "8" does not mean the same thing in both

| Trainer | Timestep sampling | Source |
|---|---|---|
| legacy (`acestep/training/`) | 8 discrete values from the turbo shift=3.0 schedule, sampled uniformly | `acestep/training/configs.py:84-86, 104-105`; the values at `acestep/training/trainer.py:61-69`, used at `:399` and `:1343` |
| v2 (`acestep/training_v2/`) | continuous, logit-normal (`mu = -0.4`, `sigma = 1.0`) | `acestep/training_v2/timestep_sampling.py:20-72`; parameters at `acestep/training_v2/configs.py:145-149` |

The legacy schedule is literally the turbo **inference** schedule reused for training —
the source comment reads "same as inference" (`acestep/training/trainer.py:60`) and
`num_inference_steps` is annotated "Fixed: turbo uses 8 steps"
(`acestep/training/configs.py:105`). So the two trainers differ by more than sampling
method: one trains on exactly the timesteps Turbo later generates with, the other
samples a continuous distribution. A recipe tuned under one regime is not evidence
about the other. (Under the convention above, `mu = -0.4` puts the median of the
logit-normal at `sigmoid(-0.4) ≈ 0.40` — a mild bias toward the cleaner half of the
range.)

`data_proportion` looks like a knob and is not one. Its default is 0.5
(`acestep/training_v2/configs.py:151`), but every current variant calls the sampler
with `use_meanflow=False`, which forces `data_proportion = 1.0` and therefore `r = t`
for the whole batch (`acestep/training_v2/timestep_sampling.py:35-37, 63-67`; call site
`acestep/training_v2/fixed_lora_module.py:332`). Changing it has no effect.

### 5.3 `mu`, `sigma` and `data_proportion` are read from the checkpoint, not from flags

There is no CLI flag for any of the three. `build_configs` opens the **checkpoint's**
`config.json` and takes `timestep_mu`, `timestep_sigma` and `data_proportion` from it
(`acestep/training_v2/cli/config_builder.py:66-88`). The file is found by mapping the
variant to its folder — `turbo` → `acestep-v15-turbo`, `base` → `acestep-v15-base`,
`sft` → `acestep-v15-sft` (`acestep/training_v2/cli/args.py:18-21`) — with a literal
subfolder of the same name as the second attempt, so a renamed fine-tune is still found
(`config_builder.py:23-44`).

Three consequences:

- **The sampler follows the checkpoint.** Swapping `--checkpoint-dir` to a fine-tune
  whose `config.json` carries different values changes the timestep distribution with no
  flag and no log line. If two runs disagree on loss shape, compare their `config.json`
  before comparing hyperparameters.
- **The fallbacks are the documented defaults.** Missing file, or a `JSONDecodeError`:
  `mu = -0.4`, `sigma = 1.0`, `data_proportion = 0.5`
  (`config_builder.py:66-68, 79-84`); a parse failure logs
  `[Side-Step] Failed to parse …` first. `--base-model` is consulted **only** when the
  config file does not exist (`:86-93`), so on a normal install it is inert.
- **What those numbers mean.** With `mu = -0.4` the median of the logit-normal is
  `sigmoid(-0.4) ≈ 0.40`, and under the convention of §5.1 that is a timestep just
  closer to clean data than to noise — half the steps sit below it. `sigma = 1.0` is wide
  enough that the tails still reach both extremes, which is why a run can look
  well-behaved while most of its gradient comes from the cleaner end.

`--shift` and `--num-inference-steps` sit in the same argument group and look like they
control this. They do not: both are stored on the config and never read by the v2
trainer (§12).

## 6. The loss is an unmasked mean over the full padded tensor

The flow-matching loss is a plain elementwise MSE over the whole padded latent batch,
with no `attention_mask` applied to the loss itself
(`acestep/training_v2/fixed_lora_module.py:354`; the mask reaches the decoder at
`:346`, but the loss never uses it).

Consequence: the default `batch_size = 1` (`acestep/training/configs.py:107`) is what
keeps this harmless. Raise the batch size and padding positions enter the loss, where
the target is noise, so part of the adapter's capacity is spent predicting nothing. The
wider the duration spread inside a batch, the larger the dilution — if you raise the
batch size for speed, keep durations comparable.

## 7. CFG dropout: what it actually drops, and what it buys

`cfg_ratio` defaults to 0.15 (`acestep/training_v2/configs.py:142`). On the v2 path it
is applied by replacing the **fused** encoder output with the model's learned null
embedding (`acestep/training_v2/fixed_lora_module.py:312-318`, calling
`apply_cfg_dropout`; the underlying mechanism in
`acestep/models/turbo/modeling_acestep_v15_turbo.py:1741-1749`). Caption and lyric are
packed into a single `encoder_hidden_states` tensor before that point, and
`encoder_attention_mask` is left unmodified — the mask stays valid while the *content*
becomes unconditional.

Consequence: roughly 15% of steps are fully unconditional — no caption **and no
lyrics**. There is no caption-only dropout and no lyric-only dropout anywhere in the
training path. That is the mechanism behind the annotation guide's rule that the lyric
stream is not a protected channel and that a feature should be stated once, in the
stream it belongs to.

Two path differences worth knowing:

- **Stack.** On the legacy path the dropout lives inside the model's `forward`, with
  0.15 as the signature default (`acestep/models/{sft,base}/modeling_acestep_v15_base.py:1723`
  and `:1744-1749`), and the legacy trainer never passes a `cfg_ratio`
  (`acestep/training/trainer.py` has no such argument), so legacy runs are pinned to
  0.15 with no flag to change it. The `--cfg-ratio` CLI flag is a v2 feature.
- **Variant.** Turbo carries a `null_condition_emb`
  (`acestep/models/turbo/modeling_acestep_v15_turbo.py:1622`), so it does train with
  CFG dropout — the widespread claim that "Turbo has no CFG support" is true only at
  inference, where `diffusion_guidance_scale` is forced to 1.0 with CFG folded into the
  distillation (`:1833-1849, 1893-1895`).

So the training variant decides whether the finished adapter can be steered at all: a
Turbo-trained adapter has its scale pinned at 1.0 and cannot be prompted harder or
softer, while still having been trained with dropout. Decide the variant before
preprocessing.

Corollary, correcting a related claim: `bpm`, `keyscale` and `timesignature` are **not**
CFG guidance inputs. They reach the model only as the `# Metas` text block inside the
same 256-token prompt as the caption, so they are prompt conditioning. Accuracy still
matters, because an inaccurate value is taught to the adapter as fact — but no guidance
mechanism reads it.

## 8. `--model-variant` defaults to Turbo, not SFT

`model_variant` defaults to `"turbo"` (`acestep/training_v2/configs.py:159`). Omit the
flag and both preprocessing and training run on the Turbo variant, with no error and no
warning — and the usual "the two must match" rule is then satisfied *consistently
wrong*, which is worse than a mismatch because nothing surfaces. Pass
`--model-variant sft` explicitly.

## 9. Optimiser, schedule, warmup — and two silent overrides

Defaults (`acestep/training/configs.py:106-118`): `learning_rate = 1e-4`,
`batch_size = 1`, `gradient_accumulation_steps = 4`, `max_epochs = 100`,
`save_every_n_epochs = 10`, `warmup_steps = 100`, `weight_decay = 0.01`,
`max_grad_norm = 1.0`, `seed = 42`; cosine schedule
(`acestep/training_v2/configs.py:126`). Published "1000 epochs on 12 songs" field notes
were run under different values, so their epoch counts do not transfer to a default
run.

Two behaviours in `acestep/training_v2/optim.py` will otherwise surprise you:

1. **Warmup is clamped.** `warmup_steps = min(warmup_steps, max(1, total_steps // 10))`
   (`:144`). Asking for 100 warmup steps on a 200-step run gives you 20, silently. On
   small datasets the warmup you configured is usually not the warmup you got, so a
   schedule comparison across dataset sizes compares different curves.
2. **Prodigy overrides you twice.** Selecting Prodigy substitutes `lr = 1.0` whenever
   your LR is exactly the `1e-4` default (`:93`) — a deliberately chosen value reads
   through, the default does not — and forces a constant scheduler unless you already
   chose `constant`/`constant_with_warmup` (`:136-141`). A cosine schedule you set is
   discarded, not blended.

The v2 CLI ships the same values as the dataclass for everything in that list
(`acestep/training_v2/cli/args.py:265-272` and `:306`), so a `fixed` run with no flags
and a Gradio run with no edits share an optimiser configuration. The exception is the
adapter triple — rank, alpha, dropout — which is not in this list at all; see the
per-surface table in §3.3.

### 9.1 Three fallbacks that are quieter than they should be

- **`--precision auto` means bf16 on every CUDA device, full stop.** `detect_gpu` maps
  `auto` to `bf16` whenever the device type is `cuda` or `xpu`, to `fp16` on `mps`, and
  to `fp32` on CPU (`acestep/training_v2/gpu_utils.py:113-119`). There is no
  compute-capability check anywhere in `training_v2` — `torch.cuda.is_bf16_supported`
  and `get_device_capability` are not called. On a pre-Ampere card that lacks native
  bf16, the run still reports bf16. Pass `--precision fp16` explicitly rather than
  trusting `auto` on older hardware.
- **`adamw8bit` degrades to AdamW without failing.** If `bitsandbytes` is not
  importable, the factory logs `[Side-Step] bitsandbytes not installed -- falling back
  to AdamW` and continues as `adamw` (`acestep/training_v2/optim.py:58-64`). The run
  succeeds with a materially larger optimiser-state footprint than you budgeted for —
  which on a tight card is an OOM several hundred steps in, not at startup.
- **Multi-GPU is DDP through Fabric, and `--batch-size` is per device.** With
  `--num-devices > 1` the trainer builds `Fabric(devices=n, strategy="ddp")`
  (`acestep/training_v2/trainer_fixed.py:256-272`), capping `n` to the visible GPU count
  and logging when it does (`:239-252`). The DataLoader is constructed with
  `batch_size=cfg.batch_size` per process (`:149`), so the effective batch is
  `batch_size × num_devices × gradient_accumulation_steps`. Doubling the device count
  without halving the batch or the accumulation steps doubles the effective batch and
  silently changes the optimisation, not just the speed.

## 10. Checkpoints, resume, and what the artefact is

- **Where checkpoints land.** `checkpoints/epoch_<N>_loss_<L>/` under the output
  directory, written every `save_every_n_epochs` epochs
  (`acestep/training_v2/trainer_fixed.py:516`). The loss is embedded in the directory
  name, so the path changes between runs — a script that hard-codes a checkpoint path
  points at the wrong epoch after the next run.
- **`--resume-from` fails soft.** A non-existent path prints
  `[WARN] Resume path not found (will train from scratch)` to stderr and continues
  (`acestep/training_v2/cli/validation.py:55`). A run that looks like it resumed can
  therefore be a fresh run; grep the log for that warning rather than trusting the exit
  code.
- **LoKR saves differently from LoRA.** LoKR goes through `save_lokr_weights` →
  `lokr_weights.safetensors` with `{"algo": "lokr", "format": "lycoris"}` metadata
  (`acestep/training/lokr_utils.py:185-208`); LoRA uses the PEFT path
  (`acestep/training_v2/trainer_helpers.py:169-184`). The saver refuses to hide a
  configuration error: if `adapter_type` is `"lokr"` but no LyCORIS network is
  attached, it raises rather than silently writing a LoRA-shaped file
  (`acestep/training_v2/trainer_helpers.py:156-166`).
- **`offload_encoder`** (default `False`, `acestep/training_v2/configs.py:135`) moves
  `vae`, `text_encoder`, `lyric_encoder`, `music_encoder`, `timbre_encoder`,
  `condition_projection` and `attention_pooler` to CPU after setup
  (`acestep/training_v2/trainer_helpers.py:121-140`, invoked at
  `acestep/training_v2/trainer_fixed.py:368-372`). Nothing moves them back, and on this
  code path nothing asks them to: there is no sampling code in the trainer to break
  (§12). Unlike a diffusion trainer with mid-run previews, turning this on costs you
  nothing beyond the one-time transfer to CPU.

## 11. Flags that do not do what their help text says

Every entry below was read out of the code, not the `--help` string. They are collected
here because each one is a plausible thing to try while debugging, and each one will
consume an evening before it becomes obvious that nothing was listening.

| Flag / field | Help text | What actually happens |
|---|---|---|
| `--sample-every-n-epochs` | "Generate audio sample every N epochs; 0=disabled" (`args.py:314`) | Parsed, stored on the config (`config_builder.py:175`, `configs.py:196`) and serialised into `to_dict` (`configs.py:276`). No training code reads it: a repo-wide grep for `sample_every_n_epochs` returns only the CLI, the dataclass and the wizard. There is no sampling pipeline in either trainer, so the flag cannot produce a preview at any value. |
| `--shift` | "Noise schedule shift (turbo=3.0, base/sft=1.0)" (`args.py:273`) | Inherited unchanged from the legacy dataclass (`acestep/training/configs.py:104`) and stored (`config_builder.py:138`). Neither trainer reads it: v2 samples a continuous logit-normal from `mu`/`sigma` (§5.3), and the legacy trainer's eight timesteps are a hard-coded literal (`acestep/training/trainer.py:61-69`). |
| `--num-inference-steps` | "Inference steps for timestep schedule (turbo=8, base/sft=50)" (`args.py:274`) | Same as `--shift`: stored (`config_builder.py:139`), never read. Passing `50` for an SFT run changes nothing about the timesteps it trains on. |
| `--bias {all,lora_only}` | "Bias training mode" (`args.py:290`) | The gradients are enabled by PEFT and immediately re-frozen by the `lora_`-substring filter (§3.4). |
| `--base-model` | "Base model a fine-tune was trained from … Used to condition timestep sampling. Auto-detected for official models." (`args.py:179-186`) | Consulted **only** when the checkpoint's `config.json` is missing (`config_builder.py:86-93`); on a normal install that file exists, so the flag never runs (§5.3). |
| `data_proportion` (config field) | Documented as the sample/meanflow split | Forced to `1.0` by `use_meanflow=False` on every current call path (§5.2). |
| `module_config`, `auto_estimate` (config fields) | Would let an estimate select target modules | No CLI flag parses them; nothing reads them (§3.5). |
| `vram_profile` (config field) | Selects a VRAM preset | No CLI flag parses it, and nothing reads it; the presets only work because the wizard copies their numbers into the answers (§3.3). |

The pattern is worth naming: this codebase has a layer of configuration fields that were
added for a feature that then landed elsewhere or not at all. A field appearing in
`to_dict()` is not evidence that anything consumes it — the check is a grep for readers,
which §13 lists for the ones that matter.

## 12. What inference does with the artefact

Training is half of the contract; the loader on the generation side decides whether your
file is usable at all.

**LoKR is rebuilt from the file's own metadata.** `lokr_weights.safetensors` stores the
adapter configuration as a JSON string under the `lokr_config` key in the safetensors
metadata (`acestep/training/lokr_utils.py:185-208`). At load time the inference handler
detects a LoKR by filename (`lokr_weights.safetensors`) **or** by the presence of that
key, then parses it back into a `LoKRConfig` and constructs the LyCORIS network from
those values — `linear_dim`, `linear_alpha`, `factor`, `weight_decompose` and the rest
(`acestep/core/generation/handler/lora/lifecycle.py:17-37, 87-182`).

Two practical consequences:

- The adapter's rank is **not** in its filename and cannot be inferred from the tensor
  shapes by a human reading a directory listing. It lives in the metadata. A tool that
  re-saves the file and drops safetensors metadata (a naive merge, some converters, a
  "clean up" script) turns the artefact into a blob the loader no longer recognises as
  LoKR, even though the weights are intact.
- Because the network is rebuilt from metadata, a LoKR whose config was edited
  post-training will load with the edited shape and then fail on weight shapes — the
  error surfaces at load, not at training, and it names tensor sizes rather than the
  file you edited.

**LoRA is refused on a quantized model.** `add_lora` returns an error immediately when
the service was initialised with quantization: *"LoRA loading is not supported on
quantized models"* (`lifecycle.py:196-205`). This is a property of how PEFT adapters
attach to a quantized linear layer, not a bug you can work around with a flag. An
adapter trained in bf16 therefore cannot be auditioned on a 4-bit or 8-bit service; it
needs a service re-initialised without quantization.

**The scale is a 0–1 dial, and 1.0 is the point you trained.** `set_lora_scale` clamps
its argument into `[0.0, 1.0]` (`acestep/core/generation/handler/lora/controls.py:83-105`,
the clamp at `:105`) and applies it per adapter name. Training happens at scale 1.0 with
`alpha/rank = 2.0` applied inside the adapter, so 1.0 reproduces the behaviour you
trained and lower values interpolate toward the base model. There is no 1.5 to "push" an
under-trained adapter: asking for one is silently clamped to 1.0, which looks like the
dial working and is actually the dial ignoring you.

**Adapters swap, they do not stack.** A second LoRA does load into the same `PeftModel`
(`lifecycle.py:273-280`) — but the loader then calls `set_active_adapter` and
`decoder.set_adapter(effective_name)` with that one name
(`acestep/core/generation/handler/lora/lifecycle.py:296-303`), so the newly added
adapter becomes *the* active one and the previous one goes dormant rather than
contributing. A LoKR cannot join at all: once PEFT is loaded, adding a LoKR returns
*"LoKr cannot be added as a second adapter when PEFT is already loaded"* (`:275-277`).

So "train two adapters and use both" does not produce the sum of their behaviours; it
produces whichever one `set_adapter` last selected. Check the active adapter before
concluding that an adapter failed to learn anything.

## 13. Verification: the greps that prove it on your install

Line numbers move between releases; these checks do not.

```bash
# timestep convention: t=1 must multiply the noise term
grep -n 'xt = t_' acestep/training_v2/fixed_lora_module.py acestep/training/trainer.py

# regression target is noise - data, not data - noise
grep -n 'flow = x1 - x0' acestep/training_v2/fixed_lora_module.py acestep/training/trainer.py

# warmup clamp and Prodigy overrides
grep -n 'warmup_steps = min' acestep/training_v2/optim.py
grep -n 'lr if lr != 1e-4' acestep/training_v2/optim.py

# adapter targets are a default list, and the LoKR filter fails open
grep -n 'target_modules' acestep/training/configs.py
grep -n 'if not lokr_param_list' -A3 acestep/training/lokr_utils.py

# DoRA defaults differ per surface
grep -rn -E 'weight_decompose: bool = |lokr_weight_decompose: bool = ' acestep/

# --bias is re-frozen by the lora_-substring filter a few lines after PEFT enables it
grep -n 'bias=lora_config.bias' -A16 acestep/training/lora_injection.py

# --sample-every-n-epochs has no reader: every hit is a store, none a load
grep -rn 'sample_every_n_epochs' acestep/

# --shift / --num-inference-steps have no reader either
grep -rn 'num_inference_steps\|\.shift' acestep/training_v2/trainer_fixed.py \
    acestep/training_v2/fixed_lora_module.py acestep/training_v2/timestep_sampling.py

# LoKR metadata round-trip: written by one function, parsed by the other
grep -n 'lokr_config' acestep/training/lokr_utils.py \
    acestep/core/generation/handler/lora/lifecycle.py
```

## See also

- [`Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md`](Comprehensive_LoRA_LoKR_Dataset_Annotation_Guide_updated.md) —
  annotation rules, metadata, lyric convention, and the annotation-side consequences of
  the mechanics above (its section 8).
- [`Artist_Scoped_Caption_Tag_Generation.md`](Artist_Scoped_Caption_Tag_Generation.md) —
  the artist-scoped caption and bracket template feature (specified, not implemented).
- [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) — how documentation in this repository is
  verified and cited.

