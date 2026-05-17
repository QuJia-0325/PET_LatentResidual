# V18 LoRA Param Count Audit - 2026-05-17

## Verdict

**No implementation bug was found. Do not stop the running V18 training.**

The correct V18 decoder LoRA adapter parameter count is **589,824**.

The `884,736` expectation comes from using the encoder / ViT-B dimensions (`hidden_size=768`, `intermediate_size=3072`, `num_hidden_layers=12`) instead of the actual MAE decoder dimensions used by this RAE checkpoint:

```json
"decoder_hidden_size": 512,
"decoder_intermediate_size": 2048,
"decoder_num_hidden_layers": 8
```

V18 injects LoRA into `rae.decoder.decoder_layers`, not into the encoder blocks. Therefore `last_n_blocks=2` means decoder layers **6 and 7** of an 8-layer decoder, not layers 10 and 11 of a 12-layer ViT encoder.

## Why 589,824 Is Correct

V18 uses:

- `rank = 32`
- `last_n_blocks = 2`
- 6 wrapped Linear modules per decoder layer:
  `query`, `key`, `value`, `attention.output.dense`, `intermediate.dense`, `output.dense`
- actual decoder dimensions:
  `hidden = 512`, `mlp = 2048`

For one decoder layer:

| Module group | Linear shapes | Formula | Params |
|---|---:|---:|---:|
| Attention q/k/v/out | 4 x 512 -> 512 | `4 * 32 * (512 + 512)` | 131,072 |
| MLP fc1/fc2 | 512 -> 2048 and 2048 -> 512 | `2 * 32 * (512 + 2048)` | 163,840 |
| **Per decoder layer** | 6 Linear |  | **294,912** |
| **Two decoder layers** | 12 Linear | `2 * 294,912` | **589,824** |

The remote `884,736` number is the result of substituting `hidden=768`, `mlp=3072`, and 2 decoder layers:

```text
2 * [4 * 32 * (768 + 768) + 2 * 32 * (768 + 3072)] = 884,736
```

That formula is internally consistent, but it is not the architecture loaded by `/home/qujiaxiang/project/RAE/vit-mae/config.json`.

## Checkpoint Audit

Command environment:

```bash
cd /home/qujiaxiang/project/PET_LatentResidual
/home/qujiaxiang/.conda/envs/rae/bin/python <audit-snippet>
```

Checkpoint inspected:

```text
/data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/step_180000.pt
```

Result:

```text
checkpoint: /data_2/qujiaxiang/outputs/PET_LatentResidual/review_0517_runs/V18_decoder_lora/run/first_hop_224_v18_decoder_lora/step_180000.pt
decoder_lora_tensors: 24
decoder_lora_numel: 589824
wrapped_layers: ['6', '7']
6.attention.attention.key: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
6.attention.attention.query: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
6.attention.attention.value: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
6.attention.output.dense: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
6.intermediate.dense: 81,920 (lora_A=(32, 512), lora_B=(2048, 32))
6.output.dense: 81,920 (lora_A=(32, 2048), lora_B=(512, 32))
7.attention.attention.key: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
7.attention.attention.query: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
7.attention.attention.value: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
7.attention.output.dense: 32,768 (lora_A=(32, 512), lora_B=(512, 32))
7.intermediate.dense: 81,920 (lora_A=(32, 512), lora_B=(2048, 32))
7.output.dense: 81,920 (lora_A=(32, 2048), lora_B=(512, 32))
state_dict_group_numel: {'other_state_tensors': 1252421, 'backbone_state_tensors': 207864788, 'rae_state_tensors': 115873612}
```

Interpretation:

- All expected modules are wrapped: attention q/k/v/out and MLP fc1/fc2 are present.
- MLP LoRA is not missing and is not rank-reduced.
- The smaller count is fully explained by the actual decoder width `512/2048`.
- The log line `wrapped 12 Linear modules ... numel=589,824` is correct.

## Parameter Count Semantics

There are three different parameter-count questions. They should not be mixed.

| Question | Answer | Meaning |
|---|---:|---|
| Newly added V18 decoder LoRA adapter params | **589,824** | Only `rae.decoder.decoder_layers.{6,7}.*.lora_A/B` |
| Optimizer trainable params at V18 launch | **209,509,657** | Existing trainable backbone + first-hop modules + decoder LoRA |
| Checkpoint state_dict tensor count | larger than trainable count | Includes frozen RAE/base decoder tensors and persistent buffers |

The launch log already separates the optimizer groups:

```text
[optimizer] group backbone: params=207,667,412
[optimizer] group first_hop: params=1,252,421
[optimizer] decoder_lora param group: tensors=24, numel=589,824
```

So if the question is **"how many extra parameters did V18 add?"**, the answer is **589,824**.

If the question is **"how many parameters are being optimized in this run?"**, the answer is **207,667,412 + 1,252,421 + 589,824 = 209,509,657** because the current V18 config continues training the backbone and first-hop components instead of freezing them.

## Stale Documentation / Comment Issue

Two existing texts can mislead readers:

- `review/0517/V18_decoder_lora/V18_decoder_lora.yaml` comments mention "ViTB decoder total 12 layers -> layer 10, 11".
- `review/0517/V18_decoder_lora/RAE_ARCHITECTURE_FINDING_20260517.md` also describes `decoder_layers` as 12 layers for ViTB.

These are stale assumptions. The loaded config shows `decoder_num_hidden_layers=8`, and the checkpoint confirms V18 wrapped layers 6 and 7. Per the audit task instruction, this audit does **not** modify YAML/Python files. A later cleanup commit should fix the comments/docs to prevent repeated confusion.

## Decision

- Match expected `884,736`? **No**, because that expectation uses the wrong dimensions.
- Match log `589,824`? **Yes**.
- Missing fc1/fc2 wrap? **No**.
- Wrong rank? **No**.
- Need to stop/restart V18? **No**.

V18 can continue running.
