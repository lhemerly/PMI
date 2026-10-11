# Local content factory MVP

One command writes a prompt, Bonsai answer, speech WAV, per-stage receipts, progress log, and a final result containing the output paths. Text and speech run sequentially under one lease. Piper remains the backward-compatible default; Kokoro is selected explicitly with `--speech-model kokoro-82m`.

```sh
python3 -B content_factory.py speak \
  --prompt-file first-run-prompt.txt \
  --models bonsai-stock \
  --out /home/luizh/Documents/Codex/2026-10-08/task-8/first-bonsai-speech-job-007 \
  --lease /home/luizh/Documents/Codex/2026-10-08/content-factory-shared.lease
```

The command prints `text:bonsai-stock` and `speech:piper-cpu` stage states as they change. The job folder contains `progress.jsonl`, the answer under `stage-0-bonsai-stock/answer.txt`, the WAV at `speech/answer.wav`, and the final `result.json` with the paths. A failure is recorded as `StoppedNoRetry`; completed stage files remain available.

If the text stage passed and speech failed, resume only speech with a new output folder:

```sh
python3 -B content_factory.py resume-speech \
  --resume-from /path/to/failed-text-plus-speech-job \
  --out /path/to/new-speech-resume-job \
  --lease /home/luizh/Documents/Codex/2026-10-08/content-factory-shared.lease
```

Resume verifies the saved Bonsai request, answer, natural completion, placement, cleanup, and postflight receipts. It does not construct a text backend. It uses Piper by default; select Kokoro with `--speech-model kokoro-82m --speech-voice pf_dora` when resuming to Brazilian Portuguese. The shared lease, fresh CPU speech admission, owner cancellation, host/VRAM supervisors, WAV validation, and no-retry behavior remain in force. Use a fresh output directory for each attempt.

An optional local ComfyUI image stage can be appended after speech:

```sh
python3 -B content_factory.py speak-image \
  --prompt-file first-run-prompt.txt \
  --models bonsai-stock \
  --out /path/to/job \
  --lease /home/luizh/Documents/Codex/2026-10-08/content-factory-shared.lease \
  --comfy-endpoint http://127.0.0.1:8188
```

`speak` and `resume-speech` are unchanged and never construct an image backend. The image stage runs last, so a failed image attempt still leaves valid text and speech receipts. The selected route is **ComfyUI with FLUX.2 Klein 4B distilled**: the adapter now submits that model's real native graph (`UNETLoader`/`CLIPLoader`/`VAELoader`, `CLIPTextEncode` with zeroed negative conditioning, `CFGGuider`, `Flux2Scheduler`, `SamplerCustomAdvanced`, and one `SaveImage`), flattened from the pinned official workflow snapshot in [`workflows/`](workflows/). The old seven-node generic checkpoint graph has been removed and is refused by the validator.

Graph compatibility is established from that pinned source; host support is not. ComfyUI v0.39.0, PyTorch 2.14.1+rocm7.2, and the three Klein model files are installed per [the setup receipt](.runtime/comfyui/SETUP-RECEIPT.md). `image-plan` checks only local file presence and byte counts; it reports the receipt SHA-256 values without rehashing the weights. Execution on this RX 6600 XT under the existing 7 GiB global VRAM limit remains unqualified, and the platform execution hold still applies. See [`IMAGE_ADAPTER.md`](IMAGE_ADAPTER.md) and [`KLEIN-IMPLEMENTATION-HANDOFF.md`](KLEIN-IMPLEMENTATION-HANDOFF.md).

Prompts must be nonempty UTF-8, contain no NUL, and be at most 4096 bytes. This checkout has reviewed fit evidence for the exact welcome prompt; other prompts need their own reviewed fit evidence before execution. The welcome prompt uses a 512-token context, 64-token output cap, and a 240-second whole-job bound. RAM admission is 16 GiB with a 12 GiB floor; RSS is capped at 12 GiB, address space at 16 GiB, and global VRAM at 7 GiB. Piper is CPU-only and writes mono 16-bit PCM at 22,050 Hz, up to 30 seconds. Existing platform execution gates still apply; the CLI cannot approve its own launch.

## Explicit Kokoro speech selection

`speak`, `speak-image` and `resume-speech` accept `--speech-model piper-cpu|kokoro-82m`. Piper remains the default so existing commands and receipts keep their original route. Kokoro accepts `--speech-voice af_heart` for English or `--speech-voice pf_dora` for Brazilian Portuguese; `PMI_KOKORO_VOICE` is the environment default when the CLI flag is omitted. An unsupported model or voice is rejected rather than mapped to Piper.

Use `speech-plan --speech-model kokoro-82m --speech-voice pf_dora` for an offline readiness report. It checks the expected local model, config, voice, runtime, and integrity-seal paths by filesystem metadata only and reports identity/runtime gates separately. No Kokoro model, voice pack or runtime is installed here. The `kokoro==0.9.4` synthesis worker is implemented but unrun and unqualified; it requires CPython `>=3.10,<3.13` (Python 3.12 is suitable), a CPU PyTorch build, the package dependencies and system `espeak-ng`, plus the reviewed local input seal. An explicit Kokoro selection stops before text generation until assets, runtime and the reviewed seal are supplied. Piper remains at mono PCM16 22,050 Hz; Kokoro receipts use mono PCM16 24,000 Hz and include the selected voice/language, verified input hashes, the exact runtime version, and qualification status.

For the current PT-BR selection, the safe next command is:

```sh
python3 -B content_factory.py speech-plan --prompt-file first-run-prompt.txt --models bonsai-stock --speech-model kokoro-82m --speech-voice pf_dora --out /tmp/pmi-kokoro-pf-dora-plan.json
```

This writes only an offline readiness plan. It does not create a job, run text or speech, or contact a service.

The supported model-free check is:

```sh
python3 -B -m unittest -q test_contract test_speech
```

Contract-only outputs are labeled `ContractOnly`; they are not inference. [`demo/`](demo/README.md) contains a prior verified real run as evidence, not a new MVP integrated run.

## Model intent and the Kokoro CPU adapter

[`model_intent.py`](model_intent.py) is a typed declaration of every model role per modality. It keeps *installation* (is the artifact here, and has anybody established its identity) separate from *runtime qualification* (has the runtime that would load it been qualified here), because merging those two is how a stock artifact inherits somebody else's evidence. Unknown identities are recorded as `None`, which always means unknown, never verified.

The intended text model is `OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF` `PTQ1_0`, observed upstream revision `6db27547a8f25667cdb43a28d15cc5d33937b7dc`. It is absent here, no weight hash is known, and it is unqualified. The installed stock Bonsai artifact keeps its preserved receipts and its own scoped welcome-prompt evidence; that is historical baseline and is **not** transferred to Heretic. `qwen3-stock` and `qwen25-modified` stay `not-selected` and are never substituted.

[`kokoro_backend.py`](kokoro_backend.py) and [`kokoro_worker.py`](kokoro_worker.py) implement explicit local CPU synthesis for `hexgrad/Kokoro-82M`, model revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, against the official `kokoro==0.9.4` API. The worker supplies local `config.json`, weight, and `.pt` voice paths with offline hub mode; `af_heart` uses English and `pf_dora` uses Brazilian Portuguese. It emits bounded mono PCM16 at 24 kHz and cleans up in an owned child process. This source was not run with a real runtime or model and does not qualify host support. Piper remains the approved auxiliary CPU baseline and is unchanged; the 22,050 Hz and 24 kHz contracts are separate and each validator refuses the other's rate.

Image is unchanged: ComfyUI plus FLUX.2 Klein 4B distilled with the same pinned workflow, where graph compatibility is a source property and host/runtime qualification stays unverified. Video remains unresolved and unavailable. See [`MODEL_INTENT.md`](MODEL_INTENT.md).

## Research operating mode

PMI now has a durable serial propose/evaluate/review/revise harness with trusted executable text checks, best-candidate preservation, budgets, status, resume and cancellation. Start with `python3 -B content_factory.py research-plan --spec-file research/example-spec.json`. See [research/README.md](research/README.md) for controls and current live-inference blockers. No real discovery or model qualification is established by the deterministic tests.
