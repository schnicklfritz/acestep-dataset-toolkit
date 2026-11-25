# Kaggle removal — remaining work

Branch `remove-assistant`, HEAD `7506e89`.

## Done
AI assistant removed. FlowLayout re-entrancy guard + theme repolish skip.
Kaggle lyrics transcription, lyrics alignment, Demucs stems and structural
captioning — each removed with its kernel and tests.

## Remaining, in order
5. Probe + credentials: `workers/kaggle_probe.py`, the credential cluster in
   `dataset_manager.py`, `kaggle_*` widgets in `ui/settings_tab.py`, `kaggle_*`
   keys in `config.py`, its tests.
6. Caption run + ACE-Step page: `workers/caption.py`,
   `modules/caption_kaggle_run.py`, `ui/ace_step_tab.py`,
   `kernels/caption_kernel.py`, `scripts/build_caption_notebook.py`,
   `modules/kaggle.py`.
   BEFORE deleting, re-home into `ui/caption_tab.py` the five controls
   `save_pipeline_defaults` reads unguarded: model weights dataset, max tokens,
   max audio duration, batch size, prompt add-on. Never use getattr guards.
7. When the suite matters: `DatasetManager.closeEvent` shows a modal
   "Save Before Closing?" when samples exist and blocks forever offscreen.
   Guard it with an env flag like `ACESTEP_SKIP_LYRICS_GATE`, then add a
   conftest teardown that releases only the windows the current test created.

## Must survive
`modules/caption_spec.py`, `modules/caption_quality.py`, `ui/caption_tab.py` as
editor, MVSEP dialog, local captioning, whisperx + gemini alignment, exporters.

## Landmines
- Baton dead end `d-99fc` claims the suite cannot run in one process; its proof
  was a SIGKILL from a faulty watchdog. Disregard it.
- Never run two pytest processes at once.
- Offscreen plus a modal dialog equals an infinite block.

## Method
Prove unreachable with `grep -F` before deleting. Stop if a symbol has a live
caller outside the commit. Full suite green after each commit. Do not push.
Do not touch `.venv/`, `deletions/`, `/home/fritz/42`, dataset directories.
