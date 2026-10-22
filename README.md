---
title: Class Quality Assessor
emoji: ♟️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Class Quality Assessor

Upload an online class recording (or paste a link). The app grades it **Poor / Average / Good / Excellent** and explains
the grade with timestamped quotes and measurements. Built for the Humaniti AI build challenge; all components are free.

## How it works

```
recording ─┬─ audio ──► Whisper (windowed; non-English windows are translated to English) ──► transcript
           │              └─► loudness / silence / noise measurements
           └─ frames ─► CLIP (what is on screen) + YuNet (faces) + brand-logo check (optional)

transcript ─► AI judge (Gemini / Groq / any OpenAI-compatible API / local Ollama)
              labels teaching moves per 2-minute segment, with verbatim quotes   ──► quotes verified against the transcript
                                                                                  ──► scores from frequencies per observed minute
measured signals (lighting, face visible, audio, brand) ─────────────────────────────┘ ──► grade + evidence + confidence
```

* **Opening, closing and three mid-class slices** are analysed (about 23 min of a ~55 min class): openings and closings carry most of the signal and it keeps CPU time down.
* **Dense labelling**: the judge must account for every move type in every segment; every claimed move needs a quote that really exists in the transcript (hallucinated evidence is dropped, timestamps corrected).
* **Frequencies, not yes/no**: praise per minute, probing-question share, etc. A single "no problem" does not make a warm class.
* **Honest degradation**: an unusable transcript (silent, garbled, unsupported language) is *not graded on text*; translated classes and demo/trial sessions are flagged; confidence is lowered.
* **Fallback chain**: preferred provider → other Gemini models (free quotas are per model) → Groq/other → local Ollama → measured-signals estimate. Failed providers are skipped for a cooldown (circuit breaker).

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python -m spacy download en_core_web_sm                # name masking before any text leaves the machine
cp .env.example .env                                   # add GEMINI_API_KEY (free key from Google AI Studio)
python scripts/check_llm.py                            # verify the AI provider works
streamlit run app.py                                   # http://localhost:8501
```

ffmpeg is required (`winget install Gyan.FFmpeg`, `apt install ffmpeg`, or it falls back to the bundled `imageio-ffmpeg`).

Command line: `python scripts/assess_one.py path/to/video.mp4`.

## Configuration (`.env` or environment variables)

| Setting | Meaning |
|---|---|
| `LLM_PROVIDER` | `gemini` \| `groq` \| `openai` \| `ollama` \| `jev` \| `none` \| `auto`. One setting switches the AI judge (also selectable in the UI sidebar). |
| `ALLOW_EXTERNAL_LLM=1` | Explicit consent to send **de-identified transcript text** (never audio/video/frames) to a cloud API. Cloud providers refuse to run without it. |
| `GEMINI_API_KEY`, `GEMINI_MODELS` | Key and an ordered model list; free-tier quotas are per model so the chain rotates through them. |
| `ESCALATE_TO`, `JEV_*` | Optional Jev (yes/no decider) mode; see `.env.example`. |
| `WHISPER_MODEL` | `tiny` / `base` / `small` (default) / `medium`. |
| `APP_ACCESS_CODE` | If set, the web app asks for this code first (use it on a public deployment). |
| `BRAND_TEMPLATE` | Path to a reference logo image for the branded-background check; empty disables it. |
| `JUDGE_MODE` | `moves` (default) or `checklist` (older yes/no approach, kept for comparison). |

Logs go to stderr and `.cache/assessor.log` with per-stage timings; the report's "How this assessment was produced" panel shows the same.

## Deploy on Hugging Face Spaces (free)

1. Create a Space at huggingface.co/new-space → **SDK: Docker**, hardware **CPU basic (free)**.
2. Push this repo to the Space's git remote (the `Dockerfile` and the front matter above are all it needs).
3. Space → **Settings → Variables and secrets**:
   * secrets: `GEMINI_API_KEY`, `APP_ACCESS_CODE`
   * variables: `LLM_PROVIDER=gemini`, `ALLOW_EXTERNAL_LLM=1`, `GEMINI_MODELS=<comma-separated models>`
4. Wait for the build (it downloads the models into the image), then open the Space. Free Spaces sleep after inactivity: open it once shortly before anyone tests it.

## Evaluation (10 labelled sample classes; reviewer's `feedbacks.pdf` is the ground truth)

`python scripts/evaluate.py --provider gemini` (features are cached; `scripts/extract_features.py` builds them once).

| Approach | Exact tier | Within 1 | Ranking agreement |
|---|---|---|---|
| Measured signals only (no AI) | 20% | 90% | 72% |
| Local Hugging Face NLI decider only | 20% | 70% | 58% |
| Gemini yes/no checklist | 30% | 50% | 58% |
| **Gemini teaching-moves (dense labelling)** | **60%** | 70% | **78%** |

Caveats: 10 classes only (2 per tier) and mostly the weaker flash-lite model; the reviewers' own agreement on this task is low
in the research literature, so ranking agreement and within-one-tier accuracy are the fairer measures. Good and Excellent classes
were graded one tier low. Re-run with stronger models when quota allows.

Latency on a 2-core CPU: speech-to-text 5-8 min per class (the bottleneck), video 0.5-1 min, AI judging 10-40 s.

## Privacy

* Recordings are processed in a temporary folder and deleted afterwards; nothing is stored. Video frames are analysed locally.
* Only de-identified transcript text can go to a cloud LLM, and only with `ALLOW_EXTERNAL_LLM=1`. Name masking uses an NER model and can miss unusual names. Use local Ollama (`LLM_PROVIDER=ollama`) for fully private processing.
* Free cloud tiers have their own data terms (e.g. Google's free Gemini tier may use submitted content to improve its products): check them before use.

## Known limitations

* Speaker separation (tutor vs children) is weak (about 56% balanced accuracy) and is not used for judging.
* Quota and overload on free LLM tiers: results can fall back to a weaker model or to a measured-signals estimate (always flagged in the report).
* Zero-shot NLI verification of claims was tested and dropped: it wrongly rejected valid events.
* Brand-logo detection is specific to the configured reference image.

## Tests

`python -m pytest tests -q` (about 60 tests: providers, fallback/circuit breaker, scoring, judge, access gate, pipeline).
