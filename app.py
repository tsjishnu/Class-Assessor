"""Streamlit front-end: upload (or link) a class recording -> grade + evidenced observations."""
from __future__ import annotations

import hmac
import json
import os
import threading
import time

import streamlit as st

from assessor import llm, media, pipeline

st.set_page_config(page_title="Class Quality Assessor", page_icon="♟️", layout="wide")

GRADE_COLOR = {"Excellent": "#1a7f37", "Good": "#2f81f7", "Average": "#bf8700", "Poor": "#cf222e"}
ICON = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️"}
_LOCK = threading.Lock()  # one assessment at a time on a small free CPU
ALLOWED = ["mp4", "mov", "mkv", "webm", "m4a", "mp3", "wav"]


def badge(grade: str) -> str:
    c = GRADE_COLOR.get(grade, "#666")
    return f'<span style="background:{c};color:#fff;padding:.35rem 1rem;border-radius:999px;font-size:1.6rem;font-weight:700">{grade}</span>'


def save_upload(uploaded, dest_dir: str) -> str:
    path = os.path.join(dest_dir, "input" + os.path.splitext(uploaded.name)[1].lower())
    with open(path, "wb") as f:
        while True:
            chunk = uploaded.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    return path


def render(report: dict, summary: str, feats: dict):
    st.markdown(badge(report["grade"]), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score", f"{report['score']}/100")
    c2.metric("Confidence", report["confidence"])
    c3.metric("Recording length", report["signals"]["duration"])
    st.write(summary)
    for fl in report["flags"]:
        st.warning(fl)

    left, right = st.columns(2)
    for col, title, items, empty in ((left, "What went well", report["strengths"], "No clear strengths were detected."),
                                     (right, "What to improve", report["improvements"], "No significant problems were detected.")):
        with col:
            st.subheader(title)
            if not items:
                st.caption(empty)
            for s in items:
                st.markdown(f"**{s['dimension']}**  \n{s['text']}")
                for e in s["evidence"]:
                    st.caption(f"↳ {e}")

    st.subheader("Assessment by dimension")
    for d in report["dimensions"]:
        score = "—" if d["score"] is None else f"{d['score']:.0f}"
        with st.expander(f"{d['title']} — {d['label']} ({score}/100)"):
            if d["score"] is not None:
                st.progress(min(max(d["score"] / 100, 0), 1))
            for o in d["observations"]:
                st.markdown(f"{ICON.get(o['impact'], '')} {o['text']}")
                for e in o["evidence"]:
                    st.caption(f"↳ {e}")
            for sk in d["skipped"]:
                st.caption(sk)

    with st.expander("How this assessment was produced"):
        sg, ai = report["signals"], report["ai"]
        if ai.get("model"):
            where = "locally on this server (private)" if ai.get("local") else "via an external free LLM API (de-identified text only)"
            st.write(f"**AI judge:** {ai['provider']} / {ai['model']}, {where}. Dimensions judged: {ai['dimensions_judged']}/6.")
            if ai.get("fallbacks"):
                st.caption("Providers skipped before this one answered: " + "; ".join(ai["fallbacks"]))
            if ai.get("decided") is not None:
                st.write(f"**Jev decider:** {ai['decided']} questions answered confidently, {ai['uncertain']} uncertain"
                         + (f", {ai['escalated']} escalated to {ai['escalated_to']}." if ai.get("escalated_to") else " (no generative provider configured to escalate to)."))
        else:
            st.write("**AI judge:** not available; transcript dimensions use measured proxies only.")
        st.write(f"**Time windows analysed:** {', '.join(sg['windows_analysed'])}")
        st.write(f"**Video frames analysed:** {sg['frames_analysed']}  ·  **Transcribed segments:** {sg['transcribed_segments']}")
        if sg.get("screen_content"):
            st.write("**On-screen content (AI zero-shot):** " + ", ".join(f"{k} {v:.0%}" for k, v in sg["screen_content"].items()))
        if sg.get("language"):
            st.write(f"**Detected language:** {sg['language']}")
        st.caption("Timings (s): " + json.dumps(feats.get("timings", {})))
    st.download_button("Download full report (JSON)", json.dumps({"report": report, "summary": summary}, indent=2), "class_assessment.json", "application/json")


def check_access() -> None:
    """Optional access code (APP_ACCESS_CODE). On a public deployment this stops strangers from using the operator's AI quota."""
    code = os.environ.get("APP_ACCESS_CODE", "")
    if not code or st.session_state.get("authorised"):
        return
    st.title("♟️ Class Quality Assessor")
    entered = st.text_input("Access code", type="password", help="Provided with the submission.")
    if entered:
        if hmac.compare_digest(entered.encode(), code.encode()):
            st.session_state["authorised"] = True
            st.rerun()
        st.error("Incorrect access code.")
    st.stop()


@st.cache_resource(show_spinner=False)
def warm_up() -> bool:
    """Load the heavy models once in the background so the first assessment does not wait for them."""
    def _load():
        try:
            from assessor import speech, vision

            speech.get_model()
            vision._clip_models()
        except Exception as e:  # warm-up is an optimisation only
            pipeline.perf.log.warning("model warm-up failed: %s", e)

    threading.Thread(target=_load, name="warmup", daemon=True).start()
    return True


def pick_provider():
    """Sidebar selector over the providers that are configured right now (switch live; default from LLM_PROVIDER)."""
    avail = llm.available_providers()
    default = llm.resolve_provider()
    options = list(avail) + ["none"]
    idx = options.index(default.name if default and default.name in avail else "none") if (default and default.name in avail) else len(options) - 1
    with st.sidebar:
        st.header("AI judge")
        choice = st.selectbox("Provider", options, index=idx, format_func=lambda n: "none (measured proxies only)" if n == "none" else f"{n}  ·  {avail[n].model}  ·  {'local' if avail[n].local else 'cloud'}")
        missing = [n for n in ("gemini", "groq", "openai", "ollama", "jev") if n not in avail]
        if missing:
            with st.expander("Not configured"):
                for n in missing:
                    st.caption(f"**{n}**: {llm.explain_unavailable(n)}")
        st.caption("Change the default with LLM_PROVIDER in .env. Cloud providers receive de-identified transcript text only.")
    return avail.get(choice)


def main():
    check_access()
    warm_up()
    provider = pick_provider()
    st.title("♟️ Class Quality Assessor")
    st.caption(
        "Upload an online class recording. AI models transcribe the audio, look at sampled video frames and judge the class "
        "against a classroom-quality rubric, then grade it **Poor / Average / Good / Excellent** and explain why with quotes and measurements."
    )
    if provider is None:
        st.warning("No AI judge is configured: only measured proxies will be used (low confidence). See README to start Ollama or set a free API key.")
    elif provider.local:
        st.info(f"Privacy: the recording is processed on this server and deleted afterwards; the AI judge ({provider.model}) runs locally. Nothing is sent to a third party.", icon="🔒")
    else:
        st.info(f"Privacy: audio and video stay on this server. Only de-identified transcript text is sent to {provider.name}, enabled explicitly by the operator.", icon="🔒")

    tab_up, tab_link = st.tabs(["Upload a file", "Paste a link"])
    with tab_up:
        uploaded = st.file_uploader("Class recording", type=ALLOWED, help="Zoom/Meet recordings (MP4 etc.). Up to ~1 GB.")
    with tab_link:
        link = st.text_input("Direct video URL or public Google Drive link")
    with st.expander("Options"):
        brand = st.checkbox("Check the branded virtual background", value=True, help="Uses assets/logo_template.png; turn off for recordings without the brand.")

    if st.button("Assess class", type="primary", disabled=not (uploaded or link.strip())):
        if not _LOCK.acquire(blocking=False):
            st.info("Another assessment is running on this server; waiting for it to finish...")
            _LOCK.acquire()
        work = media.make_workdir()
        try:
            with st.status("Processing recording...", expanded=True) as status:
                msg = st.empty()
                t0 = time.time()
                if uploaded:
                    msg.write("Receiving file...")
                    path = save_upload(uploaded, work)
                else:
                    path = media.fetch_link(link, work, msg.write)
                report, summary, feats = pipeline.run(path, progress=msg.write, brand_check=brand, provider=provider)
                status.update(label=f"Done in {time.time() - t0:.0f}s", state="complete", expanded=False)
            render(report, summary, feats)
        except Exception as e:
            st.error(f"Could not assess this recording: {e}")
        finally:
            media.cleanup(work)
            _LOCK.release()


main()
