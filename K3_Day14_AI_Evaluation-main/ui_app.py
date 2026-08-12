"""Premium Streamlit demo for the Day 14 evaluation pipeline.

The UI never fabricates benchmark artifacts. Results are shown only after a
real OpenAI run has produced ``actual_answers.json`` and the evaluation core
has generated ``benchmark_results.json``.
"""

from __future__ import annotations

import json
import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from domain_assistant import DomainAssistant, OpenAIGenerator, generate_actual_answers
from evaluate_answers import build_evaluation_artifact, load_evaluation_inputs
from populate_live_reports import main as populate_live_reports
from template import BenchmarkRunner, FailureAnalyzer, RAGASEvaluator
from validate_golden_dataset import build_contract, validate_dataset


ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "golden_dataset.json"
CORPUS_PATH = ROOT / "data" / "student_services"
ACTUAL_PATH = ROOT / "artifacts" / "actual_answers.json"
BENCHMARK_PATH = ROOT / "artifacts" / "benchmark_results.json"

DIFFICULTY_COLORS = {
    "easy": "#36d399",
    "medium": "#60a5fa",
    "hard": "#f59e0b",
    "adversarial": "#fb7185",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def dataset_status() -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    dataset = read_json(GOLDEN_PATH) or {}
    contract = build_contract(CORPUS_PATH)
    errors, stats = validate_dataset(dataset, contract)
    return errors, stats, dataset


def benchmark_dataframe(benchmark: dict[str, Any]) -> pd.DataFrame:
    rows = benchmark.get("results", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "context_recall": "Context Recall",
            "context_precision": "Context Precision",
            "faithfulness": "Faithfulness",
            "relevance": "Relevance",
            "completeness": "Completeness",
            "overall": "Overall",
        }
    )


def metric_card(label: str, value: str, note: str, tone: str = "mint") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_label(score: float) -> str:
    if score >= 0.8:
        return "Good"
    if score >= 0.6:
        return "Needs work"
    return "Significant issue"


def metric_gate(score: float, threshold: float) -> tuple[str, str]:
    if score >= threshold:
        return "PASS", "mint"
    return "FAIL", "rose"


def render_overview(
    errors: list[str], stats: dict[str, Any], benchmark: dict[str, Any] | None
) -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">NORTHSTAR · EVALUATION COMMAND CENTER</div>
          <h1>Measure what your AI<br><span>actually does.</span></h1>
          <p>Golden dataset → live RAG → five metrics → failure analysis → regression gate.</p>
          <div class="hero-tags"><b>20 QA</b><b>10 sources</b><b>5 metrics</b><b>Real OpenAI run</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = benchmark.get("summary", {}) if benchmark else {}
    cols = st.columns(4)
    with cols[0]:
        metric_card("Dataset", "PASS" if not errors else "FAIL", "Evidence provenance", "mint" if not errors else "rose")
    with cols[1]:
        metric_card("Coverage", f"{len(stats['used_documents'])}/10", "Source documents", "blue")
    with cols[2]:
        pass_rate = summary.get("pass_rate")
        metric_card("Pass rate", f"{pass_rate:.0%}" if pass_rate is not None else "—", "Live benchmark only", "amber")
    with cols[3]:
        metric_card("API evidence", "REAL" if benchmark else "PENDING", "No synthetic result", "mint" if benchmark else "slate")

    st.markdown("### Pipeline architecture")
    st.markdown(
        """
        <div class="pipeline">
          <div><span>01</span><b>Golden Dataset</b><small>20 stratified QA + verbatim evidence</small></div>
          <i>→</i><div><span>02</span><b>BM25 + OpenAI</b><small>Real retrieval traces and answers</small></div>
          <i>→</i><div><span>03</span><b>Evaluation Core</b><small>Recall, precision, grounding, relevance</small></div>
          <i>→</i><div><span>04</span><b>Failure Lab</b><small>Clusters, 5 Whys, regression gate</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What is measured")
    names = [
        ("Context Recall", "Did retrieval find all required evidence?"),
        ("Context Precision", "Did relevant evidence rank before noise?"),
        ("Faithfulness", "Is the answer grounded in gold context?"),
        ("Relevance", "Does the answer address the user's intent?"),
        ("Completeness", "Are conditions, dates and exceptions preserved?"),
    ]
    cols = st.columns(5)
    for column, (name, description) in zip(cols, names):
        with column:
            st.markdown(f"<div class='mini-card'><b>{name}</b><small>{description}</small></div>", unsafe_allow_html=True)


def run_saved_evaluation(actual_artifact: dict[str, Any]) -> dict[str, Any]:
    save_json(ACTUAL_PATH, actual_artifact)
    qa_pairs, answers_by_question = load_evaluation_inputs(GOLDEN_PATH, ACTUAL_PATH)

    def recorded_agent(question: str) -> str:
        return answers_by_question[question]

    runner = BenchmarkRunner()
    results = runner.run(qa_pairs, recorded_agent, RAGASEvaluator())
    summary = runner.generate_report(results)
    benchmark = build_evaluation_artifact(results, summary, FailureAnalyzer())
    save_json(BENCHMARK_PATH, benchmark)
    return benchmark


def render_run_lab(errors: list[str], dataset: dict[str, Any]) -> None:
    st.markdown("## Run a real benchmark")
    st.caption("The API key stays in this process for the run and is never written to an artifact or source file.")
    if errors:
        st.error("Golden dataset is invalid. Fix validation errors before inference.")
        return

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Runtime configuration")
        api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-…", help="Used only for this live run.")
        model = st.text_input("Model", value="gpt-4o-mini")
        top_k = st.slider("Retrieved chunks per question", 2, 8, 5)
        selected_id = st.selectbox(
            "Smoke-test question",
            [pair["id"] for pair in dataset.get("qa_pairs", [])],
            index=0,
        )

        smoke_clicked = st.button("Test API with 1 question", use_container_width=True)
        full_clicked = st.button("Run all 20 questions", type="primary", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="run-card">
              <div class="live-dot"></div><b>Truth-first execution</b>
              <p>Every answer is returned by the configured OpenAI model. Retrieval chunks, BM25 scores, model name and timestamp are persisted for audit.</p>
              <ul><li>No expected-answer leakage</li><li>No placeholder benchmark</li><li>Stops on API error</li></ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not (smoke_clicked or full_clicked):
        return
    if not api_key.strip():
        st.error("Enter a real OpenAI API key before running.")
        return
    if not model.strip():
        st.error("Model name cannot be empty.")
        return

    original_key = os.environ.get("OPENAI_API_KEY")
    original_model = os.environ.get("OPENAI_MODEL")
    os.environ["OPENAI_API_KEY"] = api_key.strip()
    os.environ["OPENAI_MODEL"] = model.strip()
    try:
        if smoke_clicked:
            pair = next(item for item in dataset["qa_pairs"] if item["id"] == selected_id)
            with st.status(f"Calling {model.strip()} for {selected_id}…", expanded=True) as status:
                assistant = DomainAssistant.from_corpus(CORPUS_PATH, top_k=top_k)
                response = assistant.answer_with_trace(pair["question"])
                status.update(label="Live API smoke test completed", state="complete")
            st.success("Real model response received. This smoke test did not create a benchmark artifact.")
            st.markdown("#### Actual answer")
            st.write(response.actual_answer)
            with st.expander("Retrieved trace", expanded=True):
                for rank, chunk in enumerate(response.retrieved_chunks, start=1):
                    st.markdown(f"**#{rank} · {chunk.source_doc} · BM25 {chunk.score:.4f}**")
                    st.caption(chunk.text)
        else:
            progress_bar = st.progress(0.0, text="Preparing live benchmark…")
            log_box = st.empty()
            log_lines: list[str] = []

            def progress(message: str) -> None:
                log_lines.append(message)
                if len(log_lines) > 8:
                    del log_lines[0]
                log_box.code("\n".join(log_lines), language=None)
                if "/20" in message:
                    try:
                        completed = int(message.split("/20")[0].split()[-1])
                        progress_bar.progress(completed / 20, text=message)
                    except (ValueError, IndexError):
                        pass

            actual = generate_actual_answers(
                GOLDEN_PATH,
                CORPUS_PATH,
                top_k=top_k,
                progress=progress,
            )
            benchmark = run_saved_evaluation(actual)
            if populate_live_reports() != 0:
                raise RuntimeError("Real artifacts were saved, but the worksheet report could not be populated")
            progress_bar.progress(1.0, text="20/20 answers evaluated and saved")
            st.session_state["benchmark_complete"] = True
            st.success(
                f"Real benchmark complete — pass rate {benchmark['summary']['pass_rate']:.1%}. Open Results and Failure Lab."
            )
    except Exception as exc:  # show provider/runtime errors without fabricating output
        st.error(f"Live run stopped: {exc}")
        st.info("No complete benchmark result was invented. Fix the key/model/network issue and run again.")
    finally:
        if original_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original_key
        if original_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = original_model


def render_results(benchmark: dict[str, Any] | None) -> None:
    st.markdown("## Benchmark results")
    if not benchmark:
        st.info("No real benchmark artifact exists yet. Run the 20-question API benchmark first.")
        return
    summary = benchmark["summary"]
    frame = benchmark_dataframe(benchmark)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Pass rate", f"{summary['pass_rate']:.1%}", f"{summary['passed']}/{summary['total']} cases", "mint")
    with cols[1]:
        metric_card("Faithfulness", f"{summary['avg_faithfulness']:.3f}", score_label(summary["avg_faithfulness"]), "blue")
    with cols[2]:
        metric_card("Relevance", f"{summary['avg_relevance']:.3f}", score_label(summary["avg_relevance"]), "amber")
    with cols[3]:
        metric_card("Completeness", f"{summary['avg_completeness']:.3f}", score_label(summary["avg_completeness"]), "rose")

    st.markdown("### Metric verify")
    gate_rows = [
        ("Pass rate", summary["pass_rate"], 0.70, "Gate for overall benchmark health"),
        ("Faithfulness", summary["avg_faithfulness"], 0.70, "Safety and grounding gate"),
        ("Relevance", summary["avg_relevance"], 0.65, "Intent coverage gate"),
        ("Completeness", summary["avg_completeness"], 0.65, "Condition/exception coverage gate"),
    ]
    gate_cols = st.columns(len(gate_rows))
    for column, (label, value, threshold, note) in zip(gate_cols, gate_rows):
        status, tone = metric_gate(float(value), threshold)
        with column:
            metric_card(
                label,
                f"{float(value):.3f}" if label != "Pass rate" else f"{float(value):.1%}",
                f"{status} · threshold {threshold:.2f} · {note}",
                tone,
            )

    score_columns = ["Context Recall", "Context Precision", "Faithfulness", "Relevance", "Completeness"]
    chart_left, chart_right = st.columns([1.15, 0.85])
    with chart_left:
        averages = [float(frame[column].mean()) for column in score_columns]
        radar = go.Figure(
            go.Scatterpolar(r=averages + averages[:1], theta=score_columns + score_columns[:1], fill="toself", line_color="#36d399", fillcolor="rgba(54,211,153,.20)")
        )
        radar.update_layout(
            title="Metric fingerprint",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#27344a"), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dbe7f5", height=420, margin=dict(l=35, r=35, t=65, b=25), showlegend=False,
        )
        st.plotly_chart(radar, use_container_width=True)
    with chart_right:
        by_difficulty = frame.groupby("difficulty")["Overall"].mean().reset_index()
        bars = px.bar(by_difficulty, x="difficulty", y="Overall", color="difficulty", color_discrete_map=DIFFICULTY_COLORS, range_y=[0, 1], title="Overall by difficulty")
        bars.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dbe7f5", height=420, showlegend=False, margin=dict(l=25, r=25, t=65, b=25))
        bars.update_yaxes(gridcolor="#27344a")
        st.plotly_chart(bars, use_container_width=True)

    st.markdown("### Case matrix")
    display_columns = ["id", "difficulty", *score_columns, "Overall", "passed", "failure_type"]
    st.dataframe(
        frame[display_columns].style.format({column: "{:.3f}" for column in [*score_columns, "Overall"]}),
        use_container_width=True,
        hide_index=True,
        height=570,
    )
    st.download_button(
        "Download benchmark JSON",
        data=json.dumps(benchmark, ensure_ascii=False, indent=2),
        file_name="benchmark_results.json",
        mime="application/json",
    )


def render_failure_lab(benchmark: dict[str, Any] | None) -> None:
    st.markdown("## Failure analysis lab")
    if not benchmark:
        st.info("Failure analysis becomes available after a real benchmark run.")
        return
    frame = benchmark_dataframe(benchmark)
    failures = frame[~frame["passed"]].sort_values("Overall")
    if failures.empty:
        st.success("All cases passed the core quality gate. Inspect the lowest scores for preventive improvements.")
        failures = frame.sort_values("Overall").head(3)

    left, right = st.columns([0.72, 1.28])
    with left:
        counts = benchmark["failure_analysis"].get("counts", {})
        if counts:
            pie = px.pie(names=list(counts), values=list(counts.values()), hole=0.62, color=list(counts), color_discrete_sequence=["#fb7185", "#f59e0b", "#60a5fa", "#36d399"])
            pie.update_layout(title="Failure clusters", paper_bgcolor="rgba(0,0,0,0)", font_color="#dbe7f5", height=360, margin=dict(l=20, r=20, t=55, b=15), showlegend=True)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.markdown("<div class='empty-ring'>0<small>failures</small></div>", unsafe_allow_html=True)
    with right:
        st.markdown("#### Priority actions")
        for index, suggestion in enumerate(benchmark["failure_analysis"].get("suggestions", []), start=1):
            st.markdown(f"<div class='action-row'><span>{index:02d}</span><p>{suggestion}</p></div>", unsafe_allow_html=True)

    st.markdown("### Lowest-scoring cases")
    for _, row in failures.head(3).iterrows():
        with st.expander(f"{row['id']} · {row['failure_type'] or 'preventive review'} · overall {row['Overall']:.3f}", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Faithfulness", f"{row['Faithfulness']:.3f}")
            cols[1].metric("Relevance", f"{row['Relevance']:.3f}")
            cols[2].metric("Completeness", f"{row['Completeness']:.3f}")
            st.markdown("**Question**")
            st.write(row["question"])
            st.markdown("**Actual answer**")
            st.write(row["actual_answer"])

    st.markdown("### Improvement log")
    st.markdown(benchmark["failure_analysis"].get("improvement_log", "No failures."))


def render_trace_explorer(
    benchmark: dict[str, Any] | None, actual: dict[str, Any] | None, dataset: dict[str, Any]
) -> None:
    st.markdown("## Audit trace explorer")
    if not benchmark or not actual:
        st.info("Both real answer and benchmark artifacts are required for trace inspection.")
        return
    result_by_id = {item["id"]: item for item in benchmark["results"]}
    answer_by_id = {item["id"]: item for item in actual["answers"]}
    gold_by_id = {item["id"]: item for item in dataset["qa_pairs"]}
    chosen = st.selectbox("Case", list(result_by_id), format_func=lambda item_id: f"{item_id} · {result_by_id[item_id]['difficulty']} · {result_by_id[item_id]['overall']:.3f}")
    result, answer, gold = result_by_id[chosen], answer_by_id[chosen], gold_by_id[chosen]

    st.markdown(f"### {chosen} · {gold['question']}")
    cols = st.columns(5)
    for column, key, label in zip(cols, ["context_recall", "context_precision", "faithfulness", "relevance", "completeness"], ["Ctx recall", "Ctx precision", "Faithfulness", "Relevance", "Completeness"]):
        column.metric(label, f"{result[key]:.3f}")
    actual_col, expected_col = st.columns(2)
    with actual_col:
        st.markdown("#### Actual model answer")
        safe_actual = html.escape(answer["actual_answer"])
        st.markdown(f"<div class='answer-panel actual'>{safe_actual}</div>", unsafe_allow_html=True)
    with expected_col:
        st.markdown("#### Expected answer")
        safe_expected = html.escape(gold["expected_answer"])
        st.markdown(f"<div class='answer-panel expected'>{safe_expected}</div>", unsafe_allow_html=True)

    retrieved_tab, gold_tab = st.tabs(["Retrieved chunks", "Gold evidence"])
    with retrieved_tab:
        for rank, chunk in enumerate(answer["retrieved_contexts"], start=1):
            st.markdown(f"**#{rank} · {chunk['source_doc']} · {chunk['chunk_id']} · BM25 {chunk['score']:.4f}**")
            st.caption(chunk["text"])
            st.divider()
    with gold_tab:
        for rank, evidence in enumerate(gold["contexts"], start=1):
            st.markdown(f"**Evidence {rank} · {evidence['source_doc']}**")
            st.caption(evidence["text"])
            st.divider()


def render_dataset(errors: list[str], stats: dict[str, Any], dataset: dict[str, Any]) -> None:
    st.markdown("## Golden dataset studio")
    if errors:
        st.error(f"Validator found {len(errors)} error(s).")
        for error in errors:
            st.write(f"- {error}")
    else:
        st.success("PASS — structure, stratification and verbatim evidence provenance are valid.")
    cols = st.columns(4)
    counts = stats["difficulty_counts"]
    for column, difficulty in zip(cols, ["easy", "medium", "hard", "adversarial"]):
        column.metric(difficulty.title(), counts[difficulty])

    for pair in dataset.get("qa_pairs", []):
        with st.expander(f"{pair['id']} · {pair['difficulty'].title()} · {pair['question']}"):
            st.markdown("**Expected answer**")
            st.write(pair["expected_answer"])
            st.markdown("**Evidence provenance**")
            for context in pair["contexts"]:
                st.caption(f"{context['source_doc']} — {context['text']}")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#08111f; --panel:#101c2f; --line:#24344c; --text:#eaf2fb; --muted:#8da2bd; --mint:#36d399; }
        .stApp { background:radial-gradient(circle at 18% 0%,#12314a 0,transparent 35%),radial-gradient(circle at 88% 12%,#183042 0,transparent 30%),#08111f; color:var(--text); }
        html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
        h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
        [data-testid="stSidebar"] { background:#0b1626; border-right:1px solid #1e3047; }
        [data-testid="stHeader"] { background:rgba(8,17,31,.72); backdrop-filter:blur(16px); }
        .block-container { max-width:1460px; padding-top:2rem; padding-bottom:5rem; }
        .hero { padding:64px 68px; border:1px solid #26425a; border-radius:28px; background:linear-gradient(125deg,rgba(13,34,53,.97),rgba(12,24,42,.86)); position:relative; overflow:hidden; margin-bottom:24px; box-shadow:0 30px 90px rgba(0,0,0,.26); }
        .hero:after { content:''; position:absolute; width:420px;height:420px;border-radius:50%;right:-140px;top:-220px;background:radial-gradient(circle,rgba(54,211,153,.28),transparent 68%); }
        .eyebrow { color:#78e8c0; font-size:.78rem; font-weight:700; letter-spacing:.2em; }
        .hero h1 { font-size:clamp(3rem,6vw,6rem); line-height:.92; margin:20px 0; max-width:900px; }
        .hero h1 span { color:#52ddb0; }
        .hero p { color:#a9bad0; font-size:1.12rem; max-width:720px; }
        .hero-tags { display:flex;gap:10px;flex-wrap:wrap;margin-top:26px; }
        .hero-tags b { background:#142941;border:1px solid #29435f;border-radius:99px;padding:9px 14px;font-size:.82rem;color:#cfe0f2; }
        .metric-card { min-height:148px;padding:22px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#101d30,#0d1828);position:relative;overflow:hidden; }
        .metric-card:after { content:'';position:absolute;width:70px;height:70px;border-radius:50%;right:-22px;bottom:-28px;opacity:.35; }
        .metric-card.mint:after{background:#36d399}.metric-card.blue:after{background:#60a5fa}.metric-card.amber:after{background:#f59e0b}.metric-card.rose:after{background:#fb7185}.metric-card.slate:after{background:#64748b}
        .metric-label{color:#8da2bd;text-transform:uppercase;letter-spacing:.12em;font-size:.7rem;font-weight:700}.metric-value{font-family:'Space Grotesk';font-size:2.2rem;font-weight:700;margin:12px 0 4px}.metric-note{color:#91a5bf;font-size:.78rem}
        .pipeline{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:13px;align-items:center;margin:12px 0 30px}.pipeline>div{background:#0f1b2d;border:1px solid #243750;border-radius:16px;padding:20px;min-height:132px}.pipeline span{display:block;color:#4ee0ae;font-size:.72rem;letter-spacing:.16em;margin-bottom:14px}.pipeline b,.pipeline small{display:block}.pipeline small{color:#89a0bb;margin-top:7px;line-height:1.45}.pipeline i{color:#4c6380;font-style:normal;font-size:1.4rem}
        .mini-card{background:#0e1a2b;border:1px solid #223650;border-radius:14px;padding:18px;min-height:124px}.mini-card b,.mini-card small{display:block}.mini-card small{color:#8fa4bd;line-height:1.45;margin-top:8px}
        .run-card{border:1px solid #285a50;background:linear-gradient(145deg,#102b2a,#0e1b29);border-radius:20px;padding:28px;min-height:310px}.run-card p,.run-card li{color:#9cb1c8;line-height:1.6}.live-dot{display:inline-block;width:10px;height:10px;background:#36d399;border-radius:50%;margin-right:10px;box-shadow:0 0 0 7px rgba(54,211,153,.1)}
        .action-row{display:flex;gap:18px;align-items:flex-start;padding:18px 0;border-bottom:1px solid #203049}.action-row span{color:#36d399;font-family:'Space Grotesk';font-weight:700}.action-row p{margin:0;color:#c4d3e3}.empty-ring{width:220px;height:220px;border-radius:50%;border:24px solid #19352f;margin:40px auto;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:3.2rem;color:#36d399}.empty-ring small{display:block;font-size:.8rem;color:#8da2bd}
        .answer-panel{border-radius:16px;padding:22px;min-height:180px;line-height:1.65;border:1px solid #273b55}.answer-panel.actual{background:#0f2132}.answer-panel.expected{background:#122820;border-color:#285444}
        div[data-testid="stDataFrame"]{border:1px solid #23364e;border-radius:14px;overflow:hidden}div[data-baseweb="tab-list"]{gap:8px}button[data-baseweb="tab"]{background:#101c2e;border-radius:10px;padding:10px 18px}
        .stButton>button{border-radius:12px;min-height:48px;font-weight:700}.stButton>button[kind="primary"]{background:linear-gradient(90deg,#20b985,#38d8a4);color:#052018;border:0}
        @media(max-width:900px){.hero{padding:36px 26px}.pipeline{grid-template-columns:1fr}.pipeline i{transform:rotate(90deg);text-align:center}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Northstar Eval Command Center", page_icon="◉", layout="wide")
    inject_styles()
    errors, stats, dataset = dataset_status()
    benchmark = read_json(BENCHMARK_PATH)
    actual = read_json(ACTUAL_PATH)

    with st.sidebar:
        st.markdown("## ◉ Northstar Eval")
        st.caption("AI Evaluation · Day 14")
        page = st.radio("Navigate", ["Overview", "Run API", "Results", "Failure Lab", "Trace Explorer", "Golden Dataset"], label_visibility="collapsed")
        st.divider()
        st.markdown("**System status**")
        st.write("🟢 Dataset validated" if not errors else "🔴 Dataset invalid")
        st.write("🟢 Real answers saved" if actual else "⚪ Real answers pending")
        st.write("🟢 Benchmark ready" if benchmark else "⚪ Benchmark pending")
        st.divider()
        st.caption("Results appear only from persisted real API artifacts. No sample scores are displayed as evidence.")

    if page == "Overview":
        render_overview(errors, stats, benchmark)
    elif page == "Run API":
        render_run_lab(errors, dataset)
    elif page == "Results":
        render_results(benchmark)
    elif page == "Failure Lab":
        render_failure_lab(benchmark)
    elif page == "Trace Explorer":
        render_trace_explorer(benchmark, actual, dataset)
    else:
        render_dataset(errors, stats, dataset)


if __name__ == "__main__":
    main()
