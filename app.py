import json
import os
from datetime import datetime
from typing import List, Literal, Optional

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="InterviewIQ | AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_QUESTION_COUNT = 5
MAX_ANSWER_CHARS = 8000

# ---------------------------------------------------------------------
# Structured response models
# ---------------------------------------------------------------------
class InterviewQuestion(BaseModel):
    question: str = Field(description="One clear interview question only.")
    question_focus: str = Field(description="The skill or competency this question assesses.")


class AnswerEvaluation(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall answer score from 1 to 10.")
    strengths: List[str] = Field(description="Specific strengths demonstrated in the candidate's answer.")
    improvements: List[str] = Field(description="Specific, actionable ways to improve the answer.")
    star_feedback: str = Field(description="Feedback on Situation, Task, Action, and Result structure where relevant.")
    clarity_feedback: str = Field(description="Feedback on organization, clarity, and relevance.")
    technical_feedback: str = Field(description="Feedback on technical accuracy and depth, or why this is not applicable.")
    sample_answer: str = Field(description="A strong, realistic example answer tailored to the role and question.")
    short_verdict: str = Field(description="One-sentence overall assessment.")


class InterviewSummary(BaseModel):
    overall_feedback: str = Field(description="A balanced overall assessment of the complete interview.")
    top_strengths: List[str] = Field(description="The candidate's strongest recurring skills.")
    priority_improvements: List[str] = Field(description="The most important improvements to work on next.")
    preparation_plan: List[str] = Field(description="Practical next steps for interview preparation.")
    closing_message: str = Field(description="Encouraging, realistic closing note.")


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#1d2340; --muted:#75809a; --line:#e7eaf3; --purple:#6d5ce8; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
        background:
          radial-gradient(circle at 90% 0%, rgba(129,105,255,.12), transparent 26rem),
          radial-gradient(circle at 0% 35%, rgba(71,160,213,.07), transparent 25rem),
          #f7f8fc;
        color:var(--ink);
    }
    [data-testid="stHeader"] { background:rgba(247,248,252,.82); }
    [data-testid="stSidebar"] { background:#17182d; border-right:1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] * { color:#eef0ff !important; }
    [data-testid="stSidebar"] .stCaption { color:#b8bdd8 !important; }
    .hero {
        padding:1.65rem 1.85rem; border-radius:24px; color:white;
        background:linear-gradient(118deg,#20213f 0%,#45418e 55%,#785be0 100%);
        box-shadow:0 18px 44px rgba(43,39,104,.18);
        position:relative; overflow:hidden; margin-bottom:1.25rem;
    }
    .hero:after { content:"✦"; position:absolute; right:7%; top:-46px; font-size:170px; color:rgba(255,255,255,.08); }
    .hero-kicker { color:#d9d5ff; font-size:.75rem; font-weight:700; letter-spacing:.15em; }
    .hero h1 { color:white; font:800 clamp(2rem,4vw,2.8rem)/1.12 'Manrope',sans-serif; margin:.4rem 0 .55rem; }
    .hero p { color:#e7e5ff; margin:0; max-width:690px; }
    .section-title { font:800 1.12rem 'Manrope',sans-serif; color:#222743; margin:.25rem 0 .5rem; }
    .muted { color:#7c869d; font-size:.88rem; }
    .surface { background:white; border:1px solid var(--line); border-radius:17px; padding:1rem 1.15rem; box-shadow:0 8px 25px rgba(31,41,75,.035); }
    .step { background:#fff; border:1px solid var(--line); border-radius:15px; padding:.85rem 1rem; }
    .step-label { color:#808aa0; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; }
    .step-value { color:#29264f; font:800 1.2rem 'Manrope',sans-serif; }
    .stButton > button[kind="primary"] { background:linear-gradient(100deg,#6658e8,#8b5cf6); border:0; border-radius:11px; font-weight:700; min-height:2.75rem; box-shadow:0 7px 16px rgba(102,88,232,.18); }
    .stButton > button { border-radius:11px; }
    .stTextArea textarea, .stTextInput input { border-radius:12px !important; border-color:#dfe3ef !important; background:#fdfdff !important; }
    div[data-testid="stExpander"] { border:1px solid #e8eaf3; border-radius:13px; background:#fff; }
    .question-card { background:white; border:1px solid #e6e4fb; border-left:4px solid #7564e8; border-radius:16px; padding:1.15rem 1.25rem; box-shadow:0 8px 24px rgba(43,39,104,.045); }
    .question-kicker { color:#7768db; font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
    .question-text { color:#222743; font:700 1.25rem/1.5 'Manrope',sans-serif; margin-top:.35rem; }
    .score-ring { font:800 2rem 'Manrope',sans-serif; color:#5e50d3; }
    .tag { display:inline-block; background:#f0edff; color:#5c4bd0; padding:.2rem .55rem; border-radius:20px; font-size:.72rem; font-weight:700; }
    .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; font-weight:700; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
def initialize_state():
    defaults = {
        "interview_status": "setup",  # setup, answering, evaluated, complete
        "role": "",
        "experience": "Entry level (0–2 years)",
        "interview_type": "Mixed",
        "question_target": DEFAULT_QUESTION_COUNT,
        "questions": [],
        "current_question": None,
        "current_answer": "",
        "evaluations": [],
        "summary": None,
        "started_at": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ---------------------------------------------------------------------
# API and prompt helpers
# ---------------------------------------------------------------------
def get_api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret = ""
    return str(secret or os.getenv("GROQ_API_KEY", "")).strip()


def get_client() -> Groq:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Add it to Streamlit Cloud → App settings → Secrets, "
            "or set it as an environment variable locally."
        )
    return Groq(api_key=api_key)


def call_json(prompt: str, system_message: str, max_tokens: int = 2200) -> dict:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The AI returned an empty response. Please try again.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The AI returned invalid JSON. Please try again.") from exc


def interview_context() -> str:
    return (
        f"Target role: {st.session_state.role}\n"
        f"Experience level: {st.session_state.experience}\n"
        f"Interview type: {st.session_state.interview_type}\n"
    )


def generate_question() -> InterviewQuestion:
    asked = [item["question"] for item in st.session_state.questions]
    previous_questions = "\n".join(f"- {q}" for q in asked) if asked else "None yet."
    prompt = f"""
Create exactly ONE realistic interview question for this candidate.

{interview_context()}
Questions already asked (do not repeat or lightly rephrase these):
{previous_questions}

Question number: {len(asked) + 1} of {st.session_state.question_target}.

Interview type guidance:
- Behavioral: focus on real past experiences, teamwork, ownership, conflict, leadership, learning, or problem solving.
- Technical: focus on role-relevant concepts, practical scenarios, architecture, tools, analysis, or technical decisions.
- Mixed: alternate between behavioral and technical/role-specific questions.

Return a JSON object with:
- question: one concise but meaningful question, no multiple unrelated questions.
- question_focus: the competency being assessed.

Tailor difficulty to the candidate's experience level. Do not include an answer or hints.
"""
    payload = call_json(
        prompt,
        "You are a professional hiring interviewer. Ask one tailored question at a time. Return only valid JSON.",
        max_tokens=500,
    )
    return InterviewQuestion.model_validate(payload)


def evaluate_answer(question: str, answer: str) -> AnswerEvaluation:
    prompt = f"""
Evaluate this candidate's interview answer fairly and constructively.

CANDIDATE PROFILE:
{interview_context()}

INTERVIEW QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

Return a JSON object with exactly these fields:
- score: integer from 1 to 10. Score relevance, completeness, clarity, and correctness for the candidate's level.
- strengths: list of 2–4 specific things done well. Avoid generic praise.
- improvements: list of 2–4 actionable improvements. Mention missing detail or structure where relevant.
- star_feedback: assess Situation, Task, Action, Result alignment. For technical questions, assess problem/context, approach, implementation, and outcome instead.
- clarity_feedback: assess structure, directness, and communication.
- technical_feedback: assess technical accuracy and depth for technical/role-specific questions; otherwise say not applicable and explain briefly.
- sample_answer: provide a realistic, strong example answer tailored to the role and experience level. Do not invent personal facts about the candidate. For behavioral questions, use a clearly illustrative example the candidate should adapt truthfully.
- short_verdict: one sentence summarizing the answer.

Be specific, supportive, and honest. Do not reward unsupported claims. Do not penalize a beginner for not having senior-level experience.
"""
    payload = call_json(
        prompt,
        "You are an experienced interview coach. Evaluate answers with consistent standards and return only valid JSON.",
        max_tokens=1800,
    )
    return AnswerEvaluation.model_validate(payload)


def generate_summary() -> InterviewSummary:
    records = st.session_state.evaluations
    transcript = []
    for i, item in enumerate(records, 1):
        transcript.append(
            f"Question {i}: {item['question']}\n"
            f"Candidate answer: {item['answer']}\n"
            f"Score: {item['evaluation']['score']}/10\n"
            f"Strengths: {', '.join(item['evaluation']['strengths'])}\n"
            f"Improvements: {', '.join(item['evaluation']['improvements'])}\n"
        )
    prompt = f"""
Prepare a final coaching report based on this completed mock interview.

{interview_context()}
Average score: {sum(x['evaluation']['score'] for x in records) / max(len(records), 1):.1f}/10
Interview transcript and individual evaluations:
{chr(10).join(transcript)}

Return a JSON object with:
- overall_feedback: balanced summary of performance, patterns, and readiness.
- top_strengths: 3–5 recurring strengths, grounded in the answers.
- priority_improvements: 3–5 highest-impact areas to improve.
- preparation_plan: 3–5 practical, specific next steps.
- closing_message: encouraging but realistic closing note.

Do not claim the mock interview guarantees hiring success. Base conclusions only on the supplied answers.
"""
    payload = call_json(
        prompt,
        "You are a senior career coach writing a concise, evidence-based interview report. Return only valid JSON.",
        max_tokens=1800,
    )
    return InterviewSummary.model_validate(payload)


# ---------------------------------------------------------------------
# Transcript export
# ---------------------------------------------------------------------
def build_report_text() -> str:
    lines = [
        "INTERVIEWIQ — MOCK INTERVIEW REPORT",
        "=" * 42,
        f"Role: {st.session_state.role}",
        f"Experience: {st.session_state.experience}",
        f"Interview type: {st.session_state.interview_type}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    evaluations = st.session_state.evaluations
    if evaluations:
        average = sum(x["evaluation"]["score"] for x in evaluations) / len(evaluations)
        lines.append(f"Average score: {average:.1f}/10")
    lines.append("\nQUESTION-BY-QUESTION REVIEW\n" + "-" * 28)
    for i, item in enumerate(evaluations, 1):
        ev = item["evaluation"]
        lines.extend([
            f"\nQUESTION {i}: {item['question']}",
            f"Focus: {item.get('focus', '')}",
            f"\nYOUR ANSWER:\n{item['answer']}",
            f"\nSCORE: {ev['score']}/10",
            f"ASSESSMENT: {ev['short_verdict']}",
            "\nSTRENGTHS:",
            *[f"- {x}" for x in ev["strengths"]],
            "\nAREAS TO IMPROVE:",
            *[f"- {x}" for x in ev["improvements"]],
            f"\nSTAR / STRUCTURE FEEDBACK:\n{ev['star_feedback']}",
            f"\nCLARITY FEEDBACK:\n{ev['clarity_feedback']}",
            f"\nTECHNICAL FEEDBACK:\n{ev['technical_feedback']}",
            f"\nSAMPLE STRONG ANSWER:\n{ev['sample_answer']}",
            "\n" + "-" * 42,
        ])
    summary = st.session_state.summary
    if summary:
        lines.extend([
            "\nFINAL PERFORMANCE SUMMARY",
            "=" * 28,
            f"\nOverall feedback:\n{summary['overall_feedback']}",
            "\nTop strengths:",
            *[f"- {x}" for x in summary["top_strengths"]],
            "\nPriority improvements:",
            *[f"- {x}" for x in summary["priority_improvements"]],
            "\nPreparation plan:",
            *[f"- {x}" for x in summary["preparation_plan"]],
            f"\nClosing message:\n{summary['closing_message']}",
        ])
    return "\n".join(lines)


def reset_interview():
    for key in [
        "interview_status", "role", "experience", "interview_type",
        "question_target", "questions", "current_question", "current_answer",
        "evaluations", "summary", "started_at", "last_error",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    initialize_state()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 InterviewIQ")
    st.caption("Practice with purpose. Improve with feedback.")
    st.divider()
    st.markdown("### Session guide")
    st.markdown(
        """
        **01 · Set your target**  
        Choose a role and interview format.

        **02 · Answer one question**  
        Respond as you would in a real interview.

        **03 · Get coached**  
        Review your score, strengths, and sample answer.

        **04 · See your report**  
        Finish with a performance summary.
        """
    )
    st.divider()
    st.markdown("**AI engine**")
    st.code(MODEL_NAME, language="text")
    st.caption("Groq API · Pydantic-validated JSON")
    if st.session_state.interview_status != "setup":
        st.divider()
        if st.button("↻ Start a new interview", use_container_width=True):
            reset_interview()
            st.rerun()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">PRACTICE · REFLECT · LEVEL UP</div>
      <h1>Walk into your next<br>interview with confidence.</h1>
      <p>A focused mock interview with one question at a time, practical coaching, and a complete performance report.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------
if st.session_state.interview_status == "setup":
    st.markdown('<div class="section-title">01 / Build your interview session</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Tell InterviewIQ what opportunity you are preparing for. Your questions will be tailored to this profile.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            role_choice = st.selectbox(
                "Target job role",
                [
                    "Software Engineer",
                    "Frontend Developer",
                    "Backend Developer",
                    "Full-Stack Developer",
                    "Web Developer",
                    "Data Analyst",
                    "Product Manager",
                    "UI/UX Designer",
                    "Cybersecurity Analyst",
                    "AI / Machine Learning Engineer",
                    "Digital Marketing Specialist",
                    "Other",
                ],
            )
            custom_role = ""
            if role_choice == "Other":
                custom_role = st.text_input("Enter your target role", placeholder="e.g. Cloud Solutions Architect")
            experience = st.selectbox(
                "Experience level",
                [
                    "Entry level (0–2 years)",
                    "Junior (1–3 years)",
                    "Mid-level (3–5 years)",
                    "Senior (5–8 years)",
                    "Lead / Manager (8+ years)",
                    "Career changer / Student",
                ],
            )
        with col2:
            interview_type = st.selectbox(
                "Interview type",
                ["Mixed", "Behavioral", "Technical"],
                help="Mixed combines role-specific and behavioral questions.",
            )
            question_target = st.slider("Number of questions", min_value=3, max_value=10, value=5)
            st.markdown(
                '<div class="muted">Recommended: 5 questions for a focused practice session. You can review detailed feedback after every answer.</div>',
                unsafe_allow_html=True,
            )

        final_role = custom_role.strip() if role_choice == "Other" else role_choice
        start_clicked = st.button("🚀 Start mock interview", type="primary", use_container_width=True)

    if start_clicked:
        if not final_role:
            st.warning("Please enter a target job role.")
        elif not get_api_key():
            st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets before starting.")
        else:
            st.session_state.role = final_role
            st.session_state.experience = experience
            st.session_state.interview_type = interview_type
            st.session_state.question_target = question_target
            st.session_state.questions = []
            st.session_state.evaluations = []
            st.session_state.summary = None
            st.session_state.started_at = datetime.now().isoformat(timespec="seconds")
            with st.spinner("Your interviewer is preparing the first question..."):
                try:
                    first = generate_question()
                    st.session_state.current_question = first.model_dump()
                    st.session_state.interview_status = "answering"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not start the interview: {exc}")


# ---------------------------------------------------------------------
# Active interview
# ---------------------------------------------------------------------
if st.session_state.interview_status in ("answering", "evaluated"):
    total = st.session_state.question_target
    completed = len(st.session_state.evaluations)
    question_number = completed + 1
    progress = min(completed / total, 1.0)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'<div class="step"><div class="step-label">Target role</div><div class="step-value">{st.session_state.role}</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="step"><div class="step-label">Interview format</div><div class="step-value">{st.session_state.interview_type}</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="step"><div class="step-label">Progress</div><div class="step-value">{completed} / {total}</div></div>', unsafe_allow_html=True)

    st.progress(progress, text=f"Question {min(question_number, total)} of {total} · {completed} completed")
    st.write("")

    if st.session_state.interview_status == "answering":
        current = st.session_state.current_question
        st.markdown(
            f"""
            <div class="question-card">
              <div class="question-kicker">QUESTION {question_number:02d} · {current['question_focus']}</div>
              <div class="question-text">{current['question']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        answer_key = f"answer_{question_number}"
        answer_value = st.text_area(
            "Your answer",
            key=answer_key,
            height=220,
            placeholder="Take a moment to structure your thoughts, then write your answer here...",
            max_chars=MAX_ANSWER_CHARS,
            help="For behavioral questions, consider Situation, Task, Action, and Result (STAR).",
        )
        a1, a2 = st.columns([3, 1])
        with a1:
            submit_answer = st.button("Submit answer & get feedback", type="primary", use_container_width=True)
        with a2:
            st.caption("Take your time. Quality matters more than speed.")

        if submit_answer:
            if not answer_value.strip():
                st.warning("Please write an answer before submitting.")
            elif len(answer_value.strip()) < 15:
                st.warning("Please add a little more detail so the coach can evaluate your answer fairly.")
            else:
                with st.spinner("Evaluating your answer and preparing personalized coaching..."):
                    try:
                        evaluation = evaluate_answer(current["question"], answer_value.strip())
                        st.session_state.questions.append(current)
                        st.session_state.evaluations.append({
                            "question": current["question"],
                            "focus": current["question_focus"],
                            "answer": answer_value.strip(),
                            "evaluation": evaluation.model_dump(),
                        })
                        st.session_state.interview_status = "evaluated"
                        st.rerun()
                    except ValidationError as exc:
                        st.error("The evaluation did not match the expected response structure. Please submit again.")
                        with st.expander("Technical details"):
                            st.code(str(exc))
                    except Exception as exc:
                        st.error(f"Could not evaluate this answer: {exc}")

    elif st.session_state.interview_status == "evaluated":
        item = st.session_state.evaluations[-1]
        evaluation = item["evaluation"]
        st.markdown('<div class="section-title">Your answer feedback</div>', unsafe_allow_html=True)
        score_col, verdict_col = st.columns([1, 4])
        with score_col:
            st.markdown(
                f'<div class="surface"><div class="step-label">ANSWER SCORE</div><div class="score-ring">{evaluation["score"]}<span style="font-size:1rem;color:#8991a5">/10</span></div></div>',
                unsafe_allow_html=True,
            )
        with verdict_col:
            st.markdown(f'<div class="surface"><b>Coach’s verdict</b><p style="margin:.45rem 0 0">{evaluation["short_verdict"]}</p></div>', unsafe_allow_html=True)

        good_col, improve_col = st.columns(2)
        with good_col:
            st.markdown("#### <span style='color:#16845d'>✓ Key strengths</span>", unsafe_allow_html=True)
            for strength in evaluation["strengths"]:
                st.markdown(f"- {strength}")
        with improve_col:
            st.markdown("#### <span style='color:#b7791f'>↗ Areas to improve</span>", unsafe_allow_html=True)
            for improvement in evaluation["improvements"]:
                st.markdown(f"- {improvement}")

        with st.expander("STAR / answer structure feedback", expanded=True):
            st.write(evaluation["star_feedback"])
        with st.expander("Clarity and communication feedback"):
            st.write(evaluation["clarity_feedback"])
        with st.expander("Technical depth and accuracy feedback"):
            st.write(evaluation["technical_feedback"])
        with st.expander("View a sample strong answer", expanded=True):
            st.write(evaluation["sample_answer"])
            st.caption("Use this as a learning example. Adapt it to your real experience; do not claim experiences that are not yours.")

        if len(st.session_state.evaluations) < total:
            next_label = f"Continue to question {len(st.session_state.evaluations) + 1}"
            if st.button(f"➡ {next_label}", type="primary", use_container_width=True):
                with st.spinner("Preparing your next tailored question..."):
                    try:
                        next_question = generate_question()
                        st.session_state.current_question = next_question.model_dump()
                        st.session_state.interview_status = "answering"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate the next question: {exc}")
        else:
            if st.button("🏁 Finish interview & generate report", type="primary", use_container_width=True):
                with st.spinner("Reviewing your full interview and building your performance report..."):
                    try:
                        final_summary = generate_summary()
                        st.session_state.summary = final_summary.model_dump()
                        st.session_state.interview_status = "complete"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate the final report: {exc}")


# ---------------------------------------------------------------------
# Completed interview report
# ---------------------------------------------------------------------
if st.session_state.interview_status == "complete":
    records = st.session_state.evaluations
    scores = [x["evaluation"]["score"] for x in records]
    average_score = sum(scores) / len(scores) if scores else 0
    summary = st.session_state.summary

    st.markdown('<div class="section-title">Interview complete · Your performance report</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">A complete review of your answers, progress, and recommended preparation priorities.</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Overall average", f"{average_score:.1f} / 10")
    with r2:
        st.metric("Questions completed", len(records))
    with r3:
        st.metric("Highest answer score", f"{max(scores)}/10" if scores else "—")

    st.markdown("### Overall coaching feedback")
    st.markdown(f'<div class="surface">{summary["overall_feedback"]}</div>', unsafe_allow_html=True)

    strengths_col, priority_col = st.columns(2)
    with strengths_col:
        st.markdown("#### 🌟 Top strengths")
        for strength in summary["top_strengths"]:
            st.markdown(f"- {strength}")
    with priority_col:
        st.markdown("#### 🎯 Priority improvements")
        for improvement in summary["priority_improvements"]:
            st.markdown(f"- {improvement}")

    st.markdown("### Your preparation plan")
    for index, action in enumerate(summary["preparation_plan"], 1):
        st.markdown(f"**{index}.** {action}")

    st.success(summary["closing_message"])

    st.markdown("### Question scorecard")
    for index, item in enumerate(records, 1):
        with st.expander(f"Question {index} · Score {item['evaluation']['score']}/10 · {item['focus']}"):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown("**Your answer:**")
            st.write(item["answer"])
            st.markdown(f"**Coach feedback:** {item['evaluation']['short_verdict']}")
            st.markdown("**Sample answer:**")
            st.write(item["evaluation"]["sample_answer"])

    report = build_report_text()
    download_col, restart_col = st.columns([3, 1])
    with download_col:
        st.download_button(
            "⬇ Download full Q&A transcript & coaching report",
            data=report,
            file_name="interviewiq_coaching_report.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )
    with restart_col:
        if st.button("Start again", use_container_width=True):
            reset_interview()
            st.rerun()

st.markdown(
    '<div style="text-align:center;color:#9299aa;font-size:.78rem;padding:2rem 0 1rem">INTERVIEWIQ · PRACTICE WITH PURPOSE</div>',
    unsafe_allow_html=True,
)
