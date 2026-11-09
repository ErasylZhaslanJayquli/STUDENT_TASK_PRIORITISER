"""Student Task Prioritizer — Streamlit app.

Run:  pip install -r requirements.txt && streamlit run app.py
"""
import calendar
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import task_manager as tm

st.set_page_config(page_title="Student Task Prioritizer", page_icon="🎓", layout="wide")

TODAY = date.today()
PRIORITY_COLORS = {"High": "#D64562", "Medium": "#E89A2B", "Low": "#3FA37A"}
STATUSES = ["To Do", "In Progress", "Completed"]
PAGES = [
    "🏠  Dashboard",
    "✅  My Tasks",
    "📅  Calendar",
    "🗓️  Study Schedule",
    "✨  AI Assistant",
    "📊  Analytics",
    "⚙️  Settings",
]

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def seed_tasks():
    d = lambda n: TODAY + timedelta(days=n)
    rows = [  # (title, subject, due, hours, difficulty, importance, status)
        ("Calculus homework: integrals", "Calculus", d(1), 3.0, 4, 4, "In Progress"),
        ("Intro to TS platform", "Programming", d(0), 1.5, 2, 2, "To Do"),
        ("Essay: AI in education", "English", d(5), 4.0, 3, 3, "To Do"),
        ("Physics lab report", "Physics", d(3), 3.5, 4, 4, "To Do"),
        ("Quest 1 — Python basics", "Programming", d(-1), 2.0, 2, 2, "Completed"),
        ("Practical work 2", "Foundations of AI", d(2), 2.5, 3, 3, "To Do"),
        ("Linear algebra quiz prep", "Calculus", d(7), 2.0, 3, 3, "To Do"),
        ("Read chapter 4 (ICT)", "ICT", d(9), 1.0, 1, 1, "To Do"),
        ("Group project slides", "Foundations of AI", d(6), 5.0, 5, 5, "In Progress"),
        ("Vocabulary test", "English", d(12), 1.0, 1, 1, "To Do"),
        ("Assignment 1", "Foundations of AI", d(-3), 2.0, 2, 2, "Completed"),
        ("Practical work 1", "Foundations of AI", d(-5), 2.0, 2, 2, "Completed"),
        ("Homework 3", "ICT", d(-2), 1.5, 1, 1, "Completed"),
    ]
    tasks = []
    for i, (t, s_, due, h, diff, imp, st_) in enumerate(rows, 1):
        task = tm.new_task(i, t, s_, due, h, diff, imp, st_)
        task["completed_on"] = due if st_ == "Completed" else None
        tasks.append(task)
    return tasks


SCHEDULE = [  # (weekday 0=Mon, start, end, subject, room)
    (0, "09:00", "10:30", "Calculus", "Room 301"),
    (0, "13:00", "14:30", "Programming", "Lab 12"),
    (1, "10:00", "11:30", "Foundations of AI", "Room 205"),
    (1, "15:00", "16:30", "English", "Room 110"),
    (2, "09:00", "10:30", "Physics", "Room 402"),
    (2, "12:00", "13:30", "ICT", "Lab 7"),
    (3, "11:00", "12:30", "Calculus", "Room 301"),
    (3, "14:00", "15:30", "Foundations of AI", "Lab 3"),
    (4, "09:00", "10:30", "Programming", "Lab 12"),
    (4, "11:00", "12:30", "Physics", "Room 402"),
]

if "tasks" not in st.session_state:
    st.session_state.tasks = seed_tasks()
    st.session_state.next_id = 100
    st.session_state.name = "Alex"
    st.session_state.theme = "Cream"
    st.session_state.notif = {"deadline_24h": True, "daily_digest": True, "email": False}
    st.session_state.sel_date = TODAY
    st.session_state.cal_year, st.session_state.cal_month = TODAY.year, TODAY.month
    st.session_state.week_start = TODAY - timedelta(days=TODAY.weekday())
    st.session_state.chat = [
        {"role": "assistant",
         "content": "Hi! I can tell you what to study today, explain priorities, "
                    "organize deadlines and build a plan for tomorrow."}
    ]


# --------------------------------------------------------------------------- #
# Priority model: the trained Decision Tree from ml_model.py (via task_manager.predict_task)
# --------------------------------------------------------------------------- #
def predict_task(t, today=TODAY):
    """(label, score, confidence) predicted by the Decision Tree."""
    return tm.predict_task(t, today)


def priority_score(t, today=TODAY):
    return predict_task(t, today)[1]


def priority_label(t, today=TODAY):
    return predict_task(t, today)[0]


def reasons(t):
    days = (t["due"] - TODAY).days
    out = []
    if days < 0:
        out.append(f"it is **overdue by {-days} day(s)**")
    elif days == 0:
        out.append("it is **due today**")
    elif days <= 2:
        out.append(f"the deadline is **very close ({days} day(s))**")
    elif days <= 7:
        out.append(f"the deadline is in {days} days")
    else:
        out.append(f"the deadline is still far away ({days} days)")
    if t["difficulty"] >= 4:
        out.append("the task is **hard**, so it needs more focus")
    elif t["difficulty"] <= 2:
        out.append("the task is easy, which lowers its weight")
    if t["est"] >= 3:
        out.append(f"it needs about **{t['est']:g} h**")
    if t["importance"] >= 4:
        out.append("it is **important** for your grade")
    return out


def tasks_df(include_completed=True):
    rows = []
    for t in st.session_state.tasks:
        if not include_completed and t["status"] == "Completed":
            continue
        pr, sc, conf = predict_task(t)
        rows.append({**t, "score": sc, "priority": pr, "confidence": conf})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["id", "title", "subject", "due", "est", "difficulty",
                                   "importance", "status", "completed_on", "score", "priority", "confidence"])
    return df


def get_task(tid):
    return next(t for t in st.session_state.tasks if t["id"] == tid)


def set_status(tid, status):
    t = get_task(tid)
    t["status"] = status
    t["completed_on"] = TODAY if status == "Completed" else None


def diff_label(n):
    return "Easy" if n <= 2 else "Medium" if n == 3 else "Hard"


def due_text(d):
    n = (d - TODAY).days
    if n < 0:
        return f"Overdue {-n}d"
    return "Today" if n == 0 else "Tomorrow" if n == 1 else f"{d:%a, %b %d}"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    dark = st.session_state.theme == "Dark purple"
    bg, card, text, muted, border = (
        ("#1A1230", "#251A45", "#F6EEDC", "#B9AED6", "#3A2C66") if dark
        else ("#FBF6EC", "#FFFFFF", "#1E1238", "#6E6585", "#EADFC8")
    )
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: {bg}; color: {text}; }}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp span,.stApp li {{ color: {text}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
section[data-testid="stSidebar"] {{ background: #1E1238; }}
section[data-testid="stSidebar"] * {{ color: #F6EEDC !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    padding: 12px 14px; border-radius: 14px; margin-bottom: 4px; cursor: pointer; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: #2E1F57; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{ background: #7C4DDB; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display:none; }}
.brand {{ font-weight:800; font-size:1.5rem; letter-spacing:.5px; }}
.brand small {{ color:#CDB8FF; font-size:.7rem; margin-left:6px; letter-spacing:1.5px; }}
.card {{ background:{card}; border:1px solid {border}; border-radius:20px; padding:18px 20px; margin-bottom:12px; }}
.stat {{ font-size:2rem; font-weight:800; color:#7C4DDB; line-height:1.1; }}
.muted {{ color:{muted} !important; font-size:.85rem; }}
.trow {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.ttitle {{ font-weight:700; }}
.badge {{ padding:3px 12px; border-radius:999px; font-size:.75rem; font-weight:700; color:#fff; white-space:nowrap; }}
.chip {{ display:block; font-size:.68rem; font-weight:700; color:#fff; padding:2px 6px; border-radius:6px;
    margin-top:3px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.hero {{ background:linear-gradient(120deg,#5B2EBF,#8E63E8); border-radius:24px; padding:26px 30px; margin-bottom:18px; }}
.hero h1,.hero p {{ color:#FBF6EC !important; margin:0; }}
.stButton > button, .stFormSubmitButton > button {{
    border-radius:12px; border:1px solid {border}; font-weight:700; background:{card}; color:{text}; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ border-color:#7C4DDB; color:#7C4DDB; }}
.stFormSubmitButton > button {{ background:#5B2EBF; color:#fff; border:none; }}
.stFormSubmitButton > button:hover {{ background:#7C4DDB; color:#fff; }}
.stProgress > div > div > div > div {{ background:#7C4DDB; }}
div[data-testid="stForm"] {{ background:{card}; border:1px solid {border}; border-radius:20px; }}
</style>
""",
        unsafe_allow_html=True,
    )


def task_card(t, key_prefix, show_done=True):
    pr, sc, conf = predict_task(t)
    c1, c2 = st.columns([8, 1.3], vertical_alignment="center")
    with c1:
        st.markdown(
            f"""<div class="card" style="margin-bottom:6px"><div class="trow">
            <div><div class="ttitle">{t['title']}</div>
            <div class="muted">{t['subject']} · {due_text(t['due'])} · {t['est']:g} h · {diff_label(t['difficulty'])}</div></div>
            <span class="badge" title="Decision Tree confidence: {conf:.0%}" style="background:{PRIORITY_COLORS[pr]}">{pr}</span></div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        if show_done and t["status"] != "Completed":
            if st.button("Done ✓", key=f"{key_prefix}_{t['id']}", width="stretch"):
                set_status(t["id"], "Completed")
                st.rerun()


def chip_html(t):
    pr = priority_label(t)
    op = "0.45" if t["status"] == "Completed" else "1"
    return f'<span class="chip" style="background:{PRIORITY_COLORS[pr]};opacity:{op}" title="{t["title"]}">{t["title"]}</span>'


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard():
    h = datetime.now().hour
    greet = "Good morning" if h < 12 else "Good afternoon" if h < 18 else "Good evening"
    df = tasks_df()
    active = df[df.status != "Completed"]
    done_n, total = int((df.status == "Completed").sum()), len(df)
    today_df = active[active.due <= TODAY].sort_values("score", ascending=False)

    st.markdown(
        f"""<div class="hero"><h1>{greet}, {st.session_state.name}! 👋</h1>
        <p>{TODAY:%A, %B %d} · You have {len(today_df)} task(s) to focus on today.</p></div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Due today", len(today_df)),
        ("High priority", int((active.priority == "High").sum())),
        ("In progress", int((active.status == "In Progress").sum())),
        ("Hours left", f"{active.est.sum():g} h"),
    ]
    for col, (lbl, val) in zip((c1, c2, c3, c4), stats):
        col.markdown(f'<div class="card"><div class="muted">{lbl}</div><div class="stat">{val}</div></div>',
                     unsafe_allow_html=True)

    pct = done_n / total if total else 0
    st.markdown(f"**Overall progress** — {done_n}/{total} tasks completed ({pct:.0%})")
    st.progress(pct)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Today's Tasks")
        if today_df.empty:
            st.success("Nothing due today. Nice! 🎉")
        for _, r in today_df.iterrows():
            task_card(get_task(int(r.id)), "td")

        st.subheader("High Priority Tasks")
        high = active[active.priority == "High"].sort_values("score", ascending=False).head(5)
        if high.empty:
            st.info("No high priority tasks right now.")
        for _, r in high.iterrows():
            task_card(get_task(int(r.id)), "hp")

    with right:
        st.subheader("Upcoming Deadlines")
        up = active[(active.due > TODAY) & (active.due <= TODAY + timedelta(days=7))].sort_values("due")
        if up.empty:
            st.caption("No deadlines in the next 7 days.")
        for _, r in up.iterrows():
            c = PRIORITY_COLORS[r.priority]
            st.markdown(
                f"""<div class="card" style="border-left:6px solid {c};padding:12px 16px;margin-bottom:8px">
                <div class="ttitle">{r.title}</div>
                <div class="muted">{r.subject} · {due_text(r.due)}</div></div>""",
                unsafe_allow_html=True,
            )

        st.subheader("Quick Add Task")
        with st.form("quick_add", clear_on_submit=True):
            title = st.text_input("Title")
            subject = st.text_input("Subject")
            a, b = st.columns(2)
            due = a.date_input("Due date", TODAY + timedelta(days=3))
            est = b.number_input("Estimated time (h)", 0.5, 24.0, 2.0, 0.5)
            diff = st.slider("Difficulty", 1, 5, 3)
            imp = st.slider("Importance", 1, 5, 3)
            if st.form_submit_button("Add task", width="stretch"):
                if title.strip():
                    st.session_state.tasks.append(tm.new_task(
                        st.session_state.next_id, title, subject, due, est, diff, imp))
                    st.session_state.next_id += 1
                    st.rerun()
                else:
                    st.warning("Please enter a title.")


def page_tasks():
    st.title("My Tasks")
    df = tasks_df()
    f1, f2, f3, f4 = st.columns(4)
    subj = f1.multiselect("Subject", sorted(df.subject.unique()))
    stat = f2.multiselect("Status", STATUSES, default=["To Do", "In Progress"])
    prio = f3.multiselect("ML Priority", ["High", "Medium", "Low"])
    sort = f4.selectbox("Sort by", ["Priority (high → low)", "Due date", "Subject"])

    if subj: df = df[df.subject.isin(subj)]
    if stat: df = df[df.status.isin(stat)]
    if prio: df = df[df.priority.isin(prio)]
    df = tm.sort_df(df, "priority" if sort.startswith("Priority") else "due" if sort == "Due date" else "subject")

    view = pd.DataFrame({
        "id": df.id, "Task": df.title, "Subject": df.subject, "Due Date": df.due,
        "Estimated Time (h)": df.est, "Difficulty": df.difficulty.map(diff_label), "Importance": df.importance,
        "ML Priority": df.priority.map(lambda p: {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}[p]),
        "Status": df.status,
    }).reset_index(drop=True)

    st.caption("Change a task's status right in the table. Priority is predicted by the Decision Tree from days left, difficulty, hours and importance.")
    edited = st.data_editor(
        view, hide_index=True, width="stretch", key="task_editor",
        disabled=[c for c in view.columns if c != "Status"],
        column_config={
            "id": None,
            "Due Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
            "Status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
        },
    )
    changed = False
    for i, row in edited.iterrows():
        if row["Status"] != view.loc[i, "Status"]:
            set_status(int(row["id"]), row["Status"])
            changed = True
    if changed:
        st.rerun()

    with st.expander("➕ Add a new task"):
        with st.form("add_task", clear_on_submit=True):
            a, b = st.columns(2)
            title = a.text_input("Title")
            subject = b.text_input("Subject")
            c, d, e, f = st.columns(4)
            due = c.date_input("Due date", TODAY + timedelta(days=3))
            est = d.number_input("Estimated time (h)", 0.5, 24.0, 2.0, 0.5)
            diff = e.slider("Difficulty", 1, 5, 3)
            imp = f.slider("Importance", 1, 5, 3)
            if st.form_submit_button("Add task") and title.strip():
                st.session_state.tasks.append(tm.new_task(
                    st.session_state.next_id, title, subject, due, est, diff, imp))
                st.session_state.next_id += 1
                st.rerun()

    with st.expander("🗑️ Delete a task"):
        opts = {f"{t['title']} ({t['subject']})": t["id"] for t in st.session_state.tasks}
        pick = st.selectbox("Task", list(opts))
        if st.button("Delete") and pick is not None:
            st.session_state.tasks = tm.delete_task(st.session_state.tasks, opts[pick])
            st.rerun()

    with st.expander("💾 Save / load tasks"):
        st.caption("Tasks live in this browser session only. Download a CSV to keep them and upload it later to restore.")
        st.download_button("Download tasks.csv", tm.tasks_to_csv(st.session_state.tasks),
                           file_name="tasks.csv", mime="text/csv")
        up = st.file_uploader("Load tasks from CSV", type="csv")
        if up is not None and st.button("Load file"):
            try:
                st.session_state.tasks = tm.tasks_from_csv(up)
                st.session_state.next_id = tm.next_id(st.session_state.tasks)
                st.rerun()
            except ValueError as err:
                st.error(str(err))


def tasks_on(d):
    return [t for t in st.session_state.tasks if t["due"] == d]


def page_calendar():
    st.title("Calendar")
    mode = st.radio("View", ["Month View", "Week View"], horizontal=True, label_visibility="collapsed")
    st.markdown(
        "".join(f'<span class="badge" style="background:{c};margin-right:8px">{p}</span>'
                for p, c in PRIORITY_COLORS.items()), unsafe_allow_html=True)
    st.write("")

    if mode == "Month View":
        y, m = st.session_state.cal_year, st.session_state.cal_month
        n1, n2, n3 = st.columns([1, 4, 1])
        if n1.button("◀ Prev", width="stretch"):
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
            st.session_state.cal_year, st.session_state.cal_month = y, m
            st.rerun()
        n2.markdown(f"<h3 style='text-align:center;margin:0'>{calendar.month_name[m]} {y}</h3>", unsafe_allow_html=True)
        if n3.button("Next ▶", width="stretch"):
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
            st.session_state.cal_year, st.session_state.cal_month = y, m
            st.rerun()

        heads = st.columns(7)
        for col, name in zip(heads, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            col.markdown(f"<div class='muted' style='text-align:center;font-weight:700'>{name}</div>", unsafe_allow_html=True)
        for week in calendar.Calendar(0).monthdatescalendar(y, m):
            cols = st.columns(7)
            for col, d in zip(cols, week):
                with col:
                    sel = d == st.session_state.sel_date
                    label = f"{'●' if d == TODAY else ''}{d.day}"
                    if st.button(label, key=f"day_{d}", width="stretch",
                                 type="primary" if sel else "secondary", disabled=(d.month != m)):
                        st.session_state.sel_date = d
                        st.rerun()
                    ts = tasks_on(d) if d.month == m else []
                    html = "".join(chip_html(t) for t in ts[:2])
                    if len(ts) > 2:
                        html += f"<span class='muted'>+{len(ts) - 2} more</span>"
                    st.markdown(f"<div style='min-height:58px'>{html}</div>", unsafe_allow_html=True)
    else:
        ws = st.session_state.week_start
        n1, n2, n3 = st.columns([1, 4, 1])
        if n1.button("◀ Prev", width="stretch"):
            st.session_state.week_start = ws - timedelta(days=7); st.rerun()
        n2.markdown(f"<h3 style='text-align:center;margin:0'>{ws:%b %d} – {(ws + timedelta(days=6)):%b %d, %Y}</h3>", unsafe_allow_html=True)
        if n3.button("Next ▶", width="stretch"):
            st.session_state.week_start = ws + timedelta(days=7); st.rerun()
        cols = st.columns(7)
        for i, col in enumerate(cols):
            d = ws + timedelta(days=i)
            with col:
                if st.button(f"{d:%a %d}", key=f"wday_{d}", width="stretch",
                             type="primary" if d == st.session_state.sel_date else "secondary"):
                    st.session_state.sel_date = d; st.rerun()
                ts = sorted(tasks_on(d), key=lambda t: -priority_score(t))
                hrs = sum(t["est"] for t in ts if t["status"] != "Completed")
                st.markdown(f"<div class='muted'>{hrs:g} h load</div>", unsafe_allow_html=True)
                st.markdown("".join(chip_html(t) for t in ts) or "<span class='muted'>—</span>", unsafe_allow_html=True)

    st.divider()
    d = st.session_state.sel_date
    st.subheader(f"Tasks on {d:%A, %B %d}")
    ts = sorted(tasks_on(d), key=lambda t: -priority_score(t))
    if not ts:
        st.caption("No deadlines on this day.")
    for t in ts:
        task_card(t, "cal")


def free_slots(weekday):
    def mins(s): h, m = map(int, s.split(":")); return h * 60 + m
    busy = sorted((mins(s), mins(e)) for wd, s, e, *_ in SCHEDULE if wd == weekday)
    cur, out = 8 * 60, []
    for s, e in busy + [(20 * 60, 20 * 60)]:
        if s - cur >= 60:
            out.append((cur, s))
        cur = max(cur, e)
    fmt = lambda x: f"{x // 60:02d}:{x % 60:02d}"
    return [f"{fmt(a)}–{fmt(b)} ({(b - a) / 60:g} h)" for a, b in out]


def page_schedule():
    st.title("Study Schedule")
    st.caption("University classes and free windows (08:00–20:00) you can use for studying.")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    cols = st.columns(7)
    for i, (col, name) in enumerate(zip(cols, days)):
        with col:
            today_mark = " •" if i == TODAY.weekday() else ""
            st.markdown(f"**{name[:3]}{today_mark}**")
            classes = [c for c in SCHEDULE if c[0] == i]
            for _, s, e, subj, room in classes:
                st.markdown(
                    f"""<div class="card" style="padding:10px 12px;margin-bottom:6px;border-left:5px solid #7C4DDB">
                    <div class="muted">{s}–{e}</div><div class="ttitle" style="font-size:.85rem">{subj}</div>
                    <div class="muted">{room}</div></div>""", unsafe_allow_html=True)
            if not classes:
                st.caption("No classes")
            for slot in free_slots(i):
                st.markdown(
                    f"<div style='background:#E8DDFB;color:#3B1E80;border-radius:10px;padding:6px 10px;"
                    f"font-size:.75rem;font-weight:700;margin-bottom:6px'>🟣 Free {slot}</div>",
                    unsafe_allow_html=True)


def plan_for(day):
    df = tasks_df(include_completed=False)
    if df.empty:
        return "You have no active tasks — enjoy the free time! 🎉"
    free = free_slots(day.weekday())
    top = df.sort_values("score", ascending=False).head(3)
    lines = [f"**Plan for {day:%A, %B %d}:**"]
    for i, (_, r) in enumerate(top.iterrows()):
        slot = free[i] if i < len(free) else "any free time"
        lines.append(f"{i + 1}. **{r.title}** ({r.subject}, {r.priority}) — {r.est:g} h · {slot}")
    lines.append("\nTip: take a 10-minute break every 50 minutes.")
    return "\n".join(lines)


def ai_reply(msg):
    m = msg.lower()
    df = tasks_df(include_completed=False)
    if "why" in m or "почему" in m:
        for t in st.session_state.tasks:
            words = [w for w in (t["title"] + " " + t["subject"]).lower().replace(":", " ").split() if len(w) > 3]
            if any(w in m for w in words):
                pr, sc, conf = predict_task(t)
                return (f"**{t['title']}** is **{pr}** priority (Decision Tree, confidence {conf:.0%}) because "
                        + ", ".join(reasons(t)) + ".")
        return "Tell me which task you mean, e.g. “Why is my Calculus homework High Priority?”"
    if "tomorrow" in m or "завтра" in m:
        return plan_for(TODAY + timedelta(days=1))
    if "organize" in m or "deadline" in m or "дедлайн" in m:
        if df.empty:
            return "No active deadlines. 🎉"
        lines = ["**Your deadlines, most urgent first:**"]
        for _, r in df.sort_values(["due", "score"], ascending=[True, False]).iterrows():
            lines.append(f"- {due_text(r.due)} — **{r.title}** ({r.subject}) · {r.priority}")
        lines.append("\nStart with the High items; batch the Low ones into a free evening.")
        return "\n".join(lines)
    if "today" in m or "сегодня" in m or "study" in m or "plan" in m:
        if "plan" in m and "today" not in m:
            return plan_for(TODAY)
        top = df.sort_values("score", ascending=False).head(3)
        if top.empty:
            return "You're all caught up!"
        lines = ["**Focus on these today:**"]
        for _, r in top.iterrows():
            lines.append(f"- **{r.title}** ({r.subject}) — {r.priority}, {due_text(r.due)}, ~{r.est:g} h")
        return "\n".join(lines)
    return ("I can help with:\n- “What should I study today?”\n- “Help me organize my deadlines.”\n"
            "- “Why is my Calculus homework High Priority?”\n- “Create a study plan for tomorrow.”")


def page_ai():
    st.title("AI Assistant")
    cols = st.columns(4)
    prompts = ["What should I study today?", "Help me organize my deadlines.",
               "Why is my Calculus homework High Priority?", "Create a study plan for tomorrow."]
    clicked = None
    for c, p in zip(cols, prompts):
        if c.button(p, width="stretch"):
            clicked = p
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    msg = st.chat_input("Ask about your tasks…") or clicked
    if msg:
        st.session_state.chat.append({"role": "user", "content": msg})
        st.session_state.chat.append({"role": "assistant", "content": ai_reply(msg)})
        st.rerun()


def page_analytics():
    st.title("Analytics")
    df = tasks_df()
    done = df[df.status == "Completed"]
    rate = len(done) / len(df) if len(df) else 0
    c1, c2, c3 = st.columns(3)
    for c, (l, v) in zip((c1, c2, c3), [("Completed tasks", len(done)),
                                         ("Completion rate", f"{rate:.0%}"),
                                         ("Hours completed", f"{done.est.sum():g} h")]):
        c.markdown(f'<div class="card"><div class="muted">{l}</div><div class="stat">{v}</div></div>', unsafe_allow_html=True)

    purple = ["#4B1FA8", "#7C4DDB", "#B79CF0", "#D9C9F7", "#E8B04A"]
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10))

    a, b = st.columns(2)
    with a:
        pc = df.priority.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        pc.columns = ["Priority", "Tasks"]
        fig = px.bar(pc, x="Priority", y="Tasks", color="Priority", color_discrete_map=PRIORITY_COLORS,
                     title="Tasks by Priority")
        fig.update_layout(showlegend=False, **layout)
        st.plotly_chart(fig, width="stretch")
    with b:
        w = df.groupby("subject").est.sum().reset_index()
        fig = px.pie(w, names="subject", values="est", hole=0.55, title="Workload by Subject (hours)",
                     color_discrete_sequence=purple)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")

    days = [TODAY - timedelta(days=i) for i in range(6, -1, -1)]
    wk = pd.DataFrame({"Day": [f"{d:%a %d}" for d in days],
                       "Completed": [int((done.completed_on == d).sum()) for d in days]})
    fig = px.area(wk, x="Day", y="Completed", markers=True, title="Weekly Progress (completed per day)",
                  color_discrete_sequence=["#7C4DDB"])
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Completion rate by status**")
    st.progress(rate)
    st.caption(f"{len(done)} of {len(df)} tasks completed · "
               f"{int((df.status == 'In Progress').sum())} in progress · {int((df.status == 'To Do').sum())} to do")


def page_settings():
    st.title("Settings")
    t1, t2, t3, t4 = st.tabs(["Profile", "Theme", "Notifications", "Model"])
    with t1:
        name = st.text_input("Display name", st.session_state.name)
        st.text_input("University", "Astana IT University")
        st.text_input("Major", "Artificial Intelligence")
        if st.button("Save profile"):
            st.session_state.name = name.strip() or "Alex"
            st.success("Profile saved.")
    with t2:
        theme = st.radio("Theme", ["Cream", "Dark purple"], index=["Cream", "Dark purple"].index(st.session_state.theme))
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
    with t3:
        n = st.session_state.notif
        n["deadline_24h"] = st.toggle("Remind me 24 hours before a deadline", n["deadline_24h"])
        n["daily_digest"] = st.toggle("Daily digest of today's tasks", n["daily_digest"])
        n["email"] = st.toggle("Also send by email", n["email"])
    with t4:
        st.markdown("**Priority model:** Decision Tree (scikit-learn). Inputs: days left, difficulty, estimated hours, importance.")
        mpath = Path(__file__).resolve().parent / "ml_outputs" / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Test accuracy", f"{m['test_accuracy']:.0%}")
            b.metric("Majority baseline", f"{m['baseline_test_accuracy']:.0%}")
            c.metric("Macro F1", f"{m['test_macro_f1']:.2f}")
            st.caption(f"Trained on {m['n_train']} synthetic tasks, tested on {m['n_test']}. "
                       "The test set is small, so treat the real accuracy as roughly 60–70%.")
        else:
            st.caption("Run `python ml_model.py` to create ml_outputs/metrics.json.")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
inject_css()
with st.sidebar:
    st.markdown('<div class="brand">TASKR<small>PRIORITIZER</small></div>', unsafe_allow_html=True)
    st.markdown("<div style='opacity:.6;font-size:.75rem;letter-spacing:1.5px;margin:18px 0 6px'>STUDY</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("<hr style='opacity:.2'>", unsafe_allow_html=True)
    st.markdown(f"**{st.session_state.name}**  \n<span style='opacity:.6;font-size:.8rem'>Student</span>", unsafe_allow_html=True)

{
    PAGES[0]: page_dashboard, PAGES[1]: page_tasks, PAGES[2]: page_calendar,
    PAGES[3]: page_schedule, PAGES[4]: page_ai, PAGES[5]: page_analytics, PAGES[6]: page_settings,
}[page]()
