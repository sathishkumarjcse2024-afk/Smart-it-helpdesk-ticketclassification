import io, re, sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "sample_tickets.csv"
DB_PATH = APP_DIR / "helpdesk.db"
CATEGORIES = ["Account Access", "Hardware", "Software", "Network", "Application Support"]
ROUTES = {
    "Account Access": "Identity & Access Queue",
    "Hardware": "Desktop Support Queue",
    "Software": "Software Support Queue",
    "Network": "Network Operations Queue",
    "Application Support": "Business Applications Queue",
}
PRIORITY_WORDS = {
    "High": ["urgent", "critical", "outage", "down", "cannot work", "security", "all users"],
    "Medium": ["slow", "intermittent", "error", "failed", "not working"],
}

def clean_text(value):
    text = str(value or "")
    # Mask common secret-like values before processing or display.
    text = re.sub(r'(?i)\b(password|passwd|token|api[_ -]?key)\s*[:=]\s*\S+', r'\1=[REDACTED]', text)
    text = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[EMAIL]', text, flags=re.I)
    text = re.sub(r'\b\d{8,}\b', '[LONG_NUMBER]', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    replacements = {"pls": "please", "pw": "password", "wifi": "wireless network", "pc": "computer"}
    for old, new in replacements.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)
    return text

def infer_priority(text):
    t = clean_text(text)
    if any(word in t for word in PRIORITY_WORDS["High"]):
        return "High"
    if any(word in t for word in PRIORITY_WORDS["Medium"]):
        return "Medium"
    return "Low"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    df["clean_issue"] = df["issue"].fillna("").map(clean_text)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(df["clean_issue"], df["category"])
    return model, df

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            ticket_id TEXT,
            issue TEXT,
            predicted_category TEXT,
            confidence REAL,
            suggested_queue TEXT,
            confirmed_category TEXT,
            reason TEXT,
            model_version TEXT
        )""")

def save_feedback(ticket_id, issue, pred, confidence, queue, confirmed, reason):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO feedback (created_at,ticket_id,issue,predicted_category,confidence,suggested_queue,confirmed_category,reason,model_version) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), ticket_id, clean_text(issue), pred, confidence, queue, confirmed, reason, "tfidf-logreg-v1")
        )

def get_feedback():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)

st.set_page_config(page_title="Smart IT Helpdesk", page_icon="🎫", layout="wide")
st.title("🎫 Smart IT Helpdesk Ticket Classification")
st.caption("AI-assisted classification, priority estimation, queue recommendations, and staff review. Predictions are suggestions—not final decisions.")
model, training_df = train_model()
init_db()

tab1, tab2, tab3, tab4 = st.tabs(["Classify Ticket", "Batch Upload", "Review & Feedback", "Dashboard & Evaluation"])

with tab1:
    st.subheader("Enter a support ticket")
    col1, col2 = st.columns(2)
    with col1:
        ticket_id = st.text_input("Ticket ID (optional)", value="TKT-001")
        issue = st.text_area("Issue description", placeholder="Example: I cannot log in to my college email account.", height=150)
        service = st.selectbox("Service", ["Email", "Laptop/Desktop", "Wi-Fi/Network", "Operating System", "Business Application", "Other"])
    with col2:
        submitted_time = st.date_input("Ticket date")
        context = st.text_input("Additional context (optional)")
        st.info("Do not enter passwords, API keys, or other secrets. Common secret patterns are masked automatically.")
    if st.button("Classify ticket", type="primary"):
        if len(clean_text(issue)) < 5:
            st.warning("Please enter a clearer issue description (at least a few words).")
        else:
            safe_issue = clean_text(issue)
            probabilities = model.predict_proba([safe_issue])[0]
            classes = model.named_steps["classifier"].classes_
            best_idx = probabilities.argmax()
            category = classes[best_idx]
            confidence = float(probabilities[best_idx])
            priority = infer_priority(issue + " " + context)
            queue = ROUTES.get(category, "General IT Support Queue")
            st.session_state["last_result"] = {
                "ticket_id": ticket_id, "issue": issue, "category": category,
                "confidence": confidence, "priority": priority, "queue": queue,
                "service": service, "date": str(submitted_time)
            }
            if confidence < 0.45:
                st.warning("Low confidence: send this ticket for staff review before routing.")
            a, b, c = st.columns(3)
            a.metric("Suggested category", category)
            b.metric("Estimated priority", priority)
            c.metric("Model confidence", f"{confidence:.1%}")
            st.write("**Suggested queue:**", queue)
            st.write("**Evidence:** the model matched words and phrases in the issue against labelled example tickets using TF-IDF features.")
            st.write("**Service/context captured:**", service, "|", context or "No extra context")
            st.caption("Priority is based on simple keyword rules and must be confirmed by staff. The system does not close tickets automatically.")
    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Staff review")
        confirmed = st.selectbox("Confirm or correct category", CATEGORIES, index=CATEGORIES.index(result["category"]) if result["category"] in CATEGORIES else 0)
        reason = st.text_input("Reason for correction / review note")
        if st.button("Save review"):
            save_feedback(result["ticket_id"], result["issue"], result["category"], result["confidence"], result["queue"], confirmed, reason)
            st.success("Review saved to local SQLite history.")

with tab2:
    st.subheader("Classify multiple tickets from CSV")
    st.write("Required column: `issue`. Optional columns: `ticket_id`, `service`, `date`.")
    upload = st.file_uploader("Upload ticket CSV", type=["csv"])
    if upload:
        try:
            batch = pd.read_csv(upload)
            if "issue" not in batch.columns:
                st.error("CSV must contain an 'issue' column.")
            elif len(batch) > 5000:
                st.error("For safety and performance, upload at most 5,000 rows at a time.")
            else:
                batch["safe_issue"] = batch["issue"].fillna("").map(clean_text)
                valid = batch["safe_issue"].str.len() >= 5
                if not valid.all():
                    st.warning(f"{int((~valid).sum())} short/empty ticket(s) need manual review.")
                probs = model.predict_proba(batch.loc[valid, "safe_issue"])
                classes = model.named_steps["classifier"].classes_
                rows = []
                for idx, p in zip(batch.index[valid], probs):
                    best = int(p.argmax())
                    cat = classes[best]
                    rows.append({
                        "ticket_id": batch.loc[idx].get("ticket_id", f"ROW-{idx+1}"),
                        "issue": clean_text(batch.loc[idx, "issue"]),
                        "category": cat,
                        "confidence": round(float(p[best]), 4),
                        "priority": infer_priority(batch.loc[idx, "issue"]),
                        "suggested_queue": ROUTES.get(cat, "General IT Support Queue"),
                        "review_required": float(p[best]) < 0.45
                    })
                result_df = pd.DataFrame(rows)
                st.dataframe(result_df, use_container_width=True)
                st.download_button("Download classified CSV", result_df.to_csv(index=False).encode("utf-8"), "classified_tickets.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process this file. Check the CSV format and try again. Details: {exc}")

with tab3:
    st.subheader("Review history")
    feedback = get_feedback()
    if feedback.empty:
        st.info("No staff feedback has been saved yet. Classify a ticket and save a review first.")
    else:
        st.dataframe(feedback, use_container_width=True)
        st.download_button("Export review history", feedback.to_csv(index=False).encode("utf-8"), "review_history.csv", "text/csv")

with tab4:
    st.subheader("Training-data evaluation")
    st.write(f"Training examples: **{len(training_df)}** | Categories: **{training_df['category'].nunique()}** | Model version: **tfidf-logreg-v1**")
    st.warning("Metrics below use a held-out split of the included synthetic demonstration data. They are not a guarantee of performance on real tickets.")
    X_train, X_test, y_train, y_test = train_test_split(
        training_df["issue"].map(clean_text), training_df["category"],
        test_size=0.25, random_state=42, stratify=training_df["category"]
    )
    eval_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    st.metric("Hold-out accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.write("**Per-class precision, recall and F1**")
    st.dataframe(report_df, use_container_width=True)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(training_df["category"].unique()))
    cm_df = pd.DataFrame(cm, index=sorted(training_df["category"].unique()), columns=sorted(training_df["category"].unique()))
    st.write("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)
    st.write("**Queue volume in training examples**")
    st.bar_chart(training_df["category"].value_counts())
    st.caption("Use approved data, check class imbalance, validate routes with support staff, and version the model before production use.")
