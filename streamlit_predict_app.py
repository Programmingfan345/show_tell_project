import os
import smtplib
import mysql.connector
import joblib
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# NLTK
# =========================
nltk.download("punkt")
try:
    nltk.download("punkt_tab")  # harmless if missing
except:
    pass

# =========================
# Secrets helper
# =========================
def get_secret(name: str):
    # DO NOT read env here; step A = secrets-only for admin unlock
    try:
        return st.secrets[name]
    except Exception:
        return None

EMAIL_ADDRESS = get_secret("EMAIL_ADDRESS")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD")
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    st.error("Email creds missing. Set EMAIL_ADDRESS and EMAIL_PASSWORD (Gmail App Password) in .streamlit/secrets.toml.")
    st.stop()

# =========================
# Model + Vectorizer
# =========================
def load_model_and_vectorizer():
    try:
        model = joblib.load("LogisticRegression_All_shots_data_model.pkl")
        vectorizer = joblib.load("LogisticRegression_All_shots_data_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()

def predict_sentences(sentences, model, vectorizer):
    tokens = [" ".join(nltk.word_tokenize(s.lower())) for s in sentences]
    return model.predict(vectorizer.transform(tokens))

# =========================
# DB helpers
# =========================
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=get_secret("DB_HOST"),
            port=int(get_secret("DB_PORT")),
            database=get_secret("DB_NAME"),
            user=get_secret("DB_USER"),
            password=get_secret("DB_PASSWORD"),
            autocommit=False,
        )
    except mysql.connector.Error as err:
        st.error(f"Database Connection Error: {err}")
        return None

def get_or_create_student(full_name: str, email: str):
    """Return students.student_id, creating the row if needed (email unique)."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        email_l = (email or "").strip().lower()
        cur.execute(
            """
            INSERT INTO students (full_name, email)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE full_name = VALUES(full_name)
            """,
            (full_name.strip(), email_l)
        )
        if cur.lastrowid:
            student_id = cur.lastrowid
        else:
            cur.execute("SELECT student_id FROM students WHERE email=%s", (email_l,))
            row = cur.fetchone()
            student_id = row[0] if row else None
        conn.commit()
        return student_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (student): {err}")
        return None
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

def get_or_create_week(week_number: int, label: str | None = None):
    """Return weeks.week_id, creating the row if needed (week_number unique)."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        lbl = label if label else f"Week {int(week_number)}"
        cur.execute(
            """
            INSERT INTO weeks (week_number, label)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE label = COALESCE(VALUES(label), label)
            """,
            (int(week_number), lbl)
        )
        if cur.lastrowid:
            week_id = cur.lastrowid
        else:
            cur.execute("SELECT week_id FROM weeks WHERE week_number=%s", (int(week_number),))
            row = cur.fetchone()
            week_id = row[0] if row else None
        conn.commit()
        return week_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (week): {err}")
        return None
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

def has_existing_submission(student_id: int, week_id: int) -> bool:
    """True if this student already has a submission for this week."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM student_inputs WHERE student_id=%s AND week_id=%s LIMIT 1",
            (student_id, week_id),
        )
        return cur.fetchone() is not None
    except mysql.connector.Error as err:
        st.error(f"DB error checking existing submission: {err}")
        return False
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

def insert_submission_and_sentences(
    student_id, week_id,
    student_name, email, title, story,
    total, show, tell, reflection, comments,
    sentence_rows  # list of (sentence_idx, sentence_text, model_label, student_agree_int)
):
    """Insert one parent row + all sentence rows. Caller must pre-check duplicates."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()

        # Parent row
        cur.execute(
            """
            INSERT INTO student_inputs
              (student_id, week_id,
               student_name, email, title, story,
               total_sentences, show_sentences, tell_sentences,
               reflection, comments)
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s)
            """,
            (student_id, week_id,
             student_name, email, title, story,
             total, show, tell,
             reflection, comments)
        )
        input_id = cur.lastrowid

        # Sentence rows
        cur.executemany(
            """
            INSERT INTO student_sentences
              (input_id, week_id, sentence_idx, sentence_text, model_label, student_agree)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [(input_id, week_id, idx, text, label, agree)
             for (idx, text, label, agree) in sentence_rows]
        )

        conn.commit()
        return input_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (rolled back): {err}")
        return None
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

# =========================
# Email
# =========================
def send_feedback_email(email, student_name, title, summary, feedback_list, reflection, comment):
    changed = sum(1 for item in feedback_list if not item["agree"])
    details = "🧾 Sentence-by-sentence feedback:\n"
    for item in feedback_list:
        status = "✅ Agreed" if item["agree"] else "❌ Did NOT agree"
        details += f"- [{item['label']}] {item['sentence']}\n  ➤ {status}\n\n"

    msg = EmailMessage()
    msg["Subject"] = f"📊 Feedback for Your Data Story: {title}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    msg["Reply-To"] = EMAIL_ADDRESS
    msg.set_content(f"""
Dear {student_name},

Thank you for submitting your data story titled "{title}". Our system analyzed your submission and identified a total of {summary["total_sentences"]} sentences. Of these, {summary["show_sentences"]} were categorized as 'Show' and {summary["tell_sentences"]} as 'Tell'. You disagreed with the model's classification on {changed} sentence(s).

{details}
Your comment:
"{comment if comment else 'No additional comment provided.'}"

Your reflection:
"{reflection if reflection else 'No reflection provided.'}"

Best regards,
The Data Story Feedback Team
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        st.success(f"✅ Email sent to {email}")
    except smtplib.SMTPAuthenticationError as e:
        st.error("❌ Gmail auth failed. Double-check the 16-char App Password (no spaces).")
        st.exception(e)
    except Exception as e:
        st.error("❌ Failed to send email.")
        st.exception(e)

# =========================
# Admin-only week control (SECRETS-ONLY)
# =========================
def read_admin_key() -> str:
    try:
        return str(st.secrets.get("ADMIN_KEY", "") or "")
    except Exception:
        return ""

def current_week_default() -> int:
    try:
        return int(st.secrets.get("CURRENT_WEEK", 5))
    except Exception:
        return 5

if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

# Always resync week from secrets unless admin is unlocked
if not st.session_state.get("admin_ok"):
    st.session_state.week_number = current_week_default()

with st.sidebar:
    st.subheader("Admin")
    admin_key_input = st.text_input("Admin key", type="password")
    c1, c2 = st.columns(2)
    if c1.button("Unlock"):
        expected = read_admin_key().strip()
        entered = (admin_key_input or "").strip()
        if not expected:
            st.error("ADMIN_KEY missing in .streamlit/secrets.toml. Set it and restart the app.")
        elif entered == expected:
            st.session_state.admin_ok = True
            st.success("Admin unlocked")
        else:
            st.session_state.admin_ok = False
            st.error("Invalid key")
    if c2.button("Lock"):
        st.session_state.admin_ok = False
        st.info("Admin locked")

# Editable week only if unlocked
if st.session_state.get("admin_ok"):
    st.session_state.week_number = st.number_input(
        "Week number (admin)",
        min_value=1, max_value=52,
        value=int(st.session_state.week_number),
        step=1
    )
else:
    st.markdown(f"**Week:** {int(st.session_state.week_number)}")

# =========================
# UI flow
# =========================
st.title("✨ Show or Tell Prediction App ✨")
st.markdown("### Data Story Prompt")
st.image("chart_prompt.png", caption="Use this chart to write your data story.")
st.write("---")

if "page" not in st.session_state:
    st.session_state.page = "input"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# INPUT
if st.session_state.page == "input":
    student_name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    title = st.text_input("Enter a title for your data story:")
    input_text = st.text_area("Write your data story here:")
    stories = [input_text.strip()] if input_text.strip() else []

    if st.button("Analyze"):
        if not student_name or not email or not title:
            st.error("Please fill in your name, email, and story title before continuing.")
        elif stories:
            # Optional early block before analysis
            _sid = get_or_create_student(student_name, email)
            _wid = get_or_create_week(st.session_state.week_number)
            if _sid and _wid and has_existing_submission(_sid, _wid):
                st.error(f"You've already submitted for Week {int(st.session_state.week_number)}. Resubmissions are closed.")
            else:
                st.session_state.page = "results"
                st.session_state.stories = stories
                st.session_state.student_name = student_name
                st.session_state.student_email = email
                st.session_state.story_title = title

# RESULTS
if st.session_state.page == "results":
    stories = st.session_state.stories
    name = st.session_state.student_name
    email_addr = st.session_state.student_email
    story_title = st.session_state.story_title
    week_number = int(st.session_state.week_number)

    model, vectorizer = load_model_and_vectorizer()

    if not st.session_state.analysis_done:
        st.markdown("## Sentence Analysis")
        feedback_data, sentence_rows = [], []
        total = show = tell = 0

        for story in stories:
            sentences = nltk.sent_tokenize(story)
            predictions = predict_sentences(sentences, model, vectorizer)

            for i, (sent, label) in enumerate(zip(sentences, predictions)):
                label_text = "Show" if label == 0 else "Tell"
                color = "green" if label == 0 else "red"
                st.markdown(
                    f"<span style='color:{color}'><b>{label_text}:</b> {sent}</span>",
                    unsafe_allow_html=True
                )
                agree = st.checkbox("I agree with the model's label", key=f"agree_{i}")
                feedback_data.append({"sentence": sent, "label": label_text, "agree": agree})
                sentence_rows.append((i, sent, label_text, 1 if agree else 0))

            total += len(predictions)
            show += sum(1 for p in predictions if p == 0)
            tell += sum(1 for p in predictions if p == 1)

        # Persist
        st.session_state.student_feedback = feedback_data
        st.session_state.sentence_rows = sentence_rows
        st.session_state.total_sentences = total
        st.session_state.show_sentences = show
        st.session_state.tell_sentences = tell

        st.markdown("## Summary")
        st.write(f"Week: {week_number}")
        st.write(f"Total Sentences: {total}")
        st.write(f"Show Sentences: {show}")
        st.write(f"Tell Sentences: {tell}")

        fig, ax = plt.subplots()
        ax.bar(["Show", "Tell"], [show, tell])  # default colors
        ax.set_ylabel("Number of Sentences")
        ax.set_title("Show vs Tell Breakdown")
        st.pyplot(fig)

        st.markdown("## Comment")
        st.session_state.common_reason = st.text_area("Add your thoughts or reasons for disagreement")

        if st.button("Next: Reflection & Email"):
            st.session_state.analysis_done = True
            st.session_state.feedback_complete = True

    elif st.session_state.get("feedback_complete"):
        st.markdown("### Reflection")
        reflection = st.text_area("What did you learn from this feedback?", key="reflection")

        if st.button("Submit Feedback & Send Email"):
            # Resolve identities
            student_id = get_or_create_student(name, email_addr)
            week_id = get_or_create_week(week_number)
            if not student_id or not week_id:
                st.error("Could not resolve student/week. Aborting save.")
            else:
                # 🔒 Disallow resubmission
                if has_existing_submission(student_id, week_id):
                    st.error(f"You've already submitted for Week {week_number}. Resubmissions are closed.")
                    st.stop()

                input_id = insert_submission_and_sentences(
                    student_id, week_id,
                    name, email_addr, story_title, st.session_state.stories[0],
                    st.session_state.total_sentences,
                    st.session_state.show_sentences,
                    st.session_state.tell_sentences,
                    reflection, st.session_state.common_reason,
                    st.session_state.sentence_rows
                )

                if input_id:
                    summary = {
                        "total_sentences": st.session_state.total_sentences,
                        "show_sentences": st.session_state.show_sentences,
                        "tell_sentences": st.session_state.tell_sentences,
                    }
                    send_feedback_email(
                        email_addr, name, story_title, summary,
                        st.session_state.student_feedback,
                        reflection,
                        st.session_state.common_reason
                    )
                    st.success(f" Feedback submitted (input_id={input_id}, week={week_number}) and email sent!")
                else:
                    st.error("Could not save submission. Email not sent.")

        if st.button("Restart"):
            st.session_state.clear()
            st.rerun()
