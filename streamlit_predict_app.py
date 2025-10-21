import os
import smtplib
import mysql.connector
import joblib
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# NLTK
# -------------------------
nltk.download("punkt")
nltk.download("punkt_tab")  # harmless if not present

# -------------------------
# Secrets helper
# -------------------------
def get_secret(name: str):
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return None

EMAIL_ADDRESS = get_secret("EMAIL_ADDRESS")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD")
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    st.error("Email creds missing. Set EMAIL_ADDRESS and EMAIL_PASSWORD (16-char Gmail App Password) via env vars or .streamlit/secrets.toml.")
    st.stop()

# -------------------------
# Model + Vectorizer
# -------------------------
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

# -------------------------
# DB
# -------------------------
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            port=int(st.secrets["DB_PORT"]),
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            autocommit=False,
        )
    except mysql.connector.Error as err:
        st.error(f"Database Connection Error: {err}")
        return None

def get_or_create_student(full_name: str, email: str):
    """Return students.student_id, creating the row if needed (email is unique)."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        # normalize email to lowercase; triggers also enforce this
        email_l = email.strip().lower()
        cur.execute(
            """
            INSERT INTO students (full_name, email)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE full_name = VALUES(full_name)
            """,
            (full_name.strip(), email_l)
        )
        # If inserted, lastrowid will be set; if updated, lastrowid can be 0 -> select id.
        if cur.lastrowid:
            student_id = cur.lastrowid
        else:
            cur.execute("SELECT student_id FROM students WHERE email = %s", (email_l,))
            student_id = cur.fetchone()[0]
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
    """Return weeks.week_id, creating the row if needed (week_number is unique)."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        lbl = label if label else f"Week {week_number}"
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
            cur.execute("SELECT week_id FROM weeks WHERE week_number = %s", (int(week_number),))
            week_id = cur.fetchone()[0]
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

def insert_submission_and_sentences(
    student_id, week_id,
    student_name, email, title, story,
    total, show, tell, reflection, comments,
    agreed_show, agreed_tell, disagreed_show, disagreed_tell,
    sentence_rows  # list of tuples: (sentence_idx, sentence_text, model_label, student_agree_int)
):
    """
    Inserts one row into student_inputs (with student_id & week_id),
    then all related student_sentences rows (also with week_id).
    Returns input_id on success.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()

        # 1) Insert parent (submission)
        cur.execute(
            """
            INSERT INTO student_inputs
              (student_id, week_id,
               student_name, email, title, story,
               total_sentences, show_sentences, tell_sentences,
               agreed_show, agreed_tell, disagreed_show, disagreed_tell,
               reflection, comments)
            VALUES (%s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s)
            """,
            (student_id, week_id,
             student_name, email, title, story,
             total, show, tell,
             agreed_show, agreed_tell, disagreed_show, disagreed_tell,
             reflection, comments)
        )
        input_id = cur.lastrowid

        # 2) Insert all sentences with week_id
        cur.executemany(
            """
            INSERT INTO student_sentences
              (input_id, week_id, sentence_idx, sentence_text, model_label, student_agree)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [(input_id, week_id, idx, text, label, agree) for (idx, text, label, agree) in sentence_rows]
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

# -------------------------
# Email
# -------------------------
def send_feedback_email(email, student_name, title, summary, feedback_list, reflection, comment,
                        agreed_show, agreed_tell, disagreed_show, disagreed_tell):
    changed = sum(1 for item in feedback_list if not item["agree"])
    sentence_feedback = "🧾 Sentence-by-sentence feedback:\n"
    for item in feedback_list:
        status = "✅ Agreed" if item["agree"] else "❌ Did NOT agree"
        sentence_feedback += f"- [{item['label']}] {item['sentence']}\n  ➤ {status}\n\n"

    msg = EmailMessage()
    msg["Subject"] = f"📊 Feedback for Your Data Story: {title}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    msg["Reply-To"] = EMAIL_ADDRESS
    msg.set_content(f"""
Dear {student_name},

Thank you for submitting your data story titled "{title}". Our system analyzed your submission and identified a total of {summary["total_sentences"]} sentences. Of these, {summary["show_sentences"]} were categorized as 'Show' and {summary["tell_sentences"]} as 'Tell'. You disagreed with the model's classification on {changed} sentence(s).

Your checkbox selections:
• Agreed (Show): {agreed_show}
• Agreed (Tell): {agreed_tell}
• Disagreed (Show): {disagreed_show}
• Disagreed (Tell): {disagreed_tell}

Below is a detailed review of each sentence, showing how the model labeled it and whether you agreed:

{sentence_feedback}

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

# -------------------------
# UI
# -------------------------
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
    week_number = st.number_input("Week number", min_value=1, max_value=52, value=5, step=1)  # NEW
    input_text = st.text_area("Write your data story here:")
    stories = [input_text.strip()] if input_text.strip() else []

    if st.button("Analyze"):
        if not student_name or not email or not title:
            st.error("Please fill in your name, email, and story title before continuing.")
        elif stories:
            st.session_state.page = "results"
            st.session_state.stories = stories
            st.session_state.student_name = student_name
            st.session_state.student_email = email
            st.session_state.story_title = title
            st.session_state.week_number = int(week_number)

# RESULTS
if st.session_state.page == "results":
    stories = st.session_state.stories
    name = st.session_state.student_name
    email_addr = st.session_state.student_email
    story_title = st.session_state.story_title
    week_number = st.session_state.week_number
    model, vectorizer = load_model_and_vectorizer()

    if not st.session_state.analysis_done:
        st.markdown("## Sentence Analysis")
        feedback_data = []     # [{sentence, label, agree}]
        sentence_rows = []     # [(idx, text, label, agree_int)] for DB
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

        # Tallies by label
        agreed_show = sum(1 for item in feedback_data if item["label"] == "Show" and item["agree"])
        agreed_tell = sum(1 for item in feedback_data if item["label"] == "Tell" and item["agree"])
        disagreed_show = sum(1 for item in feedback_data if item["label"] == "Show" and not item["agree"])
        disagreed_tell = sum(1 for item in feedback_data if item["label"] == "Tell" and not item["agree"])

        # Persist in session
        st.session_state.student_feedback = feedback_data
        st.session_state.sentence_rows = sentence_rows
        st.session_state.total_sentences = total
        st.session_state.show_sentences = show
        st.session_state.tell_sentences = tell
        st.session_state.agreed_show = agreed_show
        st.session_state.agreed_tell = agreed_tell
        st.session_state.disagreed_show = disagreed_show
        st.session_state.disagreed_tell = disagreed_tell

        st.markdown("## Comment")
        st.session_state.common_reason = st.text_area("Add your thoughts or reasons for disagreement")

        st.markdown("## Summary")
        st.write(f"Week: {week_number}")
        st.write(f"Total Sentences: {total}")
        st.write(f"Show Sentences: {show}")
        st.write(f"Tell Sentences: {tell}")
        st.write(f"✔️ Agreed (Show): {agreed_show}")
        st.write(f"✔️ Agreed (Tell): {agreed_tell}")

        fig, ax = plt.subplots()
        ax.bar(["Show", "Tell"], [show, tell], color=["green", "red"])
        ax.set_ylabel("Number of Sentences")
        ax.set_title("Show vs Tell Breakdown")
        st.pyplot(fig)

        if st.button("Next: Reflection & Email"):
            st.session_state.analysis_done = True
            st.session_state.feedback_complete = True

    elif st.session_state.get("feedback_complete"):
        st.markdown("### Reflection")
        reflection = st.text_area("What did you learn from this feedback?", key="reflection")

        if st.button("Submit Feedback & Send Email"):
            summary = {
                "total_sentences": st.session_state.total_sentences,
                "show_sentences": st.session_state.show_sentences,
                "tell_sentences": st.session_state.tell_sentences,
            }

            # Get or create identities
            student_id = get_or_create_student(name, email_addr)
            week_id = get_or_create_week(st.session_state.week_number)
            if not student_id or not week_id:
                st.error("Could not resolve student/week. Aborting save.")
            else:
                # Insert parent + sentences in one transaction
                input_id = insert_submission_and_sentences(
                    student_id, week_id,
                    name, email_addr, story_title, st.session_state.stories[0],
                    summary["total_sentences"], summary["show_sentences"], summary["tell_sentences"],
                    reflection, st.session_state.common_reason,
                    st.session_state.agreed_show, st.session_state.agreed_tell,
                    st.session_state.disagreed_show, st.session_state.disagreed_tell,
                    st.session_state.sentence_rows
                )

                if input_id:
                    send_feedback_email(
                        email_addr, name, story_title, summary,
                        st.session_state.student_feedback,
                        reflection,
                        st.session_state.common_reason,
                        st.session_state.agreed_show,
                        st.session_state.agreed_tell,
                        st.session_state.disagreed_show,
                        st.session_state.disagreed_tell
                    )
                    st.success(f" Feedback submitted (input_id={input_id}, week_id={week_id}) and email sent!")
                else:
                    st.error("Could not save submission. Email not sent.")

        if st.button("Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
