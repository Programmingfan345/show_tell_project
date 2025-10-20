import streamlit as st
import joblib
import nltk
import os
import smtplib
import mysql.connector
import matplotlib.pyplot as plt
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK tokenizer

# make sure path is: C:\Data_Story\AWS_SPRING\AWS_SPRING
#  
# Type this in terminal
# python -m streamlit run streamlit_predict_app.py

nltk.download("punkt")
nltk.download("punkt_tab")

# EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS") or st.secrets.get("EMAIL_ADDRESS")
# EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") or st.secrets.get("EMAIL_PASSWORD")

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

# EMAIL_ADDRESS = "studentfeedbacks44@gmail.com"
# EMAIL_PASSWORD = "blueFire123#"
# EMAIL_PASSWORD = "fhwezznkcxmxfxdo"
#smtpbrzzAGP#$563
#slonrzTBAu5641$#
#omfl otsv sdqr qjos
#uxkb crgw sbsx jjrh

# --- Load model & vectorizer ---
def load_model_and_vectorizer():
    try:
        model = joblib.load("LogisticRegression_All_shots_data_model.pkl")
        vectorizer = joblib.load("LogisticRegression_All_shots_data_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()

# --- Predict sentence labels
def predict_sentences(sentences, model, vectorizer):
    tokens = [" ".join(nltk.word_tokenize(s.lower())) for s in sentences]
    return model.predict(vectorizer.transform(tokens))

# --- Connect to MySQL
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=st.secrets["DB_HOST"],
            port=int(st.secrets["DB_PORT"]),
            database=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            # later (recommended): ssl_ca="C:/path/to/rds-combined-ca-bundle.pem"
        )
    except mysql.connector.Error as err:
        st.error(f"Database Connection Error: {err}")
        return None

# def get_db_connection():
#     try:
#         return mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="slothFace@48",
#             # password="Vigi@2004",
#             database="Showtell",
#             port=3306
#         )
#     except mysql.connector.Error as err:
#         print("Database Connection Error:", err)
#         return None

# --- Insert into database
def insert_student_data(student_name, email, title, story, total, show, tell, reflection, comments):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            # "INSERT INTO student_inputs (name, email, title, story, total_sentences, show_sentences, tell_sentences, reflection, week, comments) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            # (name, email, title, story, total, show, tell, reflection, "Week 5", comments)
            "INSERT INTO student_inputs (student_name, email, title, story, total_sentences, show_sentences, tell_sentences, reflection, week_number, comments) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (student_name, email, title, story, total, show, tell, reflection, "Week 5", comments)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print("⚠️ MySQL Error:", err)

def count_rows():
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM student_inputs;")
        n = cur.fetchone()[0]
        return n
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

# --- Send feedback email
def send_feedback_email(email, student_name, title, summary, feedback_list, reflection, comment):
    changed = sum(1 for item in feedback_list if not item["agree"])
    sentence_feedback = "🧾 Sentence-by-sentence feedback:\n"
    for item in feedback_list:
        status = "✅ Agreed" if item["agree"] else "❌ Did NOT agree"
        sentence_feedback += f"- [{item['label']}] {item['sentence']}\n  ➤ {status}\n\n"

    # 1) Build the message
    msg = EmailMessage()
    msg["Subject"] = f"📊 Feedback for Your Data Story: {title}"
    msg["From"] = EMAIL_ADDRESS          # must match the authenticated Gmail
    msg["To"] = email
    msg["Reply-To"] = EMAIL_ADDRESS
    msg.set_content(f"""
Dear {student_name},

Thank you for submitting your data story titled "{title}". Our system analyzed your submission and identified a total of {summary["total_sentences"]} sentences. Of these, {summary["show_sentences"]} were categorized as 'Show' and {summary["tell_sentences"]} as 'Tell'. You disagreed with the model's classification on {changed} sentence(s).

Below is a detailed review of each sentence, showing how the model labeled it and whether you agreed:

{sentence_feedback}

Your comment:
"{comment if comment else 'No additional comment provided.'}"

Your reflection:
"{reflection if reflection else 'No reflection provided.'}"

We appreciate your thoughtful participation and hope this feedback helps you enhance your data storytelling skills.

Best regards,
The Data Story Feedback Team
""")

    # 2) Send it
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  # 16-char Gmail App Password
            smtp.send_message(msg)
        st.success(f"✅ Email sent to {email}")
    except smtplib.SMTPAuthenticationError as e:
        st.error("❌ Gmail auth failed. Double-check the 16-char App Password (no spaces).")
        st.exception(e)
    except Exception as e:
        st.error("❌ Failed to send email.")
        st.exception(e)


# --- Streamlit UI ---
st.title("✨ Show or Tell Prediction App ✨")
st.markdown("###Data Story Prompt")
st.image("chart_prompt.png", caption="Use this chart to write your data story.")
st.write("---")

if "page" not in st.session_state:
    st.session_state.page = "input"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- INPUT PAGE ---
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
            st.session_state.page = "results"
            st.session_state.stories = stories
            st.session_state.student_name = student_name
            st.session_state.student_email = email
            st.session_state.story_title = title

# --- RESULTS PAGE ---
if st.session_state.page == "results":
    stories = st.session_state.stories
    name = st.session_state.student_name
    email = st.session_state.student_email
    title = st.session_state.story_title
    model, vectorizer = load_model_and_vectorizer()

    if not st.session_state.analysis_done:
        st.markdown("##Sentence Analysis")
        feedback_data = []

        total = show = tell = 0

        for story in stories:
            sentences = nltk.sent_tokenize(story)
            predictions = predict_sentences(sentences, model, vectorizer)

            for i, (sent, label) in enumerate(zip(sentences, predictions)):
                label_text = "Show" if label == 0 else "Tell"
                color = "green" if label == 0 else "red"

                st.markdown(f"<span style='color:{color}'><b>{label_text}:</b> {sent}</span>", unsafe_allow_html=True)
                agree = st.checkbox("I agree with the model's label", key=f"agree_{i}")
                feedback_data.append({"sentence": sent, "label": label_text, "agree": agree})

            total += len(predictions)
            show += sum(1 for p in predictions if p == 0)
            tell += sum(1 for p in predictions if p == 1)

        st.session_state.student_feedback = feedback_data
        st.session_state.total_sentences = total
        st.session_state.show_sentences = show
        st.session_state.tell_sentences = tell

        st.markdown("## Comment")
        st.session_state.common_reason = st.text_area("Add your thoughts or reasons for disagreement")

        st.markdown("##Summary")
        st.write(f"Total Sentences: {total}")
        st.write(f"Show Sentences: {show}")
        st.write(f"Tell Sentences: {tell}")

        fig, ax = plt.subplots()
        ax.bar(["Show", "Tell"], [show, tell], color=["green", "red"])
        ax.set_ylabel("Number of Sentences")
        ax.set_title("Show vs Tell Breakdown")
        st.pyplot(fig)

        if st.button("Next: Reflection & Email"):
            st.session_state.analysis_done = True
            st.session_state.feedback_complete = True

    elif st.session_state.get("feedback_complete"):
        st.markdown("###Reflection")
        reflection = st.text_area("What did you learn from this feedback?", key="reflection")

        if st.button("Submit Feedback & Send Email"):
            summary = {
                "total_sentences": st.session_state.total_sentences,
                "show_sentences": st.session_state.show_sentences,
                "tell_sentences": st.session_state.tell_sentences,
            }
            insert_student_data(
                name, email, title, stories[0],
                summary["total_sentences"],
                summary["show_sentences"],
                summary["tell_sentences"],
                reflection,
                st.session_state.common_reason
            )
            send_feedback_email(
                email, name, title, summary,
                st.session_state.student_feedback,
                reflection,
                st.session_state.common_reason
            )
            st.success(" Feedback submitted and email sent!")

        if st.button("Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()