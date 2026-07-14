import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer Chatbot")
st.write("Upload your Resume (PDF) and get an instant analysis.")

# ----------------------------
# Skills Database
# ----------------------------
skills_list = [
    "python",
    "sql",
    "machine learning",
    "data science",
    "excel",
    "power bi",
    "java",
    "html",
    "css",
    "c++",
    "javascript"
]

job_description = """
Python
SQL
Machine Learning
Data Science
HTML
CSS
Java
Power BI
Excel
"""

# ----------------------------
# Upload Resume
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose Resume PDF",
    type=["pdf"]
)

# ----------------------------
# Extract Text Function
# ----------------------------
def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text.lower()

# ----------------------------
# Process Resume
# ----------------------------
if uploaded_file is not None:

    resume_text = extract_text(uploaded_file)

    st.subheader("📄 Resume Text")
    st.write(resume_text)
    # ----------------------------
    # Skills Found
    # ----------------------------

    found_skills = []

    for skill in skills_list:
        if skill in resume_text:
            found_skills.append(skill)

    st.subheader("✅ Skills Found")
    st.write(found_skills)

    # ----------------------------
    # Missing Skills
    # ----------------------------

    missing_skills = []

    for skill in skills_list:
        if skill not in resume_text:
            missing_skills.append(skill)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

    # ----------------------------
    # Resume Score
    # ----------------------------

    score = len(found_skills) * 10

    if score > 100:
        score = 100

    st.subheader("📊 Resume Score")
    st.success(f"{score} / 100")

    # ----------------------------
    # Chatbot Suggestion
    # ----------------------------

    st.subheader("💬 Chatbot Suggestion")

    if score >= 80:
        st.success("Excellent Resume! Your profile is strong.")
    elif score >= 60:
        st.warning("Good Resume. Add more projects and certifications.")
    else:
        st.error("Need Improvement. Add more technical skills and projects.")

    # ----------------------------
    # Job Match
    # ----------------------------

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(vectors)[0][1]

    st.subheader("🎯 Job Match Percentage")
    st.info(f"{round(similarity * 100, 2)} %")
    # ----------------------------
    # Pie Chart
    # ----------------------------

    st.subheader("📈 Skills Analysis Chart")

    labels = ["Found Skills", "Missing Skills"]
    sizes = [len(found_skills), len(missing_skills)]

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.axis("equal")

    st.pyplot(fig1)

    # ----------------------------
    # Bar Chart
    # ----------------------------

    st.subheader("📊 Skills Comparison")

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.bar(
        ["Found", "Missing"],
        [len(found_skills), len(missing_skills)]
    )

    ax2.set_ylabel("Number of Skills")
    ax2.set_title("Resume Skills")

    st.pyplot(fig2)

    # ----------------------------
    # Final Summary
    # ----------------------------

    st.subheader("📋 Resume Summary")

    st.write("**Resume Score:**", score, "/100")
    st.write("**Skills Found:**", len(found_skills))
    st.write("**Missing Skills:**", len(missing_skills))
    st.write("**Job Match:**", round(similarity * 100, 2), "%")

    st.success("🎉 Resume Analysis Completed Successfully!")