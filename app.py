from flask import Flask, render_template, request, jsonify
import os
import re
from collections import Counter
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

app = Flask(__name__)

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "false").strip().lower() in {"1", "true", "yes", "on"}

model = None
if ENABLE_GEMINI and genai and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as exc:
        print(f"Gemini initialization failed: {exc}")

# Configure upload folder and allowed extensions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "were", "will", "with", "you", "your", "this", "these", "those", "our",
    "their", "they", "we", "can", "should", "must", "using", "use", "used"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def tokenize(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-/]{1,}\b", text.lower())
    return [word for word in words if word not in COMMON_STOPWORDS]


def unique_keywords(text):
    return set(tokenize(text))


def top_keywords(text, limit=12):
    return [word for word, _ in Counter(tokenize(text)).most_common(limit)]


def format_list(items):
    return "\n".join(f"- {item}" for item in items)


def local_resume_analysis(resume_text, job_description):
    resume_keywords = unique_keywords(resume_text)
    job_keywords = unique_keywords(job_description)
    matched = sorted(job_keywords & resume_keywords)
    missing = sorted(job_keywords - resume_keywords)

    match_score = int((len(matched) / max(len(job_keywords), 1)) * 100)
    strongest_matches = matched[:12] or ["No strong keyword overlap was detected."]
    biggest_gaps = missing[:12] or ["No major keyword gaps were detected."]

    strength_items = [
        f"Estimated keyword alignment score: {match_score}%.",
        f"Matched keywords: {', '.join(strongest_matches)}",
    ]
    if any(char.isdigit() for char in resume_text):
        strength_items.append("The resume includes numeric or measurable content, which usually helps demonstrate impact.")
    else:
        strength_items.append("Consider adding measurable achievements with numbers, percentages, or outcomes.")

    improvement_items = [
        f"Missing or weak keywords: {', '.join(biggest_gaps)}",
        "Tailor the summary, skills, and experience sections more directly to the target role.",
        "Mirror the job description's terminology where it truthfully matches your experience.",
    ]

    recommendation_items = [
        "Add a dedicated skills section with role-specific tools, platforms, and frameworks.",
        "Rewrite experience bullets to emphasize outcomes, ownership, and quantified results.",
        "Place the most relevant achievements near the top of the resume for faster recruiter review.",
    ]

    return "\n\n".join([
        "S K I L L S   M A T C H",
        format_list(strength_items),
        "A R E A S   F O R   I M P R O V E M E N T",
        format_list(improvement_items),
        "R E C O M M E N D A T I O N S",
        format_list(recommendation_items),
    ])


def local_ats_analysis(resume_text):
    lowered = resume_text.lower()
    common_sections = ["summary", "experience", "education", "skills", "projects", "certifications"]
    found_sections = [section.title() for section in common_sections if section in lowered]
    missing_sections = [section.title() for section in common_sections if section not in lowered]
    keyword_samples = top_keywords(resume_text)

    score = 50
    if found_sections:
        score += min(len(found_sections) * 5, 20)
    if any(char.isdigit() for char in resume_text):
        score += 10
    if len(resume_text.split()) > 250:
        score += 10
    if any(symbol in resume_text for symbol in ["|", "\t"]):
        score -= 5
    score = max(0, min(score, 100))

    return "\n\n".join([
        "FORMATTING",
        format_list([
            "The resume appears text-readable, which is a good starting point for ATS parsing.",
            "Use standard section headings and simple left-aligned formatting.",
            "Avoid tables, text boxes, multi-column layouts, headers/footers, and image-only content.",
        ]),
        "KEYWORDS",
        format_list([
            f"Common keywords detected: {', '.join(keyword_samples) if keyword_samples else 'No strong keywords detected.'}",
            "Add exact role-specific terms from your target job descriptions where accurate.",
            "Include tools, platforms, certifications, and domain terms in a clear skills section.",
        ]),
        "STRUCTURE",
        format_list([
            f"Sections detected: {', '.join(found_sections) if found_sections else 'No common sections detected.'}",
            f"Sections to consider adding: {', '.join(missing_sections[:4]) if missing_sections else 'Core sections look present.'}",
            "Keep job titles, company names, dates, and bullet points consistent across roles.",
        ]),
        "CONTENT",
        format_list([
            "Use bullet points that start with action verbs and end with measurable outcomes where possible.",
            "Spell out abbreviations at least once if they are important search terms.",
            "Save the final resume as a clean PDF or DOCX after checking that text remains selectable.",
        ]),
        "OPTIMIZATION",
        format_list([
            "Customize the resume for each job instead of sending the same generic version everywhere.",
            "Move the strongest and most relevant experience closer to the top.",
            "Add metrics such as percentages, revenue, time saved, users supported, or project scale.",
        ]),
        "SCORE",
        format_list([f"Estimated ATS compatibility score: {score}/100"]),
    ])


def ask_gemini(prompt):
    if not model:
        raise RuntimeError("Gemini is not configured. Set GOOGLE_API_KEY to enable AI responses.")

    response = model.generate_content(prompt)
    return getattr(response, "text", "").strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    if PyPDF2 is None:
        print("PyPDF2 is not installed. PDF parsing is unavailable.")
        return text

    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    if Document is None:
        print("python-docx is not installed. DOCX parsing is unavailable.")
        return text

    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ats_analysis")
def ats_analysis():
    return render_template("ats_analysis.html")

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    job_description = request.form.get("job_description", "")
    resume_text = ""

    # Check if a file was uploaded
    if 'resume_file' in request.files:
        file = request.files['resume_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(filepath)
            elif file_extension == 'docx':
                resume_text = extract_text_from_docx(filepath)
            elif file_extension == 'txt':
                resume_text = extract_text_from_txt(filepath)

            if os.path.exists(filepath):
                os.remove(filepath)
        elif file:
            return jsonify({"analysis": "Invalid file type. Allowed types are: pdf, docx, txt"})
    elif 'resume_text' in request.form:
        resume_text = request.form['resume_text']

    job_description = job_description.strip()
    resume_text = resume_text.strip()

    if not resume_text or not job_description:
        return jsonify({"analysis": "Please provide both the resume and the job description."})

    try:
        prompt = f"""Analyze the following resume text against the following job description. Provide a comprehensive summary of how well the resume aligns with the job description, clearly highlighting key strengths and specific areas for improvement. Focus on matching skills, relevant experience, quantifiable achievements, and overall suitability for the role.

                Maintain a structured format for all replies using clearly highlighted headers. Format headers by placing them on their own line and using all caps with spaces between letters (like "S K I L L S   M A T C H").

                Resume Text: 
                {resume_text}

                Job Description:
                {job_description}

                Analysis:
                """
        analysis = ask_gemini(prompt)
        if analysis:
            return jsonify({"analysis": analysis})
    except Exception as e:
        print("Error:", e)

    return jsonify({"analysis": local_resume_analysis(resume_text, job_description)})

@app.route("/analyze_ats", methods=["POST"])
def analyze_ats():
    resume_text = ""

    # Check if a file was uploaded
    if 'resume_file' in request.files:
        file = request.files['resume_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(filepath)
            elif file_extension == 'docx':
                resume_text = extract_text_from_docx(filepath)
            elif file_extension == 'txt':
                resume_text = extract_text_from_txt(filepath)

            if os.path.exists(filepath):
                os.remove(filepath)
        elif file:
            return jsonify({"analysis": "Invalid file type. Allowed types are: pdf, docx, txt"})
    elif 'resume_text' in request.form:
        resume_text = request.form['resume_text']

    resume_text = resume_text.strip()

    if not resume_text:
        return jsonify({"analysis": "Please provide the resume for ATS analysis."})

    try:
        prompt = f"""Analyze the following resume text for Applicant Tracking System (ATS) compatibility. Provide a detailed assessment covering these aspects:
        
        1. Formatting: Is the resume using ATS-friendly formatting? Check for proper headings, standard fonts, and lack of complex layouts.
        2. Keywords: Does the resume include sufficient relevant keywords for the target industry/role?
        3. Structure: Is the resume properly structured with clear sections (Experience, Education, Skills)?
        4. Content: Does the content avoid elements that might confuse ATS (tables, headers/footers, images)?
        5. Optimization: Provide specific recommendations for improving ATS compatibility.
        6. Score: Give an estimated ATS compatibility score (0-100) based on the analysis.
        
        For each point, provide:
        - Current status (what's working well)
        - Potential issues
        - Specific improvement suggestions
        
        Resume Text:
        {resume_text}
        
        ATS Analysis:
        """
        analysis = ask_gemini(prompt)
        if analysis:
            return jsonify({"analysis": analysis})
    except Exception as e:
        print("Error during ATS analysis:", e)

    return jsonify({"analysis": local_ats_analysis(resume_text)})

@app.route("/ask_resume_question", methods=["POST"])
def ask_resume_question():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please provide a question or try speaking again."})

    try:
        prompt = f"""Answer the following resume question clearly and concisely:
        
        Question: {question}
        
        Answer in this format:
        **Answer:** [your answer here]
        **Tips:** [additional tips if relevant]"""
        
        answer = ask_gemini(prompt)
        if answer:
            return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error processing question: {e}")

    fallback_answer = (
        "**Answer:** Focus on tailoring your resume to the target role, using clear section headings, "
        "relevant keywords, and measurable achievements.\n"
        "**Tips:** Add role-specific skills, keep formatting simple, and quantify impact wherever possible."
    )
    return jsonify({"answer": fallback_answer})



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
