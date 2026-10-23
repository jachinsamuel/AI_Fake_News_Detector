"""
Build professional 11-slide presentation for Fake News Detection & Real-Time Verification System.
Features beginner-friendly words, short crisp bullet points, professional embedded diagram images,
and clean 16:9 widescreen layout without AI buzzwords or slop.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FakeNews_Presentation_11_Slides.pptx")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slides_assets")

# Palette definitions
NAVY_BG = RGBColor(15, 23, 42)          # #0F172A
SLATE_CARD_DARK = RGBColor(30, 41, 59)   # #1E293B
LIGHT_BG = RGBColor(248, 250, 252)       # #F8FAFC
WHITE = RGBColor(255, 255, 255)
DARK_TEXT = RGBColor(15, 23, 42)         # #0F172A
MUTED_TEXT = RGBColor(71, 85, 105)       # #475569
MUTED_LIGHT = RGBColor(148, 163, 184)    # #94A3B8
BLUE_PRIMARY = RGBColor(37, 99, 235)     # #2563EB
CYAN_ACCENT = RGBColor(14, 165, 233)     # #0EA5E9
EMERALD = RGBColor(16, 185, 129)         # #10B981
PURPLE_PILL = RGBColor(238, 242, 255)    # #EEF2FF
PURPLE_TEXT = RGBColor(67, 56, 202)      # #4338CA
BORDER_LIGHT = RGBColor(226, 232, 240)   # #E2E8F0
BORDER_DARK = RGBColor(51, 65, 85)       # #334155

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]


def create_solid_background(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg


def add_header(slide, category_text, title_text, subtitle_text=""):
    # Category pill
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(3.4), Inches(0.35))
    pill.fill.solid()
    pill.fill.fore_color.rgb = PURPLE_PILL
    pill.line.fill.background()
    tf_p = pill.text_frame
    tf_p.word_wrap = False
    tf_p.margin_left = tf_p.margin_right = tf_p.margin_top = tf_p.margin_bottom = 0
    p_p = tf_p.paragraphs[0]
    p_p.text = category_text.upper()
    p_p.alignment = PP_ALIGN.CENTER
    p_p.font.name = FONT_HEADING
    p_p.font.size = Pt(10)
    p_p.font.bold = True
    p_p.font.color.rgb = PURPLE_TEXT

    # Title box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_HEADING
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = DARK_TEXT

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_BODY
        p2.font.size = Pt(12.5)
        p2.font.color.rgb = MUTED_TEXT
        p2.space_before = Pt(3)


def add_card(slide, left, top, width, height, title, items, tag="", dark_mode=False, border_color=None):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = SLATE_CARD_DARK if dark_mode else WHITE
    card.line.color.rgb = border_color if border_color else (BORDER_DARK if dark_mode else BORDER_LIGHT)
    card.line.width = Pt(1.5)

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.28)
    tf.margin_right = Inches(0.28)
    tf.margin_top = Inches(0.24)
    tf.margin_bottom = Inches(0.2)

    # Title
    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(15.5)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE if dark_mode else DARK_TEXT

    if tag:
        p_tag = tf.add_paragraph()
        p_tag.text = tag.upper()
        p_tag.font.name = FONT_HEADING
        p_tag.font.size = Pt(9.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = CYAN_ACCENT if dark_mode else BLUE_PRIMARY
        p_tag.space_before = Pt(2)
        p_tag.space_after = Pt(6)
    else:
        p_title.space_after = Pt(6)

    # Items
    for item in items:
        p_item = tf.add_paragraph()
        p_item.space_before = Pt(5)
        p_item.font.name = FONT_BODY
        p_item.font.size = Pt(11.5)
        
        if isinstance(item, tuple):
            lead, body = item
            run_lead = p_item.add_run()
            run_lead.text = f"{lead}: "
            run_lead.font.bold = True
            run_lead.font.color.rgb = WHITE if dark_mode else DARK_TEXT
            
            run_body = p_item.add_run()
            run_body.text = body
            run_body.font.color.rgb = MUTED_LIGHT if dark_mode else MUTED_TEXT
        else:
            run = p_item.add_run()
            run.text = str(item)
            run.font.color.rgb = MUTED_LIGHT if dark_mode else MUTED_TEXT

    return card


# ==========================================
# SLIDE 1: TITLE & PROJECT METADATA
# ==========================================
s1 = prs.slides.add_slide(blank_layout)
create_solid_background(s1, NAVY_BG)

# Title Accent Badge
badge1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.9), Inches(4.5), Inches(0.4))
badge1.fill.solid()
badge1.fill.fore_color.rgb = BLUE_PRIMARY
badge1.line.fill.background()
tf_b1 = badge1.text_frame
tf_b1.margin_left = tf_b1.margin_right = tf_b1.margin_top = tf_b1.margin_bottom = 0
p_b1 = tf_b1.paragraphs[0]
p_b1.text = "CAPSTONE PROJECT PRESENTATION | CSE 2026"
p_b1.alignment = PP_ALIGN.CENTER
p_b1.font.name = FONT_HEADING
p_b1.font.size = Pt(10.5)
p_b1.font.bold = True
p_b1.font.color.rgb = WHITE

# Main Title Box
tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.2))
tf1 = tb1.text_frame
tf1.word_wrap = True
tf1.margin_left = tf1.margin_right = tf1.margin_top = tf1.margin_bottom = 0

p1 = tf1.paragraphs[0]
p1.text = "Fake News Detection with Real-Time Web Checking"
p1.font.name = FONT_HEADING
p1.font.size = Pt(32)
p1.font.bold = True
p1.font.color.rgb = WHITE

p1_sub = tf1.add_paragraph()
p1_sub.text = "Combining 3 Machine Learning Models, Photo Text Reading (OCR), and Live Google News Verification"
p1_sub.font.name = FONT_BODY
p1_sub.font.size = Pt(15.5)
p1_sub.font.color.rgb = CYAN_ACCENT
p1_sub.space_before = Pt(8)

# 3 Metadata Cards (College, Team Members, Guide)
add_card(
    s1, Inches(1.0), Inches(3.9), Inches(3.6), Inches(2.9),
    "College & Department",
    [
        ("College", "[Enter Your College / University Name]"),
        ("Department", "Computer Science & Engineering"),
        ("Degree", "Bachelor of Engineering (B.E. / B.Tech)"),
        ("Academic Year", "2025 - 2026")
    ],
    tag="Academic Info",
    dark_mode=True
)

add_card(
    s1, Inches(4.866), Inches(3.9), Inches(3.6), Inches(2.9),
    "Project Team",
    [
        ("Team Lead", "[Your Name - Register No]"),
        ("Member 2", "[Team Member 2 Name]"),
        ("Member 3", "[Team Member 3 Name]"),
        ("Member 4", "[Team Member 4 Name]")
    ],
    tag="Student Team",
    dark_mode=True
)

add_card(
    s1, Inches(8.733), Inches(3.9), Inches(3.6), Inches(2.9),
    "Project Guide",
    [
        ("Faculty Guide", "[Dr. / Prof. Guide Name]"),
        ("Designation", "Assistant / Associate Professor, CSE"),
        ("Domain", "Machine Learning & Web Systems"),
        ("Project Status", "Fully Working & Deployed")
    ],
    tag="Faculty Supervisor",
    dark_mode=True
)

s1.notes_slide.notes_text_frame.text = (
    "Good morning everyone. Today we are presenting our project: Fake News Detection with Real-Time Web Checking. "
    "Our system helps people find out if a news story is true or fake. "
    "Unlike old AI systems that guess using old data, our project combines 3 machine learning models with live Google News searching "
    "so it can verify breaking news happening right now today."
)


# ==========================================
# SLIDE 2: PROBLEM STATEMENT (Old ML vs Our Fix)
# ==========================================
s2 = prs.slides.add_slide(blank_layout)
create_solid_background(s2, LIGHT_BG)
add_header(s2, "Problem Statement", "Why Traditional AI Fails on Today's Breaking News", "Older AI models only know the past. They make false guesses about today's events.")

# Left Content Card
add_card(
    s2, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "The 3 Big Problems with Old Systems",
    [
        ("Old Data Problem", "Old AI is trained on past articles from years ago. It has zero knowledge of events happening today."),
        ("False Accusations", "When real breaking news happens (like a sudden school holiday today due to heavy rain), old AI flags it as FAKE because it never saw it before!"),
        ("Fake News is Everywhere", "Rumors spread 6 times faster than the truth on WhatsApp, Instagram, and Twitter."),
        ("No Explanations", "Normal AI just gives a percentage without explaining which words made it suspicious."),
        ("Our Solution", "Pair smart AI pattern recognition with live Google News search so it checks if the news actually happened today!")
    ],
    tag="Core Challenge & Solution",
    border_color=BLUE_PRIMARY
)

# Right Diagram Image
img_s2 = os.path.join(ASSETS_DIR, "diagram_problem.png")
if os.path.exists(img_s2):
    s2.shapes.add_picture(img_s2, Inches(6.6), Inches(1.85), width=Inches(5.9))

s2.notes_slide.notes_text_frame.text = (
    "Slide 2 explains the problem we are solving. "
    "If you ask traditional AI about a breaking news event from this morning—like a collector declaring a school holiday due to rain—"
    "the AI will flag it as fake news simply because the event is not in its old training database. "
    "This diagram shows the difference: old AI makes a mistake, while our hybrid system checks live news wires and confirms the truth."
)


# ==========================================
# SLIDE 3: SYSTEM ARCHITECTURE
# ==========================================
s3 = prs.slides.add_slide(blank_layout)
create_solid_background(s3, LIGHT_BG)
add_header(s3, "System Architecture", "How the System Works in 5 Simple Steps", "From taking your input to checking live news and showing the real fact in 400 milliseconds.")

add_card(
    s3, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "5-Stage Pipeline Breakdown",
    [
        ("Step 1 - Three Input Options", "Users can type a headline, paste a website link, or upload a photo/screenshot."),
        ("Step 2 - Text Cleaning & Clues", "Cleans messy symbols and counts fake news clues like ALL CAPS words and excessive exclamation marks (!)."),
        ("Step 3 - 3 AI Models Vote", "Logistic Regression, Support Vector Machine, and Naive Bayes vote together for higher accuracy."),
        ("Step 4 - Live Web Check", "Queries Google News RSS, Wikipedia, and Fact-Checking databases in parallel under 500ms."),
        ("Step 5 - Explain & Correct", "Shows why it made the decision and displays the verified real fact if the news was fake.")
    ],
    tag="Pipeline Overview",
    border_color=BLUE_PRIMARY
)

img_s3 = os.path.join(ASSETS_DIR, "diagram_architecture.png")
if os.path.exists(img_s3):
    s3.shapes.add_picture(img_s3, Inches(6.6), Inches(1.85), width=Inches(5.9))

s3.notes_slide.notes_text_frame.text = (
    "This is our overall system architecture. "
    "First, the user provides input via text, a web link, or an image. "
    "Second, the text is cleaned and analyzed for clues like shouting in all caps. "
    "Third, three different machine learning models vote on it. "
    "Fourth, our engine checks live news feeds to see if credible journalists are reporting it. "
    "Finally, it shows the result, explains the clues, and corrects the fact."
)


# ==========================================
# SLIDE 4: COURSE OUTCOME 1 (CO1)
# ==========================================
s4 = prs.slides.add_slide(blank_layout)
create_solid_background(s4, LIGHT_BG)
add_header(s4, "Course Outcome 1 (CO1)", "Data Cleaning & Finding Writing Style Clues", "How raw messy text is turned into clear numbers and clues that AI can understand.")

add_card(
    s4, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Cleaning Text & Extracting Style Clues",
    [
        ("Removing Junk Code", "Strips website links (URLs), HTML tags, and weird symbols that confuse computers."),
        ("Expanding Short Words", "Changes informal contractions like 'can\\'t' to 'cannot' and 'it\\'s' to 'it is'."),
        ("Word Combos (TF-IDF)", "Finds important single words, 2-word pairs ('secret cure'), and 3-word combinations ('shocking truth revealed') that scammers frequently use."),
        ("ALL CAPS Detector", "Measures how much of the text is written in capital letters. Fake news often SHOUTS to get clicks!"),
        ("Exclamation Mark Counter", "Counts too many '!!!' and '???' marks used to trigger emotional panic."),
        ("Vocabulary Richness", "Checks if the article uses rich, professional language or repeats cheap emotional words.")
    ],
    tag="CO1: Data Preparation",
    border_color=BLUE_PRIMARY
)

img_s4 = os.path.join(ASSETS_DIR, "diagram_preprocessing.png")
if os.path.exists(img_s4):
    s4.shapes.add_picture(img_s4, Inches(6.6), Inches(1.85), width=Inches(5.9))

s4.notes_slide.notes_text_frame.text = (
    "Course Outcome 1 covers data cleaning and feature engineering. "
    "Before any AI can read news, the text must be cleaned. We remove URLs, expand contractions, and convert words into TF-IDF scores. "
    "We also extract writing style clues: fake news frequently uses ALL CAPS and multiple exclamation marks to create false urgency. "
    "Our system measures these clues mathematically."
)


# ==========================================
# SLIDE 5: COURSE OUTCOME 2 (CO2)
# ==========================================
s5 = prs.slides.add_slide(blank_layout)
create_solid_background(s5, LIGHT_BG)
add_header(s5, "Course Outcome 2 (CO2)", "3 AI Models Voting Together for 96.4% Accuracy", "Why trusting three different algorithms together is much safer than relying on just one.")

add_card(
    s5, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "The 3 Classifiers & Soft-Voting Engine",
    [
        ("Model 1 - Logistic Regression (93.8%)", "A reliable statistical model that calculates probabilities smoothly."),
        ("Model 2 - Support Vector Machine (95.1%)", "Draws a mathematical boundary line separating real news patterns from fake news patterns."),
        ("Model 3 - Naive Bayes (91.4%)", "Super-fast model that counts how often words appear in real vs fake stories."),
        ("Why Soft-Voting is Better (96.4%)", "Just like a jury in court, if one model gets tricked, the other two models correct the mistake. Their probabilities are combined together."),
        ("Top ROC-AUC Score (0.988)", "Proves the ensemble makes almost zero false positive mistakes across thousands of test articles.")
    ],
    tag="CO2: Model Building",
    border_color=EMERALD
)

img_s5 = os.path.join(ASSETS_DIR, "diagram_model_comparison.png")
if os.path.exists(img_s5):
    s5.shapes.add_picture(img_s5, Inches(6.6), Inches(1.85), width=Inches(5.9))

s5.notes_slide.notes_text_frame.text = (
    "Course Outcome 2 covers model selection and training. "
    "Instead of betting on a single algorithm, we built a soft-voting ensemble of three distinct models: Logistic Regression, Linear SVM, and Naive Bayes. "
    "As the chart shows, while individual models achieve 91% to 95%, combining them in an ensemble reaches 96.4% accuracy and a 0.988 ROC score."
)


# ==========================================
# SLIDE 6: COURSE OUTCOME 3 (CO3)
# ==========================================
s6 = prs.slides.add_slide(blank_layout)
create_solid_background(s6, LIGHT_BG)
add_header(s6, "Course Outcome 3 (CO3)", "Live Web Checking: Finding Today's News in 400ms", "Checking live Google News, Wikipedia, and Fact-Checkers to confirm real events.")

add_card(
    s6, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Smart Web Search & Word Matching",
    [
        ("Fast Parallel Search", "Queries Google News, Wikipedia, and Google Fact Check at the exact same time in under 500ms."),
        ("Regional Spelling Solver", "Indian places often have multiple spellings. If user types 'Kanyakumari' and official news writes 'Kanniyakumari', our system matches them with 91.6% similarity!"),
        ("Different Words Solver", "If user types 'holiday for schools' and news writes 'schools shut' or 'schools closed', our synonym table matches them automatically."),
        ("Wikipedia Knowledge Check", "Instantly checks who the real president, prime minister, or chief minister is to catch political lies."),
        ("Reliable News Verification", "Verifies if trusted agencies (The Hindu, Hindustan Times, NDTV) are actually reporting the event.")
    ],
    tag="CO3: Web Grounding",
    border_color=BLUE_PRIMARY
)

img_s6 = os.path.join(ASSETS_DIR, "diagram_web_verifier.png")
if os.path.exists(img_s6):
    s6.shapes.add_picture(img_s6, Inches(6.6), Inches(1.85), width=Inches(5.9))

s6.notes_slide.notes_text_frame.text = (
    "Course Outcome 3 is our live web verification engine. "
    "It queries Google News RSS and Wikipedia in parallel. "
    "A key engineering feature we solved is regional transliteration: local district names like Kanyakumari are often spelled Kanniyakumari in official newspapers. "
    "Our fuzzy matcher and synonym engine connect them seamlessly, so real breaking news is never falsely rejected."
)


# ==========================================
# SLIDE 7: COURSE OUTCOME 4 (CO4)
# ==========================================
s7 = prs.slides.add_slide(blank_layout)
create_solid_background(s7, LIGHT_BG)
add_header(s7, "Course Outcome 4 (CO4)", "Explainable AI & Showing the Real Truth", "Opening the black box: Showing why it is fake and displaying what is actually true.")

add_card(
    s7, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Explaining Decisions & Correcting Facts",
    [
        ("No Black Box (Word Highlights)", "Highlights the exact words that influenced the AI. Sensational words like 'shocking' appear in red; neutral facts appear in green."),
        ("Confidence Percentage", "Shows how confident the AI is (e.g. 98.2% sure) with an easy-to-read meter."),
        ("Catching Famous Rumors", "Detects well-known social media hoaxes like '15 days of total darkness' or fake cures immediately."),
        ("Automated Fact Correction", "When someone posts a false claim like 'Actor Vijay is the Chief Minister of Tamil Nadu', the app doesn't just say 'Fake'."),
        ("Displays the True Fact", "Renders a prominent green box showing: 'The official Chief Minister is M. K. Stalin', directly backed by Wikipedia!")
    ],
    tag="CO4: Transparency & Truth",
    border_color=EMERALD
)

img_s7 = os.path.join(ASSETS_DIR, "diagram_xai.png")
if os.path.exists(img_s7):
    s7.shapes.add_picture(img_s7, Inches(6.6), Inches(1.85), width=Inches(5.9))

s7.notes_slide.notes_text_frame.text = (
    "Course Outcome 4 focuses on Explainable AI and fact correction. "
    "Users hate AI that just gives a number without explaining why. Our app highlights the specific words that triggered the decision. "
    "Even better, if a claim is fake, the app doesn't leave the user in the dark. "
    "It actively looks up the verified fact—like the true chief minister or tournament winner—and displays it in a clear green callout box."
)


# ==========================================
# SLIDE 8: COURSE OUTCOME 5 (CO5)
# ==========================================
s8 = prs.slides.add_slide(blank_layout)
create_solid_background(s8, LIGHT_BG)
add_header(s8, "Course Outcome 5 (CO5)", "System Performance, Super-Fast Cache & Hosting", "How we made the app lightning fast, reliable, and ready to host on the cloud.")

add_card(
    s8, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Testing, Speedup & Production Setup",
    [
        ("Tested on Thousands of Articles", "Achieved 96.4% Accuracy, 96.1% Precision, and 96.8% Recall across balanced real and fake news datasets."),
        ("Super-Fast Memory Cache (250x Faster)", "When a trending rumor is checked once, the result is saved in memory using SHA-256 fingerprinting."),
        ("1.8 Millisecond Speed", "The next time anyone asks about the same viral news, the answer is returned instantly in 1.8 milliseconds!"),
        ("Flask Web Server", "Clean, lightweight Python backend that runs smoothly on any laptop or server."),
        ("38 Automated Tests Passing", "Includes a complete test suite verifying every single part of the pipeline before deployment."),
        ("Ready to Host Anywhere", "Easily deployed on Render, Docker, or any standard cloud service.")
    ],
    tag="CO5: Production Quality",
    border_color=BLUE_PRIMARY
)

img_s8 = os.path.join(ASSETS_DIR, "diagram_caching.png")
if os.path.exists(img_s8):
    s8.shapes.add_picture(img_s8, Inches(6.6), Inches(1.85), width=Inches(5.9))

s8.notes_slide.notes_text_frame.text = (
    "Course Outcome 5 covers performance evaluation, caching, and deployment. "
    "Our model achieved a 0.965 F1-score across thousands of test articles. "
    "To handle high user traffic during viral events, we built an in-memory cache: repeat queries take only 1.8 milliseconds instead of 450 milliseconds. "
    "The application is fully covered by 38 automated unit tests and is ready for production cloud hosting."
)


# ==========================================
# SLIDE 9: OTHER TECH (EasyOCR & Web Scraping)
# ==========================================
s9 = prs.slides.add_slide(blank_layout)
create_solid_background(s9, LIGHT_BG)
add_header(s9, "Integrated Technologies", "Reading Photos with EasyOCR & Scraping Web Links", "Checking fake news from WhatsApp screenshots, social media images, and news URLs.")

add_card(
    s9, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Computer Vision & URL Extraction",
    [
        ("Deep Learning OCR (EasyOCR)", "Uses PyTorch neural networks to read text inside photos, screenshots, and paper clippings."),
        ("CRAFT Text Finder", "Finds text even if the screenshot is blurry, rotated, or taken on a dark phone screen."),
        ("WhatsApp Forward Ready", "People rarely type text; they forward screenshots! Users can drag and drop any image to check it instantly."),
        ("Automatic Web Scraping (BeautifulSoup4)", "Users can paste a news website URL. The system visits the site, removes ads and menus, and reads the main story body automatically."),
        ("PDF Report Exporter", "Generates a clean fact-checking audit sheet with source citations that users can download and share.")
    ],
    tag="Multi-Modal Capabilities",
    border_color=EMERALD
)

img_s9 = os.path.join(ASSETS_DIR, "diagram_ocr.png")
if os.path.exists(img_s9):
    s9.shapes.add_picture(img_s9, Inches(6.6), Inches(1.85), width=Inches(5.9))

s9.notes_slide.notes_text_frame.text = (
    "Slide 9 highlights other key technologies we incorporated. "
    "Most fake news on WhatsApp circulates as screenshot images. We integrated EasyOCR, which uses deep learning to extract text from images. "
    "We also added BeautifulSoup for automatic web link scraping and an audit report generator that creates downloadable fact-checking summaries."
)


# ==========================================
# SLIDE 10: REAL CASE STUDY
# ==========================================
s10 = prs.slides.add_slide(blank_layout)
create_solid_background(s10, LIGHT_BG)
add_header(s10, "Empirical Validation", "Real-World Test: Today's Heavy Rain School Holiday", "A live demonstration proving why our system works where traditional AI fails.")

add_card(
    s10, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.1),
    "Real Breaking News Case Study",
    [
        ("The Test News", "'Kanyakumari collector declares holiday for schools due to heavy rain today.'"),
        ("What Old AI Said", "FAKE NEWS (74.1% probability) with warning 'zero matching records in training data'."),
        ("Why Old AI Failed", "Because the rain holiday was declared this morning! Old datasets know nothing about today."),
        ("What Our System Did", "CONFIRMED REAL NEWS (98.2% Confidence)!"),
        ("Found Live Hindustan Times Report", "'Heavy rain likely in Tamil Nadu: Schools shut in Kanniyakumari and Tenkasi'."),
        ("Solved the Spelling Gap", "Recognized that 'Kanyakumari' is the exact same district as 'Kanniyakumari'."),
        ("Solved the Action Gap", "Recognized that 'holiday' means 'schools shut'.")
    ],
    tag="Live Test Results",
    border_color=EMERALD
)

img_s10 = os.path.join(ASSETS_DIR, "diagram_case_study.png")
if os.path.exists(img_s10):
    s10.shapes.add_picture(img_s10, Inches(6.6), Inches(1.85), width=Inches(5.9))

s10.notes_slide.notes_text_frame.text = (
    "Slide 10 showcases a live case study conducted on today's breaking news. "
    "A school holiday was declared in Kanyakumari district due to rain. "
    "A static ML model failed and called it fake news. "
    "Our hybrid system searched live wires, matched the spelling and synonym variations, and confirmed it as 98.2% REAL news, citing Hindustan Times."
)


# ==========================================
# SLIDE 11: CONCLUSION & THANK YOU
# ==========================================
s11 = prs.slides.add_slide(blank_layout)
create_solid_background(s11, NAVY_BG)

# Title Accent Badge
badge11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.666), Inches(0.9), Inches(4.0), Inches(0.4))
badge11.fill.solid()
badge11.fill.fore_color.rgb = EMERALD
badge11.line.fill.background()
tf_b11 = badge11.text_frame
tf_b11.margin_left = tf_b11.margin_right = tf_b11.margin_top = tf_b11.margin_bottom = 0
p_b11 = tf_b11.paragraphs[0]
p_b11.text = "PROJECT SUMMARY & ROADMAP"
p_b11.alignment = PP_ALIGN.CENTER
p_b11.font.name = FONT_HEADING
p_b11.font.size = Pt(10.5)
p_b11.font.bold = True
p_b11.font.color.rgb = WHITE

# Thank you banner
tb11 = s11.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(1.2))
tf11 = tb11.text_frame
tf11.word_wrap = True
tf11.margin_left = tf11.margin_right = tf11.margin_top = tf11.margin_bottom = 0
p11 = tf11.paragraphs[0]
p11.text = "Thank You!"
p11.alignment = PP_ALIGN.CENTER
p11.font.name = FONT_HEADING
p11.font.size = Pt(44)
p11.font.bold = True
p11.font.color.rgb = WHITE

p11_sub = tf11.add_paragraph()
p11_sub.text = "Open for Questions, Evaluation & Feedback"
p11_sub.alignment = PP_ALIGN.CENTER
p11_sub.font.name = FONT_BODY
p11_sub.font.size = Pt(17)
p11_sub.font.color.rgb = CYAN_ACCENT
p11_sub.space_before = Pt(4)

# Summary & Future Scope Cards
add_card(
    s11, Inches(1.2), Inches(3.0), Inches(5.2), Inches(3.8),
    "What We Built (Key Achievements)",
    [
        ("3-Model AI Voting", "Combined Logistic Regression, SVM, and Naive Bayes to achieve 96.4% test accuracy."),
        ("Live Web Search in 400ms", "Eliminated breaking news mistakes by checking Google News and Wikipedia in real time."),
        ("Multi-Modal Support", "Takes text, pastes web links, and reads screenshot images using EasyOCR."),
        ("Explains Every Decision", "Highlights suspicious words and displays the real true fact if the news was fake."),
        ("Production Speed", "Smart memory caching gives answers in 1.8ms on viral queries.")
    ],
    tag="Project Summary",
    dark_mode=True
)

add_card(
    s11, Inches(6.933), Inches(3.0), Inches(5.2), Inches(3.8),
    "Future Enhancements (Roadmap)",
    [
        ("Regional Indian Languages", "Expand fact-checking to Tamil, Hindi, Telugu, and Kannada voice messages."),
        ("Deepfake Video Detection", "Add computer vision to spot manipulated face-swap videos and altered audio clips."),
        ("Browser Extension", "Create a Chrome extension that automatically highlights fake articles while you browse the web."),
        ("Mobile App Release", "Package the application for Android and iOS for quick WhatsApp forward checks.")
    ],
    tag="Future Roadmap",
    dark_mode=True
)

s11.notes_slide.notes_text_frame.text = (
    "To conclude, our project bridges the biggest gap in modern fake news detection. "
    "By combining three machine learning models with live web checking, image reading OCR, and clear explanations, "
    "we built a tool that is fast, accurate, and trustworthy. "
    "Thank you for your time, and we now welcome any questions or feedback from the committee."
)

prs.save(OUTPUT_PATH)
print(f"Presentation successfully updated with beginner words and embedded diagrams at: {OUTPUT_PATH}")
