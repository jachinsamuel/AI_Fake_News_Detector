"""
Generate custom, clean, high-resolution diagram images for the presentation slides.
All diagrams use simple, beginner-friendly visual cues and modern aesthetics.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slides_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'


def create_diagram_problem():
    """Slide 2: Visual comparison of Old ML vs Our System."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    # Card 1: Traditional ML (Fails)
    rect1 = patches.FancyBboxPatch((0.5, 5.2), 9, 4.2, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor='#FFFFFF', edgecolor='#EF4444', linewidth=2)
    ax.add_patch(rect1)

    ax.text(0.9, 8.8, "TRADITIONAL STATIC ML", fontsize=12, fontweight='bold', color='#EF4444')
    ax.text(0.9, 8.2, "Input: 'Schools closed today in Kanyakumari due to heavy rain'", fontsize=10.5, color='#0F172A', style='italic')
    
    # Process box
    p1 = patches.FancyBboxPatch((1.0, 6.0), 3.5, 1.6, boxstyle="round,pad=0.1", facecolor='#FEE2E2', edgecolor='#EF4444')
    ax.add_patch(p1)
    ax.text(2.75, 7.0, "Model Trained in 2021", fontsize=10, fontweight='bold', color='#991B1B', ha='center')
    ax.text(2.75, 6.4, "Has never seen today's event", fontsize=9, color='#7F1D1D', ha='center')

    # Arrow
    ax.annotate('', xy=(5.5, 6.8), xytext=(4.7, 6.8),
                arrowprops=dict(arrowstyle="->", color='#EF4444', lw=2))

    # Result box
    r1 = patches.FancyBboxPatch((5.7, 5.8), 3.4, 1.8, boxstyle="round,pad=0.1", facecolor='#FEE2E2', edgecolor='#B91C1C')
    ax.add_patch(r1)
    ax.text(7.4, 7.0, "FLAGS AS FAKE (74%)", fontsize=11, fontweight='bold', color='#B91C1C', ha='center')
    ax.text(7.4, 6.3, "False Negative! Old models\ncannot know breaking news.", fontsize=8.5, color='#7F1D1D', ha='center')

    # Card 2: Our Hybrid System (Succeeds)
    rect2 = patches.FancyBboxPatch((0.5, 0.4), 9, 4.4, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor='#FFFFFF', edgecolor='#10B981', linewidth=2)
    ax.add_patch(rect2)

    ax.text(0.9, 4.2, "OUR HYBRID SYSTEM (ML + LIVE WEB SEARCH)", fontsize=12, fontweight='bold', color='#059669')
    ax.text(0.9, 3.6, "Input: 'Schools closed today in Kanyakumari due to heavy rain'", fontsize=10.5, color='#0F172A', style='italic')

    # Dual process boxes
    p2a = patches.FancyBboxPatch((1.0, 1.2), 3.5, 1.8, boxstyle="round,pad=0.1", facecolor='#D1FAE5', edgecolor='#10B981')
    ax.add_patch(p2a)
    ax.text(2.75, 2.3, "ML Voting + Live Search", fontsize=10, fontweight='bold', color='#065F46', ha='center')
    ax.text(2.75, 1.6, "Finds live Hindustan Times wire\n& Google News reports today", fontsize=8.5, color='#047857', ha='center')

    # Arrow
    ax.annotate('', xy=(5.5, 2.1), xytext=(4.7, 2.1),
                arrowprops=dict(arrowstyle="->", color='#10B981', lw=2))

    # Result box
    r2 = patches.FancyBboxPatch((5.7, 1.0), 3.4, 2.0, boxstyle="round,pad=0.1", facecolor='#D1FAE5', edgecolor='#047857')
    ax.add_patch(r2)
    ax.text(7.4, 2.3, "CONFIRMED REAL (98.2%)", fontsize=11, fontweight='bold', color='#065F46', ha='center')
    ax.text(7.4, 1.5, "Live Corroboration with\nverified news links cited!", fontsize=8.5, color='#047857', ha='center')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_problem.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_architecture():
    """Slide 3: End-to-End System Flowchart."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    steps = [
        ("1. Input", "Text, Web URL,\nor Screenshot Image", "#3B82F6", "#DBEAFE", 0.6),
        ("2. Clean & Extract", "Remove noise, get\nTF-IDF & ALL CAPS", "#8B5CF6", "#EDE9FE", 2.5),
        ("3. 3-Model Voting", "Logistic + SVM +\nNaive Bayes vote", "#0284C7", "#E0F2FE", 4.4),
        ("4. Live Web Check", "Google News RSS &\nWikipedia Knowledge", "#F59E0B", "#FEF3C7", 6.3),
        ("5. Final Answer", "Real/Fake score +\nTrue Fact + citations", "#10B981", "#D1FAE5", 8.2)
    ]

    for title, desc, border, fill, x in steps:
        box = patches.FancyBboxPatch((x, 3.2), 1.6, 3.8, boxstyle="round,pad=0.15,rounding_size=0.2",
                                    facecolor=fill, edgecolor=border, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.8, 6.3, title, fontsize=10, fontweight='bold', color=border, ha='center')
        ax.text(x + 0.8, 4.7, desc, fontsize=8.5, color='#1E293B', ha='center', va='center')

        # Add connector arrow
        if x < 8.0:
            ax.annotate('', xy=(x + 1.85, 5.1), xytext=(x + 1.62, 5.1),
                        arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=2, mutation_scale=12))

    # Bottom summary badge
    badge = patches.FancyBboxPatch((1.5, 0.8), 7.0, 1.6, boxstyle="round,pad=0.1",
                                  facecolor='#0F172A', edgecolor='#334155')
    ax.add_patch(badge)
    ax.text(5.0, 1.8, "TOTAL RESPONSE TIME: UNDER 500 MILLISECONDS", fontsize=11, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(5.0, 1.2, "Dual Verification: Fast AI prediction + Live news facts checked together.", fontsize=9, color='#E2E8F0', ha='center')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_architecture.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_preprocessing():
    """Slide 4 (CO1): Text cleaning & feature pipeline."""
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    # Raw Text Box
    b1 = patches.FancyBboxPatch((0.5, 6.5), 4.2, 2.8, boxstyle="round,pad=0.1", facecolor='#FEE2E2', edgecolor='#EF4444', lw=1.5)
    ax.add_patch(b1)
    ax.text(0.8, 8.8, "RAW UNSTRUCTURED TEXT", fontsize=10.5, fontweight='bold', color='#991B1B')
    ax.text(0.8, 8.0, "'SHOCKING!! Click here: http://bit.ly\nAliens landed in Delhi TODAY!!!'", fontsize=9, color='#1E293B', style='italic')
    ax.text(0.8, 7.0, "Problems: Links, ALL CAPS, noise,\ntoo many exclamation marks.", fontsize=8.5, color='#7F1D1D')

    # Arrow Down
    ax.annotate('', xy=(2.6, 5.4), xytext=(2.6, 6.3),
                arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=2.5, mutation_scale=15))
    ax.text(2.8, 5.8, "Text Cleaning", fontsize=9.5, fontweight='bold', color='#2563EB')

    # Clean Text Box
    b2 = patches.FancyBboxPatch((0.5, 1.0), 4.2, 4.0, boxstyle="round,pad=0.1", facecolor='#DBEAFE', edgecolor='#2563EB', lw=1.5)
    ax.add_patch(b2)
    ax.text(0.8, 4.4, "CLEANED & NORMALIZED TEXT", fontsize=10.5, fontweight='bold', color='#1E40AF')
    ax.text(0.8, 3.6, "1. Remove URLs, tags, and symbols\n2. Expand contractions (can't -> cannot)\n3. Lowercase words for dictionary match\n4. Remove temporal noise ('today', 'breaking')", fontsize=8.5, color='#1E3A8A')
    ax.text(0.8, 1.6, "Result: 'aliens landed in delhi'", fontsize=9.5, fontweight='bold', color='#1E40AF', style='italic')

    # Arrow Right
    ax.annotate('', xy=(5.2, 3.0), xytext=(4.8, 3.0),
                arrowprops=dict(arrowstyle="-|>", color='#10B981', lw=2.5, mutation_scale=15))

    # Feature Extraction Box
    b3 = patches.FancyBboxPatch((5.4, 1.0), 4.1, 8.3, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#10B981', lw=2)
    ax.add_patch(b3)
    ax.text(5.7, 8.7, "FEATURE EXTRACTION (CLUES)", fontsize=11, fontweight='bold', color='#065F46')

    features = [
        ("TF-IDF Word Tokens", "Finds important 1-word, 2-word,\nand 3-word combinations.", "#D1FAE5"),
        ("ALL CAPS Ratio", "Measures shouting in text\n(Heavy signal of fake news).", "#FEF3C7"),
        ("Exclamation Mark Count", "Counts ! and ? marks showing\nsensational emotional push.", "#FEE2E2"),
        ("Word Richness (TTR)", "Distinguishes real news writing\nfrom cheap clickbait repetition.", "#E0F2FE")
    ]

    y_pos = 7.0
    for title, desc, col in features:
        fb = patches.FancyBboxPatch((5.6, y_pos - 0.2), 3.7, 1.4, boxstyle="round,pad=0.08", facecolor=col, edgecolor='#94A3B8', lw=1)
        ax.add_patch(fb)
        ax.text(5.8, y_pos + 0.8, title, fontsize=9.5, fontweight='bold', color='#0F172A')
        ax.text(5.8, y_pos + 0.1, desc, fontsize=8, color='#334155')
        y_pos -= 1.8

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_preprocessing.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_model_comparison():
    """Slide 5 (CO2): Model Comparison Bar Chart."""
    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=200)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#FFFFFF')

    models = ['Multinomial\nNaive Bayes', 'Logistic\nRegression', 'Linear SVM\n(Calibrated)', 'Soft-Voting\nEnsemble']
    accuracy = [91.4, 93.8, 95.1, 96.4]
    f1_scores = [91.2, 93.5, 94.9, 96.5]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#3B82F6', edgecolor='#1D4ED8')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#10B981', edgecolor='#047857')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold', color='#0F172A')
    ax.set_title('Model Performance Comparison (Test Dataset)', fontsize=13, fontweight='bold', color='#0F172A', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold', color='#1E293B')
    ax.set_ylim(85, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add data labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1D4ED8')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_model_comparison.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_web_verifier():
    """Slide 6 (CO3): Live Web Search & Semantic Matcher."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    # Card 1: Spelling & Synonyms Solver
    c1 = patches.FancyBboxPatch((0.5, 5.2), 9.0, 4.2, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#2563EB', lw=1.8)
    ax.add_patch(c1)
    ax.text(0.8, 8.8, "STEP 1: SMART QUERY & SYNONYM MATCHING", fontsize=11, fontweight='bold', color='#2563EB')
    
    # Subbox A: Transliteration
    sa = patches.FancyBboxPatch((0.8, 5.7), 4.1, 2.7, boxstyle="round,pad=0.1", facecolor='#EFF6FF', edgecolor='#93C5FD')
    ax.add_patch(sa)
    ax.text(1.0, 7.8, "Regional Spelling Matcher", fontsize=10, fontweight='bold', color='#1E40AF')
    ax.text(1.0, 6.2, "User types: 'Kanyakumari'\nNews spells: 'Kanniyakumari'\nMatch Score: 91.6% (Recognized as SAME!)", fontsize=8.5, color='#1E3A8A')

    # Subbox B: Synonyms
    sb = patches.FancyBboxPatch((5.1, 5.7), 4.1, 2.7, boxstyle="round,pad=0.1", facecolor='#EFF6FF', edgecolor='#93C5FD')
    ax.add_patch(sb)
    ax.text(5.3, 7.8, "Action Synonym Matcher", fontsize=10, fontweight='bold', color='#1E40AF')
    ax.text(5.3, 6.2, "User types: 'holiday for schools'\nNews spells: 'schools shut / closed'\nSynonym Table: Exact match recognized!", fontsize=8.5, color='#1E3A8A')

    # Card 2: 4 Live Sources Checked Simultaneously
    c2 = patches.FancyBboxPatch((0.5, 0.4), 9.0, 4.4, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#10B981', lw=1.8)
    ax.add_patch(c2)
    ax.text(0.8, 4.2, "STEP 2: 4 SOURCES QUERIED IN PARALLEL (< 500ms)", fontsize=11, fontweight='bold', color='#059669')

    sources = [
        ("Google News RSS", "Live headlines from global & regional news wires", 0.8),
        ("Wikipedia REST API", "Checks leaders, prime ministers, & capitals", 2.9),
        ("Google Fact Check API", "Checks if Snopes or BoomLive debunked it", 5.0),
        ("GNews / NewsAPI", "Backup live news streams for breaking events", 7.1)
    ]

    for title, desc, x in sources:
        box = patches.FancyBboxPatch((x, 1.0), 1.9, 2.7, boxstyle="round,pad=0.08", facecolor='#ECFDF5', edgecolor='#10B981', lw=1)
        ax.add_patch(box)
        ax.text(x + 0.95, 3.1, title, fontsize=8.5, fontweight='bold', color='#065F46', ha='center')
        ax.text(x + 0.95, 1.9, desc, fontsize=7.5, color='#047857', ha='center')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_web_verifier.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_xai():
    """Slide 7 (CO4): Explainable AI & Truth Correction."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    # Card 1: Feature Attribution (Explainable AI)
    c1 = patches.FancyBboxPatch((0.5, 5.2), 9.0, 4.3, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#2563EB', lw=1.8)
    ax.add_patch(c1)
    ax.text(0.8, 8.9, "1. EXPLAINABLE AI: SHOWING EXACT WORD CLUES", fontsize=11, fontweight='bold', color='#2563EB')
    
    ax.text(0.8, 8.2, "Headline Checked: 'Shocking secret cure for diabetes announced by doctor'", fontsize=9.5, color='#0F172A', style='italic')

    # Word Pills
    words = [
        ("shocking", "-8.4 (Fake Clue)", "#FEE2E2", "#991B1B"),
        ("secret cure", "-9.2 (Fake Clue)", "#FEE2E2", "#991B1B"),
        ("diabetes", "+0.2 (Topic)", "#E2E8F0", "#334155"),
        ("announced", "+2.5 (Journalistic)", "#D1FAE5", "#065F46")
    ]
    x_w = 0.8
    for w, val, fill, txt_c in words:
        wp = patches.FancyBboxPatch((x_w, 6.2), 1.9, 1.4, boxstyle="round,pad=0.08", facecolor=fill, edgecolor=txt_c, lw=1)
        ax.add_patch(wp)
        ax.text(x_w + 0.95, 7.1, f"'{w}'", fontsize=9.5, fontweight='bold', color=txt_c, ha='center')
        ax.text(x_w + 0.95, 6.5, val, fontsize=8, color=txt_c, ha='center')
        x_w += 2.1

    # Card 2: Truth Box (Fact Correction)
    c2 = patches.FancyBboxPatch((0.5, 0.5), 9.0, 4.3, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#10B981', lw=2)
    ax.add_patch(c2)
    ax.text(0.8, 4.2, "2. AUTOMATED FACT CORRECTION (WHAT IS ACTUALLY TRUE?)", fontsize=11, fontweight='bold', color='#059669')
    
    # False claim callout
    fc = patches.FancyBboxPatch((0.8, 2.5), 4.1, 1.3, boxstyle="round,pad=0.08", facecolor='#FEE2E2', edgecolor='#EF4444')
    ax.add_patch(fc)
    ax.text(1.0, 3.2, "User Claim: 'Vijay is the Chief Minister of Tamil Nadu'", fontsize=8, color='#991B1B')
    ax.text(1.0, 2.7, "AI Verdict: FAKE NEWS (98.5%)", fontsize=8.5, fontweight='bold', color='#B91C1C')

    # Arrow
    ax.annotate('', xy=(5.5, 3.1), xytext=(5.0, 3.1),
                arrowprops=dict(arrowstyle="-|>", color='#10B981', lw=2))

    # Real Fact Green Box
    rc = patches.FancyBboxPatch((5.7, 1.2), 3.5, 2.6, boxstyle="round,pad=0.1", facecolor='#ECFDF5', edgecolor='#059669', lw=1.5)
    ax.add_patch(rc)
    ax.text(5.9, 3.3, "VERIFIED GROUND TRUTH:", fontsize=9.5, fontweight='bold', color='#065F46')
    ax.text(5.9, 2.1, "The official Chief Minister of Tamil\nNadu is M. K. Stalin. C. Joseph Vijay\nis an Indian actor and leader of TVK.", fontsize=8.5, color='#047857')
    ax.text(5.9, 1.4, "[Verified via Wikipedia Knowledge Graph]", fontsize=7.5, color='#059669', style='italic')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_xai.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_caching():
    """Slide 8 (CO5): Performance & Cache Speedup."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.8), dpi=200)
    fig.patch.set_facecolor('#F8FAFC')

    # Chart 1: Latency Speedup
    ax1.set_facecolor('#FFFFFF')
    bars = ['First Query\n(Live Web)', 'Cached Query\n(Instant)']
    latencies = [450, 1.8]
    colors = ['#3B82F6', '#10B981']

    b = ax1.bar(bars, latencies, color=colors, edgecolor='#1E293B', width=0.5)
    ax1.set_ylabel('Response Time (Milliseconds)', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax1.set_title('Query Latency Speedup', fontsize=11, fontweight='bold', color='#0F172A', pad=10)
    ax1.set_ylim(0, 520)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    ax1.text(0, 465, '450 ms', ha='center', fontsize=10, fontweight='bold', color='#1D4ED8')
    ax1.text(1, 15, '1.8 ms\n(250x Faster!)', ha='center', fontsize=9.5, fontweight='bold', color='#047857')

    # Chart 2: Test Metrics
    ax2.set_facecolor('#FFFFFF')
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    scores = [96.4, 96.1, 96.8, 96.5, 98.8]

    y_pos = np.arange(len(metrics))
    b2 = ax2.barh(y_pos, scores, color='#0284C7', edgecolor='#0369A1', height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=9.5, fontweight='bold', color='#1E293B')
    ax2.set_xlim(85, 102)
    ax2.set_xlabel('Score (%)', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax2.set_title('Production Evaluation Metrics', fontsize=11, fontweight='bold', color='#0F172A', pad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)

    for bar in b2:
        width = bar.get_width()
        ax2.text(width + 0.6, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                 va='center', fontsize=9, fontweight='bold', color='#0369A1')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_caching.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_ocr():
    """Slide 9: OCR Multi-Modal Pipeline."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    steps = [
        ("Step 1: Upload Image", "User drags & drops\nWhatsApp screenshot,\nfake headline, or photo", "#3B82F6", "#DBEAFE", 0.6),
        ("Step 2: EasyOCR Detector", "CRAFT deep network\nfinds all text regions\n(even rotated/blurry)", "#8B5CF6", "#EDE9FE", 3.7),
        ("Step 3: Text Extracted", "ResNet-LSTM transcribes\nclean words & sends to\nFake News Detector", "#10B981", "#D1FAE5", 6.8)
    ]

    for title, desc, border, fill, x in steps:
        box = patches.FancyBboxPatch((x, 3.2), 2.5, 4.5, boxstyle="round,pad=0.15", facecolor=fill, edgecolor=border, lw=2)
        ax.add_patch(box)
        ax.text(x + 1.25, 7.0, title, fontsize=10.5, fontweight='bold', color=border, ha='center')
        ax.text(x + 1.25, 5.0, desc, fontsize=9, color='#1E293B', ha='center', va='center')

        if x < 6.0:
            ax.annotate('', xy=(x + 2.9, 5.4), xytext=(x + 2.6, 5.4),
                        arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=2.5, mutation_scale=14))

    # Bottom Callout: Web Scraping Included
    callout = patches.FancyBboxPatch((0.6, 0.8), 8.7, 1.8, boxstyle="round,pad=0.1", facecolor='#FFFFFF', edgecolor='#0284C7', lw=1.5)
    ax.add_patch(callout)
    ax.text(1.0, 1.9, "BONUS: AUTOMATED URL SCRAPING (BeautifulSoup4)", fontsize=10, fontweight='bold', color='#0284C7')
    ax.text(1.0, 1.2, "User can also paste any web link (e.g., news article or blog). The system scrapes the article body automatically.", fontsize=8.5, color='#334155')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_ocr.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def create_diagram_case_study():
    """Slide 10: Real Case Study Comparison."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')

    # Claim Banner
    banner = patches.FancyBboxPatch((0.5, 8.0), 9.0, 1.6, boxstyle="round,pad=0.1", facecolor='#0F172A', edgecolor='#334155')
    ax.add_patch(banner)
    ax.text(5.0, 9.0, "BREAKING NEWS CASE STUDY (OCTOBER 2026)", fontsize=10.5, fontweight='bold', color='#38BDF8', ha='center')
    ax.text(5.0, 8.3, "'Kanyakumari collector declares holiday for schools due to heavy rain today.'", fontsize=9.5, color='#FFFFFF', style='italic', ha='center')

    # Column 1: Without Web Verification
    c1 = patches.FancyBboxPatch((0.5, 1.0), 4.3, 6.5, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#EF4444', lw=2)
    ax.add_patch(c1)
    ax.text(0.8, 6.8, "WITHOUT LIVE WEB SEARCH", fontsize=11, fontweight='bold', color='#EF4444')
    ax.text(0.8, 6.1, "Static ML Model only:", fontsize=9, color='#64748B')
    
    # Red badge
    rb = patches.FancyBboxPatch((0.8, 4.6), 3.7, 1.2, boxstyle="round,pad=0.08", facecolor='#FEE2E2', edgecolor='#EF4444')
    ax.add_patch(rb)
    ax.text(2.65, 5.2, "PREDICTED: FAKE (74%)", fontsize=10, fontweight='bold', color='#B91C1C', ha='center')
    ax.text(2.65, 4.8, "'Zero matching articles found in old dataset'", fontsize=7.5, color='#7F1D1D', ha='center')

    ax.text(0.8, 3.8, "Why did it fail?", fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(0.8, 2.0, "1. Old training dataset has never\n   seen this rain event.\n2. Official news wrote 'Kanniyakumari'\n   (with double 'n' and 'i').\n3. News headlines said 'schools shut'\n   instead of 'holiday'.", fontsize=8.5, color='#334155')

    # Column 2: With Our Live Verification
    c2 = patches.FancyBboxPatch((5.2, 1.0), 4.3, 6.5, boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#10B981', lw=2)
    ax.add_patch(c2)
    ax.text(5.5, 6.8, "WITH OUR HYBRID VERIFIER", fontsize=11, fontweight='bold', color='#059669')
    ax.text(5.5, 6.1, "ML + Parallel Web Intelligence:", fontsize=9, color='#64748B')

    # Green badge
    gb = patches.FancyBboxPatch((5.5, 4.6), 3.7, 1.2, boxstyle="round,pad=0.08", facecolor='#D1FAE5', edgecolor='#10B981')
    ax.add_patch(gb)
    ax.text(7.35, 5.2, "PREDICTED: REAL (98.2%)", fontsize=10, fontweight='bold', color='#065F46', ha='center')
    ax.text(7.35, 4.8, "Corroborated by Hindustan Times wire", fontsize=7.5, color='#047857', ha='center')

    ax.text(5.5, 3.8, "Why did it succeed?", fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.text(5.5, 2.0, "1. Transliteration table matched\n   Kanyakumari <-> Kanniyakumari.\n2. Synonym table matched\n   holiday <-> schools shut.\n3. Found live Hindustan Times report\n   published just 2 hours ago!", fontsize=8.5, color='#334155')

    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "diagram_case_study.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    create_diagram_problem()
    create_diagram_architecture()
    create_diagram_preprocessing()
    create_diagram_model_comparison()
    create_diagram_web_verifier()
    create_diagram_xai()
    create_diagram_caching()
    create_diagram_ocr()
    create_diagram_case_study()
    print("All diagram images successfully generated!")
