/**
 * News Authenticity Analyzer — Client Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // Inputs & Telemetry
    const newsInput = document.getElementById("news-input");
    const wordCountSpan = document.getElementById("word-count");
    const charCountSpan = document.getElementById("char-count");
    const readingTimeSpan = document.getElementById("reading-time");
    
    // Actions & Buttons
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const spinner = document.getElementById("loading-spinner");
    const btnText = analyzeBtn.querySelector(".btn-text");
    const errorBanner = document.getElementById("error-banner");
    const errorMessage = document.getElementById("error-message");

    // Results Elements
    const resultCard = document.getElementById("result-card");
    const resultBox = resultCard.querySelector(".result-box");
    const verdictTag = document.getElementById("verdict-tag");
    const verdictHeading = document.getElementById("verdict-heading");
    const confidenceVal = document.getElementById("confidence-val");
    const confidenceBar = document.getElementById("confidence-bar");
    const featureTagsContainer = document.getElementById("feature-tags-container");
    const explanationText = document.getElementById("explanation-text");
    const statModel = document.getElementById("stat-model");
    const statLatency = document.getElementById("stat-latency");
    const statTokens = document.getElementById("stat-tokens");

    let sampleArticles = [];

    // 1. Text Counter
    function updateTextStats() {
        const text = newsInput.value.trim();
        const chars = text.length;
        const words = text ? text.split(/\s+/).filter(w => w.length > 0).length : 0;
        const readMins = Math.max(1, Math.ceil(words / 200));

        charCountSpan.textContent = `${chars.toLocaleString()} characters`;
        wordCountSpan.textContent = `${words.toLocaleString()} words`;
        readingTimeSpan.textContent = words > 0 ? `~${readMins} min read` : "0 min read";
    }

    newsInput.addEventListener("input", () => {
        updateTextStats();
        hideError();
    });

    // 2. Fetch Sample Articles
    async function loadSamples() {
        try {
            const res = await fetch("/api/examples");
            const data = await res.json();
            if (data.status === "success" && data.examples) {
                sampleArticles = data.examples;
                bindSampleButtons();
            }
        } catch (err) {
            console.warn("Could not load online samples", err);
        }
    }

    function bindSampleButtons() {
        const buttons = document.querySelectorAll(".sample-pill");
        buttons.forEach(btn => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.getAttribute("data-index"), 10);
                if (sampleArticles[idx]) {
                    const sample = sampleArticles[idx];
                    newsInput.value = `${sample.title}\n\n${sample.text}`;
                    updateTextStats();
                    hideError();
                    newsInput.focus();
                    
                    document.querySelector(".editor-section").scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });
                }
            });
        });
    }

    // 3. Clear Button
    clearBtn.addEventListener("click", () => {
        newsInput.value = "";
        updateTextStats();
        hideError();
        resultCard.classList.add("hidden");
        newsInput.focus();
    });

    // 4. Keyboard Shortcut: Ctrl + Enter
    newsInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            analyzeText();
        }
    });

    // 5. Analyze Action
    analyzeBtn.addEventListener("click", analyzeText);

    async function analyzeText() {
        const text = newsInput.value.trim();
        if (!text) {
            showError("Please enter or paste news article text before analyzing.");
            return;
        }

        const words = text.split(/\s+/).filter(w => w.length > 0).length;
        if (words < 3 && text.length < 15) {
            showError("Input is too short. Please provide at least 3 words or a headline for reliable analysis.");
            return;
        }

        hideError();
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                showError(data.message || data.error || "An error occurred during analysis.");
                setLoading(false);
                return;
            }

            displayResult(data);
        } catch (err) {
            showError("Could not reach the analysis server. Please check that Flask is running.");
        } finally {
            setLoading(false);
        }
    }

    function displayResult(data) {
        const isReal = data.prediction.toUpperCase() === "REAL";

        // Toggle real / fake classes
        resultBox.classList.remove("is-real", "is-fake");
        resultBox.classList.add(isReal ? "is-real" : "is-fake");

        // Verdict labels
        if (isReal) {
            verdictTag.textContent = "LIKELY AUTHENTIC";
            verdictHeading.textContent = "Text matches patterns of authentic reporting";
        } else {
            verdictTag.textContent = "LIKELY MISINFORMATION";
            verdictHeading.textContent = "Text matches patterns commonly seen in fake news";
        }

        // Confidence
        const confPct = Math.round(data.confidence * 10) / 10;
        confidenceVal.textContent = `${confPct.toFixed(1)}%`;
        
        confidenceBar.style.width = "0%";
        setTimeout(() => {
            confidenceBar.style.width = `${confPct}%`;
        }, 50);

        // Tags Matrix
        featureTagsContainer.innerHTML = "";
        const details = data.feature_details || [];
        if (details.length > 0) {
            details.forEach(item => {
                const chip = document.createElement("span");
                const dirClass = item.direction ? item.direction.toLowerCase() : (isReal ? 'real' : 'fake');
                chip.className = `word-chip ${dirClass}`;
                chip.textContent = item.word;
                chip.title = `Weight: ${item.score || 'N/A'}`;
                featureTagsContainer.appendChild(chip);
            });
        } else if (data.important_features && data.important_features.length > 0) {
            data.important_features.forEach(word => {
                const chip = document.createElement("span");
                chip.className = `word-chip ${isReal ? 'real' : 'fake'}`;
                chip.textContent = word;
                featureTagsContainer.appendChild(chip);
            });
        } else {
            featureTagsContainer.innerHTML = '<span class="word-chip">Standard vocabulary</span>';
        }

        // Explanation & Metadata
        explanationText.textContent = data.explanation;
        statModel.textContent = `Model: ${data.model || "Linear SVM"}`;
        statLatency.textContent = `Latency: ${data.processing_time_ms}ms`;
        statTokens.textContent = `Tokens: ${data.stats ? data.stats.cleaned_tokens_count : 0}`;

        // Unhide result
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function setLoading(isLoading) {
        if (isLoading) {
            spinner.classList.remove("hidden");
            btnText.textContent = "Analyzing...";
            analyzeBtn.disabled = true;
        } else {
            spinner.classList.add("hidden");
            btnText.textContent = "Analyze Article";
            analyzeBtn.disabled = false;
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove("hidden");
        errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function hideError() {
        errorBanner.classList.add("hidden");
    }

    // Initialize
    loadSamples();
});
