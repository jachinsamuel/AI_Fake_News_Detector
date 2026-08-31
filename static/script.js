/**
 * Minimalist Fake News Detector Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // Inputs & Telemetry
    const newsInput = document.getElementById("news-input");
    const wordCountSpan = document.getElementById("word-count");
    const charCountSpan = document.getElementById("char-count");
    
    // Actions & Buttons
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const spinner = document.getElementById("loading-spinner");
    const btnText = analyzeBtn.querySelector(".btn-text");
    const errorBanner = document.getElementById("error-banner");
    const errorMessage = document.getElementById("error-message");

    // Results Elements
    const resultCard = document.getElementById("result-card");
    const verdictTag = document.getElementById("verdict-tag");
    const confidenceVal = document.getElementById("confidence-val");
    const confidenceBar = document.getElementById("confidence-bar");
    const featureTagsContainer = document.getElementById("feature-tags-container");
    const explanationText = document.getElementById("explanation-text");

    // 1. Text Counter
    function updateTextStats() {
        const text = newsInput.value.trim();
        const chars = text.length;
        const words = text ? text.split(/\s+/).filter(w => w.length > 0).length : 0;

        charCountSpan.textContent = `${chars.toLocaleString()} characters`;
        wordCountSpan.textContent = `${words.toLocaleString()} words`;
    }

    newsInput.addEventListener("input", () => {
        updateTextStats();
        hideError();
    });

    // 2. Clear Button
    clearBtn.addEventListener("click", () => {
        newsInput.value = "";
        updateTextStats();
        hideError();
        resultCard.classList.add("hidden");
        newsInput.focus();
    });

    // 3. Keyboard Shortcut: Ctrl + Enter
    newsInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            analyzeText();
        }
    });

    // 4. Analyze Action
    analyzeBtn.addEventListener("click", analyzeText);

    async function analyzeText() {
        const text = newsInput.value.trim();
        if (!text) {
            showError("Please enter or paste a news article before analyzing.");
            return;
        }

        const words = text.split(/\s+/).filter(w => w.length > 0).length;
        if (words < 3 && text.length < 15) {
            showError("Input text is too short. Please provide at least 3 words or a headline.");
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
            showError("Could not reach the analysis server. Ensure Flask backend is running.");
        } finally {
            setLoading(false);
        }
    }

    function displayResult(data) {
        const isReal = data.prediction.toUpperCase() === "REAL";

        // Set state class
        resultCard.classList.remove("is-real", "is-fake");
        resultCard.classList.add(isReal ? "is-real" : "is-fake");

        // Verdict labels
        verdictTag.textContent = isReal ? "✓ REAL NEWS" : "⚠ FAKE NEWS";

        // Confidence
        const confPct = Math.round(data.confidence * 10) / 10;
        confidenceVal.textContent = `${confPct.toFixed(1)}%`;
        
        confidenceBar.style.width = "0%";
        setTimeout(() => {
            confidenceBar.style.width = `${confPct}%`;
        }, 50);

        // Tags List
        featureTagsContainer.innerHTML = "";
        const details = data.feature_details || [];
        if (details.length > 0) {
            details.forEach(item => {
                const tag = document.createElement("span");
                const dirClass = item.direction ? item.direction.toLowerCase() : (isReal ? 'real' : 'fake');
                tag.className = `word-tag ${dirClass}`;
                tag.textContent = item.word;
                featureTagsContainer.appendChild(tag);
            });
        } else if (data.important_features && data.important_features.length > 0) {
            data.important_features.forEach(word => {
                const tag = document.createElement("span");
                tag.className = `word-tag ${isReal ? 'real' : 'fake'}`;
                tag.textContent = word;
                featureTagsContainer.appendChild(tag);
            });
        } else {
            featureTagsContainer.innerHTML = '<span class="word-tag">General vocabulary</span>';
        }

        // Explanation text
        explanationText.textContent = data.explanation;

        // Reveal card
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
            btnText.textContent = "Analyze News";
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
});
