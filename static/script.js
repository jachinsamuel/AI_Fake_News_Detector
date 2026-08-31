/**
 * Fake News Detector Controller with Live Web Verification
 */

document.addEventListener("DOMContentLoaded", () => {
    // Inputs & Telemetry
    const newsInput = document.getElementById("news-input");
    const wordCountSpan = document.getElementById("word-count");
    const charCountSpan = document.getElementById("char-count");
    const checkWebToggle = document.getElementById("check-web-toggle");
    
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

    // Live Web Verification Elements
    const webVerificationBox = document.getElementById("web-verification-box");
    const webVerdictBadge = document.getElementById("web-verdict-badge");
    const webSummaryText = document.getElementById("web-summary-text");
    const sourcesContainer = document.getElementById("sources-container");
    const sourcesList = document.getElementById("sources-list");
    const factChecksContainer = document.getElementById("fact-checks-container");
    const factChecksList = document.getElementById("fact-checks-list");

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

        const checkWeb = checkWebToggle ? checkWebToggle.checked : true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ 
                    text: text,
                    check_web: checkWeb
                })
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

        // Render Live Web Verification & Sources
        const web = data.web_verification;
        if (web && web.status === "SUCCESS") {
            webVerificationBox.classList.remove("hidden");
            webSummaryText.textContent = web.web_summary || "Live web analysis completed.";

            // Set web verdict badge
            webVerdictBadge.className = "web-status-badge";
            if (web.is_debunked) {
                webVerdictBadge.textContent = "Debunked by Fact-Checkers";
                webVerdictBadge.classList.add("debunked");
            } else if (web.web_verdict === "CORROBORATED_BY_LIVE_NEWS") {
                webVerdictBadge.textContent = "Corroborated by News Outlets";
                webVerdictBadge.classList.add("corroborated");
            } else if (web.sources_count > 0) {
                webVerdictBadge.textContent = `${web.sources_count} Live Articles Found`;
                webVerdictBadge.classList.add("corroborated");
            } else {
                webVerdictBadge.textContent = "No Live Matches";
            }

            // Live News Sources
            sourcesList.innerHTML = "";
            const sources = web.live_sources || [];
            if (sources.length > 0) {
                sourcesContainer.classList.remove("hidden");
                sources.forEach(s => {
                    const a = document.createElement("a");
                    a.href = s.url || "#";
                    a.target = "_blank";
                    a.rel = "noopener noreferrer";
                    a.className = "source-item";
                    a.innerHTML = `
                        <span class="source-title" title="${s.title}">${s.title}</span>
                        <span class="source-meta">${s.source} ${s.published_at ? '• ' + s.published_at : ''}</span>
                    `;
                    sourcesList.appendChild(a);
                });
            } else {
                sourcesContainer.classList.add("hidden");
            }

            // Fact Check Reviews
            factChecksList.innerHTML = "";
            const checks = web.fact_checks || [];
            if (checks.length > 0) {
                factChecksContainer.classList.remove("hidden");
                checks.forEach(fc => {
                    const a = document.createElement("a");
                    a.href = fc.url || "#";
                    a.target = "_blank";
                    a.rel = "noopener noreferrer";
                    a.className = "fact-check-item";
                    a.innerHTML = `
                        <span><strong>${fc.publisher}:</strong> ${fc.claim}</span>
                        <span style="font-weight:700;">Rating: ${fc.rating}</span>
                    `;
                    factChecksList.appendChild(a);
                });
            } else {
                factChecksContainer.classList.add("hidden");
            }

        } else {
            webVerificationBox.classList.add("hidden");
        }

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
