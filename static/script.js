/**
 * Fake News Detector Controller with URL Scraping, Wikipedia Grounding & Caching
 */

document.addEventListener("DOMContentLoaded", () => {
    // Mode Switcher Elements
    const tabText = document.getElementById("tab-text");
    const tabUrl = document.getElementById("tab-url");
    const panelText = document.getElementById("panel-text");
    const panelUrl = document.getElementById("panel-url");
    const urlInput = document.getElementById("url-input");
    const fetchUrlBtn = document.getElementById("fetch-url-btn");
    const urlSpinner = document.getElementById("url-spinner");
    const scrapedMeta = document.getElementById("scraped-meta");
    const scrapedTitle = document.getElementById("scraped-title");
    const scrapedSource = document.getElementById("scraped-source");

    // Text & Telemetry Inputs
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
    const statModel = document.getElementById("stat-model");
    const confidenceVal = document.getElementById("confidence-val");
    const confidenceBar = document.getElementById("confidence-bar");
    const featureTagsContainer = document.getElementById("feature-tags-container");
    const explanationText = document.getElementById("explanation-text");

    // Wikipedia & Live Web Elements
    const wikiBox = document.getElementById("wiki-grounding-box");
    const wikiDesc = document.getElementById("wiki-desc");
    const wikiSnippet = document.getElementById("wiki-snippet");
    const wikiLink = document.getElementById("wiki-link");

    const webVerificationBox = document.getElementById("web-verification-box");
    const webVerdictBadge = document.getElementById("web-verdict-badge");
    const webSummaryText = document.getElementById("web-summary-text");
    const sourcesContainer = document.getElementById("sources-container");
    const sourcesList = document.getElementById("sources-list");
    const factChecksContainer = document.getElementById("fact-checks-container");
    const factChecksList = document.getElementById("fact-checks-list");

    const telemetryLatency = document.getElementById("telemetry-latency");
    const telemetryCache = document.getElementById("telemetry-cache");
    const exportPdfBtn = document.getElementById("export-pdf-btn");

    // Voice & Speech Recognition Elements
    const voiceBtn = document.getElementById("voice-btn");
    const voiceBtnText = document.getElementById("voice-btn-text");
    const voiceStatusBanner = document.getElementById("voice-status-banner");
    const stopVoiceBtn = document.getElementById("stop-voice-btn");

    let lastAnalysisData = null;

    // Speech-to-Text Recognition Setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isRecording = false;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = () => {
            isRecording = true;
            if (voiceBtn) voiceBtn.classList.add("listening");
            if (voiceBtnText) voiceBtnText.textContent = "Listening...";
            if (voiceStatusBanner) voiceStatusBanner.classList.remove("hidden");
        };

        recognition.onresult = (event) => {
            let transcript = "";
            for (let i = 0; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }
            newsInput.value = transcript;
            updateTextStats();
        };

        recognition.onerror = (event) => {
            console.warn("Speech recognition error:", event.error);
            if (event.error === "not-allowed") {
                showError("Microphone permission was denied. Please allow microphone access in your browser settings.");
            } else if (event.error !== "no-speech") {
                showError(`Speech notice: ${event.error}`);
            }
            stopVoiceRecording(false);
        };

        recognition.onend = () => {
            stopVoiceRecording(false);
        };
    }

    function startVoiceRecording() {
        if (!recognition) {
            showError("Speech recognition is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.");
            return;
        }
        hideError();
        try {
            recognition.start();
        } catch (e) {
            console.warn(e);
        }
    }

    function stopVoiceRecording(autoAnalyze = false) {
        isRecording = false;
        if (voiceBtn) {
            voiceBtn.classList.remove("listening");
            voiceBtnText.textContent = "Voice Input";
        }
        if (voiceStatusBanner) {
            voiceStatusBanner.classList.add("hidden");
        }
        if (recognition) {
            try {
                recognition.stop();
            } catch (e) {}
        }

        if (autoAnalyze && newsInput.value.trim().length > 10) {
            analyzeText();
        }
    }

    if (voiceBtn) {
        voiceBtn.addEventListener("click", () => {
            if (isRecording) {
                stopVoiceRecording(true);
            } else {
                startVoiceRecording();
            }
        });
    }

    if (stopVoiceBtn) {
        stopVoiceBtn.addEventListener("click", () => {
            stopVoiceRecording(true);
        });
    }

    // 1. Mode Switcher
    tabText.addEventListener("click", () => {
        tabText.classList.add("active");
        tabUrl.classList.remove("active");
        panelText.classList.remove("hidden");
        panelUrl.classList.add("hidden");
        newsInput.focus();
    });

    tabUrl.addEventListener("click", () => {
        tabUrl.classList.add("active");
        tabText.classList.remove("active");
        panelUrl.classList.remove("hidden");
        panelText.classList.add("hidden");
        urlInput.focus();
    });

    // 2. URL Scraper Fetch
    fetchUrlBtn.addEventListener("click", handleUrlFetch);
    urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleUrlFetch();
        }
    });

    async function handleUrlFetch() {
        const url = urlInput.value.trim();
        if (!url) {
            showError("Please paste a valid news article URL.");
            return;
        }

        hideError();
        setScrapeLoading(true);

        try {
            const resp = await fetch("/api/scrape-url", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });
            const data = await resp.json();

            if (!resp.ok || data.error) {
                showError(data.message || data.error || "Could not scrape article from link.");
                setScrapeLoading(false);
                return;
            }

            const article = data.data;
            newsInput.value = article.combined_text;
            updateTextStats();

            // Show scraped preview
            scrapedTitle.textContent = article.title;
            scrapedSource.textContent = `Source: ${article.source} • ${article.word_count} words extracted`;
            scrapedMeta.classList.remove("hidden");

            // Auto-trigger analysis
            analyzeText();
        } catch (err) {
            showError("Network error connecting to scraper service.");
        } finally {
            setScrapeLoading(false);
        }
    }

    function setScrapeLoading(isLoading) {
        if (isLoading) {
            urlSpinner.classList.remove("hidden");
            fetchUrlBtn.disabled = true;
        } else {
            urlSpinner.classList.add("hidden");
            fetchUrlBtn.disabled = false;
        }
    }

    // 3. Text Counter
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

    // 4. Clear Button
    clearBtn.addEventListener("click", () => {
        newsInput.value = "";
        urlInput.value = "";
        scrapedMeta.classList.add("hidden");
        updateTextStats();
        hideError();
        resultCard.classList.add("hidden");
        newsInput.focus();
    });

    // 5. Keyboard Shortcut: Ctrl + Enter
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            analyzeText();
        }
    });

    // 6. Analyze Action
    analyzeBtn.addEventListener("click", analyzeText);

    async function analyzeText() {
        const text = newsInput.value.trim();
        if (!text) {
            showError("Please enter news text or fetch an article URL before analyzing.");
            return;
        }

        const words = text.split(/\s+/).filter(w => w.length > 0).length;
        if (words < 3 && text.length < 15) {
            showError("Input text is too short. Please provide at least 3 words or a complete headline.");
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

        // State classes
        resultCard.classList.remove("is-real", "is-fake");
        resultCard.classList.add(isReal ? "is-real" : "is-fake");

        // Verdict & Model
        verdictTag.textContent = isReal ? "✓ REAL NEWS" : "⚠ FAKE NEWS";
        statModel.textContent = data.model || "Ensemble Classifier";

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
        } else {
            featureTagsContainer.innerHTML = '<span class="word-tag">General vocabulary</span>';
        }

        // Explanation text
        explanationText.textContent = data.explanation;

        // Wikipedia Grounding
        const web = data.web_verification;
        if (web && web.wikipedia_grounding && web.wikipedia_grounding.is_grounded) {
            wikiBox.classList.remove("hidden");
            wikiDesc.textContent = `${web.wikipedia_grounding.entity} — ${web.wikipedia_grounding.description}`;
            wikiSnippet.textContent = web.wikipedia_grounding.extract_snippet;
            wikiLink.href = web.wikipedia_grounding.url;
        } else {
            wikiBox.classList.add("hidden");
        }

        // Live Web Verification & Sources
        if (web && web.status === "SUCCESS") {
            webVerificationBox.classList.remove("hidden");
            webSummaryText.textContent = web.web_summary || "Live web analysis completed.";

            // Web verdict badge
            webVerdictBadge.className = "web-status-badge";
            if (web.is_debunked) {
                webVerdictBadge.textContent = "Debunked by Fact-Checkers";
                webVerdictBadge.classList.add("debunked");
            } else if (web.is_uncorroborated_hoax) {
                webVerdictBadge.textContent = "Uncorroborated Hoax";
                webVerdictBadge.classList.add("debunked");
            } else if (web.web_verdict.includes("WIKIPEDIA") || web.web_verdict === "CORROBORATED_BY_LIVE_NEWS") {
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

        // Telemetry
        const timeMs = data.processing_time_ms !== undefined ? data.processing_time_ms : 300;
        telemetryLatency.textContent = `Latency: ${timeMs}ms`;
        if (data.cached) {
            telemetryCache.classList.remove("hidden");
        } else {
            telemetryCache.classList.add("hidden");
        }

        // Store for PDF export
        lastAnalysisData = {
            ...data,
            input_text: newsInput.value.trim()
        };

        // Reveal card
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // 7. Export Formal Fact-Check PDF Report
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener("click", async () => {
            if (!lastAnalysisData) {
                showError("Please analyze a news claim first before exporting a report.");
                return;
            }

            try {
                const resp = await fetch("/export-report", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(lastAnalysisData)
                });

                if (!resp.ok) {
                    showError("Could not generate report from server.");
                    return;
                }

                const html = await resp.text();
                const printWindow = window.open("", "_blank");
                if (printWindow) {
                    printWindow.document.open();
                    printWindow.document.write(html);
                    printWindow.document.close();
                } else {
                    showError("Popup blocked. Please allow popups to open and print the PDF report.");
                }
            } catch (err) {
                showError("Failed to generate fact-check certificate.");
            }
        });
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
