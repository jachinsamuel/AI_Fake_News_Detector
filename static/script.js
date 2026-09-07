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

    // OCR Mode & Dropzone Elements
    const tabImage = document.getElementById("tab-image");
    const panelImage = document.getElementById("panel-image");
    const ocrDropzone = document.getElementById("ocr-dropzone");
    const imageFileInput = document.getElementById("image-file-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const imagePreviewContainer = document.getElementById("image-preview-container");
    const imagePreviewImg = document.getElementById("image-preview-img");
    const removeImageBtn = document.getElementById("remove-image-btn");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const ocrProgressBox = document.getElementById("ocr-progress-box");
    const ocrStatusText = document.getElementById("ocr-status-text");
    const ocrPct = document.getElementById("ocr-pct");
    const ocrProgressBar = document.getElementById("ocr-progress-bar");
    const ocrResultBox = document.getElementById("ocr-result-box");
    const ocrExtractedText = document.getElementById("ocr-extracted-text");

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

    // Forensic Stylometry Elements
    const stylometryBox = document.getElementById("stylometry-box");
    const styleVerdictBadge = document.getElementById("style-verdict-badge");
    const sensationalismBar = document.getElementById("sensationalism-bar");
    const sensationalismVal = document.getElementById("sensationalism-val");
    const attributionBar = document.getElementById("attribution-bar");
    const attributionVal = document.getElementById("attribution-val");
    const lexicalPill = document.getElementById("lexical-pill");
    const punctPill = document.getElementById("punct-pill");

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
            if (voiceBtn) {
                voiceBtn.classList.add("listening");
                voiceBtn.title = "Listening... Click to stop";
            }
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
            voiceBtn.title = "Dictate headline via microphone";
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
    function switchMode(mode) {
        tabText.classList.toggle("active", mode === "text");
        tabUrl.classList.toggle("active", mode === "url");
        if (tabImage) tabImage.classList.toggle("active", mode === "image");

        panelText.classList.toggle("hidden", mode !== "text");
        panelUrl.classList.toggle("hidden", mode !== "url");
        if (panelImage) panelImage.classList.toggle("hidden", mode !== "image");

        if (mode === "text") newsInput.focus();
        else if (mode === "url") urlInput.focus();
        else if (mode === "image" && ocrDropzone) ocrDropzone.focus();
    }

    tabText.addEventListener("click", () => switchMode("text"));
    tabUrl.addEventListener("click", () => switchMode("url"));
    if (tabImage) tabImage.addEventListener("click", () => switchMode("image"));

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

    // 2.5. OCR Image & Screenshot Engine
    let isProcessingOcr = false;

    if (ocrDropzone && imageFileInput) {
        ocrDropzone.addEventListener("click", (e) => {
            if (e.target !== removeImageBtn && !e.target.closest("#remove-image-btn")) {
                imageFileInput.click();
            }
        });

        ocrDropzone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                imageFileInput.click();
            }
        });

        // Drag & Drop
        ocrDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();
            ocrDropzone.classList.add("drag-over");
        });

        ocrDropzone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            e.stopPropagation();
            ocrDropzone.classList.remove("drag-over");
        });

        ocrDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            ocrDropzone.classList.remove("drag-over");
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleImageFile(e.dataTransfer.files[0]);
            }
        });

        imageFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleImageFile(e.target.files[0]);
            }
        });
    }

    // Global Clipboard Paste (Ctrl+V) anywhere on page
    document.addEventListener("paste", (e) => {
        // If user is pasting into a text input or textarea, let default paste happen
        if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) {
            return;
        }

        const clipboard = e.clipboardData || window.clipboardData;
        if (!clipboard || !clipboard.items) return;

        for (let i = 0; i < clipboard.items.length; i++) {
            const item = clipboard.items[i];
            if (item.type && item.type.indexOf("image") !== -1) {
                const file = item.getAsFile();
                if (file) {
                    e.preventDefault();
                    switchMode("image");
                    handleImageFile(file, "Pasted Screenshot");
                    break;
                }
            }
        }
    });

    if (removeImageBtn) {
        removeImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetOcrState();
        });
    }

    function resetOcrState() {
        if (imageFileInput) imageFileInput.value = "";
        if (imagePreviewContainer) imagePreviewContainer.classList.add("hidden");
        if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
        if (ocrProgressBox) ocrProgressBox.classList.add("hidden");
        if (ocrResultBox) ocrResultBox.classList.add("hidden");
        if (ocrExtractedText) ocrExtractedText.value = "";
        if (imagePreviewImg) imagePreviewImg.src = "";
    }

    async function handleImageFile(file, customName = null) {
        if (!file || !file.type.startsWith("image/")) {
            showError("Please select a valid image file (PNG, JPG, WEBP).");
            return;
        }

        hideError();
        const fileName = customName || file.name || "screenshot.png";
        const fileSizeKb = (file.size / 1024).toFixed(1);

        const reader = new FileReader();
        reader.onload = (event) => {
            if (imagePreviewImg) imagePreviewImg.src = event.target.result;
            if (previewFilename) previewFilename.textContent = fileName;
            if (previewFilesize) previewFilesize.textContent = `${fileSizeKb} KB`;
            if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
            if (imagePreviewContainer) imagePreviewContainer.classList.remove("hidden");
            
            // Execute OCR extraction
            processOcr(event.target.result);
        };
        reader.readAsDataURL(file);
    }

    async function processOcr(imageDataUrl) {
        if (isProcessingOcr) return;
        isProcessingOcr = true;

        if (ocrProgressBox) ocrProgressBox.classList.remove("hidden");
        if (ocrResultBox) ocrResultBox.classList.add("hidden");
        if (ocrProgressBar) ocrProgressBar.style.width = "8%";
        if (ocrStatusText) ocrStatusText.textContent = "Loading Tesseract OCR Engine...";
        if (ocrPct) ocrPct.textContent = "8%";

        try {
            if (typeof Tesseract === "undefined") {
                throw new Error("Tesseract.js OCR library is still loading or could not be reached. Please check your internet connection.");
            }

            const worker = await Tesseract.createWorker("eng", 1, {
                logger: (m) => {
                    if (m.status === "recognizing text") {
                        const pct = Math.min(99, Math.round(m.progress * 100));
                        if (ocrProgressBar) ocrProgressBar.style.width = `${pct}%`;
                        if (ocrStatusText) ocrStatusText.textContent = `Recognizing text (${pct}%)...`;
                        if (ocrPct) ocrPct.textContent = `${pct}%`;
                    } else if (m.status && ocrStatusText) {
                        ocrStatusText.textContent = `${m.status.charAt(0).toUpperCase() + m.status.slice(1)}...`;
                    }
                }
            });

            const ret = await worker.recognize(imageDataUrl);
            await worker.terminate();

            const rawText = ret.data.text || "";
            // Clean up lines, artifacts, and excessive whitespace
            const cleaned = rawText
                .replace(/\r\n/g, "\n")
                .split("\n")
                .map(l => l.trim())
                .filter(l => l.length > 0)
                .join(" ")
                .replace(/\s+/g, " ")
                .trim();

            if (ocrProgressBar) ocrProgressBar.style.width = "100%";
            if (ocrPct) ocrPct.textContent = "100%";
            if (ocrStatusText) ocrStatusText.textContent = "Text extracted successfully!";

            setTimeout(() => {
                if (ocrProgressBox) ocrProgressBox.classList.add("hidden");
            }, 600);

            if (!cleaned || cleaned.length < 5) {
                showError("Could not detect legible text from this image. Please ensure the screenshot has clear headline text.");
                return;
            }

            if (ocrExtractedText) ocrExtractedText.value = cleaned;
            if (ocrResultBox) ocrResultBox.classList.remove("hidden");
            newsInput.value = cleaned;
            updateTextStats();

            // Auto-trigger veracity verification
            analyzeText();

        } catch (err) {
            console.error("OCR Exception:", err);
            if (ocrProgressBox) ocrProgressBox.classList.add("hidden");
            showError(`OCR Error: ${err.message || "Failed to parse text from image."}`);
        } finally {
            isProcessingOcr = false;
        }
    }

    if (ocrExtractedText) {
        ocrExtractedText.addEventListener("input", () => {
            newsInput.value = ocrExtractedText.value;
            updateTextStats();
        });
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
        resetOcrState();
        updateTextStats();
        hideError();
        resultCard.classList.add("hidden");
        if (stylometryBox) stylometryBox.classList.add("hidden");
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

        // Instant Client-Side Session Cache (0.05ms repeat responses)
        const clientCacheKey = `fnd_cache_${checkWeb}_${text.toLowerCase().replace(/\s+/g, ' ')}`;
        try {
            const cachedPayload = sessionStorage.getItem(clientCacheKey);
            if (cachedPayload) {
                const parsed = JSON.parse(cachedPayload);
                parsed.processing_time_ms = 0.05;
                parsed.cached = true;
                displayResult(parsed);
                setLoading(false);
                return;
            }
        } catch (e) {}

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
            try {
                sessionStorage.setItem(clientCacheKey, JSON.stringify(data));
            } catch (e) {}
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

        // Forensic Stylometry Breakdown
        const sty = data.stylometry;
        if (sty && stylometryBox) {
            stylometryBox.classList.remove("hidden");
            styleVerdictBadge.textContent = sty.style_verdict || "Neutral";
            
            // Style badge appearance
            styleVerdictBadge.className = "style-badge";
            if (sty.style_verdict.includes("Clickbait") || sty.sensationalism_level === "High") {
                styleVerdictBadge.classList.add("style-clickbait");
            } else if (sty.style_verdict.includes("Journalistic") || sty.attribution_level === "High") {
                styleVerdictBadge.classList.add("style-journalistic");
            } else {
                styleVerdictBadge.classList.add("style-neutral");
            }

            // Sensationalism meter
            const sensPct = Math.min(100, Math.round(sty.sensationalism_density * 100));
            sensationalismBar.style.width = `${Math.max(4, sensPct)}%`;
            sensationalismVal.textContent = `${sty.sensationalism_level} (${sensPct}%)`;
            if (sty.sensationalism_level === "High") {
                sensationalismBar.style.backgroundColor = "#ef4444";
            } else if (sty.sensationalism_level === "Moderate") {
                sensationalismBar.style.backgroundColor = "#f59e0b";
            } else {
                sensationalismBar.style.backgroundColor = "#10b981";
            }

            // Attribution meter
            const attrPct = Math.min(100, Math.round(sty.attribution_score * 100));
            attributionBar.style.width = `${Math.max(4, attrPct)}%`;
            attributionVal.textContent = `${sty.attribution_level} (${attrPct}%)`;

            // Lexical Diversity
            const ttrPct = Math.round(sty.lexical_diversity * 100);
            lexicalPill.textContent = `${ttrPct}% TTR`;

            // Punctuation & Caps
            if (sty.punctuation_dramatism > 0.35 || sty.uppercase_ratio > 0.2) {
                punctPill.textContent = "Elevated / Dramatic";
                punctPill.className = "metric-pill pill-warn";
            } else {
                punctPill.textContent = "Objective / Standard";
                punctPill.className = "metric-pill pill-ok";
            }
        } else if (stylometryBox) {
            stylometryBox.classList.add("hidden");
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

    // Copy Summary Action
    const copySummaryBtn = document.getElementById("copy-summary-btn");
    if (copySummaryBtn) {
        copySummaryBtn.addEventListener("click", () => {
            if (!lastAnalysisData) return;
            const verdict = lastAnalysisData.prediction.toUpperCase() === "REAL" ? "REAL NEWS" : "FAKE NEWS";
            const textSample = newsInput.value.trim().slice(0, 140);
            const copyContent = `[${verdict} • ${lastAnalysisData.confidence}%]\n"${textSample}..."\n\nExplanation: ${lastAnalysisData.explanation}\n\nVerified by AI Fake News Detector with Live Web AI`;
            navigator.clipboard.writeText(copyContent).then(() => {
                const orig = copySummaryBtn.innerHTML;
                copySummaryBtn.innerHTML = "<span>✓</span><span>Copied!</span>";
                setTimeout(() => { copySummaryBtn.innerHTML = orig; }, 2000);
            });
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
