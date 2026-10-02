/**
 * script.js
 * =============================================================================
 * Credit Card Fraud Detection - Interactive Client Logic
 * Handles client-side validation, asynchronous model inference, sample loading,
 * and dynamic visual updates for prediction results.
 * =============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const btnPredict = document.getElementById("btn-predict");
    const btnText = document.getElementById("btn-text");
    const btnIcon = document.getElementById("btn-icon");
    const btnSpinner = document.getElementById("btn-spinner");
    const btnReset = document.getElementById("btn-reset");
    const btnLoadLegit = document.getElementById("btn-load-legit");
    const btnLoadFraud = document.getElementById("btn-load-fraud");

    const resultContainer = document.getElementById("result-container");
    const resultCard = document.getElementById("result-card");
    const resultLabel = document.getElementById("result-label");
    const resultBadge = document.getElementById("result-badge");
    const resultMessage = document.getElementById("result-message");
    const resultStatusIcon = document.getElementById("result-status-icon");
    const resultConfidence = document.getElementById("result-confidence");
    const legitProbText = document.getElementById("legit-prob-text");
    const fraudProbText = document.getElementById("fraud-prob-text");
    const legitProgressBar = document.getElementById("legit-progress-bar");
    const fraudProgressBar = document.getElementById("fraud-progress-bar");
    const alertContainer = document.getElementById("dynamic-alert-container");

    // All 30 expected feature field names
    const featureKeys = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ];

    // Benchmark sample data embedded for instant client-side offline loading
    const samples = {
        legitimate: {
            Time: 0.0,
            V1: -1.359807, V2: -0.072781, V3: 2.536347, V4: 1.378155,
            V5: -0.338321, V6: 0.462388, V7: 0.239599, V8: 0.098698,
            V9: 0.363787, V10: 0.090794, V11: -0.551600, V12: -0.617801,
            V13: -0.991390, V14: -0.311169, V15: 1.468177, V16: -0.470401,
            V17: 0.207971, V18: 0.025791, V19: 0.403993, V20: 0.251412,
            V21: -0.018307, V22: 0.277838, V23: -0.110474, V24: 0.066928,
            V25: 0.128539, V26: -0.189115, V27: 0.133558, V28: -0.021053,
            Amount: 149.62
        },
        fraud: {
            Time: 406.0,
            V1: -2.312227, V2: 1.951992, V3: -1.609851, V4: 3.997906,
            V5: -0.522188, V6: -1.426545, V7: -2.537387, V8: 1.391657,
            V9: -2.770089, V10: -2.772272, V11: 3.202033, V12: -2.899907,
            V13: -0.595222, V14: -4.289254, V15: 0.389724, V16: -1.140747,
            V17: -2.830056, V18: -0.016822, V19: 0.416956, V20: 0.126911,
            V21: 0.517232, V22: -0.035049, V23: -0.465211, V24: 0.320198,
            V25: 0.044519, V26: 0.177840, V27: 0.261145, V28: -0.143276,
            Amount: 0.00
        }
    };

    /**
     * Displays a dismissible banner alert.
     */
    function showAlert(message, type = "warning") {
        alertContainer.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show shadow-sm border-0 mb-4" role="alert">
                <div class="d-flex align-items-center">
                    <i class="bi ${type === 'danger' ? 'bi-x-octagon-fill' : type === 'success' ? 'bi-check-circle-fill' : 'bi-exclamation-triangle-fill'} fs-4 me-3 flex-shrink-0"></i>
                    <div>${message}</div>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        alertContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function clearAlert() {
        alertContainer.innerHTML = "";
    }

    /**
     * Loads a sample preset into the input fields.
     */
    function populateFields(sampleData, label) {
        clearAlert();
        let count = 0;
        for (const [key, value] of Object.entries(sampleData)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;
                input.classList.remove("is-invalid");
                count++;
            }
        }
        showAlert(`Successfully populated <strong>${count} features</strong> from <em>${label}</em> sample. Click "Predict Transaction" to run classification.`, "info");
    }

    if (btnLoadLegit) {
        btnLoadLegit.addEventListener("click", () => {
            populateFields(samples.legitimate, "Legitimate Transaction");
        });
    }

    if (btnLoadFraud) {
        btnLoadFraud.addEventListener("click", () => {
            populateFields(samples.fraud, "Fraudulent Transaction");
        });
    }

    /**
     * Resets form, styles, and hidden results.
     */
    if (btnReset) {
        btnReset.addEventListener("click", () => {
            form.reset();
            clearAlert();
            featureKeys.forEach(key => {
                const input = document.getElementById(key);
                if (input) {
                    input.classList.remove("is-invalid");
                }
            });
            if (resultContainer) {
                resultContainer.classList.add("d-none");
            }
        });
    }

    /**
     * Validates that all 30 fields are present and are valid numbers.
     */
    function validateForm() {
        const payload = {};
        const errors = [];

        featureKeys.forEach(key => {
            const input = document.getElementById(key);
            if (!input) return;

            const val = input.value.trim();
            if (val === "") {
                input.classList.add("is-invalid");
                errors.push(`Field '${key}' is required.`);
            } else if (isNaN(Number(val))) {
                input.classList.add("is-invalid");
                errors.push(`Field '${key}' must be a valid number.`);
            } else {
                input.classList.remove("is-invalid");
                payload[key] = parseFloat(val);
            }
        });

        return { isValid: errors.length === 0, payload, errors };
    }

    /**
     * Updates the DOM with the prediction result.
     */
    function renderResult(result) {
        resultContainer.classList.remove("d-none");

        // Reset classes
        resultCard.classList.remove("legit-card", "fraud-card");

        const isFraud = result.prediction === 1;

        if (isFraud) {
            resultCard.classList.add("fraud-card");
            resultLabel.textContent = "Fraudulent Transaction Detected";
            resultBadge.textContent = "High Risk Anomaly";
            resultMessage.textContent = "Transaction characteristics strongly resemble high-risk fraudulent behavior detected by the Logistic Regression classifier.";
            resultStatusIcon.innerHTML = '<i class="bi bi-shield-slash-fill text-danger"></i>';
        } else {
            resultCard.classList.add("legit-card");
            resultLabel.textContent = "Legitimate Transaction";
            resultBadge.textContent = "Secure Verified";
            resultMessage.textContent = "Transaction metrics are consistent with standard legitimate user spending patterns.";
            resultStatusIcon.innerHTML = '<i class="bi bi-shield-check-fill text-success"></i>';
        }

        // Confidence and probabilities
        if (result.confidence_percent !== undefined && result.confidence_percent !== null) {
            resultConfidence.textContent = `${result.confidence_percent}%`;
        } else {
            resultConfidence.textContent = "N/A";
        }

        const legitPercent = result.legit_prob_percent !== null ? result.legit_prob_percent : 0;
        const fraudPercent = result.fraud_prob_percent !== null ? result.fraud_prob_percent : 0;

        legitProbText.textContent = `${legitPercent}%`;
        fraudProbText.textContent = `${fraudPercent}%`;

        legitProgressBar.style.width = `${legitPercent}%`;
        fraudProgressBar.style.width = `${fraudPercent}%`;

        // Update Real-World Risk Analytics & Banking Action Banner
        const actionCodeBadge = document.getElementById("action-code-badge");
        const actionTitle = document.getElementById("action-title");
        const actionNote = document.getElementById("action-note");
        const actionIcon = document.getElementById("action-icon");
        const riskScoreDisplay = document.getElementById("risk-score-display");
        const riskScoreLabel = document.getElementById("risk-score-label");

        if (actionCodeBadge && result.action_code) {
            actionCodeBadge.textContent = result.action_code;
        }
        if (actionTitle && result.recommended_action) {
            actionTitle.textContent = result.recommended_action;
        }
        if (actionNote && result.action_note) {
            actionNote.textContent = result.action_note;
        }
        if (actionIcon) {
            if (result.action_code === "BLOCK") {
                actionIcon.innerHTML = '<i class="bi bi-slash-circle-fill text-danger fs-3"></i>';
            } else if (result.action_code === "STEP_UP_OTP") {
                actionIcon.innerHTML = '<i class="bi bi-shield-exclamation text-warning fs-3"></i>';
            } else {
                actionIcon.innerHTML = '<i class="bi bi-patch-check-fill text-success fs-3"></i>';
            }
        }
        if (riskScoreDisplay && result.risk_score !== undefined) {
            riskScoreDisplay.innerHTML = `${result.risk_score}<span class="fs-5 text-secondary">/100</span>`;
            if (result.risk_score >= 70) {
                riskScoreDisplay.className = "display-6 fw-bold my-1 text-danger";
                riskScoreLabel.textContent = "Critical Anomaly Threshold";
            } else if (result.risk_score >= 30) {
                riskScoreDisplay.className = "display-6 fw-bold my-1 text-warning";
                riskScoreLabel.textContent = "Moderate Review Range";
            } else {
                riskScoreDisplay.className = "display-6 fw-bold my-1 text-success";
                riskScoreLabel.textContent = "Safe Low-Risk Zone";
            }
        }

        // Render Top Risk Drivers (Pushed toward Fraud)
        const riskDriversList = document.getElementById("risk-drivers-list");
        if (riskDriversList) {
            if (result.top_risk_drivers && result.top_risk_drivers.length > 0) {
                riskDriversList.innerHTML = result.top_risk_drivers.map(d => `
                    <div class="driver-row p-2 rounded mb-2 bg-dark">
                        <div class="d-flex justify-content-between small mb-1">
                            <strong class="text-white font-monospace">${d.feature}</strong>
                            <span class="text-danger fw-bold">+${d.contribution.toFixed(3)} log-odds</span>
                        </div>
                        <div class="progress progress-dark" style="height: 5px;">
                            <div class="progress-bar bg-danger" style="width: ${Math.min(d.abs_contribution * 15, 100)}%;"></div>
                        </div>
                        <div class="d-flex justify-content-between text-muted x-small mt-1">
                            <span>${d.description}</span>
                            <span class="font-monospace">val: ${d.value} | weight: ${d.coefficient}</span>
                        </div>
                    </div>
                `).join("");
            } else {
                riskDriversList.innerHTML = '<p class="text-muted small mb-0">No strong risk drivers detected for this transaction.</p>';
            }
        }

        // Render Top Safety Factors (Pushed toward Legitimacy)
        const safetyDriversList = document.getElementById("safety-drivers-list");
        if (safetyDriversList) {
            if (result.top_safety_drivers && result.top_safety_drivers.length > 0) {
                safetyDriversList.innerHTML = result.top_safety_drivers.map(d => `
                    <div class="driver-row p-2 rounded mb-2 bg-dark">
                        <div class="d-flex justify-content-between small mb-1">
                            <strong class="text-white font-monospace">${d.feature}</strong>
                            <span class="text-success fw-bold">${d.contribution.toFixed(3)} log-odds</span>
                        </div>
                        <div class="progress progress-dark" style="height: 5px;">
                            <div class="progress-bar bg-success" style="width: ${Math.min(d.abs_contribution * 15, 100)}%;"></div>
                        </div>
                        <div class="d-flex justify-content-between text-muted x-small mt-1">
                            <span>${d.description}</span>
                            <span class="font-monospace">val: ${d.value} | weight: ${d.coefficient}</span>
                        </div>
                    </div>
                `).join("");
            } else {
                safetyDriversList.innerHTML = '<p class="text-muted small mb-0">No strong safety factors detected for this transaction.</p>';
            }
        }

        // Setup Copy Audit button handler
        const btnCopyAudit = document.getElementById("btn-copy-audit");
        if (btnCopyAudit) {
            btnCopyAudit.onclick = () => {
                const report = [
                    "=== BANKING FRAUD RISK AUDIT REPORT ===",
                    `Classification: ${result.label}`,
                    `Risk Score: ${result.risk_score} / 100`,
                    `Fraud Probability: ${fraudPercent}%`,
                    `Legitimate Probability: ${legitPercent}%`,
                    `Recommended Action: ${result.recommended_action || 'N/A'}`,
                    `Action Protocol: ${result.action_note || 'N/A'}`,
                    "Top Risk Factors: " + (result.top_risk_drivers || []).map(d => `${d.feature} (+${d.contribution.toFixed(3)})`).join(", "),
                    "Top Safety Factors: " + (result.top_safety_drivers || []).map(d => `${d.feature} (${d.contribution.toFixed(3)})`).join(", "),
                    "Timestamp: " + new Date().toISOString()
                ].join("\n");

                navigator.clipboard.writeText(report).then(() => {
                    const originalText = btnCopyAudit.innerHTML;
                    btnCopyAudit.innerHTML = '<i class="bi bi-check2 me-1"></i> Copied to Clipboard!';
                    setTimeout(() => {
                        btnCopyAudit.innerHTML = originalText;
                    }, 2500);
                });
            };
        }

        // Smooth scroll to result card
        resultContainer.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    /**
     * Intercept form submit for asynchronous prediction
     */
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            clearAlert();

            const validation = validateForm();
            if (!validation.isValid) {
                showAlert(`Please correct the errors in the form: ${validation.errors[0]} (and ${validation.errors.length - 1} more)`, "warning");
                return;
            }

            // Set loading state
            btnPredict.disabled = true;
            btnSpinner.classList.remove("d-none");
            btnIcon.classList.add("d-none");
            btnText.textContent = "Classifying Transaction...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(validation.payload)
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Inference server error occurred.");
                }

                renderResult(data.result);
            } catch (err) {
                console.error("Prediction error:", err);
                showAlert(`Inference Failed: ${err.message}`, "danger");
            } finally {
                // Restore button state
                btnPredict.disabled = false;
                btnSpinner.classList.add("d-none");
                btnIcon.classList.remove("d-none");
                btnText.textContent = "Predict Transaction";
            }
        });
    }
});
