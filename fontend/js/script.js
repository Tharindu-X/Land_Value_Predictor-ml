// Land Price Predictor - JavaScript
// Handles form submissions, API calls, and dynamic content

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const analysisButtons = document.querySelectorAll('.analysis-btn');
    const predictionSection = document.getElementById('predictionSection');
    const investmentSection = document.getElementById('investmentSection');
    const predictionForm = document.getElementById('predictionForm');
    const investmentForm = document.getElementById('investmentForm');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');

    // Analysis type switching
    analysisButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active state
            analysisButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Show/hide sections
            const type = this.dataset.type;
            if (type === 'prediction') {
                predictionSection.style.display = 'block';
                investmentSection.style.display = 'none';
            } else {
                predictionSection.style.display = 'none';
                investmentSection.style.display = 'block';
            }

            // Hide results
            resultsSection.style.display = 'none';
        });
    });

    // Prediction form submission
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = {
            area_type: parseInt(document.getElementById('predArea').value),
            predict_year: parseInt(document.getElementById('predYear').value)
        };

        // Validate
        if (formData.predict_year < 1994) {
            alert('Year must be 1994 or later');
            return;
        }

        // Show loading
        resultsSection.style.display = 'none';
        loadingSpinner.style.display = 'block';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            // Hide loading
            loadingSpinner.style.display = 'none';

            if (result.success) {
                displayPredictionResults(result.data);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            loadingSpinner.style.display = 'none';
            alert('Network error: ' + error.message);
        }
    });

    // Investment form submission
    investmentForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = {
            area_type: parseInt(document.getElementById('invArea').value),
            purchase_price: parseFloat(document.getElementById('purchasePrice').value),
            purchase_year: parseInt(document.getElementById('purchaseYear').value),
            sell_year: parseInt(document.getElementById('sellYear').value),
            num_perches: parseFloat(document.getElementById('numPerches').value)
        };

        // Validate
        if (formData.purchase_price <= 0) {
            alert('Purchase price must be positive');
            return;
        }
        if (formData.sell_year <= formData.purchase_year) {
            alert('Sell year must be after purchase year');
            return;
        }
        if (formData.num_perches <= 0) {
            alert('Number of perches must be positive');
            return;
        }

        // Show loading
        resultsSection.style.display = 'none';
        loadingSpinner.style.display = 'block';

        try {
            const response = await fetch('/api/investment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            // Hide loading
            loadingSpinner.style.display = 'none';

            if (result.success) {
                displayInvestmentResults(result.data);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            loadingSpinner.style.display = 'none';
            alert('Network error: ' + error.message);
        }
    });

    // Display prediction results
    function displayPredictionResults(data) {
        const html = `
            <div class="result-header">
                <h2 class="result-title">
                    <svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                    </svg>
                    Price Prediction Report
                </h2>
                ${data.is_extrapolation ? 
                    `<span class="result-badge badge-warning">
                        <svg width="16" height="16" fill="currentColor">
                            <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
                        </svg>
                        Extrapolation
                    </span>` : 
                    `<span class="result-badge badge-success">
                        <svg width="16" height="16" fill="currentColor">
                            <path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/>
                        </svg>
                        Within Range
                    </span>`
                }
            </div>

            <div class="result-grid">
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4z"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Location</h3>
                    </div>
                    <div class="result-card-value">${data.area_name}</div>
                    <div class="result-card-subvalue">Prediction Year: ${data.predict_year}</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 6v6l4 2"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Predicted Price</h3>
                    </div>
                    <div class="result-card-value">Rs. ${formatNumber(data.predicted_price)}</div>
                    <div class="result-card-subvalue">per Perch</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Reliability Score</h3>
                    </div>
                    <div class="result-card-value">${data.reliability_score}%</div>
                    <div class="result-card-subvalue">${data.reliability_score >= 80 ? 'High Confidence' : data.reliability_score >= 60 ? 'Moderate' : 'Lower Confidence'}</div>
                </div>
            </div>

            ${data.confidence_lower && data.confidence_upper ? `
                <div class="confidence-range">
                    <h4>${Math.round(data.confidence_level * 100)}% Confidence Interval</h4>
                    <div class="range-values">
                        <div class="range-item">
                            <div class="range-label">Lower Bound</div>
                            <div class="range-value">Rs. ${formatNumber(data.confidence_lower)}</div>
                        </div>
                        <div class="range-item">
                            <div class="range-label">Upper Bound</div>
                            <div class="range-value">Rs. ${formatNumber(data.confidence_upper)}</div>
                        </div>
                    </div>
                </div>
            ` : ''}

            <div class="info-box">
                <h4>Context & Historical Data</h4>
                <ul class="info-list">
                    <li>
                        <span>Latest Known Price (${data.latest_year})</span>
                        <strong>Rs. ${formatNumber(data.latest_price)}</strong>
                    </li>
                    ${data.years_diff > 0 ? `
                        <li>
                            <span>Expected Change</span>
                            <strong>${data.price_change_percent >= 0 ? '+' : ''}${data.price_change_percent.toFixed(2)}% over ${data.years_diff} years</strong>
                        </li>
                    ` : ''}
                </ul>
            </div>

            ${data.is_extrapolation ? `
                <div class="warning-box">
                    <h4>
                        <svg width="20" height="20" fill="currentColor">
                            <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
                        </svg>
                        Warning - Extrapolation
                    </h4>
                    <p>This prediction extends <strong>${data.extrapolation_years} years</strong> beyond training data (2024). 
                    Accuracy decreases with longer time horizons. Use these predictions with appropriate caution and consider them as estimates with moderate to high uncertainty.</p>
                </div>
            ` : ''}
        `;

        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Display investment results
    function displayInvestmentResults(data) {
        const profitPositive = data.profit >= 0;
        const html = `
            <div class="result-header">
                <h2 class="result-title">
                    <svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                    </svg>
                    Investment Analysis Report
                </h2>
                ${data.is_extrapolation ? 
                    `<span class="result-badge badge-warning">
                        <svg width="16" height="16" fill="currentColor">
                            <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
                        </svg>
                        Extrapolation
                    </span>` : 
                    `<span class="result-badge badge-success">
                        <svg width="16" height="16" fill="currentColor">
                            <path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/>
                        </svg>
                        Analysis Complete
                    </span>`
                }
            </div>

            <div class="result-grid">
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4z"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Property Details</h3>
                    </div>
                    <div class="result-card-value">${data.num_perches} Perches</div>
                    <div class="result-card-subvalue">${data.area_name}</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="3" width="18" height="18" rx="2"/>
                                <path d="M8 12h8m-8 4h8"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Total Investment</h3>
                    </div>
                    <div class="result-card-value">Rs. ${formatNumber(data.total_investment)}</div>
                    <div class="result-card-subvalue">Rs. ${formatNumber(data.purchase_price)} per perch</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 6v6l4 2"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Holding Period</h3>
                    </div>
                    <div class="result-card-value">${data.holding_period_years} Years</div>
                    <div class="result-card-subvalue">${data.purchase_year} → ${data.sell_year}</div>
                </div>
            </div>

            <div class="result-grid">
                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Estimated Profit</h3>
                    </div>
                    <div class="result-card-value" style="color: ${profitPositive ? '#10b981' : '#ef4444'}">
                        Rs. ${formatNumber(data.profit)}
                    </div>
                    <div class="result-card-subvalue">Total Return: Rs. ${formatNumber(data.total_return)}</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">Return on Investment</h3>
                    </div>
                    <div class="result-card-value">${data.roi.toFixed(2)}%</div>
                    <div class="result-card-subvalue">ROI over ${data.holding_period_years} years</div>
                </div>

                <div class="result-card">
                    <div class="result-card-header">
                        <div class="result-card-icon">
                            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 3v18h18"/>
                                <path d="M7 16l4-8 4 4 4-8"/>
                            </svg>
                        </div>
                        <h3 class="result-card-title">CAGR</h3>
                    </div>
                    <div class="result-card-value">${data.cagr.toFixed(2)}%</div>
                    <div class="result-card-subvalue">Per Year Growth Rate</div>
                </div>
            </div>

            ${data.profit_lower && data.profit_upper ? `
                <div class="confidence-range">
                    <h4>${Math.round(data.confidence_level * 100)}% Confidence Interval</h4>
                    <div class="range-values">
                        <div class="range-item">
                            <div class="range-label">Profit Range (Lower)</div>
                            <div class="range-value">Rs. ${formatNumber(data.profit_lower)}</div>
                            <div class="range-label" style="margin-top: 0.5rem">ROI: ${data.roi_lower.toFixed(2)}%</div>
                        </div>
                        <div class="range-item">
                            <div class="range-label">Profit Range (Upper)</div>
                            <div class="range-value">Rs. ${formatNumber(data.profit_upper)}</div>
                            <div class="range-label" style="margin-top: 0.5rem">ROI: ${data.roi_upper.toFixed(2)}%</div>
                        </div>
                    </div>
                </div>
            ` : ''}

            <div class="info-box">
                <h4>Investment Summary</h4>
                <ul class="info-list">
                    <li>
                        <span>Predicted Selling Price per Perch</span>
                        <strong>Rs. ${formatNumber(data.predicted_price)}</strong>
                    </li>
                    <li>
                        <span>Purchase Price per Perch</span>
                        <strong>Rs. ${formatNumber(data.purchase_price)}</strong>
                    </li>
                    <li>
                        <span>Price Appreciation</span>
                        <strong>${(((data.predicted_price / data.purchase_price) - 1) * 100).toFixed(2)}%</strong>
                    </li>
                </ul>
            </div>

            ${data.is_extrapolation ? `
                <div class="warning-box">
                    <h4>
                        <svg width="20" height="20" fill="currentColor">
                            <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
                        </svg>
                        Warning - Extrapolation
                    </h4>
                    <p>This prediction extends <strong>${data.extrapolation_years} years</strong> beyond training data (2024). 
                    Reliability Score: <strong>${data.reliability_score}%</strong>. 
                    ${data.extrapolation_years >= 10 ? 
                        'For long-term investments, consider re-checking predictions every 2-3 years with updated data for better accuracy. ' : 
                        'Consider this a reasonable estimate with moderate uncertainty. '}
                    Use these projections as ONE data point in your investment decision, not the sole indicator.</p>
                </div>
            ` : ''}
        `;

        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Format numbers with commas
    function formatNumber(num) {
        return num.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
});
