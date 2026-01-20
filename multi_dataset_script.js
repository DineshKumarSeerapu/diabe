// Multi-Dataset Diabetes Prediction App JavaScript
class MultiDatasetDiabetesApp {
    constructor() {
        this.currentDataset = 'combined';
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.resultsSection = document.getElementById('resultsSection');

        this.initializeApp();
    }

    initializeApp() {
        this.setupDatasetSelection();
        this.checkModelStatus();
        this.setupEventListeners();
    }

    setupDatasetSelection() {
        const datasetCards = document.querySelectorAll('.dataset-card');

        datasetCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove active class from all cards
                datasetCards.forEach(c => c.classList.remove('active'));

                // Add active class to clicked card
                card.classList.add('active');

                // Update current dataset
                this.currentDataset = card.dataset.dataset;

                // Switch forms
                this.switchForm();

                // Check model status for new dataset
                this.checkModelStatus();
            });
        });
    }

    switchForm() {
        const standardForm = document.getElementById('standard-form');
        const iraqiForm = document.getElementById('iraqi-form');

        if (this.currentDataset === 'iraqi') {
            standardForm.style.display = 'none';
            iraqiForm.style.display = 'block';
        } else {
            standardForm.style.display = 'block';
            iraqiForm.style.display = 'none';
        }

        // Reset results
        this.resultsSection.style.display = 'none';
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/api/model/status');
            const status = await response.json();

            const modelStatus = document.getElementById('modelStatus');
            const currentStatus = status[this.currentDataset];

            if (currentStatus && currentStatus.trained && currentStatus.available) {
                modelStatus.innerHTML = '<i class="fas fa-check-circle"></i><span>Model Ready: ' + 
                                       this.getDatasetDisplayName() + '</span>';
                modelStatus.classList.remove('status-error');
                modelStatus.classList.add('status-success');
            } else {
                modelStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Model Training: ' + 
                                       this.getDatasetDisplayName() + '</span>';
                modelStatus.classList.remove('status-success');
                modelStatus.classList.add('status-error');
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            const modelStatus = document.getElementById('modelStatus');
            modelStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Connection Error</span>';
            modelStatus.classList.add('status-error');
        }
    }

    getDatasetDisplayName() {
        const names = {
            'pima': 'Pima Indian Dataset',
            'frankfurt': 'Frankfurt Hospital Dataset',
            'iraqi': 'Iraqi Patient Dataset',
            'combined': 'Combined Multi-Dataset'
        };
        return names[this.currentDataset] || this.currentDataset;
    }

    setupEventListeners() {
        // Predict button
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.handlePrediction();
        });

        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetForms();
        });

        // Form validation
        this.setupFormValidation();
    }

    setupFormValidation() {
        const forms = [document.getElementById('standardForm'), document.getElementById('iraqiForm')];

        forms.forEach(form => {
            if (form) {
                const inputs = form.querySelectorAll('input, select');
                inputs.forEach(input => {
                    input.addEventListener('input', () => {
                        this.validateInput(input);
                    });
                });
            }
        });
    }

    validateInput(input) {
        input.classList.remove('input-warning', 'input-error');

        const value = parseFloat(input.value);
        if (input.type === 'number') {
            const min = parseFloat(input.min) || 0;
            const max = parseFloat(input.max) || Infinity;

            if (isNaN(value) || value < min || value > max) {
                input.classList.add('input-error');
                return false;
            }
        }

        return true;
    }

    async handlePrediction() {
        const formData = this.getFormData();

        if (!formData) {
            this.showAlert('Please fill in all fields correctly.', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset: this.currentDataset,
                    data: formData
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showAlert(error.message || 'Error making prediction. Please try again.', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    getFormData() {
        let form;
        if (this.currentDataset === 'iraqi') {
            form = document.getElementById('iraqiForm');
        } else {
            form = document.getElementById('standardForm');
        }

        const formData = new FormData(form);
        const data = {};
        let isValid = true;

        for (let [key, value] of formData.entries()) {
            if (value === '') {
                isValid = false;
                break;
            }

            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                isValid = false;
                break;
            }

            data[key] = numValue;
        }

        return isValid ? data : null;
    }

    displayResults(result) {
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });

        const predictionResult = document.getElementById('predictionResult');
        const riskDisplay = document.getElementById('riskDisplay');
        const explanationContent = document.getElementById('explanationContent');
        const datasetInfoContent = document.getElementById('datasetInfoContent');

        if (result.target_type === 'binary') {
            this.displayBinaryResults(result);
        } else {
            this.displayMulticlassResults(result);
        }

        // Display explanation
        explanationContent.textContent = result.explanation || 'Explanation not available.';

        // Display dataset information
        datasetInfoContent.innerHTML = `
            <strong>Dataset:</strong> ${this.getDatasetDisplayName()}<br>
            <strong>Model Type:</strong> ${result.target_type === 'binary' ? 'Binary Classification' : 'Multi-class Classification'}<br>
            <strong>Features Used:</strong> ${result.target_type === 'binary' ? '8 clinical parameters' : '12 laboratory parameters'}
        `;
    }

    displayBinaryResults(result) {
        const predictionResult = document.getElementById('predictionResult');
        const riskDisplay = document.getElementById('riskDisplay');

        // Prediction
        if (result.prediction === 1) {
            predictionResult.textContent = 'Diabetes Risk Detected';
            predictionResult.className = 'prediction-result prediction-positive';
        } else {
            predictionResult.textContent = 'No Diabetes Risk';
            predictionResult.className = 'prediction-result prediction-negative';
        }

        // Risk display
        const diabetesProbability = result.probability.diabetes * 100;
        const riskLevel = result.risk_level;

        riskDisplay.innerHTML = `
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${diabetesProbability}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <span style="font-weight: 600;">${diabetesProbability.toFixed(1)}% Risk</span>
                <span class="risk-level risk-${riskLevel.toLowerCase().replace(' ', '-')}">${riskLevel}</span>
            </div>
        `;
    }

    displayMulticlassResults(result) {
        const predictionResult = document.getElementById('predictionResult');
        const riskDisplay = document.getElementById('riskDisplay');

        // Prediction
        const classNames = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'];
        const prediction = classNames[result.prediction];

        predictionResult.textContent = prediction;
        predictionResult.className = `prediction-result class-result ${prediction.toLowerCase().replace('-', '-')}`;

        // Probability display for all classes
        let probDisplay = '<div class="probability-bars">';
        Object.entries(result.probabilities).forEach(([className, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            let color = '#4CAF50';
            if (className === 'Pre-Diabetic') color = '#FFC107';
            if (className === 'Diabetic') color = '#f44336';

            probDisplay += `
                <div class="prob-bar">
                    <div class="prob-label">${className}:</div>
                    <div class="prob-visual">
                        <div class="prob-fill" style="width: ${percentage}%; background-color: ${color};"></div>
                    </div>
                    <div class="prob-text">${percentage}%</div>
                </div>
            `;
        });
        probDisplay += '</div>';

        riskDisplay.innerHTML = probDisplay;
    }

    resetForms() {
        const forms = [document.getElementById('standardForm'), document.getElementById('iraqiForm')];

        forms.forEach(form => {
            if (form) {
                form.reset();

                // Reset to default values
                if (form.id === 'standardForm') {
                    document.getElementById('pregnancies').value = '0';
                    document.getElementById('glucose').value = '120';
                    document.getElementById('bloodPressure').value = '80';
                    document.getElementById('skinThickness').value = '20';
                    document.getElementById('insulin').value = '80';
                    document.getElementById('bmi').value = '25.0';
                    document.getElementById('diabetesPedigree').value = '0.5';
                    document.getElementById('age').value = '30';
                } else if (form.id === 'iraqiForm') {
                    document.getElementById('gender').value = '0';
                    document.getElementById('age_iraqi').value = '50';
                    document.getElementById('fbs').value = '7.0';
                    document.getElementById('bun').value = '5.0';
                    document.getElementById('cr').value = '70';
                    document.getElementById('chol').value = '5.0';
                    document.getElementById('tg').value = '2.0';
                    document.getElementById('bmi_iraqi').value = '28.0';
                    document.getElementById('ldl').value = '3.0';
                    document.getElementById('vldl').value = '1.0';
                    document.getElementById('hdl').value = '1.2';
                    document.getElementById('hba1c').value = '7.0';
                }

                // Remove validation classes
                const inputs = form.querySelectorAll('input, select');
                inputs.forEach(input => {
                    input.classList.remove('input-warning', 'input-error');
                });
            }
        });

        this.resultsSection.style.display = 'none';
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        alert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease;
            background: ${type === 'error' ? '#f44336' : '#2196F3'};
        `;

        document.body.appendChild(alert);

        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
}

// Add CSS animations
const styles = `
@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.input-error {
    border-color: #f44336 !important;
    box-shadow: 0 0 0 3px rgba(244, 67, 54, 0.1) !important;
}

.input-warning {
    border-color: #FFC107 !important;
    box-shadow: 0 0 0 3px rgba(255, 193, 7, 0.1) !important;
}
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MultiDatasetDiabetesApp();
});