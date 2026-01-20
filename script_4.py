# Create directories and files for the multi-dataset system
import os

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# Now create the HTML template
multi_dataset_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Dataset Diabetes Prediction - Advanced AI System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/multi_dataset_style.css') }}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <i class="fas fa-database"></i>
                <h1>Multi-Dataset Diabetes Prediction System</h1>
                <p>Advanced AI with Pima, Frankfurt, and Iraqi Datasets | Ensemble Methods & Explainable AI</p>
            </div>
        </header>

        <!-- Dataset Selection -->
        <div class="dataset-selection">
            <h2><i class="fas fa-chart-bar"></i> Select Dataset & Model</h2>
            <div class="dataset-cards">
                <div class="dataset-card" data-dataset="pima">
                    <div class="dataset-icon">
                        <i class="fas fa-user-friends"></i>
                    </div>
                    <h3>Pima Indian Dataset</h3>
                    <p>768 samples | 8 features | Binary classification</p>
                    <div class="dataset-features">
                        <small>Classic NIDDK diabetes dataset with Pima Indian heritage patients</small>
                    </div>
                </div>

                <div class="dataset-card" data-dataset="frankfurt">
                    <div class="dataset-icon">
                        <i class="fas fa-hospital"></i>
                    </div>
                    <h3>Frankfurt Hospital</h3>
                    <p>2000 samples | 8 features | Binary classification</p>
                    <div class="dataset-features">
                        <small>Large-scale German hospital diabetes dataset</small>
                    </div>
                </div>

                <div class="dataset-card" data-dataset="iraqi">
                    <div class="dataset-icon">
                        <i class="fas fa-flask"></i>
                    </div>
                    <h3>Iraqi Patient Dataset</h3>
                    <p>1000 samples | 12 features | Multi-class classification</p>
                    <div class="dataset-features">
                        <small>Comprehensive lab analysis with 3 diabetes classes</small>
                    </div>
                </div>

                <div class="dataset-card active" data-dataset="combined">
                    <div class="dataset-icon">
                        <i class="fas fa-layer-group"></i>
                    </div>
                    <h3>Combined Dataset</h3>
                    <p>3768 samples | 8+ features | Advanced ensemble</p>
                    <div class="dataset-features">
                        <small>Harmonized multi-dataset with feature mapping</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Status -->
        <div class="status-section">
            <div class="status-cards">
                <div class="status-card" id="modelStatus">
                    <i class="fas fa-cog fa-spin"></i>
                    <span>Checking model status...</span>
                </div>
            </div>
        </div>

        <!-- Input Forms -->
        <div class="main-content">
            <!-- Pima/Frankfurt/Combined Form -->
            <div class="form-section" id="standard-form">
                <h2><i class="fas fa-user-md"></i> Patient Information</h2>
                <form id="standardForm" class="diabetes-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="pregnancies"><i class="fas fa-baby"></i> Pregnancies</label>
                            <input type="number" id="pregnancies" name="Pregnancies" min="0" max="20" value="0" required>
                            <small>Number of times pregnant</small>
                        </div>
                        <div class="form-group">
                            <label for="glucose"><i class="fas fa-tint"></i> Glucose (mg/dL)</label>
                            <input type="number" id="glucose" name="Glucose" min="0" max="300" value="120" required>
                            <small>Plasma glucose concentration</small>
                        </div>
                        <div class="form-group">
                            <label for="bloodPressure"><i class="fas fa-heart"></i> Blood Pressure (mmHg)</label>
                            <input type="number" id="bloodPressure" name="BloodPressure" min="0" max="200" value="80" required>
                            <small>Diastolic blood pressure</small>
                        </div>
                        <div class="form-group">
                            <label for="skinThickness"><i class="fas fa-ruler"></i> Skin Thickness (mm)</label>
                            <input type="number" id="skinThickness" name="SkinThickness" min="0" max="100" value="20" required>
                            <small>Triceps skin fold thickness</small>
                        </div>
                        <div class="form-group">
                            <label for="insulin"><i class="fas fa-syringe"></i> Insulin (μU/mL)</label>
                            <input type="number" id="insulin" name="Insulin" min="0" max="900" value="80" required>
                            <small>2-hour serum insulin</small>
                        </div>
                        <div class="form-group">
                            <label for="bmi"><i class="fas fa-weight"></i> BMI (kg/m²)</label>
                            <input type="number" id="bmi" name="BMI" min="0" max="70" step="0.1" value="25.0" required>
                            <small>Body Mass Index</small>
                        </div>
                        <div class="form-group">
                            <label for="diabetesPedigree"><i class="fas fa-dna"></i> Diabetes Pedigree Function</label>
                            <input type="number" id="diabetesPedigree" name="DiabetesPedigreeFunction" min="0" max="3" step="0.001" value="0.5" required>
                            <small>Genetic diabetes risk factor</small>
                        </div>
                        <div class="form-group">
                            <label for="age"><i class="fas fa-calendar-alt"></i> Age (years)</label>
                            <input type="number" id="age" name="Age" min="18" max="120" value="30" required>
                            <small>Age in years</small>
                        </div>
                    </div>
                </form>
            </div>

            <!-- Iraqi Form -->
            <div class="form-section" id="iraqi-form" style="display: none;">
                <h2><i class="fas fa-flask"></i> Laboratory Analysis - Iraqi Dataset</h2>
                <form id="iraqiForm" class="diabetes-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="gender"><i class="fas fa-venus-mars"></i> Gender</label>
                            <select id="gender" name="Gender" required>
                                <option value="0">Female</option>
                                <option value="1">Male</option>
                            </select>
                            <small>Patient gender</small>
                        </div>
                        <div class="form-group">
                            <label for="age_iraqi"><i class="fas fa-calendar-alt"></i> Age (years)</label>
                            <input type="number" id="age_iraqi" name="Age" min="20" max="79" value="50" required>
                            <small>Age in years</small>
                        </div>
                        <div class="form-group">
                            <label for="fbs"><i class="fas fa-tint"></i> FBS (mmol/L)</label>
                            <input type="number" id="fbs" name="FBS" min="3" max="25" step="0.1" value="7.0" required>
                            <small>Fasting Blood Sugar</small>
                        </div>
                        <div class="form-group">
                            <label for="bun"><i class="fas fa-vial"></i> BUN (mmol/L)</label>
                            <input type="number" id="bun" name="BUN" min="1" max="15" step="0.1" value="5.0" required>
                            <small>Blood Urea Nitrogen</small>
                        </div>
                        <div class="form-group">
                            <label for="cr"><i class="fas fa-microscope"></i> Cr (μmol/L)</label>
                            <input type="number" id="cr" name="Cr" min="20" max="300" value="70" required>
                            <small>Creatinine level</small>
                        </div>
                        <div class="form-group">
                            <label for="chol"><i class="fas fa-heart"></i> Cholesterol (mmol/L)</label>
                            <input type="number" id="chol" name="Chol" min="2" max="12" step="0.1" value="5.0" required>
                            <small>Total cholesterol</small>
                        </div>
                        <div class="form-group">
                            <label for="tg"><i class="fas fa-droplet"></i> TG (mmol/L)</label>
                            <input type="number" id="tg" name="TG" min="0.5" max="8" step="0.1" value="2.0" required>
                            <small>Triglycerides</small>
                        </div>
                        <div class="form-group">
                            <label for="bmi_iraqi"><i class="fas fa-weight"></i> BMI (kg/m²)</label>
                            <input type="number" id="bmi_iraqi" name="BMI" min="18" max="50" step="0.1" value="28.0" required>
                            <small>Body Mass Index</small>
                        </div>
                        <div class="form-group">
                            <label for="ldl"><i class="fas fa-chart-line"></i> LDL (mmol/L)</label>
                            <input type="number" id="ldl" name="LDL" min="0.5" max="8" step="0.1" value="3.0" required>
                            <small>Low Density Lipoprotein</small>
                        </div>
                        <div class="form-group">
                            <label for="vldl"><i class="fas fa-chart-area"></i> VLDL (mmol/L)</label>
                            <input type="number" id="vldl" name="VLDL" min="0.1" max="3" step="0.1" value="1.0" required>
                            <small>Very Low Density Lipoprotein</small>
                        </div>
                        <div class="form-group">
                            <label for="hdl"><i class="fas fa-chart-pie"></i> HDL (mmol/L)</label>
                            <input type="number" id="hdl" name="HDL" min="0.3" max="3" step="0.1" value="1.2" required>
                            <small>High Density Lipoprotein</small>
                        </div>
                        <div class="form-group">
                            <label for="hba1c"><i class="fas fa-thermometer-half"></i> HbA1C (%)</label>
                            <input type="number" id="hba1c" name="HbA1C" min="4" max="18" step="0.1" value="7.0" required>
                            <small>Glycated Hemoglobin</small>
                        </div>
                    </div>
                </form>
            </div>

            <!-- Form Actions -->
            <div class="form-actions">
                <button type="button" class="btn-predict" id="predictBtn">
                    <i class="fas fa-brain"></i> Predict Diabetes Risk
                </button>
                <button type="button" class="btn-reset" id="resetBtn">
                    <i class="fas fa-undo"></i> Reset Form
                </button>
            </div>

            <!-- Results Section -->
            <div class="results-section" id="resultsSection" style="display: none;">
                <h2><i class="fas fa-chart-line"></i> Prediction Results</h2>
                
                <div class="result-cards" id="resultCards">
                    <div class="result-card prediction-card">
                        <div class="result-icon"><i class="fas fa-diagnoses"></i></div>
                        <div class="result-content">
                            <h3>Prediction</h3>
                            <div class="prediction-result" id="predictionResult"></div>
                        </div>
                    </div>
                    <div class="result-card probability-card">
                        <div class="result-icon"><i class="fas fa-percentage"></i></div>
                        <div class="result-content">
                            <h3>Risk Assessment</h3>
                            <div class="risk-display" id="riskDisplay"></div>
                        </div>
                    </div>
                </div>

                <div class="explanation-card">
                    <h3><i class="fas fa-lightbulb"></i> AI Explanation</h3>
                    <div class="explanation-content" id="explanationContent"></div>
                </div>

                <div class="dataset-info-card">
                    <h3><i class="fas fa-info-circle"></i> Model Information</h3>
                    <div class="dataset-info-content" id="datasetInfoContent"></div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>&copy; 2024 Multi-Dataset Diabetes Prediction System</p>
            <p><small>Powered by Advanced Ensemble AI & Explainable AI</small></p>
        </footer>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <p>Processing with AI models...</p>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/multi_dataset_script.js') }}"></script>
</body>
</html>'''

# Save HTML template
with open('templates/multi_dataset_index.html', 'w') as f:
    f.write(multi_dataset_html)

print("✅ Created: templates/multi_dataset_index.html")

# Create the updated CSS file
multi_dataset_css = '''/* Multi-Dataset Diabetes Prediction System Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.header-content i {
    font-size: 3rem;
    color: #667eea;
    margin-bottom: 15px;
}

.header h1 {
    color: #333;
    font-size: 2.5rem;
    margin-bottom: 10px;
    font-weight: 700;
}

.header p {
    color: #666;
    font-size: 1.1rem;
}

/* Dataset Selection */
.dataset-selection {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.dataset-selection h2 {
    color: #333;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.dataset-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.dataset-card {
    background: white;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
}

.dataset-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.dataset-card.active {
    border-color: #667eea;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.dataset-card.active .dataset-features small {
    color: rgba(255, 255, 255, 0.9);
}

.dataset-icon {
    font-size: 2rem;
    margin-bottom: 15px;
    color: #667eea;
}

.dataset-card.active .dataset-icon {
    color: white;
}

.dataset-card h3 {
    margin-bottom: 10px;
    font-size: 1.3rem;
}

.dataset-card p {
    font-weight: 600;
    margin-bottom: 10px;
}

.dataset-features small {
    color: #666;
    font-style: italic;
}

/* Status Section */
.status-section {
    margin-bottom: 30px;
}

.status-cards {
    display: flex;
    gap: 15px;
    justify-content: center;
    flex-wrap: wrap;
}

.status-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    padding: 15px 25px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.status-card.status-success {
    background: rgba(76, 175, 80, 0.1);
    border-left: 4px solid #4CAF50;
}

.status-card.status-error {
    background: rgba(244, 67, 54, 0.1);
    border-left: 4px solid #f44336;
}

/* Form Sections */
.form-section {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.form-section h2 {
    color: #333;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.form-group {
    display: flex;
    flex-direction: column;
}

.form-group label {
    font-weight: 600;
    margin-bottom: 8px;
    color: #333;
    display: flex;
    align-items: center;
    gap: 8px;
}

.form-group input,
.form-group select {
    padding: 12px;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-group small {
    color: #666;
    margin-top: 5px;
    font-size: 0.9rem;
}

/* Form Actions */
.form-actions {
    display: flex;
    gap: 15px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 30px;
}

.btn-predict, .btn-reset, .btn-train {
    padding: 15px 30px;
    border: none;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 10px;
}

.btn-predict {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.btn-predict:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
}

.btn-reset {
    background: #f8f9fa;
    color: #666;
    border: 2px solid #ddd;
}

.btn-reset:hover {
    background: #e9ecef;
}

.btn-train {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
}

/* Results Section */
.results-section {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.5s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.results-section h2 {
    color: #333;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Result Cards */
.result-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.result-card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.result-card:hover {
    transform: translateY(-3px);
}

.result-icon {
    font-size: 2.5rem;
    margin-bottom: 15px;
    color: #667eea;
}

.result-content h3 {
    color: #333;
    margin-bottom: 15px;
    font-size: 1.2rem;
}

.prediction-result {
    font-size: 1.4rem;
    font-weight: 700;
    padding: 10px;
    border-radius: 8px;
}

.prediction-positive {
    background: rgba(244, 67, 54, 0.1);
    color: #f44336;
}

.prediction-negative {
    background: rgba(76, 175, 80, 0.1);
    color: #4CAF50;
}

.risk-display {
    margin-top: 10px;
}

.probability-bar {
    width: 100%;
    height: 20px;
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 10px;
}

.probability-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #FFC107, #f44336);
    border-radius: 10px;
    transition: width 1s ease;
}

.risk-level {
    font-size: 1.2rem;
    font-weight: 700;
    padding: 8px 15px;
    border-radius: 20px;
}

.risk-low {
    background: rgba(76, 175, 80, 0.1);
    color: #4CAF50;
}

.risk-moderate {
    background: rgba(255, 193, 7, 0.1);
    color: #FFC107;
}

.risk-high {
    background: rgba(244, 67, 54, 0.1);
    color: #f44336;
}

/* Multiclass Results */
.multiclass-prediction {
    text-align: center;
    margin-bottom: 20px;
}

.class-result {
    font-size: 1.5rem;
    font-weight: 700;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.class-result.non-diabetic {
    background: rgba(76, 175, 80, 0.1);
    color: #4CAF50;
}

.class-result.pre-diabetic {
    background: rgba(255, 193, 7, 0.1);
    color: #FFC107;
}

.class-result.diabetic {
    background: rgba(244, 67, 54, 0.1);
    color: #f44336;
}

.probability-bars {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.prob-bar {
    display: flex;
    align-items: center;
    gap: 10px;
}

.prob-label {
    min-width: 120px;
    font-weight: 600;
}

.prob-visual {
    flex: 1;
    height: 25px;
    background: #f0f0f0;
    border-radius: 12px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 1s ease;
}

.prob-text {
    min-width: 50px;
    font-weight: 600;
    text-align: right;
}

/* Explanation and Info Cards */
.explanation-card, .dataset-info-card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.explanation-card h3, .dataset-info-card h3 {
    color: #333;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.explanation-content, .dataset-info-content {
    color: #555;
    line-height: 1.8;
    white-space: pre-line;
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: none;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.loading-content {
    background: white;
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: rgba(255, 255, 255, 0.8);
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 15px;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .form-grid {
        grid-template-columns: 1fr;
    }
    
    .form-actions {
        flex-direction: column;
    }
    
    .dataset-cards {
        grid-template-columns: 1fr;
    }
    
    .result-cards {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 480px) {
    .header-content i {
        font-size: 2rem;
    }
    
    .header h1 {
        font-size: 1.5rem;
    }
    
    .form-section, .results-section {
        padding: 20px;
    }
}'''

# Save CSS file
with open('static/css/multi_dataset_style.css', 'w') as f:
    f.write(multi_dataset_css)

print("✅ Created: static/css/multi_dataset_style.css")

# Create the JavaScript file
multi_dataset_js = '''// Multi-Dataset Diabetes Prediction App JavaScript
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
});'''

# Save JavaScript file
with open('static/js/multi_dataset_script.js', 'w') as f:
    f.write(multi_dataset_js)

print("✅ Created: static/js/multi_dataset_script.js")

# Create updated requirements
updated_requirements = '''Flask==2.3.3
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
catboost==0.26
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1
lime==0.2.0.1
Werkzeug==2.3.7
gunicorn==21.2.0
scipy==1.11.1'''

with open('multi_dataset_requirements.txt', 'w') as f:
    f.write(updated_requirements)

print("✅ Created: multi_dataset_requirements.txt")

print("\n🎉 MULTI-DATASET DIABETES PREDICTION SYSTEM CREATED!")
print("=" * 60)
print("📁 Complete file structure:")
print("├── pima_diabetes_dataset.csv")
print("├── frankfurt_diabetes_dataset.csv") 
print("├── iraqi_diabetes_dataset.csv")
print("├── multi_dataset_diabetes_predictor.py")
print("├── multi_dataset_app.py")
print("├── multi_dataset_requirements.txt")
print("├── templates/")
print("│   └── multi_dataset_index.html")
print("└── static/")
print("    ├── css/")
print("    │   └── multi_dataset_style.css")
print("    └── js/")
print("        └── multi_dataset_script.js")

print("\n🚀 TO RUN THE SYSTEM:")
print("1. pip install -r multi_dataset_requirements.txt")
print("2. python multi_dataset_app.py")
print("3. Open http://localhost:5000")

print("\n✨ FEATURES INTEGRATED:")
print("✅ Pima Indian Diabetes Dataset (PIMA) - 768 samples, 8 features")
print("✅ Frankfurt Hospital Germany Dataset - 2000 samples, 8 features") 
print("✅ Iraqi Patient Dataset (IPDD) - 1000 samples, 12 features")
print("✅ Combined multi-dataset with feature harmonization")
print("✅ Binary and multiclass classification")
print("✅ Advanced ensemble methods (XGBoost, LightGBM, etc.)")
print("✅ Explainable AI with SHAP integration")
print("✅ Responsive web interface with dataset switching")
print("✅ Real-time model training and evaluation")