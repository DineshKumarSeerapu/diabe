# Create updated HTML template for multi-dataset support
multi_dataset_html = '''
<!DOCTYPE html>
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
                <div class="status-card" id="performanceStatus" style="display: none;">
                    <i class="fas fa-chart-line"></i>
                    <span>Performance metrics loading...</span>
                </div>
            </div>
        </div>

        <!-- Input Forms (Different for each dataset) -->
        <div class="main-content">
            <!-- Pima/Frankfurt Form -->
            <div class="form-section" id="pima-frankfurt-form">
                <h2><i class="fas fa-user-md"></i> Patient Information - Pima/Frankfurt Dataset</h2>
                <form id="pimaFrankfurtForm" class="diabetes-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="pregnancies">
                                <i class="fas fa-baby"></i> Pregnancies
                            </label>
                            <input type="number" id="pregnancies" name="Pregnancies" min="0" max="20" value="0" required>
                            <small>Number of times pregnant</small>
                        </div>

                        <div class="form-group">
                            <label for="glucose">
                                <i class="fas fa-tint"></i> Glucose (mg/dL)
                            </label>
                            <input type="number" id="glucose" name="Glucose" min="0" max="300" value="120" required>
                            <small>Plasma glucose concentration</small>
                        </div>

                        <div class="form-group">
                            <label for="bloodPressure">
                                <i class="fas fa-heart"></i> Blood Pressure (mmHg)
                            </label>
                            <input type="number" id="bloodPressure" name="BloodPressure" min="0" max="200" value="80" required>
                            <small>Diastolic blood pressure</small>
                        </div>

                        <div class="form-group">
                            <label for="skinThickness">
                                <i class="fas fa-ruler"></i> Skin Thickness (mm)
                            </label>
                            <input type="number" id="skinThickness" name="SkinThickness" min="0" max="100" value="20" required>
                            <small>Triceps skin fold thickness</small>
                        </div>

                        <div class="form-group">
                            <label for="insulin">
                                <i class="fas fa-syringe"></i> Insulin (μU/mL)
                            </label>
                            <input type="number" id="insulin" name="Insulin" min="0" max="900" value="80" required>
                            <small>2-hour serum insulin</small>
                        </div>

                        <div class="form-group">
                            <label for="bmi">
                                <i class="fas fa-weight"></i> BMI (kg/m²)
                            </label>
                            <input type="number" id="bmi" name="BMI" min="0" max="70" step="0.1" value="25.0" required>
                            <small>Body Mass Index</small>
                        </div>

                        <div class="form-group">
                            <label for="diabetesPedigree">
                                <i class="fas fa-dna"></i> Diabetes Pedigree Function
                            </label>
                            <input type="number" id="diabetesPedigree" name="DiabetesPedigreeFunction" 
                                   min="0" max="3" step="0.001" value="0.5" required>
                            <small>Genetic diabetes risk factor</small>
                        </div>

                        <div class="form-group">
                            <label for="age">
                                <i class="fas fa-calendar-alt"></i> Age (years)
                            </label>
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
                            <label for="gender">
                                <i class="fas fa-venus-mars"></i> Gender
                            </label>
                            <select id="gender" name="Gender" required>
                                <option value="0">Female</option>
                                <option value="1">Male</option>
                            </select>
                            <small>Patient gender</small>
                        </div>

                        <div class="form-group">
                            <label for="age_iraqi">
                                <i class="fas fa-calendar-alt"></i> Age (years)
                            </label>
                            <input type="number" id="age_iraqi" name="Age" min="20" max="79" value="50" required>
                            <small>Age in years</small>
                        </div>

                        <div class="form-group">
                            <label for="fbs">
                                <i class="fas fa-tint"></i> FBS (mmol/L)
                            </label>
                            <input type="number" id="fbs" name="FBS" min="3" max="25" step="0.1" value="7.0" required>
                            <small>Fasting Blood Sugar</small>
                        </div>

                        <div class="form-group">
                            <label for="bun">
                                <i class="fas fa-vial"></i> BUN (mmol/L)
                            </label>
                            <input type="number" id="bun" name="BUN" min="1" max="15" step="0.1" value="5.0" required>
                            <small>Blood Urea Nitrogen</small>
                        </div>

                        <div class="form-group">
                            <label for="cr">
                                <i class="fas fa-microscope"></i> Cr (μmol/L)
                            </label>
                            <input type="number" id="cr" name="Cr" min="20" max="300" value="70" required>
                            <small>Creatinine level</small>
                        </div>

                        <div class="form-group">
                            <label for="chol">
                                <i class="fas fa-heart"></i> Cholesterol (mmol/L)
                            </label>
                            <input type="number" id="chol" name="Chol" min="2" max="12" step="0.1" value="5.0" required>
                            <small>Total cholesterol</small>
                        </div>

                        <div class="form-group">
                            <label for="tg">
                                <i class="fas fa-droplet"></i> TG (mmol/L)
                            </label>
                            <input type="number" id="tg" name="TG" min="0.5" max="8" step="0.1" value="2.0" required>
                            <small>Triglycerides</small>
                        </div>

                        <div class="form-group">
                            <label for="bmi_iraqi">
                                <i class="fas fa-weight"></i> BMI (kg/m²)
                            </label>
                            <input type="number" id="bmi_iraqi" name="BMI" min="18" max="50" step="0.1" value="28.0" required>
                            <small>Body Mass Index</small>
                        </div>

                        <div class="form-group">
                            <label for="ldl">
                                <i class="fas fa-chart-line"></i> LDL (mmol/L)
                            </label>
                            <input type="number" id="ldl" name="LDL" min="0.5" max="8" step="0.1" value="3.0" required>
                            <small>Low Density Lipoprotein</small>
                        </div>

                        <div class="form-group">
                            <label for="vldl">
                                <i class="fas fa-chart-area"></i> VLDL (mmol/L)
                            </label>
                            <input type="number" id="vldl" name="VLDL" min="0.1" max="3" step="0.1" value="1.0" required>
                            <small>Very Low Density Lipoprotein</small>
                        </div>

                        <div class="form-group">
                            <label for="hdl">
                                <i class="fas fa-chart-pie"></i> HDL (mmol/L)
                            </label>
                            <input type="number" id="hdl" name="HDL" min="0.3" max="3" step="0.1" value="1.2" required>
                            <small>High Density Lipoprotein</small>
                        </div>

                        <div class="form-group">
                            <label for="hba1c">
                                <i class="fas fa-thermometer-half"></i> HbA1C (%)
                            </label>
                            <input type="number" id="hba1c" name="HbA1C" min="4" max="18" step="0.1" value="7.0" required>
                            <small>Glycated Hemoglobin</small>
                        </div>
                    </div>
                </form>
            </div>

            <!-- Form Actions -->
            <div class="form-actions">
                <button type="button" class="btn-predict" id="predictBtn">
                    <i class="fas fa-brain"></i>
                    Predict Diabetes Risk
                </button>
                <button type="button" class="btn-reset" id="resetBtn">
                    <i class="fas fa-undo"></i>
                    Reset Form
                </button>
                <button type="button" class="btn-train" id="trainBtn" style="display: none;">
                    <i class="fas fa-cogs"></i>
                    Train Model
                </button>
            </div>

            <!-- Results Section -->
            <div class="results-section" id="resultsSection" style="display: none;">
                <h2><i class="fas fa-chart-line"></i> Prediction Results</h2>
                
                <!-- Binary Results -->
                <div class="binary-results" id="binaryResults">
                    <div class="result-cards">
                        <div class="result-card prediction-card" id="predictionCard">
                            <div class="result-icon">
                                <i class="fas fa-diagnoses"></i>
                            </div>
                            <div class="result-content">
                                <h3>Prediction</h3>
                                <div class="prediction-result" id="predictionResult"></div>
                            </div>
                        </div>

                        <div class="result-card probability-card">
                            <div class="result-icon">
                                <i class="fas fa-percentage"></i>
                            </div>
                            <div class="result-content">
                                <h3>Risk Probability</h3>
                                <div class="probability-display">
                                    <div class="probability-bar">
                                        <div class="probability-fill" id="probabilityFill"></div>
                                    </div>
                                    <div class="probability-text" id="probabilityText"></div>
                                </div>
                            </div>
                        </div>

                        <div class="result-card risk-card" id="riskCard">
                            <div class="result-icon">
                                <i class="fas fa-exclamation-triangle"></i>
                            </div>
                            <div class="result-content">
                                <h3>Risk Level</h3>
                                <div class="risk-level" id="riskLevel"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Multiclass Results -->
                <div class="multiclass-results" id="multiclassResults" style="display: none;">
                    <div class="multiclass-prediction">
                        <h3>Classification Result</h3>
                        <div class="class-result" id="classResult"></div>
                    </div>
                    
                    <div class="class-probabilities">
                        <h3>Class Probabilities</h3>
                        <div class="probability-bars" id="probabilityBars"></div>
                    </div>
                </div>

                <div class="explanation-card">
                    <h3><i class="fas fa-lightbulb"></i> AI Explanation</h3>
                    <div class="explanation-content" id="explanationContent"></div>
                </div>

                <div class="recommendations-card">
                    <h3><i class="fas fa-user-md"></i> Health Recommendations</h3>
                    <div class="recommendations-content" id="recommendationsContent"></div>
                </div>

                <div class="dataset-info-card">
                    <h3><i class="fas fa-info-circle"></i> Dataset Information</h3>
                    <div class="dataset-info-content" id="datasetInfoContent"></div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>&copy; 2024 Multi-Dataset Diabetes Prediction System. Powered by Advanced Ensemble AI & XAI.</p>
            <p><small>Supporting Pima, Frankfurt, and Iraqi datasets with state-of-the-art machine learning.</small></p>
        </footer>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <p id="loadingText">Analyzing your health data...</p>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/multi_dataset_script.js') }}"></script>
</body>
</html>
'''

# Save the updated HTML template
with open('templates/multi_dataset_index.html', 'w') as f:
    f.write(multi_dataset_html)

print("✅ Created: templates/multi_dataset_index.html")
print("🎨 Features:")
print("- Dataset selection interface")
print("- Dynamic form switching")
print("- Pima/Frankfurt form (8 features)")
print("- Iraqi dataset form (12 features)")
print("- Binary and multiclass result display")
print("- Advanced UI/UX with dataset cards")