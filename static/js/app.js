// Global variables
let currentSection = 'dashboard';
let preprocessedData = null;
let trainedModels = null;
let isDataLoaded = false;
let isModelsTrained = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeEventListeners();
    updateStepStatuses();
});

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetSection = this.getAttribute('data-section');
            showSection(targetSection);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Step card navigation
    const stepCards = document.querySelectorAll('.step-card');
    stepCards.forEach(card => {
        card.addEventListener('click', function() {
            const targetSection = this.getAttribute('data-step');
            if (targetSection) {
                showSection(targetSection);
                
                // Update nav
                navLinks.forEach(l => l.classList.remove('active'));
                const targetNav = document.querySelector(`[data-section="${targetSection}"]`);
                if (targetNav) targetNav.classList.add('active');
            }
        });
    });
}

function showSection(sectionId) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        currentSection = sectionId;
    }
}

// Event listeners
function initializeEventListeners() {
    // Data preprocessing
    const loadDataBtn = document.getElementById('load-data-btn');
    if (loadDataBtn) {
        loadDataBtn.addEventListener('click', loadAndProcessData);
    }

    // Model training
    const trainModelsBtn = document.getElementById('train-models-btn');
    if (trainModelsBtn) {
        trainModelsBtn.addEventListener('click', trainModels);
    }

    // Prediction form
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', makePrediction);
    }
}

// Data preprocessing functions
async function loadAndProcessData() {
    showLoading(true, 'Loading and processing dataset...');
    
    try {
        const response = await fetch('/api/preprocess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            preprocessedData = result.data;
            displayPreprocessingResults(result.data);
            isDataLoaded = true;
            updateStepStatuses();
            updateQuickStats();
            
            // Enable training button
            const trainBtn = document.getElementById('train-models-btn');
            if (trainBtn) trainBtn.disabled = false;
            
            showSuccess('Data preprocessing completed successfully!');
        } else {
            showError('Error preprocessing data: ' + result.error);
        }
    } catch (error) {
        showError('Error loading data: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayPreprocessingResults(data) {
    // Show results container
    const resultsContainer = document.getElementById('preprocessing-results');
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
    }

    // Display data summary
    const summaryContainer = document.getElementById('data-summary');
    if (summaryContainer) {
        summaryContainer.innerHTML = `
            <div class="stat-item">
                <h5>Total Samples</h5>
                <p>${data.total_samples}</p>
            </div>
            <div class="stat-item">
                <h5>Training Samples</h5>
                <p>${data.train_samples}</p>
            </div>
            <div class="stat-item">
                <h5>Test Samples</h5>
                <p>${data.test_samples}</p>
            </div>
            <div class="stat-item">
                <h5>Features</h5>
                <p>${data.features}</p>
            </div>
            <div class="stat-item">
                <h5>Average Rentals</h5>
                <p>${Math.round(data.data_stats.mean_rentals)}</p>
            </div>
            <div class="stat-item">
                <h5>Max Rentals</h5>
                <p>${data.data_stats.max_rentals}</p>
            </div>
        `;
    }

    // Display data table
    displayDataTable(data.sample_data);
}

function displayDataTable(sampleData) {
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');
    
    if (!tableHeader || !tableBody || !sampleData.length) return;

    // Create header
    const headers = ['Date', 'Season', 'Weather', 'Temperature', 'Humidity', 'Wind Speed', 'Weekend', 'Holiday', 'Rentals'];
    tableHeader.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;

    // Create rows
    const rows = sampleData.slice(0, 10).map(row => `
        <tr>
            <td>${row.date}</td>
            <td>${getSeasonName(row.season)}</td>
            <td>${getWeatherName(row.weather_situation)}</td>
            <td>${row.temperature}°C</td>
            <td>${row.humidity}%</td>
            <td>${row.wind_speed} km/h</td>
            <td>${row.is_weekend ? 'Yes' : 'No'}</td>
            <td>${row.is_holiday ? 'Yes' : 'No'}</td>
            <td><strong>${row.count}</strong></td>
        </tr>
    `).join('');
    
    tableBody.innerHTML = rows;
}

// Model training functions
async function trainModels() {
    showLoading(true, 'Training machine learning models...');
    
    // Show progress container
    const progressContainer = document.getElementById('training-progress');
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }

    // Simulate training progress
    const progressFill = document.getElementById('progress-fill');
    const statusText = document.getElementById('training-status');
    
    const models = ['Linear Regression', 'Random Forest', 'Gradient Boosting'];
    let progress = 0;
    
    for (let i = 0; i < models.length; i++) {
        if (statusText) statusText.textContent = `Training ${models[i]}...`;
        
        // Simulate progress
        for (let j = 0; j < 33; j++) {
            progress++;
            if (progressFill) progressFill.style.width = progress + '%';
            await new Promise(resolve => setTimeout(resolve, 50));
        }
    }

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            trainedModels = result.results;
            displayTrainingResults(result.results);
            isModelsTrained = true;
            updateStepStatuses();
            updateQuickStats();
            
            // Enable evaluation and prediction sections
            updateEvaluationSection();
            updatePredictionSection();
            
            showSuccess('Model training completed successfully!');
        } else {
            showError('Error training models: ' + result.error);
        }
    } catch (error) {
        showError('Error training models: ' + error.message);
    } finally {
        showLoading(false);
        if (progressContainer) progressContainer.style.display = 'none';
    }
}

function displayTrainingResults(results) {
    const resultsContainer = document.getElementById('training-results');
    const modelResultsContainer = document.getElementById('model-results');
    
    if (!resultsContainer || !modelResultsContainer) return;

    resultsContainer.style.display = 'block';
    
    let resultsHTML = '';
    
    Object.entries(results).forEach(([modelName, metrics]) => {
        resultsHTML += `
            <div class="model-result">
                <h4>${modelName}</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">${metrics.test_mae}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${metrics.test_rmse}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${metrics.test_r2}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${metrics.test_mape}%</div>
                        <div class="metric-label">MAPE</div>
                    </div>
                </div>
            </div>
        `;
    });
    
    modelResultsContainer.innerHTML = resultsHTML;
}

// Evaluation functions
function updateEvaluationSection() {
    const evaluationContainer = document.getElementById('evaluation-container');
    const chartsContainer = document.getElementById('charts-container');
    
    if (!evaluationContainer || !chartsContainer || !trainedModels) return;

    evaluationContainer.innerHTML = `
        <div class="evaluation-summary">
            <h3>Model Performance Summary</h3>
            <p>Comparison of ${Object.keys(trainedModels).length} trained models:</p>
        </div>
    `;
    
    chartsContainer.style.display = 'grid';
    
    // Create charts
    createComparisonChart();
    createAccuracyChart();
    loadFeatureImportance();
}

function createComparisonChart() {
    const canvas = document.getElementById('comparison-chart');
    if (!canvas || !trainedModels) return;

    const ctx = canvas.getContext('2d');
    const models = Object.keys(trainedModels);
    const mae_data = models.map(model => trainedModels[model].test_mae);
    const r2_data = models.map(model => trainedModels[model].test_r2);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'MAE (Lower is Better)',
                    data: mae_data,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    label: 'R² Score (Higher is Better)',
                    data: r2_data,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1,
                    type: 'line',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'MAE'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'R² Score'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                }
            }
        }
    });
}

function createAccuracyChart() {
    const canvas = document.getElementById('accuracy-chart');
    if (!canvas || !trainedModels) return;

    const ctx = canvas.getContext('2d');
    const bestModel = Object.entries(trainedModels).reduce((best, [name, metrics]) => 
        metrics.test_r2 > best[1].test_r2 ? [name, metrics] : best
    );

    const predictions = bestModel[1].predictions.slice(0, 50);
    const actual = bestModel[1].actual.slice(0, 50);

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Predictions vs Actual',
                data: predictions.map((pred, i) => ({x: actual[i], y: pred})),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)'
            }, {
                label: 'Perfect Prediction',
                data: [{x: Math.min(...actual), y: Math.min(...actual)}, 
                       {x: Math.max(...actual), y: Math.max(...actual)}],
                borderColor: 'rgba(239, 68, 68, 1)',
                borderDash: [5, 5],
                type: 'line',
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Actual Values'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Predicted Values'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `Prediction Accuracy - ${bestModel[0]}`
                }
            }
        }
    });
}

async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/feature_importance?model=Random Forest');
        const result = await response.json();
        
        if (result.success && result.importance) {
            createFeatureImportanceChart(result.importance);
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

function createFeatureImportanceChart(importance) {
    const canvas = document.getElementById('importance-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const features = Object.keys(importance);
    const values = Object.values(importance);

    new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: features.map(f => f.replace('_', ' ').toUpperCase()),
            datasets: [{
                label: 'Feature Importance',
                data: values,
                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                borderColor: 'rgba(139, 92, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Importance (Random Forest)'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                }
            }
        }
    });
}

// Prediction functions
function updatePredictionSection() {
    const predictionContainer = document.getElementById('prediction-container');
    const formContainer = document.getElementById('prediction-form-container');
    
    if (!predictionContainer || !formContainer) return;

    predictionContainer.style.display = 'none';
    formContainer.style.display = 'block';
}

async function makePrediction(e) {
    e.preventDefault();
    
    const season = parseInt(document.getElementById('season').value);
    const weather = parseInt(document.getElementById('weather').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const wind = parseFloat(document.getElementById('wind').value);
    const isWeekend = document.getElementById('weekend').checked ? 1 : 0;
    const isHoliday = document.getElementById('holiday').checked ? 1 : 0;
    const modelName = document.getElementById('model-select').value;

    // Create feature array (matching the order expected by the model)
    const month = new Date().getMonth() + 1;
    const weekday = new Date().getDay();
    
    const features = [
        season,
        isWeekend,
        isHoliday,
        weather,
        temperature,
        humidity,
        wind,
        Math.sin(2 * Math.PI * month / 12), // month_sin
        Math.cos(2 * Math.PI * month / 12), // month_cos
        Math.sin(2 * Math.PI * weekday / 7), // weekday_sin
        Math.cos(2 * Math.PI * weekday / 7), // weekday_cos
        temperature < 0 ? 0 : temperature < 10 ? 1 : temperature < 20 ? 2 : temperature < 30 ? 3 : 4, // temp_category
        (temperature * humidity) / 100, // temp_humidity
        season * weather // season_weather
    ];

    showLoading(true, 'Making prediction...');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                features: features,
                model: modelName
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictionResult(result.prediction, modelName);
        } else {
            showError('Error making prediction: ' + result.error);
        }
    } catch (error) {
        showError('Error making prediction: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayPredictionResult(prediction, modelName) {
    const resultContainer = document.getElementById('prediction-result');
    const predictionNumber = document.getElementById('prediction-number');
    const usedModel = document.getElementById('used-model');
    const confidence = document.getElementById('confidence');
    const predictionTime = document.getElementById('prediction-time');
    
    if (!resultContainer) return;

    if (predictionNumber) predictionNumber.textContent = Math.round(prediction);
    if (usedModel) usedModel.textContent = modelName;
    if (confidence) {
        // Simple confidence calculation based on model performance
        const modelMetrics = trainedModels[modelName];
        const confidenceLevel = modelMetrics && modelMetrics.test_r2 > 0.8 ? 'High' : 
                               modelMetrics && modelMetrics.test_r2 > 0.6 ? 'Medium' : 'Low';
        confidence.textContent = confidenceLevel;
    }
    if (predictionTime) predictionTime.textContent = new Date().toLocaleString();
    
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

// Utility functions
function updateStepStatuses() {
    const steps = [
        { id: 'step-preprocessing', completed: isDataLoaded },
        { id: 'step-training', completed: isModelsTrained, blocked: !isDataLoaded },
        { id: 'step-evaluation', completed: isModelsTrained, blocked: !isModelsTrained },
        { id: 'step-prediction', completed: isModelsTrained, blocked: !isModelsTrained }
    ];

    steps.forEach(step => {
        const element = document.getElementById(step.id);
        if (!element) return;

        element.className = 'step-status';
        
        if (step.completed) {
            element.classList.add('completed');
            element.innerHTML = '<i class="fas fa-check"></i><span>Completed</span>';
        } else if (step.blocked) {
            element.classList.add('blocked');
            element.innerHTML = '<i class="fas fa-lock"></i><span>Blocked</span>';
        } else {
            element.classList.add('pending');
            element.innerHTML = '<i class="fas fa-clock"></i><span>Pending</span>';
        }
    });
}

function updateQuickStats() {
    if (preprocessedData) {
        const samplesEl = document.getElementById('stat-samples');
        const featuresEl = document.getElementById('stat-features');
        if (samplesEl) samplesEl.textContent = preprocessedData.total_samples;
        if (featuresEl) featuresEl.textContent = preprocessedData.features;
    }

    if (trainedModels) {
        const modelsEl = document.getElementById('stat-models');
        const accuracyEl = document.getElementById('stat-accuracy');
        if (modelsEl) modelsEl.textContent = Object.keys(trainedModels).length;
        
        if (accuracyEl) {
            const bestR2 = Math.max(...Object.values(trainedModels).map(m => m.test_r2));
            accuracyEl.textContent = bestR2.toFixed(3);
        }
    }
}

function showLoading(show, message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    
    if (!overlay) return;
    
    if (show) {
        if (text) text.textContent = message;
        overlay.style.display = 'flex';
    } else {
        overlay.style.display = 'none';
    }
}

function showSuccess(message) {
    // Simple success notification (you can enhance this)
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #d1fae5;
        color: #059669;
        padding: 1rem 2rem;
        border-radius: 8px;
        border: 1px solid #34d399;
        z-index: 1001;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        document.body.removeChild(notification);
    }, 3000);
}

function showError(message) {
    // Simple error notification (you can enhance this)
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #fee2e2;
        color: #dc2626;
        padding: 1rem 2rem;
        border-radius: 8px;
        border: 1px solid #fca5a5;
        z-index: 1001;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        document.body.removeChild(notification);
    }, 5000);
}

// Helper functions
function getSeasonName(season) {
    const seasons = { 1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter' };
    return seasons[season] || 'Unknown';
}

function getWeatherName(weather) {
    const weathers = { 1: 'Clear', 2: 'Mist', 3: 'Light Rain', 4: 'Heavy Rain' };
    return weathers[weather] || 'Unknown';
}
