// API Base URL
const API_BASE = '/api';

// State
let availableModels = [];

// DOM Elements
const registerForm = document.getElementById('register-form');
const recognizeForm = document.getElementById('recognize-form');
const registerUpload = document.getElementById('register-upload');
const recognizeUpload = document.getElementById('recognize-upload');
const registerFile = document.getElementById('register-file');
const recognizeFile = document.getElementById('recognize-file');
const registerPreview = document.getElementById('register-preview');
const recognizePreview = document.getElementById('recognize-preview');
const registerResult = document.getElementById('register-result');
const recognizeResult = document.getElementById('recognize-result');
const matchesContainer = document.getElementById('matches-container');
const matchesList = document.getElementById('matches-list');
const recognizeInfo = document.getElementById('recognize-info');
const facesList = document.getElementById('faces-list');
const registerBtn = document.getElementById('register-btn');
const recognizeBtn = document.getElementById('recognize-btn');
const registerModelsContainer = document.getElementById('register-models');
const recognizeModelsContainer = document.getElementById('recognize-models');
const ensembleGroup = document.getElementById('ensemble-group');
const ensembleMethod = document.getElementById('ensemble-method');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea(registerUpload, registerFile, registerPreview);
    setupUploadArea(recognizeUpload, recognizeFile, recognizePreview);
    loadModels();
    loadFaces();
});

// Load available models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        availableModels = await response.json();
        renderModelSelectors();
    } catch (error) {
        console.error('Error loading models:', error);
        registerModelsContainer.innerHTML = '<p class="loading-text">Erro ao carregar modelos</p>';
        recognizeModelsContainer.innerHTML = '<p class="loading-text">Erro ao carregar modelos</p>';
    }
}

// Render model checkboxes
function renderModelSelectors() {
    const registerHtml = availableModels.map((model, i) => `
        <input type="checkbox" id="reg-model-${model.name}"
               class="model-checkbox"
               name="register-models"
               value="${model.name}"
               ${model.name === 'ArcFace' ? 'checked' : ''}>
        <label for="reg-model-${model.name}" class="model-label" title="${model.description}">
            ${model.display_name}
            <span class="model-badge">${model.embedding_size}d</span>
        </label>
    `).join('');

    const recognizeHtml = availableModels.map((model, i) => `
        <input type="checkbox" id="rec-model-${model.name}"
               class="model-checkbox"
               name="recognize-models"
               value="${model.name}"
               ${model.name === 'ArcFace' ? 'checked' : ''}
               onchange="updateEnsembleVisibility()">
        <label for="rec-model-${model.name}" class="model-label" title="${model.description}">
            ${model.display_name}
            <span class="model-badge">${model.embedding_size}d</span>
        </label>
    `).join('');

    registerModelsContainer.innerHTML = registerHtml;
    recognizeModelsContainer.innerHTML = recognizeHtml;
}

// Update ensemble method visibility based on selected models
function updateEnsembleVisibility() {
    const selected = document.querySelectorAll('input[name="recognize-models"]:checked');
    ensembleGroup.hidden = selected.length <= 1;
}

// Get selected models from a container
function getSelectedModels(name) {
    const checkboxes = document.querySelectorAll(`input[name="${name}"]:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

// Setup upload area with drag & drop
function setupUploadArea(uploadArea, fileInput, preview) {
    const uploadContent = uploadArea.querySelector('.upload-content');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showPreview(files[0], preview, uploadContent);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            showPreview(e.target.files[0], preview, uploadContent);
        }
    });
}

// Show image preview
function showPreview(file, preview, uploadContent) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        uploadContent.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Reset upload area
function resetUpload(uploadArea, fileInput, preview) {
    const uploadContent = uploadArea.querySelector('.upload-content');
    fileInput.value = '';
    preview.src = '';
    preview.hidden = true;
    uploadContent.style.display = 'block';
}

// Show result message
function showResult(element, message, isError = false) {
    element.textContent = message;
    element.className = 'result-message ' + (isError ? 'error' : 'success');
}

// Clear result message
function clearResult(element) {
    element.textContent = '';
    element.className = 'result-message';
}

// Set button loading state
function setLoading(btn, loading) {
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    btn.disabled = loading;
    btnText.hidden = loading;
    btnLoading.hidden = !loading;
}

// Register face
registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearResult(registerResult);

    const name = document.getElementById('name').value.trim();
    const file = registerFile.files[0];
    const selectedModels = getSelectedModels('register-models');

    if (!name) {
        showResult(registerResult, 'Por favor, insira um nome.', true);
        return;
    }

    if (!file) {
        showResult(registerResult, 'Por favor, selecione uma imagem.', true);
        return;
    }

    if (selectedModels.length === 0) {
        showResult(registerResult, 'Por favor, selecione pelo menos um modelo.', true);
        return;
    }

    setLoading(registerBtn, true);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', file);
    formData.append('models', selectedModels.join(','));

    try {
        const response = await fetch(`${API_BASE}/faces/register`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            const modelsUsed = data.models_used.join(', ');
            showResult(registerResult, `${data.message} (Modelos: ${modelsUsed})`);
            document.getElementById('name').value = '';
            resetUpload(registerUpload, registerFile, registerPreview);
            loadFaces();
        } else {
            showResult(registerResult, data.detail || data.message || 'Erro ao registrar face.', true);
        }
    } catch (error) {
        console.error('Error:', error);
        showResult(registerResult, 'Erro de conexao com o servidor.', true);
    } finally {
        setLoading(registerBtn, false);
    }
});

// Recognize face
recognizeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearResult(recognizeResult);
    matchesContainer.hidden = true;

    const file = recognizeFile.files[0];
    const selectedModels = getSelectedModels('recognize-models');
    const ensembleMethodValue = ensembleMethod.value;

    if (!file) {
        showResult(recognizeResult, 'Por favor, selecione uma imagem.', true);
        return;
    }

    if (selectedModels.length === 0) {
        showResult(recognizeResult, 'Por favor, selecione pelo menos um modelo.', true);
        return;
    }

    setLoading(recognizeBtn, true);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('models', selectedModels.join(','));
    formData.append('ensemble_method', ensembleMethodValue);

    try {
        const response = await fetch(`${API_BASE}/faces/recognize`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            if (data.matches && data.matches.length > 0) {
                showResult(recognizeResult, data.message);
                displayMatches(data.matches, data.models_used, data.ensemble_method);
            } else {
                showResult(recognizeResult, data.message || 'Nenhuma correspondencia encontrada.', true);
            }
        } else {
            showResult(recognizeResult, data.detail || 'Erro ao reconhecer face.', true);
        }
    } catch (error) {
        console.error('Error:', error);
        showResult(recognizeResult, 'Erro de conexao com o servidor.', true);
    } finally {
        setLoading(recognizeBtn, false);
    }
});

// Display matches
function displayMatches(matches, modelsUsed, ensembleMethodUsed) {
    matchesList.innerHTML = '';
    matchesContainer.hidden = false;

    // Show info about models used
    let infoHtml = `<strong>Modelos:</strong> ${modelsUsed.join(', ')}`;
    if (ensembleMethodUsed) {
        const methodLabels = {
            'average': 'Media',
            'weighted': 'Media Ponderada',
            'voting': 'Votacao',
            'min': 'Minimo',
            'max': 'Maximo'
        };
        infoHtml += ` | <strong>Ensemble:</strong> ${methodLabels[ensembleMethodUsed] || ensembleMethodUsed}`;
    }
    recognizeInfo.innerHTML = infoHtml;

    matches.forEach(match => {
        const confidence = Math.round(match.confidence * 100);
        const matchEl = document.createElement('div');
        matchEl.className = 'match-item';

        let modelDistancesHtml = '';
        if (match.model_distances) {
            const distances = Object.entries(match.model_distances)
                .map(([model, dist]) => `${model}: ${(dist * 100).toFixed(1)}%`)
                .join(', ');
            modelDistancesHtml = `<div class="match-models">Distancias: ${distances}</div>`;
        }

        matchEl.innerHTML = `
            <img src="${API_BASE}/faces/${match.id}/image" alt="${match.name}" class="match-image">
            <div class="match-info">
                <div class="match-name">${escapeHtml(match.name)}</div>
                <div class="match-confidence">Confianca: ${confidence}%</div>
                ${modelDistancesHtml}
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
            </div>
        `;
        matchesList.appendChild(matchEl);
    });
}

// Load registered faces
async function loadFaces() {
    try {
        const response = await fetch(`${API_BASE}/faces`);
        const faces = await response.json();

        if (faces.length === 0) {
            facesList.innerHTML = '<p class="empty-message">Nenhuma face registrada</p>';
            return;
        }

        facesList.innerHTML = '';
        faces.forEach(face => {
            const faceEl = document.createElement('div');
            faceEl.className = 'face-card';
            const modelsText = face.models ? face.models.join(', ') : 'N/A';
            faceEl.innerHTML = `
                <img src="${API_BASE}/faces/${face.id}/image" alt="${face.name}" class="face-image">
                <div class="face-name" title="${escapeHtml(face.name)}">${escapeHtml(face.name)}</div>
                <div class="face-models" title="${modelsText}">${modelsText}</div>
                <button class="btn btn-danger" onclick="deleteFace('${face.id}')">Remover</button>
            `;
            facesList.appendChild(faceEl);
        });
    } catch (error) {
        console.error('Error loading faces:', error);
        facesList.innerHTML = '<p class="empty-message">Erro ao carregar faces</p>';
    }
}

// Delete face
async function deleteFace(id) {
    if (!confirm('Tem certeza que deseja remover esta face?')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/faces/${id}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadFaces();
        } else {
            const data = await response.json();
            alert(data.detail || 'Erro ao remover face.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Erro de conexao com o servidor.');
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
