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
const recognizePreviewWrapper = document.getElementById('recognize-preview-wrapper');
const recognizeCanvas = document.getElementById('recognize-canvas');
const registerZipInfo = document.getElementById('register-zip-info');
const recognizeZipInfo = document.getElementById('recognize-zip-info');
const registerZipName = document.getElementById('register-zip-name');
const recognizeZipName = document.getElementById('recognize-zip-name');
const registerResult = document.getElementById('register-result');
const recognizeResult = document.getElementById('recognize-result');
const registerBulkResults = document.getElementById('register-bulk-results');
const bulkRecognizeResults = document.getElementById('bulk-recognize-results');
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
const nameGroup = document.getElementById('name-group');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea(registerUpload, registerFile, registerPreview, registerZipInfo, registerZipName);
    setupUploadArea(recognizeUpload, recognizeFile, recognizePreview, recognizeZipInfo, recognizeZipName, recognizePreviewWrapper, recognizeCanvas);
    loadModels();
    loadFaces();
});

// Check if file is a ZIP
function isZipFile(file) {
    return file.name.toLowerCase().endsWith('.zip') ||
           file.type === 'application/zip' ||
           file.type === 'application/x-zip-compressed';
}

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
    // For registration: select ALL available models by default
    const registerHtml = availableModels.map((model, i) => `
        <input type="checkbox" id="reg-model-${model.name}"
               class="model-checkbox"
               name="register-models"
               value="${model.name}"
               ${model.available ? 'checked' : 'disabled'}>
        <label for="reg-model-${model.name}" class="model-label ${!model.available ? 'model-unavailable' : ''}" title="${model.description}${!model.available ? ' (Indisponivel)' : ''}">
            ${model.display_name}
            <span class="model-badge">${model.embedding_size}d</span>
            ${!model.available ? '<span class="model-status-unavailable">X</span>' : ''}
        </label>
    `).join('');

    // For recognition: only ArcFace by default (user can select more)
    const recognizeHtml = availableModels.map((model, i) => `
        <input type="checkbox" id="rec-model-${model.name}"
               class="model-checkbox"
               name="recognize-models"
               value="${model.name}"
               ${model.available && model.name === 'ArcFace' ? 'checked' : ''}
               ${!model.available ? 'disabled' : ''}
               onchange="updateEnsembleVisibility()">
        <label for="rec-model-${model.name}" class="model-label ${!model.available ? 'model-unavailable' : ''}" title="${model.description}${!model.available ? ' (Indisponivel)' : ''}">
            ${model.display_name}
            <span class="model-badge">${model.embedding_size}d</span>
            ${!model.available ? '<span class="model-status-unavailable">X</span>' : ''}
        </label>
    `).join('');

    registerModelsContainer.innerHTML = registerHtml;
    recognizeModelsContainer.innerHTML = recognizeHtml;

    // Show count of available models
    const availableCount = availableModels.filter(m => m.available).length;
    const totalCount = availableModels.length;
    console.log(`Models loaded: ${availableCount}/${totalCount} available`);
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
// previewWrapper and canvas are optional — only used for the recognize panel
function setupUploadArea(uploadArea, fileInput, preview, zipInfo, zipName, previewWrapper = null, canvas = null) {
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
            showFilePreview(files[0], preview, uploadContent, zipInfo, zipName, previewWrapper, canvas);
            updateNameFieldVisibility(fileInput);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            showFilePreview(e.target.files[0], preview, uploadContent, zipInfo, zipName, previewWrapper, canvas);
            updateNameFieldVisibility(fileInput);
        }
    });
}

// Show file preview (image or ZIP info)
// previewWrapper: optional wrapper <div> for the canvas overlay (recognize panel only)
// canvas: optional <canvas> to clear when a new image is selected
function showFilePreview(file, preview, uploadContent, zipInfo, zipName, previewWrapper = null, canvas = null) {
    const visibleEl = previewWrapper || preview;

    if (isZipFile(file)) {
        // Show ZIP info, hide image preview
        visibleEl.hidden = true;
        zipInfo.hidden = false;
        zipName.textContent = file.name;
        uploadContent.style.display = 'none';
        // Clear any previous face boxes
        if (canvas) clearCanvas(canvas);
    } else {
        // Show image preview
        zipInfo.hidden = true;
        if (canvas) clearCanvas(canvas);
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            visibleEl.hidden = false;
            uploadContent.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

// Update name field visibility based on file type
function updateNameFieldVisibility(fileInput) {
    if (fileInput === registerFile && nameGroup) {
        const file = fileInput.files[0];
        if (file && isZipFile(file)) {
            nameGroup.style.display = 'none';
        } else {
            nameGroup.style.display = 'block';
        }
    }
}

// Reset upload area
function resetUpload(uploadArea, fileInput, preview, zipInfo, previewWrapper = null, canvas = null) {
    const uploadContent = uploadArea.querySelector('.upload-content');
    fileInput.value = '';
    preview.src = '';
    if (previewWrapper) previewWrapper.hidden = true;
    else preview.hidden = true;
    if (zipInfo) zipInfo.hidden = true;
    uploadContent.style.display = 'block';
    if (nameGroup) nameGroup.style.display = 'block';
    if (canvas) clearCanvas(canvas);
}

// Clear canvas drawing
function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── Face bounding-box drawing ────────────────────────────────────────────────

const BBOX_COLORS = [
    '#00e676', '#ff1744', '#2979ff', '#ffea00',
    '#ff6d00', '#d500f9', '#00e5ff', '#76ff03'
];

// Return black or white for best contrast against a hex color
function contrastColor(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 ? '#000000' : '#ffffff';
}

// Draw bounding boxes and name labels on the recognize canvas
function drawFaceBboxes(faces) {
    const img = recognizePreview;
    const canvas = recognizeCanvas;

    const render = () => {
        // Use natural image resolution as canvas coordinate space —
        // CSS scales the canvas visually to match the displayed image size.
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Scale line widths / fonts relative to image size
        const lineW    = Math.max(3, canvas.width / 180);
        const fontSize = Math.max(18, Math.round(canvas.width / 32));
        const pad      = Math.round(fontSize * 0.35);

        faces.forEach((face, idx) => {
            if (!face.bbox) return;

            const [x, y, w, h] = face.bbox;
            const color = BBOX_COLORS[idx % BBOX_COLORS.length];

            // ── bounding box ──
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth   = lineW;
            ctx.shadowColor = 'rgba(0,0,0,0.6)';
            ctx.shadowBlur  = lineW * 2;
            ctx.strokeRect(x, y, w, h);
            ctx.restore();

            // ── label text ──
            let label = 'Desconhecido';
            if (face.matches && face.matches.length > 0) {
                const best = face.matches[0];
                const pct  = Math.round(best.confidence * 100);
                label = `${best.name}  ${pct}%`;
            }

            ctx.font = `bold ${fontSize}px sans-serif`;
            const textW  = ctx.measureText(label).width;
            const boxW   = textW + pad * 2;
            const boxH   = fontSize + pad * 1.6;

            // Place label above the box; if no room above, place below
            const labelTop = y - boxH - lineW >= 0
                ? y - boxH - lineW
                : y + h + lineW;

            // Background pill
            ctx.save();
            ctx.fillStyle = color;
            ctx.shadowColor = 'rgba(0,0,0,0.5)';
            ctx.shadowBlur  = 6;
            ctx.beginPath();
            const r = boxH / 3;
            ctx.roundRect(x, labelTop, boxW, boxH, r);
            ctx.fill();
            ctx.restore();

            // Text
            ctx.save();
            ctx.fillStyle = contrastColor(color);
            ctx.font      = `bold ${fontSize}px sans-serif`;
            ctx.fillText(label, x + pad, labelTop + fontSize + pad * 0.4);
            ctx.restore();
        });
    };

    // Image may already be decoded (it was shown in the preview before recognition)
    if (img.complete && img.naturalWidth > 0) {
        render();
    } else {
        img.addEventListener('load', render, { once: true });
    }
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

// Register face(s)
registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearResult(registerResult);
    registerBulkResults.hidden = true;

    const name = document.getElementById('name').value.trim();
    const file = registerFile.files[0];
    const selectedModels = getSelectedModels('register-models');

    if (!file) {
        showResult(registerResult, 'Por favor, selecione uma imagem ou arquivo ZIP.', true);
        return;
    }

    if (selectedModels.length === 0) {
        showResult(registerResult, 'Por favor, selecione pelo menos um modelo.', true);
        return;
    }

    // Check if it's a ZIP file
    if (isZipFile(file)) {
        await registerFromZip(file, selectedModels);
    } else {
        // Single image registration
        if (!name) {
            showResult(registerResult, 'Por favor, insira um nome para a imagem.', true);
            return;
        }
        await registerSingleImage(name, file, selectedModels);
    }
});

// Register single image
async function registerSingleImage(name, file, selectedModels) {
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
            resetUpload(registerUpload, registerFile, registerPreview, registerZipInfo, null, null);
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
}

// Register from ZIP file
async function registerFromZip(file, selectedModels) {
    setLoading(registerBtn, true);

    const formData = new FormData();
    formData.append('zipfile_upload', file);
    formData.append('models', selectedModels.join(','));

    try {
        const response = await fetch(`${API_BASE}/faces/register-zip`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showResult(registerResult, data.message);
            displayBulkRegisterResults(data);
            resetUpload(registerUpload, registerFile, registerPreview, registerZipInfo, null, null);
            loadFaces();
        } else {
            showResult(registerResult, data.detail || 'Erro ao processar arquivo ZIP.', true);
        }
    } catch (error) {
        console.error('Error:', error);
        showResult(registerResult, 'Erro de conexao com o servidor.', true);
    } finally {
        setLoading(registerBtn, false);
    }
}

// Display bulk register results
function displayBulkRegisterResults(data) {
    registerBulkResults.hidden = false;

    let html = `
        <h4>Resultado do Registro em Lote</h4>
        <div class="bulk-summary">
            <div class="bulk-summary-item">
                <span>Total de Pessoas:</span>
                <span>${data.total_persons}</span>
            </div>
            <div class="bulk-summary-item">
                <span>Total de Imagens:</span>
                <span>${data.total_images}</span>
            </div>
            <div class="bulk-summary-item success">
                <span>Registros com Sucesso:</span>
                <span>${data.successful_registrations}</span>
            </div>
            <div class="bulk-summary-item error">
                <span>Falhas:</span>
                <span>${data.failed_registrations}</span>
            </div>
            <div class="bulk-summary-item">
                <span>Modelos:</span>
                <span>${data.models_used.join(', ')}</span>
            </div>
        </div>
    `;

    if (data.results && data.results.length > 0) {
        html += '<div class="bulk-items">';
        data.results.forEach(result => {
            const statusClass = result.images_failed === 0 ? 'success' :
                               (result.images_success === 0 ? 'error' : 'partial');
            const statusText = result.images_failed === 0 ? 'OK' :
                              (result.images_success === 0 ? 'Falha' : 'Parcial');

            html += `
                <div class="bulk-item">
                    <div class="bulk-item-header">
                        <span class="bulk-item-name">${escapeHtml(result.name)}</span>
                        <span class="bulk-item-status ${statusClass}">${statusText}</span>
                    </div>
                    <div class="bulk-item-details">
                        ${result.images_success}/${result.images_processed} imagens registradas
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    registerBulkResults.innerHTML = html;
}

// Recognize face(s)
recognizeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearResult(recognizeResult);
    matchesContainer.hidden = true;
    bulkRecognizeResults.hidden = true;

    const file = recognizeFile.files[0];
    const selectedModels = getSelectedModels('recognize-models');
    const ensembleMethodValue = ensembleMethod.value;

    if (!file) {
        showResult(recognizeResult, 'Por favor, selecione uma imagem ou arquivo ZIP.', true);
        return;
    }

    if (selectedModels.length === 0) {
        showResult(recognizeResult, 'Por favor, selecione pelo menos um modelo.', true);
        return;
    }

    // Check if it's a ZIP file
    if (isZipFile(file)) {
        await recognizeFromZip(file, selectedModels, ensembleMethodValue);
    } else {
        await recognizeSingleImage(file, selectedModels, ensembleMethodValue);
    }
});

// Recognize single image
async function recognizeSingleImage(file, selectedModels, ensembleMethodValue) {
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
            // Check for multi-face response
            if (data.faces && data.faces.length > 0) {
                showResult(recognizeResult, data.message);
                displayMultiFaceMatches(data.faces, data.models_used, data.ensemble_method, data.faces_detected);
                drawFaceBboxes(data.faces);
            } else if (data.matches && data.matches.length > 0) {
                // Fallback for single face (backward compatibility)
                showResult(recognizeResult, data.message);
                displayMatches(data.matches, data.models_used, data.ensemble_method);
                // Wrap single-face matches in a face-like structure for drawing
                drawFaceBboxes([{ bbox: null, matches: data.matches }]);
            } else {
                showResult(recognizeResult, data.message || 'Nenhuma correspondencia encontrada.', true);
                clearCanvas(recognizeCanvas);
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
}

// Recognize from ZIP file
async function recognizeFromZip(file, selectedModels, ensembleMethodValue) {
    setLoading(recognizeBtn, true);

    const formData = new FormData();
    formData.append('zipfile_upload', file);
    formData.append('models', selectedModels.join(','));
    formData.append('ensemble_method', ensembleMethodValue);

    try {
        const response = await fetch(`${API_BASE}/faces/recognize-zip`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showResult(recognizeResult, data.message);
            displayBulkRecognizeResults(data);
        } else {
            showResult(recognizeResult, data.detail || 'Erro ao processar arquivo ZIP.', true);
        }
    } catch (error) {
        console.error('Error:', error);
        showResult(recognizeResult, 'Erro de conexao com o servidor.', true);
    } finally {
        setLoading(recognizeBtn, false);
    }
}

// Display bulk recognize results (with multi-face support)
function displayBulkRecognizeResults(data) {
    bulkRecognizeResults.hidden = false;

    // Count statistics
    const totalFaces = data.results.reduce((sum, r) => sum + (r.faces_detected || 0), 0);
    const totalMatches = data.results.reduce((sum, r) => {
        if (r.faces) {
            return sum + r.faces.reduce((fSum, f) => fSum + (f.matches ? f.matches.length : 0), 0);
        }
        return sum;
    }, 0);
    const withoutFace = data.results.filter(r => !r.success).length;

    let html = `
        <h4>Resultado do Reconhecimento em Lote</h4>
        <div class="bulk-summary">
            <div class="bulk-summary-item">
                <span>Imagens Processadas:</span>
                <span>${data.processed_images}</span>
            </div>
            <div class="bulk-summary-item">
                <span>Total de Faces Detectadas:</span>
                <span>${totalFaces}</span>
            </div>
            <div class="bulk-summary-item success">
                <span>Total de Correspondencias:</span>
                <span>${totalMatches}</span>
            </div>
            <div class="bulk-summary-item error">
                <span>Imagens Sem Face:</span>
                <span>${withoutFace}</span>
            </div>
            <div class="bulk-summary-item">
                <span>Modelos:</span>
                <span>${data.models_used.join(', ')}</span>
            </div>
            ${data.ensemble_method ? `
            <div class="bulk-summary-item">
                <span>Ensemble:</span>
                <span>${data.ensemble_method}</span>
            </div>
            ` : ''}
        </div>
    `;

    if (data.results && data.results.length > 0) {
        html += '<div class="bulk-items">';
        data.results.forEach(result => {
            const facesDetected = result.faces_detected || 0;
            const faceMatches = result.faces ? result.faces.reduce((sum, f) => sum + (f.matches ? f.matches.length : 0), 0) : 0;
            const statusClass = !result.success ? 'error' : (faceMatches > 0 ? 'success' : 'partial');
            const statusText = !result.success ? 'Sem Face' : `${facesDetected} Face(s), ${faceMatches} Match(es)`;

            html += `
                <div class="bulk-item">
                    <div class="bulk-item-header">
                        <span class="bulk-item-name">${escapeHtml(result.filename)}</span>
                        <span class="bulk-item-status ${statusClass}">${statusText}</span>
                    </div>
            `;

            // Display faces with matches
            if (result.faces && result.faces.length > 0) {
                result.faces.forEach((face, faceIdx) => {
                    if (face.matches && face.matches.length > 0) {
                        html += `<div class="bulk-item-face">`;
                        html += `<div class="face-label">Face ${face.face_index + 1}${face.bbox ? ` (${face.bbox[0]},${face.bbox[1]})` : ''}:</div>`;
                        html += '<div class="bulk-item-matches">';
                        face.matches.forEach(match => {
                            const confidence = Math.round(match.confidence * 100);
                            html += `
                                <div class="bulk-match">
                                    <span class="bulk-match-name">${escapeHtml(match.name)}</span>
                                    <span class="bulk-match-confidence">${confidence}%</span>
                                </div>
                            `;
                        });
                        html += '</div></div>';
                    } else {
                        html += `<div class="bulk-item-face">`;
                        html += `<div class="face-label">Face ${face.face_index + 1}: <span class="no-match">Sem correspondencia</span></div>`;
                        html += '</div>';
                    }
                });
            }

            html += '</div>';
        });
        html += '</div>';
    }

    bulkRecognizeResults.innerHTML = html;
}

// Display matches for multiple faces in single image
function displayMultiFaceMatches(faces, modelsUsed, ensembleMethodUsed, facesDetected) {
    matchesList.innerHTML = '';
    matchesContainer.hidden = false;

    // Show info about models used and faces detected
    let infoHtml = `<strong>Faces Detectadas:</strong> ${facesDetected} | <strong>Modelos:</strong> ${modelsUsed.join(', ')}`;
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

    faces.forEach((face, faceIndex) => {
        // Create face section
        const faceSection = document.createElement('div');
        faceSection.className = 'face-section';

        const faceHeader = document.createElement('div');
        faceHeader.className = 'face-header';
        const bboxInfo = face.bbox ? ` (posicao: ${face.bbox[0]}, ${face.bbox[1]})` : '';
        faceHeader.innerHTML = `<strong>Face ${face.face_index + 1}</strong>${bboxInfo}`;
        faceSection.appendChild(faceHeader);

        if (face.matches && face.matches.length > 0) {
            face.matches.forEach(match => {
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
                faceSection.appendChild(matchEl);
            });
        } else {
            const noMatch = document.createElement('div');
            noMatch.className = 'no-match-message';
            noMatch.textContent = 'Sem correspondencia encontrada';
            faceSection.appendChild(noMatch);
        }

        matchesList.appendChild(faceSection);
    });
}

// Display matches (single image, single face - backward compatibility)
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
