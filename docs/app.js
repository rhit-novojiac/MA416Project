/**
 * Microscopic Fungi Classifier - Client-side ONNX Runtime Web Application
 * CSSE/MA416 Deep Learning Term Project
 */

// Species Metadata & Clinical Context
const CLASS_INFO = {
  H1: {
    code: "H1",
    name: "Trichophyton rubrum",
    common: "Athlete's foot & Onychomycosis (nail fungus)",
    notes: "The most common anthropophilic dermatophyte worldwide. Primarily causes athlete's foot (tinea pedis) and fungal nail infections. Microscopically characterized by slender clavate microconidia and sparse cigar-shaped macroconidia."
  },
  H2: {
    code: "H2",
    name: "Trichophyton mentagrophytes",
    common: "Inflammatory ringworm & tinea corporis",
    notes: "Zoophilic and anthropophilic dermatophyte frequently causing inflammatory skin lesions, ringworm, and kerion. Forms distinctive spherical microconidia in dense clusters and spiral hyphae."
  },
  H3: {
    code: "H3",
    name: "Epidermophyton floccosum",
    common: "Jock itch (tinea cruris) & skin infections",
    notes: "Strict anthropophilic dermatophyte affecting the groin, feet, and nails, but not hair. Diagnostic hallmark includes smooth, club-shaped or 'beaver-tail' macroconidia grouped in clusters without microconidia."
  },
  H5: {
    code: "H5",
    name: "Microsporum canis",
    common: "Zoophilic ringworm (cats & dogs to humans)",
    notes: "Zoophilic dermatophyte primarily hosted by domestic cats and dogs, highly contagious to humans. Characteristic thick-walled, echinulate (spiny), spindle-shaped macroconidia with curved apical beaks."
  },
  H6: {
    code: "H6",
    name: "Microsporum gypseum",
    common: "Geophilic dermatophyte (soil exposure)",
    notes: "Geophilic fungus inhabiting warm, humid soils, transmitted through direct ground exposure. Produces abundant symmetrical, rough-walled, ellipsoid macroconidia with rounded terminal ends."
  }
};

const CLASS_NAMES = ["H1", "H2", "H3", "H5", "H6"];

// Normalization Constants (ImageNet standards used during ResNet-18 training)
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// State
let currentSession = null;
let currentModelPath = "";
const sessionCache = new Map();
let currentImage = null;

// DOM Elements
const modelSelect = document.getElementById("modelSelect");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const progressBarContainer = document.getElementById("progressBarContainer");
const progressBar = document.getElementById("progressBar");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzonePrompt = document.getElementById("dropzonePrompt");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const btnClear = document.getElementById("btnClear");
const inferenceTimeEl = document.getElementById("inferenceTime");
const predictionBanner = document.getElementById("predictionBanner");
const speciesInfoText = document.getElementById("speciesInfoText");
const sampleButtons = document.querySelectorAll(".sample-btn");

/**
 * Updates UI status indicator
 */
function setStatus(state, message) {
  statusDot.className = "status-indicator " + state;
  statusText.textContent = message;
}

/**
 * Loads an ONNX model into ONNX Runtime Web session
 */
async function loadModel(modelPath) {
  if (sessionCache.has(modelPath)) {
    currentSession = sessionCache.get(modelPath);
    currentModelPath = modelPath;
    setStatus("ready", "Model Ready (Cached)");
    if (currentImage) runInference();
    return;
  }

  try {
    setStatus("loading", "Fetching ONNX model weights...");
    progressBarContainer.style.display = "block";
    progressBar.style.width = "40%";

    // Configure ONNX Runtime Web options
    ort.env.wasm.numThreads = 2;
    ort.env.wasm.proxy = false;

    progressBar.style.width = "75%";
    setStatus("loading", "Compiling WebAssembly execution graph...");

    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"]
    });

    progressBar.style.width = "100%";
    sessionCache.set(modelPath, session);
    currentSession = session;
    currentModelPath = modelPath;

    setTimeout(() => {
      progressBarContainer.style.display = "none";
      progressBar.style.width = "0%";
    }, 400);

    setStatus("ready", "Model Active (ONNX WASM)");
    if (currentImage) runInference();
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
    progressBarContainer.style.display = "none";
    setStatus("error", "Error loading model. Check console.");
  }
}

/**
 * Preprocesses an HTML Image into a normalized Float32 planar tensor [1, 3, 224, 224]
 */
function preprocessImage(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");

  // Center crop & resize to 224x224 maintaining aspect ratio
  const aspect = img.naturalWidth / img.naturalHeight;
  let sWidth, sHeight, sx, sy;

  if (aspect > 1) {
    sHeight = img.naturalHeight;
    sWidth = img.naturalHeight;
    sx = (img.naturalWidth - sWidth) / 2;
    sy = 0;
  } else {
    sWidth = img.naturalWidth;
    sHeight = img.naturalWidth;
    sx = 0;
    sy = (img.naturalHeight - sHeight) / 2;
  }

  ctx.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, 224, 224);
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const { data } = imageData; // RGBA uint8

  const totalPixels = 224 * 224;
  const floatData = new Float32Array(3 * totalPixels);

  // Planar layout: [R0, R1... G0, G1... B0, B1...] normalized with ImageNet statistics
  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    floatData[totalPixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    floatData[2 * totalPixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
}

/**
 * Executes model inference and renders probabilities
 */
async function runInference() {
  if (!currentSession || !currentImage) return;

  try {
    inferenceTimeEl.textContent = "Inferring...";
    const inputTensor = preprocessImage(currentImage);

    const startTime = performance.now();
    const results = await currentSession.run({ input: inputTensor });
    const endTime = performance.now();

    const latency = Math.round(endTime - startTime);
    inferenceTimeEl.textContent = `Inference: ${latency} ms (WASM)`;

    const probs = Array.from(results.probabilities.data);

    // Find top predicted class
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    const topCode = CLASS_NAMES[maxIdx];
    const topInfo = CLASS_INFO[topCode];
    const topPercentage = (maxProb * 100).toFixed(2);

    // Update Top Prediction Banner
    predictionBanner.innerHTML = `
      <div class="banner-result">
        <div class="banner-left">
          <span class="banner-label">Predicted Species</span>
          <span class="banner-species">${topCode}: <em>${topInfo.name}</em></span>
        </div>
        <div class="banner-confidence">
          <span class="conf-value">${topPercentage}%</span>
          <span class="conf-label">Confidence</span>
        </div>
      </div>
    `;

    // Update Probability Bars
    CLASS_NAMES.forEach((cls, idx) => {
      const p = probs[idx];
      const percentStr = (p * 100).toFixed(1) + "%";
      const valEl = document.getElementById("val-" + cls);
      const fillEl = document.getElementById("fill-" + cls);
      const rowEl = valEl.closest(".prob-row");

      if (valEl) valEl.textContent = percentStr;
      if (fillEl) fillEl.style.width = percentStr;

      if (rowEl) {
        if (idx === maxIdx) {
          rowEl.classList.add("top-rank");
        } else {
          rowEl.classList.remove("top-rank");
        }
      }
    });

    // Update Species Context Card
    speciesInfoText.innerHTML = `
      <strong>${topCode} - ${topInfo.name}</strong> (<em>${topInfo.common}</em>)<br><br>
      ${topInfo.notes}
    `;
  } catch (err) {
    console.error("Inference execution failed:", err);
    inferenceTimeEl.textContent = "Inference Error";
  }
}

/**
 * Handles image selection
 */
function displayImage(src) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    currentImage = img;
    previewImage.src = src;
    dropzonePrompt.style.display = "none";
    previewContainer.style.display = "flex";
    runInference();
  };
  img.src = src;
}

/**
 * Clears current image and prediction displays
 */
function clearImage() {
  currentImage = null;
  fileInput.value = "";
  previewImage.src = "";
  previewContainer.style.display = "none";
  dropzonePrompt.style.display = "flex";
  inferenceTimeEl.textContent = "Ready";

  predictionBanner.innerHTML = `
    <div class="banner-placeholder">Upload or click an image above to run classification.</div>
  `;

  CLASS_NAMES.forEach((cls) => {
    const valEl = document.getElementById("val-" + cls);
    const fillEl = document.getElementById("fill-" + cls);
    const rowEl = valEl.closest(".prob-row");
    if (valEl) valEl.textContent = "0.0%";
    if (fillEl) fillEl.style.width = "0%";
    if (rowEl) rowEl.classList.remove("top-rank");
  });

  speciesInfoText.textContent = "Select an image to see details on pathogenicity, morphology, and clinical relevance.";
}

// Event Listeners
modelSelect.addEventListener("change", (e) => {
  loadModel(e.target.value);
});

dropzone.addEventListener("click", (e) => {
  if (e.target !== btnClear) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (evt) => displayImage(evt.target.result);
    reader.readAsDataURL(file);
  }
});

// Drag and drop
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (evt) => displayImage(evt.target.result);
    reader.readAsDataURL(file);
  }
});

// Paste support
window.addEventListener("paste", (e) => {
  const items = e.clipboardData.items;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf("image") !== -1) {
      const blob = items[i].getAsFile();
      const reader = new FileReader();
      reader.onload = (evt) => displayImage(evt.target.result);
      reader.readAsDataURL(blob);
      break;
    }
  }
});

btnClear.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImage();
});

sampleButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const src = btn.getAttribute("data-src");
    displayImage(src);
  });
});

// Gallery logic for 100 random DeFungi test images
const galleryGrid = document.getElementById("galleryGrid");
const filterChips = document.querySelectorAll(".filter-chip");
let allGalleryItems = [];

async function initGallery() {
  if (!galleryGrid) return;
  try {
    const res = await fetch("gallery/manifest.json");
    if (!res.ok) return;
    allGalleryItems = await res.json();
    renderGallery("all");
  } catch (err) {
    console.warn("Could not load gallery manifest:", err);
  }
}

function renderGallery(filterClass) {
  if (!galleryGrid) return;
  galleryGrid.innerHTML = "";
  const filtered = filterClass === "all" 
    ? allGalleryItems 
    : allGalleryItems.filter(item => item.class === filterClass);

  filtered.forEach(item => {
    const btn = document.createElement("button");
    btn.className = "gallery-item";
    btn.title = `True Class: ${item.class} (${CLASS_INFO[item.class]?.name || item.class}) - ${item.original_name}`;
    btn.innerHTML = `
      <img src="${item.src}" alt="${item.class}" loading="lazy">
      <span class="gallery-badge">${item.class}</span>
    `;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".gallery-item").forEach(el => el.classList.remove("active-item"));
      btn.classList.add("active-item");
      displayImage(item.src);
    });
    galleryGrid.appendChild(btn);
  });
}

filterChips.forEach(chip => {
  chip.addEventListener("click", () => {
    filterChips.forEach(c => c.classList.remove("active-item", "active"));
    chip.classList.add("active");
    renderGallery(chip.getAttribute("data-filter"));
  });
});

// Initial startup
window.addEventListener("DOMContentLoaded", () => {
  loadModel(modelSelect.value);
  initGallery();
});


