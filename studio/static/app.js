// StickerDeskew Studio - Advanced Multi-Queue & Clipboard Management

const state = {
  queue: [],              // Array of { id, name, base64, path, thumbnail, status, stages, metadata }
  activeItemId: null,     // Currently active queue item ID
  isProcessing: false,    // Whether a pipeline request is currently executing
  currentImagePath: null,
  currentImageBase64: null,
  currentFileName: null,
  viewMode: "result",     // 'result', 'raw', 'stages'
  activeStage: "final",
  stages: {},
  metadata: null,
  manualRotation: 0,
  horizontalSkew: 0,
  verticalSkew: 0,
  flipH: false,
  flipV: false,
  characterBbox: null,    // [x, y, w, h] in natural image pixel coords
  characterHighlightMask: null,
  isRoiMode: false,
  highlightBrushSize: 32,
  brushMode: "paint",     // 'paint' or 'erase'
  isDrawingHighlight: false,
  hasHighlightStrokes: false,
  roiStart: null,
  isInpaintMode: false,
  inpaintBrushSize: 28,
  isDrawingInpaint: false,
  hasInpaintStrokes: false
};

// DOM References
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const folderInput = document.getElementById("folder-input");
const btnBrowseFolder = document.getElementById("btn-browse-folder");
const btnBrowseFiles = document.getElementById("btn-browse-files");
const btnEmptyFolder = document.getElementById("btn-empty-folder");
const currentFileLabel = document.getElementById("current-file-label");
const btnRunScript = document.getElementById("btn-run-script");
const btnProcessMain = document.getElementById("btn-process-main");
const btnHeaderDownload = document.getElementById("btn-header-download");
const btnCopyClipboard = document.getElementById("btn-copy-clipboard");
const btnClearWorkspace = document.getElementById("btn-clear-workspace");
const btnBrowseEmpty = document.getElementById("btn-browse-empty");

// Queue, Local Folder & Auto-Process DOM
const toggleAutoProcess = document.getElementById("toggle-auto-process");
const toggleSaveLocal = document.getElementById("toggle-save-local");
const queueBadgeCount = document.getElementById("queue-badge-count");
const queueItemsContainer = document.getElementById("queue-items-container");
const btnClearQueue = document.getElementById("btn-clear-queue");
const btnProcessQueue = document.getElementById("btn-process-queue");
const btnSaveLocalQueue = document.getElementById("btn-save-local-queue");
const btnOpenOutputFolder = document.getElementById("btn-open-output-folder");
const btnSidebarSaveLocal = document.getElementById("btn-sidebar-save-local");
const btnOpenFolderSidebar = document.getElementById("btn-open-folder-sidebar");
const toastMsg = document.getElementById("toast-msg");

// Status & Progress DOM
const statusIndicator = document.getElementById("pipeline-status");
const statusText = statusIndicator.querySelector(".status-text");
const loadingOverlay = document.getElementById("loading-overlay");
const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
const loadingPercent = document.getElementById("loading-percent");
const loadingStepDesc = document.getElementById("loading-step-desc");
const loadingTitle = document.getElementById("loading-title");
const btnCancelProcess = document.getElementById("btn-cancel-process");

let currentAbortController = null;
let currentProgressInterval = null;

function showLoadingOverlay(title = "Processing Sticker...", stepDesc = "Initializing pipeline...", initialPct = 6) {
  if (!loadingOverlay) return;
  if (loadingTitle) loadingTitle.textContent = title;
  if (loadingStepDesc) loadingStepDesc.textContent = stepDesc;
  if (loadingPercent) loadingPercent.textContent = `${initialPct}%`;
  if (pipelineProgressBar) pipelineProgressBar.style.width = `${initialPct}%`;
  loadingOverlay.classList.remove("hidden");
  loadingOverlay.style.display = "flex";
}

function hideLoadingOverlay() {
  if (!loadingOverlay) return;
  loadingOverlay.classList.add("hidden");
  loadingOverlay.style.display = "none";
}

// Canvas DOM
const emptyWorkspaceView = document.getElementById("empty-workspace-view");
const displayWrapper = document.getElementById("display-wrapper");
const mainStickerImg = document.getElementById("main-sticker-img");
const stageSubBar = document.getElementById("stage-sub-bar");

// Rotation & Transform DOM
const btnRotCCW = document.getElementById("btn-rot-ccw");
const btnRotCW = document.getElementById("btn-rot-cw");
const btnFlipH = document.getElementById("btn-flip-h");
const btnRotReset = document.getElementById("btn-rot-reset");
const sliderRotation = document.getElementById("slider-rotation");
const badgeRot = document.getElementById("badge-rot");
const sliderHorizontalSkew = document.getElementById("slider-horizontal-skew");
const badgeHorizontalSkew = document.getElementById("badge-horizontal-skew");
const btnResetSkew = document.getElementById("btn-reset-skew");
const sliderVerticalSkew = document.getElementById("slider-vertical-skew");
const badgeVerticalSkew = document.getElementById("badge-vertical-skew");
const btnResetVSkew = document.getElementById("btn-reset-vskew");
const btnBakeTransforms = document.getElementById("btn-bake-transforms");
const btnResetAllTransforms = document.getElementById("btn-reset-all-transforms");
const btnGenerateSixVariants = document.getElementById("btn-generate-six-variants");
const btnHeaderSixVariants = document.getElementById("btn-header-six-variants");
const selectDeskewMode = document.getElementById("select-deskew-mode");

// 6 Cutout Variations Modal & Inline Switcher DOM
const modalCutoutVariants = document.getElementById("modal-cutout-variants");
const cutoutVariantsGrid = document.getElementById("cutout-variants-grid");
const btnCloseCutoutModal = document.getElementById("btn-close-cutout-modal");
const btnRerunVariants = document.getElementById("btn-rerun-variants");
const inlineVariantsBar = document.getElementById("inline-variants-bar");
const inlineVariantsChips = document.getElementById("inline-variants-chips");
const btnOpenVariantsModal = document.getElementById("btn-open-variants-modal");

// In-Tab Web Browser & Character Snipper DOM
const btnOpenBrowser = document.getElementById("btn-open-browser");
const btnBrowserShortcut = document.getElementById("btn-browser-shortcut");
const btnHeaderSnip = document.getElementById("btn-header-snip");
const btnEmptySnip = document.getElementById("btn-empty-snip");
const webBrowserModal = document.getElementById("web-browser-modal");
const webBrowserWindow = document.getElementById("web-browser-window");
const btnBrowserClose = document.getElementById("btn-browser-close");
const btnBrowserFullscreen = document.getElementById("btn-browser-fullscreen");
const btnBrowserBack = document.getElementById("btn-browser-back");
const btnBrowserForward = document.getElementById("btn-browser-forward");
const btnBrowserRefresh = document.getElementById("btn-browser-refresh");
const browserUrlInput = document.getElementById("browser-url-input");
const btnBrowserGo = document.getElementById("btn-browser-go");
const btnBrowserSnipScreen = document.getElementById("btn-browser-snip-screen");
const btnBrowserPasteSnip = document.getElementById("btn-browser-paste-snip");
const btnBrowserStartSnip = document.getElementById("btn-browser-start-snip");
const btnBrowserImportFull = document.getElementById("btn-browser-import-full");
const bookmarkChips = document.querySelectorAll(".bookmark-chip");
const browserViewportWrapper = document.getElementById("browser-viewport-wrapper");
const browserIframe = document.getElementById("browser-iframe");
const browserImageViewer = document.getElementById("browser-image-viewer");
const browserTargetImage = document.getElementById("browser-target-image");
const browserSnipCanvas = document.getElementById("browser-snip-canvas");
const snipConfirmFloating = document.getElementById("snip-confirm-floating");
const snipDimsBadge = document.getElementById("snip-dims-badge");
const btnConfirmSnip = document.getElementById("btn-confirm-snip");
const btnCancelSnip = document.getElementById("btn-cancel-snip");
const snipHintBanner = document.getElementById("snip-hint-banner");
const btnExitSnipMode = document.getElementById("btn-exit-snip-mode");

// Background & Smudge Removal DOM
const selectAiModel = document.getElementById("select-ai-model");
const toggleSmudgeCleaner = document.getElementById("toggle-smudge-cleaner");
const smudgeSensitivityContainer = document.getElementById("smudge-sensitivity-container");
const sliderSmudgeSensitivity = document.getElementById("slider-smudge-sensitivity");
const badgeSmudge = document.getElementById("badge-smudge");
const sliderAlphaThreshold = document.getElementById("slider-alpha-threshold");
const badgeAlpha = document.getElementById("badge-alpha");
const btnExtractChar = document.getElementById("btn-extract-char");
const btnExtractFull = document.getElementById("btn-extract-full");

// Pre-Processing (Anime De-Shine & Cel Restorer) DOM
const toggleShineRemover = document.getElementById("toggle-shine-remover");
const shineOptionsContainer = document.getElementById("shine-options-container");
const sliderShineStrength = document.getElementById("slider-shine-strength");
const badgeShine = document.getElementById("badge-shine");
const sliderColorTiers = document.getElementById("slider-color-tiers");
const badgeColorTiers = document.getElementById("badge-color-tiers");
const toggleFlatCel = document.getElementById("toggle-flat-cel");
const toggleCleanHairGaps = document.getElementById("toggle-clean-hair-gaps");
const btnAutoCel = document.getElementById("btn-auto-cel");
const badgeAutoCelSummary = document.getElementById("badge-auto-cel-summary");

// Color Pop (Float32 OKLab) DOM
const toggleColorPop = document.getElementById("toggle-color-pop");
const colorPopOptionsContainer = document.getElementById("color-pop-options-container");
const selectColorPopPreset = document.getElementById("select-color-pop-preset");
const sliderColorPopVibrance = document.getElementById("slider-color-pop-vibrance");
const badgeColorPopVibrance = document.getElementById("badge-color-pop-vibrance");
const sliderColorPopClarity = document.getElementById("slider-color-pop-clarity");
const badgeColorPopClarity = document.getElementById("badge-color-pop-clarity");
const toggleAiLora = document.getElementById("toggle-ai-lora");
const aiLoraOptionsContainer = document.getElementById("ai-lora-options-container");
const selectAiLoraPreset = document.getElementById("select-ai-lora-preset");
const btnDownloadLoras = document.getElementById("btn-download-loras");
const loraStatusMsg = document.getElementById("lora-status-msg");

const pillStageWatermark = document.getElementById("pill-stage-watermark");
const pillStageCel = document.getElementById("pill-stage-cel");
const pillStageColorPop = document.getElementById("pill-stage-color-pop");

// Finishing Effects & Highlighting DOM
const toggleBorder = document.getElementById("toggle-border");
const borderSliderContainer = document.getElementById("border-slider-container");
const selectHighlightMode = document.getElementById("select-highlight-mode");
const pickerHighlightColor = document.getElementById("picker-highlight-color");
const swatchBtns = document.querySelectorAll(".swatch-btn");
const sliderBorderWidth = document.getElementById("slider-border-width");
const badgeBorder = document.getElementById("badge-border");
const sliderBorderSmoothing = document.getElementById("slider-border-smoothing");
const badgeBorderSmoothing = document.getElementById("badge-border-smoothing");
const sliderGlowRadius = document.getElementById("slider-glow-radius");
const badgeGlowRadius = document.getElementById("badge-glow-radius");
let selectedHighlightColor = "#ffffff";
const toggleSuperRes = document.getElementById("toggle-super-res");
const superResOptionsContainer = document.getElementById("super-res-options-container");
const selectEnhancerModel = document.getElementById("select-enhancer-model");
const selectEnhancerScale = document.getElementById("select-enhancer-scale");
const togglePreserveColors = document.getElementById("toggle-preserve-colors");

// Character ROI Highlighting DOM
const canvasImageWrapper = document.getElementById("canvas-image-wrapper");
const selectionBox = document.getElementById("selection-box");
const highlightCanvas = document.getElementById("highlight-canvas");
const btnToggleRoiMode = document.getElementById("btn-toggle-roi-mode");
const btnViewerRoi = document.getElementById("btn-viewer-roi");
const roiBtnText = document.getElementById("roi-btn-text");
const roiStatusText = document.getElementById("roi-status-text");
const btnClearRoi = document.getElementById("btn-clear-roi");
const sliderHighlightBrush = document.getElementById("slider-highlight-brush");
const badgeHighlightBrush = document.getElementById("badge-highlight-brush");
const roiHighlighterControls = document.getElementById("roi-highlighter-controls");
const btnBrushPaint = document.getElementById("btn-brush-paint");
const btnBrushNegative = document.getElementById("btn-brush-negative");
const btnBrushErase = document.getElementById("btn-brush-erase");

// AI Inpaint / Fill DOM
const btnInpaintBrush = document.getElementById("btn-inpaint-brush");
const inpaintFloatingBar = document.getElementById("inpaint-floating-bar");
const inpaintCanvas = document.getElementById("inpaint-canvas");
const sliderInpaintBrush = document.getElementById("slider-inpaint-brush");
const badgeInpaintSize = document.getElementById("badge-inpaint-size");
const btnInpaintClear = document.getElementById("btn-inpaint-clear");
const btnInpaintRun = document.getElementById("btn-inpaint-run");
const btnInpaintText = document.getElementById("btn-inpaint-text");
const btnInpaintClose = document.getElementById("btn-inpaint-close");
const selectInpaintModel = document.getElementById("select-inpaint-model");
const btnAutoFillHoles = document.getElementById("btn-auto-fill-holes");
const btnAutoFillText = document.getElementById("btn-auto-fill-text");


// Metric DOM
const statTime = document.getElementById("stat-time");
const statDims = document.getElementById("stat-dims");
const statModel = document.getElementById("stat-model");
const statDeskew = document.getElementById("stat-deskew");
const statRot = document.getElementById("stat-rot");

// Sample Buttons
const btnLoadMarine = document.getElementById("btn-load-marine");
const btnLoadPeeker = document.getElementById("btn-load-peeker");
const btnLoadSquare = document.getElementById("btn-load-square");

// -------------------------------------------------------------
// Toast Notifications
// -------------------------------------------------------------
let toastTimer = null;
function showToast(text) {
  if (!toastMsg) return;
  toastMsg.textContent = text;
  toastMsg.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastMsg.classList.remove("show");
  }, 2200);
}

// -------------------------------------------------------------
// Memory-Safe Helpers: Async File Reading & Fast Downscaled Thumbnails
// -------------------------------------------------------------
function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(file);
  });
}

// Ultra-fast, off-thread thumbnail generator supporting Files, Blobs and strings without memory leaks
async function generateSmallThumbnail(source, maxDim = 96) {
  if (!source) return "";

  // 1. If source is a File or Blob, decode directly with createImageBitmap (runs off-thread in C++, 20-50x faster)
  if (source instanceof Blob || source instanceof File) {
    try {
      if (typeof createImageBitmap === "function") {
        let bmp;
        try {
          bmp = await createImageBitmap(source, {
            resizeWidth: maxDim,
            resizeHeight: maxDim,
            resizeQuality: "medium"
          });
        } catch (_) {
          bmp = await createImageBitmap(source);
        }
        if (bmp) {
          const scale = Math.min(maxDim / Math.max(bmp.width, 1), maxDim / Math.max(bmp.height, 1), 1);
          const tw = Math.max(1, Math.round(bmp.width * scale));
          const th = Math.max(1, Math.round(bmp.height * scale));
          const cvs = document.createElement("canvas");
          cvs.width = tw;
          cvs.height = th;
          const ctx = cvs.getContext("2d");
          ctx.drawImage(bmp, 0, 0, tw, th);
          bmp.close(); // Crucial: immediately release native bitmap memory
          return cvs.toDataURL("image/webp", 0.8) || cvs.toDataURL("image/jpeg", 0.7);
        }
      }
    } catch (_) {}

    // Fallback for Blob: Object URL (fast, avoids multi-MB base64 strings in memory)
    return new Promise((resolve) => {
      const url = URL.createObjectURL(source);
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(maxDim / Math.max(img.naturalWidth || 1, 1), maxDim / Math.max(img.naturalHeight || 1, 1), 1);
        const tw = Math.max(1, Math.round((img.naturalWidth || maxDim) * scale));
        const th = Math.max(1, Math.round((img.naturalHeight || maxDim) * scale));
        const cvs = document.createElement("canvas");
        cvs.width = tw;
        cvs.height = th;
        const ctx = cvs.getContext("2d");
        ctx.drawImage(img, 0, 0, tw, th);
        URL.revokeObjectURL(url);
        img.src = "";
        resolve(cvs.toDataURL("image/webp", 0.8) || cvs.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        img.src = "";
        resolve("");
      };
      img.src = url;
    });
  }

  // 2. If source is a string (data URL or file path)
  if (typeof source === "string") {
    if (source.length < 4096 && source.startsWith("data:")) return source;
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const nw = img.naturalWidth || img.width || maxDim;
        const nh = img.naturalHeight || img.height || maxDim;
        const scale = Math.min(maxDim / Math.max(nw, 1), maxDim / Math.max(nh, 1), 1);
        const tw = Math.max(1, Math.round(nw * scale));
        const th = Math.max(1, Math.round(nh * scale));
        const cvs = document.createElement("canvas");
        cvs.width = tw;
        cvs.height = th;
        const ctx = cvs.getContext("2d");
        ctx.drawImage(img, 0, 0, tw, th);
        img.src = ""; // Release memory
        resolve(cvs.toDataURL("image/webp", 0.8) || cvs.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => {
        img.src = "";
        resolve("");
      };
      img.src = source;
    });
  }

  return "";
}

async function ensureItemBase64(item) {
  if (!item) return null;
  if (item.base64) return item.base64;
  if (item.file) {
    try {
      item.base64 = await readFileAsDataUrl(item.file);
      return item.base64;
    } catch (e) {
      console.warn("Could not read file for item:", item.name, e);
      return null;
    }
  }
  return null;
}

// -------------------------------------------------------------
// Clipboard Paste Handler (Ctrl+V anywhere on the page)
// -------------------------------------------------------------
window.addEventListener("paste", async (e) => {
  if (!e.clipboardData || !e.clipboardData.items) return;
  const items = e.clipboardData.items;
  let foundImage = false;

  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf("image") !== -1) {
      const file = items[i].getAsFile();
      if (file) {
        foundImage = true;
        try {
          if (browserState && browserState.isOpen) {
            const base64Data = await readFileAsDataUrl(file);
            loadPastedImageIntoBrowserModal(base64Data);
          } else {
            const now = new Date();
            const timeStr = now.toTimeString().split(" ")[0];
            const name = `Pasted_${timeStr}`;
            const [thumb, base64Data] = await Promise.all([
              generateSmallThumbnail(file, 96),
              readFileAsDataUrl(file)
            ]);
            addToQueue({
              file: file,
              base64: base64Data,
              thumbnail: thumb,
              name: name,
              source: "paste"
            });
          }
        } catch (err) {
          console.error("Paste read error:", err);
        }
      }
    }
  }

  if (foundImage) {
    if (!browserState || !browserState.isOpen) {
      showToast("📋 Image pasted from clipboard!");
    }
  }
});

// -------------------------------------------------------------
// Drag-and-Drop, Multi-File & Folder Scanning Upload
// -------------------------------------------------------------
if (dropzone) {
  dropzone.addEventListener("click", (e) => {
    // If clicking a sub-button inside dropzone header, don't trigger fileInput
    if (e.target.closest(".btn-folder-load-pill")) return;
    if (fileInput) fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--primary)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "";
  });

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";

    // Support dropping whole folders via Webkit DataTransferItemList
    const items = e.dataTransfer.items;
    if (items && items.length && items[0].webkitGetAsEntry) {
      const fileList = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry();
        if (entry) {
          await scanFileSystemEntry(entry, fileList);
        }
      }
      if (fileList.length > 0) {
        handleMultipleFiles(fileList);
        return;
      }
    }

    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      handleMultipleFiles(e.dataTransfer.files);
    }
  });
}

// Recursively traverse folder hierarchies from drag-and-drop
async function scanFileSystemEntry(entry, fileList) {
  if (entry.isFile) {
    return new Promise((resolve) => {
      entry.file((file) => {
        if (file && (file.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|tiff?|avif)$/i.test(file.name))) {
          fileList.push(file);
        }
        resolve();
      }, () => resolve());
    });
  } else if (entry.isDirectory) {
    return new Promise((resolve) => {
      const dirReader = entry.createReader();
      const readEntries = () => {
        dirReader.readEntries(async (entries) => {
          if (!entries.length) {
            resolve();
          } else {
            for (const subEntry of entries) {
              await scanFileSystemEntry(subEntry, fileList);
            }
            readEntries();
          }
        }, () => resolve());
      };
      readEntries();
    });
  }
}

if (fileInput) {
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) {
      handleMultipleFiles(e.target.files);
      fileInput.value = ""; // Reset so selecting same files again triggers change
    }
  });
}

if (folderInput) {
  folderInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) {
      handleMultipleFiles(e.target.files);
      folderInput.value = ""; // Reset so selecting folder again triggers change
    }
  });
}

if (btnBrowseFolder) {
  btnBrowseFolder.addEventListener("click", (e) => {
    e.stopPropagation();
    if (folderInput) folderInput.click();
  });
}

if (btnBrowseFiles) {
  btnBrowseFiles.addEventListener("click", (e) => {
    e.stopPropagation();
    if (fileInput) fileInput.click();
  });
}

if (btnEmptyFolder) {
  btnEmptyFolder.addEventListener("click", () => {
    if (folderInput) folderInput.click();
  });
}

async function handleMultipleFiles(files) {
  const imageFiles = Array.from(files).filter(f => {
    return f && (f.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|tiff?|avif)$/i.test(f.name || ""));
  });

  if (!imageFiles.length) {
    showToast("⚠️ No compatible image files found in selection");
    return;
  }

  const count = imageFiles.length;
  showToast(`⏳ Loading ${count} image${count > 1 ? "s" : ""} into queue...`);

  // Controlled concurrency (4 at a time) for blistering-fast loading without freezing or memory spikes
  const CONCURRENCY = 4;
  const itemsToAdd = [];

  for (let i = 0; i < count; i += CONCURRENCY) {
    const chunk = imageFiles.slice(i, i + CONCURRENCY);
    const chunkResults = await Promise.all(chunk.map(async (f, idx) => {
      try {
        const isFirst = (i + idx) === 0 && state.queue.length === 0;
        // For the first item, read base64 immediately so workspace displays it instantly.
        // For remaining items, defer base64 reading on-demand to achieve lightning speed and protect RAM!
        const thumbPromise = generateSmallThumbnail(f, 96);
        const base64Promise = isFirst ? readFileAsDataUrl(f) : Promise.resolve(null);
        const [thumb, base64Data] = await Promise.all([thumbPromise, base64Promise]);

        return {
          name: f.name || `Image_${Date.now()}`,
          file: f,
          base64: base64Data,
          thumbnail: thumb,
          source: "upload"
        };
      } catch (err) {
        console.warn("Could not load image:", f.name, err);
        return null;
      }
    }));

    for (const res of chunkResults) {
      if (res) itemsToAdd.push(res);
    }

    // Yield to the event loop so the UI stays 100% fluid
    if (i + CONCURRENCY < count) {
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (itemsToAdd.length > 0) {
    addBatchToQueue(itemsToAdd);
    showToast(`📁 Added ${itemsToAdd.length} image${itemsToAdd.length > 1 ? "s" : ""} to queue!`);
  }
}

// -------------------------------------------------------------
// Queue Management System
// -------------------------------------------------------------
function addBatchToQueue(itemsList) {
  if (!itemsList || !itemsList.length) return;

  const wasEmpty = state.queue.length === 0;
  let firstNewId = null;

  for (const itemData of itemsList) {
    const item = {
      id: "item_" + Date.now() + "_" + Math.random().toString(36).substr(2, 6),
      name: itemData.name || "Untitled Sticker",
      file: itemData.file || null,
      base64: itemData.base64 || null,
      path: itemData.path || null,
      thumbnail: itemData.thumbnail || "",
      status: "queued", // 'queued' | 'processing' | 'done' | 'error'
      stages: null,
      metadata: null,
      savedPath: null
    };
    if (!firstNewId) firstNewId = item.id;
    state.queue.push(item);

    // Asynchronously generate thumbnail if missing (without blocking queue loading)
    if (!item.thumbnail) {
      const source = item.file || item.base64;
      if (source) {
        generateSmallThumbnail(source, 96).then(th => {
          if (th) {
            item.thumbnail = th;
            const thumbEl = document.querySelector(`.queue-item-row[onclick*="${item.id}"] .queue-thumb`);
            if (thumbEl) thumbEl.src = th;
          }
        });
      }
    }
  }

  // Render queue tray ONCE for the entire batch (prevents DOM thrashing)
  renderQueueTray();

  // If queue was empty or nothing was selected yet, select the first newly added item
  if (wasEmpty || !state.activeItemId) {
    if (firstNewId) {
      selectQueueItem(firstNewId);
    }
  }

  // Auto-Process only if the user explicitly has Auto-Process enabled
  if (toggleAutoProcess && toggleAutoProcess.checked && !state.isProcessing) {
    triggerQueueProcessor();
  }
}

function addToQueue(itemData) {
  addBatchToQueue([itemData]);
}

function selectQueueItem(itemId) {
  const item = state.queue.find(q => q.id === itemId);
  if (!item) return;

  state.activeItemId = item.id;
  state.currentImagePath = item.path;
  state.currentFileName = item.name;
  if (currentFileLabel) currentFileLabel.textContent = item.name;

  renderQueueTray();

  if (item.stages) {
    // Already processed: display results instantly!
    state.stages = item.stages;
    state.metadata = item.metadata;
    state.currentImageBase64 = (item.stages && (item.stages["final"] || item.stages["raw"])) || item.base64 || null;
    if (emptyWorkspaceView) emptyWorkspaceView.classList.add("hidden");
    if (displayWrapper) displayWrapper.classList.remove("hidden");
    updateDisplayView();
    updateMetrics(item.metadata);
    if (mainStickerImg) mainStickerImg.style.transform = "";
    statusText.textContent = "Ready";
    statusIndicator.classList.remove("busy");
  } else {
    // Unprocessed queue item: show in workspace ready for Run Script
    state.stages = {};
    state.metadata = null;
    if (emptyWorkspaceView) emptyWorkspaceView.classList.add("hidden");
    if (displayWrapper) displayWrapper.classList.remove("hidden");

    if (item.base64) {
      state.currentImageBase64 = item.base64;
      mainStickerImg.src = item.base64;
    } else if (item.file) {
      // Instant display using object URL (0ms latency, zero memory copy)
      const objUrl = URL.createObjectURL(item.file);
      mainStickerImg.src = objUrl;
      // Also ensure base64 is ready for processing in the background
      ensureItemBase64(item).then(b64 => {
        if (state.activeItemId === item.id) {
          state.currentImageBase64 = b64;
        }
      });
    } else {
      state.currentImageBase64 = null;
      mainStickerImg.src = item.path || "";
    }
    applyLiveTransforms();
    statusText.textContent = item.status === "processing" ? "Processing..." : "Queued (Ready to Process)";
    if (item.status === "processing") {
      statusIndicator.classList.add("busy");
    } else {
      statusIndicator.classList.remove("busy");
    }
  }
}

function removeQueueItem(itemId, event) {
  if (event) event.stopPropagation();
  const index = state.queue.findIndex(q => q.id === itemId);
  if (index === -1) return;

  state.queue.splice(index, 1);

  if (state.activeItemId === itemId) {
    if (state.queue.length > 0) {
      const nextIdx = Math.min(index, state.queue.length - 1);
      selectQueueItem(state.queue[nextIdx].id);
    } else {
      clearWorkspace();
    }
  } else {
    renderQueueTray();
  }
}

function handleQueueItemClick(itemId) {
  const item = state.queue.find(q => q.id === itemId);
  if (!item) return;
  const wasActive = state.activeItemId === itemId;
  selectQueueItem(itemId);
  // If user clicks on an already active queued sticker, process it immediately!
  if (wasActive && item.status === "queued" && !state.isProcessing) {
    runPipeline();
  }
}

function processSingleQueueItem(itemId, event) {
  if (event) event.stopPropagation();
  const item = state.queue.find(q => q.id === itemId);
  if (!item) return;
  if (state.isProcessing) {
    showToast("⏳ Pipeline is currently busy. Please wait...");
    return;
  }
  item.status = "queued";
  selectQueueItem(itemId);
  runPipeline();
}

function processAllQueue() {
  if (state.isProcessing) {
    showToast("⏳ Pipeline is already processing an item...");
    return;
  }
  if (!state.queue || state.queue.length === 0) {
    showToast("ℹ️ Queue is empty. Drop or paste images first!");
    return;
  }
  const hasQueued = state.queue.some(q => q.status === "queued");
  if (!hasQueued) {
    // If all items are done or errored, re-queue them so "Process All" runs
    state.queue.forEach(q => {
      q.status = "queued";
    });
    renderQueueTray();
    showToast("🔄 Re-queued all images for processing");
  }
  if (toggleAutoProcess) toggleAutoProcess.checked = true;
  triggerQueueProcessor();
}

function renderQueueTray() {
  if (!queueItemsContainer) return;
  const count = state.queue.length;
  if (queueBadgeCount) queueBadgeCount.textContent = `${count}`;

  if (count === 0) {
    queueItemsContainer.innerHTML = `
      <div class="queue-empty-text" id="queue-empty-text">
        No items in queue. Press <kbd>Ctrl+V</kbd> or drop images anytime.
      </div>
    `;
    return;
  }

  let html = "";
  for (let item of state.queue) {
    const isActive = item.id === state.activeItemId;
    const thumbSrc = item.thumbnail || (item.stages && item.stages["final"]) || "";

    let statusHtml = "";
    let actionBtnHtml = "";
    if (item.status === "queued") {
      statusHtml = `<span class="queue-item-status queued">⏳ Queued</span>`;
      actionBtnHtml = `<button class="btn-run-single-item" title="Process this sticker now" onclick="processSingleQueueItem('${item.id}', event)">▶</button>`;
    } else if (item.status === "processing") {
      statusHtml = `<span class="queue-item-status processing">🔄 Processing</span>`;
    } else if (item.status === "done") {
      statusHtml = `<span class="queue-item-status done">✓ Ready</span>`;
    } else if (item.status === "error") {
      statusHtml = `<span class="queue-item-status error">✕ Error</span>`;
      actionBtnHtml = `<button class="btn-run-single-item retry" title="Retry processing" onclick="processSingleQueueItem('${item.id}', event)">🔄</button>`;
    }

    html += `
      <div class="queue-item-row ${isActive ? "active" : ""}" onclick="handleQueueItemClick('${item.id}')">
        <img class="queue-thumb" src="${thumbSrc}" alt="thumb" onerror="this.style.opacity=0.3">
        <div class="queue-item-info">
          <span class="queue-item-name" title="${item.name}">${item.name}</span>
          ${statusHtml}
        </div>
        <div class="queue-item-actions">
          ${actionBtnHtml}
          <button class="btn-remove-queue-item" title="Remove" onclick="removeQueueItem('${item.id}', event)">✕</button>
        </div>
      </div>
    `;
  }

  queueItemsContainer.innerHTML = html;
}

// -------------------------------------------------------------
// Clear Workspace
// -------------------------------------------------------------
function clearWorkspace() {
  state.activeItemId = null;
  state.currentImagePath = null;
  state.currentImageBase64 = null;
  state.currentFileName = null;
  state.stages = {};
  state.metadata = null;
  state.manualRotation = 0;
  state.flipH = false;
  state.flipV = false;
  state.characterBbox = null;
  state.isRoiMode = false;
  state.roiStart = null;
  if (selectionBox) selectionBox.classList.add("hidden");
  if (canvasImageWrapper) canvasImageWrapper.classList.remove("selecting");
  if (btnToggleRoiMode) btnToggleRoiMode.classList.remove("active");
  if (btnViewerRoi) btnViewerRoi.classList.remove("active");
  if (roiBtnText) roiBtnText.textContent = "Box Tool";
  if (roiStatusText) {
    roiStatusText.textContent = "No character box set";
    roiStatusText.style.color = "var(--text-dim)";
  }
  if (btnClearRoi) btnClearRoi.classList.add("hidden");

  state.manualRotation = 0;
  state.horizontalSkew = 0;
  state.verticalSkew = 0;
  state.flipH = false;
  state.flipV = false;
  if (sliderRotation) sliderRotation.value = 0;
  if (badgeRot) badgeRot.textContent = "0°";
  if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
  if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
  if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
  if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";
  if (badgeAutoCelSummary) badgeAutoCelSummary.classList.add("hidden");
  if (mainStickerImg) mainStickerImg.style.transform = "";
  if (currentFileLabel) currentFileLabel.textContent = "No image loaded";

  state.cutoutVariants = [];
  state.activeVariantIndex = -1;
  if (inlineVariantsBar) inlineVariantsBar.classList.add("hidden");
  if (inlineVariantsChips) inlineVariantsChips.innerHTML = "";

  if (emptyWorkspaceView) emptyWorkspaceView.classList.remove("hidden");
  if (displayWrapper) displayWrapper.classList.add("hidden");
  if (mainStickerImg) mainStickerImg.src = "";
  if (stageSubBar) stageSubBar.classList.add("hidden");

  if (statTime) statTime.textContent = "--";
  if (statDims) statDims.textContent = "--";
  if (statRot) statRot.textContent = "0°";
  statusText.textContent = "Ready";
  statusIndicator.classList.remove("busy");

  renderQueueTray();
}

function clearAllQueue() {
  state.queue = [];
  clearWorkspace();
  showToast("🗑️ Queue cleared");
}

if (btnClearWorkspace) btnClearWorkspace.addEventListener("click", clearWorkspace);
if (btnClearSample) btnClearSample.addEventListener("click", clearWorkspace);
if (btnClearQueue) btnClearQueue.addEventListener("click", clearAllQueue);
if (btnProcessQueue) btnProcessQueue.addEventListener("click", processAllQueue);
if (btnBrowseEmpty) btnBrowseEmpty.addEventListener("click", () => fileInput.click());

if (toggleAutoProcess) {
  toggleAutoProcess.addEventListener("change", () => {
    if (toggleAutoProcess.checked && !state.isProcessing) {
      triggerQueueProcessor();
    }
  });
}

// -------------------------------------------------------------
// Sliders and Input Bindings (Smooth Real-time Live Transforms)
// -------------------------------------------------------------
function applyLiveTransforms() {
  if (!mainStickerImg) return;
  const rot = state.manualRotation || 0;
  const skewX = state.horizontalSkew || 0;
  const skewY = state.verticalSkew || 0;
  const flipX = state.flipH ? -1 : 1;
  const flipY = state.flipV ? -1 : 1;

  let t = "";
  if (flipX !== 1 || flipY !== 1) t += `scale(${flipX}, ${flipY}) `;
  if (rot !== 0) t += `rotate(${rot}deg) `;
  if (skewX !== 0) t += `skewX(${skewX}deg) `;
  if (skewY !== 0) t += `skewY(${skewY}deg) `;
  mainStickerImg.style.transform = t.trim();
  mainStickerImg.style.transformOrigin = "center center";

  // Real-time metric update in bottom bar
  if (statRot) {
    let parts = [`${rot.toFixed(0)}°`];
    if (skewX !== 0) parts.push(`SkX:${skewX > 0 ? "+" : ""}${skewX.toFixed(1)}°`);
    if (skewY !== 0) parts.push(`SkY:${skewY > 0 ? "+" : ""}${skewY.toFixed(1)}°`);
    statRot.textContent = parts.join(" ");
  }
}

if (sliderRotation) {
  sliderRotation.addEventListener("input", (e) => {
    state.manualRotation = parseInt(e.target.value);
    if (badgeRot) badgeRot.textContent = `${state.manualRotation}°`;
    applyLiveTransforms();
  });
}

if (sliderHorizontalSkew) {
  sliderHorizontalSkew.addEventListener("input", (e) => {
    state.horizontalSkew = parseFloat(e.target.value);
    const sign = state.horizontalSkew > 0 ? "+" : "";
    if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = `${sign}${state.horizontalSkew.toFixed(1)}°`;
    applyLiveTransforms();
  });
}

if (sliderVerticalSkew) {
  sliderVerticalSkew.addEventListener("input", (e) => {
    state.verticalSkew = parseFloat(e.target.value);
    const sign = state.verticalSkew > 0 ? "+" : "";
    if (badgeVerticalSkew) badgeVerticalSkew.textContent = `${sign}${state.verticalSkew.toFixed(1)}°`;
    applyLiveTransforms();
  });
}

if (btnResetSkew) {
  btnResetSkew.addEventListener("click", () => {
    state.horizontalSkew = 0;
    if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
    if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
    applyLiveTransforms();
  });
}

if (btnResetVSkew) {
  btnResetVSkew.addEventListener("click", () => {
    state.verticalSkew = 0;
    if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
    if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";
    applyLiveTransforms();
  });
}

if (btnResetAllTransforms) {
  btnResetAllTransforms.addEventListener("click", () => {
    state.manualRotation = 0;
    state.horizontalSkew = 0;
    state.verticalSkew = 0;
    state.flipH = false;
    state.flipV = false;
    if (sliderRotation) sliderRotation.value = 0;
    if (badgeRot) badgeRot.textContent = "0°";
    if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
    if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
    if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
    if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";
    applyLiveTransforms();
    showToast("↺ All manual skews and angles reset");
  });
}

if (btnBakeTransforms) {
  btnBakeTransforms.addEventListener("click", bakeManualTransforms);
}

if (btnGenerateSixVariants) {
  btnGenerateSixVariants.addEventListener("click", runSixVariants);
}
if (btnHeaderSixVariants) {
  btnHeaderSixVariants.addEventListener("click", runSixVariants);
}

if (sliderSmudgeSensitivity) {
  sliderSmudgeSensitivity.addEventListener("input", (e) => {
    badgeSmudge.textContent = `${e.target.value}%`;
  });
}

if (sliderAlphaThreshold) {
  sliderAlphaThreshold.addEventListener("input", (e) => {
    badgeAlpha.textContent = e.target.value;
  });
}

toggleSmudgeCleaner.addEventListener("change", () => {
  smudgeSensitivityContainer.classList.toggle("hidden", !toggleSmudgeCleaner.checked);
});

if (toggleShineRemover) {
  toggleShineRemover.addEventListener("change", () => {
    shineOptionsContainer.classList.toggle("hidden", !toggleShineRemover.checked);
    if (toggleShineRemover.checked) {
      triggerCelLivePreview();
    } else {
      const revertImg = (state.stages && (state.stages["character_art"] || state.stages["final"] || state.stages["raw"])) || state.currentImageBase64;
      if (revertImg && mainStickerImg) {
        mainStickerImg.src = revertImg;
      }
    }
  });
}

// Debounced Live Cel & De-Shine Preview
let celPreviewDebounceTimer = null;
function triggerCelLivePreview() {
  if (!toggleShineRemover || !toggleShineRemover.checked || !hasLoadedImage()) return;
  clearTimeout(celPreviewDebounceTimer);
  celPreviewDebounceTimer = setTimeout(async () => {
    try {
      const activeImg = (state.stages && (state.stages["character_art"] || state.stages["raw"])) || state.currentImageBase64;
      const res = await fetch("/api/cel-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: activeImg || state.currentImageBase64,
          image_path: state.currentImagePath,
          strength: sliderShineStrength ? parseInt(sliderShineStrength.value) : 60,
          color_tiers: sliderColorTiers ? parseInt(sliderColorTiers.value) : 40,
          flat_cel_look: toggleFlatCel ? toggleFlatCel.checked : false
        })
      });
      const data = await res.json();
      if (data.success && data.image_base64) {
        if (mainStickerImg) mainStickerImg.src = data.image_base64;
        if (!state.stages) state.stages = {};
        state.stages["cel_restored"] = data.image_base64;
      }
    } catch (e) {
      console.warn("Live Cel preview failed:", e);
    }
  }, 70);
}

if (sliderShineStrength) {
  sliderShineStrength.addEventListener("input", (e) => {
    badgeShine.textContent = `${e.target.value}%`;
    triggerCelLivePreview();
  });
}
if (sliderColorTiers) {
  sliderColorTiers.addEventListener("input", (e) => {
    if (badgeColorTiers) badgeColorTiers.textContent = `${e.target.value} Bands`;
    triggerCelLivePreview();
  });
}
if (toggleFlatCel) {
  toggleFlatCel.addEventListener("change", () => {
    triggerCelLivePreview();
  });
}
// Color Pop (Float32 OKLab) Controls & Live Preview
if (toggleColorPop) {
  toggleColorPop.addEventListener("change", () => {
    if (colorPopOptionsContainer) {
      colorPopOptionsContainer.classList.toggle("hidden", !toggleColorPop.checked);
    }
    if (toggleColorPop.checked) {
      triggerColorPopLivePreview();
    }
  });
}

let colorPopDebounceTimer = null;
function triggerColorPopLivePreview() {
  if (!toggleColorPop || !toggleColorPop.checked || !hasLoadedImage()) return;
  clearTimeout(colorPopDebounceTimer);
  colorPopDebounceTimer = setTimeout(async () => {
    try {
      const res = await fetch("/api/color-pop-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: state.currentImageBase64,
          image_path: state.currentImagePath,
          preset: selectColorPopPreset ? selectColorPopPreset.value : "anime_pop",
          ai_lora_enabled: toggleAiLora ? toggleAiLora.checked : false,
          ai_lora_preset: selectAiLoraPreset ? selectAiLoraPreset.value : "lora_shinkai",
          vibrance: sliderColorPopVibrance ? parseFloat(sliderColorPopVibrance.value) : 1.15,
          clarity: sliderColorPopClarity ? parseFloat(sliderColorPopClarity.value) : 0.05
        })
      });
      const data = await res.json();
      if (data.success && data.image_base64) {
        if (mainStickerImg) mainStickerImg.src = data.image_base64;
        if (!state.stages) state.stages = {};
        state.stages["color_enhanced"] = data.image_base64;
      }
    } catch (e) {
      console.warn("Live Color Pop preview failed:", e);
    }
  }, 120);
}

if (selectColorPopPreset) {
  selectColorPopPreset.addEventListener("change", () => {
    triggerColorPopLivePreview();
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}
if (sliderColorPopVibrance) {
  sliderColorPopVibrance.addEventListener("input", (e) => {
    if (badgeColorPopVibrance) badgeColorPopVibrance.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
    triggerColorPopLivePreview();
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}
if (sliderColorPopClarity) {
  sliderColorPopClarity.addEventListener("input", (e) => {
    if (badgeColorPopClarity) badgeColorPopClarity.textContent = `${parseFloat(e.target.value).toFixed(2)}`;
    triggerColorPopLivePreview();
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}

// Function to load and sync real LoRA presets from the server
async function syncServerLoraPresets() {
  if (!selectAiLoraPreset) return;
  try {
    const res = await fetch("/api/lora-presets");
    const data = await res.json();
    if (data.success && data.presets) {
      const curVal = selectAiLoraPreset.value;
      selectAiLoraPreset.innerHTML = "";
      for (const [pid, pcfg] of Object.entries(data.presets)) {
        if (pid.startsWith("lora_") || pcfg.format === "safetensors") {
          const opt = document.createElement("option");
          opt.value = pid;
          opt.textContent = pcfg.name || pid;
          selectAiLoraPreset.appendChild(opt);
        }
      }
      if (selectAiLoraPreset.querySelector(`option[value="${curVal}"]`)) {
        selectAiLoraPreset.value = curVal;
      } else {
        selectAiLoraPreset.value = "lora_real_vibrant";
      }
    }
  } catch (err) {
    console.warn("Could not sync server LoRA presets:", err);
  }
}

// AI LoRA Controls & Downloader
if (toggleAiLora) {
  toggleAiLora.addEventListener("change", () => {
    if (aiLoraOptionsContainer) {
      aiLoraOptionsContainer.classList.toggle("hidden", !toggleAiLora.checked);
    }
    if (toggleAiLora.checked) {
      syncServerLoraPresets();
      triggerColorPopLivePreview();
    }
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}

if (selectAiLoraPreset) {
  selectAiLoraPreset.addEventListener("change", () => {
    triggerColorPopLivePreview();
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}

if (btnDownloadLoras) {
  btnDownloadLoras.addEventListener("click", async () => {
    const origHtml = btnDownloadLoras.innerHTML;
    btnDownloadLoras.disabled = true;
    btnDownloadLoras.textContent = "⏳ Downloading...";
    if (loraStatusMsg) loraStatusMsg.textContent = "Downloading & verifying real .safetensors LoRAs...";

    try {
      const res = await fetch("/api/download-lora-presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      const data = await res.json();
      if (data.success && data.presets) {
        if (selectAiLoraPreset) {
          const curVal = selectAiLoraPreset.value;
          selectAiLoraPreset.innerHTML = "";
          for (const [pid, pcfg] of Object.entries(data.presets)) {
            if (pid.startsWith("lora_") || pcfg.format === "safetensors") {
              const opt = document.createElement("option");
              opt.value = pid;
              opt.textContent = pcfg.name || pid;
              selectAiLoraPreset.appendChild(opt);
            }
          }
          if (selectAiLoraPreset.querySelector(`option[value="${curVal}"]`)) {
            selectAiLoraPreset.value = curVal;
          } else {
            selectAiLoraPreset.value = "lora_real_vibrant";
          }
        }
        showToast("📥 Successfully downloaded real anime .safetensors LoRAs!");
        if (loraStatusMsg) {
          loraStatusMsg.textContent = `✓ ${data.message || "Real .safetensors LoRA presets ready"}`;
          loraStatusMsg.style.color = "#4ade80";
        }
        triggerColorPopLivePreview();
        if (typeof scheduleAutosave === "function") scheduleAutosave();
      } else {
        showToast("⚠️ Failed to download presets: " + (data.error || "Unknown error"));
        if (loraStatusMsg) loraStatusMsg.textContent = data.error || "Download error";
      }
    } catch (err) {
      console.error("Error downloading LoRAs:", err);
      showToast("⚠️ Could not reach server for preset download.");
      if (loraStatusMsg) loraStatusMsg.textContent = "Network error";
    } finally {
      btnDownloadLoras.disabled = false;
      btnDownloadLoras.innerHTML = origHtml;
    }
  });
}

// Auto Cel Analysis & Recommended Settings Button
if (btnAutoCel) {
  btnAutoCel.addEventListener("click", async () => {
    if (!hasLoadedImage()) {
      showToast("⚠️ Load an image first to run Auto Cel analysis!");
      return;
    }
    const origText = btnAutoCel.innerHTML;
    btnAutoCel.disabled = true;
    btnAutoCel.innerHTML = "⏳ Analyzing Artwork & Glare...";
    showLoadingOverlay("AI Cel & De-Shine Restorer", "Analyzing artwork lighting, glare, and calculating optimal cel tiers...", 35);

    const activeImg = (state.stages && (state.stages["character_art"] || state.stages["raw"])) || state.currentImageBase64;

    try {
      const res = await fetch("/api/auto-cel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: activeImg || state.currentImageBase64,
          image_path: state.currentImagePath
        })
      });
      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      // Activate Cel Restorer & apply recommended parameters
      if (toggleShineRemover) {
        toggleShineRemover.checked = true;
        if (shineOptionsContainer) shineOptionsContainer.classList.remove("hidden");
      }
      if (sliderShineStrength) {
        sliderShineStrength.value = data.recommended_strength;
        badgeShine.textContent = `${data.recommended_strength}%`;
      }
      if (sliderColorTiers) {
        sliderColorTiers.value = data.recommended_tiers;
        if (badgeColorTiers) badgeColorTiers.textContent = `${data.recommended_tiers} Bands`;
      }

      if (badgeAutoCelSummary) {
        const m = data.metrics || {};
        badgeAutoCelSummary.innerHTML = `✨ <strong>Auto Cel:</strong> Glare ${m.glare_ratio_pct || 0}% • Line Density ${m.edge_density_pct || 0}%<br>Applied: <strong>${data.recommended_strength}%</strong> de-shine, <strong>${data.recommended_tiers}</strong> color tiers.`;
        badgeAutoCelSummary.classList.remove("hidden");
      }

      // Directly apply preview if returned from server
      if (data.preview_base64) {
        if (mainStickerImg) mainStickerImg.src = data.preview_base64;
        if (!state.stages) state.stages = {};
        state.stages["cel_restored"] = data.preview_base64;
      } else {
        triggerCelLivePreview();
      }

      showToast(`✨ Auto Cel Applied: ${data.recommended_strength}% suppression, ${data.recommended_tiers} tiers`);
    } catch (err) {
      console.error("Auto Cel error:", err);
      showToast(`❌ Auto Cel Error: ${err.message}`);
    } finally {
      btnAutoCel.disabled = false;
      btnAutoCel.innerHTML = origText;
      hideLoadingOverlay();
    }
  });
}

// Highlighting Controls (Swatches, Color Picker, Glow Radius)
if (swatchBtns) {
  swatchBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      swatchBtns.forEach(b => {
        b.classList.remove("active");
        b.style.borderColor = "rgba(255,255,255,0.2)";
      });
      btn.classList.add("active");
      btn.style.borderColor = "#818cf8";
      selectedHighlightColor = btn.dataset.color;
      if (pickerHighlightColor) pickerHighlightColor.value = selectedHighlightColor;
    });
  });
}

if (pickerHighlightColor) {
  pickerHighlightColor.addEventListener("input", (e) => {
    selectedHighlightColor = e.target.value;
    swatchBtns.forEach(b => {
      b.classList.remove("active");
      b.style.borderColor = "rgba(255,255,255,0.2)";
    });
  });
}

if (sliderGlowRadius && badgeGlowRadius) {
  sliderGlowRadius.addEventListener("input", (e) => {
    badgeGlowRadius.textContent = `${e.target.value} px`;
  });
}

sliderBorderWidth.addEventListener("input", (e) => {
  badgeBorder.textContent = `${e.target.value} px`;
});

if (sliderBorderSmoothing) {
  sliderBorderSmoothing.addEventListener("input", (e) => {
    const v = parseInt(e.target.value);
    if (badgeBorderSmoothing) {
      if (v === 0) badgeBorderSmoothing.textContent = "0% (Sharp)";
      else if (v === 50) badgeBorderSmoothing.textContent = "50% (Smooth)";
      else if (v === 100) badgeBorderSmoothing.textContent = "100% (Ultra)";
      else badgeBorderSmoothing.textContent = `${v}%`;
    }
    if (typeof scheduleAutosave === "function") scheduleAutosave();
  });
}

toggleBorder.addEventListener("change", () => {
  borderSliderContainer.classList.toggle("hidden", !toggleBorder.checked);
});

btnExtractChar.addEventListener("click", () => {
  btnExtractChar.classList.add("active");
  btnExtractFull.classList.remove("active");
});

btnExtractFull.addEventListener("click", () => {
  btnExtractFull.classList.add("active");
  btnExtractChar.classList.remove("active");
});

// Super-Resolution Option Visibility & Interaction Listeners
if (toggleSuperRes) {
  toggleSuperRes.addEventListener("change", () => {
    if (superResOptionsContainer) {
      superResOptionsContainer.classList.toggle("sr-disabled", !toggleSuperRes.checked);
      superResOptionsContainer.classList.remove("hidden");
    }
  });
}

if (selectEnhancerModel) {
  selectEnhancerModel.addEventListener("change", () => {
    if (toggleSuperRes) {
      toggleSuperRes.checked = true;
    }
    if (superResOptionsContainer) {
      superResOptionsContainer.classList.remove("sr-disabled");
      superResOptionsContainer.classList.remove("hidden");
    }
  });
}

if (selectEnhancerScale) {
  selectEnhancerScale.addEventListener("change", () => {
    if (toggleSuperRes) {
      toggleSuperRes.checked = true;
    }
    if (superResOptionsContainer) {
      superResOptionsContainer.classList.remove("sr-disabled");
      superResOptionsContainer.classList.remove("hidden");
    }
  });
}

// -------------------------------------------------------------
// Character Highlighter Brush Tool (Freehand character isolation)
// -------------------------------------------------------------
function clearHighlightCanvas() {
  if (!highlightCanvas) return;
  const ctx = highlightCanvas.getContext("2d");
  ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
  state.hasHighlightStrokes = false;
  state.characterHighlightMask = null;
  state.characterBbox = null;
  if (selectionBox) selectionBox.classList.add("hidden");
  if (roiStatusText) {
    roiStatusText.textContent = "No character highlighted";
    roiStatusText.style.color = "var(--text-dim)";
  }
  if (btnClearRoi) btnClearRoi.classList.add("hidden");
}

function syncHighlightCanvasSize() {
  if (!highlightCanvas || !mainStickerImg) return;
  const nw = mainStickerImg.naturalWidth || 512;
  const nh = mainStickerImg.naturalHeight || 512;
  if (highlightCanvas.width !== nw || highlightCanvas.height !== nh) {
    const oldW = highlightCanvas.width;
    const oldH = highlightCanvas.height;
    if (oldW > 0 && oldH > 0 && state.hasHighlightStrokes) {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = oldW;
      tempCanvas.height = oldH;
      tempCanvas.getContext("2d").drawImage(highlightCanvas, 0, 0);
      highlightCanvas.width = nw;
      highlightCanvas.height = nh;
      highlightCanvas.getContext("2d").drawImage(tempCanvas, 0, 0, nw, nh);
    } else {
      highlightCanvas.width = nw;
      highlightCanvas.height = nh;
    }
  }
}

function setRoiMode(active) {
  if (active && !hasLoadedImage()) {
    showToast("⚠️ Load an image before opening mask tool");
    return;
  }
  if (active && state.isInpaintMode) {
    setInpaintMode(false);
  }
  state.isRoiMode = active;
  if (canvasImageWrapper) canvasImageWrapper.classList.toggle("selecting", active);
  if (btnToggleRoiMode) btnToggleRoiMode.classList.toggle("active", active);
  if (btnViewerRoi) btnViewerRoi.classList.toggle("active", active);
  if (roiBtnText) roiBtnText.textContent = active ? "Painting..." : "Mask Tool";
  if (roiHighlighterControls) roiHighlighterControls.classList.toggle("hidden", !active);

  if (highlightCanvas) {
    highlightCanvas.classList.toggle("hidden", !active);
    if (active) {
      syncHighlightCanvasSize();
    }
  }

  if (active) {
    showToast("🎨 Mask Tool: Cyan = Keep character • Red = Cut/Exclude unwanted parts");
  }
}

if (btnToggleRoiMode) {
  btnToggleRoiMode.addEventListener("click", () => {
    setRoiMode(!state.isRoiMode);
  });
}

if (btnViewerRoi) {
  btnViewerRoi.addEventListener("click", () => {
    setRoiMode(!state.isRoiMode);
  });
}

if (btnBrushPaint) {
  btnBrushPaint.addEventListener("click", () => {
    state.brushMode = "paint";
    btnBrushPaint.classList.add("active");
    if (btnBrushNegative) btnBrushNegative.classList.remove("active");
    if (btnBrushErase) btnBrushErase.classList.remove("active");
    showToast("🖌️ Keep brush: Paint over parts to keep");
  });
}

if (btnBrushNegative) {
  btnBrushNegative.addEventListener("click", () => {
    state.brushMode = "negative";
    btnBrushNegative.classList.add("active");
    if (btnBrushPaint) btnBrushPaint.classList.remove("active");
    if (btnBrushErase) btnBrushErase.classList.remove("active");
    showToast("🚫 Exclude brush: Paint over parts to strictly cut out");
  });
}

if (btnBrushErase) {
  btnBrushErase.addEventListener("click", () => {
    state.brushMode = "erase";
    btnBrushErase.classList.add("active");
    if (btnBrushPaint) btnBrushPaint.classList.remove("active");
    if (btnBrushNegative) btnBrushNegative.classList.remove("active");
    showToast("🧽 Clear eraser: Erase painted brush strokes");
  });
}

if (sliderHighlightBrush) {
  sliderHighlightBrush.addEventListener("input", (e) => {
    state.highlightBrushSize = parseInt(e.target.value);
    if (badgeHighlightBrush) badgeHighlightBrush.textContent = `${state.highlightBrushSize}px`;
  });
}

if (btnClearRoi) {
  btnClearRoi.addEventListener("click", (e) => {
    e.stopPropagation();
    clearHighlightCanvas();
    showToast("Character mask cleared");
  });
}

// Character highlighter canvas pointer events
if (highlightCanvas) {
  function getHighlightCanvasPos(e) {
    const rect = highlightCanvas.getBoundingClientRect();
    const scaleX = highlightCanvas.width / (rect.width || 1);
    const scaleY = highlightCanvas.height / (rect.height || 1);
    return {
      x: Math.max(0, Math.min(highlightCanvas.width, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(highlightCanvas.height, (e.clientY - rect.top) * scaleY)),
      scaleX: scaleX
    };
  }

  highlightCanvas.addEventListener("pointerdown", (e) => {
    if (!state.isRoiMode) return;
    if (e.button !== 0) return;
    e.preventDefault();
    try { highlightCanvas.setPointerCapture(e.pointerId); } catch (_) {}
    state.isDrawingHighlight = true;
    state.hasHighlightStrokes = true;

    const ctx = highlightCanvas.getContext("2d");
    const pos = getHighlightCanvasPos(e);
    const actualBrushWidth = (state.highlightBrushSize || 32) * pos.scaleX;

    const isErase = state.brushMode === "erase";
    const isNegative = state.brushMode === "negative";

    ctx.globalCompositeOperation = isErase ? "destination-out" : "source-over";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = actualBrushWidth;

    if (isErase) {
      ctx.strokeStyle = "rgba(0, 0, 0, 1.0)";
      ctx.fillStyle = "rgba(0, 0, 0, 1.0)";
    } else if (isNegative) {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.70)";
      ctx.fillStyle = "rgba(239, 68, 68, 0.70)";
    } else {
      ctx.strokeStyle = "rgba(0, 229, 255, 0.65)";
      ctx.fillStyle = "rgba(0, 229, 255, 0.65)";
    }

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, actualBrushWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    if (roiStatusText) {
      if (isErase) {
        roiStatusText.textContent = "🧽 Erasing Mask Strokes";
        roiStatusText.style.color = "#fbbf24";
      } else if (isNegative) {
        roiStatusText.textContent = "🚫 Exclude Mask Painted (Cut Out)";
        roiStatusText.style.color = "#f87171";
      } else {
        roiStatusText.textContent = "✨ Keep Mask Painted (Isolate)";
        roiStatusText.style.color = "#00e5ff";
      }
    }
    if (btnClearRoi) btnClearRoi.classList.remove("hidden");
  });

  highlightCanvas.addEventListener("pointermove", (e) => {
    if (!state.isRoiMode || !state.isDrawingHighlight) return;
    e.preventDefault();
    const ctx = highlightCanvas.getContext("2d");
    const pos = getHighlightCanvasPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  });

  const endHighlightDrawing = (e) => {
    if (state.isDrawingHighlight) {
      const ctx = highlightCanvas.getContext("2d");
      ctx.beginPath();
      state.isDrawingHighlight = false;
      try { highlightCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
    }
  };

  highlightCanvas.addEventListener("pointerup", endHighlightDrawing);
  highlightCanvas.addEventListener("pointercancel", endHighlightDrawing);
}

// -------------------------------------------------------------
// Big-LaMa Neural Inpainting & Fill Brush Tool
// -------------------------------------------------------------
function clearInpaintCanvas() {
  if (!inpaintCanvas) return;
  const ctx = inpaintCanvas.getContext("2d");
  ctx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
  state.hasInpaintStrokes = false;
}

function syncInpaintCanvasSize() {
  if (!inpaintCanvas || !mainStickerImg) return;
  const nw = mainStickerImg.naturalWidth || 512;
  const nh = mainStickerImg.naturalHeight || 512;
  if (inpaintCanvas.width !== nw || inpaintCanvas.height !== nh) {
    inpaintCanvas.width = nw;
    inpaintCanvas.height = nh;
    clearInpaintCanvas();
  }
}

function setInpaintMode(active) {
  if (active && !hasLoadedImage()) {
    showToast("⚠️ Load an image before opening the brush tool");
    return;
  }
  if (active && state.isRoiMode) {
    setRoiMode(false);
  }
  state.isInpaintMode = active;
  if (btnInpaintBrush) btnInpaintBrush.classList.toggle("active", active);
  if (inpaintFloatingBar) inpaintFloatingBar.classList.toggle("hidden", !active);
  if (inpaintCanvas) {
    inpaintCanvas.classList.toggle("hidden", !active);
    if (active) {
      syncInpaintCanvasSize();
    }
  }
  if (active) {
    showToast("🎨 Brush over what you want to remove or fill in!");
  }
}

if (btnInpaintBrush) {
  btnInpaintBrush.addEventListener("click", () => {
    setInpaintMode(!state.isInpaintMode);
  });
}

if (btnInpaintClose) {
  btnInpaintClose.addEventListener("click", () => {
    setInpaintMode(false);
  });
}

if (sliderInpaintBrush) {
  sliderInpaintBrush.addEventListener("input", (e) => {
    state.inpaintBrushSize = parseInt(e.target.value);
    if (badgeInpaintSize) badgeInpaintSize.textContent = `${state.inpaintBrushSize} px`;
  });
}

// Model selector: update button text when model changes
if (selectInpaintModel) {
  selectInpaintModel.addEventListener("change", (e) => {
    const modelLabels = { anime: "AnimeLaMa", general: "Big-LaMa" };
    const label = modelLabels[e.target.value] || e.target.value;
    if (btnInpaintText) btnInpaintText.textContent = `Fill In (${label})`;
  });
}

if (btnInpaintClear) {
  btnInpaintClear.addEventListener("click", () => {
    clearInpaintCanvas();
    showToast("Brush strokes cleared");
  });
}

// Inpaint canvas pointer / brush drawing
if (inpaintCanvas) {
  function getCanvasPos(e) {
    const rect = inpaintCanvas.getBoundingClientRect();
    const scaleX = inpaintCanvas.width / (rect.width || 1);
    const scaleY = inpaintCanvas.height / (rect.height || 1);
    return {
      x: Math.max(0, Math.min(inpaintCanvas.width, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(inpaintCanvas.height, (e.clientY - rect.top) * scaleY)),
      scaleX: scaleX
    };
  }

  inpaintCanvas.addEventListener("pointerdown", (e) => {
    if (!state.isInpaintMode) return;
    if (e.button !== 0) return;
    e.preventDefault();
    try { inpaintCanvas.setPointerCapture(e.pointerId); } catch (_) {}
    state.isDrawingInpaint = true;
    state.hasInpaintStrokes = true;

    const ctx = inpaintCanvas.getContext("2d");
    const pos = getCanvasPos(e);

    const actualBrushWidth = state.inpaintBrushSize * pos.scaleX;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = actualBrushWidth;
    ctx.strokeStyle = "rgba(244, 63, 94, 0.72)";
    ctx.fillStyle = "rgba(244, 63, 94, 0.72)";

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, actualBrushWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  });

  inpaintCanvas.addEventListener("pointermove", (e) => {
    if (!state.isInpaintMode || !state.isDrawingInpaint) return;
    e.preventDefault();
    const ctx = inpaintCanvas.getContext("2d");
    const pos = getCanvasPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  });

  const endDrawing = (e) => {
    if (state.isDrawingInpaint) {
      const ctx = inpaintCanvas.getContext("2d");
      ctx.beginPath();
      state.isDrawingInpaint = false;
      try { inpaintCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
    }
  };

  inpaintCanvas.addEventListener("pointerup", endDrawing);
  inpaintCanvas.addEventListener("pointercancel", endDrawing);
}

if (mainStickerImg) {
  mainStickerImg.addEventListener("load", () => {
    if (state.isInpaintMode) {
      syncInpaintCanvasSize();
    }
    if (state.isRoiMode) {
      syncHighlightCanvasSize();
    }
  });
}

// Inpainting execution with Big-LaMa
async function runInpaintFill() {
  if (!hasLoadedImage()) {
    showToast("⚠️ No image loaded to fill");
    return;
  }
  if (!inpaintCanvas || !state.hasInpaintStrokes) {
    showToast("⚠️ Highlight an obstruction with the brush first!");
    return;
  }

  // Generate pure black & white binary mask from inpaintCanvas
  const w = inpaintCanvas.width;
  const h = inpaintCanvas.height;
  const offCanvas = document.createElement("canvas");
  offCanvas.width = w;
  offCanvas.height = h;
  const offCtx = offCanvas.getContext("2d");

  // Background black
  offCtx.fillStyle = "#000000";
  offCtx.fillRect(0, 0, w, h);

  const strokeImgData = inpaintCanvas.getContext("2d").getImageData(0, 0, w, h);
  const maskImgData = offCtx.createImageData(w, h);
  let markedCount = 0;

  for (let i = 0; i < strokeImgData.data.length; i += 4) {
    const alpha = strokeImgData.data[i + 3];
    if (alpha > 20) {
      maskImgData.data[i] = 255;
      maskImgData.data[i + 1] = 255;
      maskImgData.data[i + 2] = 255;
      maskImgData.data[i + 3] = 255;
      markedCount++;
    } else {
      maskImgData.data[i] = 0;
      maskImgData.data[i + 1] = 0;
      maskImgData.data[i + 2] = 0;
      maskImgData.data[i + 3] = 255;
    }
  }

  if (markedCount < 15) {
    showToast("⚠️ Highlight an obstruction with the brush first!");
    return;
  }

  offCtx.putImageData(maskImgData, 0, 0);
  const maskBase64 = offCanvas.toDataURL("image/png");

  // Determine current image base64
  let currentImgBase64 = state.currentImageBase64;
  if (state.stages && state.stages[state.activeStage]) {
    currentImgBase64 = state.stages[state.activeStage];
  } else if (mainStickerImg && mainStickerImg.src && mainStickerImg.src.startsWith("data:")) {
    currentImgBase64 = mainStickerImg.src;
  }

  // Fallback: extract directly from rendered sticker element if base64 not yet generated
  if (!currentImgBase64 && mainStickerImg && mainStickerImg.complete && mainStickerImg.naturalWidth > 0) {
    try {
      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = mainStickerImg.naturalWidth;
      snapCanvas.height = mainStickerImg.naturalHeight;
      const sCtx = snapCanvas.getContext("2d");
      sCtx.drawImage(mainStickerImg, 0, 0);
      currentImgBase64 = snapCanvas.toDataURL("image/png");
    } catch (e) {
      console.warn("Could not capture image from element:", e);
    }
  }

  // Set UI busy state
  const currentModelType = selectInpaintModel ? selectInpaintModel.value : "anime";
  const modelLabels = { anime: "AnimeLaMa", general: "Big-LaMa" };
  const modelLabel = modelLabels[currentModelType] || currentModelType;
  if (btnInpaintRun) {
    btnInpaintRun.disabled = true;
    if (btnInpaintText) btnInpaintText.textContent = `${modelLabel} Filling...`;
  }
  if (statusIndicator) {
    statusIndicator.classList.add("busy");
    if (statusText) statusText.textContent = `${modelLabel} Filling...`;
  }
  showLoadingOverlay(`AI Neural Inpainting (${modelLabel})`, `Reconstructing painted area with ${modelLabel}...`, 40);

  try {
    const res = await fetch("/api/inpaint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: currentImgBase64,
        image_path: state.currentImagePath,
        mask_base64: maskBase64,
        dilate_px: 8,
        model_type: currentModelType
      })
    });

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || "Inpainting failed.");
    }

    // Update current active image with inpainted result!
    const filledBase64 = data.image_base64;
    state.currentImageBase64 = filledBase64;
    state.currentImagePath = null; // now direct memory base64
    mainStickerImg.src = filledBase64;

    // Update stages so raw / current stages reflect the clean character
    if (!state.stages) state.stages = {};
    state.stages["raw"] = filledBase64;
    state.stages[state.activeStage] = filledBase64;

    // Update active queue item
    if (state.activeItemId) {
      const item = state.queue.find(q => q.id === state.activeItemId);
      if (item) {
        item.base64 = filledBase64;
        item.thumbnail = filledBase64;
        if (item.stages) {
          item.stages["raw"] = filledBase64;
          item.stages[state.activeStage] = filledBase64;
        }
        renderQueueTray();
      }
    }

    clearInpaintCanvas();
    showToast(`✨ Filled in seamlessly with ${modelLabel}!`);
  } catch (err) {
    console.error("[Inpaint Error]", err);
    showToast(`❌ Fill Error: ${err.message}`);
  } finally {
    if (btnInpaintRun) {
      btnInpaintRun.disabled = false;
      if (btnInpaintText) btnInpaintText.textContent = `Fill In (${modelLabel})`;
    }
    if (statusIndicator) {
      statusIndicator.classList.remove("busy");
      if (statusText) statusText.textContent = "Ready";
    }
    hideLoadingOverlay();
  }
}

if (btnInpaintRun) {
  btnInpaintRun.addEventListener("click", runInpaintFill);
}

if (btnAutoFillHoles) {
  btnAutoFillHoles.addEventListener("click", async () => {
    if (!hasLoadedImage()) {
      showToast("⚠️ Load an image first");
      return;
    }

    const currentModelType = selectInpaintModel ? selectInpaintModel.value : "anime";
    btnAutoFillHoles.disabled = true;
    if (btnAutoFillText) btnAutoFillText.textContent = "Detecting & Healing...";
    if (statusIndicator) {
      statusIndicator.classList.add("busy");
      if (statusText) statusText.textContent = "Auto-filling cutout holes...";
    }
    showLoadingOverlay("AI Neural Fill: Auto-Fill Holes", "Scanning and healing cutout voids with AnimeLaMa...", 45);

    let currentImgBase64 = state.currentImageBase64;
    if (state.stages && state.stages[state.activeStage]) {
      currentImgBase64 = state.stages[state.activeStage];
    } else if (mainStickerImg && mainStickerImg.src && mainStickerImg.src.startsWith("data:")) {
      currentImgBase64 = mainStickerImg.src;
    }

    try {
      const res = await fetch("/api/auto-fill-holes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: currentImgBase64,
          image_path: state.currentImagePath,
          model_type: currentModelType
        })
      });

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Auto-fill failed");
      }

      const filledBase64 = data.image_base64;
      state.currentImageBase64 = filledBase64;
      state.currentImagePath = null;
      mainStickerImg.src = filledBase64;

      if (!state.stages) state.stages = {};
      state.stages["raw"] = filledBase64;
      state.stages[state.activeStage] = filledBase64;

      if (state.activeItemId) {
        const item = state.queue.find(q => q.id === state.activeItemId);
        if (item) {
          item.base64 = filledBase64;
          item.thumbnail = filledBase64;
          if (item.stages) {
            item.stages["raw"] = filledBase64;
            item.stages[state.activeStage] = filledBase64;
          }
        }
      }

      const count = data.holes_filled || 0;
      if (count > 0) {
        showToast(`✨ Auto-healed ${count} cutout void${count > 1 ? "s" : ""}!`);
      } else {
        showToast("ℹ️ No internal cutout holes found to heal.");
      }
    } catch (err) {
      console.error(err);
      showToast(`❌ ${err.message || "Auto-fill error"}`);
    } finally {
      btnAutoFillHoles.disabled = false;
      if (btnAutoFillText) btnAutoFillText.textContent = "✨ Auto-Fill Holes";
      if (statusIndicator) {
        statusIndicator.classList.remove("busy");
        if (statusText) statusText.textContent = "Ready";
      }
      hideLoadingOverlay();
    }
  });
}

// 1-Click Rotation Actions (Instant live transforms without reloading or resetting)
btnRotCCW.addEventListener("click", () => {
  state.manualRotation = (state.manualRotation - 90 + 360) % 360;
  if (state.manualRotation > 180) state.manualRotation -= 360;
  if (sliderRotation) sliderRotation.value = state.manualRotation;
  if (badgeRot) badgeRot.textContent = `${state.manualRotation}°`;
  applyLiveTransforms();
});

btnRotCW.addEventListener("click", () => {
  state.manualRotation = (state.manualRotation + 90) % 360;
  if (state.manualRotation > 180) state.manualRotation -= 360;
  if (sliderRotation) sliderRotation.value = state.manualRotation;
  if (badgeRot) badgeRot.textContent = `${state.manualRotation}°`;
  applyLiveTransforms();
});

btnFlipH.addEventListener("click", () => {
  state.flipH = !state.flipH;
  applyLiveTransforms();
});

btnRotReset.addEventListener("click", () => {
  state.manualRotation = 0;
  state.horizontalSkew = 0;
  state.verticalSkew = 0;
  state.flipH = false;
  state.flipV = false;
  if (sliderRotation) sliderRotation.value = 0;
  if (badgeRot) badgeRot.textContent = "0°";
  if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
  if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
  if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
  if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";
  applyLiveTransforms();
  showToast("↺ Rotation & Skews Reset to 0°");
});

function hasLoadedImage() {
  const active = state.queue.find(q => q.id === state.activeItemId);
  return Boolean(state.currentImagePath || state.currentImageBase64 || (active && (active.base64 || active.file || active.path)));
}

// -------------------------------------------------------------
// View Mode Tabs
// -------------------------------------------------------------
document.querySelectorAll(".mode-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    state.viewMode = tab.dataset.view;
    updateDisplayView();
  });
});

document.querySelectorAll(".stage-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".stage-pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    state.activeStage = pill.dataset.stage;
    renderCurrentStage();
  });
});

function updateDisplayView() {
  if (!hasLoadedImage()) return;

  // Show/hide intermediate pre-processing pills if stages exist
  if (pillStageWatermark) {
    pillStageWatermark.classList.toggle("hidden", !state.stages["watermark_cleaned"]);
  }
  if (pillStageCel) {
    pillStageCel.classList.toggle("hidden", !state.stages["cel_restored"]);
  }
  if (pillStageColorPop) {
    pillStageColorPop.classList.toggle("hidden", !state.stages["color_enhanced"]);
  }

  if (state.viewMode === "result") {
    stageSubBar.classList.add("hidden");
    mainStickerImg.src = state.stages["final"] || "";
  } else if (state.viewMode === "raw") {
    stageSubBar.classList.add("hidden");
    mainStickerImg.src = state.stages["raw"] || "";
  } else if (state.viewMode === "stages") {
    stageSubBar.classList.remove("hidden");
    renderCurrentStage();
  }
}

function renderCurrentStage() {
  const src = state.stages[state.activeStage] || state.stages["final"];
  if (src) {
    mainStickerImg.src = src;
  }
}

// -------------------------------------------------------------
// Dynamic Progress Bar Animation
// -------------------------------------------------------------
function startProgressAnimation() {
  let currentPct = 6;
  showLoadingOverlay("Processing Sticker...", "Initializing pipeline...", currentPct);

  const pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/progress?t=${Date.now()}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.success && typeof data.percent === "number") {
        if (data.percent >= 100) {
          currentPct = 100;
          if (pipelineProgressBar) pipelineProgressBar.style.width = "100%";
          if (loadingPercent) loadingPercent.textContent = "100%";
          if (loadingTitle) loadingTitle.textContent = data.status || "Asset ready!";
          if (loadingStepDesc) loadingStepDesc.textContent = "Finalizing asset & loading...";
        } else if (data.percent > 0) {
          currentPct = Math.max(currentPct, Math.min(99, Math.round(data.percent)));
          if (pipelineProgressBar) pipelineProgressBar.style.width = `${currentPct}%`;
          if (loadingPercent) loadingPercent.textContent = `${currentPct}%`;
          if (data.status && loadingTitle) loadingTitle.textContent = data.status;
          if (data.detail && loadingStepDesc) loadingStepDesc.textContent = data.detail;
        }
      }
    } catch (_) {
      // Ignore transient polling errors
    }
  }, 120);

  return pollInterval;
}

// -------------------------------------------------------------
// Auto-Processor Queue Trigger
// -------------------------------------------------------------
function triggerQueueProcessor() {
  if (state.isProcessing) return;

  // Find next queued item
  const nextItem = state.queue.find(q => q.status === "queued");
  if (nextItem) {
    runPipeline(nextItem.id);
  }
}

// -------------------------------------------------------------
// Cancellation Support
// -------------------------------------------------------------
function cancelCurrentProcess() {
  if (!state.isProcessing && loadingOverlay.classList.contains("hidden")) {
    return;
  }

  if (currentAbortController) {
    try {
      currentAbortController.abort();
    } catch (_) {}
    currentAbortController = null;
  }

  if (currentProgressInterval) {
    clearInterval(currentProgressInterval);
    currentProgressInterval = null;
  }

  state.isProcessing = false;
  hideLoadingOverlay();
  statusIndicator.classList.remove("busy");
  statusText.textContent = "Cancelled";

  let activeItem = state.queue.find(q => q.id === state.activeItemId);
  if (activeItem && activeItem.status === "processing") {
    if (activeItem.stages && activeItem.stages["final"]) {
      activeItem.status = "done";
    } else {
      activeItem.status = "queued";
    }
    renderQueueTray();
  }

  showToast("🛑 Process cancelled");
}

// -------------------------------------------------------------
// Main Pipeline Execution
// -------------------------------------------------------------
async function runPipeline(targetItemId = null) {
  // If targetItemId specified, select it
  if (targetItemId) {
    selectQueueItem(targetItemId);
  }

  // Get active queue item
  let activeItem = state.queue.find(q => q.id === state.activeItemId);
  if (!activeItem || activeItem.status === "done" || activeItem.status === "error") {
    // Auto-advance to next queued item if available
    const pendingItem = state.queue.find(q => q.status === "queued");
    if (pendingItem) {
      selectQueueItem(pendingItem.id);
      activeItem = pendingItem;
    }
  }

  if (!hasLoadedImage()) {
    showToast("⚠️ Drop, paste (Ctrl+V), or select an image first!");
    if (dropzone) {
      dropzone.style.borderColor = "var(--accent)";
      setTimeout(() => { dropzone.style.borderColor = ""; }, 1200);
    }
    return;
  }

  if (!activeItem) {
    // If not in queue yet, add it
    activeItem = {
      id: "item_" + Date.now(),
      name: state.currentFileName || "Current Sticker",
      base64: state.currentImageBase64,
      path: state.currentImagePath,
      thumbnail: state.currentImageBase64 || state.currentImagePath,
      status: "processing",
      stages: null,
      metadata: null
    };
    state.queue.push(activeItem);
    state.activeItemId = activeItem.id;
  }

  // Ensure base64 is resolved if loaded via on-demand file reference
  if (!activeItem.base64 && activeItem.file) {
    await ensureItemBase64(activeItem);
    state.currentImageBase64 = activeItem.base64;
  }

  activeItem.status = "processing";
  state.isProcessing = true;
  renderQueueTray();

  statusText.textContent = "Processing...";
  statusIndicator.classList.add("busy");
  showLoadingOverlay("Processing Sticker...", "Initializing pipeline...", 6);

  // Abort any lingering request before starting new process
  if (currentAbortController) {
    try { currentAbortController.abort(); } catch (_) {}
  }
  currentAbortController = new AbortController();

  if (currentProgressInterval) {
    clearInterval(currentProgressInterval);
  }
  currentProgressInterval = startProgressAnimation();

  const deskewVal = selectDeskewMode.value;
  let deskewParam = deskewVal;
  let pcaAlign = false;
  let pcaMode = "neutral";

  if (deskewVal === "peeker") {
    deskewParam = "baseline";
    pcaAlign = false;
    pcaMode = "neutral";
  } else if (deskewVal === "upright") {
    deskewParam = "moments";
    pcaAlign = true;
    pcaMode = "bottom_upright";
  } else if (deskewVal === "quad") {
    deskewParam = "quad";
    pcaAlign = false;
    pcaMode = "neutral";
  } else if (deskewVal === "auto") {
    deskewParam = "auto";
    pcaAlign = false;
    pcaMode = "neutral";
  } else if (deskewVal === "character") {
    deskewParam = "character";
    pcaAlign = false;
    pcaMode = "neutral";
  } else {
    deskewParam = "none";
    pcaAlign = false;
    pcaMode = "neutral";
  }

  // Extract positive keep mask and negative exclusion mask from canvas if painted
  let characterHighlightMaskData = null;
  let characterNegativeMaskData = null;
  let computedBbox = state.characterBbox;

  if (state.hasHighlightStrokes && highlightCanvas && highlightCanvas.width > 0 && highlightCanvas.height > 0) {
    const hw = highlightCanvas.width;
    const hh = highlightCanvas.height;

    const posCanvas = document.createElement("canvas");
    posCanvas.width = hw;
    posCanvas.height = hh;
    const posCtx = posCanvas.getContext("2d");
    posCtx.fillStyle = "#000000";
    posCtx.fillRect(0, 0, hw, hh);
    const posMaskData = posCtx.createImageData(hw, hh);

    const negCanvas = document.createElement("canvas");
    negCanvas.width = hw;
    negCanvas.height = hh;
    const negCtx = negCanvas.getContext("2d");
    negCtx.fillStyle = "#000000";
    negCtx.fillRect(0, 0, hw, hh);
    const negMaskData = negCtx.createImageData(hw, hh);

    const strokeImgData = highlightCanvas.getContext("2d").getImageData(0, 0, hw, hh);
    let minX = hw, minY = hh, maxX = 0, maxY = 0;
    let posCount = 0;
    let negCount = 0;

    for (let i = 0; i < strokeImgData.data.length; i += 4) {
      const alpha = strokeImgData.data[i + 3];
      if (alpha > 15) {
        const r = strokeImgData.data[i];
        const b = strokeImgData.data[i + 2];
        const pixelIdx = i / 4;
        const px = pixelIdx % hw;
        const py = Math.floor(pixelIdx / hw);

        // Check if Red / Exclude stroke
        if (r > 150 && b < 130) {
          negMaskData.data[i] = 255;
          negMaskData.data[i + 1] = 255;
          negMaskData.data[i + 2] = 255;
          negMaskData.data[i + 3] = 255;
          negCount++;
        } else {
          // Cyan / Positive Keep stroke
          posMaskData.data[i] = 255;
          posMaskData.data[i + 1] = 255;
          posMaskData.data[i + 2] = 255;
          posMaskData.data[i + 3] = 255;
          posCount++;

          if (px < minX) minX = px;
          if (px > maxX) maxX = px;
          if (py < minY) minY = py;
          if (py > maxY) maxY = py;
        }
      }
    }

    if (posCount > 10) {
      posCtx.putImageData(posMaskData, 0, 0);
      characterHighlightMaskData = posCanvas.toDataURL("image/png");
      const bw = Math.max(1, maxX - minX + 1);
      const bh = Math.max(1, maxY - minY + 1);
      computedBbox = [minX, minY, bw, bh];
    }

    if (negCount > 10) {
      negCtx.putImageData(negMaskData, 0, 0);
      characterNegativeMaskData = negCanvas.toDataURL("image/png");
    }
  }

  const payload = {
    image_path: activeItem.path || state.currentImagePath,
    image_base64: activeItem.base64 || state.currentImageBase64,
    model_name: selectAiModel.value,
    extract_mode: btnExtractChar.classList.contains("active") ? "character_only" : "full_sticker",
    deskew_mode: deskewParam,
    alpha_threshold: parseInt(sliderAlphaThreshold.value),
    clean_shadow_smudges: toggleSmudgeCleaner.checked,
    edge_inset_px: 0,
    margin_inset_px: 0,
    padding_px: 12,
    defringe: true,
    ensure_connected: true,
    add_diecut_border: toggleBorder.checked,
    border_mode: selectHighlightMode ? selectHighlightMode.value : "solid",
    border_color: selectedHighlightColor || "#ffffff",
    border_thickness_px: parseInt(sliderBorderWidth.value),
    border_smoothing: sliderBorderSmoothing ? parseInt(sliderBorderSmoothing.value) : 50,
    highlight_glow_radius: sliderGlowRadius ? parseInt(sliderGlowRadius.value) : 20,
    highlight_box_padding: 16,
    enhance_resolution: toggleSuperRes.checked,
    enhance_scale: selectEnhancerScale ? parseInt(selectEnhancerScale.value) : 2,
    enhance_model: selectEnhancerModel ? selectEnhancerModel.value : "ultrasharp_lite",
    preserve_colors: togglePreserveColors ? togglePreserveColors.checked : true,
    character_bbox: computedBbox,
    character_highlight_mask: characterHighlightMaskData,
    character_negative_mask: characterNegativeMaskData,
    pca_align: pcaAlign,
    pca_align_mode: pcaMode,
    manual_rotation_deg: state.manualRotation,
    horizontal_skew_deg: state.horizontalSkew || 0,
    vertical_skew_deg: state.verticalSkew || 0,
    flip_horizontal: state.flipH,
    flip_vertical: state.flipV,
    remove_watermarks: false,
    watermark_sensitivity: 50,
    watermark_region: "full",
    clean_hair_gaps: toggleCleanHairGaps ? toggleCleanHairGaps.checked : true,
    remove_shine: toggleShineRemover ? toggleShineRemover.checked : false,
    shine_strength: sliderShineStrength ? parseInt(sliderShineStrength.value) : 60,
    color_tiers: sliderColorTiers ? parseInt(sliderColorTiers.value) : 40,
    flat_cel_look: toggleFlatCel ? toggleFlatCel.checked : false,
    color_pop_preset: toggleColorPop && toggleColorPop.checked && selectColorPopPreset ? selectColorPopPreset.value : "off",
    ai_lora_enabled: toggleColorPop && toggleColorPop.checked && toggleAiLora && toggleAiLora.checked,
    ai_lora_preset: selectAiLoraPreset ? selectAiLoraPreset.value : "lora_shinkai",
    color_pop_vibrance: toggleColorPop && toggleColorPop.checked && sliderColorPopVibrance ? parseFloat(sliderColorPopVibrance.value) : 1.0,
    color_pop_clarity: toggleColorPop && toggleColorPop.checked && sliderColorPopClarity ? parseFloat(sliderColorPopClarity.value) : 0.0
  };

  const shouldSaveLocal = toggleSaveLocal ? toggleSaveLocal.checked : true;
  const isBatchMode = state.queue.filter(q => q.status === "queued").length > 0;

  payload.save_to_local = shouldSaveLocal;
  payload.output_dir = "output_results";
  payload.output_filename = (activeItem.name || "sticker").replace(/\.[^/.]+$/, "");
  payload.lightweight_response = isBatchMode;

  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal
    });

    const data = await res.json();
    if (currentProgressInterval) {
      clearInterval(currentProgressInterval);
      currentProgressInterval = null;
    }
    currentAbortController = null;

    if (data.success || data.status === "success") {
      if (pipelineProgressBar) pipelineProgressBar.style.width = "100%";
      if (loadingPercent) loadingPercent.textContent = "100%";
      if (loadingStepDesc) loadingStepDesc.textContent = "Asset ready!";

      activeItem.status = "done";
      activeItem.metadata = data.metadata;
      if (data.saved_path) {
        activeItem.savedPath = data.saved_path;
      }

      if (isBatchMode) {
        // Memory optimization: Keep only final output in batch mode to prevent tab heap crashes
        activeItem.stages = { "final": data.stages["final"] };
        activeItem.thumbnail = data.stages["final"] || activeItem.thumbnail;
        activeItem.base64 = null; // Free uncompressed base64 data
      } else {
        activeItem.stages = data.stages;
        activeItem.thumbnail = data.stages["final"] || activeItem.thumbnail;
      }

      state.stages = activeItem.stages;
      state.metadata = data.metadata;

      // Clean up stage blobs for older completed items so memory stays under ~100MB
      state.queue.forEach(q => {
        if (q.id !== activeItem.id && q.status === "done" && q.stages) {
          if (q.stages["final"]) {
            q.stages = { "final": q.stages["final"] };
          }
          q.base64 = null;
        }
      });

      renderQueueTray();

      if (data.saved_filename && shouldSaveLocal) {
        showToast(`✓ Processed & saved to output_results/${data.saved_filename}`);
      }

      if (emptyWorkspaceView) emptyWorkspaceView.classList.add("hidden");
      if (displayWrapper) displayWrapper.classList.remove("hidden");

      setTimeout(() => {
        try {
          if (mainStickerImg) mainStickerImg.style.transform = "";
          // Reset manual angle/skews now that they are baked into the returned image
          state.manualRotation = 0;
          state.horizontalSkew = 0;
          state.verticalSkew = 0;
          state.flipH = false;
          state.flipV = false;
          if (sliderRotation) sliderRotation.value = 0;
          if (badgeRot) badgeRot.textContent = "0°";
          if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
          if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
          if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
          if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";

          updateDisplayView();
          updateMetrics(data.metadata);
          if (statusText) statusText.textContent = "Ready";
          if (statusIndicator) statusIndicator.classList.remove("busy");

          // Ensure paintbrush tool stays active and doesn't disappear on generate!
          if (state.isRoiMode) {
            setRoiMode(true);
          }
        } catch (postErr) {
          console.error("Error updating display post-process:", postErr);
        } finally {
          hideLoadingOverlay();
          state.isProcessing = false;
        }

        // If Auto-Process is checked, trigger next queued item after state.isProcessing is cleared
        if (toggleAutoProcess && toggleAutoProcess.checked) {
          setTimeout(triggerQueueProcessor, 100);
        }
      }, 50);
    } else {
      console.error("Pipeline returned failure:", data.error);
      activeItem.status = "error";
      state.isProcessing = false;
      renderQueueTray();
      statusText.textContent = data.error || "Error";
      statusIndicator.classList.remove("busy");
      hideLoadingOverlay();

      if (toggleAutoProcess && toggleAutoProcess.checked) {
        setTimeout(triggerQueueProcessor, 400);
      }
    }
  } catch (err) {
    if (currentProgressInterval) {
      clearInterval(currentProgressInterval);
      currentProgressInterval = null;
    }
    currentAbortController = null;

    if (err.name === "AbortError" || (err.message && err.message.toLowerCase().includes("abort"))) {
      console.log("Pipeline processing aborted by user.");
      hideLoadingOverlay();
      return;
    }

    console.error("Pipeline network error:", err);
    if (activeItem) {
      activeItem.status = "error";
    }
    state.isProcessing = false;
    renderQueueTray();
    statusText.textContent = "Network error";
    statusIndicator.classList.remove("busy");
    hideLoadingOverlay();

    if (toggleAutoProcess && toggleAutoProcess.checked) {
      setTimeout(triggerQueueProcessor, 400);
    }
  }
}

function updateMetrics(meta) {
  if (meta) {
    // If metadata has original/final dimensions, update aspect ratio
    const w = meta.final_size ? meta.final_size[0] : 0;
    const h = meta.final_size ? meta.final_size[1] : 0;
    if (w > 0 && h > 0) {
      exportState.originalWidth = w;
      exportState.originalHeight = h;
      exportState.aspectRatio = w / h;
      if (exportState.isAspectLocked) {
        if (exportState.unit === 'inches') {
          exportState.inchesHeight = Math.round((exportState.inchesWidth / exportState.aspectRatio) * 100) / 100;
        } else {
          exportState.targetHeight = Math.max(1, Math.round(exportState.targetWidth / exportState.aspectRatio));
        }
      }
      if (typeof updateAllExportUi === 'function') {
        updateAllExportUi();
      }
    }
  }
  if (!meta) return;
  if (statTime && meta.execution_time_total_s !== undefined) {
    statTime.textContent = `${meta.execution_time_total_s}s`;
  }
  if (statDims && meta.final_size) {
    statDims.textContent = `${meta.final_size[0]} × ${meta.final_size[1]} px`;
  }

  const modelNames = {
    "birefnet-general": "BiRefNet",
    "isnet-anime": "ISNet Anime",
    "u2net": "U2-Net",
    "digital": "Digital"
  };
  if (statModel && selectAiModel) {
    statModel.textContent = modelNames[selectAiModel.value] || selectAiModel.value;
  }
  if (statDeskew && selectDeskewMode && selectDeskewMode.selectedIndex >= 0 && selectDeskewMode.options[selectDeskewMode.selectedIndex]) {
    statDeskew.textContent = selectDeskewMode.options[selectDeskewMode.selectedIndex].text.split("(")[0].trim();
  }
  let parts = [`${(state.manualRotation || 0).toFixed(0)}°`];
  if (state.horizontalSkew) parts.push(`SkX:${state.horizontalSkew > 0 ? "+" : ""}${state.horizontalSkew.toFixed(1)}°`);
  if (state.verticalSkew) parts.push(`SkY:${state.verticalSkew > 0 ? "+" : ""}${state.verticalSkew.toFixed(1)}°`);
  if (statRot) statRot.textContent = parts.join(" ");
}

// -------------------------------------------------------------
// Bake Manual Transforms (Angle, Skew, Flip) Client-Side Instantly
// -------------------------------------------------------------
function bakeManualTransforms() {
  if (!hasLoadedImage() || !mainStickerImg || !mainStickerImg.src) {
    showToast("⚠️ No image loaded to bake transforms");
    return;
  }
  const rot = state.manualRotation || 0;
  const hSkew = state.horizontalSkew || 0;
  const vSkew = state.verticalSkew || 0;
  const flipH = state.flipH;
  const flipV = state.flipV;

  if (rot === 0 && hSkew === 0 && vSkew === 0 && !flipH && !flipV) {
    showToast("ℹ️ No active skew, rotation, or flip to bake");
    return;
  }

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;

    const rotRad = (rot * Math.PI) / 180;
    const hTan = Math.tan((hSkew * Math.PI) / 180);
    const vTan = Math.tan((vSkew * Math.PI) / 180);

    const padW = Math.ceil(Math.abs(h * hTan));
    const padH = Math.ceil(Math.abs(w * vTan));
    const skewedW = w + padW;
    const skewedH = h + padH;

    const cos = Math.abs(Math.cos(rotRad));
    const sin = Math.abs(Math.sin(rotRad));
    const finalW = Math.ceil(skewedW * cos + skewedH * sin);
    const finalH = Math.ceil(skewedW * sin + skewedH * cos);

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, finalW);
    canvas.height = Math.max(1, finalH);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotRad);
    ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    ctx.transform(1, vTan, hTan, 1, 0, 0);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();

    const bakedDataUrl = canvas.toDataURL("image/png");
    mainStickerImg.src = bakedDataUrl;
    state.stages["final"] = bakedDataUrl;
    if (state.stages["deskewed"]) state.stages["deskewed"] = bakedDataUrl;
    state.currentImageBase64 = bakedDataUrl;

    const activeItem = state.queue.find(item => item.id === state.activeItemId);
    if (activeItem) {
      activeItem.thumbnail = bakedDataUrl;
      activeItem.stages = activeItem.stages || {};
      activeItem.stages["final"] = bakedDataUrl;
    }

    state.manualRotation = 0;
    state.horizontalSkew = 0;
    state.verticalSkew = 0;
    state.flipH = false;
    state.flipV = false;

    if (sliderRotation) sliderRotation.value = 0;
    if (badgeRot) badgeRot.textContent = "0°";
    if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
    if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
    if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
    if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";

    mainStickerImg.style.transform = "";
    renderQueueTray();
    showToast("✨ Skew & angle permanently baked into image!");
  };
  img.src = state.stages["final"] || mainStickerImg.src;
}

function getActiveImageWithTransforms(callback) {
  const src = state.stages["final"] || (mainStickerImg ? mainStickerImg.src : null);
  if (!src) {
    callback(null);
    return;
  }
  const rot = state.manualRotation || 0;
  const hSkew = state.horizontalSkew || 0;
  const vSkew = state.verticalSkew || 0;
  const flipH = state.flipH;
  const flipV = state.flipV;

  if (rot === 0 && hSkew === 0 && vSkew === 0 && !flipH && !flipV) {
    callback(src);
    return;
  }

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;

    const rotRad = (rot * Math.PI) / 180;
    const hTan = Math.tan((hSkew * Math.PI) / 180);
    const vTan = Math.tan((vSkew * Math.PI) / 180);

    const padW = Math.ceil(Math.abs(h * hTan));
    const padH = Math.ceil(Math.abs(w * vTan));
    const skewedW = w + padW;
    const skewedH = h + padH;

    const cos = Math.abs(Math.cos(rotRad));
    const sin = Math.abs(Math.sin(rotRad));
    const finalW = Math.ceil(skewedW * cos + skewedH * sin);
    const finalH = Math.ceil(skewedW * sin + skewedH * cos);

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, finalW);
    canvas.height = Math.max(1, finalH);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotRad);
    ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    ctx.transform(1, vTan, hTan, 1, 0, 0);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();

    callback(canvas.toDataURL("image/png"));
  };
  img.onerror = () => callback(src);
  img.src = src;
}

// -------------------------------------------------------------
// 6-Variant Parallel Background Removal (Cutout & Mask Variations)
// -------------------------------------------------------------
const SIX_CUTOUT_PRESETS = [
  {
    name: "Anime Fine Edge (ISNet)",
    tag: "ISNet Anime",
    badge: "Hair Loops Cleaned",
    desc: "SOTA anime segmentation model with fine hair loop & limb cavity cleanup.",
    settings: {
      model_name: "isnet-anime",
      extract_mode: "character_only",
      alpha_threshold: 15,
      clean_hair_gaps: true,
      clean_shadow_smudges: true,
      smudge_sensitivity: 50,
      edge_inset_px: 0
    }
  },
  {
    name: "Ultra-Crisp (BiRefNet SOTA)",
    tag: "BiRefNet",
    badge: "Razor-Sharp Silhouette",
    desc: "Bilateral high-resolution neural network with sharp boundary details.",
    settings: {
      model_name: "birefnet-general",
      extract_mode: "character_only",
      alpha_threshold: 20,
      clean_hair_gaps: true,
      clean_shadow_smudges: true,
      smudge_sensitivity: 50,
      edge_inset_px: 0
    }
  },
  {
    name: "Heavy De-Haze (35α Cutoff)",
    tag: "Aggressive",
    badge: "35α Cutoff • 1px Inset",
    desc: "Strips faint background haze, fuzzy halo fringes, and drop shadows.",
    settings: {
      model_name: "isnet-anime",
      extract_mode: "character_only",
      alpha_threshold: 35,
      clean_shadow_smudges: true,
      smudge_sensitivity: 85,
      edge_inset_px: 1,
      clean_hair_gaps: true
    }
  },
  {
    name: "Soft Wisps & Feathers (5α)",
    tag: "Gentle",
    badge: "Translucent Gradients",
    desc: "Preserves soft semi-transparent strands, wisps, smoke, and fine gradients.",
    settings: {
      model_name: "birefnet-general",
      extract_mode: "character_only",
      alpha_threshold: 5,
      clean_shadow_smudges: false,
      smudge_sensitivity: 30,
      edge_inset_px: 0,
      clean_hair_gaps: false
    }
  },
  {
    name: "Universal Cutout (U2-Net)",
    tag: "U2-Net",
    badge: "Solid Core Contrast",
    desc: "Universal salient object detection with solid core contrast.",
    settings: {
      model_name: "u2net",
      extract_mode: "character_only",
      alpha_threshold: 25,
      clean_hair_gaps: true,
      clean_shadow_smudges: true,
      smudge_sensitivity: 50,
      edge_inset_px: 0
    }
  },
  {
    name: "Full Die-Cut Sticker Contour",
    tag: "Full Sticker",
    badge: "Outer Border Preserved",
    desc: "Retains the entire physical sticker silhouette including white vinyl margin.",
    settings: {
      model_name: "isnet-anime",
      extract_mode: "full_sticker",
      alpha_threshold: 15,
      clean_shadow_smudges: true,
      smudge_sensitivity: 50,
      edge_inset_px: 0
    }
  }
];

function openCutoutVariantsModal() {
  if (modalCutoutVariants) {
    modalCutoutVariants.classList.remove("hidden");
  }
}

function closeCutoutVariantsModal() {
  if (modalCutoutVariants) {
    modalCutoutVariants.classList.add("hidden");
  }
}

if (btnCloseCutoutModal) {
  btnCloseCutoutModal.addEventListener("click", closeCutoutVariantsModal);
}
if (modalCutoutVariants) {
  modalCutoutVariants.addEventListener("click", (e) => {
    if (e.target === modalCutoutVariants) {
      closeCutoutVariantsModal();
    }
  });
}
if (btnOpenVariantsModal) {
  btnOpenVariantsModal.addEventListener("click", openCutoutVariantsModal);
}
if (btnRerunVariants) {
  btnRerunVariants.addEventListener("click", () => runSixVariants());
}

function renderCutoutVariantsGrid() {
  if (!cutoutVariantsGrid) return;
  cutoutVariantsGrid.innerHTML = "";

  state.cutoutVariants.forEach((variant, idx) => {
    const card = document.createElement("div");
    card.className = `cutout-variant-card ${state.activeVariantIndex === idx ? "selected" : ""}`;
    card.dataset.index = idx;

    const isBusy = variant.status === "busy";
    const isError = variant.status === "error";
    const isDone = variant.status === "done";
    const thumbSrc = isDone ? (variant.stages["final"] || variant.stages["character_art"] || variant.thumbnail) : "";

    card.innerHTML = `
      <div class="variant-card-header">
        <div>
          <div class="variant-card-title">${variant.name}</div>
          <div style="font-size: 0.68rem; color: #38bdf8; font-weight: 600; margin-top: 2px;">${variant.badge}</div>
        </div>
        <span class="variant-card-tag">${variant.tag}</span>
      </div>
      <div class="variant-card-desc">${variant.desc}</div>
      <div class="variant-preview-box">
        ${isBusy ? `
          <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; color: #818cf8;">
            <div style="width: 28px; height: 28px; border: 3px solid rgba(99, 102, 241, 0.25); border-top-color: #818cf8; border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
            <span style="font-size: 0.72rem; font-weight: 600;">Segmenting...</span>
          </div>
        ` : isError ? `
          <div style="color: #f43f5e; font-size: 0.75rem; text-align: center; padding: 12px;">⚠️ Segmentation failed</div>
        ` : `
          <img src="${thumbSrc}" class="variant-preview-img" alt="${variant.name}">
        `}
      </div>
      <button type="button" class="variant-select-btn ${state.activeVariantIndex === idx ? "active" : ""}" ${!isDone ? "disabled" : ""}>
        ${state.activeVariantIndex === idx ? "✓ Selected (Active)" : "✓ Select This Cutout"}
      </button>
    `;

    const selectBtn = card.querySelector(".variant-select-btn");
    if (selectBtn && isDone) {
      selectBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectCutoutVariant(idx);
        closeCutoutVariantsModal();
      });
    }

    card.addEventListener("click", () => {
      if (isDone) {
        selectCutoutVariant(idx);
      }
    });

    cutoutVariantsGrid.appendChild(card);
  });
}

function updateInlineVariantsBar() {
  if (!inlineVariantsBar || !inlineVariantsChips) return;
  if (!state.cutoutVariants || state.cutoutVariants.length === 0) {
    inlineVariantsBar.classList.add("hidden");
    return;
  }

  inlineVariantsBar.classList.remove("hidden");
  inlineVariantsChips.innerHTML = "";

  state.cutoutVariants.forEach((v, idx) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `variant-chip ${state.activeVariantIndex === idx ? "active" : ""}`;
    chip.textContent = `${idx + 1}. ${v.tag}`;
    chip.title = `${v.name} (${v.badge})`;
    chip.addEventListener("click", () => selectCutoutVariant(idx));
    inlineVariantsChips.appendChild(chip);
  });
}

function selectCutoutVariant(idx) {
  const variant = state.cutoutVariants[idx];
  if (!variant || variant.status !== "done") return;

  state.activeVariantIndex = idx;
  state.stages = variant.stages;
  state.metadata = variant.metadata;

  if (mainStickerImg) {
    mainStickerImg.src = variant.stages["final"] || variant.stages["character_art"];
    mainStickerImg.style.transform = "";
  }
  state.manualRotation = 0;
  state.horizontalSkew = 0;
  state.verticalSkew = 0;
  state.flipH = false;
  state.flipV = false;
  if (sliderRotation) sliderRotation.value = 0;
  if (badgeRot) badgeRot.textContent = "0°";
  if (sliderHorizontalSkew) sliderHorizontalSkew.value = 0;
  if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = "0°";
  if (sliderVerticalSkew) sliderVerticalSkew.value = 0;
  if (badgeVerticalSkew) badgeVerticalSkew.textContent = "0°";

  const activeItem = state.queue.find(item => item.id === state.activeItemId);
  if (activeItem) {
    activeItem.stages = variant.stages;
    activeItem.metadata = variant.metadata;
    activeItem.thumbnail = variant.stages["final"] || activeItem.thumbnail;
    activeItem.activeVariantIndex = idx;
  }

  updateDisplayView();
  updateMetrics(variant.metadata);
  renderQueueTray();
  renderCutoutVariantsGrid();
  updateInlineVariantsBar();
  showToast(`✓ Applied cutout: ${variant.name}`);
}

async function runSixVariants() {
  const activeItem = state.queue.find(item => item.id === state.activeItemId);
  const rawBase64 = state.currentImageBase64 || (activeItem ? activeItem.base64 : null);
  const rawPath = state.currentImagePath || (activeItem ? activeItem.path : null);

  if (!rawBase64 && !rawPath) {
    showToast("⚠️ Please load an image first before generating 6 cutout variations");
    return;
  }

  // Open the comparison modal immediately
  openCutoutVariantsModal();

  // Initialize the 6 variants without touching or flooding state.queue!
  state.cutoutVariants = SIX_CUTOUT_PRESETS.map((preset, idx) => ({
    index: idx,
    name: preset.name,
    tag: preset.tag,
    badge: preset.badge,
    desc: preset.desc,
    status: "busy",
    stages: {},
    metadata: null
  }));

  state.activeVariantIndex = -1;
  renderCutoutVariantsGrid();
  updateInlineVariantsBar();
  showToast("🚀 Generating 6 cutout variations in parallel...");

  // Execute all 6 background removals in parallel
  const promises = SIX_CUTOUT_PRESETS.map(async (preset, idx) => {
    const payload = {
      image_path: rawPath,
      image_base64: rawBase64,
      ...preset.settings,
      defringe: true,
      ensure_connected: true,
      preserve_colors: true,
      color_pop_preset: "off",
      remove_shine: false,
      add_diecut_border: false,
      enhance_resolution: false,
      deskew_mode: selectDeskewMode ? selectDeskewMode.value : "auto",
      pca_align: false,
      manual_rotation_deg: 0,
      horizontal_skew_deg: 0,
      vertical_skew_deg: 0
    };

    try {
      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if ((data.status === "success" || data.success) && data.stages) {
        state.cutoutVariants[idx].status = "done";
        state.cutoutVariants[idx].stages = data.stages;
        state.cutoutVariants[idx].metadata = data.metadata;
      } else {
        state.cutoutVariants[idx].status = "error";
      }
    } catch (e) {
      console.error(`Cutout variant ${idx + 1} failed:`, e);
      state.cutoutVariants[idx].status = "error";
    }
    renderCutoutVariantsGrid();
  });

  await Promise.all(promises);

  // If active item doesn't have a finished state yet, auto-select the first successful variant
  const firstDoneIdx = state.cutoutVariants.findIndex(v => v.status === "done");
  if (firstDoneIdx !== -1 && state.activeVariantIndex === -1) {
    selectCutoutVariant(firstDoneIdx);
  }
  updateInlineVariantsBar();
  showToast("✓ All 6 cutout variations ready! Click any cutout to apply.");
}

// -------------------------------------------------------------
// In-Workspace & Modal Export Settings, Inches Support & Quick Batch Export
// -------------------------------------------------------------
const exportState = {
  isOpen: false,
  unit: "inches",          // 'inches' or 'px'
  dpi: 300,                // 300 standard print, 150 draft, 600 ultra
  inchesWidth: 3.0,        // width in inches
  inchesHeight: 3.0,       // height in inches
  originalWidth: 0,        // px
  originalHeight: 0,       // px
  aspectRatio: 1.0,        // width / height
  isAspectLocked: true,    // default true to prevent weird skewing/stretching
  targetWidth: 900,        // px
  targetHeight: 900,       // px
  fitMode: "fit",          // 'fit' (contain / pad) or 'stretch'
  format: "png"            // 'png', 'webp', 'jpeg'
};

// DOM references for Export Modal
const modalExportSize = document.getElementById("modal-export-size");
const btnCloseExportModal = document.getElementById("btn-close-export-modal");
const btnCancelExportModal = document.getElementById("btn-cancel-export-modal");
const btnDoCustomExport = document.getElementById("btn-do-custom-export");
const exportWidthInput = document.getElementById("export-width-input");
const exportHeightInput = document.getElementById("export-height-input");
const btnToggleAspectLock = document.getElementById("btn-toggle-aspect-lock");
const lockStatusText = document.getElementById("lock-status-text");
const exportSkewProtectionBadge = document.getElementById("export-skew-protection-badge");
const exportUnlockedOptions = document.getElementById("export-unlocked-options");
const exportPreviewThumb = document.getElementById("export-preview-thumb");
const exportOriginalDims = document.getElementById("export-original-dims");
const exportTargetDims = document.getElementById("export-target-dims");
const exportAspectRatioText = document.getElementById("export-aspect-ratio-text");
const btnExportText = document.getElementById("btn-export-text");
const btnModalUnitInches = document.getElementById("btn-modal-unit-inches");
const btnModalUnitPx = document.getElementById("btn-modal-unit-px");
const selectModalDpi = document.getElementById("select-modal-dpi");
const modalDpiRow = document.getElementById("modal-dpi-row");

// DOM references for Sidebar In-Workspace Export Block
const btnUnitInches = document.getElementById("btn-unit-inches");
const btnUnitPx = document.getElementById("btn-unit-px");
const selectExportDpi = document.getElementById("select-export-dpi");
const badgeCurrentDpi = document.getElementById("badge-current-dpi");
const sidebarDpiGroup = document.getElementById("sidebar-dpi-group");
const sidebarWidthInput = document.getElementById("sidebar-width-input");
const sidebarHeightInput = document.getElementById("sidebar-height-input");
const sidebarUnitLabelW = document.getElementById("sidebar-unit-label-w");
const sidebarUnitLabelH = document.getElementById("sidebar-unit-label-h");
const btnSidebarAspectLock = document.getElementById("btn-sidebar-aspect-lock");
const flightSummaryText = document.getElementById("flight-summary-text");
const flightAspectTag = document.getElementById("flight-aspect-tag");
const selectSidebarFormat = document.getElementById("select-sidebar-format");
const btnExportCurrentSidebar = document.getElementById("btn-export-current-sidebar");
const labelExportCurrentBtn = document.getElementById("label-export-current-btn");
const btnSidebarExportAll = document.getElementById("btn-sidebar-export-all");
const btnExportAllQueue = document.getElementById("btn-export-all-queue");

// Helper: Convert Inches to Pixels
function inchesToPx(inches, dpi = exportState.dpi) {
  return Math.max(16, Math.round(inches * dpi));
}

// Helper: Convert Pixels to Inches
function pxToInches(px, dpi = exportState.dpi) {
  return Math.round((px / dpi) * 100) / 100;
}

// Synchronize calculations and all UI elements (Sidebar + Modal)
function updateAllExportUi() {
  const isInch = exportState.unit === "inches";
  const dpi = exportState.dpi || 300;
  const ratio = exportState.aspectRatio || 1.0;

  // Compute px and inches
  if (isInch) {
    exportState.targetWidth = inchesToPx(exportState.inchesWidth, dpi);
    exportState.targetHeight = inchesToPx(exportState.inchesHeight, dpi);
  } else {
    exportState.inchesWidth = pxToInches(exportState.targetWidth, dpi);
    exportState.inchesHeight = pxToInches(exportState.targetHeight, dpi);
  }

  const wPx = exportState.targetWidth;
  const hPx = exportState.targetHeight;
  const wIn = exportState.inchesWidth.toFixed(isInch ? 2 : 2);
  const hIn = exportState.inchesHeight.toFixed(isInch ? 2 : 2);

  // 1. Sidebar Elements
  if (sidebarUnitLabelW) sidebarUnitLabelW.textContent = isInch ? "in" : "px";
  if (sidebarUnitLabelH) sidebarUnitLabelH.textContent = isInch ? "in" : "px";

  if (sidebarWidthInput) {
    sidebarWidthInput.step = isInch ? "0.1" : "1";
    sidebarWidthInput.value = isInch ? exportState.inchesWidth : exportState.targetWidth;
  }
  if (sidebarHeightInput) {
    sidebarHeightInput.step = isInch ? "0.1" : "1";
    sidebarHeightInput.value = isInch ? exportState.inchesHeight : exportState.targetHeight;
  }

  if (btnUnitInches) btnUnitInches.classList.toggle("active", isInch);
  if (btnUnitPx) btnUnitPx.classList.toggle("active", !isInch);
  if (sidebarDpiGroup) sidebarDpiGroup.classList.toggle("hidden", !isInch);
  if (badgeCurrentDpi) badgeCurrentDpi.textContent = `${dpi} DPI`;
  if (selectExportDpi) selectExportDpi.value = String(dpi);

  if (flightSummaryText) {
    flightSummaryText.textContent = `${wIn}" x ${hIn}" (${wPx} x ${hPx} px @ ${dpi} DPI)`;
  }
  if (flightAspectTag) {
    flightAspectTag.textContent = `${ratio.toFixed(2)}:1`;
  }

  if (btnSidebarAspectLock) {
    btnSidebarAspectLock.classList.toggle("locked", exportState.isAspectLocked);
    btnSidebarAspectLock.title = exportState.isAspectLocked
      ? "Aspect Ratio Locked: Width & height scale proportionally"
      : "Aspect Ratio Unlocked: Free width and height";
  }

  if (labelExportCurrentBtn) {
    labelExportCurrentBtn.textContent = `Export Sticker (${isInch ? wIn + '"' : wPx + 'px'} ${exportState.format.toUpperCase()})`;
  }

  if (selectSidebarFormat) {
    selectSidebarFormat.value = exportState.format;
  }

  // 2. Modal Elements
  if (exportWidthInput) {
    exportWidthInput.value = isInch ? exportState.inchesWidth : exportState.targetWidth;
  }
  if (exportHeightInput) {
    exportHeightInput.value = isInch ? exportState.inchesHeight : exportState.targetHeight;
  }
  const modalUnitLabels = document.querySelectorAll(".export-unit-label");
  modalUnitLabels.forEach(l => l.textContent = isInch ? "in" : "px");

  if (btnModalUnitInches) btnModalUnitInches.classList.toggle("active", isInch);
  if (btnModalUnitPx) btnModalUnitPx.classList.toggle("active", !isInch);
  if (modalDpiRow) modalDpiRow.classList.toggle("hidden", !isInch);
  if (selectModalDpi) selectModalDpi.value = String(dpi);

  if (exportOriginalDims) {
    const origInW = pxToInches(exportState.originalWidth || wPx, dpi);
    const origInH = pxToInches(exportState.originalHeight || hPx, dpi);
    exportOriginalDims.textContent = `${exportState.originalWidth || wPx} x ${exportState.originalHeight || hPx} px (${origInW}" x ${origInH}")`;
  }
  if (exportTargetDims) {
    exportTargetDims.textContent = `${wIn}" x ${hIn}" (${wPx} x ${hPx} px)`;
  }
  if (exportAspectRatioText) {
    exportAspectRatioText.textContent = `${ratio.toFixed(2)} : 1 (${ratio > 1 ? "Landscape" : ratio < 1 ? "Portrait" : "Square"})`;
  }

  if (btnToggleAspectLock) {
    btnToggleAspectLock.classList.toggle("locked", exportState.isAspectLocked);
    btnToggleAspectLock.classList.toggle("unlocked", !exportState.isAspectLocked);
    const iconLocked = btnToggleAspectLock.querySelector(".icon-locked");
    const iconUnlocked = btnToggleAspectLock.querySelector(".icon-unlocked");
    if (iconLocked) iconLocked.classList.toggle("hidden", !exportState.isAspectLocked);
    if (iconUnlocked) iconUnlocked.classList.toggle("hidden", exportState.isAspectLocked);
  }
  if (lockStatusText) {
    lockStatusText.textContent = exportState.isAspectLocked ? "Locked" : "Unlocked";
  }
  if (exportSkewProtectionBadge) {
    exportSkewProtectionBadge.classList.toggle("hidden", !exportState.isAspectLocked);
  }
  if (exportUnlockedOptions) {
    exportUnlockedOptions.classList.toggle("hidden", exportState.isAspectLocked);
  }
  if (btnExportText) {
    btnExportText.textContent = `Download ${wIn}" (${wPx}x${hPx}px) ${exportState.format.toUpperCase()}`;
  }

  // Sync format chips
  document.querySelectorAll(".export-format-chip").forEach(c => {
    const r = c.querySelector("input");
    if (r) {
      c.classList.toggle("active", r.value === exportState.format);
      r.checked = r.value === exportState.format;
    }
  });

  scheduleAutosave();
}

// Set active unit (inches or px)
function setExportUnit(unit) {
  if (exportState.unit === unit) return;
  exportState.unit = unit;
  showToast(`Export Unit set to ${unit === "inches" ? "Inches (with DPI scaling)" : "Pixels"}`);
  updateAllExportUi();
}

// Unit switch event listeners
if (btnUnitInches) btnUnitInches.addEventListener("click", () => setExportUnit("inches"));
if (btnUnitPx) btnUnitPx.addEventListener("click", () => setExportUnit("px"));
if (btnModalUnitInches) btnModalUnitInches.addEventListener("click", () => setExportUnit("inches"));
if (btnModalUnitPx) btnModalUnitPx.addEventListener("click", () => setExportUnit("px"));

// DPI change listeners
if (selectExportDpi) {
  selectExportDpi.addEventListener("change", (e) => {
    exportState.dpi = parseInt(e.target.value) || 300;
    updateAllExportUi();
    showToast(`Print DPI set to ${exportState.dpi} DPI`);
  });
}
if (selectModalDpi) {
  selectModalDpi.addEventListener("change", (e) => {
    exportState.dpi = parseInt(e.target.value) || 300;
    updateAllExportUi();
    showToast(`Print DPI set to ${exportState.dpi} DPI`);
  });
}

// Width & Height input handlers
function handleWidthInput(val) {
  const num = parseFloat(val);
  if (!num || num <= 0) return;
  document.querySelectorAll(".sidebar-preset-chip, .export-preset-btn").forEach(b => b.classList.remove("active"));

  if (exportState.unit === "inches") {
    exportState.inchesWidth = num;
    if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
      exportState.inchesHeight = Math.round((num / exportState.aspectRatio) * 100) / 100;
    }
  } else {
    exportState.targetWidth = Math.round(num);
    if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
      exportState.targetHeight = Math.max(1, Math.round(num / exportState.aspectRatio));
    }
  }
  updateAllExportUi();
}

function handleHeightInput(val) {
  const num = parseFloat(val);
  if (!num || num <= 0) return;
  document.querySelectorAll(".sidebar-preset-chip, .export-preset-btn").forEach(b => b.classList.remove("active"));

  if (exportState.unit === "inches") {
    exportState.inchesHeight = num;
    if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
      exportState.inchesWidth = Math.round((num * exportState.aspectRatio) * 100) / 100;
    }
  } else {
    exportState.targetHeight = Math.round(num);
    if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
      exportState.targetWidth = Math.max(1, Math.round(num * exportState.aspectRatio));
    }
  }
  updateAllExportUi();
}

if (sidebarWidthInput) sidebarWidthInput.addEventListener("input", (e) => handleWidthInput(e.target.value));
if (sidebarHeightInput) sidebarHeightInput.addEventListener("input", (e) => handleHeightInput(e.target.value));
if (exportWidthInput) exportWidthInput.addEventListener("input", (e) => handleWidthInput(e.target.value));
if (exportHeightInput) exportHeightInput.addEventListener("input", (e) => handleHeightInput(e.target.value));

// Toggle Aspect Ratio Lock
function toggleAspectLock() {
  exportState.isAspectLocked = !exportState.isAspectLocked;
  if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
    // Re-align height to current width
    if (exportState.unit === "inches") {
      exportState.inchesHeight = Math.round((exportState.inchesWidth / exportState.aspectRatio) * 100) / 100;
    } else {
      exportState.targetHeight = Math.max(1, Math.round(exportState.targetWidth / exportState.aspectRatio));
    }
    showToast("Aspect Ratio Locked: Dimensions scale proportionally");
  } else {
    showToast("Aspect Ratio Unlocked: Free width and height");
  }
  updateAllExportUi();
}

if (btnSidebarAspectLock) btnSidebarAspectLock.addEventListener("click", toggleAspectLock);
if (btnToggleAspectLock) btnToggleAspectLock.addEventListener("click", toggleAspectLock);

// Format change
if (selectSidebarFormat) {
  selectSidebarFormat.addEventListener("change", (e) => {
    exportState.format = e.target.value;
    updateAllExportUi();
  });
}

// Preset Chips in Sidebar (Inches Presets)
document.querySelectorAll(".sidebar-preset-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const inch = parseFloat(chip.dataset.inch);
    if (!inch) return;
    document.querySelectorAll(".sidebar-preset-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");

    exportState.unit = "inches";
    exportState.inchesWidth = inch;
    if (exportState.isAspectLocked && exportState.aspectRatio > 0) {
      exportState.inchesHeight = Math.round((inch / exportState.aspectRatio) * 100) / 100;
    } else {
      exportState.inchesHeight = inch;
    }
    updateAllExportUi();
    showToast(`Set sticker size to ${inch}" (${exportState.targetWidth}x${exportState.targetHeight} px @ ${exportState.dpi} DPI)`);
  });
});

// Quick scale preset buttons in Modal
document.querySelectorAll(".export-preset-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const preset = btn.dataset.preset;
    document.querySelectorAll(".export-preset-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    const origW = exportState.originalWidth || 512;
    const origH = exportState.originalHeight || 512;

    exportState.unit = "px";
    if (preset === "1x") {
      exportState.targetWidth = origW;
      exportState.targetHeight = origH;
    } else if (preset === "1.5x") {
      exportState.targetWidth = Math.round(origW * 1.5);
      exportState.targetHeight = Math.round(origH * 1.5);
    } else if (preset === "2x") {
      exportState.targetWidth = origW * 2;
      exportState.targetHeight = origH * 2;
    } else if (preset === "4x") {
      exportState.targetWidth = origW * 4;
      exportState.targetHeight = origH * 4;
    } else if (preset === "512" || preset === "1024" || preset === "2048" || preset === "4096") {
      const fixedDim = parseInt(preset);
      if (exportState.isAspectLocked) {
        if (origW >= origH) {
          exportState.targetWidth = fixedDim;
          exportState.targetHeight = Math.max(1, Math.round(fixedDim / exportState.aspectRatio));
        } else {
          exportState.targetHeight = fixedDim;
          exportState.targetWidth = Math.max(1, Math.round(fixedDim * exportState.aspectRatio));
        }
      } else {
        exportState.targetWidth = fixedDim;
        exportState.targetHeight = fixedDim;
      }
    }
    updateAllExportUi();
  });
});

// Fit mode radio buttons
document.querySelectorAll("input[name='export-fit-mode']").forEach(radio => {
  radio.addEventListener("change", (e) => {
    exportState.fitMode = e.target.value;
    document.querySelectorAll(".export-radio-btn").forEach(l => l.classList.remove("active"));
    radio.closest(".export-radio-btn").classList.add("active");
  });
});

// Format radio buttons in modal
document.querySelectorAll("input[name='export-format']").forEach(radio => {
  radio.addEventListener("change", (e) => {
    exportState.format = e.target.value;
    updateAllExportUi();
  });
});

// Modal open and close
function openExportModal() {
  getActiveImageWithTransforms((src) => {
    if (!src) {
      showToast("No image loaded to export");
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const origW = img.naturalWidth || img.width || 512;
      const origH = img.naturalHeight || img.height || 512;

      exportState.originalWidth = origW;
      exportState.originalHeight = origH;
      exportState.aspectRatio = origW / Math.max(1, origH);
      exportState.isOpen = true;

      if (exportPreviewThumb) exportPreviewThumb.src = src;
      updateAllExportUi();

      if (modalExportSize) {
        modalExportSize.classList.remove("hidden");
        modalExportSize.style.display = "flex";
      }
    };
    img.src = src;
  });
}

function closeExportModal() {
  exportState.isOpen = false;
  if (modalExportSize) {
    modalExportSize.classList.add("hidden");
    modalExportSize.style.display = "none";
  }
}

if (btnCloseExportModal) btnCloseExportModal.addEventListener("click", closeExportModal);
if (btnCancelExportModal) btnCancelExportModal.addEventListener("click", closeExportModal);

// High-quality canvas render single image helper
function renderImageToExportBlob(imgSrc, targetW, targetH, isLocked, fitMode, format, callback) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const srcW = img.naturalWidth || img.width;
    const srcH = img.naturalHeight || img.height;

    const canvas = document.createElement("canvas");
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d");

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    if (format === "jpeg") {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, targetW, targetH);
    }

    if (isLocked || fitMode === "fit") {
      const scale = Math.min(targetW / srcW, targetH / srcH);
      const drawW = Math.round(srcW * scale);
      const drawH = Math.round(srcH * scale);
      const drawX = Math.round((targetW - drawW) / 2);
      const drawY = Math.round((targetH - drawH) / 2);

      if (scale < 0.5) {
        let stepCanvas = document.createElement("canvas");
        let curW = srcW;
        let curH = srcH;
        stepCanvas.width = curW;
        stepCanvas.height = curH;
        let stepCtx = stepCanvas.getContext("2d");
        stepCtx.imageSmoothingEnabled = true;
        stepCtx.imageSmoothingQuality = "high";
        stepCtx.drawImage(img, 0, 0);

        while (curW * 0.5 > drawW && curH * 0.5 > drawH) {
          const nextW = Math.round(curW * 0.5);
          const nextH = Math.round(curH * 0.5);
          const halfCanvas = document.createElement("canvas");
          halfCanvas.width = nextW;
          halfCanvas.height = nextH;
          const halfCtx = halfCanvas.getContext("2d");
          halfCtx.imageSmoothingEnabled = true;
          halfCtx.imageSmoothingQuality = "high";
          halfCtx.drawImage(stepCanvas, 0, 0, nextW, nextH);
          stepCanvas = halfCanvas;
          curW = nextW;
          curH = nextH;
        }
        ctx.drawImage(stepCanvas, drawX, drawY, drawW, drawH);
      } else {
        ctx.drawImage(img, drawX, drawY, drawW, drawH);
      }
    } else {
      ctx.drawImage(img, 0, 0, targetW, targetH);
    }

    const mimeTypes = {
      png: "image/png",
      webp: "image/webp",
      jpeg: "image/jpeg"
    };
    const mime = mimeTypes[format] || "image/png";
    const quality = format === "png" ? undefined : 0.95;

    canvas.toBlob((blob) => {
      callback(blob);
    }, mime, quality);
  };
  img.onerror = () => callback(null);
  img.src = imgSrc;
}

// Download blob directly as file
function triggerBrowserDownload(blob, filename) {
  const blobUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = filename;
  link.href = blobUrl;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(blobUrl), 1500);
}

// Execute single custom export (Sidebar or Modal)
function executeCustomExport() {
  getActiveImageWithTransforms((src) => {
    if (!src) {
      showToast("No active image to export");
      return;
    }

    const targetW = exportState.targetWidth;
    const targetH = exportState.targetHeight;
    const isLocked = exportState.isAspectLocked;
    const fitMode = exportState.fitMode;
    const format = exportState.format || "png";

    renderImageToExportBlob(src, targetW, targetH, isLocked, fitMode, format, (blob) => {
      if (!blob) {
        showToast("Export generation failed");
        return;
      }
      const filenamePrefix = state.currentFileName ? state.currentFileName.replace(/\.[^/.]+$/, "") : "sticker";
      const ext = format === "jpeg" ? "jpg" : format;
      const sizeTag = exportState.unit === "inches"
        ? `${exportState.inchesWidth}x${exportState.inchesHeight}in`
        : `${targetW}x${targetH}px`;
      const filename = `${filenamePrefix}_${sizeTag}_${exportState.dpi}dpi.${ext}`;

      triggerBrowserDownload(blob, filename);
      closeExportModal();
      showToast(`Exported ${filename} successfully!`);
    });
  });
}

// Quick Export / Export All Finished (Unzipped)
async function exportAllFinishedItems() {
  const finishedItems = state.queue.filter(item => {
    return item.status === "done" && item.stages && item.stages["final"];
  });

  if (finishedItems.length === 0) {
    // If no queue items are marked done, check if current workspace has a completed sticker
    const currentSrc = state.stages && state.stages["final"];
    if (currentSrc) {
      showToast("No queue items found, exporting active finished sticker...");
      executeCustomExport();
      return;
    }
    showToast("No finished stickers to export. Process queue items first!");
    return;
  }

  showToast(`Exporting ${finishedItems.length} finished sticker${finishedItems.length > 1 ? "s" : ""} unzipped...`);

  const targetW = exportState.targetWidth;
  const targetH = exportState.targetHeight;
  const isLocked = exportState.isAspectLocked;
  const fitMode = exportState.fitMode;
  const format = exportState.format || "png";
  const ext = format === "jpeg" ? "jpg" : format;

  let exportedCount = 0;

  for (let i = 0; i < finishedItems.length; i++) {
    const item = finishedItems[i];
    const src = item.stages["final"];
    if (!src) continue;

    await new Promise((resolve) => {
      renderImageToExportBlob(src, targetW, targetH, isLocked, fitMode, format, (blob) => {
        if (blob) {
          const rawName = item.name ? item.name.replace(/\.[^/.]+$/, "") : `sticker_${i + 1}`;
          const sizeTag = exportState.unit === "inches"
            ? `${exportState.inchesWidth}x${exportState.inchesHeight}in`
            : `${targetW}x${targetH}px`;
          const filename = `${rawName}_${sizeTag}_${exportState.dpi}dpi.${ext}`;
          triggerBrowserDownload(blob, filename);
          exportedCount++;
        }
        // Stagger browser downloads by 350ms to ensure browser doesn't drop parallel downloads
        setTimeout(resolve, 350);
      });
    });
  }

  showToast(`Successfully exported ${exportedCount} sticker${exportedCount > 1 ? "s" : ""} to Downloads!`);
}

// -------------------------------------------------------------
// Direct Local Folder Saving & Explorer Opening
// -------------------------------------------------------------
async function openLocalOutputFolder() {
  try {
    const res = await fetch("/api/open-output-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_dir: "output_results" })
    });
    const data = await res.json();
    if (data.success) {
      showToast("📂 Opened output_results/ in File Explorer!");
    } else {
      showToast(`⚠️ Could not open folder: ${data.error || "Unknown error"}`);
    }
  } catch (err) {
    console.error("Open folder error:", err);
    showToast("⚠️ Could not open folder");
  }
}

async function saveBatchToLocalFolder() {
  const finishedItems = state.queue.filter(item => {
    return item.status === "done" && item.stages && item.stages["final"];
  });

  if (finishedItems.length === 0) {
    const currentSrc = state.stages && state.stages["final"];
    if (currentSrc) {
      showToast("💾 Saving active sticker to output_results/...");
      const filename = (state.currentFileName || "sticker").replace(/\.[^/.]+$/, "") + "_clean.png";
      try {
        const res = await fetch("/api/save-local", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: currentSrc,
            filename: filename,
            output_dir: "output_results"
          })
        });
        const data = await res.json();
        if (data.success) {
          showToast(`✓ Saved to output_results/${filename}!`);
        } else {
          showToast(`❌ Save error: ${data.error}`);
        }
      } catch (e) {
        showToast("❌ Network error saving sticker");
      }
      return;
    }
    showToast("⚠️ No finished stickers to save. Process queued items first!");
    return;
  }

  showToast(`💾 Saving ${finishedItems.length} finished sticker${finishedItems.length > 1 ? "s" : ""} to output_results/...`);

  const itemsToSave = finishedItems.map((item, idx) => {
    const rawName = item.name ? item.name.replace(/\.[^/.]+$/, "") : `sticker_${idx + 1}`;
    return {
      filename: `${rawName}_clean.png`,
      image_base64: item.stages["final"]
    };
  });

  try {
    const res = await fetch("/api/save-local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        items: itemsToSave,
        output_dir: "output_results"
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`✓ Successfully saved ${data.saved_count} sticker${data.saved_count > 1 ? "s" : ""} to output_results/!`);
    } else {
      showToast(`❌ Error saving: ${data.error}`);
    }
  } catch (err) {
    console.error("Save local error:", err);
    showToast("❌ Network error saving stickers to local folder");
  }
}

// Attach export event listeners
if (btnExportCurrentSidebar) btnExportCurrentSidebar.addEventListener("click", executeCustomExport);
if (btnSidebarExportAll) btnSidebarExportAll.addEventListener("click", exportAllFinishedItems);
if (btnExportAllQueue) btnExportAllQueue.addEventListener("click", exportAllFinishedItems);
if (btnSaveLocalQueue) btnSaveLocalQueue.addEventListener("click", saveBatchToLocalFolder);
if (btnSidebarSaveLocal) btnSidebarSaveLocal.addEventListener("click", saveBatchToLocalFolder);
if (btnOpenOutputFolder) btnOpenOutputFolder.addEventListener("click", openLocalOutputFolder);
if (btnOpenFolderSidebar) btnOpenFolderSidebar.addEventListener("click", openLocalOutputFolder);

if (btnDoCustomExport) {
  btnDoCustomExport.addEventListener("click", executeCustomExport);
}

// Top Bar Header Export button opens custom dimension export modal
btnHeaderDownload.addEventListener("click", openExportModal);
btnRunScript.addEventListener("click", runPipeline);
btnProcessMain.addEventListener("click", runPipeline);

// Copy PNG directly to clipboard
btnCopyClipboard.addEventListener("click", async () => {
  getActiveImageWithTransforms((src) => {
    if (!src) return;
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const cvs = document.createElement("canvas");
        cvs.width = img.width;
        cvs.height = img.height;
        const ctx = cvs.getContext("2d");
        ctx.drawImage(img, 0, 0);
        cvs.toBlob(async (blob) => {
          try {
            await navigator.clipboard.write([
              new ClipboardItem({ "image/png": blob })
            ]);
            showToast("✓ Copied PNG to Clipboard!");
            btnCopyClipboard.textContent = "✓ Copied!";
            setTimeout(() => {
              btnCopyClipboard.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy PNG`;
            }, 1800);
          } catch (e) {
            console.error("Clipboard copy failed:", e);
          }
        }, "image/png");
      };
      img.src = src;
    } catch (err) {
      console.error("Copy failed:", err);
    }
  });
});

// Listener for external messages (e.g. from error cards inside proxied iframes)
window.addEventListener("message", (e) => {
  if (e.data && e.data.action === "start_screen_snip") {
    startNativeScreenSnip();
  }
});

if (btnCancelProcess) {
  btnCancelProcess.addEventListener("click", cancelCurrentProcess);
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (modalExportSize && !modalExportSize.classList.contains("hidden")) {
      closeExportModal();
      return;
    }
    if (modalCutoutVariants && !modalCutoutVariants.classList.contains("hidden")) {
      closeCutoutVariantsModal();
      return;
    }
    if (browserState.isOpen) {
      closeBrowserModal();
      return;
    }
    if (state.isProcessing || !loadingOverlay.classList.contains("hidden")) {
      cancelCurrentProcess();
    }
  }
});

// -------------------------------------------------------------
// In-Tab Web Browser & Character Snipping Tool
// -------------------------------------------------------------
const browserState = {
  isOpen: false,
  isFullscreen: false,
  isSnipMode: false,
  isDrawing: false,
  startX: 0,
  startY: 0,
  currentRect: null,
  capturedImage: null,
  originalWidth: 0,
  originalHeight: 0
};

function openBrowserModal(initialUrl) {
  browserState.isOpen = true;
  webBrowserModal.classList.remove("hidden");
  const targetUrl = initialUrl || (browserUrlInput ? browserUrlInput.value : "") || "https://safebooru.org";
  if (browserUrlInput) {
    browserUrlInput.value = targetUrl;
  }
  loadBrowserUrl(targetUrl);
}

function closeBrowserModal() {
  browserState.isOpen = false;
  webBrowserModal.classList.add("hidden");
  exitSnippingMode();
  if (browserState.isFullscreen) {
    toggleBrowserFullscreen(false);
  }
}

function toggleBrowserFullscreen(force) {
  browserState.isFullscreen = typeof force === "boolean" ? force : !browserState.isFullscreen;
  if (browserState.isFullscreen) {
    webBrowserWindow.classList.add("fullscreen");
    btnBrowserFullscreen.textContent = "🗗";
  } else {
    webBrowserWindow.classList.remove("fullscreen");
    btnBrowserFullscreen.textContent = "⛶";
  }
  if (browserState.isSnipMode) {
    syncSnipCanvasSize();
    renderSnipOverlay();
  }
}

function loadBrowserUrl(url) {
  if (!url) return;
  url = url.trim();
  if (!url.startsWith("http://") && !url.startsWith("https://") && !url.startsWith("data:")) {
    if (url.includes(".") && !url.includes(" ")) {
      url = "https://" + url;
    } else {
      url = "https://images.google.com/search?tbm=isch&q=" + encodeURIComponent(url);
    }
  }
  if (browserUrlInput) {
    browserUrlInput.value = url;
  }

  // Check if it is a direct image link or data URL
  const isDirectImage = url.startsWith("data:image/") ||
    /\.(png|jpg|jpeg|webp|gif|svg)(\?.*)?$/i.test(url) ||
    url.includes("images.unsplash.com") ||
    url.includes("picsum.photos");

  if (isDirectImage) {
    browserIframe.classList.add("hidden");
    browserImageViewer.classList.remove("hidden");

    const proxySrc = url.startsWith("data:") ? url : `/api/web-proxy?url=${encodeURIComponent(url)}`;
    browserTargetImage.src = proxySrc;

    browserTargetImage.onload = () => {
      browserState.capturedImage = browserTargetImage;
      browserState.originalWidth = browserTargetImage.naturalWidth;
      browserState.originalHeight = browserTargetImage.naturalHeight;
    };
  } else {
    browserImageViewer.classList.add("hidden");
    browserIframe.classList.remove("hidden");
    browserIframe.src = `/api/web-proxy?url=${encodeURIComponent(url)}`;
  }
}

// -------------------------------------------------------------
// Load Pasted Screenshot Directly Into Browser Modal
// -------------------------------------------------------------
function loadPastedImageIntoBrowserModal(dataUrl) {
  if (!dataUrl) return;
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    browserIframe.classList.add("hidden");
    browserImageViewer.classList.remove("hidden");

    browserState.capturedImage = img;
    browserState.originalWidth = img.naturalWidth || 800;
    browserState.originalHeight = img.naturalHeight || 600;

    browserTargetImage.crossOrigin = "anonymous";
    browserTargetImage.src = dataUrl;

    showToast("📋 Pasted screenshot loaded! Drag a box over your character to snip.");
    enterSnippingMode();
  };
  img.onerror = () => {
    showToast("⚠️ Failed to load pasted image");
  };
  img.src = dataUrl;
}

// 📸 Native Screen Capture Snipping
async function startNativeScreenSnip() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    showToast("⚠️ Screen capture not supported in this browser. Use Ctrl+V or Win+Shift+S to paste screenshots!");
    return;
  }

  let stream = null;
  try {
    showToast("📸 Select the window, tab, or screen to snip...");
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { cursor: "always" },
      audio: false
    });

    if (!stream) return;

    const track = stream.getVideoTracks()[0];
    if (!track) {
      throw new Error("No video track found in screen stream");
    }

    const video = document.createElement("video");
    video.muted = true;
    video.volume = 0;
    video.playsInline = true;
    video.autoplay = true;
    video.srcObject = stream;

    // Await metadata and video dimensions
    await new Promise((resolve, reject) => {
      let done = false;
      const onReady = () => {
        if (!done && video.videoWidth > 0 && video.videoHeight > 0) {
          done = true;
          resolve();
        }
      };

      video.onloadedmetadata = onReady;
      video.onloadeddata = onReady;
      video.oncanplay = onReady;

      const timer = setInterval(() => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          clearInterval(timer);
          onReady();
        }
      }, 40);

      setTimeout(() => {
        clearInterval(timer);
        if (!done) {
          if (video.videoWidth > 0) resolve();
          else reject(new Error("Timeout waiting for screen stream frame"));
        }
      }, 3500);

      video.play().catch(e => console.warn("Video play notice:", e));
    });

    // Wait for frame to be decoded
    if ("requestVideoFrameCallback" in video) {
      await new Promise(r => video.requestVideoFrameCallback(r));
    } else {
      await new Promise(r => setTimeout(r, 120));
    }

    const vw = video.videoWidth || 1920;
    const vh = video.videoHeight || 1080;

    const canvas = document.createElement("canvas");
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, vw, vh);

    // Stop stream tracks immediately
    stream.getTracks().forEach(t => t.stop());
    stream = null;

    const dataUrl = canvas.toDataURL("image/png");
    if (!dataUrl || dataUrl.length < 500) {
      throw new Error("Captured screenshot frame was empty");
    }

    // Ensure browser modal is open so the user can snip immediately
    if (!browserState.isOpen) {
      browserState.isOpen = true;
      webBrowserModal.classList.remove("hidden");
    }

    // Switch to image viewer with the captured screen
    browserIframe.classList.add("hidden");
    browserImageViewer.classList.remove("hidden");

    browserState.capturedImage = canvas;
    browserState.originalWidth = vw;
    browserState.originalHeight = vh;

    browserTargetImage.crossOrigin = "anonymous";
    browserTargetImage.src = dataUrl;

    showToast("📸 Screenshot captured! Now drag a box over your character to snip.");
    enterSnippingMode();
  } catch (err) {
    if (stream) {
      try { stream.getTracks().forEach(t => t.stop()); } catch (_) {}
    }
    if (err.name === "NotAllowedError" || (err.message && err.message.toLowerCase().includes("permission"))) {
      showToast("ℹ️ Screen capture cancelled.");
    } else {
      console.error("Screen capture error:", err);
      showToast(`⚠️ Screen capture notice: ${err.message || err}`);
    }
  }
}

// ✂️ Snipping Overlay & Marquee System
function enterSnippingMode() {
  browserState.isSnipMode = true;
  browserState.currentRect = null;
  browserSnipCanvas.classList.remove("hidden");
  snipHintBanner.classList.remove("hidden");
  snipConfirmFloating.classList.add("hidden");

  syncSnipCanvasSize();
  renderSnipOverlay();
}

function exitSnippingMode() {
  browserState.isSnipMode = false;
  browserState.isDrawing = false;
  browserState.currentRect = null;
  browserSnipCanvas.classList.add("hidden");
  snipHintBanner.classList.add("hidden");
  snipConfirmFloating.classList.add("hidden");
}

function syncSnipCanvasSize() {
  const rect = browserViewportWrapper.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  browserSnipCanvas.width = rect.width * dpr;
  browserSnipCanvas.height = rect.height * dpr;
  browserSnipCanvas.style.width = `${rect.width}px`;
  browserSnipCanvas.style.height = `${rect.height}px`;

  const ctx = browserSnipCanvas.getContext("2d");
  ctx.scale(dpr, dpr);
}

function renderSnipOverlay() {
  if (!browserState.isSnipMode) return;
  const ctx = browserSnipCanvas.getContext("2d");
  const rect = browserViewportWrapper.getBoundingClientRect();

  ctx.clearRect(0, 0, rect.width, rect.height);

  // Dark semi-transparent veil
  ctx.fillStyle = "rgba(5, 8, 16, 0.65)";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (browserState.currentRect && browserState.currentRect.w > 4 && browserState.currentRect.h > 4) {
    const { x, y, w, h } = browserState.currentRect;

    // Clear cutout hole for selected character
    ctx.clearRect(x, y, w, h);

    // Glowing cyan/blue border
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#38bdf8";
    ctx.shadowColor = "#0284c7";
    ctx.shadowBlur = 8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;

    // Corner crosshairs / grips
    const gripLen = 8;
    ctx.fillStyle = "#ec4899";
    ctx.fillRect(x - 2, y - 2, gripLen, 3);
    ctx.fillRect(x - 2, y - 2, 3, gripLen);

    ctx.fillRect(x + w - gripLen + 2, y - 2, gripLen, 3);
    ctx.fillRect(x + w - 1, y - 2, 3, gripLen);

    ctx.fillRect(x - 2, y + h - 1, gripLen, 3);
    ctx.fillRect(x - 2, y + h - gripLen + 2, 3, gripLen);

    ctx.fillRect(x + w - gripLen + 2, y + h - 1, gripLen, 3);
    ctx.fillRect(x + w - 1, y + h - gripLen + 2, 3, gripLen);
  }
}

// Mouse interaction on the snipping canvas
if (browserSnipCanvas) {
  browserSnipCanvas.addEventListener("mousedown", (e) => {
    if (!browserState.isSnipMode) return;
    const rect = browserViewportWrapper.getBoundingClientRect();
    browserState.isDrawing = true;
    browserState.startX = e.clientX - rect.left;
    browserState.startY = e.clientY - rect.top;
    browserState.currentRect = { x: browserState.startX, y: browserState.startY, w: 0, h: 0 };
    snipConfirmFloating.classList.add("hidden");
  });

  window.addEventListener("mousemove", (e) => {
    if (!browserState.isSnipMode || !browserState.isDrawing) return;
    const rect = browserViewportWrapper.getBoundingClientRect();
    const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    const rx = Math.min(browserState.startX, curX);
    const ry = Math.min(browserState.startY, curY);
    const rw = Math.abs(curX - browserState.startX);
    const rh = Math.abs(curY - browserState.startY);

    browserState.currentRect = { x: rx, y: ry, w: rw, h: rh };
    renderSnipOverlay();
  });

  window.addEventListener("mouseup", (e) => {
    if (!browserState.isSnipMode || !browserState.isDrawing) return;
    browserState.isDrawing = false;

    if (browserState.currentRect && browserState.currentRect.w > 15 && browserState.currentRect.h > 15) {
      const r = browserState.currentRect;
      snipDimsBadge.textContent = `${Math.round(r.w)} × ${Math.round(r.h)} px`;

      const vRect = browserViewportWrapper.getBoundingClientRect();
      const pillX = Math.max(10, Math.min(vRect.width - 320, r.x + r.w / 2 - 150));
      const pillY = Math.min(vRect.height - 60, r.y + r.h + 12);

      snipConfirmFloating.style.left = `${pillX}px`;
      snipConfirmFloating.style.top = `${pillY}px`;
      snipConfirmFloating.classList.remove("hidden");
    } else {
      snipConfirmFloating.classList.add("hidden");
    }
  });
}

// Calculate rendered bounds inside object-fit: contain
function getContainedImageGeometry(imgEl, containerRect) {
  const nw = browserState.originalWidth || imgEl.naturalWidth || imgEl.width || 800;
  const nh = browserState.originalHeight || imgEl.naturalHeight || imgEl.height || 600;

  const containerAspect = containerRect.width / containerRect.height;
  const imgAspect = nw / nh;

  let renderW, renderH, renderX, renderY;
  if (imgAspect > containerAspect) {
    renderW = containerRect.width;
    renderH = containerRect.width / imgAspect;
    renderX = 0;
    renderY = (containerRect.height - renderH) / 2;
  } else {
    renderH = containerRect.height;
    renderW = containerRect.height * imgAspect;
    renderX = (containerRect.width - renderW) / 2;
    renderY = 0;
  }
  return {
    renderX,
    renderY,
    renderW,
    renderH,
    naturalWidth: nw,
    naturalHeight: nh
  };
}

// Confirm Snipping & Send to Studio Queue
if (btnConfirmSnip) {
  btnConfirmSnip.addEventListener("click", () => {
    if (!browserState.currentRect || browserState.currentRect.w < 10 || browserState.currentRect.h < 10) {
      showToast("⚠️ Please drag a box over your character first");
      return;
    }
    const r = browserState.currentRect;

    // Case 1: Image or captured canvas is loaded in image viewer
    if (browserState.capturedImage) {
      try {
        const vRect = browserViewportWrapper.getBoundingClientRect();
        const geom = getContainedImageGeometry(browserTargetImage, vRect);

        const relX = r.x - geom.renderX;
        const relY = r.y - geom.renderY;

        const scaleX = geom.naturalWidth / geom.renderW;
        const scaleY = geom.naturalHeight / geom.renderH;

        const natX = Math.max(0, Math.min(geom.naturalWidth, relX * scaleX));
        const natY = Math.max(0, Math.min(geom.naturalHeight, relY * scaleY));
        const natW = Math.max(10, Math.min(geom.naturalWidth - natX, r.w * scaleX));
        const natH = Math.max(10, Math.min(geom.naturalHeight - natY, r.h * scaleY));

        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = Math.round(natW);
        cropCanvas.height = Math.round(natH);
        const cropCtx = cropCanvas.getContext("2d");

        cropCtx.drawImage(
          browserState.capturedImage,
          natX, natY, natW, natH,
          0, 0, cropCanvas.width, cropCanvas.height
        );

        const croppedBase64 = cropCanvas.toDataURL("image/png");
        if (!croppedBase64 || croppedBase64 === "data:," || croppedBase64.length < 100) {
          throw new Error("Cropped screenshot export was empty");
        }

        addToQueue({
          base64: croppedBase64,
          name: `Web_Snip_${Date.now().toString().slice(-4)}`,
          source: "snip"
        });

        closeBrowserModal();
        showToast("✂️ Character snipped & imported into Studio!");
        return;
      } catch (cropErr) {
        console.error("Cropping error:", cropErr);
        showToast("⚠️ Could not crop screenshot: " + (cropErr.message || cropErr));
      }
    }

    // Case 2: Webpage in iframe without a screen frame yet - automatically capture frame!
    showToast("📸 Capturing screenshot frame to snip...");
    startNativeScreenSnip();
  });
}

// Import entire loaded image or captured frame into Studio
if (btnBrowserImportFull) {
  btnBrowserImportFull.addEventListener("click", () => {
    try {
      if (browserState.capturedImage) {
        let dataUrl = "";
        if (browserState.capturedImage instanceof HTMLCanvasElement) {
          dataUrl = browserState.capturedImage.toDataURL("image/png");
        } else if (browserState.capturedImage.src) {
          dataUrl = browserState.capturedImage.src;
        }
        if (dataUrl && dataUrl.startsWith("data:")) {
          addToQueue({
            base64: dataUrl,
            name: `Screen_${Date.now().toString().slice(-4)}`,
            source: "screenshot"
          });
          closeBrowserModal();
          showToast("📥 Screenshot imported into Studio!");
          return;
        }
      } else if (browserTargetImage && browserTargetImage.src && !browserTargetImage.src.includes("about:blank")) {
        const src = browserTargetImage.src;
        if (src.startsWith("data:")) {
          addToQueue({
            base64: src,
            name: `Web_Full_${Date.now().toString().slice(-4)}`,
            source: "web"
          });
          closeBrowserModal();
          showToast("📥 Image imported into Studio!");
          return;
        }
      }
    } catch (err) {
      console.warn("Import full error:", err);
    }
    // If viewing webpage in iframe: capture screen frame!
    startNativeScreenSnip();
  });
}

// Wire up all Browser & Snipper buttons
if (btnOpenBrowser) {
  btnOpenBrowser.addEventListener("click", () => openBrowserModal());
}
if (btnBrowserShortcut) {
  btnBrowserShortcut.addEventListener("click", () => openBrowserModal());
}
if (btnHeaderSnip) {
  btnHeaderSnip.addEventListener("click", () => startNativeScreenSnip());
}
if (btnEmptySnip) {
  btnEmptySnip.addEventListener("click", () => startNativeScreenSnip());
}
if (btnBrowserClose) {
  btnBrowserClose.addEventListener("click", () => closeBrowserModal());
}
if (btnBrowserFullscreen) {
  btnBrowserFullscreen.addEventListener("click", () => toggleBrowserFullscreen());
}
if (btnBrowserGo) {
  btnBrowserGo.addEventListener("click", () => loadBrowserUrl(browserUrlInput.value));
}
if (browserUrlInput) {
  browserUrlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadBrowserUrl(browserUrlInput.value);
  });
}
if (btnBrowserBack) {
  btnBrowserBack.addEventListener("click", () => {
    try { browserIframe.contentWindow.history.back(); } catch (e) {}
  });
}
if (btnBrowserForward) {
  btnBrowserForward.addEventListener("click", () => {
    try { browserIframe.contentWindow.history.forward(); } catch (e) {}
  });
}
if (btnBrowserRefresh) {
  btnBrowserRefresh.addEventListener("click", () => loadBrowserUrl(browserUrlInput.value));
}
if (btnBrowserSnipScreen) {
  btnBrowserSnipScreen.addEventListener("click", startNativeScreenSnip);
}
if (btnBrowserPasteSnip) {
  btnBrowserPasteSnip.addEventListener("click", async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.read) {
        const clipboardItems = await navigator.clipboard.read();
        for (const item of clipboardItems) {
          const imageType = item.types.find(t => t.startsWith("image/"));
          if (imageType) {
            const blob = await item.getType(imageType);
            const reader = new FileReader();
            reader.onload = (e) => loadPastedImageIntoBrowserModal(e.target.result);
            reader.readAsDataURL(blob);
            return;
          }
        }
      }
      showToast("💡 Press Ctrl+V right now to paste your screenshot!");
    } catch (e) {
      showToast("💡 Press Ctrl+V right now to paste your screenshot!");
    }
  });
}
if (btnBrowserStartSnip) {
  btnBrowserStartSnip.addEventListener("click", () => {
    if (browserState.isSnipMode) {
      exitSnippingMode();
      return;
    }
    // If no captured frame exists yet, capture screen/window first!
    if (!browserState.capturedImage) {
      startNativeScreenSnip();
    } else {
      enterSnippingMode();
    }
  });
}
if (btnCancelSnip) {
  btnCancelSnip.addEventListener("click", () => {
    browserState.currentRect = null;
    snipConfirmFloating.classList.add("hidden");
    renderSnipOverlay();
  });
}
if (btnExitSnipMode) {
  btnExitSnipMode.addEventListener("click", exitSnippingMode);
}
bookmarkChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const url = chip.getAttribute("data-url");
    if (url) {
      if (browserUrlInput) browserUrlInput.value = url;
      loadBrowserUrl(url);
    }
  });
});

// -------------------------------------------------------------
// Application Settings Autosave & Restore (localStorage)
// Automatically remembers all user choices when leaving / reloading the app!
// -------------------------------------------------------------
const SETTINGS_STORAGE_KEY = "sticker_deskew_studio_settings_v1";

function saveStudioSettings() {
  try {
    const settings = {
      // Model & Segmentation
      aiModel: selectAiModel ? selectAiModel.value : "isnet-anime",
      cleanHairGaps: toggleCleanHairGaps ? toggleCleanHairGaps.checked : true,
      cleanSmudges: toggleSmudgeCleaner ? toggleSmudgeCleaner.checked : false,
      smudgeSensitivity: sliderSmudgeSensitivity ? sliderSmudgeSensitivity.value : "50",
      alphaThreshold: sliderAlphaThreshold ? sliderAlphaThreshold.value : "15",
      deskewMode: selectDeskewMode ? selectDeskewMode.value : "auto",

      // Shine & Cel Restorer
      removeShine: toggleShineRemover ? toggleShineRemover.checked : false,
      shineStrength: sliderShineStrength ? sliderShineStrength.value : "60",
      colorTiers: sliderColorTiers ? sliderColorTiers.value : "40",
      flatCel: toggleFlatCel ? toggleFlatCel.checked : false,

      // Color Pop & AI LoRA
      colorPop: toggleColorPop ? toggleColorPop.checked : false,
      colorPopPreset: selectColorPopPreset ? selectColorPopPreset.value : "anime_pop",
      colorPopVibrance: sliderColorPopVibrance ? sliderColorPopVibrance.value : "1.15",
      colorPopClarity: sliderColorPopClarity ? sliderColorPopClarity.value : "0.05",
      aiLora: toggleAiLora ? toggleAiLora.checked : false,
      aiLoraPreset: selectAiLoraPreset ? selectAiLoraPreset.value : "lora_shinkai",

      // Finishing Effects & Borders
      border: toggleBorder ? toggleBorder.checked : false,
      highlightMode: selectHighlightMode ? selectHighlightMode.value : "diecut",
      highlightColor: selectedHighlightColor || "#ffffff",
      borderWidth: sliderBorderWidth ? sliderBorderWidth.value : "14",
      borderSmoothing: sliderBorderSmoothing ? sliderBorderSmoothing.value : "50",
      glowRadius: sliderGlowRadius ? sliderGlowRadius.value : "20",
      superRes: toggleSuperRes ? toggleSuperRes.checked : false,
      enhancerModel: selectEnhancerModel ? selectEnhancerModel.value : "ultrasharp_lite",
      enhancerScale: selectEnhancerScale ? selectEnhancerScale.value : "2",
      preserveColors: togglePreserveColors ? togglePreserveColors.checked : true,

      // Queue & Tools
      autoProcess: toggleAutoProcess ? toggleAutoProcess.checked : false,
      saveLocal: toggleSaveLocal ? toggleSaveLocal.checked : true,
      inpaintModel: selectInpaintModel ? selectInpaintModel.value : "anime",
      inpaintBrushSize: sliderInpaintBrush ? sliderInpaintBrush.value : "28",
      highlightBrushSize: sliderHighlightBrush ? sliderHighlightBrush.value : "32",

      // Current Transforms
      rotation: state.manualRotation || 0,
      horizontalSkew: state.horizontalSkew || 0,
      verticalSkew: state.verticalSkew || 0,
      flipH: !!state.flipH,
      flipV: !!state.flipV,

      // Export settings
      exportUnit: exportState.unit || "inches",
      exportDpi: exportState.dpi || 300,
      exportInchesWidth: exportState.inchesWidth || 3.0,
      exportInchesHeight: exportState.inchesHeight || 3.0,
      exportTargetWidth: exportState.targetWidth || 900,
      exportTargetHeight: exportState.targetHeight || 900,
      exportAspectLocked: exportState.isAspectLocked !== undefined ? exportState.isAspectLocked : true,
      exportFormat: exportState.format || "png"
    };

    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch (err) {
    console.warn("[Studio Settings] Failed to save settings to localStorage:", err);
  }
}

let autosaveDebounceTimer = null;
function scheduleAutosave() {
  clearTimeout(autosaveDebounceTimer);
  autosaveDebounceTimer = setTimeout(saveStudioSettings, 150);
}

function restoreStudioSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (!s || typeof s !== "object") return;

    // 1. Model & Segmentation
    if (selectAiModel && s.aiModel !== undefined) selectAiModel.value = s.aiModel;
    if (toggleCleanHairGaps && s.cleanHairGaps !== undefined) toggleCleanHairGaps.checked = !!s.cleanHairGaps;
    if (toggleSmudgeCleaner && s.cleanSmudges !== undefined) {
      toggleSmudgeCleaner.checked = !!s.cleanSmudges;
      if (smudgeSensitivityContainer) smudgeSensitivityContainer.classList.toggle("hidden", !toggleSmudgeCleaner.checked);
    }
    if (sliderSmudgeSensitivity && s.smudgeSensitivity !== undefined) {
      sliderSmudgeSensitivity.value = s.smudgeSensitivity;
      if (badgeSmudge) badgeSmudge.textContent = `${s.smudgeSensitivity}%`;
    }
    if (sliderAlphaThreshold && s.alphaThreshold !== undefined) {
      sliderAlphaThreshold.value = s.alphaThreshold;
      if (badgeAlpha) badgeAlpha.textContent = `${s.alphaThreshold} / 255`;
    }
    if (selectDeskewMode && s.deskewMode !== undefined) selectDeskewMode.value = s.deskewMode;

    // 2. Shine & Cel Restorer
    if (toggleShineRemover && s.removeShine !== undefined) {
      toggleShineRemover.checked = !!s.removeShine;
      if (shineOptionsContainer) shineOptionsContainer.classList.toggle("hidden", !toggleShineRemover.checked);
    }
    if (sliderShineStrength && s.shineStrength !== undefined) {
      sliderShineStrength.value = s.shineStrength;
      if (badgeShine) badgeShine.textContent = `${s.shineStrength}%`;
    }
    if (sliderColorTiers && s.colorTiers !== undefined) {
      sliderColorTiers.value = s.colorTiers;
      if (badgeColorTiers) badgeColorTiers.textContent = `${s.colorTiers} Bands`;
    }
    if (toggleFlatCel && s.flatCel !== undefined) toggleFlatCel.checked = !!s.flatCel;

    // 3. Color Pop & AI LoRA
    if (toggleColorPop && s.colorPop !== undefined) {
      toggleColorPop.checked = !!s.colorPop;
      if (colorPopOptionsContainer) colorPopOptionsContainer.classList.toggle("hidden", !toggleColorPop.checked);
    }
    if (selectColorPopPreset && s.colorPopPreset !== undefined) selectColorPopPreset.value = s.colorPopPreset;
    if (sliderColorPopVibrance && s.colorPopVibrance !== undefined) {
      sliderColorPopVibrance.value = s.colorPopVibrance;
      if (badgeColorPopVibrance) badgeColorPopVibrance.textContent = `${parseFloat(s.colorPopVibrance).toFixed(2)}x`;
    }
    if (sliderColorPopClarity && s.colorPopClarity !== undefined) {
      sliderColorPopClarity.value = s.colorPopClarity;
      if (badgeColorPopClarity) badgeColorPopClarity.textContent = `${parseFloat(s.colorPopClarity).toFixed(2)}`;
    }
    if (toggleAiLora && s.aiLora !== undefined) {
      toggleAiLora.checked = !!s.aiLora;
      if (aiLoraOptionsContainer) aiLoraOptionsContainer.classList.toggle("hidden", !toggleAiLora.checked);
    }
    if (selectAiLoraPreset && s.aiLoraPreset !== undefined) selectAiLoraPreset.value = s.aiLoraPreset;

    // 4. Finishing Effects & Borders
    if (toggleBorder && s.border !== undefined) {
      toggleBorder.checked = !!s.border;
      if (borderSliderContainer) borderSliderContainer.classList.toggle("hidden", !toggleBorder.checked);
    }
    if (selectHighlightMode && s.highlightMode !== undefined) selectHighlightMode.value = s.highlightMode;
    if (s.highlightColor !== undefined) {
      selectedHighlightColor = s.highlightColor;
      if (pickerHighlightColor) pickerHighlightColor.value = s.highlightColor;
      swatchBtns.forEach(btn => {
        const c = btn.getAttribute("data-color");
        btn.classList.toggle("active", c && c.toLowerCase() === s.highlightColor.toLowerCase());
      });
    }
    if (sliderBorderWidth && s.borderWidth !== undefined) {
      sliderBorderWidth.value = s.borderWidth;
      if (badgeBorder) badgeBorder.textContent = `${s.borderWidth}px`;
    }
    if (sliderBorderSmoothing && s.borderSmoothing !== undefined) {
      sliderBorderSmoothing.value = s.borderSmoothing;
      const v = parseInt(s.borderSmoothing);
      if (badgeBorderSmoothing) {
        if (v === 0) badgeBorderSmoothing.textContent = "0% (Sharp)";
        else if (v === 50) badgeBorderSmoothing.textContent = "50% (Smooth)";
        else if (v === 100) badgeBorderSmoothing.textContent = "100% (Ultra)";
        else badgeBorderSmoothing.textContent = `${v}%`;
      }
    }
    if (sliderGlowRadius && s.glowRadius !== undefined) {
      sliderGlowRadius.value = s.glowRadius;
      if (badgeGlowRadius) badgeGlowRadius.textContent = `${s.glowRadius}px`;
    }
    if (toggleSuperRes && s.superRes !== undefined) {
      toggleSuperRes.checked = !!s.superRes;
      if (superResOptionsContainer) {
        superResOptionsContainer.classList.toggle("sr-disabled", !toggleSuperRes.checked);
        superResOptionsContainer.classList.remove("hidden");
      }
    }
    if (selectEnhancerModel && s.enhancerModel !== undefined) selectEnhancerModel.value = s.enhancerModel;
    if (selectEnhancerScale && s.enhancerScale !== undefined) selectEnhancerScale.value = s.enhancerScale;
    if (togglePreserveColors && s.preserveColors !== undefined) togglePreserveColors.checked = !!s.preserveColors;

    // 5. Queue & Tools
    if (toggleAutoProcess && s.autoProcess !== undefined) toggleAutoProcess.checked = !!s.autoProcess;
    if (toggleSaveLocal && s.saveLocal !== undefined) toggleSaveLocal.checked = !!s.saveLocal;
    if (selectInpaintModel && s.inpaintModel !== undefined) selectInpaintModel.value = s.inpaintModel;
    if (sliderInpaintBrush && s.inpaintBrushSize !== undefined) {
      sliderInpaintBrush.value = s.inpaintBrushSize;
      state.inpaintBrushSize = parseInt(s.inpaintBrushSize);
      if (badgeInpaintSize) badgeInpaintSize.textContent = `${s.inpaintBrushSize}px`;
    }
    if (sliderHighlightBrush && s.highlightBrushSize !== undefined) {
      sliderHighlightBrush.value = s.highlightBrushSize;
      state.highlightBrushSize = parseInt(s.highlightBrushSize);
      if (badgeHighlightBrush) badgeHighlightBrush.textContent = `${s.highlightBrushSize}px`;
    }

    // 6. Transforms
    if (s.rotation !== undefined) {
      state.manualRotation = parseFloat(s.rotation) || 0;
      if (sliderRotation) sliderRotation.value = state.manualRotation;
      if (badgeRot) badgeRot.textContent = `${state.manualRotation}°`;
    }
    if (s.horizontalSkew !== undefined) {
      state.horizontalSkew = parseFloat(s.horizontalSkew) || 0;
      if (sliderHorizontalSkew) sliderHorizontalSkew.value = state.horizontalSkew;
      if (badgeHorizontalSkew) badgeHorizontalSkew.textContent = `${state.horizontalSkew}°`;
    }
    if (s.verticalSkew !== undefined) {
      state.verticalSkew = parseFloat(s.verticalSkew) || 0;
      if (sliderVerticalSkew) sliderVerticalSkew.value = state.verticalSkew;
      if (badgeVerticalSkew) badgeVerticalSkew.textContent = `${state.verticalSkew}°`;
    }
    if (s.flipH !== undefined) state.flipH = !!s.flipH;
    if (s.flipV !== undefined) state.flipV = !!s.flipV;

    // 7. Export Settings
    if (s.exportUnit !== undefined) exportState.unit = s.exportUnit;
    if (s.exportDpi !== undefined) exportState.dpi = parseInt(s.exportDpi) || 300;
    if (s.exportInchesWidth !== undefined) exportState.inchesWidth = parseFloat(s.exportInchesWidth) || 3.0;
    if (s.exportInchesHeight !== undefined) exportState.inchesHeight = parseFloat(s.exportInchesHeight) || 3.0;
    if (s.exportTargetWidth !== undefined) exportState.targetWidth = parseInt(s.exportTargetWidth) || 900;
    if (s.exportTargetHeight !== undefined) exportState.targetHeight = parseInt(s.exportTargetHeight) || 900;
    if (s.exportAspectLocked !== undefined) exportState.isAspectLocked = !!s.exportAspectLocked;
    if (s.exportFormat !== undefined) {
      exportState.format = s.exportFormat;
      const fmtRadio = document.querySelector(`input[name='export-format'][value='${s.exportFormat}']`);
      if (fmtRadio) {
        fmtRadio.checked = true;
        document.querySelectorAll(".export-format-chip").forEach(c => c.classList.remove("active"));
        fmtRadio.closest(".export-format-chip")?.classList.add("active");
      }
    }
    if (typeof updateAllExportUi === "function") {
      updateAllExportUi();
    }

    if (typeof applyLiveTransforms === "function") {
      applyLiveTransforms();
    }
    console.log("[Studio Settings] Restored saved user settings from localStorage.");
  } catch (err) {
    console.warn("[Studio Settings] Failed to restore settings from localStorage:", err);
  }
}

// Attach autosave hooks to all form controls and window exit events
[
  selectAiModel, toggleCleanHairGaps, toggleSmudgeCleaner, sliderSmudgeSensitivity, sliderAlphaThreshold,
  selectDeskewMode, toggleShineRemover, sliderShineStrength, sliderColorTiers, toggleFlatCel,
  toggleColorPop, selectColorPopPreset, sliderColorPopVibrance, sliderColorPopClarity,
  toggleAiLora, selectAiLoraPreset, toggleBorder, selectHighlightMode, pickerHighlightColor,
  sliderBorderWidth, sliderBorderSmoothing, sliderGlowRadius, toggleSuperRes, selectEnhancerModel, selectEnhancerScale,
  togglePreserveColors, toggleAutoProcess, toggleSaveLocal, selectInpaintModel, sliderInpaintBrush, sliderHighlightBrush,
  sliderRotation, sliderHorizontalSkew, sliderVerticalSkew
].forEach(ctrl => {
  if (!ctrl) return;
  ctrl.addEventListener("input", scheduleAutosave);
  ctrl.addEventListener("change", scheduleAutosave);
});

swatchBtns.forEach(btn => {
  btn.addEventListener("click", scheduleAutosave);
});

[btnRotCCW, btnRotCW, btnRotReset, btnFlipH, btnResetSkew, btnResetVSkew, btnResetAllTransforms, btnToggleAspectLock].forEach(btn => {
  if (btn) btn.addEventListener("click", scheduleAutosave);
});

window.addEventListener("beforeunload", saveStudioSettings);
window.addEventListener("pagehide", saveStudioSettings);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    saveStudioSettings();
  }
});

// Clean Startup: Workspace is completely empty on open, but settings are restored!
window.addEventListener("DOMContentLoaded", () => {
  clearWorkspace();
  restoreStudioSettings();
  if (typeof syncServerLoraPresets === "function") {
    syncServerLoraPresets();
  }
});

