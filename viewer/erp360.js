import * as THREE from "three";

const params = new URLSearchParams(window.location.search);
const runId = params.get("run") || params.get("run_id") || "";

/** 将路径规范为 outputs 下的相对路径（支持整段 Windows 路径或 `/outputs/...`）。 */
function normalizeImgParam(raw) {
  if (!raw || !String(raw).trim()) return "";
  let s = String(raw).trim().replace(/\\/g, "/");
  const low = s.toLowerCase();
  const needle = "/outputs/";
  const j = low.lastIndexOf(needle);
  if (j >= 0) {
    s = s.slice(j + needle.length);
  } else if (low.startsWith("outputs/")) {
    s = s.slice("outputs/".length);
  }
  return s.replace(/^\/+/, "");
}

/** 从服务器加载的 outputs 相对路径（可被 UI / ?img= 更新）。 */
let outputsImgRel = normalizeImgParam(params.get("img") || "");

/** 本机选择的文件（blob URL）。 */
let localObjectUrl = null;
let localFileName = "";

/** @type {"4x" | "1x"} */
let resMode = params.get("res") === "1x" ? "1x" : "4x";
let poseIdx = Math.max(0, parseInt(params.get("pose") || "0", 10) || 0);
let numPoses = 1;

const hudLines = document.getElementById("hud-lines");
const statusEl = document.getElementById("status");
const runControls = document.getElementById("erp360-run-controls");
const btn4x = document.getElementById("btn-4x");
const btn1x = document.getElementById("btn-1x");
const poseInput = document.getElementById("pose-input");
const posePrev = document.getElementById("pose-prev");
const poseNext = document.getElementById("pose-next");
const poseOf = document.getElementById("pose-of");

const btnPickLocal = document.getElementById("btn-pick-local");
const inputLocalFile = document.getElementById("input-local-file");
const inputOutputsRel = document.getElementById("input-outputs-rel");
const btnLoadOutputs = document.getElementById("btn-load-outputs");
const btnClearCustom = document.getElementById("btn-clear-custom");

if (outputsImgRel) {
  inputOutputsRel.value = outputsImgRel;
}

function erpUrl(pose, mode) {
  const sub = mode === "4x" ? "rgb_4x" : "rgb";
  return runId ? `runs/${runId}/erp/${sub}/pose_${pose}.png` : "";
}

function outputsToFetchUrl(rel) {
  return (
    "/outputs/" +
    rel
      .split("/")
      .filter(Boolean)
      .map((seg) => encodeURIComponent(seg))
      .join("/")
  );
}

/** 当前用于 TextureLoader 的 URL（blob、/outputs/… 或 runs/…）。 */
function getTextureLoadUrl() {
  if (localObjectUrl) return localObjectUrl;
  if (outputsImgRel) return outputsToFetchUrl(outputsImgRel);
  return erpUrl(poseIdx, resMode);
}

function isCustomMode() {
  return Boolean(outputsImgRel || localObjectUrl);
}

function clearLocalBlob() {
  if (localObjectUrl) {
    URL.revokeObjectURL(localObjectUrl);
    localObjectUrl = null;
  }
  localFileName = "";
}

function setStatus(msg, isErr = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("err", isErr);
}

function syncResButtons() {
  btn4x.classList.toggle("active", resMode === "4x");
  btn1x.classList.toggle("active", resMode === "1x");
}

function syncPoseUi() {
  poseInput.value = String(poseIdx);
  poseInput.max = String(Math.max(0, numPoses - 1));
  poseOf.textContent = numPoses > 0 ? `/ ${numPoses - 1}` : "";
}

function pushUrl() {
  const q = new URLSearchParams();
  if (localObjectUrl) {
    window.history.replaceState({}, "", window.location.pathname);
    return;
  }
  if (outputsImgRel) {
    q.set("img", outputsImgRel);
    const url = `${window.location.pathname}?${q.toString()}`;
    window.history.replaceState({}, "", url);
    return;
  }
  if (!runId) {
    window.history.replaceState({}, "", window.location.pathname);
    return;
  }
  q.set("run", runId);
  q.set("pose", String(poseIdx));
  q.set("res", resMode === "4x" ? "4x" : "1x");
  window.history.replaceState({}, "", `${window.location.pathname}?${q.toString()}`);
}

function updateHudText(extra) {
  const lines = [];
  if (localFileName) {
    lines.push("模式: <b>本机文件</b>");
    lines.push(`文件: <code>${localFileName}</code>`);
  } else if (outputsImgRel) {
    lines.push("模式: <b>服务器 outputs 贴图</b>");
    lines.push(`路径: <code>outputs/${outputsImgRel}</code>`);
  } else if (!runId) {
    lines.push("用下方 <b>选择本地图片</b> 或填写 <b>outputs 相对路径</b> 后点「从服务器加载」。");
    lines.push("也可 URL：<code>?run=…</code> 或 <code>?img=_probe/foo.png</code>");
  } else {
    lines.push(`run: <b>${runId}</b>`);
    lines.push(`贴图: <code>${erpUrl(poseIdx, resMode)}</code>`);
  }
  if (extra) lines.push(extra);
  hudLines.innerHTML = lines.join("<br />");
}

// ---- Three.js（基于 three.js equirectangular 示例：内向球 + lon/lat）----
const container = document.getElementById("app");

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1100
);

const scene = new THREE.Scene();

const sphereGeom = new THREE.SphereGeometry(500, 72, 48);
sphereGeom.scale(-1, 1, 1);

const textureLoader = new THREE.TextureLoader();
let currentTexture = null;
const material = new THREE.MeshBasicMaterial();
const mesh = new THREE.Mesh(sphereGeom, material);
scene.add(mesh);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

let lon = 0;
let lat = 0;
let onPointerDownMouseX = 0;
let onPointerDownMouseY = 0;
let onPointerDownLon = 0;
let onPointerDownLat = 0;
let dragging = false;

function onPointerDown(event) {
  if (event.isPrimary === false) return;
  dragging = true;
  onPointerDownMouseX = event.clientX;
  onPointerDownMouseY = event.clientY;
  onPointerDownLon = lon;
  onPointerDownLat = lat;
  document.addEventListener("pointermove", onPointerMove);
  document.addEventListener("pointerup", onPointerUp);
}

function onPointerMove(event) {
  if (!dragging || event.isPrimary === false) return;
  lon = (onPointerDownMouseX - event.clientX) * 0.1 + onPointerDownLon;
  lat = (event.clientY - onPointerDownMouseY) * 0.1 + onPointerDownLat;
}

function onPointerUp(event) {
  if (event.isPrimary === false) return;
  dragging = false;
  document.removeEventListener("pointermove", onPointerMove);
  document.removeEventListener("pointerup", onPointerUp);
}

function onWheel(event) {
  event.preventDefault();
  const fov = camera.fov + event.deltaY * 0.04;
  camera.fov = THREE.MathUtils.clamp(fov, 15, 90);
  camera.updateProjectionMatrix();
}

container.addEventListener("pointerdown", onPointerDown);
document.addEventListener("wheel", onWheel, { passive: false });

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onResize);

function renderFrame() {
  lat = Math.max(-85, Math.min(85, lat));
  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.MathUtils.degToRad(lon);
  const x = 500 * Math.sin(phi) * Math.cos(theta);
  const y = 500 * Math.cos(phi);
  const z = 500 * Math.sin(phi) * Math.sin(theta);
  camera.position.set(0, 0, 0);
  camera.lookAt(x, y, z);
  renderer.render(scene, camera);
}

renderer.setAnimationLoop(renderFrame);

function disposeCurrentTexture() {
  if (currentTexture) {
    currentTexture.dispose();
    currentTexture = null;
  }
  material.map = null;
  material.needsUpdate = true;
}

function applyTexture(tex) {
  disposeCurrentTexture();
  currentTexture = tex;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  material.map = tex;
  material.needsUpdate = true;
}

async function fetchMetaNumPoses() {
  if (!runId) return null;
  try {
    const res = await fetch(`runs/${runId}/meta.json`);
    if (!res.ok) return null;
    const meta = await res.json();
    const n = meta.num_poses;
    if (typeof n === "number" && n > 0) return n;
  } catch {
    /* ignore */
  }
  return null;
}

async function probePoseCountFromRgb() {
  if (!runId) return 1;
  let n = 0;
  const maxTry = 32;
  for (let i = 0; i < maxTry; i++) {
    const res = await fetch(`runs/${runId}/erp/rgb/pose_${i}.png`, { method: "HEAD" });
    if (!res.ok) break;
    n = i + 1;
  }
  return Math.max(n, 1);
}

function loadPanorama() {
  const url = getTextureLoadUrl();
  if (!url) {
    updateHudText();
    setStatus("请选择本地图片、填写 outputs 路径，或使用 ?run= / ?img=。");
    return;
  }

  setStatus(`加载中… ${url.startsWith("blob:") ? localFileName || "本地文件" : url}`);

  textureLoader.load(
    url,
    (tex) => {
      applyTexture(tex);
      const label = url.startsWith("blob:") ? `${localFileName}（本地）` : url;
      setStatus(`已加载 ${label}  (${tex.image.width}×${tex.image.height})`);
      updateHudText();
    },
    undefined,
    () => {
      disposeCurrentTexture();
      const hint = outputsImgRel
        ? "确认已运行 <code>python scripts/serve_viewer.py</code>，且文件在仓库 <code>outputs/</code> 下。"
        : resMode === "4x"
          ? "若尚未跑超分，请执行 <code>python scripts/sr_erp_4x.py --run-id …</code> 或改用「1x 原始」。"
          : "";
      updateHudText(`<span class="err">无法加载该贴图。</span> ${hint}`);
      setStatus(`加载失败: ${url.startsWith("blob:") ? localFileName : url}`, true);
    }
  );
}

async function init() {
  runControls.style.display = isCustomMode() ? "none" : "";
  syncResButtons();
  if (!isCustomMode()) {
    const n = await fetchMetaNumPoses();
    if (n != null) {
      numPoses = n;
    } else if (runId) {
      numPoses = await probePoseCountFromRgb();
    }
    poseIdx = Math.min(poseIdx, Math.max(0, numPoses - 1));
    syncPoseUi();
  }
  pushUrl();
  loadPanorama();
}

btnPickLocal.addEventListener("click", () => inputLocalFile.click());

inputLocalFile.addEventListener("change", () => {
  const file = inputLocalFile.files && inputLocalFile.files[0];
  inputLocalFile.value = "";
  if (!file) return;
  clearLocalBlob();
  outputsImgRel = "";
  inputOutputsRel.value = "";
  localFileName = file.name;
  localObjectUrl = URL.createObjectURL(file);
  runControls.style.display = "none";
  pushUrl();
  loadPanorama();
});

function loadFromOutputsInput() {
  const rel = normalizeImgParam(inputOutputsRel.value);
  if (!rel) {
    setStatus("请先填写 outputs 下的相对路径。", true);
    return;
  }
  clearLocalBlob();
  localFileName = "";
  outputsImgRel = rel;
  runControls.style.display = "none";
  pushUrl();
  loadPanorama();
}

btnLoadOutputs.addEventListener("click", loadFromOutputsInput);
inputOutputsRel.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadFromOutputsInput();
});

btnClearCustom.addEventListener("click", () => {
  clearLocalBlob();
  outputsImgRel = "";
  localFileName = "";
  inputOutputsRel.value = "";
  runControls.style.display = "";
  if (runId) {
    const n = numPoses;
    poseIdx = Math.min(poseIdx, Math.max(0, n - 1));
    syncPoseUi();
  }
  pushUrl();
  loadPanorama();
});

btn4x.addEventListener("click", () => {
  if (isCustomMode()) return;
  resMode = "4x";
  syncResButtons();
  pushUrl();
  loadPanorama();
});
btn1x.addEventListener("click", () => {
  if (isCustomMode()) return;
  resMode = "1x";
  syncResButtons();
  pushUrl();
  loadPanorama();
});

function setPose(i) {
  if (isCustomMode()) return;
  poseIdx = Math.max(0, Math.min(numPoses - 1, i));
  syncPoseUi();
  pushUrl();
  loadPanorama();
}

posePrev.addEventListener("click", () => setPose(poseIdx - 1));
poseNext.addEventListener("click", () => setPose(poseIdx + 1));
poseInput.addEventListener("change", () => {
  const v = parseInt(poseInput.value, 10);
  if (Number.isFinite(v)) setPose(v);
  else syncPoseUi();
});

init();
