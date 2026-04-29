import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const params = new URLSearchParams(window.location.search);
const runId = params.get("run") || params.get("run_id") || "";
const baseUrl = runId ? `runs/${runId}/` : "";
const plyUrl = params.get("ply") || `${baseUrl}gs/output.ply`;
const metaUrl = `${baseUrl}meta.json`;

const hud = document.getElementById("hud");
const status = document.getElementById("status");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  1000
);
camera.position.set(0, -4, 1.5);
camera.up.set(0, 0, 1);
camera.lookAt(0, 0, 1.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("app").appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 1.5);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

const spark = new SparkRenderer({ renderer });
scene.add(spark);

const grid = new THREE.GridHelper(20, 20, 0x222222, 0x111111);
grid.rotation.x = Math.PI * 0.5;
scene.add(grid);

const axes = new THREE.AxesHelper(0.6);
scene.add(axes);

let splatMesh = null;
let runMeta = null;

async function loadMeta() {
  try {
    const res = await fetch(metaUrl);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

function updateHud() {
  const lines = [];
  if (runMeta) {
    if (runMeta.run_id) lines.push(`run: ${runMeta.run_id}`);
    if (runMeta.scene) {
      lines.push(`scene: ${runMeta.scene.scene_kind || "?"}`);
      if (runMeta.scene.style) lines.push(`style: ${runMeta.scene.style}`);
      if (runMeta.scene.light) lines.push(`light: ${runMeta.scene.light}`);
      if (runMeta.scene.occupancy) lines.push(`occ: ${runMeta.scene.occupancy}`);
    }
    if (runMeta.cuboid_size) {
      const s = runMeta.cuboid_size;
      lines.push(`cuboid: ${s[0]} x ${s[1]} x ${s[2]} m`);
    }
    if (runMeta.num_poses) lines.push(`poses: ${runMeta.num_poses}`);
    if (runMeta.gs_used_real_fastgs === false) {
      lines.push("(fallback PLY -- FastGS not available)");
    }
  }
  lines.push("");
  lines.push(`ply: ${plyUrl}`);
  lines.push("WASD/arrows + mouse drag (orbit)");
  hud.textContent = lines.join("\n");
}

function focusOnSplat() {
  if (!splatMesh) return;
  splatMesh.position.set(0, 0, 0);
  splatMesh.quaternion.set(1, 0, 0, 0);
  controls.target.set(0, 0, 1.5);
  controls.update();
}

async function init() {
  runMeta = await loadMeta();
  updateHud();

  status.textContent = `loading ${plyUrl}`;
  try {
    splatMesh = new SplatMesh({
      url: plyUrl,
      onLoad: () => {
        status.textContent = `loaded ${plyUrl}`;
        focusOnSplat();
      },
    });
    scene.add(splatMesh);
  } catch (e) {
    console.error(e);
    status.textContent = `failed: ${e}`;
    status.classList.add("err");
  }
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

init();
