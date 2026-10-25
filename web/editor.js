// Reality Engine CAD editor. All data comes from the local server; nothing is preloaded.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;
const STAGES = ['keyframes', 'geometry', 'scene', 'read', 'build', 'critique', 'verify'];

function el(tag, props = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c);
  return n;
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body?.error || r.statusText);
  return body;
}
function log(msg) { const l = $('log'); l.textContent = (msg + '\n' + l.textContent).slice(0, 20000); }

// ---------------------------------------------------------------- three.js setup
const viewport = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewport.prepend(renderer.domElement);
const world = new THREE.Scene();
world.background = new THREE.Color(0x151b24);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(4, 3, 5);
world.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(4, 8, 3);
world.add(sun);
const grid = new THREE.GridHelper(20, 40, 0x3a4560, 0x222a35);
world.add(grid);
const objGroup = new THREE.Group(), camGroup = new THREE.Group(), measureGroup = new THREE.Group();
world.add(objGroup, camGroup, measureGroup);
let pointsObj = null, pointsArr = null;

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
const tc = new TransformControls(camera, renderer.domElement);
world.add(tc.getHelper());
tc.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value && !photoCam; if (!e.value) commit(); });
tc.addEventListener('objectChange', () => { const m = tc.object; if (m) { syncFromMesh(m); showProps(); } });

const UNIT = {
  box: new THREE.BoxGeometry(1, 1, 1), plane: new THREE.BoxGeometry(1, 1, 1),
  cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, 40), sphere: new THREE.SphereGeometry(0.5, 40, 24),
};

// ---------------------------------------------------------------- state
let runId = null, sceneData = null, meshes = new Map(), selected = null, dirty = false;
let history = [], hIndex = -1, mode = 'select', measurePts = [], lastMeasure = null, photoCam = null;
let pollTimer = null, loadedSceneFor = null;
const mpu = () => sceneData?.calibration?.meters_per_unit || 1;

function setDirty(v) { dirty = v; $('btn-save').textContent = v ? 'Save •' : 'Save'; }

// ---------------------------------------------------------------- objects <-> meshes
function applyToMesh(m, o) {
  m.position.set(...o.position);
  m.rotation.set(o.rotation[0] * DEG, o.rotation[1] * DEG, o.rotation[2] * DEG, 'XYZ');
  m.scale.set(o.size[0], o.kind === 'plane' ? Math.max(o.size[1], 0.002) : o.size[1], o.size[2]);
  m.material.color.set(o.color);
  m.material.emissive.set(o.id === selected ? 0x2a2210 : 0x000000);
}
function meshFor(o) {
  const m = new THREE.Mesh(UNIT[o.kind], new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.85, metalness: 0.05 }));
  m.userData.id = o.id;
  applyToMesh(m, o);
  return m;
}
function rebuildMeshes() {
  tc.detach();
  objGroup.clear();
  meshes = new Map();
  for (const o of sceneData.objects) { const m = meshFor(o); meshes.set(o.id, m); objGroup.add(m); }
  if (selected && !meshes.has(selected)) selected = null;
  attach();
  renderOutliner();
  showProps();
}
const objById = (id) => sceneData.objects.find((o) => o.id === id);
function syncFromMesh(m) {
  const o = objById(m.userData.id);
  if (!o) return;
  const r = (v) => Math.round(v * 10000) / 10000;
  o.position = [r(m.position.x), r(m.position.y), r(m.position.z)];
  o.rotation = [r(m.rotation.x / DEG), r(m.rotation.y / DEG), r(m.rotation.z / DEG)];
  o.size = [Math.max(0.001, r(Math.abs(m.scale.x))), Math.max(0.001, r(Math.abs(m.scale.y))), Math.max(0.001, r(Math.abs(m.scale.z)))];
  setDirty(true);
}

// ---------------------------------------------------------------- history
function snapshot() { return JSON.stringify(sceneData.objects); }
function commit() {
  if (!sceneData) return;
  const s = snapshot();
  if (history[hIndex] === s) return;
  history = history.slice(0, hIndex + 1);
  history.push(s);
  if (history.length > 200) history.shift();
  hIndex = history.length - 1;
  setDirty(true);
  renderOutliner();
}
function restore(i) {
  if (i < 0 || i >= history.length) return;
  hIndex = i;
  sceneData.objects = JSON.parse(history[i]);
  setDirty(true);
  rebuildMeshes();
}

// ---------------------------------------------------------------- selection + outliner + properties
function attach() {
  tc.detach();
  const m = selected && meshes.get(selected);
  if (m && !objById(selected).locked && mode === 'select') tc.attach(m);
}
function select(id) {
  selected = id;
  for (const [k, m] of meshes) m.material.emissive.set(k === id ? 0x2a2210 : 0x000000);
  attach();
  renderOutliner();
  showProps();
}
function renderOutliner() {
  const ul = $('outliner'), q = $('filter').value.toLowerCase();
  ul.replaceChildren();
  if (!sceneData) return;
  const list = sceneData.objects.filter((o) => !q || o.name.toLowerCase().includes(q) || (o.zone || '').toLowerCase().includes(q));
  $('obj-count').textContent = `(${sceneData.objects.length})`;
  for (const o of list) {
    const sw = el('span', { class: 'swatch' });
    sw.style.background = o.color;
    ul.append(el('li', {}, el('button', { type: 'button', class: o.id === selected ? 'sel' : '', onclick: () => select(o.id) },
      sw, el('span', { class: 'name', text: o.name + (o.locked ? ' (locked)' : '') }), el('span', { class: 'meta', text: o.kind }))));
  }
}
const F = ['px', 'py', 'pz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz'];
function showProps() {
  const o = selected && objById(selected);
  $('prop-form').hidden = !o;
  $('no-sel').hidden = !!o;
  if (!o) return;
  const k = mpu(), fix = (v, d = 3) => Number(v.toFixed(d));
  const vals = [...o.position.map((v) => fix(v * k)), ...o.rotation.map((v) => fix(v, 1)), ...o.size.map((v) => fix(v * k))];
  F.forEach((f, i) => { if (document.activeElement !== $('p-' + f)) $('p-' + f).value = vals[i]; });
  $('p-name').value = o.name; $('p-color').value = o.color; $('p-material').value = o.material || '';
  $('p-zone').value = o.zone || ''; $('p-notes').value = o.notes || ''; $('p-locked').checked = !!o.locked;
  $('p-meta').textContent = `${o.id} · ${o.kind} · source: ${o.source}`;
}
$('prop-form').addEventListener('change', () => {
  const o = selected && objById(selected);
  if (!o) return;
  const v = F.map((f) => Number($('p-' + f).value));
  if (v.some((x) => !Number.isFinite(x))) return showProps();
  const k = mpu();
  o.position = v.slice(0, 3).map((x) => x / k);
  o.rotation = v.slice(3, 6);
  o.size = v.slice(6, 9).map((x) => Math.max(0.001, x / k));
  o.name = $('p-name').value.trim().slice(0, 80) || o.name;
  o.color = $('p-color').value; o.material = $('p-material').value.slice(0, 80);
  o.zone = $('p-zone').value.slice(0, 80); o.notes = $('p-notes').value.slice(0, 500); o.locked = $('p-locked').checked;
  applyToMesh(meshes.get(o.id), o);
  attach();
  commit();
  showProps();
});
$('prop-form').addEventListener('submit', (e) => e.preventDefault());
$('filter').addEventListener('input', renderOutliner);

// ---------------------------------------------------------------- editing commands
function floorPointAhead() {
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2(0, 0), camera);
  const p = new THREE.Vector3();
  return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? p : orbit.target.clone();
}
function newId(kind) { return `${kind}-${Date.now().toString(36)}${Math.floor(Math.random() * 1296).toString(36)}`; }
function add(kind) {
  if (!sceneData) return;
  const p = floorPointAhead(), s = 0.5 / mpu();
  const n = sceneData.objects.filter((o) => o.kind === kind).length + 1;
  const o = { id: newId(kind), name: `${kind[0].toUpperCase() + kind.slice(1)} ${n}`, kind,
    size: kind === 'plane' ? [s * 2, 0.002, s * 2] : [s, s, s], position: [p.x, kind === 'plane' ? 0.001 : s / 2, p.z],
    rotation: [0, 0, 0], color: '#b8bcc4', material: '', locked: false, zone: '', source: 'manual', notes: '' };
  sceneData.objects.push(o);
  const m = meshFor(o); meshes.set(o.id, m); objGroup.add(m);
  select(o.id);
  commit();
}
document.querySelectorAll('[data-add]').forEach((b) => b.addEventListener('click', () => add(b.dataset.add)));
function duplicate() {
  const o = selected && objById(selected);
  if (!o) return;
  const c = { ...structuredClone(o), id: newId(o.kind), name: o.name + ' copy', locked: false, source: 'manual' };
  c.position = [o.position[0] + 0.1 / mpu(), o.position[1], o.position[2]];
  sceneData.objects.push(c);
  const m = meshFor(c); meshes.set(c.id, m); objGroup.add(m);
  select(c.id); commit();
}
function remove() {
  if (!selected) return;
  sceneData.objects = sceneData.objects.filter((o) => o.id !== selected);
  selected = null; rebuildMeshes(); commit();
}
function dropToFloor() {
  const m = selected && meshes.get(selected);
  if (!m) return;
  const box = new THREE.Box3().setFromObject(m);
  m.position.y -= box.min.y;
  syncFromMesh(m); showProps(); commit();
}
function pct(arr, q) { const s = Float64Array.from(arr).sort(); return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]; }
function fitToPoints() {
  const m = selected && meshes.get(selected);
  if (!m || !pointsArr) return log('Fit to points needs a selected object and a point cloud.');
  m.updateMatrixWorld();
  const inv = m.matrixWorld.clone().invert(), v = new THREE.Vector3();
  const xs = [], ys = [], zs = [];
  const floorEps = sceneData.floor?.detected ? 0.03 / mpu() : -Infinity;  // floor points belong to the floor
  for (let i = 0; i < pointsArr.length; i += 3) {
    if (pointsArr[i + 1] < floorEps) continue;
    v.set(pointsArr[i], pointsArr[i + 1], pointsArr[i + 2]).applyMatrix4(inv);
    if (Math.abs(v.x) <= 0.6 && Math.abs(v.y) <= 0.6 && Math.abs(v.z) <= 0.6) {
      xs.push(v.x * m.scale.x); ys.push(v.y * m.scale.y); zs.push(v.z * m.scale.z);
    }
  }
  if (xs.length < 20) return log(`Fit to points: only ${xs.length} points inside the object; move it over the points first.`);
  const lo = [pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02)], hi = [pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98)];
  const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2).applyQuaternion(m.quaternion);
  m.position.add(center);
  m.scale.set(Math.max(hi[0] - lo[0], 0.005), Math.max(hi[1] - lo[1], 0.005), Math.max(hi[2] - lo[2], 0.005));
  syncFromMesh(m); showProps(); commit();
  log(`Fit to points: used ${xs.length} reconstructed points.`);
}
$('btn-dup').onclick = duplicate; $('btn-del').onclick = remove; $('btn-floor').onclick = dropToFloor; $('btn-fit').onclick = fitToPoints;
$('btn-undo').onclick = () => restore(hIndex - 1); $('btn-redo').onclick = () => restore(hIndex + 1);
document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => setTransform(b.dataset.mode)));
function setTransform(t) {
  tc.setMode(t);
  document.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('on', b.dataset.mode === t));
}
$('btn-space').onclick = () => { tc.setSpace(tc.space === 'world' ? 'local' : 'world'); $('btn-space').textContent = tc.space === 'world' ? 'World' : 'Local'; };
function applySnap() {
  const on = $('snap').checked, step = Number($('snap-step').value) / mpu();
  tc.setTranslationSnap(on ? step : null); tc.setRotationSnap(on ? 15 * DEG : null); tc.setScaleSnap(on ? step : null);
}
$('snap').onchange = applySnap; $('snap-step').onchange = applySnap; applySnap();

// ---------------------------------------------------------------- picking + measure
const ray = new THREE.Raycaster();
ray.params.Points.threshold = 0.03;
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || tc.dragging) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  if (mode === 'measure') return measureClick();
  const hit = ray.intersectObjects(objGroup.children, false)[0];
  select(hit ? hit.object.userData.id : null);
});
function measureClick() {
  const targets = [...objGroup.children, ...(pointsObj && pointsObj.visible ? [pointsObj] : [])];
  const hit = ray.intersectObjects(targets, false)[0];
  const p = hit ? hit.point.clone() : ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
  if (!p) return;
  if (measurePts.length === 2) { measurePts = []; measureGroup.clear(); }
  measurePts.push(p);
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.015, 12, 8), new THREE.MeshBasicMaterial({ color: 0xf2a541, depthTest: false }));
  dot.position.copy(p); dot.renderOrder = 10; measureGroup.add(dot);
  if (measurePts.length === 2) {
    const g = new THREE.BufferGeometry().setFromPoints(measurePts);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xf2a541, depthTest: false }));
    line.renderOrder = 10; measureGroup.add(line);
    lastMeasure = measurePts[0].distanceTo(measurePts[1]);
    $('measure-out').textContent = `${(lastMeasure * mpu()).toFixed(3)} m  (${lastMeasure.toFixed(4)} scene units)`;
  } else $('measure-out').textContent = 'Click the second point.';
}
function setMode(m) {
  mode = m;
  $('btn-measure').classList.toggle('on', m === 'measure');
  if (m !== 'measure') { measurePts = []; }
  attach();
}
$('btn-measure').onclick = () => setMode(mode === 'measure' ? 'select' : 'measure');
$('btn-calibrate').onclick = () => {
  const L = Number($('true-len').value);
  if (!sceneData || !lastMeasure || !(L > 0)) return log('Measure a known distance first, then enter its true length.');
  sceneData.calibration = { meters_per_unit: L / lastMeasure, method: 'measured', note: `${L} m reference measured in the editor` };
  setDirty(true); applySnap(); showCalibration(); showProps();
  $('measure-out').textContent = `${L.toFixed(3)} m (calibrated)`;
};
function showCalibration() {
  const c = sceneData?.calibration;
  $('calib').textContent = c ? `Scale: 1 unit = ${c.meters_per_unit.toFixed(4)} m (${c.method.replaceAll('_', ' ')}${c.note ? ': ' + c.note : ''})` : '';
}

// ---------------------------------------------------------------- photo (view through a keyframe camera)
function viewThrough(cam) {
  photoCam = cam;
  orbit.enabled = false;
  camera.fov = cam.fov_y; camera.position.set(...cam.position); camera.quaternion.set(...cam.quaternion);
  camera.updateProjectionMatrix();
  $('photo').src = `/runs/${runId}/${cam.image}`; $('photo').hidden = false; $('photo-bar').hidden = false;
  $('photo-label').textContent = `keyframe ${cam.id} · t = ${cam.t.toFixed(2)} s`;
  $('photo').style.opacity = $('photo-opacity').value / 100;
  document.querySelectorAll('#strip button').forEach((b) => b.classList.toggle('sel', b.dataset.cam === cam.id));
  resize();
}
function exitPhoto() {
  if (!photoCam) return;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  orbit.target.copy(camera.position).addScaledVector(fwd, 2);
  photoCam = null; camera.fov = 60; camera.updateProjectionMatrix();
  orbit.enabled = true; $('photo').hidden = true; $('photo-bar').hidden = true;
  document.querySelectorAll('#strip button').forEach((b) => b.classList.remove('sel'));
  resize();
}
$('btn-exit-photo').onclick = exitPhoto;
$('photo-opacity').oninput = (e) => { $('photo').style.opacity = e.target.value / 100; };

function resize() {
  const W = viewport.clientWidth, H = viewport.clientHeight;
  let w = W, h = H;
  if (photoCam) {
    const a = photoCam.width / photoCam.height;
    if (W / H > a) w = Math.round(H * a); else h = Math.round(W / a);
    Object.assign($('photo').style, { width: w + 'px', height: h + 'px' });
  }
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

// ---------------------------------------------------------------- loading a run
function buildCameras() {
  camGroup.clear();
  const strip = $('strip');
  strip.replaceChildren();
  for (const c of sceneData.cameras) {
    const h = Math.tan(c.fov_y * DEG / 2) * 0.15, w = h * c.width / c.height, z = -0.15;
    const pts = [[0, 0, 0], [-w, -h, z], [0, 0, 0], [w, -h, z], [0, 0, 0], [w, h, z], [0, 0, 0], [-w, h, z],
      [-w, -h, z], [w, -h, z], [w, -h, z], [w, h, z], [w, h, z], [-w, h, z], [-w, h, z], [-w, -h, z]];
    const g = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts.flat(), 3));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x8b9dff }));
    lines.position.set(...c.position); lines.quaternion.set(...c.quaternion);
    camGroup.add(lines);
    const img = el('img', { src: `/runs/${runId}/${c.image}`, alt: `Keyframe at ${c.t.toFixed(1)} s`, loading: 'lazy' });
    strip.append(el('button', { type: 'button', 'data-cam': c.id, onclick: () => viewThrough(c) }, img, el('span', { text: `${c.t.toFixed(0)}s` })));
  }
  if (sceneData.cameras.length) {
    const box = new THREE.Box3();
    sceneData.cameras.forEach((c) => box.expandByPoint(new THREE.Vector3(...c.position)));
    const ctr = box.getCenter(new THREE.Vector3());
    orbit.target.set(ctr.x, 0.5, ctr.z);
    camera.position.set(ctr.x + 4, 3.5, ctr.z + 4);
  }
}
async function loadPoints() {
  if (pointsObj) { world.remove(pointsObj); pointsObj.geometry.dispose(); pointsObj = null; pointsArr = null; }
  if (!sceneData.points) return;
  const d = await api(`/runs/${runId}/${sceneData.points}`);
  pointsArr = Float32Array.from(d.positions);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pointsArr, 3));
  g.setAttribute('color', new THREE.BufferAttribute(Float32Array.from(d.colors, (c) => c / 255), 3));
  pointsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true }));
  pointsObj.visible = $('show-points').checked;
  world.add(pointsObj);
  log(`Loaded ${d.count.toLocaleString()} reconstructed points.`);
}
async function loadScene() {
  sceneData = await api(`/api/runs/${runId}/scene`);
  loadedSceneFor = runId;
  history = [snapshot()]; hIndex = 0; setDirty(false); selected = null;
  $('empty').hidden = true;
  rebuildMeshes(); buildCameras(); showCalibration(); applySnap();
  await loadPoints();
  $('btn-mjcf').href = `/api/runs/${runId}/scene.mjcf`;
  try { showVerify(await api(`/api/runs/${runId}/verify`)); } catch { $('verify-summary').textContent = 'Not verified yet.'; }
}
function renderStages(st) {
  const box = $('stages');
  box.replaceChildren(...STAGES.map((s) => {
    const v = st?.stages?.[s] || { state: 'pending' };
    return el('span', { class: `stage ${v.state}`, title: v.detail || v.state, text: s });
  }));
}
async function poll() {
  if (!runId) return;
  const id = runId;
  try {
    const st = await api(`/api/runs/${id}/status`);
    if (id !== runId) return;
    renderStages(st);
    $('log').textContent = [...st.log].reverse().join('\n');
    if (st.stages.scene?.state === 'done' && loadedSceneFor !== id) await loadScene();
    if (st.stages.verify?.state === 'done' && loadedSceneFor === id && !dirty && st.busy === false && pollTimer) {
      try { showVerify(await api(`/api/runs/${id}/verify`)); } catch { /* not written yet */ }
    }
    const running = st.busy || Object.values(st.stages).some((v) => v.state === 'running');
    clearTimeout(pollTimer);
    pollTimer = running ? setTimeout(poll, 2000) : null;
  } catch (e) { log(`status: ${e.message}`); }
}
async function loadRun(id) {
  if (dirty) log('Unsaved changes were discarded when switching runs.');
  clearTimeout(pollTimer); pollTimer = null;
  runId = id; sceneData = null; loadedSceneFor = null; exitPhoto();
  objGroup.clear(); camGroup.clear(); measureGroup.clear(); $('strip').replaceChildren(); tc.detach();
  if (pointsObj) { world.remove(pointsObj); pointsObj = null; pointsArr = null; }
  $('empty').hidden = !!id; renderOutliner(); showProps();
  if (!id) return renderStages(null);
  pollTimer = 1; await poll();
}
async function refreshRuns(pick) {
  const runs = await api('/api/runs');
  const sel = $('run-select');
  sel.replaceChildren(...(runs.length ? runs.map((r) => el('option', { value: r.id, text: r.id })) : [el('option', { value: '', text: 'No runs yet' })]));
  const id = pick || runs[0]?.id || '';
  sel.value = id;
  await loadRun(id || null);
}
$('run-select').onchange = (e) => loadRun(e.target.value || null);

// ---------------------------------------------------------------- upload, save, verify, export
$('upload').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  log(`Uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`);
  try {
    const r = await api('/api/runs', { method: 'POST', headers: { 'X-Filename': f.name }, body: f });
    await refreshRuns(r.id);
  } catch (err) { log(`Upload failed: ${err.message}`); }
  e.target.value = '';
};
async function save() {
  if (!sceneData) return;
  try { await api(`/api/runs/${runId}/scene`, { method: 'PUT', body: JSON.stringify(sceneData), headers: { 'Content-Type': 'application/json' } }); setDirty(false); log('Saved.'); }
  catch (e) { log(`Save failed: ${e.message}`); }
}
$('btn-save').onclick = save;
$('btn-verify').onclick = async () => {
  if (!sceneData) return;
  if (dirty) await save();
  $('verify-summary').textContent = 'Rendering every keyframe…';
  try { showVerify(await api(`/api/runs/${runId}/verify`, { method: 'POST' })); } catch (e) { $('verify-summary').textContent = `Verify failed: ${e.message}`; }
};
$('btn-rebuild').onclick = async () => {
  if (!runId) return;
  if (dirty) await save();
  try { await api(`/api/runs/${runId}/rebuild`, { method: 'POST' }); loadedSceneFor = null; pollTimer = 1; poll(); } catch (e) { log(`Rebuild: ${e.message}`); }
};
function showVerify(v) {
  const sum = $('verify-summary');
  sum.replaceChildren(
    el('div', { class: 'row2' },
      el('div', {}, el('div', { class: 'score', text: v.mean_ssim ?? '–' }), el('div', { class: 'muted', text: 'mean SSIM' })),
      el('div', {}, el('div', { class: 'score', text: v.points_explained != null ? Math.round(v.points_explained * 100) + '%' : '–' }), el('div', { class: 'muted', text: 'points on a surface' }))),
    el('div', { class: 'muted', text: `edge F1 ${v.mean_edge_f1 ?? '–'} · ${v.objects} objects · ${v.keyframes.length} keyframes` }));
  const ul = $('verify-list');
  ul.replaceChildren(...[...v.keyframes].sort((a, b) => a.ssim - b.ssim).map((k) =>
    el('li', {}, el('button', { type: 'button', onclick: () => {
      $('verify-img').src = `/runs/${runId}/${k.compare}?${Date.now()}`; $('verify-img').hidden = false;
      const c = sceneData?.cameras.find((x) => x.id === k.camera); if (c) viewThrough(c);
    } }, el('span', { class: 'swatch' }), el('span', { class: 'name', text: `t = ${k.t.toFixed(1)} s` }), el('span', { class: 'meta', text: `SSIM ${k.ssim} · edge ${k.edge_f1}` })))));
}
$('btn-glb').onclick = () => {
  if (!sceneData) return;
  new GLTFExporter().parse(objGroup, (res) => {
    const a = el('a', { href: URL.createObjectURL(new Blob([res], { type: 'model/gltf-binary' })), download: `${runId}.glb` });
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, (err) => log(`glTF export failed: ${err.message || err}`), { binary: true });
};
$('show-points').onchange = (e) => { if (pointsObj) pointsObj.visible = e.target.checked; };
$('show-cams').onchange = (e) => { camGroup.visible = e.target.checked; };

// ---------------------------------------------------------------- keyboard
addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, select')) return;
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && e.key.toLowerCase() === 's') { e.preventDefault(); save(); }
  else if (ctrl && e.key.toLowerCase() === 'z') { e.preventDefault(); restore(hIndex - 1); }
  else if (ctrl && e.key.toLowerCase() === 'y') { e.preventDefault(); restore(hIndex + 1); }
  else if (ctrl && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicate(); }
  else if (e.key === 'Delete' || e.key === 'Backspace') remove();
  else if (e.key === 'w') setTransform('translate');
  else if (e.key === 'e') setTransform('rotate');
  else if (e.key === 'r') setTransform('scale');
  else if (e.key === 'm') setMode(mode === 'measure' ? 'select' : 'measure');
  else if (e.key === 'Escape') { setMode('select'); exitPhoto(); }
});
addEventListener('beforeunload', (e) => { if (dirty) e.preventDefault(); });

// ---------------------------------------------------------------- boot
renderer.setAnimationLoop(() => {
  if (!photoCam) orbit.update();
  $('hud').hidden = !sceneData;
  if (sceneData) $('hud').textContent = `${sceneData.objects.length} objects · ${sceneData.cameras.length} keyframe cameras · ${mode === 'measure' ? 'measuring' : 'select'}`;
  renderer.render(world, camera);
});
api('/api/config').then((c) => { $('gemini').textContent = c.gemini ? 'Gemini: ready' : 'Gemini: add key in .env'; }).catch(() => {});
refreshRuns().catch((e) => log(`Server unreachable: ${e.message}`));
