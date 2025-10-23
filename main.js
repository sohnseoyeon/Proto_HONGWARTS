import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

/* ========================
   UI 요소
   ======================== */
const $ = (s) => document.querySelector(s);
const header = $("#glass-header");
const titleImg = $("#title-img");
const backImg = $("#back-btn-img");
const helpImg = $("#question-btn-img");
const scene = $("#scene");
const app = $("#app");
const stage = $("#stage");
const altView = $("#alt-view");
const camVideo = $("#cam");
const cursor = $("#cursor");

/* 타이틀 매핑(그대로 유지) */
const TITLE_MAP = {
  main: "./assets/titles/main_title.png",
  "r3-btn": "./assets/titles/r3_title.png",
  "k4l4-btn": "./assets/titles/k4l4_title.png",
  "l4g4-btn": "./assets/titles/l4g4_title.png",
  "p3-btn": "./assets/titles/p3_title.png",
  "g2q1-btn": "./assets/titles/g2q1_title.png",
  "q3f2-btn": "./assets/titles/q3f2_title.png",
  "f2e3-btn": "./assets/titles/f2e3_title.png",
  "f4u2-btn": "./assets/titles/f4u2_title.png",
  "ub2c3-btn": "./assets/titles/ub2c3_title.png",
  "z23z33-btn": "./assets/titles/z23z33_title.png",
  "c3af-btn": "./assets/titles/c3af_title.png",
  "c3-btn": "./assets/titles/c3_title.png",
};

/* 파노라마 텍스처 매핑(스팟 → 이미지) */
const PANO_MAP = {
  "r3-btn": "./assets/views/r3_view.png",
  "k4l4-btn": "./assets/views/k4l4_view.png",
  "l4g4-btn": "./assets/views/l4g4_view.png",
  "p3-btn": "./assets/views/p3_view.png",
  "g2q1-btn": "./assets/views/g2q1_view.png",
  "q3f2-btn": "./assets/views/q3f2_view.png",
  "f2e3-btn": "./assets/views/f2e3_view.png",
  "f4u2-btn": "./assets/views/f4u2_view.png",
  "ub2c3-btn": "./assets/views/ub2c3_view.png",
  "z23z33-btn": "./assets/views/z23z33_view.png",
  "c3af-btn": "./assets/views/c3af_view.png",
  "c3-btn": "./assets/views/c3_view.png",
};

/* 비율 스케일 유지 (네가 준 코드 유지) */
function computeScale() {
  const vw = innerWidth;
  const vh = innerHeight;
  return Math.min(vw / 3840, vh / 2160);
}
function applyScale() {
  document.documentElement.style.setProperty("--scale", computeScale());
}
addEventListener("resize", applyScale);
addEventListener("orientationchange", applyScale);
applyScale();

/* ========================
   제스처 커서 / 공통 상태
   ======================== */

// === Autorange for hand -> screen mapping ===
let obsX = { min: 1, max: 0 }, obsY = { min: 1, max: 0 };
const EDGE_MARGIN = 0.08;      // 끝 여유 (5~10%)
// const HAND_LOST_MS = 600;      // 손 락 유지 시간 (있으면 그대로)
function resetObs() { obsX = { min: 1, max: 0 }; obsY = { min: 1, max: 0 }; }

function observe(uX, uY) {
  // 관측치 갱신
  obsX.min = Math.min(obsX.min, uX); obsX.max = Math.max(obsX.max, uX);
  obsY.min = Math.min(obsY.min, uY); obsY.max = Math.max(obsY.max, uY);
  // 과도하게 좁아지지 않게 천천히 벌림
  const a = 0.02;
  obsX.min = (1 - a) * obsX.min + a * 0.0; obsX.max = (1 - a) * obsX.max + a * 1.0;
  obsY.min = (1 - a) * obsY.min + a * 0.0; obsY.max = (1 - a) * obsY.max + a * 1.0;
}
function mapAdaptive(u, minV, maxV) {
  const eps = 1e-6, span = Math.max(maxV - minV, eps);
  let t = (u - minV) / span;                        // 0~1 정규화
  t = (t - 0.5) * (1 + 2 * EDGE_MARGIN) + 0.5;     // 좌우/상하 과확장
  return Math.min(1, Math.max(0, t));              // 최종 0~1
}


// === Primary Face Lock (first-seen) ===
let primaryFaceIdx = null;
let lastFaceCentroid = null; // {x, y} in video normalized coords
const FACE_LOST_MS = 2000;   // 이 시간 이상 보이지 않으면 락 해제
let primaryFaceLastSeenAt = 0;

function faceCentroid(lms) {
  // 간단하게 눈/코 평균 중심 사용
  const L = lms[33], R = lms[263], N = lms[1];
  const cx = (L.x + R.x + N.x) / 3;
  const cy = (L.y + R.y + N.y) / 3;
  return { x: cx, y: cy };
}

function distance2(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return dx*dx + dy*dy;
}

// 여러 얼굴이 있을 때도 "처음 잡힌 얼굴"을 유지.
// 얼굴이 잠깐 가려져도 최근 위치에 가장 가까운 걸 이어받음.
// 오래(=FACE_LOST_MS) 안 보이면 락 해제 후 새로 잡힌 첫 얼굴을 다시 락.
function selectPrimaryFaceIndex(fv, nowTs) {
  const faces = fv?.faceLandmarks || [];
  if (!faces.length) return null;

  // 아직 락이 없으면 "첫 번째로 보이는 얼굴"을 락
  if (primaryFaceIdx == null) {
    primaryFaceIdx = 0;
    lastFaceCentroid = faceCentroid(faces[primaryFaceIdx]);
    primaryFaceLastSeenAt = nowTs;
    return primaryFaceIdx;
  }

  // 락이 있는데 landmarks 배열 길이가 이전과 달라졌거나, 인덱스가 벗어나면
  // '이전 중심점에 가장 가까운 얼굴'로 재매칭
  if (!faces[primaryFaceIdx]) {
    let bestI = 0, bestD = Infinity;
    for (let i = 0; i < faces.length; i++) {
      const c = faceCentroid(faces[i]);
      const d2 = distance2(c, lastFaceCentroid || c);
      if (d2 < bestD) { bestD = d2; bestI = i; }
    }
    primaryFaceIdx = bestI;
    lastFaceCentroid = faceCentroid(faces[primaryFaceIdx]);
    primaryFaceLastSeenAt = nowTs;
    return primaryFaceIdx;
  }

  // 정상적으로 보이는 경우: 같은 인덱스를 유지하되, 중심 업데이트
  lastFaceCentroid = faceCentroid(faces[primaryFaceIdx]);
  primaryFaceLastSeenAt = nowTs;
  return primaryFaceIdx;
}

function maybeUnlockPrimaryFace(nowTs) {
  if (primaryFaceIdx != null && nowTs - primaryFaceLastSeenAt > FACE_LOST_MS) {
    primaryFaceIdx = null;
    lastFaceCentroid = null;
  }
}


let running = true,
  rafId = null;
let MODE = "MAP"; // "MAP" | "PANO"
let currentSpot = null;

/* 커서 보간 */
let _cxS = innerWidth * 0.5,
  _cyS = innerHeight * 0.5;
let _cx = _cxS,
  _cy = _cyS;
const MAX_STEP = 20000; // 더 빠르고 멀리 이동 허용
const LERP_MOVE = 0.5; // 반응성 상승

/* 중앙값 버퍼 */
const MEDIAN_WINDOW = 1;
const bufX = [],
  bufY = [];
function pushBuf(buf, v) {
  buf.push(v);
  if (buf.length > MEDIAN_WINDOW) buf.shift();
}
function median(buf) {
  return buf.length ? buf[buf.length - 1] : null;
}

/* 플래시 */
function flash(x, y) {
  const f = document.createElement("div");
  Object.assign(f.style, {
    position: "fixed",
    left: `${x}px`,
    top: `${y}px`,
    width: "12px",
    height: "12px",
    borderRadius: "50%",
    transform: "translate(-50%,-50%)",
    background: "rgba(255,60,60,.9)",
    boxShadow: "0 0 0 10px rgba(255,60,60,.25)",
    pointerEvents: "none",
    zIndex: 3000,
  });
  document.body.appendChild(f);
  setTimeout(() => f.remove(), 220);
}

/* 커서 이동 */
function updateCursor(tx, ty, snap = false) {
  // 커서 시각 크기(120px)와 무관하게 히트 반지름은 별도로 작게
  const HIT_RADIUS = 0; // ← 화면 끝까지 닿게 만들 핵심! (원하면 0~2로 더 줄여도 됨)

  const W = innerWidth,
    H = innerHeight;
  // const R = (cursor?.offsetWidth || 24) / 2;
  const R = HIT_RADIUS
  tx = Math.max(R, Math.min(W - R, tx));
  ty = Math.max(R, Math.min(H - R, ty));

  if (snap) {
    _cxS = tx;
    _cyS = ty;
  } else {
    const dx = tx - _cxS,
      dy = ty - _cyS;
    const dist = Math.hypot(dx, dy);
    if (dist > MAX_STEP) {
      const r = MAX_STEP / dist;
      tx = _cxS + dx * r;
      ty = _cyS + dy * r;
    }
    _cxS = _cxS + (tx - _cxS) * LERP_MOVE;
    _cyS = _cyS + (ty - _cyS) * LERP_MOVE;
  }
  _cx = _cxS;
  _cy = _cyS;
  cursor.style.left = _cx + "px";
  cursor.style.top = _cy + "px";
  cursor.hidden = false;

  // ✅ 손 커서 hover 감지 (SVG 버튼 위인지 확인)
  const el = document.elementFromPoint(_cx, _cy);
  const allBtns = document.querySelectorAll('#scene image[id$="-btn"]');
  allBtns.forEach((btn) => btn.classList.remove("hovered"));
  if (el && el.matches && el.matches('#scene image[id$="-btn"]')) {
    el.classList.add("hovered");
  }
}

/* 클릭 합성 */
let lastClickAt = 0;
const CLICK_COOLDOWN = 500;
function clickAtCursor(x, y) {
  const now = performance.now();
  if (now - lastClickAt < CLICK_COOLDOWN) return false;
  const target = document.elementFromPoint(x | 0, y | 0);
  if (!target) return false;

  const common = {
    bubbles: true,
    cancelable: true,
    clientX: x,
    clientY: y,
    view: window,
  };
  try {
    target.dispatchEvent(
      new PointerEvent("pointerdown", {
        ...common,
        pointerId: 1,
        isPrimary: true,
        buttons: 1,
      })
    );
    target.dispatchEvent(
      new PointerEvent("pointerup", {
        ...common,
        pointerId: 1,
        isPrimary: true,
        buttons: 0,
      })
    );
  } catch {}
  target.dispatchEvent(
    new MouseEvent("mousedown", { ...common, button: 0, buttons: 1 })
  );
  target.dispatchEvent(
    new MouseEvent("mouseup", { ...common, button: 0, buttons: 0 })
  );
  target.dispatchEvent(new MouseEvent("click", { ...common, button: 0 }));

  // flash(x, y);
  cursor.classList.add("clicking");
  setTimeout(() => cursor.classList.remove("clicking"), 180);
  lastClickAt = now;
  return true;
}

/* ========================
   MediaPipe (머리/손)
   ======================== */
let fileset, face, pose, gesture;
const YAW_DIR = -1; // 지도에서 좌우 스크롤/패닝 직관 (지도에서만 사용)
const PITCH_DIR = 1;

let baseYaw = 0,
  basePitch = 0,
  yawS = 0,
  pitchS = 0;
const SMOOTH = 0.3,
  CALIB_MS = 1200;
let calibStart = performance.now();

let lastHandIdx = null,
  lastHandSide = "Right",
  lastHandSeenAt = 0;
const HAND_HYST_MS = 350,
  MIN_SWITCH_DELTA = 0.12;
let lastHandGesture = "None";
const MIN_CONF = 0.65;

/* ✨ 드리프트 방지: pitch 히스테리시스 상태 */
let _pitchActive = false;
const DEAD_Y = 0.5; // 마스크/안경에서도 좌우 감지를 위해 완화
/* 히스테리시스 임계 (시작/종료 다르게) */
const DEAD_P_LO = 0.6; // 상하도 너무 빡세지 않게 완화
const DEAD_P_HI = DEAD_P_LO + 0.12; // 움직임 시작 임계(바깥)

/* 새 관람객 인지용 상태 */
let facePresent = false; // 현재 얼굴이 안정적으로 보이는 상태인지
let lastFaceSeenAt = 0; // 마지막으로 얼굴을 본 시각
let candidateFaceStart = 0; // 얼굴이 다시 보이기 시작한 시각(안정성 확인을 위해)
const FACE_GONE_MS = 2500; // 이 시간 이상 얼굴이 안 보이면 '부재'
const FACE_STABLE_MS = 600; // 재등장 후 이 시간 유지되면 '새 관람객'
let lastFaceGoneAt = 0; // 마지막으로 부재로 판정된 시각

/* ========================
   카메라/모델 로딩
   ======================== */
async function openCam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  camVideo.srcObject = stream;
  await camVideo.play();
  camVideo.style.transform = "scaleX(-1)";
  cursor.hidden = false;
}

async function loadModels() {
  fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  face = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 4,
    minFaceDetectionConfidence: 0.3, // 마스크/모자 상황에서 탐지 민감도 완화
    minFacePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  gesture = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

const HAND_LOST_MS = 600; // 이 시간 이상 그 손이 안 보이면 새 손으로 락 전환

function chooseHandIndex(gv, now) {
  try {
    const arr = gv?.handedness ?? [];
    // 현재 락된 손이 유효하고 최근에 관측됐다면 그대로 유지
    if (lastHandIdx != null) {
      const stillThere =
        gv?.landmarks?.[lastHandIdx] && (now - lastHandSeenAt) < HAND_LOST_MS;
      if (stillThere) {
        return lastHandIdx;
      }
    }

    // 여기까지 왔다는 건 (1) 락이 없거나 (2) 오래 사라졌거나 (3) 인덱스가 무효
    if (!arr.length) return null;

    // "가장 먼저 잡힌 손" = 이번 프레임에서 보이는 가장 앞의(=index가 작은) 손으로 락
    const first = arr[0]?.[0];
    lastHandIdx = 0;
    lastHandSide = first?.categoryName || "Right";
    lastHandSeenAt = now;
    return lastHandIdx;
  } catch {
    return lastHandIdx;
  }
}


function readGesture(gv, idx) {
  try {
    if (idx != null && gv?.gestures?.[idx]?.length) {
      const top = gv.gestures[idx][0];
      return { name: top?.categoryName || "None", score: top?.score ?? 0 };
    }
    const g0 = gv?.gestures?.[0]?.[0];
    return { name: g0?.categoryName || "None", score: 0 };
  } catch {
    return { name: "None", score: 0 };
  }
}

/* 2D Kalman */
class Kalman2D {
  constructor() {
    this.x = new Float64Array([_cxS, _cyS, 0, 0]);
    this.P = this.eye(4, 200);
    this.Q_base = 20.0;
    this.R_meas = 10.0;
  }
  eye(n, s = 1) {
    const M = Array.from({ length: n }, (_, i) => Array(n).fill(0));
    for (let i = 0; i < n; i++) M[i][i] = s;
    return M;
  }
  mul(A, B) {
    const r = A.length,
      c = B[0].length,
      n = B.length,
      R = Array.from({ length: r }, () => Array(c).fill(0));
    for (let i = 0; i < r; i++)
      for (let k = 0; k < n; k++) {
        const v = A[i][k];
        for (let j = 0; j < c; j++) R[i][j] += v * B[k][j];
      }
    return R;
  }
  add(A, B) {
    return A.map((r, i) => r.map((v, j) => v + B[i][j]));
  }
  sub(A, B) {
    return A.map((r, i) => r.map((v, j) => v - B[i][j]));
  }
  tr(A) {
    return A[0].map((_, i) => A.map((r) => r[i]));
  }
  inv2(M) {
    const a = M[0][0],
      b = M[0][1],
      c = M[1][0],
      d = M[1][1],
      det = a * d - b * c || 1e-9;
    return [
      [d / det, -b / det],
      [-c / det, a / det],
    ];
  }
  predict(dt) {
    const F = [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    const a = this.Q_base,
      dt2 = dt * dt,
      dt3 = dt2 * dt,
      dt4 = dt2 * dt2;
    const q11 = 0.25 * dt4 * a,
      q13 = 0.5 * dt3 * a,
      q33 = dt2 * a;
    const Q = [
      [q11, 0, q13, 0],
      [0, q11, 0, q13],
      [q13, 0, q33, 0],
      [0, q13, 0, q33],
    ];
    const x = this.x;
    this.x = new Float64Array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]]);
    const P1 = this.mul(F, this.P),
      P2 = this.mul(P1, this.tr(F));
    this.P = this.add(P2, Q);
  }
  update(zx, zy) {
    const H = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ];
    const R = [
      [this.R_meas, 0],
      [0, this.R_meas],
    ];
    const z = [[zx], [zy]];
    const xmat = [[this.x[0]], [this.x[1]], [this.x[2]], [this.x[3]]];
    const y = this.sub(z, this.mul(H, xmat));
    const S = this.add(this.mul(this.mul(H, this.P), this.tr(H)), R);
    const Sinv = this.inv2(S);
    const K = this.mul(this.mul(this.P, this.tr(H)), Sinv);
    const Ky = this.mul(K, y);
    this.x[0] += Ky[0][0];
    this.x[1] += Ky[1][0];
    this.x[2] += Ky[2][0];
    this.x[3] += Ky[3][0];
    const I = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    const KH = this.mul(K, H),
      I_KH = this.sub(I, KH);
    this.P = this.mul(I_KH, this.P);
  }
}
const kf = new Kalman2D();

/* ========================
   파노라마 (Three.js)
   ======================== */
let renderer, scene3, camera3, mesh, panoTex;
let yaw = 0,
  pitch = 0,
  yawT = 0,
  pitchT = 0;
const PITCH_MIN = THREE.MathUtils.degToRad(-85);
const PITCH_MAX = THREE.MathUtils.degToRad(85);

function softClampPitch(t) {
  const soft = THREE.MathUtils.degToRad(10);
  if (t < PITCH_MIN)
    return PITCH_MIN + Math.tanh((t - PITCH_MIN) / soft) * soft;
  if (t > PITCH_MAX)
    return PITCH_MAX + Math.tanh((t - PITCH_MAX) / soft) * soft;
  return t;
}
function shortestAngleDelta(a, b) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function ensurePano() {
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  altView.appendChild(renderer.domElement);

  scene3 = new THREE.Scene();
  camera3 = new THREE.PerspectiveCamera(
    75,
    innerWidth / innerHeight,
    0.1,
    2000
  );
  camera3.position.set(0, 0, 0);

  const sphere = new THREE.SphereGeometry(500, 64, 48);
  sphere.scale(-1, 1, 1);
  const mat = new THREE.MeshBasicMaterial({ map: null });
  mesh = new THREE.Mesh(sphere, mat);
  scene3.add(mesh);

  addEventListener("resize", () => {
    renderer.setSize(innerWidth, innerHeight);
    camera3.aspect = innerWidth / innerHeight;
    camera3.updateProjectionMatrix();
  });
}

function loadPano(url) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        resolve(tex);
      },
      undefined,
      reject
    );
  });
}

/* ========================
   모드 전환 (+ 파노라마 진입시 재보정 트리거)
   ======================== */
function openPanoBySpot(spotId) {
  currentSpot = spotId;
  MODE = "PANO";
  document.body.classList.add("detail-open");
  titleImg.src = TITLE_MAP[spotId] || TITLE_MAP.main;
  backImg.style.display = "block";

  altView.classList.add("active");
  altView.setAttribute("aria-hidden", "false");

  // ✅ 파노라마 진입 시 캘리브레이션 다시 시작 (드리프트 억제)
  calibStart = performance.now();
  _pitchActive = false; // 히스테리시스 상태도 리셋
  // ✅ 파노라마 시점 초기화(항상 중앙에서 시작)
  yaw = 0;
  pitch = 0;
  yawT = 0;
  pitchT = 0;
  baseYaw = 0;
  basePitch = 0; // 기준도 초기화

  ensurePano();
  const url = PANO_MAP[spotId] || "assets/panos/default.jpg";
  loadPano(url)
    .then((tex) => {
      panoTex?.dispose?.();
      panoTex = tex;
      mesh.material.map = panoTex;
      mesh.material.needsUpdate = true;
      renderer.render(scene3, camera3);
    })
    .catch((err) => {
      console.error("Pano load failed:", err);
      alert(
        "파노라마 이미지를 불러오지 못했습니다. 파일 경로를 확인해 주세요."
      );
      closePano();
    });
}

function closePano() {
  MODE = "MAP";
  currentSpot = null;
  altView.classList.remove("active");
  altView.setAttribute("aria-hidden", "true");
  document.body.classList.remove("detail-open");
  titleImg.src = TITLE_MAP.main;
  backImg.style.display = "none";
}

/* 스팟 클릭 바인딩 */
Object.keys(PANO_MAP).forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("click", () => openPanoBySpot(id));
});
backImg.addEventListener("click", closePano);
altView.addEventListener("click", (e) => {
  if (e.target === altView) closePano();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePano();
});
// helpImg.addEventListener("click", () => {
//   /* TODO: 도움말 */
// });

/* ========================
   프레임 루프 (머리=스크롤/패닝/회전, 손=커서/클릭)
   ======================== */
let lastTS = -1,
  prevT = null,
  hadTipPrev = false;

async function frame() {
  if (!running) return;

  const ts = performance.now();
  if (!camVideo.videoWidth) {
    rafId = requestAnimationFrame(frame);
    return;
  }

  // 인퍼런스
  let fv = null,
    pv = null,
    gv = null;
  try {
    if (face && pose) {
      [fv, pv, gv] = await Promise.all([
        face.detectForVideo(camVideo, ts),
        pose.detectForVideo(camVideo, ts),
        lastTS % 3 === 0 && gesture
          ? gesture.recognizeForVideo(camVideo, ts)
          : Promise.resolve(null),
      ]);
    }
  } catch {}

  // Head 추정
  let yawHead = 0,
    pitchHead = 0;
  let haveMatrix = false;
  // if (fv?.faceLandmarks?.length) {
  //   const lm = fv.faceLandmarks[0];
  if (fv?.faceLandmarks?.length) {
    // 🔒 가장 먼저 잡힌 얼굴 인덱스를 고정해서 사용
    const idx = selectPrimaryFaceIndex(fv, ts);
    const lm = fv.faceLandmarks[idx];
    const leftEye = lm[33],
      rightEye = lm[263],
      nose = lm[1];
    const cx = (leftEye.x + rightEye.x) * 0.5;
    const cy = (leftEye.y + rightEye.y) * 0.5;
    const faceW =
      Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) + 1e-6;

    yawHead = ((nose.x - cx) / faceW) * 10.0;
    pitchHead = ((nose.y - cy) / faceW) * 8.0;
    const fallbackYaw = yawHead,
      fallbackPitch = pitchHead; // 좌표 기반 폴백

    try {
      // const m = fv.facialTransformationMatrixes?.[0]?.data;
      const m = fv.facialTransformationMatrixes?.[idx]?.data;
      if (m && m.length >= 16) {
        const yawRad = Math.atan2(m[2], m[10]);
        const pitchRad = Math.asin(-m[6]);
        const matYaw = (yawRad * 180) / Math.PI / 25;
        const matPitch = (pitchRad * 180) / Math.PI / 20;
        // 특정 방향(좌/상)에서 거의 0으로 수축할 때 소량 폴백을 혼합
        yawHead =
          Math.abs(matYaw) < 0.05 ? 0.8 * matYaw + 0.2 * fallbackYaw : matYaw;
        pitchHead =
          Math.abs(matPitch) < 0.05
            ? 0.8 * matPitch + 0.2 * fallbackPitch
            : matPitch;
        haveMatrix = true;
      }
    } catch {}
    // 얼굴이 검출됨 → 타임스탬프 갱신
    lastFaceSeenAt = ts;
    if (!facePresent) {
      // 재등장 후보 시작
      if (candidateFaceStart === 0) candidateFaceStart = ts;
      // 충분히 안정적으로 보였을 때 새 관람객으로 간주하고 재보정 시작
      if (ts - candidateFaceStart >= FACE_STABLE_MS) {
        facePresent = true;
        calibStart = ts; // 재보정 윈도우 재시작
        _pitchActive = false; // 히스테리시스 리셋
        baseYaw = 0;
        basePitch = 0; // 기준 초기화(윈도우 동안 다시 적응)
      }
    }
  } else {
    // 얼굴이 안 보이는 프레임
    if (facePresent && ts - lastFaceSeenAt >= FACE_GONE_MS) {
      facePresent = false;
      candidateFaceStart = 0;
      lastFaceGoneAt = ts;
    }
  }
  // 얼굴 장시간 미검출 시 face-lock 해제
maybeUnlockPrimaryFace(ts);

  if (!calibStart) calibStart = ts;
  if (ts - calibStart < CALIB_MS) {
    baseYaw = baseYaw + 0.15 * (yawHead - baseYaw);
    basePitch = basePitch + 0.15 * (pitchHead - basePitch);
  }
  // 행렬이 없는 경우(안경/마스크로 불안정)에는 제어를 중지하여 드리프트 방지
  if (!haveMatrix) {
    yawS = yawS * 0.9;
    pitchS = pitchS * 0.9;
  }
  yawHead -= baseYaw;
  pitchHead -= basePitch;
  yawS = yawS + SMOOTH * (yawHead - yawS);
  pitchS = pitchS + SMOOTH * (pitchHead - pitchS);

  // 손 → 커서 좌표
  let tip = null;
  const handIdx = chooseHandIndex(gv, ts);
  if (handIdx != null && gv?.landmarks?.[handIdx]) {
    const h = gv.landmarks[handIdx];
    if (h?.length > 8) tip = h[8]; // index tip
  }
  if (!tip && pv?.landmarks?.length) {
    const wristIdx = lastHandSide === "Left" ? 15 : 16;
    tip = pv.landmarks[0][wristIdx];
  }
  if (tip) {
    // const rawX = (1 - tip.x) * innerWidth;
    // const rawY = tip.y * innerHeight;
    const EDGE_PAD = 0.12; // 손 좌표가 대략 0.15~0.85쯤만 쓰는 현실 반영 (원하면 0.12~0.2 사이로 튜닝)

    function expand01(u) {
      const v = (u - EDGE_PAD) / (1 - 2 * EDGE_PAD);
      return Math.min(1, Math.max(0, v)); // 0~1로 클램프
    }

    const normX = expand01(1 - tip.x); // 좌우 반전 유지
    const normY = expand01(tip.y);
    const rawX = normX * innerWidth;
    const rawY = normY * innerHeight;
    pushBuf(bufX, rawX);
    pushBuf(bufY, rawY);
    const zx = median(bufX),
      zy = median(bufY);
    const dt = Math.max(1 / 120, prevT ? (ts - prevT) / 1000 : 1 / 60);
    kf.predict(dt);
    kf.update(zx, zy);
    prevT = ts;
    updateCursor(kf.x[0], kf.x[1], !hadTipPrev);
    hadTipPrev = true;
  } else hadTipPrev = false;

  /* ===== 데드존/히스테리시스/컨트롤 ===== */

  /* 1) yaw 데드존 (기존과 동일) */
  const overY = Math.max(0, Math.abs(yawS) - DEAD_Y);

  /* 2) 먼저 컨트롤 값을 '선언'해서 아래에서 참조 가능하게 */
  const yawCtl = YAW_DIR * yawS;
  const pitchCtl = PITCH_DIR * pitchS;

  /* 3) pitch 히스테리시스 */
  if (_pitchActive) {
    // 활성 상태에선 충분히 안쪽으로 들어오면 비활성
    if (Math.abs(pitchS) < DEAD_P_LO * 0.9) _pitchActive = false;
  } else {
    // 비활성 상태에선 높은 임계를 넘으면 활성
    if (Math.abs(pitchS) > DEAD_P_HI) _pitchActive = true;
  }

  /* 방향을 바꾸려고 할 땐 언제든 재활성화 (끝단에서 걸리는 현상 방지) */
  if (Math.sign(pitchCtl) !== Math.sign(pitchT - pitch)) {
    // pitchT는 목표, pitch는 현재 → 반대쪽으로 움직이려는 의도 감지
    _pitchActive = true;
  }

  /* 4) 유효 pitch 과잉량 */
  const overP = _pitchActive ? Math.max(0, Math.abs(pitchS) - DEAD_P_LO) : 0;

  const SPD = 20;
  //   const yawCtl   = YAW_DIR   * yawS;
  //   const pitchCtl = PITCH_DIR * pitchS;

  if (MODE === "MAP") {
    // 지도 패닝(가로/세로): 스테이지를 translate로 이동
    const scale = computeScale();
    const viewW = innerWidth,
      viewH = innerHeight;
    const stageW = 3840 * scale,
      stageH = 2160 * scale;
    const maxX = Math.max(0, (stageW - viewW) / 2);
    const maxY = Math.max(0, (stageH - viewH) / 2);

    let dx = 0,
      dy = 0;
    if (overY > 0)
      dx = Math.sign(yawCtl) * (0.6 + Math.pow(overY, 1.25)) * (SPD * 0.8);
    if (overP > 0) dy = Math.sign(pitchCtl) * overP * (SPD * 0.8);

    const m = stage._pan || { x: 0, y: 0 };
    m.x = THREE.MathUtils.clamp(m.x + dx, -maxX, maxX);
    m.y = THREE.MathUtils.clamp(m.y + dy, -maxY, maxY);
    stage._pan = m;

    stage.style.transform = `translate(${m.x}px, ${m.y}px)`;
    document.body.classList.add("yaw-active");
    clearTimeout(document.body._t);
    document.body._t = setTimeout(
      () => document.body.classList.remove("yaw-active"),
      120
    );
  } else if (MODE === "PANO") {
    // 파노라마 회전: 체감 방향 일치
    const sens = 1.2;
    const kBase = 0.0025 * sens * (SPD / 20);
    // 얼굴이 존재하고 캘리브 기간 이후에만 제어 적용(드리프트 억제)
    const controlEnabled = facePresent && ts - calibStart >= CALIB_MS;
    if (controlEnabled) {
      // 좌우: 마스크/안경에서 좌측 인식이 약해지는 문제 보완 (바이어스 제거, 속도 상향)
      if (overY > 0) {
        const yawSign = Math.sign(yawCtl);
        const yawSpeed = (0.3 + Math.pow(overY, 1.1)) * kBase * 18; // 더 빠르게
        yawT += yawSign * yawSpeed;
      }
      // 상하: 너무 빠르다는 피드백 → 속도 하향, 시작 임계 완화
      if (overP > 0) {
        const pSign = Math.sign(pitchCtl);
        const pSpeed = overP * kBase * 18; // 느리게
        pitchT += pSign * pSpeed;
      }
    }
    pitchT = softClampPitch(pitchT);

    // 카메라 적용/렌더
    if (renderer) {
      const s = 0.12; // 관성
      yaw += shortestAngleDelta(yaw, yawT) * s;
      pitch += (pitchT - pitch) * s;

      const quat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(pitch, yaw, 0, "YXZ")
      );
      camera3.quaternion.copy(quat);

      if (Math.abs(yawT) > 1000) {
        const n = Math.floor(yawT / (Math.PI * 2));
        const offset = n * (Math.PI * 2);
        yaw -= offset;
        yawT -= offset;
      }
      renderer.render(scene3, camera3);
    }
  }

  // 주먹-클릭
  const { name: gName, score } = readGesture(gv, handIdx);
  const becameFist =
    lastHandGesture !== "Closed_Fist" &&
    gName === "Closed_Fist" &&
    score >= MIN_CONF;
  if (becameFist) clickAtCursor(_cx, _cy);
  lastHandGesture = score >= 0.5 ? gName : "None";

  lastTS++;
  rafId = requestAnimationFrame(frame);
}

/* ========================
   부트스트랩
   ======================== */
(async function start() {
  try {
    await openCam();
    await loadModels();
    running = true;
    requestAnimationFrame(frame);
  } catch (err) {
    console.error("초기화 실패:", err);
    alert(
      "카메라/모델 초기화에 실패했습니다. HTTPS 또는 localhost 환경을 확인하세요."
    );
  }
})();
