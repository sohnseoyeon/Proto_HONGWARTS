// main.js
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
const $ = (s)=>document.querySelector(s);
const header    = $("#glass-header");
const titleImg  = $("#title-img");
const backImg   = $("#back-btn-img");
const helpImg   = $("#question-btn-img");
const scene     = $("#scene");
const app       = $("#app");
const stage     = $("#stage");
const altView   = $("#alt-view");
const camVideo  = $("#cam");
const cursor    = $("#cursor");

/* 타이틀 매핑 */
const TITLE_MAP = {
  main:"./assets/titles/main_title.png",
  "r3-btn":"./assets/titles/r3_title.png",
  "k4l4-btn":"./assets/titles/k4l4_title.png",
  "l4g4-btn":"./assets/titles/l4g4_title.png",
  "p3-btn":"./assets/titles/p3_title.png",
  "g2q1-btn":"./assets/titles/g2q1_title.png",
  "q3f2-btn":"./assets/titles/q3f2_title.png",
  "f2e3-btn":"./assets/titles/f2e3_title.png",
  "f4u2-btn":"./assets/titles/f4u2_title.png",
  "ub2c3-btn":"./assets/titles/ub2c3_title.png",
  "z23z33-btn":"./assets/titles/z23z33_title.png",
  "c3af-btn":"./assets/titles/c3af_title.png",
  "c3-btn":"./assets/titles/c3_title.png",
};

/* 파노라마 원본 매핑 */
const PANO_MAP = {
  "r3-btn":"./assets/views/r3_view.png",
  "k4l4-btn":"./assets/views/k4l4_view.png",
  "l4g4-btn":"./assets/views/l4g4_view.png",
  "p3-btn":"./assets/views/p3_view.png",
  "g2q1-btn":"./assets/views/g2q1_view.png",
  "q3f2-btn":"./assets/views/q3f2_view.png",
  "f2e3-btn":"./assets/views/f2e3_view.png",
  "f4u2-btn":"./assets/views/f4u2_view.png",
  "ub2c3-btn":"./assets/views/ub2c3_view.png",
  "z23z33-btn":"./assets/views/z23z33_view.png",
  "c3af-btn":"./assets/views/c3af_view.png",
  "c3-btn":"./assets/views/c3_view.png",
};

/* (2) 썸네일 매핑: 파일 없으면 자동으로 무시됨 */
const PANO_THUMB = {
  "r3-btn":"./assets/views/r3_view.thumb.webp",
  "k4l4-btn":"./assets/views/k4l4_view.thumb.webp",
  "l4g4-btn":"./assets/views/l4g4_view.thumb.webp",
  "p3-btn":"./assets/views/p3_view.thumb.webp",
  "g2q1-btn":"./assets/views/g2q1_view.thumb.webp",
  "q3f2-btn":"./assets/views/q3f2_view.thumb.webp",
  "f2e3-btn":"./assets/views/f2e3_view.thumb.webp",
  "f4u2-btn":"./assets/views/f4u2_view.thumb.webp",
  "ub2c3-btn":"./assets/views/ub2c3_view.thumb.webp",
  "z23z33-btn":"./assets/views/z23z33_view.thumb.webp",
  "c3af-btn":"./assets/views/c3af_view.thumb.webp",
  "c3-btn":"./assets/views/c3_view.thumb.webp",
};

/* 비율 스케일 유지 */
function computeScale(){
  const vw = innerWidth;
  const vh = innerHeight;
  return Math.min(vw/3840, vh/2160);
}
function applyScale(){
  document.documentElement.style.setProperty('--scale', computeScale());
}
addEventListener('resize', applyScale);
addEventListener('orientationchange', applyScale);
applyScale();

/* ========================
   제스처 커서 / 공통 상태
   ======================== */
let running = true, rafId = null;
let MODE = "MAP";        // "MAP" | "PANO"
let currentSpot = null;

/* 커서 보간 */
let _cxS = innerWidth * 0.5, _cyS = innerHeight * 0.5;
let _cx = _cxS, _cy = _cyS;
const MAX_STEP = 100;
const LERP_MOVE = 0.5;

/* 중앙값 버퍼 */
const MEDIAN_WINDOW = 1;
const bufX = [], bufY = [];
function pushBuf(buf, v){ buf.push(v); if(buf.length>MEDIAN_WINDOW) buf.shift(); }
function median(buf){ return buf.length ? buf[buf.length-1] : null; }

/* 플래시 */
function flash(x,y){
  const f = document.createElement("div");
  Object.assign(f.style,{
    position:"fixed", left:`${x}px`, top:`${y}px`, width:"12px", height:"12px",
    borderRadius:"50%", transform:"translate(-50%,-50%)",
    background:"rgba(255,60,60,.9)", boxShadow:"0 0 0 10px rgba(255,60,60,.25)",
    pointerEvents:"none", zIndex:3000
  });
  document.body.appendChild(f);
  setTimeout(()=>f.remove(), 220);
}

/* 커서 이동 */
function updateCursor(tx, ty, snap = false) {
  const W = innerWidth, H = innerHeight;
  const R = (cursor?.offsetWidth || 24) / 2;
  tx = Math.max(R, Math.min(W - R, tx));
  ty = Math.max(R, Math.min(H - R, ty));

  if (snap) {
    _cxS = tx; _cyS = ty;
  } else {
    const dx = tx - _cxS, dy = ty - _cyS;
    const dist = Math.hypot(dx, dy);
    if (dist > MAX_STEP) {
      const r = MAX_STEP / dist;
      tx = _cxS + dx * r;
      ty = _cyS + dy * r;
    }
    _cxS = _cxS + (tx - _cxS) * LERP_MOVE;
    _cyS = _cyS + (ty - _cyS) * LERP_MOVE;
  }
  _cx = _cxS; _cy = _cyS;
  cursor.style.left = _cx + "px";
  cursor.style.top = _cy + "px";
  cursor.hidden = false;

  // 손 커서 hover 감지
  const el = document.elementFromPoint(_cx, _cy);
  const allBtns = document.querySelectorAll('#scene image[id$="-btn"]');
  allBtns.forEach(btn => btn.classList.remove('hovered'));
  if (el && el.matches && el.matches('#scene image[id$="-btn"]')) {
    el.classList.add('hovered');
  }
}

/* 클릭 합성 */
let lastClickAt = 0;
const CLICK_COOLDOWN = 500;
function clickAtCursor(x,y){
  const now = performance.now();
  if(now - lastClickAt < CLICK_COOLDOWN) return false;
  const target = document.elementFromPoint(x|0,y|0);
  if(!target) return false;

  const common={bubbles:true,cancelable:true,clientX:x,clientY:y,view:window};
  try {
    target.dispatchEvent(new PointerEvent("pointerdown",{...common,pointerId:1,isPrimary:true,buttons:1}));
    target.dispatchEvent(new PointerEvent("pointerup",{...common,pointerId:1,isPrimary:true,buttons:0}));
  } catch {}
  target.dispatchEvent(new MouseEvent("mousedown",{...common,button:0,buttons:1}));
  target.dispatchEvent(new MouseEvent("mouseup",{...common,button:0,buttons:0}));
  target.dispatchEvent(new MouseEvent("click",{...common,button:0}));

  flash(x,y);
  cursor.classList.add("clicking");
  setTimeout(()=>cursor.classList.remove("clicking"), 180);
  lastClickAt = now;
  return true;
}

/* ========================
   MediaPipe (머리/손)
   ======================== */
let fileset, face, pose, gesture;
const YAW_DIR = -1;
const PITCH_DIR = 1;

let baseYaw = 0, basePitch = 0, yawS = 0, pitchS = 0;
const SMOOTH = 0.3, CALIB_MS = 1200;
let calibStart = performance.now();

let lastHandIdx = null, lastHandSide = 'Right', lastHandSeenAt = 0;
const HAND_HYST_MS = 350, MIN_SWITCH_DELTA = 0.12;
let lastHandGesture = "None";
const MIN_CONF = 0.65;

/* pitch 히스테리시스 */
let _pitchActive = false;
const DEAD_Y = 0.6;
const DEAD_P_LO = 0.60;
const DEAD_P_HI = DEAD_P_LO + 0.12;

async function openCam(){
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode:"user", width:{ideal:1280}, height:{ideal:720} }, audio:false
  });
  camVideo.srcObject = stream;
  await camVideo.play();
  camVideo.style.transform = "scaleX(-1)";
  cursor.hidden = false;
}

async function loadModels(){
  fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  face = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO", numFaces: 1,
  });
  pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO", numPoses: 1,
  });
  gesture = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    },
    runningMode: "VIDEO", numHands: 2,
  });
}

function chooseHandIndex(gv, now){
  try{
    const arr = gv?.handedness ?? [];
    if(!arr.length) return lastHandIdx;
    let bestIdx=lastHandIdx, bestScore=-1, bestSide=lastHandSide;
    for(let i=0;i<arr.length;i++){
      const h=arr[i]?.[0], sc=h?.score ?? -1;
      if(sc > bestScore){ bestScore=sc; bestIdx=i; bestSide=h?.categoryName || bestSide; }
    }
    if(lastHandIdx!=null && (now - lastHandSeenAt) < HAND_HYST_MS){
      const prev=arr[lastHandIdx]?.[0], prevScore=prev?.score ?? -1;
      if(prevScore>=0 && (bestScore - prevScore) < MIN_SWITCH_DELTA) return lastHandIdx;
    }
    lastHandIdx=bestIdx; lastHandSide=bestSide; lastHandSeenAt=now; return bestIdx;
  }catch{ return lastHandIdx; }
}
function readGesture(gv, idx){
  try{
    if(idx!=null && gv?.gestures?.[idx]?.length){
      const top = gv.gestures[idx][0];
      return { name: top?.categoryName || 'None', score: top?.score ?? 0 };
    }
    const g0 = gv?.gestures?.[0]?.[0];
    return { name: g0?.categoryName || 'None', score: 0 };
  }catch{ return { name:'None', score:0 }; }
}

/* 2D Kalman */
class Kalman2D {
  constructor(){
    this.x = new Float64Array([_cxS, _cyS, 0, 0]);
    this.P = this.eye(4, 400);
    this.Q_base = 8.0;
    this.R_meas = 20.0;
  }
  eye(n, s=1){ const M = Array.from({length:n},(_,i)=>Array(n).fill(0)); for(let i=0;i<n;i++) M[i][i]=s; return M; }
  mul(A,B){ const r=A.length,c=B[0].length,n=B.length,R=Array.from({length:r},()=>Array(c).fill(0)); for(let i=0;i<r;i++) for(let k=0;k<n;k++){ const v=A[i][k]; for(let j=0;j<c;j++) R[i][j]+=v*B[k][j]; } return R; }
  add(A,B){ return A.map((r,i)=>r.map((v,j)=>v+B[i][j])); }
  sub(A,B){ return A.map((r,i)=>r.map((v,j)=>v-B[i][j])); }
  tr(A){ return A[0].map((_,i)=>A.map(r=>r[i])); }
  inv2(M){ const a=M[0][0],b=M[0][1],c=M[1][0],d=M[1][1],det=a*d-b*c||1e-9; return [[d/det,-b/det],[-c/det,a/det]]; }
  predict(dt){
    const F=[[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]];
    const a=this.Q_base, dt2=dt*dt, dt3=dt2*dt, dt4=dt2*dt2;
    const q11=0.25*dt4*a, q13=0.5*dt3*a, q33=dt2*a;
    const Q=[[q11,0,q13,0],[0,q11,0,q13],[q13,0,q33,0],[0,q13,0,q33]];
    const x=this.x;
    this.x = new Float64Array([x[0]+x[2]*dt, x[1]+x[3]*dt, x[2], x[3]]);
    const P1=this.mul(F,this.P), P2=this.mul(P1,this.tr(F));
    this.P = this.add(P2,Q);
  }
  update(zx,zy){
    const H=[[1,0,0,0],[0,1,0,0]];
    const R=[[this.R_meas,0],[0,this.R_meas]];
    const z=[[zx],[zy]];
    const xmat=[[this.x[0]],[this.x[1]],[this.x[2]],[this.x[3]]];
    const y=this.sub(z,this.mul(H,xmat));
    const S=this.add(this.mul(this.mul(H,this.P),this.tr(H)),R);
    const Sinv=this.inv2(S);
    const K=this.mul(this.mul(this.P,this.tr(H)),Sinv);
    const Ky=this.mul(K,y);
    this.x[0]+=Ky[0][0]; this.x[1]+=Ky[1][0]; this.x[2]+=Ky[2][0]; this.x[3]+=Ky[3][0];
    const I=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    const KH=this.mul(K,H), I_KH=this.sub(I,KH);
    this.P=this.mul(I_KH,this.P);
  }
}
const kf = new Kalman2D();

/* ========================
   파노라마 (Three.js)
   ======================== */
let renderer, scene3, camera3, mesh, panoTex;
let yaw=0, pitch=0, yawT=0, pitchT=0;
const PITCH_MIN = THREE.MathUtils.degToRad(-85);
const PITCH_MAX = THREE.MathUtils.degToRad( 85);

function softClampPitch(t){
  const soft = THREE.MathUtils.degToRad(10);
  if (t < PITCH_MIN) return PITCH_MIN + Math.tanh((t - PITCH_MIN)/soft)*soft;
  if (t > PITCH_MAX) return PITCH_MAX + Math.tanh((t - PITCH_MAX)/soft)*soft;
  return t;
}
function shortestAngleDelta(a,b){
  let d = (b - a) % (Math.PI*2);
  if (d >  Math.PI) d -= Math.PI*2;
  if (d < -Math.PI) d += Math.PI*2;
  return d;
}

function ensurePano(){
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); // 성능 캡
  renderer.setSize(innerWidth, innerHeight);
  altView.appendChild(renderer.domElement);

  scene3 = new THREE.Scene();
  camera3 = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.1, 2000);
  camera3.position.set(0,0,0);

  const sphere = new THREE.SphereGeometry(500,64,48);
  sphere.scale(-1,1,1);
  const mat = new THREE.MeshBasicMaterial({ map: null });
  mesh = new THREE.Mesh(sphere, mat);
  scene3.add(mesh);

  addEventListener('resize', ()=>{
    renderer.setSize(innerWidth, innerHeight);
    camera3.aspect = innerWidth/innerHeight;
    camera3.updateProjectionMatrix();
  });
}

/* ========================
   텍스처 로딩 최적화 (레이스 방지 + 캐시 + 썸네일 스왑)
   ======================== */
let _panoSeq = 0;            // 항상 "마지막 클릭"만 유효하게
let _inflight = null;        // 진행 중 식별
const texCache = new Map();  // URL -> THREE.Texture

const manager = new THREE.LoadingManager();
const loader  = new THREE.TextureLoader(manager);

function getTextureCached(url){
  return new Promise((resolve, reject)=>{
    // 썸네일 파일이 실제로 없을 수도 있으니 fetch로 가볍게 존재 확인
    // (CORS 허용된 동일 오리진 전제. 외부 CDN이면 생략 가능)
    if (!url) return reject(new Error("no url"));
    const hit = texCache.get(url);
    if (hit && hit.image && hit.image.complete) return resolve(hit);

    loader.load(url, tex=>{
      tex.colorSpace = THREE.SRGBColorSpace;
      texCache.set(url, tex);
      resolve(tex);
    }, undefined, err=>reject(err));
  });
}

async function loadPanoLatest(url){
  const mySeq = ++_panoSeq;
  _inflight = mySeq;
  try{
    const tex = await getTextureCached(url);
    if (mySeq !== _panoSeq) return null; // 더 최신 클릭이 있으면 무시
    return tex;
  } finally {
    if (_inflight === mySeq) _inflight = null;
  }
}

function applyTexture(tex){
  if (!tex) return;
  panoTex?.dispose?.();
  panoTex = tex;
  mesh.material.map = panoTex;
  mesh.material.needsUpdate = true;
  renderer.render(scene3, camera3); // 첫 프레임 즉시
}

/* ========================
   모드 전환
   ======================== */
function openPanoBySpot(spotId){
  currentSpot = spotId;
  MODE = "PANO";
  document.body.classList.add('detail-open');
  titleImg.src = TITLE_MAP[spotId] || TITLE_MAP.main;
  backImg.style.display = 'block';

  altView.classList.add('active');
  altView.setAttribute('aria-hidden','false');

  // 파노라마 진입 시 재보정
  calibStart = performance.now();
  _pitchActive = false;

  ensurePano();

  // 1) 썸네일 즉시 적용 (있을 경우)
  const thumbUrl = PANO_THUMB[spotId];
  if (thumbUrl) {
    getTextureCached(thumbUrl).then(tex=>{
      // 썸네일은 레이스 무시(빠른 피드백용)
      if (MODE === "PANO" && currentSpot === spotId) applyTexture(tex);
    }).catch(()=>{ /* 썸네일 없거나 실패해도 무시 */ });
  }

  // 2) 원본 로드: 항상 "마지막"만 적용
  const url = PANO_MAP[spotId] || "assets/panos/default.jpg";
  altView.style.cursor = 'progress';
  loadPanoLatest(url)
    .then(tex => {
      if (tex) applyTexture(tex);
    })
    .catch(err => {
      console.error("Pano load failed:", err);
      alert("파노라마 이미지를 불러오지 못했습니다. 파일 경로를 확인해 주세요.");
      closePano();
    })
    .finally(() => {
      altView.style.cursor = (_inflight ? 'progress' : 'default');
    });
}

function closePano(){
  MODE = "MAP";
  currentSpot = null;
  altView.classList.remove('active');
  altView.setAttribute('aria-hidden','true');
  document.body.classList.remove('detail-open');
  titleImg.src = TITLE_MAP.main;
  backImg.style.display = 'none';
}

/* 스팟 클릭 바인딩 (로딩 중 연타 방지) */
Object.keys(PANO_MAP).forEach(id=>{
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', ()=>{
    if (_inflight) return; // 로딩 중이면 무시 (원하면 큐잉 로직으로 변경)
    openPanoBySpot(id);
  });
});
backImg.addEventListener('click', closePano);
altView.addEventListener('click', e=>{ if(e.target===altView) closePano(); });
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closePano(); });
helpImg.addEventListener('click', ()=>{/* TODO: 도움말 */});

/* ========================
   프레임 루프 (머리=패닝/회전, 손=커서/클릭)
   ======================== */
let lastTS = -1, prevT = null, hadTipPrev = false;

async function frame(){
  if(!running) return;

  const ts = performance.now();
  if(!camVideo.videoWidth){
    rafId = requestAnimationFrame(frame); return;
  }

  // 인퍼런스
  let fv=null, pv=null, gv=null;
  try{
    if(face && pose){
      [fv, pv, gv] = await Promise.all([
        face.detectForVideo(camVideo, ts),
        pose.detectForVideo(camVideo, ts),
        (lastTS % 3 === 0 && gesture) ? gesture.recognizeForVideo(camVideo, ts) : Promise.resolve(null),
      ]);
    }
  }catch{}

  // Head 추정
  let yawHead=0, pitchHead=0;
  if (fv?.faceLandmarks?.length){
    const lm = fv.faceLandmarks[0];
    const leftEye = lm[33], rightEye = lm[263], nose = lm[1];
    const cx = (leftEye.x + rightEye.x) * 0.5;
    const cy = (leftEye.y + rightEye.y) * 0.5;
    const faceW = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) + 1e-6;

    yawHead   = ((nose.x - cx) / faceW) * 10.0;
    pitchHead = ((nose.y - cy) / faceW) * 8.0;

    try {
      const m = fv.facialTransformationMatrixes?.[0]?.data;
      if (m && m.length >= 16) {
        const yawRad = Math.atan2(m[2], m[10]);
        const pitchRad = Math.asin(-m[6]);
        yawHead   = yawHead   * 0.8 + (yawRad * 180/Math.PI / 25) * 0.2;
        pitchHead = pitchHead * 0.8 + (pitchRad * 180/Math.PI / 20) * 0.2;
      }
    } catch {}
  }

  if(!calibStart) calibStart = ts;
  if(ts - calibStart < CALIB_MS){
    baseYaw   = baseYaw   + 0.15*(yawHead - baseYaw);
    basePitch = basePitch + 0.15*(pitchHead - basePitch);
  }
  yawHead   -= baseYaw;
  pitchHead -= basePitch;
  yawS   = yawS   + SMOOTH*(yawHead - yawS);
  pitchS = pitchS + SMOOTH*(pitchHead - pitchS);

  // 손 → 커서 좌표
  let tip = null;
  const handIdx = chooseHandIndex(gv, ts);
  if (handIdx != null && gv?.landmarks?.[handIdx]) {
    const h = gv.landmarks[handIdx];
    if (h?.length > 8) tip = h[8]; // index tip
  }
  if (!tip && pv?.landmarks?.length){
    const wristIdx = (lastHandSide === 'Left') ? 15 : 16;
    tip = pv.landmarks[0][wristIdx];
  }
  if(tip){
    const rawX = (1 - tip.x) * innerWidth;
    const rawY = tip.y * innerHeight;
    pushBuf(bufX, rawX); pushBuf(bufY, rawY);
    const zx = median(bufX), zy = median(bufY);
    const dt = Math.max(1/120, prevT ? (ts - prevT)/1000 : 1/60);
    kf.predict(dt); kf.update(zx, zy); prevT = ts;
    updateCursor(kf.x[0], kf.x[1], !hadTipPrev);
    hadTipPrev = true;
  } else hadTipPrev = false;

  /* 데드존/히스테리시스/컨트롤 */
  const overY = Math.max(0, Math.abs(yawS) - DEAD_Y);
  const yawCtl   = YAW_DIR   * yawS;
  const pitchCtl = PITCH_DIR * pitchS;

  if (_pitchActive) {
    if (Math.abs(pitchS) < DEAD_P_LO * 0.90) _pitchActive = false;
  } else {
    if (Math.abs(pitchS) > DEAD_P_HI) _pitchActive = true;
  }
  if (Math.sign(pitchCtl) !== Math.sign(pitchT - pitch)) _pitchActive = true;
  const overP = _pitchActive ? Math.max(0, Math.abs(pitchS) - DEAD_P_LO) : 0;

  const SPD = 20;

  if (MODE === "MAP"){
    // 지도 패닝
    const scale = computeScale();
    const viewW = innerWidth, viewH = innerHeight;
    const stageW = 3840*scale, stageH = 2160*scale;
    const maxX = Math.max(0, (stageW - viewW)/2);
    const maxY = Math.max(0, (stageH - viewH)/2);

    let dx = 0, dy = 0;
    if (overY > 0) dx = Math.sign(yawCtl)   * (0.6 + Math.pow(overY,1.25)) * (SPD * 0.8);
    if (overP > 0) dy = Math.sign(pitchCtl) * overP * (SPD * 0.8);

    const m = stage._pan || { x:0, y:0 };
    m.x = THREE.MathUtils.clamp(m.x + dx, -maxX, maxX);
    m.y = THREE.MathUtils.clamp(m.y + dy, -maxY, maxY);
    stage._pan = m;

    stage.style.transform = `translate(${m.x}px, ${m.y}px)`;
    document.body.classList.add('yaw-active');
    clearTimeout(document.body._t);
    document.body._t = setTimeout(()=>document.body.classList.remove('yaw-active'),120);
  } else if (MODE === "PANO"){
    // 파노라마 회전
    const sens = 1.2;
    const kBase = 0.0025 * sens * (SPD / 20);
    if (overY > 0) yawT   -= Math.sign(yawCtl)   * (0.6 + Math.pow(overY,1.25)) * kBase * 5;
    if (overP > 0) pitchT -= Math.sign(pitchCtl) * overP * kBase * 30;
    pitchT = softClampPitch(pitchT);

    if (renderer){
      const s = 0.12; // 관성
      yaw   += shortestAngleDelta(yaw, yawT) * s;
      pitch += (pitchT - pitch) * s;

      const quat = new THREE.Quaternion()
        .setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
      camera3.quaternion.copy(quat);

      if (Math.abs(yawT) > 1000){
        const n = Math.floor(yawT / (Math.PI*2));
        const offset = n * (Math.PI*2);
        yaw   -= offset; yawT -= offset;
      }
      renderer.render(scene3, camera3);
    }
  }

  // 주먹-클릭
  const { name:gName, score } = readGesture(gv, handIdx);
  const becameFist =
    (lastHandGesture !== "Closed_Fist") &&
    (gName === "Closed_Fist") && (score >= MIN_CONF);
  if (becameFist) clickAtCursor(_cx, _cy);
  lastHandGesture = (score >= 0.5 ? gName : "None");

  lastTS++;
  rafId = requestAnimationFrame(frame);
}

/* ========================
   부트스트랩
   ======================== */
(async function start(){
  try{
    // (선택) 서비스워커: 반복 방문 가속 (sw.js 있으면)
    if ('serviceWorker' in navigator) {
      try { navigator.serviceWorker.register('/sw.js'); } catch {}
    }

    await openCam();
    await loadModels();
    running = true;
    requestAnimationFrame(frame);

    // 유휴 시간에 썸네일 사전 로딩 (4개만 예시)
    function preloadPanosSubset(limit = 4){
      const urls = Object.values(PANO_THUMB).slice(0, limit).filter(Boolean);
      for (const url of urls) getTextureCached(url).catch(()=>{});
    }
    if ('requestIdleCallback' in window) {
      requestIdleCallback(()=>preloadPanosSubset(4), { timeout: 1500 });
    } else {
      setTimeout(()=>preloadPanosSubset(4), 1200);
    }
  }catch(err){
    console.error("초기화 실패:", err);
    alert("카메라/모델 초기화에 실패했습니다. HTTPS 또는 localhost 환경을 확인하세요.");
  }
})();
