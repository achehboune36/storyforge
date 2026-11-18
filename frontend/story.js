// Reader. Streams prose into Text nodes and opens image placeholders in place.
//
// Two rules govern everything here:
//
//   1. Prose appends to a Text node, never `innerHTML +=`. The latter reparses
//      the paragraph on every chunk, which destroys the user's text selection
//      and reflows mid-sentence.
//
//   2. An image figure is created at full size the instant its slot opens,
//      before generation starts. Because its box never changes size, the image
//      landing cannot shove text around under the reader's eyes. That single
//      detail is what makes "reading is never blocked" true rather than
//      merely claimed.

import { StoryWS } from '/ws.js';

const sessionId = new URLSearchParams(location.search).get('s');
if (!sessionId) location.href = '/';

const storyEl = document.getElementById('story');
const actionEl = document.getElementById('action');
const sendEl = document.getElementById('send');
const passEl = document.getElementById('pass');
const undoEl = document.getElementById('undo');
const statusEl = document.getElementById('status');
const connEl = document.getElementById('conn');
const modelEl = document.getElementById('modelpick');

const figures = new Map();   // image_id -> <figure>
let turnEl = null;           // current narrator turn container
let paraEl = null;           // current <p>
let textNode = null;         // live Text node being appended to
let streaming = false;

// ── scroll ──────────────────────────────────────────────────────────────
// Only follow the text if the reader is already at the bottom. Someone who
// scrolled up to re-read a paragraph must not be yanked forward.
const nearBottom = () =>
  window.innerHeight + window.scrollY >= document.body.scrollHeight - 120;
function follow() { if (nearBottom()) window.scrollTo(0, document.body.scrollHeight); }

// ── building blocks ─────────────────────────────────────────────────────

function newTurn(cls) {
  const s = document.createElement('section');
  s.className = 'turn' + (cls ? ' ' + cls : '');
  storyEl.append(s);
  return s;
}

function startPara(container) {
  paraEl = document.createElement('p');
  textNode = document.createTextNode('');
  paraEl.append(textNode);
  container.append(paraEl);
}

function appendProse(container, text) {
  if (!paraEl) startPara(container);
  for (const [i, part] of text.split(/\n{2,}/).entries()) {
    if (i > 0) startPara(container);
    if (part) textNode.appendData(part);
  }
}

function makeFigure(imageId, w, h) {
  const fig = document.createElement('figure');
  fig.className = 'slot queued';
  fig.dataset.imageId = imageId;
  if (w && h) fig.style.aspectRatio = `${w} / ${h}`;

  const img = document.createElement('img');
  const badge = document.createElement('figcaption');
  badge.className = 'badge';
  badge.innerHTML = '<span class="label">queued</span><span class="bar"><i></i></span>';
  fig.append(img, badge);
  figures.set(imageId, fig);
  return fig;
}

function setBadge(fig, label, pct) {
  const l = fig.querySelector('.label');
  if (l) l.textContent = label;
  const bar = fig.querySelector('.bar i');
  if (bar && pct != null) bar.style.width = pct + '%';
}

// ── replay ──────────────────────────────────────────────────────────────

function render(payload) {
  storyEl.innerHTML = '';
  figures.clear();
  turnEl = paraEl = textNode = null;

  let currentTurn = null, section = null;
  for (const seg of payload.segments || []) {
    if (seg.turn_id !== currentTurn) {
      currentTurn = seg.turn_id;
      section = newTurn(seg.role === 'user' ? 'action' : '');
      paraEl = textNode = null;
    }
    if (!seg.kind) continue;
    if (seg.kind === 'prose') {
      appendProse(section, seg.text || '');
      paraEl = textNode = null;
    } else if (seg.kind === 'image') {
      const info = (payload.images || {})[seg.image_id] || {};
      const fig = makeFigure(seg.image_id, info.width, info.height);
      section.append(fig);
      applyImageState(seg.image_id, info);
      paraEl = textNode = null;
    }
  }
  window.scrollTo(0, document.body.scrollHeight);
}

function applyImageState(imageId, info) {
  const fig = figures.get(imageId);
  if (!fig) return;
  if (info.status === 'ready' && info.url) {
    showFinal(fig, info.url);
  } else if (info.status === 'failed') {
    fig.className = 'slot failed';
    setBadge(fig, 'failed');
    addRetry(fig, imageId, info.error);
  } else {
    fig.className = 'slot queued';
    setBadge(fig, info.status || 'queued');
  }
}

function showFinal(fig, url) {
  const img = fig.querySelector('img');
  const pre = new Image();
  pre.onload = () => {
    // The render is the authority on its own shape — the narrator picks a
    // frame per image, so never trust the placeholder's guess once the
    // real picture is in hand.
    if (pre.naturalWidth && pre.naturalHeight) {
      fig.style.aspectRatio = `${pre.naturalWidth} / ${pre.naturalHeight}`;
    }
    img.src = url;
    img.classList.remove('preview');
    img.classList.add('shown');
    fig.className = 'slot ready';
    fig.querySelector('.badge')?.remove();
  };
  pre.src = url;
}

function addRetry(fig, imageId, error) {
  const badge = fig.querySelector('.badge');
  if (!badge || badge.querySelector('.retry')) return;
  const b = document.createElement('button');
  b.className = 'retry';
  b.textContent = 'retry';
  b.title = error || '';
  b.onclick = () => {
    ws.send('retry_image', { image_id: imageId });
    fig.className = 'slot queued';
    setBadge(fig, 'queued');
    b.remove();
  };
  badge.append(b);
}

// ── image model ─────────────────────────────────────────────────────────
// The switch is global and takes effect on the next picture: images already
// drawn keep the model that drew them, which is what the image row records.

async function loadModels() {
  try {
    const cat = await fetch('/api/models').then(r => r.json());
    modelEl.innerHTML = '';
    for (const m of cat.models) {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = m.label;
      o.title = [m.notes, `cfg ${m.cfg} · ${m.steps} steps`].filter(Boolean).join(' — ');
      o.selected = m.name === cat.active;
      modelEl.append(o);
    }
  } catch { modelEl.style.display = 'none'; }
}

modelEl.onchange = async () => {
  const prev = statusEl.textContent;
  statusEl.textContent = 'switching model…';
  try {
    const r = await fetch(`/api/models/active?name=${encodeURIComponent(modelEl.value)}`,
                          { method: 'POST' });
    if (!r.ok) throw new Error();
    const d = await r.json();
    statusEl.textContent = `next picture: ${d.model.label || d.active}`;
    setTimeout(() => { if (!streaming) statusEl.textContent = ''; }, 4000);
  } catch {
    statusEl.textContent = 'could not switch model';
    loadModels();
  }
};

// ── socket ──────────────────────────────────────────────────────────────

const ws = new StoryWS(sessionId, '', {
  onOpen() { connEl.textContent = ''; },
  onClosed() { connEl.textContent = 'reconnecting…'; },
  onRejected(r) { connEl.textContent = 'disconnected: ' + (r || ''); },

  hello() {},

  replay(d) { render(d); setBusy(d.busy); },

  turn_start(d) {
    if (d.passing) {
      // No player line to show — mark the gap instead, so the reader can see
      // where they chose to step back rather than it looking like a jump.
      const s = newTurn('passed');
      s.textContent = 'you let the moment pass';
    } else {
      const s = newTurn('action');
      s.textContent = d.action;
    }
    turnEl = newTurn('');
    paraEl = textNode = null;
    setBusy(true);
    follow();
  },

  prose(d) {
    if (!turnEl) { turnEl = newTurn(''); paraEl = textNode = null; }
    appendProse(turnEl, d.text);
    follow();
  },

  image_slot(d) {
    if (!turnEl) turnEl = newTurn('');
    turnEl.append(makeFigure(d.image_id, d.width, d.height));
    paraEl = textNode = null;
    follow();
  },

  image_state(d) {
    const fig = figures.get(d.image_id);
    if (!fig) return;
    if (d.status === 'queued') { fig.className = 'slot queued'; setBadge(fig, 'queued'); }
    if (d.status === 'running') { fig.className = 'slot running'; setBadge(fig, 'drawing', 0); }
  },

  image_progress(d) {
    const fig = figures.get(d.image_id);
    if (fig) setBadge(fig, `drawing ${d.pct}%`, d.pct);
  },

  // latent2rgb previews are abstract colour fields; blurring them reads as
  // "developing" rather than "broken", and sharpens as the steps advance.
  image_preview(d) {
    const fig = figures.get(d.image_id);
    if (!fig || fig.classList.contains('ready')) return;
    const img = fig.querySelector('img');
    img.src = d.data_url;
    img.classList.add('preview', 'shown');
  },

  image_ready(d) {
    const fig = figures.get(d.image_id);
    if (fig) showFinal(fig, d.url);
  },

  // The second pass finished minutes after the fact. Same picture, sharper:
  // swap it in quietly, no badge, no scroll.
  warning(d) {
    statusEl.textContent = d.message || 'something went wrong';
  },

  image_hires_ready(d) {
    const fig = figures.get(d.image_id);
    if (fig) showFinal(fig, d.url);
  },

  image_failed(d) {
    const fig = figures.get(d.image_id);
    if (!fig) return;
    fig.className = 'slot failed';
    setBadge(fig, 'failed');
    addRetry(fig, d.image_id, d.error);
  },

  // Someone switched the model from the lab or another tab.
  model_changed(d) {
    if (d.name) modelEl.value = d.name;
  },

  turn_end() { setBusy(false); follow(); },

  error(d) {
    statusEl.textContent = d.message || d.code;
    if (d.code !== 'busy') setBusy(false);
  },
});

// ── composer ────────────────────────────────────────────────────────────

function setBusy(on) {
  streaming = !!on;
  // Only the LLM stream gates input. Images generating in the background must
  // never stop the reader from acting — that is the whole premise.
  sendEl.disabled = streaming;
  passEl.disabled = streaming;
  undoEl.disabled = streaming;
  statusEl.innerHTML = streaming
    ? '<span class="dot">●</span> the story is being written…'
    : '';
}

function submit() {
  const text = actionEl.value.trim();
  if (!text || streaming) return;
  if (!ws.send('action', { text })) { statusEl.textContent = 'not connected'; return; }
  actionEl.value = '';
  actionEl.style.height = 'auto';
}

sendEl.onclick = submit;

// Undo: drop the last exchange and re-render from storage, so the reader can
// take the scene somewhere else. Deletes rather than hides — a hidden turn
// would still be replayed into the model's context and keep steering.
undoEl.onclick = async () => {
  if (streaming) return;
  undoEl.disabled = true;
  statusEl.textContent = 'undoing…';
  try {
    const r = await fetch(`/api/sessions/${sessionId}/rollback`, { method: 'POST' });
    if (!r.ok) throw new Error(await r.text());
    render(await r.json());
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = 'undo failed';
  } finally { undoEl.disabled = false; }
};

// Pass: the reader has nothing to add and wants the world to move on its own.
// This is the only place the narrator is allowed to advance the story itself.
passEl.onclick = () => {
  if (streaming) return;
  if (!ws.send('pass', {})) { statusEl.textContent = 'not connected'; return; }
  setBusy(true);
};
actionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
actionEl.addEventListener('input', () => {
  actionEl.style.height = 'auto';
  actionEl.style.height = Math.min(actionEl.scrollHeight, 144) + 'px';
});

loadModels();

fetch(`/api/sessions/${sessionId}`)
  .then(r => r.ok ? r.json() : Promise.reject())
  .then(d => { render(d); setBusy(d.busy); })
  .catch(() => { storyEl.innerHTML = '<p class="sub">Session not found.</p>'; })
  .finally(() => ws.connect());
