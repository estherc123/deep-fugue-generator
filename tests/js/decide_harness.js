// Drives the pairs page's decide() under stubbed DOM/fetch and reports whether a
// click produced a save, and whether the user was told anything when it did not.
//
// The failure this guards: a click that silently does nothing. alert() was the
// only signal when the reviewer field was empty, and a browser that suppresses
// repeated alerts turns every click into a no-op with no feedback at all.
const fs = require('fs'), vm = require('vm');
const js = fs.readFileSync(process.argv[2], 'utf8');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const mode = process.argv[4] || 'named';        // named | anonymous | serverfail

let posted = null, alerted = null, rejections = [];
process.on('unhandledRejection', e => rejections.push(String(e && e.message || e)));

const store = {};
function el(id) {
  if (id && store[id]) return store[id];
  const n = {
    style: {}, dataset: {}, children: [], id: id || '', className: '', value: '',
    textContent: '', innerHTML: '', disabled: false,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild(c) { this.children.push(c); return c; },
    querySelectorAll() { return []; }, querySelector() { return el(); },
    addEventListener() {}, focus() {}, remove() {}, setAttribute() {},
    getBoundingClientRect() { return { bottom: 0 }; },
  };
  if (id) store[id] = n;
  return n;
}
const audio = {
  currentTime: 0, destination: {},
  createOscillator() { return { type: '', frequency: { value: 0 }, connect() {}, start() {}, stop() {} }; },
  createGain() { return { gain: { setValueAtTime() {}, linearRampToValueAtTime() {}, setTargetAtTime() {} }, connect() {} }; },
};
const sandbox = {
  console,
  document: {
    getElementById: id => el(id), createElement: () => el(),
    querySelectorAll: () => [], querySelector: () => el(),
    addEventListener() {}, body: el(),
  },
  fetch: async (url, opts) => {
    if (opts && opts.method === 'POST') {
      posted = { url, body: opts.body };
      if (mode === 'serverfail') {
        return { ok: false, status: 500, text: async () => 'boom', json: async () => ({}) };
      }
      return { ok: true, status: 200, text: async () => '{}', json: async () => ({ ok: true }) };
    }
    return { ok: true, status: 200, json: async () => data };
  },
  alert: m => { alerted = m; },
  localStorage: { getItem: () => null, setItem() {} },
  setTimeout, clearTimeout, Math, JSON, Date, Promise, encodeURIComponent,
};
sandbox.window = sandbox;
sandbox.AudioContext = function () { return audio; };
sandbox.globalThis = sandbox;

const ctx = vm.createContext(sandbox);
try { vm.runInContext(js, ctx, { timeout: 5000 }); }
catch (e) { console.error('EVAL_ERROR: ' + e.message); process.exit(2); }

setTimeout(async () => {
  if (mode !== 'anonymous') el('reviewer').value = 'tester';
  if (typeof ctx.decide !== 'function') { console.error('NO_DECIDE'); process.exit(3); }
  try { await ctx.decide(0, 'left'); }
  catch (e) { console.error('DECIDE_THREW: ' + e.message); process.exit(4); }
  // Any visible signal: the warning banner, or the save indicator.
  const warn = store['warn'] ? store['warn'].style.display : 'none';
  const saved = store['saved'] ? store['saved'].textContent : '';
  console.log(JSON.stringify({
    posted: posted ? JSON.parse(posted.body).label : null,
    feedback: (warn === 'block') || !!saved || !!alerted,
    saved_text: saved, rejections,
  }));
}, 300);
