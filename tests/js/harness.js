// Executes a review page's <script> with stubbed DOM / fetch / Web Audio, then
// calls play() with real note data and reports what the synth scheduled.
//
// `node --check` only parses. It could not see that play() referenced an
// undeclared AC, nor that a caller passed BPM into a seconds-per-quarter
// parameter -- both of which shipped a page that looked fine and played nothing.
// This actually runs the code.
const fs = require('fs');
const vm = require('vm');

const js = fs.readFileSync(process.argv[2], 'utf8');
const notes = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

let oscillators = 0;
const stops = [];

function el() {
  const n = {
    style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    textContent: '', innerHTML: '', id: '', className: '', value: '',
    appendChild(c) { this.children.push(c); return c; },
    querySelectorAll() { return []; },
    querySelector() { return el(); },
    addEventListener() {}, focus() {}, scrollIntoView() {}, remove() {},
    setAttribute() {}, getAttribute() { return null; },
  };
  return n;
}

const audio = {
  currentTime: 0,
  destination: {},
  createOscillator() {
    oscillators++;
    return {
      type: '', frequency: { value: 0 }, connect() {},
      start() {}, stop(t) { stops.push(t); },
    };
  },
  createGain() {
    return {
      gain: {
        value: 0, setValueAtTime() {}, linearRampToValueAtTime() {},
        setTargetAtTime() {},
      },
      connect() {},
    };
  },
};

const sandbox = {
  console,
  document: {
    getElementById: () => el(),
    createElement: () => el(),
    querySelectorAll: () => [],
    querySelector: () => el(),
    addEventListener: () => {},
    body: el(),
  },
  fetch: async () => ({ json: async () => ({ items: [], verdicts: {} }) }),
  setTimeout, clearTimeout, Math, JSON, Date, Promise,
};
sandbox.window = sandbox;
sandbox.AudioContext = function () { return audio; };
sandbox.globalThis = sandbox;

const ctx = vm.createContext(sandbox);
try {
  vm.runInContext(js, ctx, { timeout: 5000 });
} catch (e) {
  console.error('EVAL_ERROR: ' + e.message);
  process.exit(2);
}

if (typeof ctx.play !== 'function') {
  console.error('NO_PLAY: page defines no play()');
  process.exit(3);
}
try {
  // 0.55 s per quarter is what every real caller passes.
  ctx.play(notes, 0.55, 0.16);
} catch (e) {
  console.error('PLAY_ERROR: ' + e.message);
  process.exit(4);
}

const span = stops.length ? Math.max(...stops) : 0;
console.log(JSON.stringify({ oscillators, span_seconds: Number(span.toFixed(2)) }));
