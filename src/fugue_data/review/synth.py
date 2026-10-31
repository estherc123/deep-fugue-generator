"""The Web Audio synth shared by every review page.

It lived in three copies, which is how a regex repair once mangled one
AudioContext constructor while leaving the others correct, and how a
per-voice timbre fix had to be applied three times by hand. One copy now;
`export_audio.py` mirrors these same numbers for offline WAV rendering and
must be changed with it.

Per-voice timbre is keyed to the voice, not to pitch height, so a line does
not change instrument as it crosses another. The lowest voice carries
partials an octave and two octaves up because its fundamental falls below
what a laptop speaker reproduces -- the ear infers the pitch from the
partials.
"""

SYNTH_JS = r"""
// Synth state lives with the synth: play() closes over both.
let AC=null, CUR=null;

function voiceParams(v, nv, gain){
  const fromBottom = (nv - 1) - v;          // 0 = lowest voice
  const isBass = fromBottom === 0;
  const isLower = fromBottom <= 1;
  return {
    gain: gain * (isBass ? 1.15 : isLower ? 1.0 : 1.0),
    oct:  isBass ? 0.95 : isLower ? 0.40 : 0.0,
    oct2: isBass ? 0.55 : isLower ? 0.18 : 0.0,
    // A slower attack on low voices removes the click that made them sound rough.
    atk:  isBass ? 0.040 : isLower ? 0.030 : 0.018,
  };
}
function play(notes, secPerQuarter, gain){
  AC = AC || new (window.AudioContext||window.webkitAudioContext)();
  if(CUR) CUR.forEach(o=>{try{o.stop()}catch(e){}});
  CUR = [];
  const nv = 1 + notes.reduce((m,n)=>Math.max(m,n.v), 0);
  const t0 = AC.currentTime + 0.06;
  notes.forEach(n=>{
    const P = voiceParams(n.v, nv, gain);
    const f = 440*Math.pow(2,(n.m-69)/12);
    const st = t0 + n.t*secPerQuarter, du = Math.max(0.10, n.d*secPerQuarter);
    const mk = (freq, amp, type) => {
      const o = AC.createOscillator(), g = AC.createGain();
      o.type = type; o.frequency.value = freq;
      g.gain.setValueAtTime(0, st);
      g.gain.linearRampToValueAtTime(amp, st + P.atk);
      g.gain.setTargetAtTime(0.0001, st + du*0.8, 0.10);
      o.connect(g); g.connect(AC.destination);
      o.start(st); o.stop(st + du + 0.4); CUR.push(o);
    };
    mk(f, P.gain, 'triangle');
    if(P.oct  > 0) mk(f*2, P.gain*P.oct,  'triangle');
    if(P.oct2 > 0) mk(f*4, P.gain*P.oct2, 'triangle');
  });
}
"""
