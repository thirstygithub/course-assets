// Boots Façade: the recompiled engine (engine.wasm) and, when the engine
// launches java.exe the way the original did, the Java AI on CheerpJ with its
// native methods bound to the recompiled nativeaniminterface.dll.
import { EngineHost } from './host.js';

const $ = s => document.querySelector(s);
const params = new URLSearchParams(location.search);
const logLines = [];
window.facadeLog = logLines;

// Debug aid: sample the main thread for `ms` (JS Self-Profiling API; the dev
// server sends the Document-Policy it needs) and list where the time went.
window.profile = async (ms = 5000) => {
  const p = new Profiler({ sampleInterval: 1, maxBufferSize: 200000 });
  await new Promise(r => setTimeout(r, ms));
  const t = await p.stop();
  const name = i => {
    const f = t.frames[i];
    return f ? `${f.name || '(anon)'}${f.resourceId !== undefined ? ' ' + (t.resources[f.resourceId] || '').split('/').pop() : ''}` : '?';
  };
  const self = new Map(), total = new Map();
  let idle = 0;
  for (const s of t.samples) {
    if (s.stackId === undefined) { idle++; continue; }
    let st = t.stacks[s.stackId];
    self.set(name(st.frameId), (self.get(name(st.frameId)) || 0) + 1);
    const seen = new Set();
    for (; st; st = st.parentId === undefined ? null : t.stacks[st.parentId]) {
      const n = name(st.frameId);
      if (!seen.has(n)) { seen.add(n); total.set(n, (total.get(n) || 0) + 1); }
    }
  }
  const top = m => [...m].sort((a, b) => b[1] - a[1]).slice(0, 25).map(([n, c]) => `${(100 * c / t.samples.length).toFixed(1)}% ${n}`);
  return { samples: t.samples.length, idle: `${(100 * idle / t.samples.length).toFixed(1)}%`, self: top(self), total: top(total) };
};

function log(line, cls) {
  logLines.push(line);
  if (logLines.length > 5000) logLines.splice(0, 1000);
  if (cls === 'err' || params.has('log')) console[cls === 'err' ? 'error' : 'log'](line);
}

function status(text) { $('#status').textContent = text; }

let javaStarted = false;

async function bootJava(host) {
  if (javaStarted) return;
  javaStarted = true;
  status('Starting the Java AI…');
  const natives = await fetch('app/natives.json').then(r => r.json());
  await cheerpjInit({
    version: 8,
    enableDebug: true,
    overrideDocumentBase: new URL('https://cdn.jsdelivr.net/gh/thirstygithub/course-assets@main/', location.href).href,
    // ?dump=ms: print each agent's status and pending steps periodically
    // (facade.web.Launcher) to the Java console
    // ?trace=N: print the first N lines of ABL's joint goal negotiation trace
    // (on by default with ?dump)
    javaProperties: [
      ...(params.has('dump') ? [`facade.threadDumpMs=${Number(params.get('dump')) || 15000}`] : []),
      ...(params.has('trace') || params.has('dump') ? [`facade.ablTrace=${Number(params.get('trace')) || 600}`] : []),
    ],
    natives: {
      ...host.java.natives(natives),
      // Missing from CheerpJ's runtime; Class.newInstance() uses it to rethrow
      // exceptions raised by reflectively invoked constructors.
      Java_sun_misc_Unsafe_throwException(lib, self, throwable) { throw throwable; },
    },
  });
  status('');
  const code = await cheerpjRunMain('facade.web.Launcher', '/app/app/facade.jar.bin', '/app/facade', '/app/app/datafiles.txt');
  log(`[java] exited with ${code}`, code ? 'err' : undefined);
}

async function start() {
  $('#start').disabled = true;
  status('Loading the engine…');
  const canvas = $('#screen');
  const host = new EngineHost({
    canvas,
    log,
    onShellExecute: () => bootJava(host).catch(e => { log(String(e.stack || e), 'err'); status(`Java failed: ${e}`); }),
  });
  window.engine = host;
  // Timestamped record of runtime failures (CheerpJ's own errors land in the
  // console without times, and the console keeps them across reloads).
  const failure = what => log(`[${(performance.now() / 1000).toFixed(1)}s] ${what} (natives ${host.java.nativeCalls}, jni ${host.java.calls})`, 'err');
  addEventListener('error', e => failure(`error: ${e.message}`));
  addEventListener('unhandledrejection', e => failure(`unhandled: ${e.reason?.stack || e.reason}`));
  // ?probe[=frames]: run the snapshot-probe build and capture the probed
  // function once the engine has drawn that many frames (tools/replay);
  // ?probe=0 captures from the start, initialization included.
  if (params.has('probe')) {
    host.wasmName = 'engine-probe.wasm';
    const after = params.get('probe') === '0' ? 0 : Number(params.get('probe')) || 300;
    const addrs = (params.get('addrs') || '').split(',').filter(Boolean).map(a => parseInt(a, 16));
    if (after === 0) host.armProbe(params.get('name') || 'snap', addrs);
    else {
      const iv = setInterval(() => { if (host.frames >= after) { clearInterval(iv); host.armProbe(params.get('name') || 'snap', addrs); } }, 100);
    }
  }
  host.audio();
  await host.load('engine/');
  host.attachInput();
  $('#overlay').remove();
  canvas.focus();
  status('');
  try {
    await host.start();
  } catch (e) {
    log(String(e.stack || e), 'err');
    status(`Engine stopped: ${e.message}`);
    window.engineError = String(e);
  }
}

$('#start').addEventListener('click', () => start());
