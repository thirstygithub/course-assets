// Host for the statically recompiled animEngineDLL (engine.wasm).
// Provides the JavaScript half of the platform layer and runs the DLL's own
// entry points: _DllMainCRTStartup, then AnimEngineWinMain.
//
// Blocking Windows calls (Sleep, GetMessage, SwapBuffers, file opens) suspend
// the wasm stack through JSPI, so the engine's original main loop runs
// unchanged and the page stays responsive.
import { GLEmu } from './gl.js';
import { JavaBridge } from './jni.js';

const ASSET_ROOT = 'facade/';             // = util/sources/facade, the engine's working dir
const GAME_DIR = ['util', 'sources', 'facade'];
const PENDING = -2;                         // JS_PENDING in recomp/runtime/hle.h

// DirectInput scan code for each Windows virtual key (US layout).
const SCAN_TO_VK = [
  0, 0x1b, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x30, 0xbd, 0xbb, 0x08, 0x09,
  0x51, 0x57, 0x45, 0x52, 0x54, 0x59, 0x55, 0x49, 0x4f, 0x50, 0xdb, 0xdd, 0x0d, 0x11, 0x41, 0x53,
  0x44, 0x46, 0x47, 0x48, 0x4a, 0x4b, 0x4c, 0xba, 0xde, 0xc0, 0x10, 0xdc, 0x5a, 0x58, 0x43, 0x56,
  0x42, 0x4e, 0x4d, 0xbc, 0xbe, 0xbf, 0x10, 0x6a, 0x12, 0x20, 0x14, 0x70, 0x71, 0x72, 0x73, 0x74,
  0x75, 0x76, 0x77, 0x78, 0x79, 0x90, 0x91, 0x24, 0x26, 0x21, 0x6d, 0x25, 0x0c, 0x27, 0x6b, 0x23,
  0x28, 0x22, 0x2d, 0x2e,
];
const VK_TO_SCAN = new Map();
SCAN_TO_VK.forEach((vk, scan) => { if (vk && !VK_TO_SCAN.has(vk)) VK_TO_SCAN.set(vk, scan); });
// Arrow keys and friends are extended keys: DirectInput reports them as 0xC8 etc.
const EXTENDED = { 0x26: 0xc8, 0x28: 0xd0, 0x25: 0xcb, 0x27: 0xcd, 0x24: 0xc7, 0x23: 0xcf, 0x21: 0xc9, 0x22: 0xd1, 0x2d: 0xd2, 0x2e: 0xd3 };
const scanOf = vk => EXTENDED[vk] ?? VK_TO_SCAN.get(vk) ?? 0;

export class EngineHost {
  constructor({ canvas, log = console.log, onShellExecute } = {}) {
    this.canvas = canvas;
    this.log = log;
    this.onShellExecute = onShellExecute;
    this.files = new Map();     // id -> { path, data: Uint8Array, size }
    this.nextFile = 1;
    this.sync = 0;   // nesting depth of callThreadSync
    this.syncWaiters = new Set();
    this.cache = new Map();     // url -> Uint8Array
    this.decoder = new TextDecoder('latin1');
    this.messages = [];         // [msg, wParam, lParam] from page input
    this.keys = new Uint8Array(256);    // by virtual key
    this.scans = new Uint8Array(256);   // by DirectInput scan code
    this.keyEvents = [];        // buffered DirectInput events
    this.wakeup = null;
    this.sounds = new Map();
    this.nextSound = 1;
    this.cursors = new Map();
    this.frames = 0;
    this.java = new JavaBridge(this);
  }

  get u8() { return new Uint8Array(this.memory.buffer); }
  get u16() { return new Uint16Array(this.memory.buffer); }

  cstr(p) {
    const u8 = this.u8;
    let e = p;
    while (u8[e]) e++;
    return this.decoder.decode(u8.subarray(p, e));
  }

  // Engine paths are Windows paths relative to util/sources/facade.
  resolve(winPath) {
    const parts = [...GAME_DIR];
    for (const seg of winPath.replace(/\\/g, '/').split('/')) {
      if (!seg || seg === '.') continue;
      if (/^[a-z]:$/i.test(seg)) { parts.length = 0; continue; }
      if (seg === '..') parts.pop(); else parts.push(seg);
    }
    const inGame = GAME_DIR.every((s, i) => parts[i]?.toLowerCase() === s);
    return inGame ? ASSET_ROOT + parts.slice(GAME_DIR.length).join('/') : null;
  }

  // ---- input from the page ----
  post(msg, w, l) {
    this.messages.push([msg, w >>> 0, l >>> 0]);
    const wake = this.wakeup;
    this.wakeup = null;
    wake?.();
  }

  attachInput(target = window) {
    const canvas = this.canvas;
    target.addEventListener('keydown', e => {
      const vk = e.keyCode & 0xff, scan = scanOf(vk);
      if (!this.keys[vk]) this.keyEvents.push((scan << 8) | 1);
      this.keys[vk] = 1; this.scans[scan] = 1;
      this.post(0x100, vk, (scan << 16) | 1);
      if (e.key.length === 1) this.post(0x102, e.key.charCodeAt(0), (scan << 16) | 1);
      else if (vk === 0x0d || vk === 0x08) this.post(0x102, vk, (scan << 16) | 1);
      if (vk === 0x08 || vk === 0x09 || (vk >= 0x21 && vk <= 0x28) || vk === 0x20) e.preventDefault();
    });
    target.addEventListener('keyup', e => {
      const vk = e.keyCode & 0xff, scan = scanOf(vk);
      this.keys[vk] = 0; this.scans[scan] = 0;
      this.keyEvents.push(scan << 8);
      this.post(0x101, vk, (scan << 16) | 0xc0000001);
    });
    target.addEventListener('blur', () => { this.keys.fill(0); this.scans.fill(0); });
    const pos = e => {
      const r = canvas.getBoundingClientRect();
      const x = Math.round((e.clientX - r.left) * canvas.width / r.width);
      const y = Math.round((e.clientY - r.top) * canvas.height / r.height);
      return ((y & 0xffff) << 16) | (x & 0xffff);
    };
    const buttons = e => (e.buttons & 1 ? 1 : 0) | (e.buttons & 2 ? 2 : 0);
    canvas.addEventListener('mousemove', e => this.post(0x200, buttons(e), pos(e)));
    canvas.addEventListener('mousedown', e => {
      this.keys[e.button === 2 ? 2 : 1] = 1;
      this.post(e.button === 2 ? 0x204 : 0x201, buttons(e), pos(e));
    });
    canvas.addEventListener('mouseup', e => {
      this.keys[e.button === 2 ? 2 : 1] = 0;
      this.post(e.button === 2 ? 0x205 : 0x202, buttons(e), pos(e));
    });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
  }

  // ---- audio (DirectSound buffers) ----
  audio() {
    if (!this.ac) this.ac = new AudioContext();
    if (this.ac.state === 'suspended') this.ac.resume();
    return this.ac;
  }

  // Music: the engine's DirectShow graphs (hle_media.c), each an <audio>.
  musicImports() {
    this.music ??= new Map();
    let nextId = 1;
    const get = id => this.music.get(id);
    return {
      js_music_open: wpath => {
        let path = '';
        for (let p = wpath; ; p += 2) {
          const c = this.u8[p] | (this.u8[p + 1] << 8);
          if (!c) break;
          path += String.fromCharCode(c);
        }
        const url = this.resolve(path);
        if (!url) { this.log(`[music] outside game dir: ${path}`); return 0; }
        const audio = new Audio(url);
        audio.preload = 'auto';
        const m = { audio, state: 0, complete: false, notify: null };
        audio.addEventListener('ended', () => {
          m.state = 0;
          m.complete = true;
          if (m.notify) this.post(m.notify.msg, 0, m.notify.param);   // EC_COMPLETE waiting
        });
        const id = nextId++;
        this.music.set(id, m);
        return id;
      },
      // ops: run, pause, stop, volume (1/100 dB), state, take EC_COMPLETE, close
      js_music_ctl: (id, op, value) => {
        const m = get(id);
        if (!m) return 0;
        switch (op) {
          case 0: m.state = 2; m.complete = false; m.audio.play().catch(e => this.log(`[music] ${e}`)); return 0;
          case 1: m.state = 1; m.audio.pause(); return 0;
          case 2: m.state = 0; m.audio.pause(); return 0;
          case 3: m.audio.volume = Math.min(1, Math.max(0, 10 ** (value / 2000))); return 0;
          case 4: return m.state;
          case 5: { const c = m.complete; m.complete = false; return c ? 1 : 0; }
          case 6: m.audio.pause(); m.audio.removeAttribute('src'); m.audio.load(); this.music.delete(id); return 0;
        }
        return 0;
      },
      js_music_time: (id, which) => {
        const a = get(id)?.audio;
        if (!a) return 0;
        return which ? (Number.isFinite(a.duration) ? a.duration : 0) : a.currentTime;
      },
      js_music_seek: (id, seconds) => { const a = get(id)?.audio; if (a) a.currentTime = Math.max(0, seconds); },
      js_music_notify: (id, hwnd, msg, param) => { const m = get(id); if (m) m.notify = hwnd ? { msg, param } : null; },
    };
  }

  soundImports() {
    const sound = id => this.sounds.get(id);
    return {
      js_sound_create: () => {
        const id = this.nextSound++;
        this.sounds.set(id, { buffer: null, src: null, volume: 1, pan: 0, rate: 1, loop: false, started: 0, bytesPerSec: 1, size: 0 });
        return id;
      },
      js_sound_data: (id, ptr, bytes, ch, rate, bits) => {
        const s = sound(id);
        if (!s || !bytes || !ch || !rate) return;
        const ac = this.audio(), frames = Math.floor(bytes / (ch * bits / 8));
        if (!frames) return;
        const buf = ac.createBuffer(ch, frames, rate);
        const mem = this.memory.buffer;
        if (bits === 16) {
          const pcm = new Int16Array(mem.slice(ptr, ptr + frames * ch * 2));
          for (let c = 0; c < ch; c++) {
            const out = buf.getChannelData(c);
            for (let i = 0; i < frames; i++) out[i] = pcm[i * ch + c] / 32768;
          }
        } else {
          const pcm = new Uint8Array(mem, ptr, frames * ch);
          for (let c = 0; c < ch; c++) {
            const out = buf.getChannelData(c);
            for (let i = 0; i < frames; i++) out[i] = (pcm[i * ch + c] - 128) / 128;
          }
        }
        s.buffer = buf;
        s.size = bytes;
        s.bytesPerSec = rate * ch * bits / 8;
        s.baseRate = rate;
      },
      js_sound_dup: id => {
        const s = sound(id), nid = this.nextSound++;
        this.sounds.set(nid, { ...s, src: null });
        return nid;
      },
      js_sound_play: (id, loop) => {
        const s = sound(id);
        if (!s?.buffer) return;
        if (s.src) { try { s.src.stop(); } catch {} }
        const ac = this.audio();
        const src = ac.createBufferSource();
        src.buffer = s.buffer;
        src.loop = !!loop;
        src.playbackRate.value = s.rate;
        const gain = ac.createGain();
        gain.gain.value = s.volume;
        const pan = ac.createStereoPanner();
        pan.pan.value = s.pan;
        src.connect(gain).connect(pan).connect(ac.destination);
        src.onended = () => { if (s.src === src) s.src = null; };
        src.start();
        Object.assign(s, { src, gainNode: gain, panNode: pan, loop: !!loop, started: ac.currentTime });
      },
      js_sound_stop: id => {
        const s = sound(id);
        if (s?.src) { const src = s.src; s.src = null; try { src.stop(); } catch {} }
      },
      js_sound_status: id => {
        const s = sound(id);
        return s?.src ? (1 | (s.loop ? 4 : 0)) : 0;
      },
      js_sound_set: (id, what, v) => {
        const s = sound(id);
        if (!s) return;
        if (what === 0) { s.volume = v <= -10000 ? 0 : Math.pow(10, v / 2000); if (s.gainNode) s.gainNode.gain.value = s.volume; }
        if (what === 1) { s.pan = Math.max(-1, Math.min(1, v / 10000)); if (s.panNode) s.panNode.pan.value = s.pan; }
        if (what === 2 && s.baseRate) { s.rate = v ? v / s.baseRate : 1; if (s.src) s.src.playbackRate.value = s.rate; }
      },
      js_sound_pos: id => {
        const s = sound(id);
        if (!s?.src || !this.ac) return 0;
        const pos = Math.floor((this.ac.currentTime - s.started) * s.rate * s.bytesPerSec);
        return s.size ? pos % s.size : 0;
      },
      js_sound_release: id => {
        const s = sound(id);
        if (s?.src) try { s.src.stop(); } catch {}
        this.sounds.delete(id);
      },
    };
  }

  imports() {
    const S = fn => new WebAssembly.Suspending(fn);
    const env = {
      js_log: (p, n) => this.log(this.decoder.decode(this.u8.subarray(p, p + n)).replace(/\s+$/, '')),
      js_now_ms: () => performance.now(),
      js_math: (op, a, b) => [Math.sin, Math.cos, Math.tan, Math.atan2, Math.pow][op](a, b),
      js_trap: (addr, msg) => {
        throw new Error(`engine trap at 0x${(addr >>> 0).toString(16)}: ${this.cstr(msg)}`);
      },
      js_unimplemented: (dll, name) => {
        throw new Error(`unimplemented import ${this.cstr(dll)}!${this.cstr(name)}`);
      },
      // Imports that may wait return PENDING and park the promise; the guest
      // then suspends in js_await (the only JSPI import besides the main
      // thread's message/frame waits). Inside callThreadSync they never wait.
      js_await: S(() => { const p = this.pending; this.pending = null; return p; }),
      js_sleep: ms => this.sync ? 0 : this.wait(new Promise(r => setTimeout(() => r(0), ms))),
      js_jni: (index, args, thread) => this.java.handle(index, args, thread),
      // Events, mutexes and critical sections (hle_kernel32.c): a guest
      // thread that has to wait parks here until something is signaled or
      // the timeout passes, then re-checks.
      js_sync_wait: ms => {
        if (this.sync) return -3;                  // SYNC_CANT_WAIT
        return this.wait(new Promise(r => {
          this.syncWaiters.add(r);
          if (ms !== 0xffffffff) setTimeout(() => { this.syncWaiters.delete(r); r(0); }, ms);
        }));
      },
      js_sync_notify: () => {
        const ws = [...this.syncWaiters];
        this.syncWaiters.clear();
        for (const r of ws) r(0);
      },
      js_probe: (addr, phase) => this.probe(addr, phase),
      js_local_time: (p, utc) => {
        const d = new Date();
        const v = utc
          ? [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDay(), d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds(), d.getUTCMilliseconds()]
          : [d.getFullYear(), d.getMonth() + 1, d.getDay(), d.getDate(), d.getHours(), d.getMinutes(), d.getSeconds(), d.getMilliseconds()];
        const dv = new DataView(this.memory.buffer);
        v.forEach((x, i) => dv.setUint16(p + 2 * i, x, true));
      },
      js_file_open: (p, write) => {
        const id = this.openFile(this.cstr(p), write);
        return id instanceof Promise ? this.wait(id) : id;
      },
      js_file_size: id => this.files.get(id)?.size ?? 0,
      js_file_read: (id, pos, dst, n) => {
        const f = this.files.get(id);
        if (!f) return 0;
        const k = Math.max(0, Math.min(n, f.size - pos));
        this.u8.set(f.data.subarray(pos, pos + k), dst);
        return k;
      },
      js_file_write: (id, pos, src, n) => {
        const f = this.files.get(id);
        if (!f) return;
        if (pos + n > f.data.length) {
          const grown = new Uint8Array(Math.max(pos + n, f.data.length * 2));
          grown.set(f.data);
          f.data = grown;
        }
        f.data.set(this.u8.subarray(src, src + n), pos);
        f.size = Math.max(f.size, pos + n);
      },
      js_file_close: id => {
        // The engine's own logs (path planning, errors) go to log files:
        // keep their text in the page log.
        const f = this.files.get(id);
        if (f?.write) {
          const text = this.decoder.decode(f.data.subarray(0, Math.min(f.size, 2000))).trim();
          if (text) this.log(`[${f.path.split(/[\\/]/).pop()}] ${text}`);
        }
        this.files.delete(id);
      },

      // window / input
      js_poll_message: m => {
        const msg = this.messages.shift();
        if (!msg) return 0;
        const dv = new DataView(this.memory.buffer);
        [0x2001, msg[0], msg[1], msg[2], performance.now() >>> 0, 0, 0].forEach((v, i) => dv.setUint32(m + 4 * i, v, true));
        return 1;
      },
      js_wait_message: S(() => new Promise(r => { this.wakeup = r; })),
      // A frame that took less than a display refresh waits for the next one;
      // a slower frame only yields to the browser (which shows the newest
      // frame at each refresh), instead of idling up to another whole
      // refresh: waiting every time made 20 ms frames run at 30 fps.
      js_swap_buffers: S(() => {
        this.gl.present();
        this.frames++;
        const took = performance.now() - (this.frameStart ?? 0);   // this frame's work
        const resume = r => () => { this.frameStart = performance.now(); r(); };
        if (took < 14) return new Promise(r => requestAnimationFrame(resume(r)));
        return new Promise(r => { this.yieldPort ??= this.makeYield(); this.yieldPort(resume(r)); });
      }),
      js_key_down: vk => this.keys[vk & 0xff],
      js_scan_down: scan => this.scans[scan & 0xff],
      js_next_key_event: () => this.keyEvents.shift() ?? 0,
      js_load_cursor: p => {
        const id = 0x4500 + this.cursors.size;
        const url = this.resolve(this.cstr(p));
        this.cursors.set(id, url ? `url("${url}"), auto` : 'auto');
        return id;
      },
      // Icon resources used as cursors: Windows puts an icon's hotspot at
      // its center. Converted to PNG, the cursor format every browser takes.
      js_load_icon: id => {
        const h = 0x4500 + this.cursors.size;
        this.cursors.set(h, 'default');
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas');
          c.width = img.width; c.height = img.height;
          c.getContext('2d').drawImage(img, 0, 0);
          this.cursors.set(h, `url("${c.toDataURL()}") ${img.width >> 1} ${img.height >> 1}, default`);
          if (this.cursor === h) this.canvas.style.cursor = this.cursors.get(h);
        };
        img.src = `${this.base}cursors/${id}.ico`;
        return h;
      },
      js_set_cursor: h => { this.cursor = h; this.canvas.style.cursor = this.cursors.get(h) || 'default'; },
      js_shell_execute: (file, params) => this.onShellExecute?.(this.cstr(file), params ? this.cstr(params) : ''),
      js_window_size: (w, h) => {
        this.log(`[engine] window ${w}x${h}`);
        this.canvas.width = w;
        this.canvas.height = h;
      },
      ...this.soundImports(),
      ...this.musicImports(),
    };
    return { env };
  }

  // ---- snapshot probes (tools/replay) ----
  // With an engine built with RECOMP_PROBES, armProbe(name) captures guest
  // memory and CPU state at the next entry to a probed function and at its
  // matching exit, then uploads both to the dev server (build/snapshots).
  // `addrs`: probed functions to capture (each once); empty = the first one hit.
  armProbe(name = 'snap', addrs = []) {
    this.probeArmed = { name, addrs: new Set(addrs), captures: new Map(), any: !addrs.length };
  }

  snapshot(addr) {
    const dv = new DataView(this.memory.buffer);
    const st = this.x.rt_probe_state();
    // probe_state (rt.c): regs[8], flags, top, fcw, fsw, df, pad, st[8] @0x38, regions[8] @0x78
    const cpu = new Uint8Array(this.memory.buffer.slice(st, st + 0x98));
    new DataView(cpu.buffer).setUint32(0x34, addr, true);   // the probed function, for the replayer
    const regions = [];
    // engine-process memory only: TEBs + engine stack, heap, VirtualAlloc area, ae + nai
    for (let i = 1; i < 4; i++) regions.push([dv.getUint32(st + 0x78 + 8 * i, true), dv.getUint32(st + 0x7c + 8 * i, true)]);
    for (const m of [this.modules.ae, this.modules.nai]) regions.push([m.base, m.base + m.size]);
    const parts = [new TextEncoder().encode('FSNP'), cpu];
    for (const [lo, hi] of regions) {
      if (hi <= lo) continue;
      const head = new DataView(new ArrayBuffer(8));
      head.setUint32(0, lo, true);
      head.setUint32(4, hi - lo, true);
      parts.push(new Uint8Array(head.buffer), new Uint8Array(this.memory.buffer.slice(lo, hi)));
    }
    return new Blob(parts);
  }

  probe(addr, phase) {
    const p = this.probeArmed;
    if (!p || (!p.any && !p.addrs.has(addr))) return;
    let c = p.captures.get(addr);
    if (!c) {
      if (phase !== 0) return;
      c = { depth: 0, pre: null, done: false };
      p.captures.set(addr, c);
    }
    if (c.done) return;
    if (phase === 0) {
      if (c.depth++ === 0) {
        c.pre = this.snapshot(addr);
        this.x.rt_trace_enable?.(1);      // traced builds record every instruction until exit
      }
      return;
    }
    if (--c.depth > 0) return;
    c.done = true;
    this.x.rt_trace_enable?.(0);
    const post = this.snapshot(addr);
    const tag = `${p.name}-${addr.toString(16)}`;
    const up = (suffix, blob) => fetch(`/__snapshot/${tag}-${suffix}.bin`, { method: 'POST', body: blob });
    const uploads = [up('pre', c.pre), up('post', post)];
    const n = this.x.rt_trace_count?.() ?? 0;
    if (n) {
      const buf = this.x.rt_trace_buf();
      uploads.push(up('trace', new Blob([this.memory.buffer.slice(buf, buf + n * 56)])));
    }
    // the imports' results, which the native replay feeds back (rt.c ImportRec)
    const ni = this.x.rt_import_count?.() ?? 0;
    if (ni) {
      const log = this.x.rt_import_log();
      uploads.push(up('imports', new Blob([this.memory.buffer.slice(log, log + ni * 16)])));
    }
    Promise.all(uploads)
      .then(() => this.log(`[probe] uploaded ${tag}`))
      .catch(e => this.log(`[probe] upload failed: ${e}`));
    c.pre = null;
    if (p.any) this.probeArmed = null;
  }

  // A zero-delay yield to the event loop (setTimeout(0) is clamped to 4 ms).
  makeYield() {
    const ch = new MessageChannel(), queue = [];
    ch.port1.onmessage = () => queue.shift()?.();
    return r => { queue.push(r); ch.port2.postMessage(0); };
  }

  // Parks `promise` for the guest's following js_await (see imports()).
  wait(promise) {
    this.pending = promise;
    return PENDING;
  }

  // Returns the file id, or a promise of it when the file has to be fetched
  // (synchronously instead inside callThreadSync).
  openFile(path, write) {
    const url = this.resolve(path);
    const open = data => {
      const id = this.nextFile++;
      this.files.set(id, { path, data, size: write ? 0 : data.length, write });
      return id;
    };
    // Logs and stageplays: kept in memory for this session.
    if (write) return open(new Uint8Array(4096));
    if (!url) { this.log(`[file] outside game dir: ${path}`); return -1; }
    const cached = this.cache.get(url);
    if (cached) return open(cached);
    const got = data => {
      if (!data) { this.log(`[file] not found: ${path}`); return -1; }
      this.cache.set(url, data);
      return open(data);
    };
    if (this.sync) {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.overrideMimeType('text/plain; charset=x-user-defined');
      xhr.send();
      if (xhr.status !== 200) return got(null);
      const s = xhr.responseText, data = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) data[i] = s.charCodeAt(i);
      return got(data);
    }
    return fetch(url).then(async r => got(r.ok ? new Uint8Array(await r.arrayBuffer()) : null));
  }

  async load(base = '') {
    this.base = base;
    const names = ['ae', 'nai', 'ae2', 'nai2'];
    const [wasm, ...mods] = await Promise.all([
      fetch(base + (this.wasmName ?? 'engine.wasm')).then(r => r.arrayBuffer()),
      ...names.map(async n => ({
        name: n,
        img: await fetch(base + n + '.img').then(r => r.arrayBuffer()),
        meta: await fetch(base + n + '.json').then(r => r.json()),
      })),
    ]);
    const imports = this.imports();
    // GL needs the memory, which only exists after instantiation; bind lazily.
    const glImports = {};
    const glNames = WebAssembly.Module.imports(await WebAssembly.compile(wasm))
      .filter(i => i.name.startsWith('gl_')).map(i => i.name);
    for (const n of glNames) glImports[n] = (...a) => this.glFns[n](...a);
    Object.assign(imports.env, glImports);
    const { instance } = await WebAssembly.instantiate(wasm, imports);
    this.x = instance.exports;
    this.memory = this.x.memory;
    this.gl = new GLEmu(this.canvas, this.memory);
    this.glFns = this.gl.imports(this);
    this.modules = Object.fromEntries(mods.map(m => [m.name, m.meta]));
    for (const m of mods) this.u8.set(new Uint8Array(m.img), m.meta.base);
    this.x.rt_capture_main_stack();
    this.x.rt_init();
    this.invoke = WebAssembly.promising(this.x.rt_invoke);
  }

  push(args) {
    for (let i = args.length - 1; i >= 0; i--) this.x.rt_push(args[i] >>> 0);
  }

  // Top-level call on the engine's main thread (args right to left).
  async callMain(addr, ...args) {
    this.x.rt_enter_main();
    this.push(args);
    return (await this.invoke(addr)) >>> 0;
  }

  // Call on a fresh guest thread: how the Java process's calls run. Returns
  // { eax, edx } (edx for 64-bit results).
  async callThread(addr, args, onThread) {
    const t = this.x.rt_thread_begin();
    onThread?.(t);
    this.x.rt_enter_shadow_stack(t);
    try {
      this.push(args);
      const eax = (await this.invoke(addr)) >>> 0;
      return { eax, edx: this.x.rt_edx() >>> 0 };
    } finally {
      this.x.rt_thread_end(t);
    }
  }

  // callThread without JSPI, for callers that must not wait (CheerpJ's async
  // natives leak until its runtime crashes). While it runs, the suspending
  // imports answer synchronously: Sleep returns at once and files are read
  // with a synchronous request.
  callThreadSync(addr, args, onThread) {
    const t = this.x.rt_thread_begin();
    onThread?.(t);
    this.x.rt_enter_shadow_stack(t);
    this.sync++;
    try {
      this.push(args);
      const eax = this.x.rt_invoke(addr) >>> 0;
      return { eax, edx: this.x.rt_edx() >>> 0 };
    } finally {
      this.sync--;
      this.x.rt_thread_end(t);
    }
  }

  async start() {
    const { ae, nai, ae2, nai2 } = this.modules;
    // Engine process: the loader initializes nativeaniminterface.dll (an
    // import of animEngineDLL.dll) first.
    this.log('[engine] engine process: DllMain(nativeaniminterface), DllMain(animEngineDLL)');
    await this.callMain(nai.entry, nai.base, 1, 0);
    await this.callMain(ae.entry, ae.base, 1, 0);
    // Java process: java.exe loads nativeaniminterface.dll, which imports
    // animEngineDLL.dll.
    this.log('[engine] java process: DllMain(animEngineDLL), DllMain(nativeaniminterface)');
    await this.callThread(ae2.entry, [ae2.base, 1, 0]);
    await this.callThread(nai2.entry, [nai2.base, 1, 0]);
    const winMain = ae.exports['?AnimEngineWinMain@@YAHPAUHINSTANCE__@@0PADH@Z'];
    this.log('[engine] AnimEngineWinMain');
    this.running = this.callMain(winMain, 0x00400000, 0, 0, 1);
    const code = await this.running;
    this.log(`[engine] AnimEngineWinMain returned ${code}`);
  }
}
