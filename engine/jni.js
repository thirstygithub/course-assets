// JNI for the recompiled nativeaniminterface.dll.
//
// Java (on CheerpJ) calls NativeAnimationInterface's natives; each one runs
// the DLL's original Java_facade_nativeinterface_* function on a fresh guest
// thread of the "java process", with a JNIEnv whose function table lands in
// handle() below.
//
// CheerpJ can't run calls into Java while Java threads sit in native methods,
// so the DLL's effects on Java objects aren't applied here: field writes and
// (void) method calls are recorded and returned with the result, and the Java
// wrapper (tools/gen_nai_shim.py -> facade.web.Jni.finish) applies them, in
// order, when the native returns. The DLL never reads back from Java objects,
// so this is equivalent.

const TYPES = ['Object', 'Boolean', 'Byte', 'Char', 'Short', 'Int', 'Long', 'Float', 'Double', 'Void'];
const FS = '\u0003', OPS = '\u0002', RET = '\u0001';

// Parameter descriptors of a method signature: ['I', 'F', '[I', 'Lfoo;', ...]
function paramTypes(sig) {
  const out = [];
  for (let i = 1; sig[i] !== ')';) {
    let j = i;
    while (sig[j] === '[') j++;
    j = sig[j] === 'L' ? sig.indexOf(';', j) + 1 : j + 1;
    out.push(sig.slice(i, j));
    i = j;
  }
  return out;
}
const returnType = sig => sig.slice(sig.indexOf(')') + 1);

export class JavaBridge {
  constructor(host) {
    this.host = host;
    // handle -> value: { param: k } (the native's k-th object argument),
    // { classOf: value }, { className }, { intArray }, or a JS string
    this.handles = new Map();
    this.nextHandle = 0x100;
    this.methods = [null];        // method ID -> { name, sig, isStatic }
    this.fields = [null];         // field ID -> { name, sig }
    this.ops = new Map();         // guest thread -> recorded operations
    this.locals = new Map();      // guest thread -> handles to free
    this.globals = new Set();
    this.calls = 0;
    this.nativeCalls = 0;
  }

  get dv() { return new DataView(this.host.memory.buffer); }
  u32(p) { return this.dv.getUint32(p, true); }
  i32(p) { return this.dv.getInt32(p, true); }
  f32(p) { return this.dv.getFloat32(p, true); }
  f64(p) { return this.dv.getFloat64(p, true); }

  // Handles created while a native call runs are local to its guest thread
  // and freed when the call returns (NewGlobalRef pins one).
  ref(v, thread = null) {
    if (v === null || v === undefined) return 0;
    const h = this.nextHandle++;
    this.handles.set(h, v);
    if (thread !== null) (this.locals.get(thread) ?? this.locals.set(thread, []).get(thread)).push(h);
    return h;
  }
  deref(h) { return h ? this.handles.get(h) : null; }

  freeLocals(thread) {
    for (const h of this.locals.get(thread) ?? []) if (!this.globals.has(h)) this.handles.delete(h);
    this.locals.delete(thread);
  }

  // Method/field IDs are stable per (name, signature), as in a real JVM; the
  // DLL looks them up on every call.
  memberId(table, name, sig, isStatic) {
    const key = `${isStatic ? 's' : 'i'} ${name} ${sig}`;
    table.index ??= new Map();
    let id = table.index.get(key);
    if (id === undefined) {
      id = table.push({ name, sig, isStatic }) - 1;
      table.index.set(key, id);
    }
    return id;
  }

  // How a recorded operation names its target (see Jni.finish).
  target(v) {
    if (v?.param !== undefined) return `p${v.param}`;
    if (v?.classOf?.param !== undefined) return `c${v.classOf.param}`;
    if (v?.className) return `n${v.className.replace(/\//g, '.')}`;
    throw new Error(`JNI: operation on an object the native wasn't given: ${JSON.stringify(v)}`);
  }

  // Encodes the arguments of a Call*Method family entry: variant 0 is "...",
  // 1 is a va_list, 2 is a jvalue array.
  readArgs(sig, variant, p) {
    const out = [];
    for (const t of paramTypes(sig)) {
      out.push(this.encode(t, p, variant === 2));
      p += variant === 2 ? 8 : (t === 'J' || t === 'D' || t === 'F' ? 8 : 4);
    }
    return out.join(',');
  }

  encode(t, p, jvalue) {
    switch (t) {
      case 'Z': case 'B': return String(this.dv.getUint8(p) ? 1 : 0);
      case 'I': case 'S': case 'C': return String(this.i32(p));
      case 'J': return this.dv.getBigInt64(p, true).toString();
      case 'F': return String(jvalue ? this.f32(p) : this.f64(p));   // varargs promote float to double
      case 'D': return String(this.f64(p));
      case '[I': {
        const a = this.deref(this.u32(p));
        return a ? Array.from(a.intArray).join(';') : '';
      }
      default: throw new Error(`JNI: unsupported argument type ${t}`);
    }
  }

  record(thread, fields) { this.ops.get(thread)?.push(fields.join(FS)); }

  // ---- the JNIEnv entry point ----
  // Writes eax/edx/float result and the stdcall pop size for hle_jni.c.
  // Synchronous, so the guest thread doesn't suspend on every JNI call.
  handle(index, a, thread) {
    this.calls++;
    const arg = i => this.u32(a + 4 * i);
    let ret = 0, pop = 0;
    const done = (v, nargs) => { ret = v >>> 0; pop = nargs * 4; };

    if (index >= 34 && index <= 63 || index >= 114 && index <= 143) {        // Call[Static]<T>Method[V|A]
      const isStatic = index >= 114;
      const k = index - (isStatic ? 114 : 34), t = TYPES[Math.floor(k / 3)], variant = k % 3;
      if (t !== 'Void') throw new Error(`JNI: Call${isStatic ? 'Static' : ''}${t}Method needs a value from Java`);
      const m = this.methods[arg(2)];
      const p = variant === 0 ? a + 12 : this.u32(a + 12);                 // va_list / jvalue* are pointers
      this.record(thread, ['C', this.target(this.deref(arg(1))), isStatic ? 1 : 0, m.name, m.sig, this.readArgs(m.sig, variant, p)]);
      pop = variant === 0 ? 0 : 16;                                         // "..." variants are cdecl
    } else if (index >= 104 && index <= 112) {                              // Set<T>Field
      const t = TYPES[index - 104], f = this.fields[arg(2)], p = a + 12;
      const v = t === 'Float' ? this.f32(p) : t === 'Double' ? this.f64(p) : t === 'Long' ? this.dv.getBigInt64(p, true)
        : t === 'Boolean' ? (this.dv.getUint8(p) ? 1 : 0) : this.i32(p);
      this.record(thread, ['S', this.target(this.deref(arg(1))), f.name, String(v)]);
      pop = t === 'Long' || t === 'Double' ? 20 : 16;
    } else {
      switch (index) {
        case 4: done(0x00010004, 1); break;                                       // GetVersion
        case 6: done(this.ref({ className: this.host.cstr(arg(1)) }, thread), 2); break; // FindClass
        case 15: done(0, 1); break;                                               // ExceptionOccurred
        case 16: case 17: done(0, 1); break;                                      // ExceptionDescribe/Clear
        case 18: throw new Error(`JNI FatalError: ${this.host.cstr(arg(1))}`);   // FatalError
        case 21: this.globals.add(arg(1)); done(arg(1), 2); break;                // NewGlobalRef
        case 22: this.globals.delete(arg(1)); done(0, 2); break;                  // DeleteGlobalRef
        case 23: done(0, 2); break;                                               // DeleteLocalRef
        case 24: done(this.deref(arg(1)) === this.deref(arg(2)) ? 1 : 0, 3); break; // IsSameObject
        case 31: done(this.ref({ classOf: this.deref(arg(1)) }, thread), 2); break; // GetObjectClass
        case 33: case 113:                                                         // Get(Static)MethodID
          done(this.memberId(this.methods, this.host.cstr(arg(2)), this.host.cstr(arg(3)), index === 113), 4);
          break;
        case 94: case 144:                                                         // Get(Static)FieldID
          done(this.memberId(this.fields, this.host.cstr(arg(2)), this.host.cstr(arg(3)), index === 144), 4);
          break;
        case 164: done(String(this.deref(arg(1)) ?? '').length, 2); break;       // GetStringLength
        case 167: done(this.ref(this.host.cstr(arg(1)), thread), 2); break;      // NewStringUTF
        case 168: done(new TextEncoder().encode(String(this.deref(arg(1)) ?? '')).length, 2); break;
        case 169: {                                                                // GetStringUTFChars
          const bytes = new TextEncoder().encode(String(this.deref(arg(1)) ?? ''));
          const p = this.host.x.rt_alloc(bytes.length + 1);
          this.host.u8.set(bytes, p);
          this.host.u8[p + bytes.length] = 0;
          if (arg(2)) this.dv.setUint8(arg(2), 1);
          done(p, 3);
          break;
        }
        case 170: this.host.x.rt_free(arg(2)); done(0, 3); break;                // ReleaseStringUTFChars
        case 171: done(this.deref(arg(1))?.intArray?.length ?? 0, 2); break;     // GetArrayLength
        case 179: done(this.ref({ intArray: new Int32Array(arg(1)) }, thread), 2); break; // NewIntArray
        case 187: {                                                                // GetIntArrayElements
          const arr = this.deref(arg(1)).intArray;
          const p = this.host.x.rt_alloc(Math.max(4, arr.length * 4));
          for (let i = 0; i < arr.length; i++) this.dv.setInt32(p + 4 * i, arr[i], true);
          if (arg(2)) this.dv.setUint8(arg(2), 1);
          done(p, 3);
          break;
        }
        case 195: {                                                                // ReleaseIntArrayElements
          const arr = this.deref(arg(1)).intArray, p = arg(2), mode = arg(3);
          if (mode !== 2) for (let i = 0; i < arr.length; i++) arr[i] = this.i32(p + 4 * i);   // not JNI_ABORT
          if (mode !== 1) this.host.x.rt_free(p);                                  // not JNI_COMMIT
          done(0, 4);
          break;
        }
        case 202: {                                                                // GetIntArrayRegion
          const arr = this.deref(arg(1)).intArray, start = arg(2), n = arg(3);
          for (let i = 0; i < n; i++) this.dv.setInt32(arg(4) + 4 * i, arr[start + i], true);
          done(0, 5);
          break;
        }
        case 210: {                                                                // SetIntArrayRegion
          const arr = this.deref(arg(1)).intArray, start = arg(2), n = arg(3);
          for (let i = 0; i < n; i++) arr[start + i] = this.i32(arg(4) + 4 * i);
          done(0, 5);
          break;
        }
        case 228: done(0, 1); break;                                              // ExceptionCheck
        default:
          throw new Error(`JNI function #${index} not implemented`);
      }
    }
    const r = new Uint32Array(this.host.memory.buffer, this.host.x.rt_jni_result(thread), 8);
    r[0] = pop; r[1] = ret; r[2] = 0; r[3] = 0;
  }

  // ---- natives for cheerpjInit ----
  // `descriptors` maps each native to its original JNI descriptor; Java calls
  // n_<name>(nullMask, <primitive and String arguments>) and keeps the other
  // object arguments itself (tools/gen_nai_shim.py).
  natives(descriptors) {
    const { host } = this;
    const nai2 = host.modules.nai2;
    const out = {};
    for (const [name, desc] of Object.entries(descriptors)) {
      const exp = Object.keys(nai2.exports).find(e => e.startsWith(`_Java_facade_nativeinterface_NativeAnimationInterface_${name}@`));
      if (!exp) continue;   // natives the DLL doesn't implement were never called by the original either
      const addr = nai2.exports[exp];
      const types = paramTypes(desc), rt = returnType(desc);
      const run = async (nullMask, prims) => {
        const env = host.x.rt_jni_env();
        const words = [env, this.classHandle ??= this.ref({ className: 'facade/nativeinterface/NativeAnimationInterface' })];
        const locals = [];
        let obj = 0, prim = 0;
        for (const t of types) {
          if (t.startsWith('L') || t.startsWith('[')) {
            if (t === 'Ljava/lang/String;') { const h = this.ref(String(prims[prim++] ?? '')); locals.push(h); words.push(h); continue; }
            const k = obj++;
            if (nullMask & (1 << k)) { words.push(0); continue; }
            const h = this.ref({ param: k });
            locals.push(h);
            words.push(h);
            continue;
          }
          const v = prims[prim++];
          if (t === 'Z') words.push(v ? 1 : 0);
          else if (t === 'F') { const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, v, true); words.push(b.getUint32(0, true)); }
          else if (t === 'J') { const big = BigInt(v); words.push(Number(big & 0xffffffffn), Number((big >> 32n) & 0xffffffffn)); }
          else words.push(v | 0);
        }
        let thread = null;
        try {
          const { eax } = await host.callThread(addr, words, t => { thread = t; this.ops.set(t, []); });
          const value = rt === 'I' ? String(eax | 0) : rt === 'Z' ? (eax ? '1' : '0')
            : rt.startsWith('L') ? String(this.deref(eax) ?? '') : '';
          const ops = this.ops.get(thread) ?? [];
          return ops.length ? value + RET + ops.join(OPS) : value;
        } finally {
          for (const h of locals) this.handles.delete(h);
          if (thread !== null) { this.freeLocals(thread); this.ops.delete(thread); }
        }
      };
      // JNI name mangling: '_' in a method name is written "_1"
      const jname = `Java_facade_nativeinterface_NativeAnimationInterface_${`n_${name}`.replace(/_/g, '_1')}`;
      out[jname] = SNAPSHOT_READS.has(name) ? this.cachedRead(run) : (lib, nullMask, ...prims) => {
        this.nativeCalls++;
        return run(nullMask, prims);
      };
    }
    return out;
  }

  // A native that only reads engine state, answered synchronously with the
  // newest result for the same arguments while a background guest thread
  // (outside CheerpJ) refreshes it. CheerpJ's runtime eventually crashes when
  // resuming Java threads from asynchronous natives (CJ3Library.restart, after
  // a few minutes of play, in 4.2 and 5.0 alike), so the sensors, which make
  // most of the calls, stay off that path. The values are at most a frame or
  // two older; the original's sensors also read the engine's last frame.
  cachedRead(run) {
    const cache = new Map();   // arguments -> { value, at, busy }
    return (lib, nullMask, ...prims) => {
      this.nativeCalls++;
      const key = nullMask + '|' + prims.join(',');
      let c = cache.get(key);
      if (!c) {
        // first call: answer when the DLL has
        c = { value: null, at: 0, busy: true };
        cache.set(key, c);
        return run(nullMask, prims).then(v => { c.value = v; c.at = performance.now(); return v; })
          .finally(() => { c.busy = false; });
      }
      if (!c.busy && performance.now() - c.at > READ_REFRESH_MS) {
        c.busy = true;
        run(nullMask, prims).then(v => { c.value = v; c.at = performance.now(); })
          .catch(e => this.host.log(`[jni] background read failed: ${e}`))
          .finally(() => { c.busy = false; });
      }
      return c.value ?? '';
    };
  }
}

// Natives that only read engine state (no side effects, nothing consumed):
// served by JavaBridge.cachedRead. Event-like reads (activations, anim cues,
// gestures, typed text, completed actions) and every command stay synchronous
// with the engine.
const SNAPSHOT_READS = new Set([
  'getObjectPosition', 'getObjectRotation', 'getObjectState', 'getObjectPickupPosition',
  'getAllObjectPositions', 'getAllObjectRotations', 'getAllObjectStagingRotations', 'getAllObjectStates',
  'getAllObjectPickupPositions', 'getAllObjectPickupRotations', 'getAllHeldObjects', 'getArmGestureBases',
  'getGazeTracking', 'getEyelidHeightL', 'getEyelidHeightR', 'getBrowShapeL', 'getBrowShapeR', 'getMouthShape',
]);
const READ_REFRESH_MS = 15;
