/**
 * A small GPU fluid: Stam's stable fluids on WebGL2 (semi-Lagrangian advection,
 * vorticity confinement, Jacobi pressure projection). Velocity lives on a coarse
 * grid, dye on a finer one. The hero pours chroma into it through twelve nozzles,
 * one per pitch class, so the picture is driven by real (decoded) HPCP frames.
 *
 * Coordinates passed in are UV: x, y in [0, 1], y up.
 */

type RGB = [number, number, number]

export interface FluidConfig {
  simRes: number
  dyeRes: number
  velocityDissipation: number
  densityDissipation: number
  pressureIterations: number
  curl: number
  splatRadius: number
  background: RGB
}

const VERT = `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv, vL, vR, vT, vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`

const HEAD = `precision highp float; precision highp sampler2D; varying vec2 vUv, vL, vR, vT, vB;`

const FRAG = {
  splat: `${HEAD}
uniform sampler2D uTarget; uniform float aspectRatio; uniform vec3 color; uniform vec2 point; uniform float radius;
void main () {
  vec2 p = vUv - point; p.x *= aspectRatio;
  gl_FragColor = vec4(texture2D(uTarget, vUv).xyz + exp(-dot(p, p) / radius) * color, 1.0);
}`,
  // twelve splats in one pass: the pitch-class nozzles
  nozzles: `${HEAD}
uniform sampler2D uTarget; uniform float aspectRatio; uniform float radius;
uniform vec2 points[12]; uniform vec3 values[12];
void main () {
  vec3 base = texture2D(uTarget, vUv).xyz;
  for (int i = 0; i < 12; i++) {
    vec2 p = vUv - points[i]; p.x *= aspectRatio;
    base += exp(-dot(p, p) / radius) * values[i];
  }
  gl_FragColor = vec4(base, 1.0);
}`,
  advection: `${HEAD}
uniform sampler2D uVelocity, uSource; uniform vec2 texelSize; uniform float dt, dissipation;
void main () {
  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
  gl_FragColor = texture2D(uSource, coord) / (1.0 + dissipation * dt);
}`,
  divergence: `${HEAD}
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).x, R = texture2D(uVelocity, vR).x;
  float T = texture2D(uVelocity, vT).y, B = texture2D(uVelocity, vB).y;
  vec2 C = texture2D(uVelocity, vUv).xy;
  if (vL.x < 0.0) L = -C.x; if (vR.x > 1.0) R = -C.x;
  if (vT.y > 1.0) T = -C.y; if (vB.y < 0.0) B = -C.y;
  gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`,
  curl: `${HEAD}
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).y, R = texture2D(uVelocity, vR).y;
  float T = texture2D(uVelocity, vT).x, B = texture2D(uVelocity, vB).x;
  gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}`,
  vorticity: `${HEAD}
uniform sampler2D uVelocity, uCurl; uniform float curl, dt;
void main () {
  float L = texture2D(uCurl, vL).x, R = texture2D(uCurl, vR).x;
  float T = texture2D(uCurl, vT).x, B = texture2D(uCurl, vB).x, C = texture2D(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= curl * C; force.y *= -1.0;
  vec2 v = texture2D(uVelocity, vUv).xy + force * dt;
  gl_FragColor = vec4(clamp(v, -1000.0, 1000.0), 0.0, 1.0);
}`,
  pressure: `${HEAD}
uniform sampler2D uPressure, uDivergence;
void main () {
  float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
  gl_FragColor = vec4((L + R + B + T - texture2D(uDivergence, vUv).x) * 0.25, 0.0, 0.0, 1.0);
}`,
  gradient: `${HEAD}
uniform sampler2D uPressure, uVelocity;
void main () {
  float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
  gl_FragColor = vec4(texture2D(uVelocity, vUv).xy - vec2(R - L, T - B), 0.0, 1.0);
}`,
  clear: `${HEAD}
uniform sampler2D uTexture; uniform float value;
void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`,
  // dye is light on a night ground; tone-map so dense dye saturates instead of clipping
  display: `${HEAD}
uniform sampler2D uTexture; uniform vec3 background;
void main () {
  vec3 c = texture2D(uTexture, vUv).rgb;
  c = 1.0 - exp(-c * 1.5);
  c = max(mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, 1.45), 0.0);
  float a = max(c.r, max(c.g, c.b));
  vec3 col = background * (1.0 - 0.5 * a) + c;
  vec2 q = vUv - 0.5;
  col *= 1.0 - 0.35 * dot(q, q);
  gl_FragColor = vec4(col, 1.0);
}`,
}

interface Target {
  tex: WebGLTexture
  fbo: WebGLFramebuffer
  w: number
  h: number
  attach: (unit: number) => number
}
interface Double {
  read: Target
  write: Target
  swap: () => void
  w: number
  h: number
}
interface Program {
  prog: WebGLProgram
  u: Record<string, WebGLUniformLocation | null>
}

export class Fluid {
  private gl: WebGL2RenderingContext
  private cfg: FluidConfig
  private p: Record<keyof typeof FRAG, Program>
  private velocity!: Double
  private dye!: Double
  private pressure!: Double
  private divergence!: Target
  private curlT!: Target
  private disposed = false

  static create(canvas: HTMLCanvasElement, cfg: FluidConfig): Fluid | null {
    const gl = canvas.getContext('webgl2', { alpha: false, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false })
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) return null
    try {
      const f = new Fluid(gl, cfg)
      // a half-float target that cannot be rendered to means no fluid on this device
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null
      return f
    } catch {
      return null
    }
  }

  private constructor(gl: WebGL2RenderingContext, cfg: FluidConfig) {
    this.gl = gl
    this.cfg = cfg
    const vs = this.shader(gl.VERTEX_SHADER, VERT)
    this.p = Object.fromEntries(Object.entries(FRAG).map(([k, src]) => [k, this.program(vs, this.shader(gl.FRAGMENT_SHADER, src))])) as Record<
      keyof typeof FRAG,
      Program
    >
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW)
    const idx = gl.createBuffer()
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.enableVertexAttribArray(0)
    this.allocate()
  }

  private shader(type: number, src: string) {
    const gl = this.gl
    const s = gl.createShader(type)!
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader')
    return s
  }

  private program(vs: WebGLShader, fs: WebGLShader): Program {
    const gl = this.gl
    const prog = gl.createProgram()!
    gl.attachShader(prog, vs)
    gl.attachShader(prog, fs)
    gl.bindAttribLocation(prog, 0, 'aPosition')
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) ?? 'link')
    const u: Program['u'] = {}
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS) as number
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(prog, i)!.name.replace(/\[0\]$/, '')
      u[name] = gl.getUniformLocation(prog, name)
    }
    return { prog, u }
  }

  private res(base: number) {
    const { drawingBufferWidth: w, drawingBufferHeight: h } = this.gl
    const aspect = w / h
    const min = Math.round(base)
    const max = Math.round(base * Math.max(aspect, 1 / aspect))
    return w > h ? { w: max, h: min } : { w: min, h: max }
  }

  private target(w: number, h: number, internal: number, format: number): Target {
    const gl = this.gl
    gl.activeTexture(gl.TEXTURE0)
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.HALF_FLOAT, null)
    const fbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
    gl.viewport(0, 0, w, h)
    gl.clearColor(0, 0, 0, 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    return {
      tex,
      fbo,
      w,
      h,
      attach(unit: number) {
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, tex)
        return unit
      },
    }
  }

  private double(w: number, h: number, internal: number, format: number): Double {
    let a = this.target(w, h, internal, format)
    let b = this.target(w, h, internal, format)
    return {
      w,
      h,
      get read() {
        return a
      },
      get write() {
        return b
      },
      swap() {
        const t = a
        a = b
        b = t
      },
    }
  }

  private free(t: Target) {
    this.gl.deleteTexture(t.tex)
    this.gl.deleteFramebuffer(t.fbo)
  }

  private allocate() {
    const gl = this.gl
    if (this.velocity) {
      for (const d of [this.velocity, this.dye, this.pressure]) {
        this.free(d.read)
        this.free(d.write)
      }
      this.free(this.divergence)
      this.free(this.curlT)
    }
    const sim = this.res(this.cfg.simRes)
    const dye = this.res(this.cfg.dyeRes)
    this.velocity = this.double(sim.w, sim.h, gl.RG16F, gl.RG)
    this.dye = this.double(dye.w, dye.h, gl.RGBA16F, gl.RGBA)
    this.pressure = this.double(sim.w, sim.h, gl.R16F, gl.RED)
    this.divergence = this.target(sim.w, sim.h, gl.R16F, gl.RED)
    this.curlT = this.target(sim.w, sim.h, gl.R16F, gl.RED)
  }

  /** Call after the canvas's backing size changes. Clears the fluid. */
  resize() {
    if (!this.disposed) this.allocate()
  }

  private use(name: keyof typeof FRAG) {
    const pr = this.p[name]
    this.gl.useProgram(pr.prog)
    return pr.u
  }

  private blit(target: Target | null) {
    const gl = this.gl
    if (target) {
      gl.viewport(0, 0, target.w, target.h)
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo)
    } else {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)
  }

  private get aspect() {
    return this.gl.drawingBufferWidth / this.gl.drawingBufferHeight
  }

  private radius(scale = 1) {
    const r = (this.cfg.splatRadius * scale) / 100
    return this.aspect > 1 ? r * this.aspect : r
  }

  /** One splat: push velocity (dx, dy in texels/s) and add dye. */
  splat(x: number, y: number, dx: number, dy: number, color: RGB, scale = 1) {
    const gl = this.gl
    const u = this.use('splat')
    gl.uniform1i(u.uTarget, this.velocity.read.attach(0))
    gl.uniform1f(u.aspectRatio, this.aspect)
    gl.uniform2f(u.point, x, y)
    gl.uniform3f(u.color, dx, dy, 0)
    gl.uniform1f(u.radius, this.radius(scale))
    gl.uniform2f(u.texelSize, 1 / this.velocity.w, 1 / this.velocity.h)
    this.blit(this.velocity.write)
    this.velocity.swap()
    gl.uniform1i(u.uTarget, this.dye.read.attach(0))
    gl.uniform3f(u.color, color[0], color[1], color[2])
    this.blit(this.dye.write)
    this.dye.swap()
  }

  /** Twelve splats in one pass each for velocity and dye. Arrays are length 24 / 36 / 36. */
  nozzles(points: Float32Array, forces: Float32Array, colors: Float32Array, scale = 1) {
    const gl = this.gl
    const u = this.use('nozzles')
    gl.uniform1f(u.aspectRatio, this.aspect)
    gl.uniform1f(u.radius, this.radius(scale))
    gl.uniform2fv(u.points, points)
    gl.uniform2f(u.texelSize, 1 / this.velocity.w, 1 / this.velocity.h)
    gl.uniform1i(u.uTarget, this.velocity.read.attach(0))
    gl.uniform3fv(u.values, forces)
    this.blit(this.velocity.write)
    this.velocity.swap()
    gl.uniform1i(u.uTarget, this.dye.read.attach(0))
    gl.uniform3fv(u.values, colors)
    this.blit(this.dye.write)
    this.dye.swap()
  }

  step(dt: number) {
    const gl = this.gl
    const v = this.velocity
    const simTexel: [number, number] = [1 / v.w, 1 / v.h]
    gl.disable(gl.BLEND)

    let u = this.use('curl')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uVelocity, v.read.attach(0))
    this.blit(this.curlT)

    u = this.use('vorticity')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uVelocity, v.read.attach(0))
    gl.uniform1i(u.uCurl, this.curlT.attach(1))
    gl.uniform1f(u.curl, this.cfg.curl)
    gl.uniform1f(u.dt, dt)
    this.blit(v.write)
    v.swap()

    u = this.use('divergence')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uVelocity, v.read.attach(0))
    this.blit(this.divergence)

    u = this.use('clear')
    gl.uniform1i(u.uTexture, this.pressure.read.attach(0))
    gl.uniform1f(u.value, 0.8)
    this.blit(this.pressure.write)
    this.pressure.swap()

    u = this.use('pressure')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uDivergence, this.divergence.attach(0))
    for (let i = 0; i < this.cfg.pressureIterations; i++) {
      gl.uniform1i(u.uPressure, this.pressure.read.attach(1))
      this.blit(this.pressure.write)
      this.pressure.swap()
    }

    u = this.use('gradient')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uPressure, this.pressure.read.attach(0))
    gl.uniform1i(u.uVelocity, v.read.attach(1))
    this.blit(v.write)
    v.swap()

    u = this.use('advection')
    gl.uniform2f(u.texelSize, ...simTexel)
    gl.uniform1i(u.uVelocity, v.read.attach(0))
    gl.uniform1i(u.uSource, v.read.attach(0))
    gl.uniform1f(u.dt, dt)
    gl.uniform1f(u.dissipation, this.cfg.velocityDissipation)
    this.blit(v.write)
    v.swap()

    gl.uniform1i(u.uVelocity, v.read.attach(0))
    gl.uniform1i(u.uSource, this.dye.read.attach(1))
    gl.uniform1f(u.dissipation, this.cfg.densityDissipation)
    this.blit(this.dye.write)
    this.dye.swap()
  }

  render() {
    const gl = this.gl
    const u = this.use('display')
    gl.uniform2f(u.texelSize, 1 / gl.drawingBufferWidth, 1 / gl.drawingBufferHeight)
    gl.uniform1i(u.uTexture, this.dye.read.attach(0))
    gl.uniform3f(u.background, ...this.cfg.background)
    this.blit(null)
  }

  /** Frees GPU memory. The context itself is left alive: the canvas may be reused (React StrictMode remounts). */
  dispose() {
    this.disposed = true
    for (const d of [this.velocity, this.dye, this.pressure]) {
      this.free(d.read)
      this.free(d.write)
    }
    this.free(this.divergence)
    this.free(this.curlT)
  }
}
