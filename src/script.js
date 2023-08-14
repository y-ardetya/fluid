import * as THREE from "three";

const VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
  }
`;

const BUFFER_A_FRAG = /* glsl */ `
  uniform vec2 uResolution;
  uniform sampler2D uChannel;
  uniform float uFrame;
  uniform float uTime;
  uniform vec2 uMouse;

  //* skew constants for 3d simplex functions *//
  const float F3 =  0.3333333;
  const float G3 =  0.1666667;

  vec3 random3(vec3 c) {
    float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
    vec3 r;
    r.z = fract(512.0*j);
    j *= .125;
    r.x = fract(512.0*j);
    j *= .125;
    r.y = fract(512.0*j);
    return r-0.5;
  }
  
  /* 3d simplex noise */
  float simplex3d(vec3 p) {
     /* 1. find current tetrahedron T and it's four vertices */
     /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
     /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
     
     /* calculate s and x */
     vec3 s = floor(p + dot(p, vec3(F3)));
     vec3 x = p - s + dot(s, vec3(G3));
     
     /* calculate i1 and i2 */
     vec3 e = step(vec3(0.0), x - x.yzx);
     vec3 i1 = e*(1.0 - e.zxy);
     vec3 i2 = 1.0 - e.zxy*(1.0 - e);
       
     /* x1, x2, x3 */
     vec3 x1 = x - i1 + G3;
     vec3 x2 = x - i2 + 2.0*G3;
     vec3 x3 = x - 1.0 + 3.0*G3;
     
     /* 2. find four surflets and store them in d */
     vec4 w, d;
     
     /* calculate surflet weights */
     w.x = dot(x, x);
     w.y = dot(x1, x1);
     w.z = dot(x2, x2);
     w.w = dot(x3, x3);
     
     /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
     w = max(0.6 - w, 0.0);
     
     /* calculate surflet components */
     d.x = dot(random3(s), x);
     d.y = dot(random3(s + i1), x1);
     d.z = dot(random3(s + i2), x2);
     d.w = dot(random3(s + 1.0), x3);
     
     /* multiply d by w^4 */
     w *= w;
     w *= w;
     d *= w;
     
     /* 3. return the sum of the four surflets */
     return dot(d, vec4(52.0));
  }
  
  /*****************************************************************************/

  vec2 pen(float t) {
    t *= 0.05;
    return 0.5 * uResolution.xy *
        vec2(simplex3d(vec3(t,0,0)) + 1.,
             simplex3d(vec3(0,t,0)) + 1.);
  }

  #define T(gl_FragCoord) texture2D(uChannel,(gl_FragCoord)/uResolution.xy)
  #define length2(gl_FragCoord) dot(gl_FragCoord, gl_FragCoord)
  #define dt 0.15
  #define K 0.2
  #define nu 0.5
  #define kappa 0.1

  void main() {
    if(uFrame < 10.0) {
      gl_FragColor = vec4(1.0);
      return;
    }
    gl_FragColor = T(gl_FragCoord.xy);

    vec4 n = T(gl_FragCoord.xy + vec2(0, 1));
    vec4 e = T(gl_FragCoord.xy + vec2(1, 0));
    vec4 s = T(gl_FragCoord.xy - vec2(0, 1));
    vec4 w = T(gl_FragCoord.xy - vec2(1, 0));

    vec4 laplacian = (n + e + s + w - 4.0 * gl_FragColor);

    vec4 dx = (e - w) / 2.0;
    vec4 dy = (n - s) / 2.0;

    //* Velocity
    float div = dx.x + dy.y;
    //* Mass Conservation
    gl_FragColor.z -= dt *(dx.z * gl_FragColor.x + dy.z * gl_FragColor.y + div * gl_FragColor.z);
    //* Semi Langrangian Advection
    gl_FragColor.xyw = T(gl_FragCoord.xy - dt * gl_FragColor.xy).xyw;
    //* Viscosity
    gl_FragColor.xyw += dt * vec3(nu, nu, kappa) * laplacian.xyw;
    //* Nullify Divergence
    gl_FragColor.xy -= K * vec2(dx.z, dy.z);
    //* External Source
    vec2 m = uMouse.xy;
    gl_FragColor.xyw += dt * exp(-length2(gl_FragCoord.xy - m) / 800.0) * vec3(m - pen(uTime -0.1), 1.0);
    //* Dissipation
    //! Very important
    gl_FragColor.w -= dt * 0.00005;
    //* Output
    gl_FragColor.xyzw = clamp(gl_FragColor.xyzw, vec4(-5.0, -5.0, 0.5, 0.0), vec4(5.0, 5.0, 3.0, 5.0));
  }
`;

const BUFFER_FINAL_FRAG = /* glsl */ `
  #define PI 3.1415926535897932384626433832795

  uniform sampler2D uChannel;
  uniform vec2 uResolution;
  uniform sampler2D uTexture1;
  uniform float uTime;

  // cosine based palette, 4 vec3 params
  vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
    {
    return a + b*cos( 6.28318*(c*t+d) );
    }

  void main () {
    vec4 img1 = texture2D(uTexture1, gl_FragCoord.xy / uResolution.xy);
    float grayscale = (img1.r + img1.g + img1.b) / 3.0;
    img1.rgb = vec3(grayscale);
    vec4 img2 = texture2D(uTexture1, gl_FragCoord.xy / uResolution.xy);
    vec4 c = texture2D(uChannel, gl_FragCoord.xy / uResolution.xy);
    vec4 finalOutput = mix(img1, img2, c.w);

    gl_FragColor.rgb =  0.6 + 0.6 * cos(6.3 * atan(c.y, c.x) / (2.0 * PI) + vec3(0.0, 23.0, 21.0)); //* Velocity
    gl_FragColor.rgb *= c.a / 5.0; //* Ink
    gl_FragColor.rgb += clamp(c.w - 1.0, 0.0, 1.0) / 10.0; //* Local Density
    gl_FragColor.a = 1.0;
    gl_FragColor = finalOutput;
  }
`;

class App {
  constructor() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.renderer = new THREE.WebGLRenderer();
    this.mousePosition = new THREE.Vector2();
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.renderer.setClearColor(0xffffff);
    this.renderer.setSize(this.width, this.height);
    document.body.appendChild(this.renderer.domElement);

    this.renderer.domElement.addEventListener("mousemove", (e) => {
      this.mousePosition.setX(e.clientX);
      this.mousePosition.setY(this.height - e.clientY);
    });
    this.targetA = new BufferManager(this.renderer, {
      width: this.width,
      height: this.height,
    });
    this.targetB = new BufferManager(this.renderer, {
      width: this.width,
      height: this.height,
    });
  }

  start() {
    const resolution = new THREE.Vector2(this.width, this.height);
    this.bufferA = new BufferShader(BUFFER_A_FRAG, {
      uChannel: { value: null },
      uResolution: { value: resolution },
      uMouse: { value: this.mousePosition },
      uFrame: { value: 10 },
      uTime: { value: 0 },
    });
    this.bufferB = new BufferShader(BUFFER_FINAL_FRAG, {
      uChannel: { value: null },
      uResolution: { value: resolution },
      uTexture1: { value: new THREE.TextureLoader().load("/1.jpg") },
      uTime: { value: 0 },
    });
    this.animate();
  }

  animate() {
    requestAnimationFrame(() => {
      //* Animate uTime
      this.bufferA.uniforms["uTime"].value += 0.01;
      this.bufferB.uniforms["uTime"].value += 0.01;
      this.bufferA.uniforms["uFrame"].value += 1;

      //* Render buffer A first
      this.bufferA.uniforms["uChannel"].value = this.targetA.readBuffer.texture;
      this.targetA.render(this.bufferA.scene, this.orthoCamera);

      //* Render buffer B
      this.bufferB.uniforms["uChannel"].value = this.targetA.readBuffer.texture;
      this.targetB.render(this.bufferB.scene, this.orthoCamera, true);

      this.animate();
    });
  }
}

class BufferShader {
  constructor(fragmentShader, uniforms = {}) {
    this.uniforms = uniforms;
    this.material = new THREE.ShaderMaterial({
      fragmentShader,
      vertexShader: VERTEX_SHADER,
      uniforms,
    });
    this.scene = new THREE.Scene();
    this.scene.add(
      new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material)
    );
  }
}

class BufferManager {
  constructor(renderer, size) {
    this.renderer = renderer;
    this.readBuffer = new THREE.WebGLRenderTarget(size.width, size.height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.FloatType,
      stencilBuffer: false,
    });

    this.writeBuffer = this.readBuffer.clone();
  }
  swap() {
    const temp = this.readBuffer;
    this.readBuffer = this.writeBuffer;
    this.writeBuffer = temp;
  }
  render(scene, camera, toScreen = false) {
    if (toScreen) {
      this.renderer.render(scene, camera);
    } else {
      this.renderer.setRenderTarget(this.writeBuffer);
      this.renderer.clear();
      this.renderer.render(scene, camera);
      this.renderer.setRenderTarget(null);
    }
    this.swap();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new App().start();
});
