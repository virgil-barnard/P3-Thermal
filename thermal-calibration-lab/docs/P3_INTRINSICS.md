# P3 intrinsic parameters: sources, derivation, and assumptions

The synthetic generator uses a **nominal intrinsic matrix derived from published
P3 specifications**. It is not a factory calibration matrix or a measurement of
an individual camera. The principal point, zero skew, and baseline distortion
are explicit modeling assumptions.

Thermal Master's P3 product page lists the following values in its **Spec**
section. These specifications were checked on September 17, 2026.
[Manufacturer specifications](https://thermalmaster.com/products/p3-thermal-camera-for-iphone-and-android)

| Published quantity | Value | Use in the generator |
|---|---:|---|
| Native detector resolution | 256 × 192 | Native image dimensions |
| Detector pixel pitch | 12 μm | Conversion from physical length to pixels |
| Lens focal length | 4.3 mm | Nominal perspective projection |
| X³IR output resolution | 512 × 384 | Enhanced output; native geometry uses 256 × 192 |
| Field of view | 40° × 30.2° | Independent consistency check |

The pinhole model projects a point expressed in camera coordinates as

$$
u = f_x\frac{X}{Z}+c_x,
\qquad
v = f_y\frac{Y}{Z}+c_y.
$$

Here, $X,Y,Z$ use a common physical unit, while $u,v,f_x,f_y,c_x,c_y$ are expressed
in pixel coordinates. OpenCV documents both this model and the approximation
of intrinsics from lens specifications under **Intrinsic parameters from camera
lens specifications**.
[OpenCV camera-model documentation](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

The focal length in pixels is the physical focal length divided by pixel pitch:

$$
f_x = \frac{f_{\mathrm{mm}}}{p_{x,\mathrm{mm}}},
\qquad
f_y = \frac{f_{\mathrm{mm}}}{p_{y,\mathrm{mm}}}.
$$

Taking the advertised pitch to apply equally along both detector axes:

$$
12\,\mu\mathrm{m}=0.012\,\mathrm{mm},
\qquad
f_x=f_y=\frac{4.3}{0.012}=358.333333\ \mathrm{pixels}.
$$

The corresponding nominal detector dimensions are $256(0.012)=3.072$ mm wide
and $192(0.012)=2.304$ mm high.

The generator **assumes a centered principal point**. Native pixel centers have
integer coordinates: horizontal indices 0–255 and vertical indices 0–191.
The first pixel footprint spans $[-0.5,0.5]$ along each axis. Consequently,

$$
c_x=\frac{256-1}{2}=127.5,
\qquad
c_y=\frac{192-1}{2}=95.5.
$$

With zero skew, the nominal native intrinsic matrix is

$$
K=
\begin{bmatrix}
358.333333 & 0 & 127.5 \\
0 & 358.333333 & 95.5 \\
0 & 0 & 1
\end{bmatrix}.
$$

The following choices are assumptions, rather than values established by the
manufacturer's specification page:

- The optical axis intersects the image center.
- Pixel axes are orthogonal and share the same pitch, giving zero skew and
  $f_x=f_y$.
- The baseline Brown–Conrady distortion coefficients are all zero, in the order
  $(k_1,k_2,p_1,p_2,k_3)$. A separate challenge scenario deliberately introduces
  synthetic distortion.
- The nominal intrinsics remain fixed. Focus-dependent changes, digital crops,
  and transformations applied by a capture pipeline require separate treatment.

The `Camera` class in `thermal_calibration/config.py` implements these defaults.
The equivalent explicit construction is:

```python
from thermal_calibration import Camera

camera = Camera(
    width=256,
    height=192,
    fx=4.3 / 0.012,
    fy=4.3 / 0.012,
    cx=(256 - 1) / 2,
    cy=(192 - 1) / 2,
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
)

K_native = camera.matrix()
K_2x = camera.matrix(scale=2)
K_3x = camera.matrix(scale=3)
K_4x = camera.matrix(scale=4)
```

The higher-resolution reference grids preserve the same field of view and
pixel-footprint boundaries. Under the generator's integer-center convention,
the coordinate transformation at scale $s$ is

$$
u_s=s(u+\tfrac12)-\tfrac12,
\qquad
v_s=s(v+\tfrac12)-\tfrac12.
$$

Substituting the native projection gives

$$
f_{x,s}=sf_x,\qquad f_{y,s}=sf_y,
\qquad
c_{x,s}=s(c_x+\tfrac12)-\tfrac12,
\qquad
c_{y,s}=s(c_y+\tfrac12)-\tfrac12.
$$

| Grid | Dimensions | $f_x=f_y$ | $c_x$ | $c_y$ |
|---|---|---:|---:|---:|
| Native | 256 × 192 | 358.333333 | 127.5 | 95.5 |
| 2× | 512 × 384 | 716.666667 | 255.5 | 191.5 |
| 3× | 768 × 576 | 1075.000000 | 383.5 | 287.5 |
| 4× | 1024 × 768 | 1433.333333 | 511.5 | 383.5 |

These are coordinate representations of the same synthetic camera on different
sampling grids. They do not assert equivalent gains in resolved detail or
describe the internal implementation of X³IR.

The advertised field of view provides a useful consistency check. For a centered,
distortion-free camera with width $W$ and height $H$,

$$
\mathrm{HFOV}=2\arctan\left(\frac{W}{2f_x}\right),
\qquad
\mathrm{VFOV}=2\arctan\left(\frac{H}{2f_y}\right).
$$

The assumed matrix gives **39.3144° × 29.9955°**, compared with the manufacturer's
**40° × 30.2°**. The values are close but do not agree exactly. Using the advertised
FOV values directly would instead give approximately $f_x=351.68$ px and
$f_y=355.79$ px. The generator uses the focal-length/pitch derivation above;
these alternative estimates illustrate the limits of nominal specifications.
[Manufacturer FOV specification](https://thermalmaster.com/products/p3-thermal-camera-for-iphone-and-android)

The dime example follows directly from the same projection. For a face-on circular
target of diameter $D$ at axial distance $Z$, approximately on-axis and with
distortion neglected, its projected geometric diameter is

$$
d_{\mathrm{pixels}}=f_x\frac{D}{Z}.
$$

For the user-supplied diameter of 17.91 mm and distance of 1016 mm:

$$
d_{\mathrm{pixels}}
=358.333333\frac{17.91}{1016}
=6.316683\ \mathrm{pixels}
\approx 6.32\ \mathrm{pixels}.
$$

The 1016 mm lens-reference distance is being used as an approximation to distance
from the projection center. The **6.32-pixel value is a model prediction**, not an
independently fitted measurement of the real image. It describes geometric
diameter before optical blur and pixel integration. Depending on subpixel
position, a silhouette of this width can intersect 7 or 8 pixel columns before
blur; faint boundary pixels therefore should not simply be counted as full
pixels of object width.

Actual calibration should fit intrinsics and distortion from known target geometry
observed at multiple poses, then verify them on held-out captures. A dime at one
distance can check local scale, but cannot establish the complete intrinsic matrix.
PSF, detector noise, and temporal response require their own measurements; they
do not follow from this intrinsic-parameter derivation.
