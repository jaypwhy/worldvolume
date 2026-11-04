# Boundary Conditions and Analytic Spectra for the Worldvolume Klein–Gordon Solver

**Status:** reference specification for `worldvolume`. Revision 1, 2026-11-04.
**Author:** theorist. **Audience:** the engineer implementing the solver and its tests.

This document fixes, once and for all, the sign conventions, the normalisation
conventions, the analytic spectra, and the finite-difference scheme for the first
release of `worldvolume`. The reference table in §5 is generated from the closed
forms in §3; every number there is reproducible from a formula stated here.

**How to read the claim markers.** Throughout I tag statements:

- **[STD]** — standard textbook material. I state it because the implementation
  depends on it and because sources differ on conventions; the citation is to a
  standard reference, not to a specific equation.
- **[DER]** — derived here, in this document. The derivation is given in full so
  you can check it. Several of these I verified symbolically with SymPy 1.14 and
  numerically with mpmath 1.3 at 50-digit precision; those checks are listed in
  §7.
- **[VERIFIED]** — a numerical fact I established by running the scheme in §4.
  The numbers are quoted so you can reproduce them.

Nothing here is a research claim. This is a specification.

---

## 1. Setup and conventions

### 1.1 Units

**[STD]** We set

$$\hbar = c = 1 .$$

Consequently every quantity carries dimensions of a power of mass: frequencies
$\omega$ and masses $m$ are in mass units; lengths $L_i$ are in inverse mass
units; the boundary coupling $\kappa$ (§2.1) is in mass units. There is no
$\hbar$ anywhere in this document, and no factor of $c$. If you ever find
yourself wanting to insert one, something has gone wrong.

### 1.2 Metric signature and the action

**[STD]** The worldvolume is $(p+1)$-dimensional. We use the **mostly-minus**
("West Coast") signature

$$\eta_{ab} = \mathrm{diag}(+1, -1, -1, \dots, -1), \qquad a,b = 0,1,\dots,p .$$

The worldvolume coordinates are $\sigma^a$, with $\sigma^0 = t$ the time and
$\sigma^i$ the spatial worldvolume directions. We work with a single real scalar
field $\phi(t,\vec x)$ of mass $m$ and the action

$$S = \tfrac12 \int dt\, d^p x \; \Big[ (\partial_t \phi)^2 - (\nabla\phi)^2 - m^2 \phi^2 \Big],
\qquad (\nabla\phi)^2 \equiv \sum_{i=1}^{p} (\partial_i \phi)^2 . \tag{1.1}$$

This is the standard normalisation: the kinetic term has the signature-induced
relative minus sign between time and space derivatives, and the mass term enters
with a minus sign relative to the kinetic term. The overall $\tfrac12$ is the
conventional one for a real scalar; it cancels out of the equation of motion and
out of everything in this document.

Varying (1.1) gives the **Klein–Gordon equation**

$$\boxed{\;\partial_t^2 \phi - \nabla^2 \phi + m^2 \phi = 0\;} \tag{1.2}$$

equivalently $(\Box + m^2)\phi = 0$ with $\Box = \eta^{ab}\partial_a\partial_b = \partial_t^2 - \nabla^2$.

> **Convention warning (the mass sign).** Many texts write the KG equation as
> $(\Box - m^2)\phi = 0$ with $\Box = \partial_t^2 - \nabla^2$; that is the *same*
> equation with $m^2 \to -m^2$. With our convention (1.2) a **normal** particle has
> $m^2 > 0$, and the bosonic open-string tachyon has $m^2 = -1/\alpha'$ — negative.
> With the other convention the tachyon has $m^2 = +1/\alpha'$ — positive. If you
> compare against a source and the sign of the tachyon's $m^2$ looks wrong, this is
> why. Our convention is fixed by the requirement that a plane wave
> $e^{-i\omega t + i k x}$ have $\omega^2 = k^2 + m^2$ with $m^2>0$ for a normal
> particle — see §1.4.

### 1.3 The domain

**[STD]** The spatial worldvolume is the rectangular box

$$\Omega = [0, L_1]\times[0,L_2], \qquad p = 2, \tag{1.3}$$

with side lengths $L_1, L_2 > 0$. Coordinates: $x \equiv \sigma^1 \in [0,L_1]$,
$y \equiv \sigma^2 \in [0,L_2]$. The boundary $\partial\Omega$ consists of four
pieces:

$$x = 0,\quad x = L_1,\quad y = 0,\quad y = L_2 .$$

Time is **not** discretised and not part of the box: the solver works in the
frequency domain, and time dependence is restored at the end (§1.6). The first
release uses $p=2$; the 1D results in §3 are the building blocks for it and are
what the unit tests should be written against first.

### 1.4 Separation of variables and the spectral problem

**[STD]** Write $\phi(t,\vec x) = e^{-i\omega t} f(\vec x)$. Substituting into
(1.2):

$$-\omega^2 f - \nabla^2 f + m^2 f = 0
\quad\Longleftrightarrow\quad
-\nabla^2 f = \underbrace{(\omega^2 - m^2)}_{\textstyle \lambda} f . \tag{1.4}$$

So we have the **eigenvalue problem for the (positive) Laplacian**:

$$\boxed{\;-\Delta f = \lambda f \ \text{ on } \Omega, \qquad f \text{ obeying the assigned BC on each boundary,}\;} \tag{1.5}$$

and the Klein–Gordon frequency follows from

$$\boxed{\;\omega^2 = \lambda + m^2\;} \tag{1.6}$$

**Three consequences, and they matter for the implementation:**

1. **The mass $m$ does nothing to the eigenfunctions and nothing to the
   spectrum of $-\Delta$.** It enters only through the shift (1.6). Every
   boundary-condition subtlety in this document lives in $\lambda$, not in
   $\omega$. The solver should compute $\lambda_n$ from the BCs alone and apply
   (1.6) as a post-processing step. This separation is what makes the reference
   table in §5 testable in two independent stages.
2. $\omega$ is real only if $\lambda + m^2 \ge 0$. For $m^2<0$ (tachyonic field)
   or for Robin with $\kappa<0$ (§3.5.6) the lowest $\lambda$ can make
   $\lambda+m^2<0$, and then $\omega = i\sqrt{-(\lambda+m^2)}$: the mode grows as
   $e^{+|\omega| t}$. **The solver must return $\omega$ as a complex number or
   return $\omega^2$ and let the caller take the root.** It must not crash on a
   negative argument of `sqrt`, and it must not silently take `abs`.
3. $\lambda$ is dimensionless-free but has dimensions of mass$^2$; with $L$ in
   inverse mass, $\lambda \sim 1/L^2$ as expected.

### 1.5 Inner product and normalisation

**[STD]** The natural inner product is

$$\langle f, g\rangle = \int_\Omega f^* g \, d^p x , \qquad \|f\|^2 = \langle f,f\rangle . \tag{1.7}$$

Because $-\Delta$ with any of the boundary conditions of §2 is self-adjoint on
this domain (§2.3), eigenfunctions belonging to distinct eigenvalues are
orthogonal in (1.7) and we may normalise

$$\langle f_m, f_n \rangle = \delta_{mn} . \tag{1.8}$$

**All eigenfunctions in §3 are normalised by (1.8) — unit $L^2$ norm, no
$1/\sqrt{2\omega}$ factor.** The $1/\sqrt{2\omega_n}$ of canonical quantisation is
a separate, later choice belonging to the mode expansion, not to the
eigenproblem. Conflating the two is trap #1 in §6.

### 1.6 Mode expansion and time evolution

**[STD]** With $\{f_n, \lambda_n\}$ from (1.5) and $\omega_n$ from (1.6), the
general real solution of (1.2) is

$$\phi(t,\vec x) = \sum_n \Big( a_n\, f_n(\vec x)\, e^{-i\omega_n t} + a_n^*\, f_n^*(\vec x)\, e^{+i\omega_n t}\Big) \tag{1.9}$$

for complex coefficients $a_n$. (For real $\lambda_n$ the $f_n$ may be taken
real, and then $f_n^* = f_n$; the $\sin$/$\cos$ modes in §3 are real.) If a
canonical normalisation is wanted, insert $1/\sqrt{2\omega_n}$ in (1.9); that is a
choice about the coefficients $a_n$, not about $f_n$.

For the time-evolution feature, the practical statement is: given initial data
$\phi(0,\vec x)$ and $\partial_t\phi(0,\vec x)$, project onto the $f_n$, evolve
each coefficient by $e^{\mp i\omega_n t}$, and resum. For $\omega_n$ imaginary
the exponential is real and growing/decaying as in §1.4 item 2.

### 1.7 Factors of $2\pi$ — read this before writing any wavenumber

**[STD]** These are the only places $2\pi$ can enter, and getting them wrong
produces a plausible spectrum that is wrong by a fixed factor.

1. **Box, not circle.** For an interval of length $L$ with Dirichlet or Neumann
   ends, the allowed wavenumbers are
   $$k_n = \frac{n\pi}{L}, \qquad n \in \mathbb{Z}_{\ge 0}\ \text{or}\ \mathbb{Z}_{\ge 1},$$
   **not** $2\pi n /L$. The latter is the momentum on a *circle* of circumference
   $L$ (periodic boundary conditions), i.e. the KK momentum $n/R$ with
   $L = 2\pi R$. Equivalently $k_n = 2\pi n/(2L)$: a standing wave on $[0,L]$ is
   half a period of a wave of wavelength $2L/n$. Careful with the comparison:
   a Dirichlet/Neumann box of length $L$ and a periodic circle of circumference
   $2L$ have the **same set of $|k|$ values** ($m\pi/L$) but **not the same
   multiplicities** — the box has one mode per $k>0$, while the circle has two
   (cos and sin) per $k>0$ plus the zero mode. So "same wavenumbers" does not
   mean "same spectrum"; and a circle of circumference $L$ has neither.
2. **Density of states.** Because the box modes are labelled by $n \ge 0$ only
   (or $n \ge 1$), the mode spacing is $\Delta k = \pi/L$ and
   $$\sum_{n} \;\longrightarrow\; \frac{L}{\pi}\int_0^\infty dk .$$
   On a circle the spacing is $2\pi/L$ but $k$ runs over all of $\mathbb{R}$, and
   the total count agrees. Using $L/2\pi$ with $k>0$ halves your mode count — a
   real error I have seen in mode-sum code.
3. **The factor 2 in the normalisation.** Non-zero modes carry $\sqrt{2/L}$
   (the $2$ is $\int_0^L\cos^2 = L/2$), but the Neumann **zero mode** carries
   $1/\sqrt{L}$. §3.3.
4. **Robin coupling.** $\kappa$ has dimensions of mass. The dimensionless
   combination is $\tilde\kappa = \kappa L$; the transcendental equation in
   dimensionless form is (3.15). If a test of the Robin spectrum fails by a
   factor of $L$, it is this.

### 1.8 Summary of conventions that differ between sources

| Quantity | This document | Common alternative | Note |
|---|---|---|---|
| Signature | $(+,-,-,-)$ | $(-,+,+,+)$ | flips the overall sign of $\Box$; the spectrum is unaffected |
| KG operator | $(\Box + m^2)\phi = 0$ | $(\Box - m^2)\phi=0$ | flips the sign of $m^2$; tachyon is $m^2<0$ here |
| Spectral problem | $-\Delta f = \lambda f$, $\omega^2 = \lambda + m^2$ | $+\Delta f = -\lambda f$ | same content |
| Robin condition | $\partial_n\phi + \kappa\phi = 0$ | $\partial_n\phi = \kappa\phi$, or $\partial_n\phi + h\phi = 0$ with $h>0$ | **flips the sign of $\kappa$**; see §2.2 |
| Normal derivative | outward normal | inward normal | flips the sign at $x=0$ only — the classic bug, §6 |
| Eigenfunction norm | $\int|f|^2 = 1$ | $1/\sqrt{2\omega}$ included | §1.5 |
| Wavenumber | $n\pi/L$ | $2\pi n/L$ | box vs circle, §1.7 |

---

## 2. Boundary conditions and their D-brane meaning

### 2.1 The three conditions, stated precisely

Let $\hat n$ be the **outward** unit normal to $\partial\Omega$. On the four
pieces of $\partial\Omega$ of the box (1.3):

| boundary | outward normal $\hat n$ | $\partial_n$ |
|---|---|---|
| $x = 0$ | $-\hat x$ | $-\partial_x$ |
| $x = L_1$ | $+\hat x$ | $+\partial_x$ |
| $y = 0$ | $-\hat y$ | $-\partial_y$ |
| $y = L_2$ | $+\hat y$ | $+\partial_y$ |

**The sign of $\partial_n$ at the lower end of each interval is negative. This is
the single most common source of silently wrong spectra in this problem; see
§6 item 1.**

**[STD]** The three boundary conditions are:

$$\textbf{Neumann (N):}\quad \partial_n \phi = 0 \quad\text{on }\partial\Omega. \tag{2.1}$$
$$\textbf{Dirichlet (D):}\quad \phi = 0 \quad\text{on }\partial\Omega. \tag{2.2}$$
$$\textbf{Robin (R):}\quad \partial_n \phi + \kappa\,\phi = 0 \quad\text{on }\partial\Omega, \qquad \kappa \in \mathbb{R}. \tag{2.3}$$

Written out with the signs from the table, on a 1D interval $[0,L]$:

$$\text{at } x=0:\quad -\phi'(0) = 0\ \ (N), \quad \phi(0)=0\ \ (D), \quad -\phi'(0) + \kappa_0\phi(0) = 0\ \ (R) \tag{2.4a}$$
$$\text{at } x=L:\quad +\phi'(L) = 0\ \ (N), \quad \phi(L)=0\ \ (D), \quad +\phi'(L) + \kappa_L\phi(L) = 0\ \ (R) \tag{2.4b}$$

Equivalently, as conditions on the derivative alone:

$$(N):\ \phi'(0)=0,\ \phi'(L)=0; \qquad (R):\ \phi'(0)=+\kappa_0\phi(0),\ \ \phi'(L)=-\kappa_L\phi(L). \tag{2.5}$$

**Note the relative minus sign in (2.5).** With the same value of $\kappa$ at
both ends, the derivative conditions at the two ends have *opposite* signs. An
implementation that writes $\phi'(0) = \kappa\phi(0)$ **and**
$\phi'(L) = \kappa\phi(L)$ is solving a different, perfectly well-posed,
self-adjoint problem — and it will converge beautifully to the wrong answer.
§6 item 1 gives the numeric signature of that bug.

The implementation should carry $\kappa$ **per boundary piece**, i.e. four
numbers $\kappa_{x0}, \kappa_{xL}, \kappa_{y0}, \kappa_{yL}$ for the box, with
$\kappa = 0$ meaning Neumann and $\kappa = \infty$ meaning Dirichlet. Storing the
condition as a tagged value (`("N",)`, `("D",)`, `("R", kappa)`) is clearer than
abusing $\kappa$, but §3.5 shows that $\kappa=0$ really is Neumann and
$\kappa\to\infty$ really is Dirichlet, so the degenerate encodings are legitimate
if you prefer them.

### 2.2 The limits $\kappa\to 0$ and $\kappa\to\infty$

**[DER]** Both limits follow immediately from (2.3) and are proved again at the
level of the spectrum in §3.5.4:

- $\kappa \to 0$: (2.3) becomes $\partial_n\phi = 0$ — **Neumann**, exactly, for
  every mode.
- $\kappa \to \infty$: divide (2.3) by $\kappa$ to get $\phi + \partial_n\phi/\kappa = 0$;
  as $\kappa\to\infty$, $\phi\to 0$ — **Dirichlet**, exactly.

So in our convention $\kappa>0$ interpolates continuously from Neumann
($\kappa=0$) to Dirichlet ($\kappa=\infty$), and this is the reason for the
choice of sign in (2.3). **A source that writes the Robin condition as
$\partial_n\phi = \kappa\phi$ has the opposite sign convention: there
$\kappa=0$ is Neumann but $\kappa\to+\infty$ is *anti*-Dirichlet and it is
$\kappa\to-\infty$ that gives Dirichlet.** If you take the Robin formulas below
from a source with the other sign, you must flip $\kappa$.

**[DER]** For a finite positive $\kappa$, the spectrum is strictly inside the two
limits in the following precise sense: writing $k_n(\kappa)$ for the $n$-th
wavenumber of the symmetric Robin problem with $\kappa$ at both ends (so
$k_n(0) = n\pi/L$),

$$k_n(\kappa) = \frac{n\pi}{L}\left(1 - \frac{2}{\kappa L}\right) + O(\kappa^{-2}), \qquad \kappa\to\infty, \tag{2.6}$$

i.e. the approach to Dirichlet is from *below* at relative rate $2/(\kappa L)$.
I verified (2.6) numerically: for $L=1$, $\kappa=10^6$, the first root is
$k_1 = 3.141586370417$ against the prediction $\pi(1-2\times10^{-6}) = 3.141586370404$,
agreeing to $1.3\times10^{-11}$.

### 2.3 Self-adjointness

**[STD]** For $f,g$ obeying the boundary conditions, Green's first identity gives

$$\int_\Omega \big( f\,(-\Delta g) - g\,(-\Delta f)\big)\,d^px = \int_{\partial\Omega}\big( g\,\partial_n f - f\,\partial_n g\big)\,dS . \tag{2.7}$$

The operator $-\Delta$ is self-adjoint on the domain cut out by the boundary
conditions iff the boundary term vanishes. It does, piece by piece:

- **Dirichlet on a piece:** $f = g = 0$ there, so the integrand vanishes.
- **Neumann on a piece:** $\partial_n f = \partial_n g = 0$ there.
- **Robin on a piece:** $\partial_n f = -\kappa f$ and $\partial_n g = -\kappa g$, so
  $g\,\partial_n f - f\,\partial_n g = -\kappa fg + \kappa fg = 0$ — **for any real $\kappa$,
  including different $\kappa$ on different boundary pieces**, and including
  $\kappa < 0$.

Hence **[STD]** the spectrum is real, eigenfunctions of distinct eigenvalues are
orthogonal in (1.7), and the eigenvalues are bounded below. Three consequences
the implementation relies on:

1. $\lambda_n \in \mathbb{R}$ always. If your eigensolver returns complex
   eigenvalues with non-zero imaginary part, you have a bug (or you assembled a
   non-symmetric matrix — see §4.3).
2. The spectrum is discrete and accumulates only at $+\infty$ (compact domain,
   coercive form), so "the lowest $N$ modes" is well defined.
3. **Sign of the spectrum.** Multiplying the eigenvalue equation by $f^*$ and
   integrating by parts, using (2.3):
   $$\lambda \int_\Omega |f|^2 = \int_\Omega |\nabla f|^2 + \sum_{\text{Robin pieces}}\kappa_i \int_{\text{piece}} |f|^2 . \tag{2.8}$$
   So $\lambda \ge 0$ whenever all Robin couplings are $\ge 0$ (with all
   Dirichlet/Neumann pieces contributing nothing), and $\lambda > 0$ unless
   $f$ is constant and every Robin $\kappa_i = 0$. **Negative $\kappa$ can and
   does produce negative eigenvalues**; §3.5.6.

### 2.4 The D-brane / open-string reading

**[STD]** In perturbative open-string theory, the embedding coordinates
$X^\mu(\tau,\sigma)$, $\mu = 0,\dots,9$, of a string with worldsheet boundary
$\sigma = 0,\pi$ obey, on each boundary, either

$$\partial_\sigma X^\mu = 0 \quad(\text{Neumann}) \qquad\text{or}\qquad X^\mu = \text{const} \quad(\text{Dirichlet}). \tag{2.9}$$

The physical content is:

- **Neumann direction** — the string endpoint is free to move in that spacetime
  direction. The endpoints of all open strings sweep out a hypersurface, and
  that hypersurface is a **D-brane**: the brane *extends along* every Neumann
  direction. The Neumann directions are the brane's **worldvolume** directions.
- **Dirichlet direction** — the string endpoint is pinned to a fixed value of
  $X^\mu$. The brane does *not* extend in that direction: it is a **transverse**
  direction, and the brane sits at a definite position in it.

A D$p$-brane in $D=10$ has $p+1$ Neumann directions (including $X^0$, since
endpoints move in time) and $9-p$ Dirichlet directions. This is the content of
the original Dirichlet-brane papers [1,2]; a textbook account with the open-string
mode expansions is [3,4] and [5, Vol. 1].

**[STD]** Two standard facts that the *field-theory* problem mirrors exactly, and
which justify the boundary conditions we impose on $\phi$:

- **T-duality exchanges N and D.** T-duality along a direction of radius $R$
  maps $R \to \alpha'/R$ and swaps $\partial_\sigma X = 0 \leftrightarrow X = \text{const}$.
  A brane extending along the direction maps to a brane localised in it. This is
  why the Neumann and Dirichlet problems have the *same* spectral structure up to
  relabelling (§3.2, §3.3), and why the mixed N/D interval of §3.4 is T-dual to a
  Neumann interval on a circle.
- **Momentum versus winding.** In a Neumann direction the endpoint has
  continuous momentum (quantised, $p = n/R$, if the direction is compact of
  radius $R$); in a Dirichlet direction there is no momentum in that direction
  but there is a fixed separation, which in the string channel becomes winding.

The field $\phi$ on the worldvolume inherits the same boundary conditions: a
worldvolume scalar fluctuation is a function of the worldvolume coordinates and
obeys N on the brane's own directions and D on the transverse ones. That is the
problem (1.5) we are solving.

### 2.5 The intersecting-brane case, and what it does to the mode structure

**[STD]** Consider two branes that intersect, or a single brane whose worldvolume
directions are split. The case the first release must support is:

> A worldvolume with **some Neumann directions and some Dirichlet directions** —
> e.g. a brane that extends along $x$ but is localised in $y$ (a "D1 inside a
> D2", or the worldvolume of a brane ending on another brane).

For the box (1.3) this means the boundary conditions are assigned **per
direction**:

- $x$ Neumann: $\partial_x\phi = 0$ at $x=0$ and $x=L_1$;
- $y$ Dirichlet: $\phi = 0$ at $y=0$ and $y=L_2$.

**The mode structure this produces** (proved in §3.6):

- The eigenfunctions **factorise**, $f_{n_1 n_2}(x,y) = f_{n_1}(x)\,f_{n_2}(y)$,
  with each factor carrying its own direction's condition.
- The eigenvalues **add**: $\lambda_{n_1 n_2} = k^{(1)2}_{n_1} + k^{(2)2}_{n_2}$.
- The **mode-number ranges differ per direction**: for a Neumann direction
  $n_i \ge 0$ (the $n_i = 0$ mode is the constant one, which exists); for a
  Dirichlet direction $n_i \ge 1$ (the constant mode is excluded). This is the
  practical meaning of "mixed": the two directions do not have the same spectrum
  of mode labels, and the ground state of the box is *not* the constant
  function. For the mixed $N_x D_y$ box the ground state is
  $f_{0,1} = L_1^{-1/2}\sqrt{2/L_2}\,\cos(0)\sin(\pi y/L_2)$, i.e. constant in
  $x$, half-wave in $y$, with
  $\lambda_{01} = (\pi/L_2)^2$. There is no zero mode at all.
- **Degeneracies are typically accidental.** For generic (incommensurate)
  $L_1/L_2$ the map $(n_1,n_2)\mapsto\lambda$ is injective and every level is
  non-degenerate; degeneracies appear when $L_1/L_2$ is rational, or by
  number-theoretic accident when $L_1 = L_2$. §3.6.2.

### 2.6 Scope note: field modes versus string modes

**[STD]** One point of terminology, because it is a live source of confusion.

- **This document** solves the *field* problem (1.5) on the worldvolume. Its
  wavenumbers are $n\pi/L$, $(n+\tfrac12)\pi/L$, or the Robin roots — always
  *fixed by the boundary conditions on the interval*.
- The **open-string** mode expansion is a different problem: for a string
  stretched between two branes with $d$ ND directions, the *oscillator* modes in
  the ND directions become half-integer, and the zero-point energy is shifted
  accordingly. Those half-integer oscillator labels are **not** the
  $(n+\tfrac12)\pi/L$ of §3.4 — they are a different quantity (oscillator index
  versus wavenumber), and they arise from the string's own $\sigma$-dependence,
  not from the field's boundary-value problem.

The numerical coincidence that both involve "half-integer" is exactly that. Do
not derive §3.4 from the ND string spectrum, and do not use §3.4 to predict
string oscillator shifts. §3.4 is the spectrum of $-\Delta$ on an interval with
one Neumann and one Dirichlet end — that is all, and that is what the solver
needs.

---

## 3. Analytic spectra

This is the heart of the document. Everything is stated for the 1D interval
$[0,L]$ and assembled in 2D in §3.6.

Throughout: $\lambda = k^2$ is the eigenvalue of $-\Delta$, $f$ the normalised
eigenfunction, and $\omega^2 = \lambda + m^2$ by (1.6).

### 3.1 The general 1D solution

**[STD]** For $\lambda = k^2 > 0$, the general solution of $-f'' = k^2 f$ is

$$f(x) = A\cos(kx) + B\sin(kx). \tag{3.1}$$

The boundary conditions at $x=0$ and $x=L$ give two homogeneous linear equations
in $(A,B)$; a non-trivial solution exists iff the determinant vanishes, which is
the **spectral condition** on $k$. Every case below is that one computation, done
explicitly.

The derivative is $f'(x) = -Ak\sin(kx) + Bk\cos(kx)$, so
$f(0) = A$, $f'(0) = Bk$, $f(L) = A\cos(kL) + B\sin(kL)$,
$f'(L) = -Ak\sin(kL) + Bk\cos(kL)$.

**Remember (2.4): the condition at $x=0$ involves $-\phi'(0)$, not $+\phi'(0)$.**

### 3.2 Pure Dirichlet: $\phi(0) = \phi(L) = 0$

**[STD]**

*Derivation.* $\phi(0) = A = 0$. Then $\phi(L) = B\sin(kL) = 0$ with $B\neq0$,
so $\sin(kL) = 0$, i.e. $kL = n\pi$ with $n \in \mathbb{Z}$. Since $k>0$ and
$k\leftrightarrow -k$ give the same function up to sign, the distinct modes are
$n = 1,2,3,\dots$ — **$n=0$ is excluded because $\sin 0 \equiv 0$**. $\square$

$$\boxed{\;k_n = \frac{n\pi}{L},\quad f_n(x) = \sqrt{\frac{2}{L}}\,\sin\!\Big(\frac{n\pi x}{L}\Big),\quad \lambda_n = \Big(\frac{n\pi}{L}\Big)^2,\qquad n = 1,2,3,\dots\;} \tag{3.2}$$

with $\omega_n^2 = (n\pi/L)^2 + m^2$. Normalisation:
$\int_0^L \sin^2(n\pi x/L)dx = L/2$, hence $\sqrt{2/L}$. Orthogonality of
distinct $n$ is elementary. No zero mode.

### 3.3 Pure Neumann: $\phi'(0) = \phi'(L) = 0$

**[STD]**

*Derivation.* From (2.4a) with $\kappa=0$: $-\phi'(0) = -Bk = 0$, so $B=0$ and
$f = A\cos(kx)$. Then $+\phi'(L) = -Ak\sin(kL) = 0$ with $A\neq0$, so
$\sin(kL) = 0$, $kL = n\pi$, $n \in \mathbb{Z}_{\ge0}$. Here **$n=0$ survives**:
$k=0$ gives $f = A\cos 0 = \text{const}$, which does satisfy $\phi'=0$ at both
ends. It must be handled separately for the normalisation, because
$\cos(kx)\to1$ as $k\to0$ is not normalised the same way as the others. $\square$

$$\boxed{\;k_n = \frac{n\pi}{L},\quad \lambda_n = \Big(\frac{n\pi}{L}\Big)^2,\qquad n = 0,1,2,\dots\;} \tag{3.3a}$$

$$f_0(x) = \frac{1}{\sqrt{L}}, \qquad f_n(x) = \sqrt{\frac{2}{L}}\,\cos\!\Big(\frac{n\pi x}{L}\Big)\ \ (n\ge1). \tag{3.3b}$$

**The zero mode.** $k_0 = 0$, $\lambda_0 = 0$, $f_0 = L^{-1/2}$. Its
normalisation is $\int_0^L (L^{-1/2})^2 dx = 1$ — note the **factor
$\sqrt{L}$ rather than $\sqrt{2/L}$**, i.e. $\sqrt2$ different from the
non-zero modes. Then
$$\omega_0^2 = m^2,\qquad \omega_0 = |m| . \tag{3.3c}$$
For $m=0$ this is a genuine massless mode with $\omega_0 = 0$: a static,
spatially constant solution of (1.2). **It is physical, not a numerical
artefact, and it must appear in the output.** §6 item 2.

Two traps specific to this case:

- The zero mode is the *only* mode whose $L^2$ norm is $L$ rather than $L/2$
  before normalisation. A loop that applies $\sqrt{2/L}$ uniformly gets
  $f_0 = \sqrt{2/L}$, whose norm is $2$, not $1$ — off by $\sqrt2$.
- The discrete spectrum must contain an eigenvalue that is **exactly** $0$ when
  both ends are Neumann (to machine precision, not approximately). §4.7.

### 3.4 Mixed Neumann/Dirichlet on a single interval — the mode-number condition

This is the case the brief flags, and the one where naive implementations go
wrong. **[DER]** (elementary, but the mode range needs care.)

Take the interval $[0,L]$ with **Neumann at $x=0$ and Dirichlet at $x=L$**:

$$\phi'(0) = 0, \qquad \phi(L) = 0 . \tag{3.4}$$

*Derivation.* $-\phi'(0) = -Bk = 0 \Rightarrow B = 0$, so
$\phi = A\cos(kx)$. Then $\phi(L) = A\cos(kL) = 0$ with $A\neq0$:

$$\cos(kL) = 0 \quad\Longrightarrow\quad kL = \tfrac{\pi}{2} + n\pi = \Big(n+\tfrac12\Big)\pi,
\qquad n \in \mathbb{Z}. \tag{3.5}$$

**Which $n$ are allowed.** $k>0$, and $k$ and $-k$ give the same function. The
set (3.5) is symmetric about $0$: $n=-1$ gives $k = -\pi/(2L) = -k_0$, i.e.
$n=-1$ duplicates $n=0$ (I checked this symbolically: $k_{-1} = -\pi/2L$,
$|k_{-1}| = |k_0|$). So the **distinct** modes are

$$\boxed{\;k_n = \frac{(2n+1)\pi}{2L} = \frac{(n+\tfrac12)\pi}{L},\qquad n = 0,1,2,\dots\;} \tag{3.6}$$

i.e. the **positive half-integer multiples of $\pi/L$** — equivalently the odd
multiples of $\pi/(2L)$. The three statements to internalise:

- **Not** $n\pi/L$. The spectrum is *not* the Dirichlet spectrum.
- **Not** $(n-\tfrac12)\pi/L$ starting at $n=1$ either — that is the same set
  relabelled, and it is fine, but then $n$ starts at $1$ and there is no
  "$n=0$" term. Pick one labelling and use it everywhere.
- **The index set is $n \ge 0$, countably infinite, no zero mode, and the lowest
  wavenumber is $\pi/(2L)$** — half the Dirichlet fundamental. This is a real
  physical difference, not a convention: the mode has a node at one end only, so
  it is a quarter-wavelength, not a half-wavelength, that fits in $L$.

$$\boxed{\;f_n(x) = \sqrt{\frac{2}{L}}\,\cos\!\Big(\frac{(n+\tfrac12)\pi x}{L}\Big),
\qquad \lambda_n = \Big(\frac{(n+\tfrac12)\pi}{L}\Big)^2,\qquad n=0,1,2,\dots\;} \tag{3.7}$$

*Normalisation.* $\int_0^L\cos^2(k_n x)dx = L/2$ whenever $\cos(k_nL) = 0$
(use $\int\cos^2 = L/2 + \sin(2kL)/(4k)$ and $\sin(2k_nL) = 2\sin(k_nL)\cos(k_nL) = 0$),
so $\sqrt{2/L}$ is right and **no special case is needed for $n=0$**. I verified
$\int_0^L f_n f_m = \delta_{nm}$ symbolically for
$(n,m) \in \{(0,0),(0,1),(0,3),(2,5)\}$.

**The mirrored case** — Dirichlet at $x=0$, Neumann at $x=L$ — has the *same*
spectrum and eigenfunctions $\sqrt{2/L}\sin((n+\tfrac12)\pi x/L)$:

$$\phi(0) = A = 0,\quad \phi'(L) = Bk\cos(kL) = 0 \Rightarrow \cos(kL)=0 . \tag{3.8}$$

Same $k_n$, same $\lambda_n$, $n\ge0$. **So "ND" and "DN" are isospectral**; the
solver should assert this as a symmetry test.

**Consistency check via images [DER].** Extend $f_n(x) = \cos(k_nx)$ evenly about
$x=0$ to $[-L,L]$. Since $f_n$ is even and $f_n(\pm L) = 0$, the extension is a
Dirichlet eigenfunction of $-\Delta$ on the interval $[-L,L]$ of length $2L$.
The Dirichlet modes on $[-L,L]$ are $\sin(m\pi(x+L)/(2L))$ with $k = m\pi/(2L)$,
and the even ones are exactly $m$ odd, i.e. $k = (2n+1)\pi/(2L)$ — precisely
(3.6). Two consequences worth using as tests:

- The ND spectrum on $[0,L]$ is the **odd-$m$ half** of the Dirichlet spectrum
  on the doubled interval $[0,2L]$.
- The mode *density* is therefore the same as the Dirichlet density on
  $[0,L]$: spacing $\pi/L$, not $\pi/(2L)$. Do not double the mode count
  because the wavenumbers look "half-integer".

### 3.5 Robin: $\partial_n\phi + \kappa\phi = 0$

Let $\kappa_0$ be the coupling at $x=0$ and $\kappa_L$ at $x=L$, so that by
(2.4)

$$-\phi'(0) + \kappa_0\phi(0) = 0, \qquad +\phi'(L) + \kappa_L\phi(L) = 0 . \tag{3.9}$$

$\kappa_0 = 0$ is Neumann at $x=0$; $\kappa_0\to\infty$ is Dirichlet at $x=0$;
and similarly at $x=L$. This single two-parameter family therefore **contains
every 1D case above as a limit**, which makes it the right thing to implement
first and the best single test of the whole scheme.

#### 3.5.1 The spectral condition [DER]

With $f = A\cos kx + B\sin kx$ and (3.9):

$$(3.9\text{a}):\quad -Bk + \kappa_0 A = 0 \quad\Longrightarrow\quad B = \frac{\kappa_0}{k}A . \tag{3.10}$$

$$(3.9\text{b}):\quad A\big(-k\sin kL + \kappa_L\cos kL\big) + B\big(k\cos kL + \kappa_L\sin kL\big) = 0 . \tag{3.11}$$

Substituting $B = (\kappa_0/k)A$ and dividing by $A\neq0$:

$$\Big(\frac{\kappa_0\kappa_L}{k} - k\Big)\sin(kL) + (\kappa_0+\kappa_L)\cos(kL) = 0,$$

i.e.

$$\boxed{\;(\kappa_0\kappa_L - k^2)\,\sin(kL) + (\kappa_0+\kappa_L)\,k\,\cos(kL) = 0\;} \tag{3.12}$$

or, where $\cos(kL)\neq0$,

$$\boxed{\;\tan(kL) = \frac{k\,(\kappa_0+\kappa_L)}{k^2 - \kappa_0\kappa_L}\;.} \tag{3.13}$$

I derived (3.12)–(3.13) by hand and re-derived them with SymPy from the $2\times2$
determinant of the boundary system; the two agree.

**Use (3.12), not (3.13), to find roots.** The $\tan$ form has poles at
$kL = (n+\tfrac12)\pi$ and a root-finder that naively bisects on it will either
miss roots or report a pole as a root. The determinant form (3.12) is smooth in
$k$, and its roots are exactly the eigenvalues. §6 item 4.

**The symmetric case** $\kappa_0 = \kappa_L = \kappa$ (the one to test first):

$$\tan(kL) = \frac{2\kappa k}{k^2-\kappa^2},\qquad\text{equivalently}\qquad
(\kappa^2-k^2)\sin(kL) + 2\kappa k\cos(kL) = 0 . \tag{3.14}$$

**Dimensionless form.** With $x = kL$ and $\tilde\kappa_i = \kappa_i L$,

$$(\tilde\kappa_0\tilde\kappa_L - x^2)\sin x + (\tilde\kappa_0+\tilde\kappa_L)\,x\cos x = 0,
\qquad \tan x = \frac{x(\tilde\kappa_0+\tilde\kappa_L)}{x^2-\tilde\kappa_0\tilde\kappa_L}. \tag{3.15}$$

Use (3.15) for root-finding when you want the answer independent of the units of
$L$: the roots $x_n$ depend only on $\tilde\kappa_i$, and $k_n = x_n/L$. This is
the cleanest way to test the Robin code, and it removes the factor-of-$L$ class
of bug in §1.7 item 4.

#### 3.5.2 The eigenfunctions and their normalisation [DER]

From (3.10), with $A$ fixed by normalisation,

$$f(x) = N\Big(\cos kx + \frac{\kappa_0}{k}\sin kx\Big), \tag{3.16}$$

which automatically satisfies the condition at $x=0$. Write $c \equiv \kappa_0/k$.
The normalisation integral is

$$I(k) \equiv \int_0^L\Big(\cos kx + c\sin kx\Big)^2 dx
= \frac{L}{2}(1+c^2) + \frac{1-c^2}{4k}\sin(2kL) + \frac{c}{k}\sin^2(kL), \tag{3.17}$$

so $N = I(k)^{-1/2}$. I derived (3.17) by hand and confirmed it with SymPy; I
also checked it against direct numerical quadrature at 50 digits for the
$\kappa_0=\kappa_L=2$, $L=1$ modes, where the closed form and the quadrature agree
to all 50 digits, the boundary residuals are $\le 10^{-49}$, and distinct modes
are orthogonal to $\le 10^{-50}$.

Two limits of (3.17) worth checking in a unit test:

- $c = 0$ (Neumann at $x=0$) and $kL = n\pi$: $I = L/2$, recovering (3.3b) with
  $\sqrt{2/L}$ (and $I=L$ for $k=0$, recovering $1/\sqrt L$).
- $c\to\infty$ (Dirichlet at $x=0$): the bracket in (3.16) is dominated by
  $\sin kx$, and $I \to L/2$ at $kL = n\pi$, recovering (3.2).

**Do not try to simplify $I(k)$ using the spectral condition** to get a closed
form like "$L/2$ always" — it is not $L/2$ except in the Neumann/Dirichlet
limits. For the Case D parameters in §5, $I(k_1) = 1.851033886878379$,
$I(k_2) = 0.742962685095034$, $I(k_3) = 0.585216171650906$: not constant, and
not $L/2 = 0.5$.

#### 3.5.3 Root structure and ordering [DER], [VERIFIED]

- **[DER] There is at least one root in each interval $(n\pi, (n+1)\pi)$ for
  $kL$**, $n = 0,1,2,\dots$, because at $kL = m\pi$ the determinant (3.12) equals
  $(\kappa_0+\kappa_L)m\pi(-1)^m$, whose sign alternates with $m$ (for
  $\kappa_0+\kappa_L>0$). So a scan for sign changes cannot *miss* a root, and
  the $n$-th smallest root satisfies $(n-1)\pi < k_nL < n\pi$.
- **[VERIFIED] For the couplings I tested there is exactly one root per
  interval** ($\kappa$ up to $10^4$ symmetric, and the asymmetric cases of
  §5), so "the $n$-th root" and "the root in the $n$-th interval" coincide. For
  Case D ($\kappa=2$, $L=1$) the roots are
  $kL = 1.7207,\ 4.0575,\ 6.8512,\ 9.8264,\ 12.8746,\ 15.9573,\ 19.0587$, i.e.
  one in each of $(0,\pi), (\pi,2\pi), (2\pi,3\pi),\dots$ — note that they are
  *not* near the half-integer points except asymptotically.
- **Small $\kappa$ is the delicate regime.** As $\kappa\to0^+$ the first root
  $k_1\to0$ (it becomes the Neumann zero mode, §3.5.5), so the first interval
  contains a root arbitrarily close to its left end. Do not assume
  $k_1 = O(1/L)$; the value depends on $\kappa$ by the table in §3.5.5.
- **Ordering matters.** The reference table's "$n$-th mode" always means the
  $n$-th *smallest positive root*, not the $n$-th root in a $\tan$-branch
  enumeration. These coincide for $\kappa>0$ but not for mixed-sign couplings.
- **Counting roots.** Since the roots interlace the poles, a scan for sign
  changes of (3.12) on a grid with step $< \pi/(2L)$ cannot miss a root, and a
  grid with step $< \pi/(4L)$ is comfortably safe. Use the determinant, bracket
  the root, then bisect or use a bracketed Newton. Do not start the scan at
  $k=0$ with a sign-change test only, because (3.12) has a *degenerate* zero at
  $k=0$ for special couplings (§3.5.5).

#### 3.5.4 The limits [DER], [VERIFIED]

From (3.12), dividing by $\kappa_0\kappa_L$ when both are large:

- $\kappa_0\to\infty,\ \kappa_L\to\infty$: $(1 - k^2/\kappa_0\kappa_L)\sin(kL) + k(1/\kappa_L + 1/\kappa_0)\cos(kL) = 0 \Rightarrow \sin(kL) = 0$ — **Dirichlet–Dirichlet** (3.2). ✓
- $\kappa_0 = 0$ (Neumann at $x=0$), $\kappa_L$ finite: (3.12) becomes $-k^2\sin(kL) + \kappa_L k\cos(kL) = 0 \Rightarrow \cos(kL) = 0$ — **ND** (3.6). ✓ (A single Robin end already forces the half-integer set!)
- $\kappa_0 = 0 = \kappa_L$: $\sin(kL) = 0$ — **Neumann–Neumann** (3.3). ✓
- $\kappa_0 = 0$, $\kappa_L\to\infty$: divide (3.12) by $\kappa_L$: $-k^2\sin(kL)/\kappa_L + k\cos(kL) \to 0 \Rightarrow \cos(kL) = 0$ — **ND** again. ✓

I confirmed all four limits numerically: with $\kappa_0=\kappa_L=10^8$ the roots
reproduce $n\pi$ to 8 digits; with $\kappa_0 = 0$, $\kappa_L = 10^8$ they
reproduce $(n+\tfrac12)\pi$ to 8 digits; with $\kappa_0=\kappa_L=0$ they are
$n\pi$ to 12 digits.

**Also [VERIFIED]:** with $\kappa_0 = \kappa_L = 10^6$, the discrete FD spectrum
in §4 reproduces the Dirichlet convergence table *exactly* to all printed
digits — the Robin code path and the Dirichlet code path agree in the limit,
which is a strong end-to-end test.

#### 3.5.5 Is there a zero mode? [DER], [VERIFIED]

**At $k=0$ the spectral condition (3.12) carries no information.** Every term
vanishes: $(\kappa_0\kappa_L - 0)\sin 0 = 0$ and $(\kappa_0+\kappa_L)\cdot0\cdot\cos0 = 0$.
So $k=0$ satisfies (3.12) *identically, for every $\kappa_0,\kappa_L$* — it is a
degeneracy of the equation, not a statement about eigenvalues. At $k=0$ the two
basis functions $\cos(kx) = 1$ and $\sin(kx)/k \to x$ are still independent, so
the $k=0$ case must be solved separately. With $f = A + Bx$:

$(3.9\text{a}):\ -B + \kappa_0A = 0 \;\Rightarrow\; B = \kappa_0A,$
$(3.9\text{b}):\ B + \kappa_L(A + BL) = 0 \;\Rightarrow\; A\big(\kappa_0 + \kappa_L + \kappa_0\kappa_L L\big) = 0 .$

$\boxed{\;k=0 \text{ is an eigenvalue} \iff \kappa_0 + \kappa_L + \kappa_0\kappa_L L = 0\;} \tag{3.18}$

(Note the condition is symmetric under $\kappa_0\leftrightarrow\kappa_L$, as it
must be under $x\to L-x$. It is **not** "$\kappa_0=0$ or $\kappa_L=-2/L$" —
that is only the symmetric-coupling specialisation.) I verified (3.18) against
the explicit boundary conditions for
$(\kappa_0,\kappa_L) = (0,0), (-2,-2), (-3,-1.5)$ (all give a zero mode) and
$(0,0.5), (0.5,0), (2,2), (-2,0), (0,-2), (-1,-2), (-2,-1)$ (none do).

Hence:

- **Neumann–Neumann**, $\kappa_0=\kappa_L=0$: (3.18) holds, the constant mode of
  §3.3 is present. In this case (3.12) reduces to $-k^2\sin(kL)=0$, whose
  positive roots are the $n\pi/L$ of (3.3) — **and whose root at $k=0$ is of
  even multiplicity ($\sim-k^3L$ near $0$), so it produces no sign change.** A
  sign-change scan of (3.12) therefore finds every Neumann mode *except* the
  zero mode. §6 item 2.
- **Symmetric, $\kappa_0=\kappa_L=\kappa$**: (3.18) gives $\kappa(2+\kappa L)=0$,
  i.e. $\kappa = 0$ (Neumann) or $\kappa = -2/L$. For $\kappa=-2/L$ the zero mode
  is $f \propto 1 - 2x/L$ (verified: zero boundary residual at both ends). Note
  it is **not** the lowest eigenvalue there — see §3.5.6.
- **One end Neumann, the other Robin with $\kappa\neq0$**: (3.18) gives
  $\kappa_L = 0$, contradiction — **no zero mode.** Consistent with (2.8), since
  the Robin piece forces $f=0$ there, hence $f\equiv0$.
- **$\kappa_0,\kappa_L > 0$: no zero mode**, and $\lambda_1 > 0$ strictly, by
  (2.8): $\lambda=0$ would force $\nabla f=0$ and $f=0$ on both Robin pieces,
  hence $f\equiv0$. **But "not zero" does not mean "order $\pi/(2L)$":** the
  smallest root depends strongly on $\kappa$ and joins the Neumann zero mode
  continuously as $\kappa\to0^+$:

  | $\kappa$ ($L=1$, both ends) | $k_1$ | $\lambda_1 = k_1^2$ | $2\kappa/L$ |
  |---|---|---|---|
  | 0.001 | 0.0447176 | 0.0020000 | 0.002 |
  | 0.01 | 0.1413036 | 0.0199667 | 0.02 |
  | 0.1 | 0.4435208 | 0.196711 | 0.2 |
  | 1 | 1.3065424 | 1.707053 | 2 |
  | 2 | 1.7206672 | 2.960696 | 4 |
  | 10 | 2.6276754 | 6.904678 | 20 |
  | $10^4$ | 3.1409645 | 9.865658 | — |

  **[VERIFIED]** For small $\kappa$, $\lambda_1 \to 2\kappa/L$ — the variational
  bound of §3.5.6 is *asymptotically attained* by the near-constant mode
  ($k_1 \approx \sqrt{2\kappa/L}$). So a test that asserts "the first Robin
  eigenvalue is near $(\pi/2L)^2$" is wrong for small $\kappa$; only for large
  $\kappa$ does the Dirichlet-like value set in (2.6).

#### 3.5.6 Negative eigenvalues: the imaginary-wavenumber equation [DER], [VERIFIED]

**A complete spectrum solver needs two equations, not one.** (3.12) is the
condition for $\lambda = k^2 > 0$. Negative eigenvalues, $\lambda = -\mu^2 < 0$,
come from $k = i\mu$ ($\mu>0$). Substituting into (3.12),
$\sin(i\mu L) = i\sinh(\mu L)$, $\cos(i\mu L) = \cosh(\mu L)$, $k^2 = -\mu^2$:

$\boxed{\;(\kappa_0\kappa_L + \mu^2)\,\sinh(\mu L) + (\kappa_0+\kappa_L)\,\mu\,\cosh(\mu L) = 0,
\qquad \lambda = -\mu^2\;} \tag{3.19}$

(Equivalently, divide by $\cosh$: $(\kappa_0\kappa_L+\mu^2)\tanh(\mu L) + (\kappa_0+\kappa_L)\mu = 0$.)
**This equation has a solution with $\mu>0$ iff at least one of $\kappa_0,\kappa_L$
is negative** — which is exactly when (2.8) allows $\lambda<0$. For
$\kappa_0,\kappa_L \ge 0$ every term on the left of (3.19) is positive, so there
are no negative eigenvalues and (3.12) alone is complete. **All of §5 is in this
safe class** ($\kappa \ge 0$), which is why the reference table needed only
(3.12).

Then:

1. **[DER]** $\lambda_{\min} \le 2\kappa/L$ for the symmetric case $\kappa_0=\kappa_L=\kappa<0$,
   by taking the trial function $f \equiv 1$ in (2.8): $\int|\nabla f|^2 = 0$ and
   the boundary terms give $2\kappa$, divided by $\int f^2 = L$. **So any
   $\kappa<0$ produces at least one negative eigenvalue** — a mode that grows
   exponentially in time by §1.4 item 2.
2. **[VERIFIED]** The number of negative eigenvalues is one for the couplings I
   checked ($\kappa = -0.5, -2, -3, -10$ at both ends, $L=1$), with $\lambda$
   matching the roots of (3.19) to 5 digits at $N=400$; for large negative
   $\kappa$ the negative eigenvalue approaches $-\kappa^2$ from above
   ($\kappa=-10$: $\lambda_{\min} = -100.003$).
3. **[VERIFIED]** At $\kappa_0=\kappa_L=-2/L$ the eigenvalues are
   $\lambda = -5.756915,\ 0,\ 31.32,\ \dots$ for $L=1$ — the zero mode of
   §3.5.5 is the *second* eigenvalue, not the first.
4. **[DER]** For $\kappa_0 = -\kappa_L = a$ (equal and opposite), (3.19) becomes
   $(-a^2+\mu^2)\sinh(\mu L)=0$, so $\mu = |a|$ and there is exactly one negative
   eigenvalue, $\lambda = -a^2$, *in addition to* the Dirichlet positive
   spectrum. This is the signature of the wrong-sign bug in §6 item 1.

**Practical instruction:** support $\kappa<0$ (it is self-adjoint and it is the
physically interesting "attractive boundary" case), but do not assume
$\lambda_{\min}\ge0$. Do not "fix" a negative eigenvalue by taking an absolute
value; it is the answer. And **do not report the spectrum as complete unless you
have solved both (3.12) and (3.19)** whenever a coupling is negative.

#### 3.5.7 Root-finding recipe

A recipe that is robust, and which I used to generate §5:

1. Work in dimensionless $x = kL$ with (3.15), or in $k$ with (3.12).
2. Scan $k$ upward from a small positive value ($10^{-8}$, not $0$) with step
   $h_{\text{scan}} = 0.001/L$; record every sign change.
3. Refine each bracket by bisection to machine precision (or use
   `scipy.optimize.brentq` on the bracket).
4. **Handle the $k=0$ root separately** per (3.18) before/after the scan.
5. **If any coupling is negative, also solve (3.19) on $\mu>0$** by the same
   scan-and-bisect, and merge $\lambda=-\mu^2$ into the eigenvalue list. Skip
   this only when $\kappa_0,\kappa_L\ge0$, where (3.19) provably has no root.
6. Sort the merged list ascending and stop when you have the requested number of
   eigenvalues; assert that the number of positive roots found matches the
   interlacing count of §3.5.3.

At 50-digit precision, scanning in `float` and refining in `mpmath` is what I
did; scanning entirely at high precision is too slow by orders of magnitude.

### 3.6 The 2D box: product structure and degeneracies

#### 3.6.1 Factorisation [STD]

The operator $-\Delta = -\partial_x^2 - \partial_y^2$ is a sum of commuting 1D
operators, and the box's boundary conditions are assigned per direction (the
$x=0,L_1$ faces carry conditions on $\phi$ as a function of $x$ only, etc.). So

$$\boxed{\;f_{n_1n_2}(x,y) = f^{(1)}_{n_1}(x)\,f^{(2)}_{n_2}(y),\qquad
\lambda_{n_1n_2} = k^{(1)2}_{n_1} + k^{(2)2}_{n_2},\qquad
\omega^2_{n_1n_2} = k^{(1)2}_{n_1} + k^{(2)2}_{n_2} + m^2\;} \tag{3.20}$$

with $f^{(i)}$, $k^{(i)}$ taken from §3.2–§3.5 for the condition on direction
$i$, and the **mode ranges taken per direction**: $n_i \ge 1$ for Dirichlet,
$n_i \ge 0$ for Neumann, $n_i \ge 0$ with half-integer labels for mixed, and the
Robin roots in increasing order.

The full list of combinations the solver must support, with the mode ranges:

| $x$-condition | $y$-condition | $n_1$ range | $n_2$ range | ground state $\lambda$ |
|---|---|---|---|---|
| D | D | $\ge 1$ | $\ge 1$ | $(\pi/L_1)^2+(\pi/L_2)^2$ |
| N | N | $\ge 0$ | $\ge 0$ | $0$ (zero mode) |
| N | D | $\ge 0$ | $\ge 1$ | $(\pi/L_2)^2$ |
| D | N | $\ge 1$ | $\ge 0$ | $(\pi/L_1)^2$ |
| R | R | root $n_1$ | root $n_2$ | $k_1^{(1)2}+k_1^{(2)2}$ |
| R | N | root $n_1$ | $\ge 0$ | $k_1^{(1)2}$ |
| R | D | root $n_1$ | $\ge 1$ | $k_1^{(1)2}+(\pi/L_2)^2$ |
| D | R, N | R, etc. | by symmetry | |

**The mode ordering in 2D is by eigenvalue, not by lexicographic $(n_1,n_2)$.** A
solver that returns modes sorted by $(n_1,n_2)$ will disagree with the reference
table's ordering; sort by $\lambda$ (with a documented tie-breaking rule for
degenerate levels).

#### 3.6.2 Degeneracies [STD], [DER]

Degeneracies are of two kinds:

**(a) Symmetry-forced.** When $L_1 = L_2$ and both directions carry the same
condition, $(n_1,n_2)$ and $(n_2,n_1)$ are degenerate. For the pure Dirichlet
square with $L_1=L_2=L$:

$$\lambda_{n_1n_2} = \Big(\frac{\pi}{L}\Big)^2\big(n_1^2+n_2^2\big), \tag{3.21}$$

so the multiplicity of the level with $n_1^2+n_2^2 = q$ is the number of
**ordered pairs of positive integers** $(a,b)$ with $a^2+b^2 = q$. This is
classical number theory [9]: with $r_2(q)$ the number of representations of $q$
as a sum of two squares of *integers* (signs and zero allowed), Jacobi's formula
gives $r_2(q) = 4\,(d_1(q)-d_3(q))$ where $d_{1,3}$ count divisors $\equiv 1,3$
mod 4; then

$$\text{mult}_{DD}(q) = \frac{r_2(q) - 4\,[\sqrt q \in \mathbb{Z}]}{4}. \tag{3.22}$$

I verified (3.22) against direct enumeration for
$q = 1,2,4,5,8,9,10,13,16,17,18,20,25$. Values for the Dirichlet square:

| $q$ | 2 | 5 | 8 | 10 | 13 | 17 | 18 | 20 | 25 | 26 | 29 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mult | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 |

Note $q=1,4,9,16$ have multiplicity **0** for Dirichlet: those are the
axis levels $(n,0)$, $(0,n)$, and $n=0$ is excluded. **For Neumann the same $q$
has a different multiplicity** because the axis modes are present: for $N$–$N$
with $L_1=L_2$ every $q$ that is a perfect square gains the $(n,0)$ and $(0,n)$
modes, and $q=0$ is a non-degenerate zero mode. So a degeneracy test must be
written per boundary-condition combination, not once for "the square".

**(b) Accidental.** For rational $L_1/L_2 = p/q$ there are exact coincidences
beyond the $n_1\leftrightarrow n_2$ swap. For the mixed $N_x D_y$ box with
$L_1 = 1$, $L_2 = 1.5$ used in §5 Case E, $\lambda_{0,3} = 4\pi^2 = 39.4784$
coincides with the *excluded* $\lambda_{2,0}$ — the near-degeneracy is broken
precisely because the Dirichlet direction excludes $n_2=0$. With
$L_1/L_2$ irrational there are no exact degeneracies at all. **Do not assume
degeneracies exist, and do not assume they are absent: count them.**

#### 3.6.3 Counting and consistency checks [STD], [VERIFIED]

Two checks the engineer can run without any reference numbers:

- **Weyl law.** In 1D, the number of modes with $\lambda \le \Lambda$ satisfies
  $N(\Lambda) = L\sqrt\Lambda/\pi + O(1)$ for both Dirichlet and Neumann (the
  boundary correction differs in sign but is $O(1)$). In 2D,
  $N(\Lambda) = A\Lambda/(4\pi) + O(\sqrt\Lambda)$ with $A = L_1L_2$. Verified
  for $L=1$: $\Lambda=10$ gives 1 mode vs Weyl $1.007$; $\Lambda=100$ gives 3 vs
  $3.183$.
- **Isospectrality.** ND and DN on the same interval have identical spectra
  (§3.4). A 2D box with $N_xD_y$ and its $D_xN_y$ mirror with $L_1\leftrightarrow L_2$
  have identical spectra. Both are exact symmetries of the scheme and both
  should hold to machine precision in the discrete solver too.

---

## 4. Finite-difference discretisation specification

The goal: assemble a sparse **symmetric generalised eigenvalue problem** whose
lowest eigenvalues converge to (3.2)–(3.7) at **second order in the grid
spacing**, uniformly across all boundary-condition combinations, and which is
exactly symmetric so that `scipy.sparse.linalg.eigsh` (which requires symmetry)
can be used and so that all eigenvalues are guaranteed real.

### 4.1 Grid

**[DER]** Nodes are **equally spaced including both endpoints**:

$$x_j = j\,h, \qquad h = \frac{L}{N+1}, \qquad j = 0,1,\dots,N+1 . \tag{4.1}$$

So there are $N+2$ nodes, $N$ of them strictly interior, and $x_0 = 0$,
$x_{N+1} = L$ **are nodes**. In 2D, the same in each direction with
$h_1 = L_1/(N_1+1)$, $h_2 = L_2/(N_2+1)$, giving an $(N_1+2)\times(N_2+2)$ grid.

Why this convention (rather than cell-centred or "interior nodes only"):

- **Neumann and Robin boundaries need a node on the boundary**, because the
  condition constrains the field *at* the boundary and the discrete operator at
  that node needs a ghost node outside. Without a boundary node you cannot
  impose (2.3) without a one-sided (lower-order) formula.
- **Dirichlet nodes are simply eliminated** (§4.4), so including them costs
  nothing: they are not unknowns.
- $h = L/(N+1)$, **not** $L/N$ and not $L/(N-1)$. Getting this wrong shifts
  every eigenvalue at first order in $h$ — it will still *converge*, just to the
  wrong values, which is the worst failure mode. The convention is fixed by
  requiring $x_{N+1} = (N+1)h = L$.

### 4.2 Stencils and ghost nodes

**[STD]** The second-order central second difference at an interior node:

$$(-\Delta_h f)_j = -\frac{f_{j-1} - 2f_j + f_{j+1}}{h^2} = \frac{-f_{j-1}+2f_j-f_{j+1}}{h^2},
\qquad 1 \le j \le N. \tag{4.2}$$

At a boundary node we use the *same* three-point formula with a **ghost node**
$x_{-1} = -h$ (or $x_{N+2} = L+h$) whose value is fixed by the boundary
condition. The ghost value is eliminated, giving a modified row.

**[DER]** The ghost relations follow from the **central** difference of the
derivative at the boundary node — not a one-sided one:

**At $x=0$** (outward normal $-\hat x$, so $\partial_n = -\partial_x$):

- Neumann, $-\phi'(0) = 0$: $\dfrac{f_1 - f_{-1}}{2h} = 0 \Rightarrow f_{-1} = f_1 . \tag{4.3a}$
- Robin, $-\phi'(0)+\kappa_0\phi(0)=0 \Rightarrow \phi'(0) = \kappa_0 f_0$:
  $\dfrac{f_1 - f_{-1}}{2h} = \kappa_0 f_0 \Rightarrow f_{-1} = f_1 - 2h\kappa_0 f_0 . \tag{4.3b}$
- Dirichlet: $f_0 = 0$, node eliminated, nothing to do.

**At $x=L$** (outward normal $+\hat x$):

- Neumann, $+\phi'(L) = 0$: $\dfrac{f_{N+2} - f_N}{2h} = 0 \Rightarrow f_{N+2} = f_N . \tag{4.4a}$
- Robin, $\phi'(L)+\kappa_L\phi(L)=0 \Rightarrow \phi'(L) = -\kappa_L f_{N+1}$:
  $\dfrac{f_{N+2}-f_N}{2h} = -\kappa_L f_{N+1} \Rightarrow f_{N+2} = f_N - 2h\kappa_L f_{N+1} . \tag{4.4b}$

**Note (4.4a)/(4.4b): the central difference at the last node uses $f_N$ and
$f_{N+2}$, not $f_{N+1}$ and $f_{N+2}$.** The latter is a one-sided difference
and destroys the second-order accuracy (and the symmetry). This is trap #5.

Substituting into (4.2) at the boundary nodes gives the **modified rows**:

$$x=0:\quad (-\Delta_h f)_0 = \frac{(2+2h\kappa_0)f_0 - 2f_1}{h^2}
\qquad(\kappa_0=0\ \text{for Neumann}), \tag{4.5a}$$

$$x=L:\quad (-\Delta_h f)_{N+1} = \frac{-2f_N + (2+2h\kappa_L)f_{N+1}}{h^2}
\qquad(\kappa_L=0\ \text{for Neumann}). \tag{4.5b}$$

**Explicit small example**, to remove all ambiguity. $N = 3$, so $h = L/4$ and
nodes $0,\dots,4$; Neumann at $x=0$, Robin $\kappa_L$ at $x=L$. The **unsymmetrised**
matrix $A'$ (with $-\Delta_h$ as the operator) is

$$A' = \frac{1}{h^2}\begin{pmatrix}
2 & -2 & 0 & 0 & 0\\
-1 & 2 & -1 & 0 & 0\\
0 & -1 & 2 & -1 & 0\\
0 & 0 & -1 & 2 & -1\\
0 & 0 & 0 & -2 & 2+2h\kappa_L
\end{pmatrix}. \tag{4.6}$$

**$A'$ is not symmetric** — the $(0,1)$ entry is $-2/h^2$ while $(1,0)$ is
$-1/h^2$. This is intrinsic to ghost elimination, not an error, and §4.3 fixes
it properly.

### 4.3 Symmetry: the weighted (generalised) eigenproblem

**[DER]** The ghost-eliminated $A'$ of (4.6) is **not** symmetric, but it *is*
self-adjoint with respect to a weighted inner product. Define the diagonal weight
vector

$$w_j = \begin{cases} \tfrac12, & j = 0 \text{ or } j = N+1 \text{ carrying N or R},\\[2pt] 1, & \text{interior nodes},\end{cases}
\qquad W = \mathrm{diag}(w_0,\dots,w_{N+1}) \tag{4.7}$$

(nodes eliminated by Dirichlet are removed from both $A'$ and $W$). Then

$$A_{\rm sym} \equiv W A' \quad\text{is symmetric}, \tag{4.8}$$

and the eigenproblem is the **symmetric generalised problem**

$$\boxed{\;A_{\rm sym}\,v = \lambda\, W\, v\;} \tag{4.9}$$

*Why (4.8) is symmetric.* $A'$ has $(0,1) = -2/h^2$, $(1,0) = -1/h^2$ and
similarly at the right end; all other entries are symmetric. Then
$(A_{\rm sym})_{01} = w_0 A'_{01} = \tfrac12(-2/h^2) = -1/h^2 = w_1A'_{10} = (A_{\rm sym})_{10}$,
and the $(0,0)$ entry is unchanged by the symmetry check since it is diagonal.
This is exactly the statement that $A'$ is self-adjoint in the inner product
$\langle u,v\rangle_W = \sum_j w_j u_j v_j$, the discrete analogue of (1.7) with
the **trapezoidal rule**. I verified numerically that
$\max|A_{\rm sym} - A_{\rm sym}^{\!\top}| = 0$ exactly (not merely small) for
D–D, N–N, N–D, R–R and R–N in 1D, and for the 2D assembly of §4.5.

**The eigenvalues of (4.9) are identical to those of the unsymmetrised
problem.** $A_{\rm sym} = WA'$ with $W$ invertible, so
$A_{\rm sym}v = \lambda Wv \iff A'v = \lambda v$. Identical $\lambda$,
identical eigenvectors *as vectors*. **But the orthonormality relation changes:**

$$\boxed{\;v_m^{\top} W v_n = \delta_{mn}\;}\qquad\text{(not } v_m^{\top}v_n = \delta_{mn}\text{).} \tag{4.10}$$

I verified this: for $\kappa=2$ at both ends with $N=20$,
$\max|V^{\top}WV - I| = 1.4\times10^{-15}$ while $\max|V^{\top}V - I| = 0.142$.
**A test that asserts `V.T @ V == I` will fail for any problem with a
Neumann or Robin boundary, and will pass only for all-Dirichlet.** Trap #3.

**Recommendation.** Assemble $A_{\rm sym}$ and $W$, and call
`scipy.sparse.linalg.eigsh(A_sym, k=nmodes, M=W, sigma=0, which='LM')` or
`scipy.linalg.eigh(A_sym.toarray(), W.toarray())` for small problems. Both
guarantee real $\lambda$ and return $W$-orthonormal eigenvectors. Do **not**
use a general non-symmetric eigensolver: it costs more, gives complex
eigenvalues for a problem that has none, and destroys the ordering guarantees.

**A note on the alternative "halve the boundary rows of $A'$" description.**
Since $A_{\rm sym} = WA'$ literally multiplies the boundary rows by $\tfrac12$,
some write the rule as "halve the first and last rows and use
$M = \mathrm{diag}(\tfrac12,1,\dots,1,\tfrac12)$". That is the same thing and is
a fine way to implement it, provided you also use $M=W$ (not $M=I$) in the
eigenproblem. Halving the rows while solving the *ordinary* eigenproblem
$A_{\rm sym}v = \lambda v$ is a **different problem** with different eigenvalues
(it corresponds to $M = \mathrm{diag}(2,1,\dots,1,2)$) and it is wrong. Trap #4.

### 4.4 Dirichlet nodes: eliminate, do not constrain

**[DER]** A Dirichlet boundary node has $f = 0$ known. Do not put it in the
unknown vector with a row $e_j^{\top}$ and a unit eigenvalue-like entry: that
makes the matrix non-symmetric and (worse) can introduce a spurious eigenvalue
if the row is scaled inconsistently. **Delete the row and column.** Its
contribution to neighbouring rows is then automatically zero because
$f_{\text{boundary}} = 0$.

Implementation: build the full $(N+2)\times(N+2)$ matrices, then
`A = A[np.ix_(active, active)]`, `W = W[active]` with `active` the boolean mask
of non-Dirichlet nodes. In 2D the active set is the **product** of the 1D active
sets, so an entire Dirichlet face is removed (including corners), which is
correct.

For an all-Dirichlet problem this reproduces the textbook tridiagonal matrix
$(1/h^2)\,\mathrm{tridiag}(-1,2,-1)$ of size $N\times N$ with mass matrix
$W = I$ — see the exact spectrum in §4.7.

### 4.5 2D assembly

**[DER]** Assemble each direction's 1D pair $(A^{(i)}_{\rm sym}, W^{(i)})$ on its
own active set, then

$$\boxed{\;A = A^{(1)}_{\rm sym}\otimes W^{(2)} + W^{(1)}\otimes A^{(2)}_{\rm sym},
\qquad M = W^{(1)}\otimes W^{(2)}\;} \tag{4.11}$$

and solve $A\,v = \lambda\, M\,v$. This is symmetric (Kronecker products of
symmetric matrices with diagonal $W$) and positive definite in $M$. Its
eigenvalues are the sums $\lambda^{(1)}_{m}+\lambda^{(2)}_{n}$ exactly, which is
the discrete counterpart of (3.20) — I verified the 2D eigenvalues reproduce the
1D sums and that $\max|A - A^{\top}| = 0$.

**Index convention** (state it in the code, because the two Kronecker orders
differ): with `np.kron(A1, W2)` the **first** factor varies slowest, so the
flattened index is $j\cdot n_2 + i$ for node $(i,j)$ — i.e. row-major with the
$y$-index fastest. Pick one, document it, and test it by checking that the
eigenvector of the $(1,1)$ mode, reshaped, is a single half-wave in each
direction.

`eigsh` on (4.11) with `M=M` and `sigma=0` uses a shift-invert; for 2D problems
up to a few thousand unknowns this is fast. For the lowest modes of a symmetric
generalised problem, `which='SM'` is slow and unreliable; prefer shift-invert
(`sigma=0` or a small positive shift).

### 4.6 Recovering eigenfunctions: the $\sqrt h$ factor

**[DER], [VERIFIED]** This is subtle and it is trap #6.

`eigh(A, W)` returns $v$ with $v^{\top}Wv = 1$, where $W$ has entries $\tfrac12$
and $1$ — **no factor of $h$**. The continuum normalisation (1.8) is
$\int|f|^2 = 1$, whose trapezoidal discretisation is

$$\sum_j h\,w_j\,u_j^2 = 1 \qquad\text{for } u_j \approx f(x_j). \tag{4.12}$$

Therefore the **function values** are

$$\boxed{\;u_j = \frac{v_j}{\sqrt h}\;} \tag{4.13}$$

and only then is $u$ comparable with the analytically normalised $f_n$ of §3.

I verified (4.13) both ways:

- Robin, $\kappa=2$ both ends, $L=1$, mode 1: with $u = v/\sqrt h$ the nodal error
  $\max_j|u_j - f_1(x_j)|$ is $2.06\times10^{-3}, 5.66\times10^{-4},
  1.49\times10^{-4}, 3.80\times10^{-5}$ for $N = 10,20,40,80$ — second order
  (ratios 3.6, 3.8, 3.9). Without the $1/\sqrt h$, $u_0 = 0.1604$ against
  $f_1(0) = 0.7350$: a factor $1/\sqrt h = \sqrt{21} = 4.583$, i.e. a 78% error.
- Mixed ND, mode 0: with $u = v/\sqrt h$ the error is $2\times10^{-15}$ — machine
  precision. (See §4.7 for why this case is nodally exact.)

So: **eigenvalues from `eigh(A, W)` are correct as they stand; eigenvectors must
be divided by $\sqrt h$ (per direction in 2D, so by $\sqrt{h_1h_2}$ in 2D) before
comparing with §3.** A test that skips this fails by a factor $\sqrt{h}$ and will
look like a convergence failure.

### 4.7 What the discrete spectrum is, exactly, and the expected convergence rate

**[STD]** The scheme is second-order accurate: the truncation error of (4.2) is
$O(h^2)$, and the boundary rows (4.5) are also $O(h^2)$ (the ghost value is
itself $O(h^2)$ accurate because the odd terms in the Taylor expansion vanish at
a Neumann/Robin point to the order needed). The eigenvalue error is then
$O(h^2)$ as well, for a reason worth stating because it tells you the
*coefficient*:

#### 4.7.1 The exact discrete spectrum for Dirichlet/Neumann data [DER], [VERIFIED]

**For pure Dirichlet and pure Neumann boundaries the discrete spectrum is known
in closed form — exactly, at finite $h$:**

$\boxed{\;\lambda_n^{(h)} = \frac{4}{h^2}\sin^2\!\Big(\frac{k_n h}{2}\Big),
\qquad k_n \text{ the \emph{exact} analytic wavenumber of §3,}\;} \tag{4.14}$

with $k_n = n\pi/L$ (Dirichlet $n\ge1$, Neumann $n\ge0$) or $(n+\tfrac12)\pi/L$
(mixed N/D, $n\ge0$). Equivalently for the Dirichlet interval,
$\lambda_m^{(h)} = \tfrac{4(N+1)^2}{L^2}\sin^2\!\big(\tfrac{m\pi}{2(N+1)}\big)$,
$m = 1,\dots,N$.

I verified (4.14) against `eigh` for **every** mode at $N = 7$ and $N = 20$:
$\max_n|\lambda_n^{(h)} - \text{(4.14)}| = 6\times10^{-14}$ (D–D),
$3\times10^{-14}$ (N–N, including the $n=0$ entry which is exactly $0$),
$9\times10^{-14}$ (N–D). That is round-off, not truncation. **This is the test to
write** — it needs no tolerance beyond round-off, and it holds at *any* $N$,
including very coarse grids where a convergence test has nothing to measure.

*Why it is exact.* The difference equation with constant coefficients has
$\cos$/$ \sin$ solutions, and for D and N data both boundary conditions are
imposed exactly by (4.3)–(4.5) (the ghost reflection $f_{-1}=f_1$ is exact for a
Neumann point, and the elimination is exact for a Dirichlet point). So the
sampled analytic eigenfunction is an exact eigenvector of the discrete operator,
with the eigenvalue (4.14) rather than $k^2$. **In 2D the same holds as a
Kronecker sum**, by (4.11):
$\lambda^{(h)} = \tfrac{4}{h_1^2}\sin^2(k_1h_1/2) + \tfrac{4}{h_2^2}\sin^2(k_2h_2/2)$
— verified to $8\times10^{-14}$ for the five lowest mixed $N_xD_y$ levels at
$N_1=N_2=8$, and $7\times10^{-13}$ at $N=16$.

Expanding (4.14) with $\theta = k_nh/2$:

$\frac{\lambda_n^{(h)}}{\lambda_n} = \Big(\frac{\sin\theta}{\theta}\Big)^2
= 1 - \frac{\theta^2}{3} + O(\theta^4),$

so

$\boxed{\;\lambda_n^{(h)} = \lambda_n\Big[1 - \frac{(k_nh)^2}{12} + O(h^4)\Big]\;} \tag{4.15}$

**This is the second test to write:** the *relative* eigenvalue error of the
$n$-th Dirichlet/Neumann mode is $-(k_nh)^2/12$ to leading order, with **no free
constant**. Verified:

| case | $N$ | mode | measured rel. error | $-(k h)^2/12$ | ratio |
|---|---|---|---|---|---|
| D–D, $L=1$ | 40 | 1 | $-4.8918\times10^{-4}$ | $-4.8927\times10^{-4}$ | 0.9998 |
| D–D, $L=1$ | 80 | 3 | $-1.1277\times10^{-3}$ | $-1.1282\times10^{-3}$ | 0.9995 |
| N–D, $L=1$ | 40 | 1 | $-1.2231\times10^{-4}$ | $-1.2232\times10^{-4}$ | 1.0000 |
| N–D, $L=1$ | 80 | 3 | $-7.8324\times10^{-4}$ | $-7.8348\times10^{-4}$ | 0.9997 |
| N–N, $L=1$ | 40 | 3 | $-1.9556\times10^{-3}$ | $-1.9571\times10^{-3}$ | 0.9992 |

**Important exception — Robin.** For Robin boundaries (4.14) is **false** and
(4.15)'s coefficient is *not* $1/12$: it is $\kappa$- and mode-dependent and
**can have either sign**. Measured at $L=1$, $\kappa=2$ both ends: (4.14)
deviates by $1.7$ in absolute terms at $N=7$ and $0.27$ at $N=20$ (so it is not
an identity there), and the relative error against the exact spectrum is
$+6.75\times10^{-5}$ for mode 1 at $N=40$ — i.e. $-0.46$ times the naive
$-(kh)^2/12$ — and $-0.71$ times for mode 3. The *rate* is still exactly 2
(measured 1.87, 1.93, 1.96, 1.98 for $N = 10,20,40,80,160$), which is what to
test. **Do not assert (4.14) or (4.15) for Robin.** The reason: the ghost-node
Robin condition contributes a boundary truncation error that enters the
eigenvalue at the same $O(h^2)$ order with its own $\kappa$-dependent
coefficient, competing with the bulk $1/12$ term.

#### 4.7.2 Measured convergence rates

Error halving per grid doubling, from the scheme of §4 for $L=1$, lowest modes:

| BC | rates ($N=10\to160$) |
|---|---|
| D–D | 1.86→1.98, 1.85→1.98, 1.84→1.98 |
| N–N | (the $n=0$ zero mode is exact) 1.86→1.98, 1.85→1.98, 1.84→1.98 for $n=1,2,3$ |
| N–D | 1.87→1.98, 1.86→1.98, 1.85→1.98 |
| R(2)–R(2) | 1.87→1.98, 1.86→1.98, 1.85→1.98 |
| R(0.5)–R(3) | 1.87→1.98, 1.86→1.98, 1.86→1.98 |

2D mixed $N_xD_y$, $L_1=1$, $L_2=1.5$, lowest six modes: rates 1.83, 1.91, 1.96
at $N = 16,32,64$ (grid $N$ per direction). **Expected rate is 2; a measured rate
of 1 means a one-sided boundary difference or a wrong $h$; a rate of 4 or better
means you are accidentally comparing against the discrete spectrum rather than
the continuum one.**

#### 4.7.3 Three exact results to assert (no tolerance needed)

1. **The closed form (4.14).** For D–D, N–N and N–D, assert
   $|\lambda_n^{(h)} - \tfrac{4}{h^2}\sin^2(k_nh/2)| < 10^{-12}$. In 2D, assert
   the Kronecker sum. This subsumes the Dirichlet-only case and is the single most
   useful test in this document.
2. **Neumann zero mode.** With Neumann at both ends, the discrete eigenvector
   for the lowest eigenvalue is the **constant vector** and the eigenvalue is
   **exactly 0**. Verified: $v = (1,1,\dots,1)$ to 12 digits, $\lambda_0 =
   5.8\times10^{-14}$ (pure round-off). If you get $\lambda_0 = O(h^2)$ instead
   of $O(10^{-14})$, your Neumann boundary row is wrong.
3. **Nodally exact trigonometric modes.** For 1D D–D, N–N and N–D, the discrete
   eigenvector equals the **sampled analytic eigenfunction** (with the *exact*
   wavenumber $k_n$, not a shifted one) to machine precision. Verified:
   $\max_j|u_j - \sqrt{2/L}\sin(n\pi x_j/L)| \le 4\times10^{-15}$ for D–D modes
   1, 3, 5 at $N=10,20$; $\max_j|u_j - \sqrt{2/L}\cos((n+\tfrac12)\pi x_j/L)|
   \le 5\times10^{-15}$ for N–D modes 1, 3, 5, including the boundary value
   $u_0 = 1.414213562 = \sqrt2$ exactly; and the N–N zero mode is the constant
   vector. Note the eigenvector is exact while the eigenvalue is (4.14) — the
   discretisation error sits entirely in the eigenvalue for these cases.
   **Robin does not have this property** (§4.6: nodal error $2\times10^{-3}$ at
   $N=10$), so use the rate test there, not an exactness test.

### 4.8 Summary of the algorithm

1. For each direction $i = 1,2$: given $L_i$, $N_i$, and the condition on each
   end, build $A^{(i)}_{\rm sym}$ and $W^{(i)}$ on the active nodes:
   - interior rows: $(-1,2,-1)/h_i^2$;
   - N/R boundary rows from (4.5);
   - Dirichlet nodes deleted;
   - halve the N/R boundary rows (equivalently form $W A'$);
   - weights $w = \tfrac12$ at N/R boundary nodes, $1$ elsewhere.
2. Assemble $A$, $M$ by (4.11).
3. Solve the symmetric generalised problem $Av = \lambda Mv$ for the lowest
   `nmodes` eigenvalues (shift-invert).
4. Convert: $u = v/\sqrt{h_1h_2}$ (per direction: $\sqrt{h_i}$), and
   $\omega_n^2 = \lambda_n + m^2$, returning $\omega$ as complex (or $\omega^2$).
5. Sort by $\lambda$ ascending, with a documented tie-break for degenerate
   levels.

---

## 5. Reference table of hard numbers

All values were computed with **mpmath 1.3 at 50-digit precision** from the
closed forms in §3 — case A–C from the elementary closed forms (3.2), (3.3),
(3.7); cases D, D2 from the roots of (3.12) found by float scan + mpmath
bisection per §3.5.7; cases E, F from (3.20). They are quoted to 12 significant
figures; the trailing digits are trustworthy (the symbolic checks in §7 agree
independently). **Every entry is traceable to the formula named in its header.**

Notation: $\lambda$ = eigenvalue of $-\Delta$; $\omega^2 = \lambda+m^2$;
$\omega = \sqrt{\omega^2}$ (real in all cases below).

### Case A — 1D, Dirichlet at both ends
$L = 1$, $m = 0.5$, both ends Dirichlet, modes $n = 1..6$.
**Formula: (3.2), $\lambda_n = (n\pi/L)^2$.**

| $n$ | $\lambda_n$ | $\omega_n^2$ | $\omega_n$ |
|---|---|---|---|
| 1 | 9.86960440109 | 10.1196044011 | 3.18113256578 |
| 2 | 39.4784176044 | 39.7284176044 | 6.30304827876 |
| 3 | 88.8264396098 | 89.0764396098 | 9.43803155376 |
| 4 | 157.913670417 | 158.163670417 | 12.5763138645 |
| 5 | 246.740110027 | 246.990110027 | 15.7159190004 |
| 6 | 355.305758439 | 355.555758439 | 18.8561862114 |

### Case B — 1D, Neumann at both ends (with the zero mode)
$L = 2$, $m = 1.0$, both ends Neumann, modes $n = 0..5$.
**Formula: (3.3), $\lambda_n = (n\pi/L)^2$, $n\ge0$; $f_0 = 1/\sqrt L$.**

| $n$ | $\lambda_n$ | $\omega_n^2$ | $\omega_n$ |
|---|---|---|---|
| 0 | 0 | 1.0 | 1.0 |
| 1 | 2.46740110027 | 3.46740110027 | 1.86209588912 |
| 2 | 9.86960440109 | 10.8696044011 | 3.29690830948 |
| 3 | 22.2066099025 | 23.2066099025 | 4.81732393580 |
| 4 | 39.4784176044 | 40.4784176044 | 6.36226513157 |
| 5 | 61.6850275068 | 62.6850275068 | 7.91738766935 |

**Test:** the $n=0$ row must be *exactly* $\lambda=0$, and
$\omega_0^2 = m^2 = 1$ exactly. This is the single most diagnostic row in the
document.

### Case C — 1D, Neumann at $x=0$, Dirichlet at $x=L$ (the mixed interval)
$L = 1$, $m = 0.25$, $n=0$ at $x=0$, $D$ at $x=1$, modes $n = 0..5$.
**Formula: (3.6)–(3.7), $\lambda_n = ((n+\tfrac12)\pi/L)^2$, $n\ge0$.**

| $n$ | $\lambda_n$ | $\omega_n^2$ | $\omega_n$ |
|---|---|---|---|
| 0 | 2.46740110027 | 2.52990110027 | 1.59056628289 |
| 1 | 22.2066099025 | 22.2691099025 | 4.71901577688 |
| 2 | 61.6850275068 | 61.7475275068 | 7.85795950020 |
| 3 | 120.902653913 | 120.965153913 | 10.9984159729 |
| 4 | 199.859489122 | 199.921989122 | 14.1393772537 |
| 5 | 298.555533133 | 298.618033133 | 17.2805680790 |

**Test:** $\lambda_0 = (\pi/2)^2 = 2.4674011$ — a *quarter* wavelength, half the
Dirichlet fundamental of Case A's $L=1$ spectrum. If your first eigenvalue is
$9.8696$, you used the Dirichlet mode set on an ND interval.

### Case D — 1D, Robin at both ends, equal couplings
$L = 1$, $m = 0$, $\kappa_0 = \kappa_L = 2.0$ (our convention, (2.3)),
modes $n = 1..7$ in ascending $k$.
**Formula: (3.14) for $k_n$, then $\lambda_n = k_n^2$, $\omega_n = k_n$ ($m=0$).**

| $n$ | $k_n = \omega_n$ | $\lambda_n = \omega_n^2$ |
|---|---|---|
| 1 | 1.72066717804 | 2.96069553758 |
| 2 | 4.05751567622 | 16.4634334628 |
| 3 | 6.85123691896 | 46.9394473198 |
| 4 | 9.82636087887 | 96.5573681218 |
| 5 | 12.8745963583 | 165.755231390 |
| 6 | 15.9573314248 | 254.636426202 |
| 7 | 19.0586688107 | 363.232856837 |

Cross-check available: each $k_n$ satisfies $\tan k_n = 4k_n/(k_n^2-4)$; I
verified $k_1$: $\tan(1.72066717804) = -6.62237964045$ and
$4k/(k^2-4) = -6.62237964045$ (agreement to $3\times10^{-14}$). Normalisation
constants for these modes: $I(k_1) = 1.851033886878379$,
$I(k_2) = 0.742962685095034$, $I(k_3) = 0.585216171650906$ from (3.17).

### Case D2 — 1D, Robin with unequal couplings
$L = 1$, $m = 0$, $\kappa_0 = 0.5$ at $x=0$, $\kappa_L = 3.0$ at $x=1$,
modes $n = 1..6$. **Formula: (3.12) / (3.13).**

| $n$ | $k_n = \omega_n$ | $\lambda_n = \omega_n^2$ |
|---|---|---|
| 1 | 1.45177421846 | 2.10764838139 |
| 2 | 3.92145780256 | 15.3778312973 |
| 3 | 6.77378599620 | 45.8841767223 |
| 4 | 9.77370803083 | 95.5253686719 |
| 5 | 12.8349221458 | 164.735226490 |
| 6 | 15.9255437179 | 253.622942712 |

**This case is the one that catches the normal-sign bug** (§6 item 1), because
the two couplings differ and the buggy spectra are then *not* accidentally equal
to a clean textbook sequence. For these parameters:

| implementation | $k_1$ | $k_2$ |
|---|---|---|
| correct, $(\kappa_0,\kappa_L)=(0.5,3.0)$ | 1.45177422 | 3.92145780 |
| bug (a), effective $(-0.5,3.0)$ | 0.73230426 | 3.68957517 |
| bug (b), effective $(0.5,-3.0)$ | 2.45788021 | 5.89717139 |

All three converge smoothly to second order; only the values tell you which
problem you solved.

### Case E — 2D, mixed: $x$ Neumann, $y$ Dirichlet
$L_1 = 1$, $L_2 = 1.5$, $m = 0.3$; Neumann at $x=0,L_1$, Dirichlet at $y=0,L_2$;
lowest 8 levels in ascending $\lambda$.
**Formula: (3.20) with (3.3) in $x$ ($n_1\ge0$) and (3.2) in $y$ ($n_2\ge1$).**

| # | $(n_1,n_2)$ | $\lambda$ | $\omega^2$ | $\omega$ |
|---|---|---|---|---|
| 1 | (0,1) | 4.38649084493 | 4.47649084493 | 2.11577192649 |
| 2 | (1,1) | 14.2560952460 | 14.3460952460 | 3.78762395784 |
| 3 | (0,2) | 17.5459633797 | 17.6359633797 | 4.19951942247 |
| 4 | (1,2) | 27.4155677808 | 27.5055677808 | 5.24457508105 |
| 5 | (0,3) | 39.4784176044 | 39.5684176044 | 6.29034320243 |
| 6 | (2,1) | 43.8649084493 | 43.9549084493 | 6.62984980594 |
| 7 | (1,3) | 49.3480220054 | 49.4380220054 | 7.03121767587 |
| 8 | (2,2) | 57.0243809841 | 57.1143809841 | 7.55740570461 |

Note the ordering is **not** lexicographic: level 5 is $(0,3)$ and level 6 is
$(2,1)$. Sort by $\lambda$. No level here is degenerate (though
$\lambda_{0,3} = 4\pi^2$ numerically equals the excluded $\lambda_{2,0}$).

### Case F — 2D, all Dirichlet, equal sides (degeneracies)
$L_1 = L_2 = 1$, $m = 0$, Dirichlet everywhere; lowest 8 levels.
**Formula: (3.20) with (3.2) both directions; (3.21)–(3.22) for multiplicities.**

| # | $(n_1,n_2)$ | $\omega^2 = \lambda$ | $\omega$ | multiplicity |
|---|---|---|---|---|
| 1 | (1,1) | 19.7392088022 | 4.44288293816 | 1 |
| 2,3 | (1,2), (2,1) | 49.3480220054 | 7.02481473104 | 2 |
| 4 | (2,2) | 78.9568352087 | 8.88576587632 | 1 |
| 5,6 | (1,3), (3,1) | 98.6960440109 | 9.93458826580 | 2 |
| 7,8 | (2,3), (3,2) | 128.304857214 | 11.3271733991 | 2 |

**Test:** the solver must return the level-2 eigenvalue *twice*. A solver that
deduplicates degenerate eigenvalues, or that sorts by $(n_1,n_2)$ without
preserving multiplicity, fails here.

---

## 6. What the engineer must not get wrong

Each item: the trap, why it is silent, and the test that catches it.

1. **The sign of the normal derivative at $x=0$.** With $\partial_n$ the
   *outward* normal, $\partial_n = -\partial_x$ at $x=0$. So the Robin condition
   (2.3) reads $\phi'(0) = +\kappa_0\phi(0)$ at the left end but
   $\phi'(L) = -\kappa_L\phi(L)$ at the right end (2.5) — **opposite signs for the
   same $\kappa$.** There are two natural ways to get this wrong, and both are
   silent:

   - **Bug (a):** using $\partial_n = +\partial_x$ at $x=0$ as well (forgetting
     the outward normal), so the condition is imposed as
     $\phi'(0) = -\kappa_0\phi(0)$, $\phi'(L) = -\kappa_L\phi(L)$. This is our
     problem with $(\kappa_0,\kappa_L)\to(-\kappa_0,+\kappa_L)$.
   - **Bug (b):** writing the same derivative formula at both ends,
     $\phi'(0) = +\kappa\phi(0)$ *and* $\phi'(L) = +\kappa\phi(L)$. This is our
     problem with $(\kappa_0,\kappa_L)\to(+\kappa_0,-\kappa_L)$.

   Both are self-adjoint and converge smoothly, so nothing in the solver
   complains. Note that the two bugs are mirror images of each other
   ($x\to L-x$), so **they produce identical spectra** — the spectrum alone
   cannot tell you which one you have.

   **The signature is dramatic and exact** [DER, VERIFIED]. Both bugs turn a
   symmetric $\kappa=a$ into $\kappa_0 = -a$, $\kappa_L = +a$ (or the reverse).
   For that case (3.12) becomes $(-a^2-k^2)\sin(kL) + 0\cdot k\cos(kL) = 0$, i.e.
   $\sin(kL)=0$: **the positive spectrum is exactly the Dirichlet spectrum
   $(n\pi/L)^2$**, for any $a$. And rotating to $k=i\mu$ (§3.5.6) gives
   $(-a^2+\mu^2)\sinh(\mu L) = 0$, i.e. $\mu = a$, so there is **also exactly one
   negative eigenvalue, $\lambda = -a^2$**. So for Case D's $\kappa=2$, $L=1$:

   | implementation | spectrum (lowest three) |
   |---|---|
   | correct | 2.96069554, 16.4634335, 46.9394473 |
   | bug (a) or (b) | $-4.000$, 9.86960440, 39.4784176 |

   I verified this by actually assembling both buggy matrices at $N=200$: each
   returns $-3.9999, 9.8694, 39.475, 88.810$, against the Dirichlet reference
   $\pi^2, 4\pi^2, 9\pi^2 = 9.8696, 39.478, 88.826$. **So the signature is: a
   Robin spectrum that comes out as $n^2\pi^2/L^2$ *with a spurious negative
   eigenvalue* at $-\kappa^2$.** Either symptom alone means you flipped one sign.
   (Case D2, unequal couplings, breaks the exact coincidence but still fails
   badly: correct $k_1 = 1.45177422$; bug (a) $0.73230426$ with no negative
   eigenvalue; bug (b) $k_1 = 2.45785654$ with a negative eigenvalue
   $\approx -9.06$.) *Test:* Case D and Case D2, plus the assertion
   $\lambda_{\min} > 0$ for all $\kappa>0$.
2. **Zero modes.** Three separate failures:
   (a) dropping the Neumann zero mode entirely;
   (b) normalising it with $\sqrt{2/L}$ instead of $1/\sqrt L$ — a $\sqrt2$ error
   (§3.3);
   (c) **never finding it at all**, because a sign-change scan of (3.12) cannot
   see it: for $\kappa_0=\kappa_L=0$ the equation is $-k^2\sin(kL)=0$, whose
   root at $k=0$ has even multiplicity ($\sim-k^3L$ near $0$) and therefore
   produces no sign change. Conversely, $k=0$ satisfies (3.12) *identically for
   every* $\kappa_0,\kappa_L$, so a root-finder that includes $k=0$ will report a
   **spurious** zero eigenvalue whenever (3.18) fails. **Handle $k=0$ exactly
   once, by the criterion (3.18), outside the scan.** *Test:* Case B row $n=0$
   exactly; discrete $\lambda_0 = O(10^{-14})$ with a constant eigenvector; and
   Case D (no zero mode) must *not* return one.
3. **Normalisation.** Three distinct traps: (i) including $1/\sqrt{2\omega}$ in
   $f_n$ (§1.5) — then $\int|f_n|^2 \ne 1$ and every comparison against §5 fails;
   (ii) using $\sqrt{2/L}$ for the Neumann zero mode (§3.3); (iii) forgetting
   that discrete eigenvectors satisfy $v^{\top}Wv=1$ and must be divided by
   $\sqrt h$ before comparing with continuum $f_n$ (§4.6) — a $\sqrt h$
   discrepancy that looks like slow convergence.
4. **The mode-number condition on the mixed interval.** It is
   $k_n = (n+\tfrac12)\pi/L$ with $n\ge0$ — not $n\pi/L$, not $n\ge1$ with
   $(n-\tfrac12)$, and with $n=-1$ excluded because it duplicates $n=0$
   ($k_{-1} = -k_0$). *Test:* Case C $\lambda_0 = 2.46740110027$; and
   $\lambda^{ND}_0 = \lambda^{DD}_1/4$.
5. **One-sided boundary differences / wrong ghost stencil.** At $x=L$ the
   central difference uses $f_N$ and $f_{N+2}$, not $f_{N+1}$ and $f_{N+2}$
   (4.4). A one-sided ghost relation still converges — at first order — and
   produces a smooth but wrong sequence. *Test:* measured convergence rate must
   be 2 (not 1), and for D/N the coefficient must be $-1/12$ (4.15).
6. **Symmetry and the mass matrix.** The ghost-eliminated matrix is not
   symmetric; symmetrise by halving the boundary rows **and** solve the
   generalised problem with $M=W$ (4.7)–(4.9). Halving the rows while solving
   the *ordinary* eigenproblem changes the eigenvalues. Solving with a general
   non-symmetric solver "works" but yields complex eigenvalues from round-off
   and unreliable ordering. *Test:* $\max|A_{\rm sym}-A_{\rm sym}^{\top}|=0$
   exactly; all eigenvalues real; $V^{\top}WV = I$ (and **not** $V^{\top}V=I$).
7. **Robin root ordering and the $\tan$ poles.** Enumerate roots by *ascending
   $k$*, from (3.12) rather than (3.13), with a scan step $<\pi/(4L)$; handle
   $k=0$ separately (3.18); do not let a $\tan$ pole be reported as a root.
   **And solve (3.19) too whenever a coupling is negative** — (3.12) alone gives
   only the positive eigenvalues, and I was myself caught by this while checking
   the wrong-sign diagnostic: a positive-only scan reports a "complete" spectrum
   that is missing the negative eigenvalue. *Test:* Case D's $k_1 < k_2 < \dots$
   against the table; the $\kappa\to0$, $\kappa\to\infty$ limits of §3.5.4; and
   for $\kappa_0=-\kappa_L=-2$, $L=1$ the spectrum must be
   $\{-4, \pi^2, 4\pi^2, \dots\}$.
8. **The grid convention $h = L/(N+1)$.** With $L/N$ or $L/(N-1)$ the scheme
   still converges (to the wrong values, at first order in $h$). *Test:* the
   exact Dirichlet closed form (4.14) to $10^{-12}$.
9. **Degenerate eigenvalues.** Do not deduplicate. A Dirichlet square returns
   the level at $49.348$ twice (Case F). Sorting by $(n_1,n_2)$ rather than by
   $\lambda$ also breaks Case E's ordering.
10. **The $2\pi$ factor.** Wavenumbers on a box of length $L$ are $n\pi/L$, not
    $2\pi n/L$; mode density is $L/\pi$ for $k>0$ (§1.7). A single stray $2\pi$
    in a wavenumber gives a spectrum wrong by $4\pi^2$ per mode — very visible,
    but a $2\pi$ in the *density* is invisible in a spectrum test and shows up
    only in mode-count/Weyl checks.
11. **`sqrt` of a negative $\omega^2$.** With $m^2<0$ or $\kappa<0$,
    $\lambda+m^2$ can be negative and $\omega$ imaginary — physically, an
    exponentially growing mode. Returning `nan`, or silently taking `abs`,
    destroys the physics. *Test:* Case D with $\kappa=-2$ must return a negative
    $\lambda_{\min}$ ($-5.756915$ for $L=1$), not an error and not $+5.756915$.
12. **Time direction treated as a box direction.** The KG equation is solved in
    the frequency domain; $t$ is not discretised and carries no boundary
    condition. Applying a BC to $t$ (e.g. "periodic in time") changes the
    problem entirely.

---

## 7. What I checked numerically, and what I did not

So the engineer knows exactly how much weight each claim carries.

**Checked symbolically (SymPy 1.14):**
- The Robin determinant (3.12) and the $\tan$ form (3.13), re-derived from the
  $2\times2$ boundary system.
- The limits of §3.5.4.
- The normalisation integral (3.17), including $I\to L/2$ at $c=0$, $kL=n\pi$.
- Orthonormality $\int_0^L f_nf_m = \delta_{nm}$ for the ND modes (3.7) on the
  pairs $(0,0),(0,1),(0,3),(2,5)$.
- The zero-mode criterion (3.18), two ways: the explicit solution
  $f = A + Bx$ giving $\kappa_0+\kappa_L+\kappa_0\kappa_L L = 0$, and direct
  boundary residuals for $f = 1-2x/L$ at $\kappa_0=\kappa_L=-2/L$ (zero at both
  ends). Checked against (3.18) for ten coupling pairs, including the
  asymmetric $(-3,-1.5)$ (zero mode) and $(-1,-2)$ (none).
- The identity behind the wrong-sign diagnostic (elementary algebra, then
  confirmed numerically by root-finding): for $\kappa_0 = -\kappa_L = a$, (3.12)
  reduces to $\sin(kL)=0$ for every $a$, so a single sign flip yields exactly the
  Dirichlet spectrum.
- The two-square multiplicity formula (3.22) against direct enumeration for
  $q = 1,2,4,5,8,9,10,13,16,17,18,20,25$.

**Checked numerically (NumPy 2.5 / SciPy 1.18):**
- The scheme of §4 against every closed form in §3: convergence rates $\to 2$
  for D–D, N–N, N–D, R–R (equal and unequal couplings), R–N, and 2D mixed.
- Exact symmetry of $A_{\rm sym}$ (zero difference, not small) for all cases and
  the 2D assembly.
- The exact discrete spectrum (4.14) against `eigh` for **every** mode at
  $N = 7$ and $N = 20$, for D–D, N–N and N–D (agreement $3$–$9\times10^{-14}$),
  and its 2D Kronecker-sum form for the five lowest mixed levels
  ($9\times10^{-14}$ at $N=8$).
- Nodal exactness of the eigenvectors for D–D and N–D modes 1, 3, 5
  ($\le 5\times10^{-15}$), and the constant N–N zero-mode vector.
- The relative-error law (4.15) for D–D, N–N, N–D (ratios 0.9982–1.0000) and
  its **failure** for Robin (ratios $-0.46$ and $0.71$).
- $V^{\top}WV = I$ to $1.4\times10^{-15}$ while $V^{\top}V - I$ reaches $0.142$.
- The $\sqrt h$ rescaling (4.13): nodal errors $2.06\times10^{-3}\to
  3.80\times10^{-5}$ (second order) for Robin; $2\times10^{-15}$ (exact) for
  N–D.
- The Neumann zero mode: constant eigenvector to 12 digits, $\lambda_0 =
  5.8\times10^{-14}$.
- The wrong-sign diagnostics of §6 item 1: both buggy assemblies ($N=200$,
  $L=1$, $\kappa=2$) return $-3.9999, 9.8694, 39.475, 88.810$ — the Dirichlet
  spectrum plus the negative eigenvalue $-\kappa^2$ predicted by (3.19). The
  Case D2 couplings give $k_1 = 0.7323$ (bug a, no negative eigenvalue) and
  $2.4579$ with $\lambda \approx -9.06$ (bug b).
- The small-$\kappa$ table of §3.5.5: $k_1$ for
  $\kappa = 10^{-3},10^{-2},0.1,1,2,10,10^4$, confirming
  $\lambda_1\to2\kappa/L$ as $\kappa\to0^+$.
- Negative eigenvalues for $\kappa<0$ matching the imaginary-wavenumber roots
  $k=i\mu$, $\lambda=-\mu^2$, to 5 digits at $N=400$; the $\kappa=-2/L$ zero mode
  appearing as the second eigenvalue.
- The $\kappa\to\infty$ law (2.6) to $1.3\times10^{-11}$.
- 2D degeneracies (Case F) and the Case E ordering.
- The reference table itself: generated with mpmath at 50 digits, then the
  $k_n$ values re-checked against $\tan k = 4k/(k^2-4)$ and against direct BC
  residuals of the normalised eigenfunctions ($\le 10^{-49}$).

**Not checked / deliberately out of scope:**
- **The full 3D (or $p$-dimensional) generalisation.** §4.5's Kronecker assembly
  is written for 2D; the extension to more directions is the obvious iteration
  but I have not tested it.
- **Higher-order schemes.** I specify and verify the second-order ghost scheme
  only. A compact fourth-order scheme would change the convergence assertions.
- **Non-rectangular or non-separable domains.** Everything here depends on
  separability; a general domain needs a different (e.g. finite-element)
  treatment and the analytic reference values in §5 would not apply.
- **The string-theoretic mode expansions of §2.4/§2.6.** These are stated as
  standard background and I did not re-derive the ND-string oscillator shifts;
  I deliberately kept them out of the solver's scope (§2.6) rather than risk
  mixing two different "half-integer" statements.
- **Very large $\kappa L$ in the Robin root-finder.** I tested to
  $\kappa L = 10^8$. Beyond that, the determinant (3.12) is dominated by the
  $\kappa_0\kappa_L$ term and root-finding in `float` will lose precision; the
  dimensionless form (3.15) is better conditioned and should be used.

**Where I was genuinely uncertain.** Two places, flagged rather than papered
over:

1. **The exactness of the nodally-exact trigonometric modes (§4.7 item 3).** I
   verified it to machine precision for N–D mode 0 and for the N–N zero mode,
   and the argument (constant-coefficient difference equation + exactly imposed
   boundary conditions) is convincing, but I did not prove it for all modes of
   all three cases, and I did not test it for high mode numbers. Treat it as a
   verified numerical fact for the low modes, not as a theorem; use the rate
   test (4.15) as the primary convergence assertion and this as a bonus.
2. **The number of negative eigenvalues for $\kappa<0$ in the asymmetric case.**
   I verified "exactly one" for the symmetric case at four couplings and for one
   asymmetric case, and proved only "at least one" (the variational bound of
   §3.5.6). I would not assert a general count.

---

## References

Standard textbooks and papers, cited by author, title, year. Where I state a
standard result I am relying on the general content of these, not on a specific
equation.

1. J. Polchinski, *Dirichlet-Branes and Ramond-Ramond Charges*, Phys. Rev. Lett.
   **75** (1995) 4724.
2. J. Polchinski, *String Theory, Vol. 1: An Introduction to the Bosonic
   String*, Cambridge University Press, 1998.
3. C. V. Johnson, *D-Branes*, Cambridge University Press, 2003.
4. A. Recknagel and V. Schomerus, *Boundary Conformal Field Theory and the
   String Theory of Open Strings*, Cambridge University Press, 2013.
5. R. Courant and D. Hilbert, *Methods of Mathematical Physics, Vol. I*,
   Interscience Publishers, 1953. (Sturm–Liouville theory; self-adjoint
   boundary conditions; Robin conditions.)
6. G. B. Arfken, H. J. Weber and F. E. Harris, *Mathematical Methods for
   Physicists*, 7th edition, Academic Press, 2013. (Orthogonal expansions,
   Sturm–Liouville problems.)
7. R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial
   Differential Equations*, SIAM, 2007. (Ghost-node treatment of boundary
   conditions; truncation-error analysis; eigenvalue convergence.)
8. K. W. Morton and D. F. Mayers, *Numerical Solution of Partial Differential
   Equations*, 2nd edition, Cambridge University Press, 2005.
9. G. H. Hardy and E. M. Wright, *An Introduction to the Theory of Numbers*,
   Oxford University Press. (Sums of two squares; the formula for $r_2(q)$.)
10. A. Sen, *Tachyon Condensation on the Brane Antibrane System*, JHEP **08**
    (1998) 012. (The open-string tachyon, $m^2 = -1/\alpha'$ in our
    convention.)
