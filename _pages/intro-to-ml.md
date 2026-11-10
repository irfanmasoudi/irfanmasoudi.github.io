---
layout: page
title: Intro to ML
permalink: /notes/intro-to-ml/
description: Linear algebra and statistics recap for machine learning.
nav: false
toc:
  sidebar: right
chart:
  plotly: true
---

This note is a recap of the mathematical background used in the Machine
Learning: linear algebra (least squares and the eigenvector problem) and
probability and statistics (distributions, confidence intervals, hypothesis
testing, and the Bayesian approach).

## Reference


- Bishop, C.M., _Pattern Recognition and Machine Learning_, 2006, Springer.
- Montgomery, D.C., Runger, G.C., _Applied Statistics and Probability for Engineers_, 2010, John Wiley & Sons.

## Linear algebra

### Least squares

Consider a dataset of $N$ inputs $x_i = (x_{i1}, \dots, x_{iD})$ with
$x_{ij} \in \mathbb{R}$, so that each input has dimension $D$, and a target
$t_i \in \mathbb{R}$ for each input. We want to predict the target $t_i$ with a
linear combination of the input $x_i$, i.e. we want to find a vector of
parameters

$$
w := (w_1, \dots, w_D)^\top
$$

that minimizes a loss function. When the loss is the sum of the squared
prediction errors, we are performing **least squares** minimization.

#### Loss function

The least squares loss is

$$
L(w) := \frac{1}{2} \sum_{i=1}^{N}
\left( t_i - \sum_{j=1}^{D} x_{ij} w_j \right)^2 .
$$

Defining the design matrix (each row is an input vector $x_i$)

$$
X =
\begin{pmatrix}
x_1 \\
\vdots \\
x_N
\end{pmatrix}
=
\begin{pmatrix}
x_{11} & \cdots & x_{1D} \\
\vdots & \ddots & \vdots \\
x_{N1} & \cdots & x_{ND}
\end{pmatrix}
\in \mathbb{R}^{N \times D},
$$

and the residual (error) vector

$$
r := t - Xw =
\begin{pmatrix}
r_1 \\
\vdots \\
r_N
\end{pmatrix}
=
\begin{pmatrix}
t_1 - \sum_{j=1}^{D} x_{1j} w_j \\
\vdots \\
t_N - \sum_{j=1}^{D} x_{Nj} w_j
\end{pmatrix}
\in \mathbb{R}^{N},
$$

the loss can be written equivalently as

$$
L(w) = \frac{1}{2} \lVert r \rVert_2^2
       = \frac{1}{2} r^\top r
       = \frac{1}{2} \sum_{i=1}^{N} r_i^2
       = \frac{1}{2} (t - Xw)^\top (t - Xw) .
$$

#### Minimizing the loss

To minimize the loss we compute its gradient with respect to each component of
$w$:

$$
\frac{\partial L(w)}{\partial w} :=
\left( \frac{\partial L(w)}{\partial w_1}, \dots,
       \frac{\partial L(w)}{\partial w_D} \right).
$$

There are two equivalent ways to compute it: an elementwise (traditional)
derivative and a matrix derivative.

**Traditional (elementwise) derivative.** Since only the term $x_{ih} w_h$
depends on $w_h$ inside the inner sum,

$$
\begin{aligned}
\left( \frac{\partial L(w)}{\partial w} \right)_h
= \frac{\partial L(w)}{\partial w_h}
&= \frac{\partial}{\partial w_h}
\left[
\frac{1}{2} \sum_{i=1}^{N}
\left( t_i - \sum_{j=1}^{D} x_{ij} w_j \right)^2
\right] \\
&= \frac{1}{2} \sum_{i=1}^{N} 2
\left( t_i - \sum_{j=1}^{D} x_{ij} w_j \right)(-x_{ih}) \\
&= -\sum_{i=1}^{N} x_{ih}
\left( t_i - \sum_{j=1}^{D} x_{ij} w_j \right).
\end{aligned}
$$

For example, with $D = 3$ the inner sum is
$x_{i1} w_1 + x_{i2} w_2 + x_{i3} w_3$; for $h = 2$ the derivative picks out the
coefficient $x_{i2}$ of $w_2$.

**Matrix derivative.** Starting from the residual vector $r = t - Xw$, and using
$\partial r / \partial w = -X$,

$$
\begin{aligned}
\frac{\partial L(w)}{\partial w}
&= \frac{\partial}{\partial w}
\left[ \frac{1}{2} (t - Xw)^\top (t - Xw) \right] \\
&= \frac{\partial}{\partial w}
\left[ \frac{1}{2} r^\top r \right]
= -X^\top r
= -X^\top (t - Xw).
\end{aligned}
$$

Written componentwise, this is

$$
\left( \frac{\partial L(w)}{\partial w} \right)_h
= -\begin{pmatrix} x_{1h} & \cdots & x_{Nh} \end{pmatrix}
\begin{pmatrix} r_1 \\ \vdots \\ r_N \end{pmatrix}
= -\sum_{i=1}^{N} x_{ih} r_i
= -\sum_{i=1}^{N} x_{ih}
\left( t_i - \sum_{j=1}^{D} x_{ij} w_j \right),
$$

which matches the elementwise derivative.

Setting the gradient to zero gives the normal equations

$$
-X^\top (t - Xw) = 0
\quad\Longrightarrow\quad
X^\top X w = X^\top t,
$$

whose solution (when $X^\top X$ is invertible) is

$$
\hat{w} = (X^\top X)^{-1} X^\top t .
$$

### Eigenvector problem

Given a square matrix $A \in \mathbb{R}^{N \times N}$, the eigenvalue
equations are

$$
A v_i = \lambda_i v_i .
$$

- The **eigenvectors** $v_1, \dots, v_N$ are the directions along which vectors
  are not deviated by the transformation $A$.
- The **eigenvalues** $\lambda_1, \dots, \lambda_N$ are the stretch factors for
  the directions $v_i$.

#### Graphical representation

When $v$ is an eigenvector of $A$, applying $A$ does not change its direction: it
only stretches it by the eigenvalue $\lambda$, so $A v$ lies on the same line
through the origin as $v$. In the interactive plot below we use

$$
A = \begin{pmatrix} 1 & 2 \\ 1 & 0 \end{pmatrix},
\qquad
v = \begin{pmatrix} 2 \\ 1 \end{pmatrix},
\qquad
\lambda = 2,
\qquad
A v = \begin{pmatrix} 4 \\ 2 \end{pmatrix} = 2v .
$$

```plotly
{
  "data": [],
  "layout": {
    "title": { "text": "Eigenvector: A v = λ v" },
    "xaxis": {
      "range": [-0.8, 5],
      "zeroline": true,
      "zerolinecolor": "#999999",
      "title": { "text": "x" },
      "showgrid": true,
      "gridcolor": "rgba(128,128,128,0.25)"
    },
    "yaxis": {
      "range": [-0.8, 3],
      "scaleanchor": "x",
      "scaleratio": 1,
      "zeroline": true,
      "zerolinecolor": "#999999",
      "title": { "text": "y" },
      "showgrid": true,
      "gridcolor": "rgba(128,128,128,0.25)"
    },
    "showlegend": false,
    "hovermode": "closest",
    "margin": { "l": 50, "r": 30, "t": 50, "b": 50 },
    "shapes": [
      { "type": "line", "x0": 2, "x1": 2, "y0": 0, "y1": 1, "line": { "color": "#999999", "width": 1, "dash": "dot" } },
      { "type": "line", "x0": 0, "x1": 2, "y0": 1, "y1": 1, "line": { "color": "#999999", "width": 1, "dash": "dot" } },
      { "type": "line", "x0": 4, "x1": 4, "y0": 0, "y1": 2, "line": { "color": "#999999", "width": 1, "dash": "dot" } },
      { "type": "line", "x0": 0, "x1": 4, "y0": 2, "y1": 2, "line": { "color": "#999999", "width": 1, "dash": "dot" } }
    ],
    "annotations": [
      { "x": 4, "y": 2, "xref": "x", "yref": "y", "ax": 0, "ay": 0, "axref": "x", "ayref": "y", "showarrow": true, "arrowhead": 3, "arrowsize": 1.5, "arrowwidth": 6, "arrowcolor": "#ff9896", "text": "" },
      { "x": 2, "y": 1, "xref": "x", "yref": "y", "ax": 0, "ay": 0, "axref": "x", "ayref": "y", "showarrow": true, "arrowhead": 3, "arrowsize": 1.5, "arrowwidth": 6, "arrowcolor": "#d62728", "text": "" },
      { "x": 3.1, "y": 2.05, "xref": "x", "yref": "y", "text": "<b>Av = λv</b>", "showarrow": false, "font": { "color": "#d62728", "size": 15 }, "align": "left" },
      { "x": 0.95, "y": 0.72, "xref": "x", "yref": "y", "text": "<b>v</b>", "showarrow": false, "font": { "color": "#d62728", "size": 15 } },
      { "x": 2, "y": -0.35, "xref": "x", "yref": "y", "text": "x", "showarrow": false, "font": { "color": "#888888", "size": 13 } },
      { "x": 4, "y": -0.35, "xref": "x", "yref": "y", "text": "λx", "showarrow": false, "font": { "color": "#888888", "size": 13 } },
      { "x": -0.45, "y": 1, "xref": "x", "yref": "y", "text": "y", "showarrow": false, "font": { "color": "#888888", "size": 13 } },
      { "x": -0.45, "y": 2, "xref": "x", "yref": "y", "text": "λy", "showarrow": false, "font": { "color": "#888888", "size": 13 } }
    ]
  }
}
```

In matrix form, for a generic pair $(\lambda, v)$,

$$
A v = \lambda v
\quad\Longrightarrow\quad
(A - \lambda I_N) v = 0 .
$$

This problem has a non-null solution only if the matrix $A - \lambda I_N$ is not
full rank, or equivalently if

$$
\lvert A - \lambda I_N \rvert = 0 .
$$

#### "Eigen" properties

- The rank of $A$ equals the number of non-zero eigenvalues.
- The determinant of $A$ equals the product of its eigenvalues:

$$
\lvert A \rvert = \prod_{i=1}^{N} \lambda_i .
$$

- The trace of $A$ equals the sum of its eigenvalues:

$$
\operatorname{Tr}(A) = \sum_{i=1}^{N} \lambda_i .
$$

#### Two useful definitions

- A matrix $A$ is **positive definite** if $x^\top A x > 0$ for all
  $x \in \mathbb{R}^N \setminus \{0\}$.
- A matrix $A$ is **semi-positive definite** if $x^\top A x \ge 0$ for all
  $x \in \mathbb{R}^N \setminus \{0\}$.

An easier way to check these properties is through the eigenvalues:

- A positive definite matrix has all positive eigenvalues, i.e.
  $\lambda_i > 0$ for all $i$.
- A semi-positive definite matrix has all non-negative eigenvalues, i.e.
  $\lambda_i \ge 0$ for all $i$.

## Probability and statistics

### Distributions

#### Discrete random variables

A discrete random variable $X$ is a variable with values in a discrete set $E$
whose value is determined by a stochastic phenomenon. We define a probability
function $P : E \to [0, 1]$ that tells how often an event in $E$ occurs:

$$
P(X = i) = \frac{\lvert i \rvert}{\lvert E \rvert} .
$$

For instance, in a 20-faced die:

$$
E = \{1, \dots, 20\},
\qquad
P(X = i) = \frac{1}{20} .
$$

A properly defined probability function satisfies

$$
0 \le P(X = i) \le 1
\qquad \text{and} \qquad
\sum_{i \in E} P(X = i) = 1 .
$$

#### Cumulative function

Assuming the events are ordered, the cumulative function defines the
probability of multiple events:

$$
F(i) := P(X \le i)
= \sum_{h=1}^{i} P(X = h)
= \sum_{h \in E,\, h \le i} \frac{\lvert h \rvert}{\lvert E \rvert} .
$$

For instance, in a 20-faced die:

$$
F(i) = \sum_{h=1}^{i} \frac{1}{20} = \frac{i}{20} .
$$

A properly defined cumulative function satisfies

$$
0 \le F(i) \le 1,
\qquad
F(i) = 0 \ \forall i < \min_{h \in E} h,
\qquad
F(i) = 1 \ \forall i \ge \max_{h \in E} h .
$$

#### Characterizing the distribution

The quantities that characterize a random variable are the expected value and
the variance (the first two moments):

$$
\mathbb{E}[X] := \sum_{i \in E} i \, P(X = i),
$$

$$
\operatorname{Var}(X) := \sum_{i \in E} \left( \mathbb{E}[X] - i \right)^2
P(X = i) .
$$

For instance, in a 20-faced die:

$$
\mathbb{E}[X] = \sum_{i=1}^{20} \frac{i}{20} = \frac{21}{2} = 10.5 .
$$

Sometimes we compute the standard deviation instead of the variance:

$$
\operatorname{std}(X) = \sqrt{\operatorname{Var}(X)} .
$$

#### Continuous random variables

If the set $E$ is not discrete, i.e. $E \subseteq \mathbb{R}$, we define the
**probability density function** (pdf):

$$
f(x) := \lim_{\delta x \to 0}
\frac{P(x \le X \le x + \delta x)}{\delta x},
$$

with properties

$$
f(x) \ge 0 \ \forall x \in \Omega,
\qquad
\int_{x \in \Omega} f(x) \, dx = 1 .
$$

The **cumulative density function** (CDF) is defined as

$$
F(x) := \int_{s \in \Omega,\, s \le x} f(s) \, ds,
$$

with properties

$$
0 \le F(x) \le 1 \ \forall x \in \Omega,
\qquad
F\left( \min_{x \in \Omega} x - \varepsilon \right) = 0 \ \forall \varepsilon > 0,
\qquad
F\left( \max_{x \in \Omega} x \right) = 1 .
$$

A **quantile of order** $\alpha$ is a point $z_\alpha \in \Omega$ such that

$$
F(z_\alpha) = 1 - \alpha,
$$

i.e. a point of the domain leaving a cumulated probability $1 - \alpha$ to its
left.

#### Characterizing continuous random variables

The expected value and the variance for a continuous random variable are

$$
\mu = \mathbb{E}[X] := \int_{x \in \Omega} x f(x) \, dx,
\qquad
\sigma^2 = \operatorname{Var}(X) :=
\int_{x \in \Omega} \left( \mathbb{E}[X] - x \right)^2 f(x) \, dx .
$$

For the Gaussian case:

$$
f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}}
e^{-\frac{(x - \mu)^2}{2\sigma^2}},
\qquad
F(x; \mu, \sigma) = \int_{-\infty}^{x} f(t; \mu, \sigma) \, dt .
$$

#### Estimating the distributions

Usually we do not have any information about the distributions, so we need to
estimate their mean and variance:

$$
\bar{X} := \frac{\sum_{i=1}^{n} x_i}{n},
\qquad
s^2 := \frac{\sum_{i=1}^{n} \left( \bar{X} - x_i \right)^2}{n - 1} .
$$

These are consistent estimators for the expected value and the variance,
respectively.

#### Central Limit Theorem

Let us consider the empirical expected value $\bar{X}$. We recall the Central
Limit Theorem (CLT):

> **Theorem (Central Limit Theorem).** Assume $\{X_1, \dots, X_n\}$ is a
> sequence of independent and identically distributed (i.i.d.) random variables
> with $\mathbb{E}[X_i] = \mu$ and $\operatorname{Var}[X_i] = \sigma^2 < \infty$.
> Then
>
> $$
> \sqrt{n} \left( \frac{\sum_{i=1}^{n} X_i}{n} - \mu \right)
> \to \mathcal{N}(0, \sigma^2),
> $$
>
> where the convergence holds in distribution.

#### Empirical distributions

Even when samples come from different distributions, the sample mean exhibits
the same limiting distribution, which is the empirical evidence of the CLT.

### Confidence intervals

We need to set a level identifying whether our estimator is "good enough". The
probability that $\bar{X} = \mathbb{E}[X]$ is zero, since the realization of the
expected value is itself a continuous random variable. We therefore build
intervals where we have high confidence that the true mean $\mathbb{E}[X]$
lies. A 95% confidence interval works 95% of the times.

Options for the confidence intervals:

**Gaussian approximation:**

$$
\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
\le \mu \le
\bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}} .
$$

**Chebyshev inequality** (requires
$\mathbb{E}[X] = \mu < \infty$ and $\operatorname{Var}[X] = \sigma^2 < \infty$):

$$
\bar{X} - \frac{\sigma}{\sqrt{n\alpha}}
\le \mu \le
\bar{X} + \frac{\sigma}{\sqrt{n\alpha}} .
$$

**Hoeffding inequality** (finite support):

$$
\bar{X} - (b - a) \sqrt{\frac{-\log(\alpha/2)}{2n}}
\le \mu \le
\bar{X} + (b - a) \sqrt{\frac{-\log(\alpha/2)}{2n}} .
$$

### Hypothesis testing

A scientific statement we may want to show is that the estimated parameter
$\bar{X}$ is equal to $\mu$, or that an estimated parameter $\bar{X}'$ is
different from another estimated parameter $\bar{X}''$.

Statistics answers by defining a **null hypothesis** $H_0$ and an **alternative
hypothesis** $H_1$:

$$
H_0 : \mu = \mu_0
\qquad \text{vs.} \qquad
H_1 : \mu \ne \mu_0,
$$

and then uses the data to provide evidence that either one or the other holds.

#### Basic Gaussian test

Given the data $\{x_1, \dots, x_n\}$ we have

$$
\bar{X} \sim \mathcal{N}\left( \mu, \frac{\sigma^2}{n} \right)
\quad\Longrightarrow\quad
t = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1) .
$$

Fixing a confidence $1 - \alpha$, with $\alpha \in (0, 1)$, the test statistic
$t$ should be close to the real mean $\mu$ with "high" probability:

$$
P\left( t < z_{\alpha/2} \ \lor \ t > z_{1-\alpha/2} \right) = \alpha .
$$

| Decision          | Fail to reject $H_0$ | Reject $H_0$           |
| ----------------- | -------------------- | ---------------------- |
| **True $H_0$**    | Correct              | Type I error ($\alpha$)|
| **True $H_1$**    | Type II error        | Correct                |

#### P-value

To avoid specifying the confidence $\alpha$, we can let the data tell us how
confident we can be about their correspondence to a specific hypothesis:

- Small p-values imply that we are confident that the $H_1$ hypothesis holds.
- Large p-values imply that we are not able to tell that the $H_0$ hypothesis
  does not hold.

### Bayesian approach

Confidence bounds include some information about the data, but they do not
allow us to incorporate information one has about the distribution parameters.
The new idea is that the expected value $\mu$ of the random variable is itself
a random variable. We use the Bayes formula to update this information:

$$
P(a \mid b) = \frac{P(b \mid a) P(a)}{P(b)} .
$$

#### Example: Bernoulli variables

Consider the Bayes formula applied to a sequence of observations:

$$
\begin{aligned}
P(\mu \mid x_1, \dots, x_t)
&= \frac{P(x_1, \dots, x_{t-1}, x_t \mid \mu) P(\mu)}
        {P(x_1, \dots, x_t)} \\
&\propto P(x_t \mid \mu) P(x_1, \dots, x_{t-1} \mid \mu) P(\mu) \\
&= P(x_t \mid \mu) P(x_{t-1} \mid \mu) P(x_1, \dots, x_{t-2} \mid \mu) P(\mu) \\
&= P(\mu) \prod_{h=1}^{t} P(x_h \mid \mu) .
\end{aligned}
$$

We incorporate information incrementally from a prior distribution $P(\mu)$.

For example, if we have a prior telling us that there were 3 successes over 10
trials, using a Beta distribution as prior for $\mu$
($\mu \sim \operatorname{Beta}(3, 7)$), the posterior is still a Beta
distribution, i.e. Bernoulli and Beta form a conjugate prior-posterior pair.
