
# Blind Source Separation Note



## Aux IVA

Blind Source Separation using independent vector analysis based on auxiliary function. This function will separate the input signal into statistically independent sources without using any prior information.

The algorithm in the determined case, i.e., when the number of sources is equal to the number of microphones, is AuxIVA [1]. When there are more microphones (the overdetermined case), a computationally cheaper variant (OverIVA) is used [2].

> ***AuxIVA***
>
> [1] N. Ono, **Stable and fast update rules for independent vector analysis based on auxiliary function technique**, Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.
>
> ***OverIVA***
>
> [2] R. Scheibler and N. Ono, Independent Vector Analysis with more Microphones than Sources, arXiv, 2019. https://arxiv.org/abs/1905.07880

### Original objective function of IVA

$$
J(\mathbf{W})=\sum_{k=1}^K \frac{1}{N_\tau}\sum_{\tau=1}^{N_\tau}G(\mathbf{y}_k(\tau))-\sum_{w=1}^{N_w}log|\det W(w)| \cr
\mathbf y_k(\tau) = (y_k(1,\tau)\cdots y_k(N_w, \tau))^t
$$

$G(y_k)$ is called *contrast function*, where
$$
G(\mathbf{y}_k(\tau)) = -\log p(\mathbf{y}_k(\tau))
$$
while doing blind source separation, the distribution of $\mathbf y_k(\tau)$ (the representation of a multivariate p.d.f. of sources) is given as prior, for example in `pyroomacoustic`, there are *Laplace* and *Gauss* distributions to choose. then considering this spherical contrast functions:
$$
G(\mathbf{y}_k) = G_R(r_k) \cr
r_k = ||\mathbf{y}_k||_2=\sqrt{\sum_{w=1}^{N_w}|y_k(w)|^2}
$$

### Original updating rule

$$
W(w) \larr W(w) + \mu(I-E[\phi_w(y) y^h(w) ])W(w) \cr
\phi_w(y) = (\phi_{1w}(y_1), \dots, \phi_{kw}(y_k))^t\cr
\phi_{kw}(y_k)=\frac{\partial G(y_k)}{\partial y_k^\star(w)}
$$
where $\phi_{kw}(y_k)$ is called *natural gradient*.

### Auxiliary function technique

*An extension of EM algorithm*

#### Aux- Introduction

$$
\begin{aligned}
\theta^\dagger &= \arg\min_\theta J(\theta)\cr
J(\theta) &= \min_\theta Q(\theta, \tilde\theta)\cr
\tilde\theta^{(i+1)} &= \arg\min_{\tilde\theta}Q(\theta^{(i)}, \tilde\theta)\cr
\theta^{(i+1)} &= \arg\min_\theta Q(\theta, \tilde\theta^{(i+1)})
\end{aligned}
$$

where $Q(\cdot)$ is called an *auxiliary function*, $\theta$ and $\theta^\dagger$ is *parameter vectors*, and $\tilde\theta$ is called *auxiliary variables*. According to above, the **monotonic decrease** of $J(\theta)$ is guaranteed. 

#### Auxiliary function of IVA contrast function

**Definition 1**:

Defining a set of real-valued functions:
$$
S_G = \{ G(z) | G(z) = G_R(||z||_2) \}
$$
where $G_R(r)$ is a continuous and differentiable function *(see [above](#original-objective-function-of-iva))*, and it is **derived from the *super-Gaussianity* of the assumed source PDF.** 

Note: in several literature, the contrast functions may as:
$$
\begin{aligned}
G_1(z) &= C\cdot ||z||_2\cr
G_2(z) &= m\cdot \log\cosh (C\cdot||z||_2)
\end{aligned}
$$
**Theorem 1**:

>For any $ G(z) = G_R(||z||_2) \in S_G$, 
>$$
>G(z) \le \frac{G_R'(r_0)}{2r_0}||z||_2^2 +(G_R(r_0)-\frac{r_0G_r'(r_0)}{2})
>$$
>holds for any $z$ and $r_0$. The equality sign is satisfied if and only if $r_0=||z||_2$.

***indicating***: the right side can be the auxiliary function for $G(z)$ (*See [above](#aux--introduction)*).



**Theorem 2**:

>For any $ G(z) = G_R(||z||_2) \in S_G$,  let
>$$
>Q(W, V)=\sum_{w=1}^{N_w} Q_w(W(w), \mathbf V(w)) \cr
>Q_w(W(w), \mathbf V(w)) = \frac{1}{2} \sum_{k=1}^K \mathbf w_k^h(w)V_k(w)\mathbf w_k(w) - \log|\det W(w)|+R
>$$
>where
>$$
>V_k(w) = E[\frac{G'_R(r_k)}{r_k}\mathbf x(w)\mathbf x^h(w)]
>$$
>and $r_k$ is a positive random variable, $R$ is a constant term independent of $W$.
>holds for any $z$ and $r_0$. The equality sign is satisfied if and only if $r_0=||z||_2$.

### Details in code

#### Common Functions / techniques

`projection_back(Y, ref)` from `pyroomacoustic`

This function computes the frequency-domain filter that minimizes the squared error to a reference signal. This is commonly used to solve the scale ambiguity in BSS. 

### Related Knowledge

***Frobenius norm***  sometimes also called the Euclidean norm (a term unfortunately also used for the vector $L^2$ norm), is matrix norm of an $m\times n$ matrix $A$, defined as $||A||_F$:
$$
\begin{aligned}
||A||_F &= [\sum_{i,j}abs(a_{i,j})^2]^{\frac{1}{2}} \cr
&= \sqrt{\sum_{i=1}^m\sum_{j=1}^n|A_{i,j}|^2} \cr
&= \sqrt{\text{Tr}(AA^h)}
\end{aligned}
$$
***Probability and Likelihood functions*** Probability is used to describe the plausibility of some data, given a value for the parameter. Likelihood is used to describe the plausibility of a value for the parameter, given some data.
*Likelihood function*
$$
L(x|y) = P(y=y_i|x)=\Pi_{j=0}^mP(y=y_i|x_i)
$$

## NMF

### $\beta$-divergence

$$
d_\beta(x|y)\xlongequal{\text{def}}
\begin{cases}
\frac{1}{\beta(\beta-1)}(x^\beta+(\beta-1)y^\beta-\beta xy^{\beta-1}) & \beta \in \mathcal{R}\quad{\rm and}\quad (x\neq0,1)\cr
x\log\frac{x}{y}-x+y=d_\text{KL}(x|y) & \beta = 1\cr
\frac{x}{y}-\log\frac{x}{y}-1=d_\text{IS}(x|y) & \beta =0
\end{cases}
$$

Note that if $\beta = 2$, $d_\beta(x|y)$ is equivalent to $d_Q(x|y)$, and a noteworthy property of $\beta$-divergence is as the following equation:
$$
d_\beta(\lambda x|\lambda y)=\lambda^\beta d_\beta(x|y)
$$

## ILRMA

## Multichannel Variational Autoencoder

### Conditional Variational Autoencoder
#### The hidden variable $Z$
$$
p(Z)=\sum_X p(Z|X)p(X)=\sum_X \mathcal{N}(0,I)p(X)=\mathcal{N}(0,I) \sum_X p(X) = \mathcal{N}(0,I)
$$
#### Log-likelihood function
$$
\begin{aligned}
\log p_\theta(s) &= \log \int q_\phi (\mathbf{z|s})\frac{p_\theta(\mathbf{s|z})p(\mathbf{z})}{q_\phi(\mathbf{z|s})} d \mathbf{z}\cr
&\ge \int q_\phi (\mathbf{z|s})\log\frac{p_\theta(\mathbf{s|z})p(\mathbf{z})}{q_\phi(\mathbf{z|s})} d \mathbf{z}\cr
&= \mathbb{E}_{\mathbf{z}\sim q_\phi (\mathbf{z|s})}[\log q_\phi(\mathbf{z|s})]-\text{KL}(q_\phi (\mathbf{z|s})\|p(\mathbf{z}))
\end{aligned}
$$

## FastMNMF

### General framework and spatial covariance models

> Duong, Ngoc Q K, Emmanuel Vincent, and Rémi Gribonval. “Under-Determined Reverberant Audio Source Separation Using a Full-Rank Spatial Covariance Model.” *IEEE Transactions on Audio, Speech, and Language Processing* 18, no. 7 (September 2010): 1830–40. https://doi.org/10.1109/TASL.2010.2050716.

#### General framework

- $v_j(n,f)$ encodes *time-varying **Spectro-Temporal Power***
- ${\bf R}_j(f)$ denotes *time-invariant **Spatial Covariance Matrix*** 
- ${\bf c}_j(n,f)\approx {\bf h}_j(f)s_j(n,f)$ denotes the STFT coefficients of the ***spatial images*** of the $j$th source, that is the contribution of this source to all mixture channels.sources 
- ${\bf h}_j(f)$ denotes the vector of filter coefficients *modeling the **acoustic path** from this source to all microphones*

So, for the Mixture signals:
$$
{\bf R}_{{\bf x}}(n,f)=\sum_{j=1}^J v_j(n,f)\,{\bf R}_j(f) = \sum_{j=1}^{J}{\bf R}_{{\bf c}_j}(n,f)
$$
In other words, the *Likelihood Function* for mixture signals given all $v$ and $R$ is:
$$
P({\bf x}\,\vert\,v,{\bf R})=\prod_{n,f}{1 \over \det (\pi{\bf R}_{{\bf x}}(n,f))}e^{-{\bf x}^H(n,f){\bf R}_{\bf x}^{-1}(n,f){\bf x}(n,f)}
$$

#### Rank-1 Convolutive Model[^WhyConvRank1?]

Most existing approaches to audio source separation rely on **narrowband approximation** of the **convolutive mixing process** by the complex-valued multiplication. The covariance matrix of $\mathbf{c}_j(n,f)$ is
$$
{\bf R}_{{\bf c}_j}(n,f)=v_j(n,f)\,{\bf R}_j(f)
$$
where ${v}_j(n,f)$ is the variance of ${\bf s}_j(n,f)$ and ${\bf R}_j(f)$ is equal to the rank-1 matrix:
$$
{\bf R}_j(f)={\bf h}_j(f){\bf h}_j^H(f)
$$
with the $I\times 1$ mixing vector ${\bf h}_j(f)$ denoting the Fourier transform of the mixing filters ${\bf h}_j(\tau)$. 

[^WhyConvRank1?]:this **Rank-1 Convolutive** model treats each *IMAGE* source also as an individual one source. So the fact is that, for *anechoic situation*, the $J$ indicates all sources, however, which are only real sources but no image ones. For ***latent** reverberant situation*(only direct and early echos), the $J$ will be extended to the number of sources which including all *real* sources and all *image* sources (based on the Image Source Method sense).

#### Rank-1 Anechoic Model

For omnidirectional microphones in an anechoic recording environment without reverberation, each mixing filter boils down to the combination of a delay
$$
\tau_{ij}={r_{ij} \over c}\quad{\rm and}\quad \kappa_{ij}={1 \over \sqrt{4\pi} r_{ij}}
$$

where $c$ is sound velocity. The spatial covariance matrix of $j$-th source is hence given by the rank-1 anechoic model.
$$
{\bf R}_j(f)={\bf a}_j(f){\bf a}_j^H(f)
$$
where the Fourier transform ${\bf a}_j(f)$ of the mixing filters, *actually the ${\bf h}_j(\tau)$ as mentioned above*, is now **parameterized** as
$$
{\bf a}_j(f)=
\begin{pmatrix}
\kappa_{1,j}e^{-2i\pi f\tau _{1,j}}\cr \vdots \cr \kappa_{I,j}e^{-2i\pi f\tau _{I,j}}
\end{pmatrix}
$$

#### Full-Rank Direct+Diffuse Model

One possible interpretation of the narrowband approximation is that the sound of each source as recorded on the microphones comes from a single spatial position at each frequency $f$, as ${\bf h}_j(f)$ or ${\bf a}_j(f)$ mentioned above. This approximation is not valid in a reverberant environment, since reverberation induces some spatial spread of each source, due to echoes at many different positions on the walls of the recording room. This spread translates into full-rank spatial covariance matrices.

The spatial image of each source is composed of two uncorrelated parts:

- a direct part, which is modeled by ${\bf a}_j(f)$ 
- a reverberant part.

The spatial covariance ${\bf R}_j(f)$ of each source is then a full-rank matrix defined as below:
$$
{\bf R}_j(f)={\bf a}_j(f){\bf a}_j^H(f) +\sigma_{\rm rev}^2{\bf \Psi}(f)
$$
and in this equation:

- $\sigma_{\rm rev}^2$ is the variance of the reverberant part
- ${\bf \Psi}_{il}(f)$ is a *function* of the distance $d_{il}$ between the $i$th and the $l$th microphone.
  - such that ${\bf \Psi}_{ii}(f) = 1$.

Assuming that the reverberant part is **diffuse**, i.e., its intensity is uniformly distributed over all possible directions.
$$
\Psi_{il}(f)={\sin(2\pi fd_{il}/c) \over 2\pi fd_{il}/c}, \sigma_{\rm rev}^{2}={4\beta ^{2} \over {\cal A}(1-\beta^{2})}
$$

- ${\cal A}$ is the total wall area

- $\beta$ the wall reflection coefficient computed from the room reverberation time $T_{60}$ via Eyring's formula:

- $$
  \beta=\exp\left\{-{13.82 \over \left({1 \over L_x}+{1 \over L_y}+{1 \over L_z}\right)cT_{60}}\right\}
  $$

#### Full-Rank Unconstrained Model

In practice, the assumption that the reverberant part is diffuse is rarely satisfied in realistically reverberant environments. Indeed, early echoes accounting for most of its energy are not uniformly distributed on the boundaries of the recording room. When performing some simulations in a rectangular room, we observed that
$$
\Psi_{il}(f)={\sin(2\pi fd_{il}/c) \over 2\pi fd_{il}/c}
$$
is valid on average when considering a large number of sources distributed at different positions in a room, but generally not valid for each individual source.

Therefore, we also investigate the modeling of each source via a full-rank unconstrained spatial covariance matrix Rj(f) whose coefficients are unrelated a priori. This model is the most general possible model for a covariance matrix. It generalizes the above three models in the sense that any matrix taking the form of
$$
\begin{aligned}
{\bf R}_j(f)&={\bf h}_j(f){\bf h}_j^H(f)\cr
{\bf R}_j(f)&={\bf a}_j(f){\bf a}_j^H(f)\cr
{\bf R}_j(f)&={\bf a}_j(f){\bf a}_j^H(f) +\sigma_{\rm rev}^2{\bf \Psi}(f)
\end{aligned}
$$
can also be considered as an unconstrained matrix. Because of this increased flexibility, this unconstrained model better fits the data as measured by the likelihood. In particular, it improves the poor fit between the model and the data observed for rank-1 models due to the fact that the narrowband approximation underlying these models does not hold for reverberant mixtures. In that sense, it circumvents the narrowband approximation to a certain extent.

The entries of ${\bf R}_j(f)$ are not directly interpretable in terms of simple geometrical quantities. The principal component of the matrix can be interpreted as a beamformer[^beamformer] pointing towards the direction of maximum output power, while the ratio between its largest eigenvalue and its trace is equal to the ratio between the output and input power of that beamformer. In moderate reverberation conditions, the former is expected to be close to the source direction of arrival (DOA) while the latter is related to the ratio between the power of direct sound and that of reverberation. However, the strength of this model is precisely that it remains valid to a certain extent in more reverberant environments, since it is the most general possible model for a covariance matrix.

[^beamformer]:B. D. van Veen and K. M. Buckley, "Beamforming: A versatile approach to spatial filtering", *IEEE ASSP Mag.*, vol. 5, no. 2, pp. 4-24, Apr. 1988.


## FastMNMF

### Full-Rank Spatial Model

#### Model Formulation

- $N$: Sources observed

- $M$: Microphones

- $X = \{ x_{ft}\}^{F,T}_{f,t=1}\in \mathbb{C}^{F\times T\times M}$:Observed multichannel *complex spectra*

Let ${\bf x}_{ftn}=[x_{ftn,1},\cdots,x_{ftn,M}]^T\in \mathbb{C}^M$: be the ***image*** of source $n$ assumed to be _circularly-symmetric complex Gaussian distributed_ as follows:
$$
{\bf x}_{ftn} \sim \mathcal{N}_\mathbb{C}({\bf 0}, \lambda_{ftn}{\bf G}_{nf})
$$
where $\lambda_{ftn}$ is the PSD of source $n$ at frequency $f$ and time $t$.

---

${\bf G}_{nf}$ is the $M\times M$ positive definite **full-rank** SCM of source $n$ at frequency $f$. 

According to the reproductive property of Gaussian distribution:
$$
{\bf x}_{ft} \sim \mathcal{N}_\mathbb{C} \left( {\bf 0}, \sum_{n=1}^N \lambda_{ftn}{\bf G}_{nf} \right)
$$

---

If ${\bf x}_{ft}, {\bf G}_{nf}, \lambda_{ftn}$ are all given, then the _posterior expectation_ of the source image ${\bf x}_{ftn}$ is obtained by **Multichannel Wiener Filtering (MWF)**:
$$
{\bf x}_{ftn} = \mathbb{E}[{\bf x}_{ftn} |{\bf x}_{ft}] = {\bf Y}_{ftn}{\bf Y}_{ft}^{-1}{\bf x}_{ft}
$$
where ${\bf Y}_{ftn}\stackrel{\text{def}}{=}\lambda_{ftn}{\bf G}_{nf}$ and ${\bf Y}_{ft}\stackrel{\text{def}}{=}\sum_{n=1}^N{\bf Y}_{ftn}$.

#### Parameter Estimation

**Goal**: Find the parameters ${\bf G}=\{{\bf G}_{nf}\}_{n,f=1}^{N,F}$ and ${\bf \Lambda}=\{\lambda_{ftn}\}_{f,t,n=1}^{F,T,N}$ that maximize the *log-likelihood* given by:
$$
\log p({\bf X}|{\bf G}, {\bf \Lambda}) \stackrel{\text c}{=} -\sum_{f,t=1}^{F,T}\left(\text{tr}\left({\bf X}_{ft}{\bf Y}_{ft}^{-1} \right) + \log{|{\bf Y}_{ft}|}\right)
$$
where ${\bf X}_{ft}\stackrel{\text{def}}{=}{\bf x}_{ft}{\bf x}_{ft}^H$ denotes the covariance matrix of a single ${\bf x}_{ft}$. 

---

***Closed-form update rule of ${\bf G}$***
$$
\begin{aligned}
{\bf A}_{nf}&\stackrel{\text{def}}{=} \sum_{t=1}^T\lambda_{ftn}{\bf Y}_{ft}^{-1}{\bf X}_{ft}{\bf Y}_{ft}^{-1},\cr
{\bf B}_{nf}&\stackrel{\text{def}}{=} \sum_{t=1}^T\lambda_{ftn}{\bf Y}_{ft}^{-1},\cr
{\bf G}_{nf}&\larr{\bf B}_{nf}^{-1}({\bf B}_{nf}{\bf G}_{nf}{\bf A}_{nf}{\bf G}_{nf})^{\frac{1}{2}}
\end{aligned}
$$

### Source Models

#### Unconstrained Source Model

*Unconstrained model*: **Uses ${\bf \Lambda}$ as free parameters.**

Using the **MM** algorithm and the **multiplicative update (MU)** rule of ${\bf \Lambda}$ is given by:
$$
\lambda_{ftn}\larr \lambda_{ftn}\sqrt{\frac{\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}{\bf X}_{ft}{\bf Y}_{ft}^{-1}\right)}{\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}\right)}}
$$

#### NMF-Based Source Model

If the PSDs $\{\lambda_{ftn}\}_{f,t=1}^{F,T}$ of source $n$ have **low-rank** structure (e.g., noise and music), the PSDs can be factorized as follows:
$$
\lambda_{ftn} = \sum_{k=1}^{K}\omega_{nkf}h_{nkt} \tag{for noise and music}
$$
Using **MM** algorithm and **MU** rules for $W$ and $H$ are given by:
$$
\omega_{nkf}\larr \omega_{nkf}\sqrt{\frac{\sum_{t=1}^{T}h_{nkt}\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}{\bf X}_{ft}{\bf Y}_{ft}^{-1}\right)}{{\sum_{t=1}^{T}h_{nkt}\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}\right)}}}\cr
h_{nkf}\larr h_{nkf}\sqrt{\frac{\sum_{f=1}^{F}\omega_{nkf}\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}{\bf X}_{ft}{\bf Y}_{ft}^{-1}\right)}{{\sum_{f=1}^{F}\omega_{nkf}\text{tr}\left({\bf G}_{ft}{\bf Y}_{ft}^{-1}\right)}}}
$$

#### DNN-Based Source Model

To represent the complicated characteristics of the PSDs $\{\lambda_{ftn}\}_{f,t=1}^{F,T}$ of a source $n$ (e.g., speech), a deep generative model can be used as follows
$$
\lambda _{ftn} = u_{nf}v_{nt}[{\bf \sigma}_{\bf \theta}^2({\bf z}_{nt})]_f \tag{for speech}
$$

- ${\bf \sigma}_{\bf \theta}^2(\cdot)$ is a nonlinear function (using *Neural Networks*) with parameters $\theta$ that maps a latent variable ${\bf z}_{nt} \in \mathbb{R}^D$ to a nonnegative spectrum ${\bf r}_{nt}\stackrel{\text{def}}{=}{\bf \sigma}_{\bf \theta}^2({\bf z}_{nt})\in\mathbb{R}_{+}^F$ at each time $t$. 
- $[\cdot]_f$ indicates the $f$-th element of a vector
- $u_{nf}\ge 0$ is a scaling factor at frequency $f$
- $v_{nt} \ge 0$ is an activation at time $t$

To update the latent variables ${\bf Z}_n=\{{\bf z}_{nt}\}_{t=1}^T$, we use Metropolis sampling[^MetropolisSampling]. A proposal ${\bf z}_{nt}^\text{new}\sim\mathcal{N}({\bf z}_{nt}^\text{old}, \epsilon I)$ is accepted with probability $\min (1, \gamma_{nt})$, where $\gamma_{nt}$ is given by:
$$
\log \gamma_{n t}=-\sum_{f=1}^{F}\left(\frac{1}{\lambda_{f t n}^{\text {new }}}-\frac{1}{\lambda_{f t n}^{\text {old }}}\right) \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1} \mathbf{X}_{f t} \mathbf{Y}_{f t}^{-1}\right) \cr -\sum_{f=1}^{F}\left(\lambda_{f t n}^{\text {new }}-\lambda_{f t n}^{\text {old }}\right) \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1}\right)
$$
where $\lambda_{ftn}^\text{new/old}=u_{nf}v_{nt}[\sigma_{\theta}^{2}(\mathrm{z}_{nt}^\text{new/old})]_{f}$. *In practice*, we update $\mathrm{Z}_n$ several times without updating $\mathrm{Y}_{ft}$ to reduce the computational cost of calculating the inverse.

In the same way as the NMF-based source model, the MU rules of $U$ and $V$ are given by
$$
u_{n f} \leftarrow u_{n f} \sqrt{\frac{\sum_{t=1}^{T} v_{n t} r_{n t f} \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1} \mathbf{X}_{f t} \mathbf{Y}_{f t}^{-1}\right)}{\sum_{t=1}^{T} v_{n t} r_{n t f} \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1}\right)}}
\cr
v_{n t} \leftarrow v_{n t} \sqrt{\frac{\sum_{f=1}^{F} u_{n f} r_{n t f} \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1} \mathbf{X}_{f t} \mathbf{Y}_{f t}^{-1}\right)}{\sum_{f=1}^{F} u_{n f} r_{n t f} \operatorname{tr}\left(\mathbf{G}_{n f} \mathbf{Y}_{f t}^{-1}\right)}}
$$

[^MetropolisSampling]: or *Metropolis–Hastings algorithm* [(wiki)](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), is a kind of _**M**arkov **C**hain **M**onte **C**arlo_ method.

### Integration of Spatial and Source Models

#### Full-Rank Spatial Covariance Analysis (FCA)

FCA [^FCA] is obtained by integrating the full-rank spatial model and the unconstrained source model. While the EM algorithm was originally used in FCA, in this paper we use the MM algorithm expected to converge faster as proposed in [^MM-FCA].

[^FCA]: N. Q. K. Duong et al., "Under-determined reverberant audio source separation using a full-rank spatial covariance model", *IEEE TASLP*, vol. 18, no. 7, pp. 1830-1840, 2010.
[^MM-FCA]: N. Ito and T. Nakatani, "Multiplicative updates and joint diagonalization based acceleration for under-determined BSS using a full-rank spatial covariance model", *GlobalSIP*, pp. 231-235, 2018.

#### Multichannel NMF (MNMF)

MNMF[^MNMF] is obtained by integrating the NMF-based source model into FCA.

[^MNMF]: H. Sawada et al., "Multichannel extensions of non-negative matrix factorization with complex-valued data", *IEEE TASLP*, vol. 21, no. 5, pp. 971-982, 2013.

#### MNMF with a Deep Prior (MNMF-DP)

MNMF-DP[^MNMF-DP] specialized for speech enhancement is obtained by integrating the full-rank spatial model and the DNN and NMF-based source models representing speech and noise sources, respectively. Assuming a source indexed by $n = 1$ corresponds to the speech, $\lambda_{ft,1}$ and $\lambda_{ft,n},(n\ge2)$ are given by $\text{(for speech)}$ and $\text{(for noise and music)}$, respectively.

[^MNMF-DP]: K. Sekiguchi et al., "Bayesian multichannel speech enhancement with a deep speech prior", *APSIPA*, pp. 1233-1239, 2018.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzYwODgzMjcxXX0=
-->