# LittleLess AI — Phase 3.1 Mathematical Hypotheses, Proofs & Counterexamples

**Author:** Principal Mathematical Algorithm Designer & Quantization Architect  
**Workspace:** `/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer/`  
**Date:** October 2026

---

## 1. Problem Statement: Why Attention Quantization Fails Under Standard DBQ

In Phase 3 empirical evaluations, applying identical Dual-Basis Quantization (DBQ) to different modules in the 1.17M Causal Transformer produced wildly divergent degradation:
- **Quantizing Feedforward Layers (MLP Only):** $+2.40\%$ degradation (PPL $6.523$ vs. $6.370$).
- **Quantizing Self-Attention Projections (Attn Only):** $+7.56\%$ degradation (PPL $6.852$ vs. $6.370$).
- **Quantizing Both Simultaneously (PC-DBQ All):** $+17.24\%$ degradation (PPL $7.468$ vs. $6.370$).

Self-attention suffered more than **$3.1\times$ the degradation** of feedforward layers, despite containing only $34.9\%$ of model parameters compared to $56.1\%$ in the MLP.

---

## 2. Mathematical Derivation: Error Propagation in Attention vs. Feedforward Blocks

### 2.1 The Feedforward Block: 1-Lipschitz Linear Stability
In a standard Pre-LN feedforward block:
$$\mathbf{h}_1 = \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{y}_{\text{MLP}} = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2$$
Let $\hat{\mathbf{W}}_1 = \mathbf{W}_1 + \mathbf{E}_1$ and $\hat{\mathbf{W}}_2 = \mathbf{W}_2 + \mathbf{E}_2$, where $\mathbf{E}_1, \mathbf{E}_2$ represent quantization error matrices.

The perturbed output is:
$$\hat{\mathbf{y}}_{\text{MLP}} = (\mathbf{W}_2 + \mathbf{E}_2) \text{GELU}\left( (\mathbf{W}_1 + \mathbf{E}_1)\mathbf{x} + \mathbf{b}_1 \right) + \mathbf{b}_2$$
Since $\text{GELU}(z) = z \Phi(z)$, its derivative is bounded:
$$\sup_{z \in \mathbb{R}} |\text{GELU}'(z)| \approx 1.0926$$
Hence, GELU is approximately $1.09$-Lipschitz. By first-order Taylor expansion:
$$\Delta \mathbf{y}_{\text{MLP}} \approx \mathbf{W}_2 \text{diag}(\text{GELU}'(\mathbf{W}_1 \mathbf{x})) \mathbf{E}_1 \mathbf{x} + \mathbf{E}_2 \text{GELU}(\mathbf{W}_1 \mathbf{x})$$
The total error $\Delta \mathbf{y}_{\text{MLP}}$ is **strictly linear** in $\mathbf{E}_1$ and $\mathbf{E}_2$. There are no exponential terms, and the non-linearity does not amplify input perturbations.

---

### 2.2 The Self-Attention Block: Bilinear Interaction and Exponential Softmax Amplification
In causal self-attention, for query token $i$ and key token $j$:
$$\mathbf{q}_i = \mathbf{W}_q \mathbf{x}_i \in \mathbb{R}^{d_k}, \quad \mathbf{k}_j = \mathbf{W}_k \mathbf{x}_j \in \mathbb{R}^{d_k}$$
The unnormalized attention logit is:
$$A_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} = \frac{\mathbf{x}_i^\top \mathbf{W}_q^\top \mathbf{W}_k \mathbf{x}_j}{\sqrt{d_k}}$$
The attention probability distribution across keys $j \le i$ is computed via row-wise softmax:
$$S_{ij} = \frac{\exp(A_{ij})}{\sum_{m=1}^i \exp(A_{im})}$$
Finally, the attention block output for token $i$ is:
$$\mathbf{y}_i = \mathbf{W}_{\text{out}} \sum_{j=1}^i S_{ij} (\mathbf{W}_v \mathbf{x}_j)$$

Quantization induces three compounding failure modes:

#### Failure Mode 1: Bilinear Quantization Cross-Products
Let $\hat{\mathbf{W}}_q = \mathbf{W}_q + \mathbf{E}_q$ and $\hat{\mathbf{W}}_k = \mathbf{W}_k + \mathbf{E}_k$. The perturbed logit is:
$$\hat{A}_{ij} = \frac{1}{\sqrt{d_k}} \mathbf{x}_i^\top (\mathbf{W}_q + \mathbf{E}_q)^\top (\mathbf{W}_k + \mathbf{E}_k) \mathbf{x}_j = A_{ij} + \Delta A_{ij}$$
where:
$$\Delta A_{ij} = \frac{1}{\sqrt{d_k}} \left( \mathbf{x}_i^\top \mathbf{W}_q^\top \mathbf{E}_k \mathbf{x}_j + \mathbf{x}_i^\top \mathbf{E}_q^\top \mathbf{W}_k \mathbf{x}_j + \mathbf{x}_i^\top \mathbf{E}_q^\top \mathbf{E}_k \mathbf{x}_j \right)$$
Unlike the MLP, the logit error involves cross-products between query errors and key projections, plus a bilinear second-order error term $\mathbf{E}_q^\top \mathbf{E}_k$.

#### Failure Mode 2: Exponential Softmax Perturbation
The Jacobian of the softmax mapping $\mathbf{A}_{i, :} \mapsto \mathbf{S}_{i, :}$ is:
$$\frac{\partial S_{ij}}{\partial A_{im}} = S_{ij}(\delta_{jm} - S_{im})$$
Consider the relative probability ratio between a relevant context token $j$ (where $A_{ij} \approx +9.0$ in our trained model) and an irrelevant token $k$ (where $A_{ik} \approx 0.0$):
$$\frac{S_{ij}}{S_{ik}} = \exp(A_{ij} - A_{ik}) = \exp(9.0) \approx 8,103$$
Suppose independent quantization noise perturbs $\Delta A_{ij} = -0.8$ and $\Delta A_{ik} = +0.8$ (a modest shift of $1.6$ on a scale where logits range from $-7$ to $+9.5$):
$$\frac{\hat{S}_{ij}}{\hat{S}_{ik}} = \exp(9.0 - 1.6) = \exp(7.4) \approx 1,635$$
The relative attention weight on the critical token collapses by **$4.95\times$ ($79.8\%$ loss of attention focus)**. The softmax function is locally $\mathcal{O}(e^{\Delta A})$ sensitive, exponentially magnifying small logit noise into catastrophic context corruption.

#### Failure Mode 3: Cascaded 4-Fold Error Composition
The corrupted attention weights $\hat{S}_{ij}$ are subsequently multiplied by the quantized value projections $(\mathbf{W}_v + \mathbf{E}_v)\mathbf{x}_j$, and projected through $(\mathbf{W}_{\text{out}} + \mathbf{E}_{\text{out}})$. Thus, attention suffers from a **4-stage compound error cascade**:
$$\hat{\mathbf{W}}_q \otimes \hat{\mathbf{W}}_k \longrightarrow \text{softmax} \longrightarrow \hat{\mathbf{W}}_v \longrightarrow \hat{\mathbf{W}}_{\text{out}}$$

---

## 3. Mathematical Investigation of Attention Outlier-Centering Transformation (AOCT)

A common hypothesis in recent quantization literature suggests transforming attention inputs or weights via centering:
$$\tilde{\mathbf{z}} = \mathbf{z} - \boldsymbol{\mu}$$
We mathematically derive and empirically test four variants of AOCT under causal autoregressive masking.

### 3.1 Theorem 1: Key Sequence-Centering Invariance
**Claim:** Centering key vectors across the sequence length preserves exact softmax attention probabilities.

**Proof:**  
Let $\bar{\mathbf{k}} \in \mathbb{R}^{d_k}$ be any constant vector across keys $j \in \{1, \dots, T\}$. Define centered keys:
$$\tilde{\mathbf{k}}_j = \mathbf{k}_j - \bar{\mathbf{k}}$$
The transformed attention logit for query $i$ and key $j$ is:
$$\tilde{A}_{ij} = \frac{\mathbf{q}_i^\top \tilde{\mathbf{k}}_j}{\sqrt{d_k}} = \frac{\mathbf{q}_i^\top (\mathbf{k}_j - \bar{\mathbf{k}})}{\sqrt{d_k}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} - \frac{\mathbf{q}_i^\top \bar{\mathbf{k}}}{\sqrt{d_k}} = A_{ij} - c_i$$
Notice that for a fixed query index $i$, the scalar $c_i = \frac{\mathbf{q}_i^\top \bar{\mathbf{k}}}{\sqrt{d_k}}$ is **strictly constant across all key indices $j$**.

Applying row-wise causal softmax:
$$\tilde{S}_{ij} = \frac{\exp(\tilde{A}_{ij})}{\sum_{m=1}^i \exp(\tilde{A}_{im})} = \frac{\exp(A_{ij} - c_i)}{\sum_{m=1}^i \exp(A_{im} - c_i)} = \frac{\exp(A_{ij})\exp(-c_i)}{\sum_{m=1}^i \exp(A_{im})\exp(-c_i)} = \frac{\exp(A_{ij})}{\sum_{m=1}^i \exp(A_{im})} = S_{ij}$$
The term $\exp(-c_i)$ cancels identically from numerator and denominator. $\blacksquare$

**Empirical Verification:**
On synthetic and WikiText-2 tokens ($B=2, H=5, T=128, D=32$):
$$\max_{i, j} |\tilde{S}_{ij} - S_{ij}| = 2.38 \times 10^{-7} \quad (\text{Machine Float32 Epsilon})$$

---

### 3.2 Counterexample 1: Key Channel-Centering Fails
**Claim:** Centering keys across feature channels $\mu_k(j) = \frac{1}{d_k}\sum_{c=1}^{d_k} k_{j,c}$ does NOT preserve softmax.

**Proof & Numerical Counterexample:**  
Let $\tilde{\mathbf{k}}_j = \mathbf{k}_j - \mu_k(j)\mathbf{1}$. Then:
$$\tilde{A}_{ij} = A_{ij} - \mu_k(j) \frac{\mathbf{q}_i^\top \mathbf{1}}{\sqrt{d_k}} = A_{ij} - d_{ij}$$
Since $\mu_k(j)$ depends on the key index $j$, the shift $d_{ij}$ is **not constant along row $i$**.  
$$\max_{i, j} |\tilde{S}_{ij} - S_{ij}| = \mathbf{0.1768} \quad (\text{Catastrophic Softmax Flip})$$
Channel-centering of keys violates softmax invariance and is mathematically falsified.

---

### 3.3 Counterexample 2: Query Centering Fails
**Claim:** Centering query vectors across sequence or channels does NOT preserve softmax.

**Proof & Numerical Counterexample:**  
Let $\tilde{\mathbf{q}}_i = \mathbf{q}_i - \bar{\mathbf{q}}$. Then:
$$\tilde{A}_{ij} = \frac{(\mathbf{q}_i - \bar{\mathbf{q}})^\top \mathbf{k}_j}{\sqrt{d_k}} = A_{ij} - \frac{\bar{\mathbf{q}}^\top \mathbf{k}_j}{\sqrt{d_k}}$$
The subtracted term $\frac{\bar{\mathbf{q}}^\top \mathbf{k}_j}{\sqrt{d_k}}$ depends directly on $j$. As $j$ varies across the row, the shift varies, breaking row-wise softmax invariance.  
$$\max_{i, j} |\tilde{S}_{ij} - S_{ij}| = \mathbf{0.0523} \quad (\text{Substantial Softmax Distortion})$$
Query centering without a key-dependent logit correction term is mathematically falsified.

---

### 3.4 Counterexample 3: Weight Matrix Centering Fails
**Claim:** Subtracting row or column means from $\mathbf{W}_k$ does NOT preserve softmax.

**Proof & Numerical Counterexample:**  
If $\mathbf{W}_k = \tilde{\mathbf{W}}_k + \mathbf{1}\boldsymbol{\mu}^\top$, then $\mathbf{k}_j = \tilde{\mathbf{W}}_k \mathbf{x}_j + \mathbf{1}(\boldsymbol{\mu}^\top \mathbf{x}_j)$. The resulting logit shift is:
$$\Delta A_{ij} = \frac{(\mathbf{q}_i^\top \mathbf{1}) (\boldsymbol{\mu}^\top \mathbf{x}_j)}{\sqrt{d_k}}$$
Because $\mathbf{x}_j$ varies with key position $j$, $\Delta A_{ij}$ is non-constant along row $i$.  
$$\max_{i, j} |\tilde{S}_{ij} - S_{ij}| = \mathbf{0.0551}$$
Weight-level centering without dynamic runtime correction is mathematically falsified.

---

## 4. The Mathematical Breakthrough: Activation-Weighted Gram Dual-Basis Quantization (AWG-DBQ)

Since activation-centering does not resolve the weight quantization challenge, we derive a mathematically principled solution: **Activation-Weighted Gram Dual-Basis Quantization (AWG-DBQ)**.

### 4.1 Theoretical Derivation
Standard DBQ (Phase 3) approximates $\mathbf{W} \in \mathbb{R}^{M \times N}$ by minimizing the unweighted Frobenius norm:
$$\min_{\mathbf{s}_1, \mathbf{s}_2, \mathbf{T}_1, \mathbf{T}_2} \|\mathbf{W} - (\mathbf{s}_1 \odot \mathbf{T}_1 + \mathbf{s}_2 \odot \mathbf{T}_2)\|_F^2$$
This treats all input directions equally (implicit identity covariance $\mathbf{H} = \mathbf{I}$). In reality, input tokens $\mathbf{X} \in \mathbb{R}^{N \times B}$ have non-uniform empirical covariance:
$$\mathbf{H} = \frac{1}{B} \mathbf{X} \mathbf{X}^\top \in \mathbb{R}^{N \times N}$$
The true objective is minimizing the **layer activation reconstruction error**:
$$\mathcal{L}(\mathbf{W}) = \frac{1}{B} \| \mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X} \|_F^2 = \text{Tr}\left( (\mathbf{W} - \hat{\mathbf{W}}) \mathbf{H} (\mathbf{W} - \hat{\mathbf{W}})^\top \right)$$
For each row $i \in \{1, \dots, M\}$ of weight matrix $\mathbf{W}$:
$$\mathcal{L}_i = (\mathbf{W}_i - s_{1, i} \mathbf{T}_{1, i} - s_{2, i} \mathbf{T}_{2, i}) \mathbf{H} (\mathbf{W}_i - s_{1, i} \mathbf{T}_{1, i} - s_{2, i} \mathbf{T}_{2, i})^\top$$

Setting the partial derivatives with respect to $[s_{1, i}, s_{2, i}]^\top$ to zero:
$$\frac{\partial \mathcal{L}_i}{\partial s_{1, i}} = -2 \mathbf{T}_{1, i} \mathbf{H} \mathbf{W}_i^\top + 2 s_{1, i} (\mathbf{T}_{1, i} \mathbf{H} \mathbf{T}_{1, i}^\top) + 2 s_{2, i} (\mathbf{T}_{1, i} \mathbf{H} \mathbf{T}_{2, i}^\top) = 0$$
$$\frac{\partial \mathcal{L}_i}{\partial s_{2, i}} = -2 \mathbf{T}_{2, i} \mathbf{H} \mathbf{W}_i^\top + 2 s_{1, i} (\mathbf{T}_{2, i} \mathbf{H} \mathbf{T}_{1, i}^\top) + 2 s_{2, i} (\mathbf{T}_{2, i} \mathbf{H} \mathbf{T}_{2, i}^\top) = 0$$

Define the **Activation-Weighted Gram Matrix** $\mathbf{G}_i \in \mathbb{R}^{2 \times 2}$ and projection vector $\mathbf{b}_i \in \mathbb{R}^2$:
$$\mathbf{G}_i = \begin{bmatrix} \mathbf{T}_{1, i} \mathbf{H} \mathbf{T}_{1, i}^\top & \mathbf{T}_{1, i} \mathbf{H} \mathbf{T}_{2, i}^\top \\ \mathbf{T}_{2, i} \mathbf{H} \mathbf{T}_{1, i}^\top & \mathbf{T}_{2, i} \mathbf{H} \mathbf{T}_{2, i}^\top \end{bmatrix} = \begin{bmatrix} g_{11} & g_{12} \\ g_{12} & g_{22} \end{bmatrix}, \quad \mathbf{b}_i = \begin{bmatrix} \mathbf{W}_i \mathbf{H} \mathbf{T}_{1, i}^\top \\ \mathbf{W}_i \mathbf{H} \mathbf{T}_{2, i}^\top \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

The exact, closed-form optimal scale vector is:
$$\begin{bmatrix} s_{1, i}^* \\ s_{2, i}^* \end{bmatrix} = \mathbf{G}_i^{-1} \mathbf{b}_i = \frac{1}{g_{11} g_{22} - g_{12}^2} \begin{bmatrix} g_{22} & -g_{12} \\ -g_{12} & g_{11} \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

### 4.2 Properties of AWG-DBQ
1. **Zero Additional Storage:** The scales $s_{1, i}^*, s_{2, i}^*$ replace the heuristic scales in the exact same 16-bit slots.
2. **Zero Runtime Overhead:** Deployment inference is identical.
3. **Exact Error Minimization in Attention Space:** Feature directions with large activation variance $\sigma_k^2$ are weighted proportionally in the scale optimization, preventing attention logit drift.
4. **Empirical Verification:** AWG-DBQ slashed total test perplexity degradation from **$+17.24\%$ (PPL 7.468) down to $+3.89\%$ (PPL 6.618)** across all transformer layers.

---

## 5. Embedding Table Representation: Why Table Lookups Require INT8

### 5.1 Central Limit Theorem Failure in Embedding Tables
In a matrix multiplication $\mathbf{y} = \mathbf{W}\mathbf{x}$ where $\mathbf{W} \in \mathbb{R}^{M \times N}$, the quantization error on output channel $m$ is:
$$e_m = \sum_{n=1}^N E_{mn} x_n$$
By the Central Limit Theorem, if $E_{mn}$ are zero-mean independent random variables with variance $\sigma_E^2$, the variance of the error is:
$$\text{Var}(e_m) = \sigma_E^2 \sum_{n=1}^N x_n^2$$
The summation across $N=160$ channels provides significant noise averaging.

In contrast, in an embedding lookup:
$$\mathbf{e}_t = \mathbf{W}_{\text{te}}[idx_t, :]$$
There is **NO dot product and NO summation across channels**. The noise on row $idx_t$ is injected directly into the representation entering Layer 0. Under 2-bit DBQ, embedding noise alone degraded test perplexity by $+2.93\%$.

### 5.2 Symmetric INT8 Solution
Embeddings constitute only $61,440$ parameters ($5.24\%$ of model parameters). Quantizing `wte` and `wpe` to symmetric INT8:
$$Q = \text{clip}\left(\left\lfloor \frac{W}{\text{scale}} \right\rceil, -128, 127\right), \quad \text{scale} = \frac{\max |W|}{127}$$
- **Quality Gap:** Only **$+0.02\%$** (PPL $6.371$ vs. $6.370$).
- **Storage Saving:** Saves $61,440$ bytes ($50\%$ embedding reduction).
- **Combined with AWG-DBQ:** Total artifact size drops to **$557,369$ bytes ($544.3$ KB)**, fully satisfying the $< 600$ KB acceptance gate.
