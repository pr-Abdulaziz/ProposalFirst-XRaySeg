# Proposal-first PIDray-Occlusion-aware Bilayer SAM

This is the concise equation guide for the proposed architecture.

[Open the Draw.io architecture](../diagrams/proposal-first-pidray-occ-hybrid.drawio)

## 1. Provenance

| Label | Meaning |
|---|---|
| **PIDray** | Taken from the PIDray paper or released code |
| **Occ** | Taken from the Occlusion-aware paper or code |
| **Proposed** | New connection or rule introduced by this hybrid |

## 2. Symbols

| Symbol | Meaning |
|---|---|
| $i$ | image index |
| $j$ | proposal or object index |
| $k\in\{1,2,3\}$ | Cascade stage |
| $\ell\in\{1,\ldots,5\}$ | AttFPN level |
| $x_i$ | input X-ray image |
| $F_i^{1:5}$ | PIDray AttFPN features |
| $\phi_i$ | image-level prohibited confidence |
| $\hat c_{ij},\hat b_{ij}$ | predicted class and box |
| $s_{ij}^{det}$ | detection confidence |
| $f_i^{img}$ | SAM image embedding |
| $f_{ij}^{sparse}$ | learned sparse prompt |
| $\hat m_{ij}^{e}$ | overlap or occluder mask |
| $\hat m_{ij}$ | final target mask |

The two feature systems must not be confused:

$$
\boxed{F_i^{1:5}\neq f_i^{img}}
$$

## 3. Training annotations

For image $x_i$:

$$
Y_i=
\left\{
(b_{ij},c_{ij},m_{ij},m_{ij}^{e})
\right\}_{j=1}^{n_i},
\qquad
y_i^0=\mathbf 1[n_i>0]
$$

- $m_{ij}$ is the target-object mask.
- $m_{ij}^{e}$ is the overlapping part of another annotated prohibited object.
- $y_i^0$ is the binary label used by PIDray $R_0$.

## 4. PIDray backbone and AttFPN

$$
\{C_i^2,C_i^3,C_i^4,C_i^5\}
=
\Phi_{\mathrm{ResNet101}}(x_i)
$$

$$
\boxed{
\{F_i^1,\ldots,F_i^5\}
=
\Phi_{\mathrm{AttFPN}}
(C_i^2,C_i^3,C_i^4,C_i^5)
}
$$

Each $F_i^\ell$ has 256 channels. A compact description of AttFPN is:

$$
\begin{aligned}
G_{i,\ell r}
&=
\operatorname{Resize}(\widetilde F_i^r,H_\ell,W_\ell),\\
C_i^\ell
&=
\sum_{r=1}^{5}A_{i,\ell r}^{ch}\odot G_{i,\ell r},\\
S_i^\ell
&=
\sum_{r=1}^{5}A_{i,\ell r}^{sp}\odot G_{i,\ell r},\\
F_i^\ell
&=
\widetilde F_i^\ell+
\operatorname{DR}_\ell(C_i^\ell+S_i^\ell).
\end{aligned}
$$

This compact equation summarizes the implementation. PIDray describes the operations separately as channel attention, spatial attention, and dependency refinement.

## 5. PIDray coarse node $R_0$

### Paper formulation

$$
F_i^0
=
\operatorname{Agg}(F_i^1,\ldots,F_i^5),
\qquad
E_{ip}=F_{ip}^0+PE_p
$$

For attention head $r$:

$$
\begin{aligned}
\alpha_{ip}^{(r)}
&=
\operatorname{softmax}_{p}
\left(
\frac{
\langle W_Q^{(r)}Q,W_K^{(r)}E_{ip}\rangle
}{\sqrt d}
\right),\\
h_i^{(r)}
&=
\sum_{p=1}^{HW}
\alpha_{ip}^{(r)}W_V^{(r)}F_{ip}^{0},\\
V_i
&=
\operatorname{FFN}
\left[
W_H\operatorname{Cat}
(h_i^{(1)},\ldots,h_i^{(N_h)})
\right].
\end{aligned}
$$

The final probability and decision are:

$$
\boxed{
\phi_i=\sigma(w_\phi^\top V_i+b_\phi)
},
\qquad
g_i^0=\mathbf 1[\phi_i>0.5]
$$

### Released-code difference

The released PIDray branch uses:

$$
\boxed{F_i^0\leftarrow F_i^5}
$$

Its inference path runs RPN and RoI processing before suppressing detections when $\phi_i\le0.5$. Therefore, released $R_0$ is not a detector compute-saving early exit.

## 6. RPN and three-stage Cascade

The RPN produces:

$$
\boxed{
\hat{\mathcal O}_i
=
\Phi_{\mathrm{rpn}}(F_i^{1:5})
=
\{\hat o_{ij}\}_{j=1}^{M_i}
}
$$

Initialize $\hat b_{ij}^{(0)}=\hat o_{ij}$. For Cascade stage $k$:

$$
\begin{aligned}
u_{ij}^{(k)}
&=
\Phi_{\mathrm{roi}}^{(k)}
(F_i^{1:4},\hat b_{ij}^{(k-1)}),\\
z_{ij}^{(k)}
&=
\Phi_{\mathrm{cls}}^{(k)}(u_{ij}^{(k)}),\\
\Delta b_{ij}^{(k)}
&=
\Phi_{\mathrm{reg}}^{(k)}(u_{ij}^{(k)}),\\
\hat b_{ij}^{(k)}
&=
\operatorname{Decode}
(\hat b_{ij}^{(k-1)},\Delta b_{ij}^{(k)}).
\end{aligned}
$$

The stages use IoU thresholds $0.5$, $0.6$, and $0.7$. The final detector returns:

$$
\boxed{
(\hat c_{ij},s_{ij}^{det},\hat b_{ij}),
\qquad
\hat b_{ij}=\hat b_{ij}^{(3)}
}
$$

## 7. Proposed proposal screening

This gate is new:

$$
q_{ij}=\phi_i s_{ij}^{det}
$$

$$
\boxed{
g_{ij}
=
\mathbf 1[\phi_i>\tau_0]
\mathbf 1[s_{ij}^{det}>\tau_d],
\qquad
\mathcal K_i=\{j:g_{ij}=1\}
}
$$

- $\phi_i$ provides global image evidence.
- $s_{ij}^{det}$ provides local proposal evidence.
- $q_{ij}$ ranks retained proposals.
- If $\mathcal K_i=\varnothing$, inference skips the SAM/Occ branch.

## 8. Detector-to-SAM bridge

For each retained proposal $j\in\mathcal K_i$:

$$
\boxed{
\hat v_{ij}
=
\Phi_{\mathrm{roi}}^{mask}
(F_i^{1:4},\hat b_{ij})
}
$$

$$
\boxed{
f_{ij}^{sparse}
=
\Phi_{\mathrm{prompt}}(\hat v_{ij}),
\qquad
f_{ij}^{sparse}\in\mathbb R^{5\times256}
}
$$

The local RoI feature is $256\times14\times14$.

## 9. SAM encoder

The image embedding is computed once and reused:

$$
\boxed{
f_i^{img}
=
\Phi_{\mathrm{enc}}(\mathcal P_S(x_i))
}
$$

For the local ViT-L model with a $512\times512$ input:

$$
f_i^{img}\in\mathbb R^{256\times32\times32}
$$

The original Occ pipeline uses $f_i^{img}$ for RPN and RoI processing. The hybrid instead uses PIDray $F_i^{1:5}$ for detection and keeps $f_i^{img}$ for mask decoding.

## 10. Occlusion-aware bilayer decoding

### Occluder decoder

$$
T_{ij}^{r}
=
\operatorname{Cat}
(t_{\mathrm{mask}},t_{\mathrm{IoU}},f_{ij}^{sparse})
$$

$$
\boxed{
(\hat m_{ij}^{e},
f_{r,ij}^{img},
f_{r,ij}^{sparse})
=
\Phi_r(T_{ij}^{r},f_i^{img})
}
$$

### Residual guidance

$$
\boxed{
f_{e,ij}^{img}
=
f_i^{img}+f_{r,ij}^{img},
\qquad
f_{e,ij}^{sparse}
=
f_{ij}^{sparse}+f_{r,ij}^{sparse}
}
$$

The overlap mask is an auxiliary output, not a direct input to $\Phi_e$:

$$
\boxed{\hat m_{ij}^{e}\not\rightarrow\Phi_e}
$$

### Occludee decoder

$$
T_{ij}^{e}
=
\operatorname{Cat}
(t_{\mathrm{mask}},t_{\mathrm{IoU}},f_{e,ij}^{sparse})
$$

$$
\boxed{
\hat m_{ij}
=
\Phi_e(T_{ij}^{e},f_{e,ij}^{img})
}
$$

### Token-order note

The Occ paper writes:

$$
\operatorname{Cat}
(t_{\mathrm{mask}},t_{\mathrm{IoU}},f^{sparse})
$$

The local decoder packs:

$$
[t_{\mathrm{IoU}},t_{\mathrm{mask}}^{1:4},f_{ij}^{sparse}]
$$

This gives $1+4+5=10$ tokens. The paper and code describe the same token roles but list them differently.

## 11. Final output

Restore the target mask to the original image:

$$
\hat m_{ij}^{orig}
=
T_i^{-1}
\left[
\operatorname{Upsample}(\hat m_{ij})
\right]
$$

$$
\boxed{
\hat{\mathcal Y}_i
=
\left\{
(\hat c_{ij},
\hat b_{ij},
\hat m_{ij}^{orig},
\hat m_{ij}^{e},
q_{ij})
:
j\in\mathcal K_i
\right\}
}
$$

The primary outputs are class, box, and target mask. The overlap mask is mainly used for training and diagnosis.

## 12. Training loss

$$
\mathcal L_{R_0}
=
\operatorname{BCE}(\phi_i,y_i^0)
$$

$$
\mathcal L_{\mathrm{det}}
=
\mathcal L_{\mathrm{rpn}}
+
\sum_{k=1}^{3}
w_k
\left[
\mathcal L_{\mathrm{cls}}^{(k)}
+
\mathbf 1_{ij}^{+}
\mathcal L_{\mathrm{reg}}^{(k)}
\right]
$$

where $(w_1,w_2,w_3)=(1,0.5,0.25)$.

$$
\mathcal L_{\mathrm{bilayer}}
=
\frac1{N_+}
\sum_{(i,j)\in\mathcal P^+}
\left[
\mathcal L_{\mathrm{pred}}(\hat m_{ij},m_{ij})
+
\lambda_e
\mathcal L_{\mathrm{pred}}
(\hat m_{ij}^{e},m_{ij}^{e})
\right]
$$

The local Occ code uses $\lambda_e=0.25$. A starting hybrid objective is:

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
0.1\mathcal L_{R_0}
+
\mathcal L_{\mathrm{det}}
+
\mathcal L_{\mathrm{bilayer}}
}
$$

This combined loss is proposed; it is not printed by either paper.

## 13. Training versus inference

Training uses matched positive RoIs:

$$
g_{ij}^{train}
=
\mathbf 1[(i,j)\text{ is a matched positive RoI}]
$$

Do not use the predicted hard $R_0$ gate to remove genuine positive mask examples.

Inference uses:

$$
g_{ij}^{test}
=
\mathbf 1[\phi_i>\tau_0]
\mathbf 1[s_{ij}^{det}>\tau_d]
$$

## 14. Component ownership

| Component | Source |
|---|---|
| ResNet-101, AttFPN, $R_0$, RPN, Cascade | PIDray |
| SAM encoder, prompt head, $\Phi_r$, residual guidance, $\Phi_e$ | Occ/RSPrompter |
| Detector-to-SAM bridge and proposal screening | proposed hybrid |
| Skipping SAM when no proposal survives | proposed hybrid |
| Combined total loss | proposed hybrid |

The original PIDray FCN mask head is excluded because $\Phi_e$ becomes the final mask head.

Training $\hat m_{ij}^{e}$ requires PIDray-A-style overlap annotations. Standard PIDray masks alone cannot supervise this auxiliary loss.

## 15. Complete flow

$$
\boxed{
x_i
\xrightarrow{\mathrm{ResNet101+AttFPN}}
F_i^{1:5}
\xrightarrow{R_0+\mathrm{RPN+Cascade}}
(\phi_i,\hat c_{ij},s_{ij}^{det},\hat b_{ij},\hat v_{ij})
\xrightarrow{\mathrm{screening}+\Phi_{\mathrm{prompt}}}
f_{ij}^{sparse}
}
$$

$$
\boxed{
x_i
\xrightarrow{\Phi_{\mathrm{enc}}}
f_i^{img},
\qquad
(f_i^{img},f_{ij}^{sparse})
\xrightarrow{\Phi_r\to\mathrm{residual}\to\Phi_e}
(\hat m_{ij}^{e},\hat m_{ij})
}
$$

The PIDray branch determines **where and what**. The bilayer SAM branch predicts the target mask using overlap-aware guidance.

## References

- [PIDray paper](../../main_ref/PIDray_paper.pdf)
- [Occlusion-aware paper](../../main_ref/Occlusion-aware.pdf)
- [Original PIDray configuration](../../references/legacy_pidray/configs/cascade_mask_rcnn_r101_with_R0R1.py)
- [Original PIDray $R_0$ implementation](../../references/legacy_pidray/mmdet/models/detectors/two_stage.py)
- [Original AttFPN implementation](../../references/legacy_pidray/mmdet/models/necks/attfpn.py)
- [Occ configuration](../../configs/occlusion/pid.py)
- [Occ bilayer decoder](../../mmdet/occlu/occlu.py)
