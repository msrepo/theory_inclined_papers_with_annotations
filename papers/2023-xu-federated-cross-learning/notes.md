---
title: "Federated Cross Learning for Medical Image Segmentation"
category: "Medical image analysis"
subcategory: "Federated learning"
short_title: "Xu 2023 — FedCross"
authors: "Xuanang Xu, Hannah H. Deng, Tianyi Chen, Tianshu Kuang, Joshua C. Barber, Daeseung Kim, Jaime Gateno, James J. Xia, Pingkun Yan"
venue: "MIDL (PMLR 227)"
year: 2023
url: "https://proceedings.mlr.press/v227/xu24a.html"
pdf_url: "https://proceedings.mlr.press/v227/xu24a/xu24a.pdf"
tags: [federated-learning, non-iid, fedavg, ensemble, uncertainty, prostate-mri, catastrophic-forgetting]
status: read
summary: "passing one model from hospital to hospital instead of averaging weights; the algebra only unrolls the FedAvg update, the gains are fractions of a Dice point, and the ensemble variant is compared with single models"
ext_links: [code|https://github.com/DIAL-RPI/FedCross]
---

## Links

- **[PMLR 227:1441–1452](https://proceedings.mlr.press/v227/xu24a.html)**: the paper (MIDL 2023, full paper track).
- **Code**: [DIAL-RPI/FedCross](https://github.com/DIAL-RPI/FedCross).
- **Baselines**: FedAvg (McMahan et al. 2017), FedProx (Li et al. 2020), FedBN (Li et al. 2021).
- **Related in this repo**: [IID representation learning](../2022-wu-iid-representation/index.html), which asks what it would take
  to make domains' distributions identical.

Xu et al., *Federated Cross Learning for Medical Image Segmentation*, MIDL 2023 (proceedings volume dated 2024).

## In one paragraph

Hospitals cannot pool patient scans, so federated learning trains a shared model by sending it to each hospital,
training locally and combining the results. The standard combination, FedAvg, averages the hospitals' weights
and is known to suffer when each hospital's data look different (non-iid). The authors give a short algebraic
argument that the averaging step is the weak link, and propose FedCross: skip averaging and pass a *single* model
from hospital to hospital in turn. Training several such models in parallel and averaging their *predictions*
(FedCrossEns) recovers centralised accuracy and gives a per-pixel disagreement map as an uncertainty estimate.

## The picture, intuition first

**FedAvg.** Picture $K$ hikers starting from the same point on a foggy mountain range, each told to walk downhill
using only their own local map. After a while the server takes the *average position* of all hikers. If the
terrain were a single bowl, the average would be a fine place to stand. On a rugged range, each hiker may have
descended into a different valley, and the midpoint between two valleys can be on a ridge. The longer each hiker
walks before the average (more local epochs $E$), the further apart the valleys.

**FedCross.** One hiker walks the mountain alone, but lets each hospital's map take over in turn: first
the model trains on hospital A for a while, then it is handed to B, then to C. No averaging, so no ridge
between valleys. The cost: the walker is always drawn towards the hospital it visited last, and the visits cannot run at the same time.

**FedCrossEns.** Send $K$ walkers along different random routes. At test time take the average of their
predicted masks; the pixel-wise standard deviation across the $K$ masks says where they disagree.

## Notation in plain words

- $\theta$: all the network's weights in one long list; $\ell(\theta,D_k)$: the training loss on hospital $k$'s data;
  $\nabla\ell$: the gradient, the direction in which the loss rises fastest, so we step opposite to it.
- $w_k=\|D_k\|/\sum_i\|D_i\|$: hospital $k$'s share of the total data; the weights add up to 1.
- $\alpha_j$: the step size at local step $j$; $J$: the number of local steps per round.
- $\theta_k^{j}$: hospital $k$'s copy of the weights after $j-1$ local steps, so $\theta_k^1=\theta$ is the broadcast model.
- "Non-iid": hospitals' data are not draws from one common distribution (different scanners, patient mixes).
- $L(\theta)=\sum_k w_k\,\ell(\theta,D_k)$: the loss the federation actually wants to minimise.
- **DSC** (Dice): twice the overlap of predicted and true masks divided by the sum of their sizes; 1 is perfect.

## The analysis (Section 2.2), step by step

Each hospital runs $J$ gradient steps (their Eq. 1): $\theta_k^{j+1}=\theta_k^j-\alpha_j\nabla\ell(\theta_k^j,D_k)$.
The server's average is $\theta'=\sum_k w_k\theta_k^{J+1}$ (Eq. 2). Substituting the local updates, and using
$\sum_k w_k=1$ and $\theta_k^1=\theta$, gives (Eq. 3)

$$\theta'=\theta-\underbrace{\alpha_1\sum_k w_k\nabla\ell(\theta,D_k)}_{=\ \alpha_1\nabla L(\theta)}-\underbrace{\sum_k w_k\sum_{j=2}^{J}\alpha_j\nabla\ell(\theta_k^j,D_k)}_{\text{gradients at each hospital's own wandering iterate}}.$$

In words: the first correction is one genuine step on the federation's loss; the second is each hospital's later steps,
taken at points that have already drifted away from $\theta$ in different directions. The authors say this term
"may not be a descent direction" and can "grow arbitrarily" with $J$, hence averaging is risky on non-iid data, and
the risk rises with the local epochs $E$. The algebra checks out (the paper drops the factor $\alpha_1$ in front of $\nabla L$).

FedCross then trains on one hospital per round, with $K$ times as many local epochs so the number of epochs
per hospital matches FedAvg. They note it equals centralised training in which the samples arrive grouped by hospital.

## Results (four prostate MRI datasets as four clients, 3D U-Net, 400 epochs in total)

| Method | Global Dice (%) | Global ASD (mm) |
|---|---|---|
| Localised (each hospital alone) | 85.08 | 2.33 |
| Centralised (pooled data) | 88.61 | 1.43 |
| FedAvg | 86.72 | 1.57 |
| FedProx | 87.64 | 1.54 |
| FedBN | 87.40 | 1.58 |
| FedCross | 87.91 | 1.57 |
| FedCrossEns | 88.73 | 1.22 |

Second experiment (Fig. 5): vary local epochs $E=1,2,4,8,16$ with $T=400/E$ rounds. FedAvg, FedProx and FedBN lose accuracy as $E$
grows; FedCross is claimed to degrade less.

## Questions and doubts

- **The analysis states a worry; it does not prove one.** Eq. 3 is an identity, not a bound. "May not be a
  descent direction" and "can grow arbitrarily" are not backed by an assumption such as smoothness of $\ell$,
  and no inequality shows the averaged model is worse than any alternative. Figure 1 is a two-client cartoon.
- **Averaging can be fine, and the cartoon does not show a failure.** In a convex bowl, the loss at the averaged
  weights is at most the average of the clients' losses (Jensen), so being "not at the minimum of $L$" is no failure by
  itself, and neither client's own minimum is a minimum of $L$ either. The real concern is non-convexity, which the
  introduction mentions but the analysis does not use.
- **The same drift term hits FedCross.** In a FedCross round, one hospital runs $J$ (or $KJ$) steps from $\theta$, so
  the model gets pulled towards that hospital's minimum, exactly the drift Eq. 3 identifies. What changes is that
  drift is not averaged with others'; the paper never argues why sequential drift is less harmful than averaged drift.
  The limitation section (forgetting of hospitals not seen recently) concedes this, and the random order with
  a small number of clients is what is relied upon.
- **The gains are small and untested where it matters.** FedCross against FedProx is 87.91 against 87.64 Dice, a
  gap well inside the spread of per-case Dice (standard deviations of 5 to 7 points). The significance stars compare
  methods with FedCrossEns only, so FedCross against FedProx and FedBN has no test. Test sets are 30% of 19, 48,
  30 and 122 subjects, so a few cases decide each column. One run per setting; no seeds.
- **The headline comparison is not like for like.** FedCrossEns is four models; FedCross, FedAvg and the others are
  one model. Four-fold training and inference compute, and an ensemble of four FedAvg models, would be the fair
  baseline. FedCrossEns reaching 88.73 against the centralised 88.61 is partly just the ensemble effect.
- **The epoch experiment is read off a figure.** Fig. 5 has no error bars and no numbers, and when $E=16$ there are
  only 25 rounds with one random client each, so the visiting pattern differs a lot between runs.
- **Uncertainty is shown, not evaluated.** The standard deviation of four masks is a map with an upper limit
  determined by only four samples, and Figure 4 is a visual example. The paper does not check that high
  uncertainty falls where errors are, or compare it with any other estimate.
- **"No extra communication" but no parallelism.** Per client, bytes sent match FedAvg, but only one hospital works at a
  time, so wall-clock time grows by about $K$ (the authors list lack of parallelism as a limitation). This is a bigger
  deal when many clients are involved, whereas the paper's setting has four.
- **Privacy is not discussed.** In FedAvg, secure aggregation can hide each hospital's update inside the sum. In FedCross
  the next hospital receives a model that was just fitted to one hospital alone, which can reveal more about that
  hospital's data than an average would. Aggregation-free also means secure aggregation is unavailable.
- **The "non-iid" here is scanner shift only.** All four datasets are prostate gland segmentation, and Fig. 3
  shows PROMISE12's intensity histogram differs. Label shift and unequal clients are not tested.

## Takeaways

- The useful insight is a plain one: averaging weights after long local training is a risk on heterogeneous
  data, and the longer the local training, the bigger the risk.
- Sequential training sidesteps averaging at the cost of forgetting, serial time and (arguably) privacy; with few
  clients and random order the cost seems small in the reported experiment.
- The better numbers for FedCrossEns come from ensembling; they would need an ensembled FedAvg to be compared
  fairly.
