# Do CNNs Rely on Texture Rather Than Shape?

## Question

**Quel est l'impact du champ réceptif sur le biais de texture des CNN ?**

On se place dans le cadre de [REFINED-BIAS](https://arxiv.org/pdf/2603.10834).

On remplace le [cue-conflict](https://arxiv.org/pdf/1811.12231) par un jeu de données séparé.

L'idée sous-jacente : il existe une corrélation entre la taille du champ réceptif (RF) et le biais de texture des CNN.

## Plan

On va tester plusieurs hypothèses (en fonction de notre temps) :

1. Sur des CNN entraînés sur ImageNet-1k, la taille du RF effectif est corrélée positivement à `Shape-Sens` et négativement à `Texture-Sens`, y compris pour les CNN modernes.
2. À recette d'entraînement comparable, les CNN à grands noyaux (RepLKNet, UniRepLKNet, ConvNeXt) ont une `Shape-Sens` plus élevée que ResNet-50.
3. Le `cue-conflict` et `REFINED-BIAS` classent les modèles différemment ($\tau$ de Kendall faible).
4. Dans un CNN contrôlé, faire varier uniquement la taille des noyaux (ou une convolution *depthwise*, ou la dilatation) modifie `Shape-Sens`. Si R5 (RF large, petit noyau) se comporte comme R3, c'est le RF qui compte, pas le noyau.

*(Les étapes 3 et 4 sont interchangeables en fonction du temps.)*

### I. Sur des CNN entraînés sur ImageNet-1k, la taille du RF effectif est corrélée positivement à `Shape-Sens` et négativement à `Texture-Sens`, y compris pour les CNN modernes

Définir les CNN modernes, réutiliser le cadre expérimental de `REFINED-BIAS` et constater la corrélation entre le RF et le biais de texture.


### II


WIP

### III

WIP

#### IV
WIP


## Glossaire

* **`cue-conflict`** : protocole de Geirhos et al. (2019) qui superpose la texture d'une classe sur la forme d'une autre (via transfert de style) pour mesurer si un modèle prédit plutôt selon la forme ou la texture.
* **`REFINED-BIAS`** : benchmark de Kim et al. (2026) qui remplace le `cue-conflict` par deux jeux d'images séparés (contours seuls pour la forme, mosaïques de patchs mélangés pour la texture) évalués par rang moyen (MRR) sur les 1 000 classes.
* **`Shape-Sens`** : sensibilité à la forme mesurée par `REFINED-BIAS` sur des images réduites aux contours de l'objet.
* **`Texture-Sens`** : sensibilité à la texture mesurée par `REFINED-BIAS` sur des mosaïques de patchs où la structure globale est détruite.

## Sources

* **REFINED-BIAS** : Kim et al., *On the Reliability of Cue Conflict and Beyond*, 2026 — [arXiv:2603.10834](https://arxiv.org/abs/2603.10834) · [GitHub](https://github.com/PumjunKim/REFINED-BIAS)
* **Cue-conflict** : Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019 — [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)
* **Effective Receptive Field (ERF)** : Luo et al., *Understanding the Effective Receptive Field in Deep CNNs*, NeurIPS 2016
* **ConvNeXt** : Liu et al., *A ConvNet for the 2020s*, CVPR 2022 — [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
* **RepLKNet** : Ding et al., *Scaling Up Your Kernels to 31×31*, CVPR 2022 — [GitHub](https://github.com/DingXiaoH/RepLKNet-pytorch)
* **UniRepLKNet** : Ding et al., *UniRepLKNet*, CVPR 2024 — [GitHub](https://github.com/AILab-CVC/UniRepLKNet)
* **Plan détaillé du projet** : [doc/plan.md](doc/plan.md)


