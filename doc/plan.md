# CNN : texture ou forme ? Le champ réceptif sous la loupe de REFINED-BIAS

> Question du projet : *Do CNNs rely on texture rather than shape?*
> Contraintes : environ 9 jours-homme, 1 GPU RTX 2000 Ada (16 Go). Les évaluations se font le jour et les entraînements contrôlés tournent la nuit.

## 1. Point de départ

- **Geirhos et al. (ICLR 2019)** : avec le *cue-conflict*, un ResNet-50 ImageNet paraît biaisé texture (shape bias ≈ 0,22).
- **REFINED-BIAS (Kim et al., 2026, arXiv 2603.10834)** remplace le cue-conflict par deux jeux d'images séparés :
  - **forme** : objets réduits à leurs contours ;
  - **texture** : mosaïques de patchs mélangés.

  La métrique est un rang (MRR) sur les 1000 classes. Le même ResNet-50 obtient alors Shape-Sens 0,563 et Texture-Sens 0,568 : **pas de biais texture**.
- REFINED-BIAS observe aussi, sur 14 anciens CNN, une corrélation forte entre la taille du champ réceptif (RF) et la sensibilité à la forme (ρ = 0,938) ou à la texture (ρ = 0,912). Les auteurs la présentent comme un **mécanisme plausible, pas causal**.
  - Ils n'ont testé **aucun CNN moderne** (ConvNeXt, grands noyaux, convolutions déformables).
  - Ils ont **exclu les transformers** de cette analyse.

Le détail de la littérature est dans la note de projet `etat-de-l-art-et-gap.md`.

## 2. Objectif et hypothèses

Utiliser REFINED-BIAS pour tester si le champ réceptif détermine la sensibilité forme/texture, en étendant l'analyse aux CNN modernes.

| | Hypothèse (falsifiable) | Niveau |
|---|---|---|
| **H1** | Sur des CNN entraînés sur ImageNet-1k, la taille du RF *effectif* est corrélée positivement à Shape-Sens et négativement à Texture-Sens, y compris pour les CNN modernes. | 1 (cœur) |
| **H2** | À recette d'entraînement comparable, les CNN à grands noyaux (RepLKNet, UniRepLKNet, ConvNeXt) ont une Shape-Sens plus élevée que ResNet-50. | 1 (cœur) |
| **H3** | Le cue-conflict et REFINED-BIAS classent les modèles différemment (τ de Kendall faible). | 2 (option) |
| **H4** | Dans un CNN contrôlé, faire varier **uniquement** la taille des noyaux (ou une convolution depthwise, ou la dilatation) change Shape-Sens. Si R5 (RF large, petit noyau) se comporte comme R3, c'est le RF qui compte, pas le noyau. | 3 (causal) |

Si H1 et H2 tiennent après les contrôles du §6, l'hypothèse « le RF explique le biais » est renforcée. Si elles tombent, la recette d'entraînement est le meilleur suspect, ce qui va dans le sens de Hermann et al. 2020.

## 3. Plan par niveaux

### Niveau 1 : le cœur (obligatoire)

1. **Reproduire** le ResNet-50 de référence de REFINED-BIAS (Shape-Sens 0,563 / Texture-Sens 0,568, à ±0,01 près). Si on n'y arrive pas, on s'arrête là et on investigue.
2. **Évaluer** environ 15 à 18 modèles (voir §4) avec le code officiel REFINED-BIAS, à 224 px. On réutilise la fonction `load_model` qui gère timm.
3. **Mesurer le RF** de chaque modèle :
   - le RF *théorique*, calculé ;
   - le RF *effectif* (ERF), par gradient de l'unité centrale de la sortie par rapport à l'entrée, moyenné sur environ 500 images (Luo et al., NeurIPS 2016). On en tire un score scalaire : le rayon contenant X % de la masse du gradient.
4. **Analyser** :
   - Spearman entre ERF et Shape-Sens / Texture-Sens ;
   - corrélation partielle en contrôlant la précision top-1 ;
   - comparaison par paires contrôlées ;
   - intervalles de confiance par bootstrap sur les images *et* sur les classes ;
   - **Texture-Sens par taille de grille.** La grille est écrite dans le nom de fichier (`16x16_brain_coral_180.png`). Répartition vérifiée : 4×4 = 900, 8×8 = 900, 16×16 = 600, 32×32 = 600 images, équilibrées entre les classes. Cette analyse teste si le lien RF ↔ texture observé par REFINED-BIAS est un artefact du mélange de patchs (faille 3).

### Niveau 2 : comparaison avec le cue-conflict (si le temps le permet)

Passer les mêmes modèles sur le cue-conflict avec la toolbox `model-vs-human` (Geirhos et al. 2021), puis calculer le τ de Kendall entre les classements obtenus par les deux métriques.

### Niveau 3 : nos propres CNN, la partie causale (GPU : RTX 2000 Ada, 16 Go)

Lancé **en parallèle** du niveau 1 : les entraînements tournent la nuit pendant que les évaluations et les analyses se font le jour.

- **Les données** : « ImageNet-100-RB », soit les 20 classes REFINED-BIAS plus 80 classes ImageNet tirées au hasard avec une graine fixe. Cela fait environ 130 000 images d'entraînement, et la validation officielle de ces 100 classes sert de test.
  - Imagenette ne convient pas : il ne contient pas ces classes.
  - **Prérequis : avoir accès à ImageNet-1k** (image-net.org ou Hugging Face, après acceptation des conditions d'utilisation).
- **Le pré-traitement** : redimensionner une fois pour toutes les images à 160 px (côté court) et les stocker. Sans ça, c'est le décodage JPEG sur le CPU qui limite la vitesse, pas le GPU.
- **L'architecture** : un réseau de type ConvNeXt « micro » (4 étages, environ 5 à 10 M de paramètres). Le bloc est une convolution depthwise k×k suivie de deux pointwise. On ne change qu'un facteur à la fois :

  | Run | Facteur testé | Noyau | Dilatation |
  |---|---|---|---|
  | R1 | référence | 3×3 | 1 |
  | R2 | taille de noyau | 7×7 | 1 |
  | R3 | taille de noyau | 13×13 | 1 |
  | R4 | taille de noyau | 31×31 | 1 |
  | R5 | **RF sans grand noyau** | 3×3 | dilatée pour atteindre le RF théorique de R3 |
  | R6 | depthwise ou pas | 3×3 dense | 1 |

  - Les paires R3/R5 séparent « grand champ réceptif » et « grand noyau ».
  - La paire R1/R6 isole le depthwise. Le nombre de paramètres change un peu d'un run à l'autre : on le rapporte.
- **La recette** est identique pour tous : AdamW, environ 60 époques à 128 px, AMP, même graine, augmentations légères (crop + flip) pour ne pas mélanger d'effet d'augmentation. On fait **2 graines** pour R1 et R4, afin de mesurer le bruit dû à la graine.
- **Le budget estimé** (ordre de grandeur, à confirmer par un run d'essai de 2 époques au J1) : environ 2 à 4 h par run pour R1, R2 et R6, plus pour R4 car les grandes convolutions depthwise sont lentes sous PyTorch. Cela fait environ 20 à 30 h de GPU au total, soit 3 nuits.
- **L'évaluation** : images REFINED-BIAS redimensionnées à la résolution de test, MRR sur les **100 labels**, ERF mesuré sur chaque run. Ces chiffres ne sont **comparables qu'entre modèles du niveau 3**, jamais avec ceux du niveau 1.

## 4. Modèles : tous entraînés sur ImageNet-1k uniquement (vérifié le 27/11/2026)

Tous les modèles ci-dessous ont un checkpoint entraîné **uniquement sur ImageNet-1k** en 224 px. On écarte toutes les variantes pré-entraînées sur IN-22k, IN-12k, LAION ou JFT.

| Rôle | Modèle (checkpoint) | Ce qui varie | Source |
|---|---|---|---|
| Ancrage sur le papier | `resnet50.tv_in1k`, `resnet18.tv_in1k`, `resnet152.tv_in1k`, `vgg16.tv_in1k` | profondeur, RF théorique | timm (poids torchvision) |
| RF très petit | BagNet-17 (et 9 / 33) | RF borné par construction | dépôt REFINED-BIAS / BagNet |
| **Même recette moderne** | `resnet50.tv2_in1k`, `resnext50_32x4d.tv2_in1k`, `regnety_016.tv2_in1k`, `resnet152.tv2_in1k` | architecture, recette torchvision V2 fixe | timm |
| **Paire contrôlée** | `resnet50.a2_in1k` contre `convnext_tiny_hnf.a2h_in1k` | ResNet contre ConvNeXt, recette timm de type A2 (vérifier dans la doc timm qu'elle est bien identique) | timm |
| Recette forte | `resnet50.a1_in1k` | référence pour les modèles « recette auteur » | timm |
| Depthwise | `mobilenetv2_100.ra_in1k`, `efficientnet_b0.ra_in1k`, `convmixer_1024_20_ks9_p14.in1k` | depthwise, noyau 3 contre 9 | timm |
| CNN modernes | `convnext_tiny.fb_in1k` (noyau 7×7 depthwise) | recette des auteurs | timm |
| Grands noyaux | RepLKNet-31B IN-1K 224 (83,5 %), UniRepLKNet-T IN-1K 224 (83,2 %) | noyaux 31×31 / 13×13 | dépôts officiels (absents de timm) |
| Déformable | InternImage-T IN-1K 224 (83,5 %) | DCNv3, RF adaptatif | dépôt officiel (absent de timm, opérateur CUDA à compiler) |
| À part | `convnextv2_tiny.fcmae_ft_in1k` | pré-entraînement auto-supervisé FCMAE **sur IN-1k** | timm |
| Contrôle non-CNN (option) | `deit_small_patch16_224.fb_in1k`, `swin_tiny_patch4_window7_224.ms_in1k` | attention | timm |

Ordre de priorité si le temps manque : ancrage, puis même recette, puis paire contrôlée, puis ConvNeXt, puis RepLKNet et UniRepLKNet. Viennent ensuite le depthwise, puis InternImage et les autres.

## 5. Calendrier indicatif (9 jours-homme)

A et B désignent deux volets de travail : évaluations (A) et entraînement (B). Si une seule personne s'en charge, elle les mène en alternance, le GPU tournant la nuit dans tous les cas.

| Jour | Le jour (volet A / B) | La nuit (GPU) |
|---|---|---|
| J1 | A : installation, reproduction de ResNet-50 (critère d'arrêt). B : sous-ensemble ImageNet-100-RB, pré-redimensionnement, code du ConvNeXt micro, run d'essai de 2 époques (mesure du temps réel) | Pré-traitement des données |
| J2 | A : modèles hors timm, code ERF. B : vérification du pipeline d'entraînement | R1 + R6 |
| J3 | A : évaluations REFINED-BIAS (niveau 1). B : évaluation de R1 et R6 | R2 + R3 |
| J4 | A : fin des évaluations + ERF. B : évaluation de R2 et R3 | R4 |
| J5 | Analyses H1/H2, corrélations partielles, analyse par grille | R5 + 2e graine de R1 |
| J6 | Niveau 3 : évaluations et analyse H4. *Option* niveau 2 (cue-conflict) | 2e graine de R4 |
| J7 | Croisement entre modèles pré-entraînés et contrôlés, figures | Marge (relance si un run a échoué) |
| J8–J9 | Rédaction, relecture | |

**Règle d'arrêt** : si au J2 le run d'essai indique plus de 6 h par run, on réduit à R1 / R3 / R4 / R5 (une seule graine) ou on descend à 112 px.

## 6. Failles connues et parades

1. **Même dataset ne veut pas dire même entraînement.** C'est la faille principale.
   - Tous nos modèles sont bien sur ImageNet-1k. En revanche, ResNet-50 torchvision V1 utilise environ 90 époques avec crop et flip, alors que ConvNeXt et RepLKNet utilisent environ 300 époques avec Mixup, CutMix, RandAugment, etc.
   - REFINED-BIAS montre lui-même que la recette change la préférence, à architecture fixe.
   - Une corrélation RF ↔ Shape-Sens sur des modèles pré-entraînés ne prouve donc rien seule.
   - *Parade* : familles à recette partagée (torchvision V2, paire A2), comparer les modèles « recette auteur » à `resnet50.a1_in1k` plutôt qu'au ResNet-50 V1, et surtout le niveau 3, qui est la seule vraie parade causale.
2. **La taille du noyau n'est pas le champ réceptif.**
   - Le RF théorique d'un ResNet-50 dépasse déjà 224 px, donc il sature et ne discrimine rien.
   - *Parade* : utiliser le RF **effectif** mesuré, et rapporter les deux.
3. **Les images texture interagissent mécaniquement avec le RF.**
   - Ces images sont des mosaïques de patchs mélangés (grilles 4×4 à 32×32, soit des patchs de 56 à 7 px).
   - Un modèle à grand RF voit les frontières incohérentes entre patchs, un modèle à petit RF voit des patchs intacts.
   - Une corrélation « petit RF ↔ forte Texture-Sens » peut donc être en partie fabriquée par le stimulus.
   - *Parade* : analyser Texture-Sens **par taille de grille**. Si l'effet du RF se concentre sur les grilles fines, c'est un artefact.
4. **Les images forme sont hors distribution.**
   - Ce sont des traits blancs sur fond noir.
   - Les modèles entraînés avec beaucoup d'augmentation peuvent mieux s'en sortir sans « utiliser plus la forme ».
   - *Parade* : rapporter la précision ImageNet-val en covariable, et discuter le lien avec la robustesse aux croquis.
5. **Classes et indices sont confondus.**
   - Les classes forme sont des objets fabriqués, les classes texture des animaux ou objets naturels, avec seulement 10 classes de chaque côté.
   - *Parade* : bootstrap **par classe**, résultats par classe en annexe, et ne jamais conclure sur la préférence seule.
6. **Peu de modèles, beaucoup de facteurs confondus.**
   - Avec 15 à 18 points, un ρ élevé peut venir de la précision, de la profondeur ou de l'époque de l'architecture.
   - *Parade* : corrélation partielle sur le top-1, hypothèses et analyses écrites **avant** de voir les résultats, intervalles de confiance plutôt que des p-values seules.
7. **Prétraitement et résolution.**
   - Chaque checkpoint a sa propre normalisation. ConvNeXt est par exemple testé à 288 px par défaut dans timm.
   - *Parade* : forcer 224 px, utiliser la normalisation de chaque modèle (`timm.data.resolve_data_config`), et vérifier la précision ImageNet-val de chaque checkpoint.
8. **Coût d'installation des modèles hors timm.**
   - InternImage demande de compiler l'opérateur DCNv3.
   - RepLKNet doit être chargé dans la bonne forme (avant ou après reparamétrisation).
   - *Parade* : limiter cela à une demi-journée maximum, puis abandonner le modèle.
9. **ConvNeXt V2 n'est pas comparable directement.** Son checkpoint IN-1k passe par un pré-entraînement auto-supervisé (FCMAE), et REFINED-BIAS montre que l'auto-supervision augmente Shape-Sens. *Parade* : le traiter à part.
10. **Le benchmark lui-même est récent.**
    - Il n'a pas été répliqué de façon indépendante, et aucune licence n'est indiquée.
    - Certains chiffres du papier sont ambigus (le 0,49 cité dans le texte).
    - *Parade* : l'étape de reproduction au J1, et ne pas redistribuer les données.
11. **« Depthwise » ou « ConvNeXt » ne sont pas une seule variable.** Dans les modèles pré-entraînés, ces choix arrivent toujours avec d'autres changements (LayerNorm, GELU, couches inversées, recette). Seul le niveau 3 isole un facteur.
12. **Portée de la réponse finale.**
    - REFINED-BIAS mesure une *sensibilité* à des indices isolés, pas une dépendance causale sur des images naturelles.
    - La réponse à la question du projet sera donc conditionnelle à cette métrique.
    - *Parade* : le dire explicitement. Si le temps le permet, croiser avec la méthode par suppression de Burgert et al. 2025.
13. **Niveau 3 : résolution et capacité.**
    - On entraîne à 128 px alors que les images REFINED-BIAS sont en 224 px. Une fois réduits, les contours blancs fins s'amincissent et se crénellent.
    - *Parade* : tester à 160 px (un léger écart entraînement/test aide en général), et vérifier une fois visuellement les images redimensionnées.
    - Les grands noyaux ajoutent aussi des paramètres : rapporter paramètres et FLOPs, et comparer à précision similaire.
14. **Niveau 3 : accès à ImageNet et mémoire.**
    - Sans accès à ImageNet-1k, le niveau 3 tombe : à vérifier **avant** le J1.
    - Avec 16 Go de VRAM, un batch d'environ 256 à 128 px en AMP doit passer, sauf peut-être pour R4.
    - *Parade* : accumulation de gradient si besoin.

## 7. Pistes en réserve (hors périmètre)

- **Point de rupture sous style transfer continu** (note `protocole-point-de-rupture-style.md`) : courbe de précision en fonction de la force de stylisation α. C'est une piste intéressante, mais elle reste sur le terrain du style transfer que REFINED-BIAS critique. Elle est mise de côté au profit de ce plan.
- Visual Anagrams / Configural Shape Score (Doshi et al., NeurIPS 2025) comme troisième métrique.

## 8. Références principales

- Geirhos et al., *ImageNet-trained CNNs are biased towards texture…*, ICLR 2019 — arXiv 1811.12231
- Kim et al., *On the Reliability of Cue Conflict and Beyond* (REFINED-BIAS), 2026 — arXiv 2603.10834 — github.com/PumjunKim/REFINED-BIAS
- Gavrikov & Keuper, *Can Biases in ImageNet Models Explain Generalization?*, CVPR 2024 — arXiv 2404.01509
- Hermann, Chen & Kornblith, *The Origins and Prevalence of Texture Bias in CNNs*, NeurIPS 2020 — arXiv 1911.09071
- Burgert et al., *ImageNet-trained CNNs are not biased towards texture*, 2025 — arXiv 2509.20234
- Luo et al., *Understanding the Effective Receptive Field in Deep CNNs*, NeurIPS 2016
- Ding et al., *Scaling Up Your Kernels to 31×31* (RepLKNet), CVPR 2022 — github.com/DingXiaoH/RepLKNet-pytorch
- Ding et al., *UniRepLKNet*, CVPR 2024 — github.com/AILab-CVC/UniRepLKNet
- Wang et al., *InternImage*, CVPR 2023 — github.com/OpenGVLab/InternImage
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022 — arXiv 2201.03545
- Brendel & Bethge, *Approximating CNNs with Bag-of-local-Features models…* (BagNet), ICLR 2019
