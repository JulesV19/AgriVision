# AgriVision : preuve de concept de segmentation d'anomalies agricoles

Backbone **DINOv3 ViT-S/16** (21 M de paramètres, gelé) et une **tête de segmentation légère** (environ 1,6 M de paramètres entraînables), entraînés sur **Agriculture-Vision 2021** en RGB.
C'est la première étape avant de passer à l'estimation de sinistres (grêle, sangliers, etc.).

## Contenu

| Fichier | Rôle |
|---|---|
| `notebooks/train_colab.ipynb` | Pipeline complet sur Colab Pro : données, entraînement, évaluation, visualisation |
| `agrivision/prepare_subset.py` | Construit un sous-ensemble équilibré à partir de l'archive officielle |
| `agrivision/data.py` | Dataset et décodage des labels multi-classes |
| `agrivision/model.py` | `DinoSegmenter` : DINOv3 gelé, 4 couches intermédiaires, décodeur conv avec skip haute résolution |
| `agrivision/losses.py` | CE « au moins une classe vraie », adaptée aux annotations qui se chevauchent, + Dice |
| `agrivision/metrics.py` | mIoU officiel Agriculture-Vision (matrice de confusion modifiée) |
| `agrivision/train.py` | Entraînement (bf16, cosine, reprise auto) et `--eval-only` |

## Démarrage sur Colab

1. Demander l'accès à [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). Le modèle est *gated* : il faut accepter la licence DINOv3.
2. Ajouter un token HF dans les secrets Colab sous le nom `HF_TOKEN`.
3. Copier ce dossier dans `MyDrive/agrivision/code`, ou le pousser sur GitHub et mettre `CODE_SOURCE = "git"`.
4. Ouvrir `notebooks/train_colab.ipynb` sur un runtime L4 ou A100 et tout exécuter.

## Données : Agriculture-Vision 2021

- Bucket public AWS Open Data : `s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz` (21 Go, sans compte AWS).
- Tuiles de 512×512 en RGB + NIR, environ 10 cm/px, prises au-dessus de parcelles aux États-Unis. Le split train compte environ 57k tuiles et le val environ 18k. Les labels du test officiel ne sont pas publics.
- 9 classes évaluées : background, double_plant, drydown, endrow, nutrient_deficiency, planter_skip, water, waterway, weed_cluster. `storm_damage` est présente dans les labels mais exclue de l'évaluation officielle. Elle est encodée (bit 9) dans le sous-ensemble pour pouvoir l'utiliser plus tard.
- Les annotations peuvent se chevaucher. Les pixels hors masque ou hors limite de parcelle sont ignorés.
- Sous-ensemble par défaut : 12k tuiles train, échantillonnées en favorisant les classes rares, 3k val et 2k test. Val et test sont tirés du val officiel sur des parcelles disjointes.
- Format : `rgb/<id>.jpg` et `lbl/<id>.png` en uint16, où bit 0 = valide et bit k = classe k.

Conditions d'utilisation du dataset : voir `Agriculture-Vision Dataset Terms of Use.pdf` dans le bucket. Licence des poids : DINOv3 License (Meta).

## Pistes après la PoC

- `--backbone-res 768` : patchs plus fins, donc des contours meilleurs, mais environ 2,5× plus lent.
- Backbone `dinov3-vits16plus` ou `dinov3-vitb16`, ou LoRA sur le ViT.
- Le backbone satellite SAT-493M n'existe qu'en ViT-L et 7B.
- Ajout du NIR via un adaptateur. Activation de `storm_damage`.
- Fine-tuning sur des données de sinistres réelles (grêle, sangliers).
