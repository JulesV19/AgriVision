"""PoC de segmentation Agriculture-Vision avec un backbone DINOv3 gelé."""

# Classes officielles du challenge Agriculture-Vision 2021 (id = indice).
CLASSES = [
    "background",
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "water",
    "waterway",
    "weed_cluster",
]
# Présente dans les labels mais exclue de l'évaluation officielle 2021.
# On l'encode quand même (bit 9) pour pouvoir l'activer plus tard (proche "grêle/tempête").
EXTRA_CLASSES = ["storm_damage"]

# Encodage des PNG uint16 du sous-ensemble :
#   bit 0      -> pixel valide (mask & boundary)
#   bit k (>0) -> pixel annoté avec la classe CLASSES[k] (ou EXTRA_CLASSES pour k >= 9)
LABEL_BITS = {name: i for i, name in enumerate(CLASSES + EXTRA_CLASSES) if i > 0}

# Normalisation des backbones DINOv3 LVD-1689M (images web).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
