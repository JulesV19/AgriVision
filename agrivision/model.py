import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DEFAULT_BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"


def conv_bn(cin, cout, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


def _blocks(vit):
    """Liste des blocs transformer (le nom de l'attribut change selon la version de transformers)."""
    n = vit.config.num_hidden_layers
    for m in vit.modules():
        if isinstance(m, nn.ModuleList) and len(m) == n:
            return m
    raise RuntimeError("blocs du ViT introuvables")


class DinoSegmenter(nn.Module):
    """Backbone DINOv3 ViT (gelé, ou derniers blocs ajustés) + petit décodeur convolutif.

    - On prend les patch tokens de plusieurs couches intermédiaires (1/16 de la résolution),
      normalisés par la LayerNorm finale (comme get_intermediate_layers(norm=True) de DINOv3).
    - Une petite branche convolutive sur l'image (1/2 et 1/4) récupère le détail fin que
      les patchs de 16 px perdent, pour des contours plus propres.
    - unfreeze_last=k : les k derniers blocs du ViT (et sa LayerNorm finale) sont aussi entraînés.
    """

    def __init__(self, num_classes, backbone=DEFAULT_BACKBONE, layers=(2, 5, 8, 11), dim=256,
                 backbone_res=None, unfreeze_last=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        cfg = self.backbone.config
        self.layers = tuple(layers)
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self.patch = cfg.patch_size
        self.backbone_res = backbone_res  # ex. 768 pour des patchs plus fins (plus lent)
        self.unfreeze_last = unfreeze_last
        self.backbone.requires_grad_(False)
        if unfreeze_last > 0:
            for blk in list(_blocks(self.backbone))[-unfreeze_last:]:
                blk.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)

        c = cfg.hidden_size
        self.fuse = nn.Sequential(conv_bn(c * len(layers), dim, k=1), conv_bn(dim, dim))
        self.stem2 = conv_bn(3, 32, stride=2)   # 1/2
        self.stem4 = conv_bn(32, 64, stride=2)  # 1/4
        self.up8 = conv_bn(dim, 128)
        self.up4 = nn.Sequential(conv_bn(128 + 64, 96), conv_bn(96, 96))
        self.up2 = nn.Sequential(conv_bn(96 + 32, 48), conv_bn(48, 48))
        self.classifier = nn.Conv2d(48, num_classes, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # aucun dropout/drop-path dans DINOv3 : le mode eval ne change que ça
        return self

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def backbone_features(self, x):
        if self.backbone_res and self.backbone_res != x.shape[-1]:
            x = F.interpolate(x, size=(self.backbone_res, self.backbone_res), mode="bilinear", align_corners=False)
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        with torch.set_grad_enabled(self.unfreeze_last > 0 and torch.is_grad_enabled()):
            out = self.backbone(pixel_values=x, output_hidden_states=True)
            n_layers = self.backbone.config.num_hidden_layers
            hidden = out.hidden_states[-n_layers:]  # sorties des blocs (sans l'embedding)
            feats = []
            for i in self.layers:
                t = self.backbone.norm(hidden[i])[:, self.n_prefix:]
                feats.append(t.transpose(1, 2).reshape(t.shape[0], -1, h, w))
        return torch.cat(feats, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        f = self.fuse(self.backbone_features(x))
        s2 = self.stem2(x)
        s4 = self.stem4(s2)

        f = F.interpolate(f, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        f = self.up8(f)
        f = F.interpolate(f, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        f = self.up4(torch.cat([f, s4], 1))
        f = F.interpolate(f, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        f = self.up2(torch.cat([f, s2], 1))
        return F.interpolate(self.classifier(f), size=(H, W), mode="bilinear", align_corners=False)

    def head_state_dict(self):
        """Poids entraînés seulement (le reste du backbone se recharge depuis HF)."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.") or k in trainable}


class SmpSegmenter(nn.Module):
    """Référence CNN classique (segmentation_models_pytorch), entièrement entraînée.

    arch ex. "DeepLabV3Plus", "FPN", "Unet" ; encoder ex. "efficientnet-b3", "resnet50".
    """

    def __init__(self, num_classes, arch="DeepLabV3Plus", encoder="efficientnet-b3"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = getattr(smp, arch)(encoder_name=encoder, encoder_weights="imagenet", classes=num_classes)

    def backbone_parameters(self):
        return list(self.net.encoder.parameters())

    def forward(self, x):
        return self.net(x)

    def head_state_dict(self):
        return self.state_dict()
