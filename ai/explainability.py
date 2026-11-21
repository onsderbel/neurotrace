"""Explicabilité : Grad-CAM (carte de chaleur) et outils d'affichage.

Grad-CAM montre les zones de l'image qui ont le plus contribué à la prédiction du modèle.
C'est une AIDE À L'INTERPRÉTATION, pas une preuve médicale : une zone chaude ne signifie
pas qu'une tumeur s'y trouve.

La partie « affichage » (overlay_heatmap, summarize_cam) n'a besoin ni de PyTorch ni du modèle.
"""
from __future__ import annotations

import cv2
import numpy as np

METHOD = "Grad-CAM"
DISCLAIMER = ("Aide à l'interprétation uniquement : les zones mises en évidence indiquent où le modèle "
              "a regardé, pas la présence prouvée d'une tumeur.")


# ---------------------------------------------------------------- affichage (sans PyTorch)

def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpose la carte de chaleur (valeurs 0..1) sur l'image RGB uint8. Retourne une image RGB uint8."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha doit être compris entre 0 et 1")
    h, w = image_rgb.shape[:2]
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    if cam.shape != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_bgr = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def summarize_cam(cam: np.ndarray, threshold: float = 0.5) -> dict:
    """Résumé chiffré de la carte de chaleur (utile pour la validation et pour l'explication écrite)."""
    cam = np.asarray(cam, dtype=np.float32)
    h, w = cam.shape
    if float(cam.max()) <= 0.0:
        return {"empty": True, "peak_xy": None, "center_of_mass_xy": None, "high_activation_area_ratio": 0.0}
    py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
    total = float(cam.sum())
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = float((cam * xs).sum() / total), float((cam * ys).sum() / total)
    return {
        "empty": False,
        # coordonnées normalisées entre 0 et 1 (0,0 = coin haut-gauche)
        "peak_xy": [round(px / (w - 1), 3), round(py / (h - 1), 3)],
        "center_of_mass_xy": [round(cx / (w - 1), 3), round(cy / (h - 1), 3)],
        "high_activation_area_ratio": round(float((cam >= threshold).mean()), 4),
    }


# ---------------------------------------------------------------- Grad-CAM (PyTorch)

class GradCAM:
    """Grad-CAM sur la dernière couche convolutive du réseau.

    Usage :  cam = GradCAM(model, model.layer4[-1])(input_tensor, target_index)
    Le modèle doit être en mode eval(). Retourne un tableau float32 (H, W) normalisé entre 0 et 1.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor, target_index: int):
        import torch
        import torch.nn.functional as F

        store = {}

        def forward_hook(_module, _inp, out):
            store["activations"] = out
            out.register_hook(lambda grad: store.__setitem__("gradients", grad))

        handle = self.target_layer.register_forward_hook(forward_hook)
        try:
            with torch.enable_grad():
                x = input_tensor.clone().detach().requires_grad_(True)
                logits = self.model(x)
                self.model.zero_grad(set_to_none=True)
                logits[0, target_index].backward()
        finally:
            handle.remove()

        acts, grads = store["activations"].detach(), store["gradients"].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)                 # importance de chaque canal
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))        # combinaison, valeurs positives seulement
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy().astype(np.float32)
        self.model.zero_grad(set_to_none=True)

        peak = float(cam.max())
        return cam / peak if peak > 0 else np.zeros_like(cam)
