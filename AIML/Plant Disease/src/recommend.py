"""Phase 10 — Disease recommendation engine.

Maps every class to a concrete agronomic action. Healthy classes return a
monitoring message. ``recommend(label, confidence)`` returns a farmer-facing
string used by ``predict.py``.
"""
from __future__ import annotations

RECOMMENDATIONS = {
    # --- Apple ---
    "Apple___Apple_Scab": "Apply protectant fungicide (captan or mancozeb) at green-tip; rake and destroy fallen leaves to cut overwintering spores.",
    "Apple___Black_Rot": "Prune out cankers and mummified fruit; apply captan/thiophanate-methyl from bud break. Remove nearby dead wood.",
    "Apple___Cedar_Apple_Rust": "Apply myclobutanil/mancozeb at pink stage; remove nearby junipers (alternate host) within ~1 km if possible.",
    "Apple___Healthy": "Leaf looks healthy. Maintain routine scouting and balanced irrigation/fertilization.",
    # --- Cherry ---
    "Cherry___Powdery_Mildew": "Apply sulfur or a DMI fungicide; improve canopy airflow with pruning and avoid excess nitrogen.",
    "Cherry___Healthy": "Leaf looks healthy. Continue regular monitoring, especially in humid spells.",
    # --- Corn ---
    "Corn___Common_Rust": "Usually low-impact; plant resistant hybrids. Apply a triazole/strobilurin fungicide only if infection is early and severe.",
    "Corn___Gray_Leaf_Spot": "Rotate crops and bury residue; use resistant hybrids and a strobilurin fungicide at tasseling if pressure is high.",
    "Corn___Northern_Leaf_Blight": "Plant resistant hybrids; apply fungicide at first lesion if before silking. Rotate and till residue.",
    "Corn___Healthy": "Leaf looks healthy. Maintain nitrogen management and scout after rain.",
    # --- Grape ---
    "Grape___Black_Rot": "Remove mummified berries and infected canes; apply mancozeb/myclobutanil from early shoot growth through fruit set.",
    "Grape___Esca": "Prune out infected wood in dry weather and seal large cuts; no cure — manage vine stress and avoid large pruning wounds.",
    "Grape___Leaf_Blight": "Apply copper or mancozeb; improve canopy ventilation and remove infected leaves to limit spread.",
    "Grape___Healthy": "Leaf looks healthy. Keep canopy open and monitor through veraison.",
    # --- Peach ---
    "Peach___Bacterial_Spot": "Use resistant cultivars and copper sprays (dormant + early season); avoid overhead irrigation and high nitrogen.",
    "Peach___Healthy": "Leaf looks healthy. Continue scouting; thin for airflow.",
    # --- Pepper ---
    "Pepper___Bacterial_Spot": "Use certified disease-free seed and resistant varieties; rotate crops and apply copper + mancozeb. Avoid working plants when wet.",
    "Pepper___Healthy": "Leaf looks healthy. Maintain drip irrigation to keep foliage dry.",
    # --- Potato ---
    "Potato___Early_Blight": "Apply chlorothalonil/mancozeb on a 7-10 day schedule; remove lower infected leaves and maintain plant vigor with adequate nitrogen.",
    "Potato___Late_Blight": "Act fast — apply copper-based or systemic fungicide; destroy infected plants, avoid overhead watering, and harvest in dry conditions.",
    "Potato___Healthy": "Leaf looks healthy. Hill soil over tubers and scout intensively in cool, wet weather.",
    # --- Strawberry ---
    "Strawberry___Leaf_Scorch": "Renovate beds and remove old infected foliage; apply captan/myclobutanil and improve airflow with proper spacing.",
    "Strawberry___Healthy": "Leaf looks healthy. Keep mulch dry and monitor after rain.",
    # --- Tomato ---
    "Tomato___Bacterial_Spot": "Use disease-free seed and resistant varieties; apply copper + mancozeb, rotate crops, and avoid overhead irrigation.",
    "Tomato___Early_Blight": "Apply chlorothalonil/mancozeb; remove lower infected leaves, mulch to block soil splash, and stake for airflow.",
    "Tomato___Late_Blight": "Urgent — apply systemic fungicide (e.g. cymoxanil); remove and destroy infected plants and avoid leaf wetness.",
    "Tomato___Healthy": "Leaf looks healthy. Maintain consistent watering and prune for airflow.",
}

_DEFAULT = ("No specific rule found for this class — isolate the plant, consult a "
            "local agronomist, and avoid overhead watering until identified.")


def recommend(label: str, confidence: float | None = None, low_conf: float = 0.6) -> str:
    base = RECOMMENDATIONS.get(label, _DEFAULT)
    if confidence is not None and confidence < low_conf:
        return (f"(Low confidence {confidence:.0%} — verify manually.) " + base)
    return base


def is_healthy(label: str) -> bool:
    return label.split("___")[-1].strip().lower() == "healthy"


if __name__ == "__main__":
    # sanity print
    for k in ("Tomato___Early_Blight", "Potato___Late_Blight", "Pepper___Bacterial_Spot"):
        print(f"{k}\n  -> {recommend(k, 0.97)}\n")
