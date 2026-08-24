# ============================================================
# Dépistage de la rétinopathie diabétique par IA — Démo Streamlit
# Projet M1 IA — DIT Dakar
#
# Lancement :  py -m streamlit run app.py
# Prérequis  : modele/modele_rd.keras et modele/config.json
#              (générés par notebook/entrainement_aptos.ipynb)
# ============================================================
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title='Dépistage RD — IA', page_icon='👁️', layout='wide')

DOSSIER_MODELE = os.path.join(os.path.dirname(__file__), 'modele')

CONSEILS = [
    ("✅ Aucun signe de rétinopathie diabétique détecté.", 'Poursuivre le suivi ophtalmologique annuel régulier.'),
    ("🔎 RD légère détectée.", 'Un contrôle ophtalmologique dans 6 à 12 mois est recommandé.'),
    ("⚠️ RD modérée détectée.", 'Consultation ophtalmologique recommandée dans les 3 mois.'),
    ("🔴 RD sévère détectée.", 'Consultation ophtalmologique recommandée sous 1 mois.'),
    ("🚨 RD proliférative détectée.", 'Prise en charge ophtalmologique urgente requise.'),
]


# ---------------- Utilitaires (cohérents avec le notebook) ----------------
def recadrer_fond_oeil(img, seuil=10, marge=0.10):
    """Recadre l'image sur la rétine (suppression des bords noirs), comme à l'entraînement."""
    gris = img.mean(axis=2)
    masque = gris > seuil
    if not masque.any():
        return img
    lignes = np.any(masque, axis=1)
    colonnes = np.any(masque, axis=0)
    h0, h1 = np.where(lignes)[0][[0, -1]]
    w0, w1 = np.where(colonnes)[0][[0, -1]]
    dh, dw = h1 - h0, w1 - w0
    h0 = max(0, int(h0 - marge * dh)); h1 = min(img.shape[0] - 1, int(h1 + marge * dh))
    w0 = max(0, int(w0 - marge * dw)); w1 = min(img.shape[1] - 1, int(w1 + marge * dw))
    return img[h0:h1 + 1, w0:w1 + 1]


def pretraiter(image_pil, taille):
    """Même chaîne de prétraitement que pendant l'entraînement."""
    img = np.array(image_pil.convert('RGB'), dtype=np.float32)
    img = tf.image.resize(img, [taille * 2, taille * 2]).numpy()
    img = recadrer_fond_oeil(img)
    img = tf.image.resize_with_pad(img, taille, taille).numpy()
    return img


def trouver_base(modele):
    for couche in modele.layers:
        if 'mobilenet' in couche.name.lower():
            return couche
    raise ValueError("Couche de base MobileNet introuvable dans le modèle.")


def gradcam_heatmap(modele, img, taille):
    """Carte de chaleur des zones ayant influencé la décision (Selvaraju et al., 2017).

    NB : en Keras 3, on ne peut pas mélanger les graphes du modèle externe et de
    la base MobileNetV2. Le sous-modèle est donc construit dans le graphe de la
    base, puis la tête (normalisation + moyenne globale + Dense) est répliquée.
    """
    base = trouver_base(modele)
    modele_base = tf.keras.Model(base.input,
                                 [base.get_layer('out_relu').output, base.output])
    couche_dense = [c for c in modele.layers if isinstance(c, tf.keras.layers.Dense)][0]
    img_batch = tf.expand_dims(img, 0)
    x = img_batch / 127.5 - 1.0                     # = couche Rescaling du modèle
    with tf.GradientTape() as tape:
        activations, caracteristiques = modele_base(x)
        moyenne = tf.reduce_mean(caracteristiques, axis=[1, 2])   # = GlobalAveragePooling2D
        probabilites = tf.nn.softmax(couche_dense(moyenne))
        indice = int(probabilites[0].numpy().argmax())
        score = probabilites[0, indice]
    gradients = tape.gradient(score, activations)[0]
    poids = gradients.numpy().mean(axis=(0, 1))
    heatmap = tf.nn.relu((activations[0].numpy() * poids).sum(axis=-1)).numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap_grande = tf.image.resize(heatmap[..., None], [taille, taille]).numpy()[..., 0]
    return heatmap_grande, indice


# ---------------- Chargement du modèle ----------------
@st.cache_resource
def charger_modele():
    with open(os.path.join(DOSSIER_MODELE, 'config.json'), encoding='utf-8') as f:
        config = json.load(f)
    modele = tf.keras.models.load_model(os.path.join(DOSSIER_MODELE, 'modele_rd.keras'))
    return modele, config


# ---------------- Interface ----------------
st.title("👁️ Détection automatique de la rétinopathie diabétique")
st.caption("Projet M1 Intelligence Artificielle — Dakar Institute of Technology · "
           "MobileNetV2 (transfer learning) · Dataset APTOS 2019 · Explicabilité Grad-CAM")

modele_ok = os.path.exists(os.path.join(DOSSIER_MODELE, 'modele_rd.keras'))
if not modele_ok:
    st.warning("Modèle introuvable : placez **modele_rd.keras** et **config.json** dans le dossier "
               "**app/modele/** (générés par le notebook d'entraînement).")
    st.stop()

modele, config = charger_modele()
CLASSES = config['classes']
TAILLE = config['img_size']

with st.sidebar:
    st.header("À propos")
    st.markdown(
        "**But :** aider au dépistage de la rétinopathie diabétique (RD) dans les zones à "
        "faible accès à un ophtalmologiste.\n\n"
        "Le modèle classe chaque photographie du fond d'œil en 5 niveaux de sévérité "
        "(0 = aucune RD → 4 = RD proliférative).\n\n"
        "La carte **Grad-CAM** montre les zones qui ont influencé la décision."
    )
    st.info("⚠️ Outil d'aide au dépistage à but académique : il ne remplace en aucun cas "
            "l'avis d'un médecin ophtalmologiste.", icon="🩺")

image_uploadee = st.file_uploader("📤 Téléverser une photographie du fond d'œil (PNG/JPG)",
                                  type=['png', 'jpg', 'jpeg'])

if image_uploadee is not None:
    image_pil = Image.open(image_uploadee)
    img = pretraiter(image_pil, TAILLE)

    probabilites = modele.predict(tf.expand_dims(img, 0), verbose=0)[0]
    prediction = int(probabilites.argmax())
    confiance = probabilites[prediction]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image analysée")
        st.image(tf.cast(img, tf.uint8).numpy(), use_container_width=True)
        titre, conseil = CONSEILS[prediction]
        if prediction == 0:
            st.success(f"**{titre}** Confiance : {confiance:.0%}. {conseil}")
        elif prediction <= 2:
            st.warning(f"**{titre}** Confiance : {confiance:.0%}. {conseil}")
        else:
            st.error(f"**{titre}** Confiance : {confiance:.0%}. {conseil}")

    with col2:
        st.subheader("Probabilités par niveau de sévérité")
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probabilites)},
                     horizontal=True, height=320)

        st.subheader("Explication — Grad-CAM")
        heatmap, _ = gradcam_heatmap(modele, img, TAILLE)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(tf.cast(img, tf.uint8).numpy())
        ax.imshow(heatmap, cmap='jet', alpha=0.40)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Zones rouges/orange = régions les plus influentes pour la décision "
                   "(lésions potentielles : hémorragies, exsudats, micro-anévrismes...).")
else:
    st.info("👆 Téléversez une image de fond d'œil pour lancer l'analyse. "
            "Des exemples sont disponibles dans le dataset APTOS "
            "([kaggle.com](https://www.kaggle.com/c/aptos2019-blindness-detection)).")
