import os
import textwrap
from functools import lru_cache

import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI


@lru_cache(maxsize=16)
def fetch_site_text(url: str) -> str:
    """
    Récupère et nettoie le texte principal d'un site web.
    """
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        return f"Erreur lors de la récupération du site : {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Supprimer scripts et styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Texte visible
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    joined = "\n".join(non_empty)

    # On tronque pour garder un prompt raisonnable
    return joined[:15000]


def get_openai_client() -> OpenAI:
    """
    Crée un client OpenAI en utilisant la clé OPENAI_KEY (Streamlit secrets).
    """
    api_key = st.secrets["OPENAI_KEY"]
    if not api_key:
        raise RuntimeError(
            "La clé OPENAI_KEY n'est pas configurée dans les *Secrets* "
            "Streamlit. Ajoutez-la avant de lancer la démo."
        )
    return OpenAI(api_key=api_key)


def call_site_assistant(user_message: str, site_url: str, site_context: str) -> str:
    """
    Appelle le modèle OpenAI avec le contenu du site comme contexte.
    """
    client = get_openai_client()

    system_prompt = textwrap.dedent(
        f"""
        Tu es un assistant conversationnel pour le site suivant : {site_url}

        Tu disposes d'un extrait de texte issu de ce site, qui reflète :
        - son activité
        - son ton de communication
        - ses principaux services / produits

        Ton rôle :
        - répondre aux visiteurs comme si tu étais le chatbot officiel du site
        - rester cohérent avec le ton et le contenu du site
        - si une information n'apparaît pas clairement dans le texte, être prudent
          et le signaler (par exemple : "cette information n'est pas précisée sur
          le site, mais en général...").

        CONTEXTE DU SITE (extrait nettoyé) :
        ---
        {site_context}
        ---

        Réponds en français, de manière claire, professionnelle et accessible
        pour un visiteur non technique.
        """
    ).strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def main():
    st.set_page_config(
        page_title="QalamIA – Démo Chatbot pour votre site",
        page_icon="💬",
        layout="centered",
    )

    st.title("💬 Démo Chatbot QalamIA – Branchez l’IA sur votre site")
    st.write(
        "Cette démo est proposée par **QalamIA**. Elle montre, en conditions réelles, "
        "comment un chatbot peut être connecté au contenu **de votre propre site** "
        "pour accueillir vos visiteurs, répondre aux questions fréquentes et mettre "
        "en avant vos services, sans changer votre site actuel."
    )

    st.markdown(
        "Si vous souhaitez **mettre en place ce type de chatbot sur votre site**, "
        "vous pouvez contacter QalamIA directement au **+212 7 79 95 51 83**."
    )

    st.markdown("### 1. Indiquez l'URL de votre site")

    with st.form(key="site_form"):
        default_url = st.session_state.get("site_url", "")
        site_url = st.text_input(
            "URL du site à utiliser pour la démo",
            value=default_url,
            placeholder="https://votre-site.com",
        )
        load_clicked = st.form_submit_button("Charger le site")

    if load_clicked and site_url:
        with st.spinner("Récupération et analyse du contenu du site..."):
            site_text = fetch_site_text(site_url)
        if site_text.startswith("Erreur lors de la récupération"):
            st.error(site_text)
            return
        st.session_state.site_url = site_url
        st.session_state.site_text = site_text
        st.session_state.chat_messages = []

    # Vérifier qu'un site est chargé
    site_url = st.session_state.get("site_url")
    site_text = st.session_state.get("site_text")

    if not site_url or not site_text:
        st.info(
            "Renseignez l'URL de votre site ci-dessus, puis cliquez sur "
            "**« Charger le site »** pour voir comment un chatbot QalamIA pourrait "
            "se comporter directement avec le contenu de votre site."
        )
        return

    st.success(f"Site chargé pour la démo : **{site_url}**")

    with st.expander("Que montre cette démo QalamIA ?", expanded=True):
        st.markdown(
            "- **Pas de jargon technique** : vous partez simplement de l’URL de votre site.\n"
            "- **Le chatbot reprend votre ton et votre contenu** : il s’appuie sur le texte "
            "public de votre site pour répondre comme un « conseiller en ligne ».\n"
            "- **Cas d’usage concrets** : accueillir vos visiteurs 24/7, répondre aux mêmes "
            "questions que votre équipe reçoit par téléphone ou WhatsApp, orienter vers vos "
            "pages importantes (services, biens, formulaires de contact…).\n"
            "- Cette démo est pensée pour un **responsable non technique** qui veut "
            "voir rapidement le résultat avant de décider d’une mise en place réelle.\n\n"
            "👉 Si, en testant cette page, vous vous dites « c’est exactement ce qu’il me faut », "
            "appelez QalamIA au **+212 7 79 95 51 83** pour parler de votre projet."
        )

    st.markdown("### 2. Discutez avec le chatbot basé sur votre site")

    # Initialiser l'historique si besoin
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # IMPORTANT : gérer le nouveau message AVANT d'afficher l'historique
    user_input = st.chat_input(
        "Posez une question comme un visiteur (par ex. : « Quels services proposez-vous ? »)"
    )

    if user_input:
        # Ajouter le message utilisateur
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input}
        )

        # Appeler le modèle et ajouter la réponse
        with st.spinner("Le chatbot QalamIA rédige sa réponse..."):
            try:
                answer = call_site_assistant(user_input, site_url, site_text)
            except Exception as e:
                answer = (
                    "Une erreur est survenue lors de l'appel à l'API OpenAI : "
                    f"{e}"
                )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

    # Afficher l'historique complet (y compris le dernier échange)
    st.subheader("Conversation – Exemple de ce que verrait un visiteur sur votre site")
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            prefix = "📱 Vous :"
        else:
            # Emoji neutre pour représenter le site / le chatbot (pas d'êtres vivants)
            prefix = "🌐 Site (Chatbot QalamIA) :"
        st.markdown(f"{prefix} {msg['content']}")

    st.markdown("---")
    st.markdown(
        "Pour transformer cette démo en un **chatbot réellement intégré à votre site**, "
        "contactez **QalamIA** au **+212 7 79 95 51 83**. Nous adaptons le chatbot à "
        "votre activité, vos langues et vos objectifs (génération de contacts, "
        "qualification de demandes, support client, etc.)."
    )


if __name__ == "__main__":
    main()


