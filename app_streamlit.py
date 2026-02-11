import os
import textwrap
from functools import lru_cache

import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI


JYRO_URL = "https://jyrorealestate.com/"


@lru_cache(maxsize=1)
def fetch_site_text(url: str) -> str:
    """
    Fetch and clean the main text from the Jyro Real Estate website.
    Cached so it only runs once per session.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error while fetching the website: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Get visible text
    text = soup.get_text(separator="\n")
    # Basic cleanup
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    joined = "\n".join(non_empty)

    # Optional: truncate to keep prompts light
    return joined[:12000]


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it as an environment variable "
            "before running the app."
        )
    return OpenAI(api_key=api_key)


def call_jyro_assistant(user_message: str, site_context: str) -> str:
    client = get_openai_client()

    system_prompt = textwrap.dedent(
        f"""
        You are a real estate assistant for Jyro Real Estate, a company
        specializing in renting and selling properties in Tetouan and
        Northern Morocco, as presented on their official website `{JYRO_URL}`.

        Use the information below as your main reference about their
        services, property types, locations, and general positioning.

        If the user asks about topics outside this context (e.g. detailed
        legal/tax advice, highly technical questions), give a high-level
        answer and encourage them to contact the agency directly for
        official information.

        WEBSITE CONTEXT (cleaned extract):
        ---
        {site_context}
        ---

        Answer in a way that is:
        - clear and professional
        - friendly and reassuring
        - focused on helping the user understand how Jyro Real Estate
          can support them in finding, renting, or buying a property.
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
        page_title="Jyro Real Estate – Website Chatbot Demo",
        page_icon="🏠",
        layout="centered",
    )

    st.title("🏠 Jyro Real Estate – Website Chatbot Demo")
    st.write(
        "This is a **live demo** of an AI chatbot connected to the content of "
        "[Jyro Real Estate](https://jyrorealestate.com/). "
        "It is designed for a real estate manager who wants to see how an "
        "assistant could answer visitors' questions directly on the website."
    )

    with st.expander("What does this demo show?", expanded=True):
        st.markdown(
            "- **24/7 assistant for your website**: the chatbot can welcome visitors, "
            "answer common questions about rentals and sales, and guide them to the "
            "right type of property.\n"
            "- **Trained on your existing website**: for this demo, it reads the public "
            "content of `jyrorealestate.com` and uses it as its main knowledge source.\n"
            "- **No technical work for you**: the chatbot can be integrated on your site "
            "with a small code snippet or an embedded widget—your team does not need to "
            "manage the AI details.\n"
            "- **Goal**: show how a visitor could interact with your brand in a more "
            "human, conversational way, while staying consistent with your website content."
        )

    # Load site context once
    with st.spinner("Loading content from Jyro Real Estate website..."):
        site_text = fetch_site_text(JYRO_URL)

    if site_text.startswith("Error while fetching the website"):
        st.error(site_text)
        st.stop()

    # Chat UI without avatars/faces
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("Conversation")

    # Display history as plain text blocks
    for msg in st.session_state.messages:
        prefix = ":iphone: You: " if msg["role"] == "user" else ":key: Assistant: "
        st.markdown(f"{prefix} {msg['content']}")

    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Type a question a visitor might ask "
            "(for example: 'Do you have villas for sale in Tetouan?')"
        )
        submitted = st.form_submit_button("Send")

    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                answer = call_jyro_assistant(prompt, site_text)
            except Exception as e:
                answer = (
                    "An error occurred while calling the OpenAI API: "
                    f"{e}"
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()



