import os
import textwrap
from functools import lru_cache
import io

import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from pypdf import PdfReader


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
    Create an OpenAI client using the OPENAI_KEY secret.
    """
    api_key = st.secrets["OPENAI_KEY"]
    if not api_key:
        raise RuntimeError(
            "OPENAI_KEY is not set in Streamlit secrets. Please configure it "
            "in the app's Settings → Secrets before running the app."
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


def extract_pdf_text(file) -> str:
    """
    Extract raw text from an uploaded PDF file.
    """
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    joined = "\n".join(texts)
    return joined[:20000]  # keep prompt size reasonable


def call_pdf_assistant(question: str, pdf_text: str) -> str:
    client = get_openai_client()

    system_prompt = textwrap.dedent(
        """
        You are an assistant that answers questions based strictly on the
        content of the following PDF document provided by the user.

        - If the answer is clearly present in the document, quote or
          summarise it in simple business English.
        - If the document does not contain the answer, say that the
          information is not in the document and respond with your best
          general guidance, clearly separating what comes from the PDF
          and what is general advice.
        """
    ).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"PDF DOCUMENT CONTENT (truncated):\n---\n{pdf_text}\n---",
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def render_website_chat_page():
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
    if "web_messages" not in st.session_state:
        st.session_state.web_messages = []

    st.subheader("Conversation")

    # Simple chat-style input: sending with Enter, no extra faces.
    # We handle the new message FIRST so it appears immediately in history.
    user_input = st.chat_input(
        "Type a question a visitor might ask "
        "(for example: 'Do you have villas for sale in Tetouan?')"
    )

    if user_input:
        st.session_state.web_messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            try:
                answer = call_jyro_assistant(user_input, site_text)
            except Exception as e:
                answer = (
                    "An error occurred while calling the OpenAI API: "
                    f"{e}"
                )

        st.session_state.web_messages.append({"role": "assistant", "content": answer})

    # Display history as plain text blocks (including the latest turn)
    for msg in st.session_state.web_messages:
        prefix = ":iphone: You: " if msg["role"] == "user" else ":key: Assistant: "
        st.markdown(f"{prefix} {msg['content']}")


def render_pdf_chat_page():
    st.title("📄 PDF Document Q&A Demo")
    st.write(
        "On this page, you can **upload a PDF document** (for example a contract, "
        "a brochure, or internal guidelines) and ask questions about its content. "
        "The assistant answers based primarily on what is inside the PDF."
    )

    # User uploads a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_source_file = uploaded_file
        pdf_source_name = uploaded_file.name
        pdf_bytes = uploaded_file.getvalue()
        if "pdf_text" not in st.session_state or st.session_state.get(
            "pdf_filename"
        ) != pdf_source_name:
            with st.spinner("Reading and indexing the PDF..."):
                # Use a fresh BytesIO for extraction to avoid pointer issues
                pdf_text = extract_pdf_text(io.BytesIO(pdf_bytes))
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_filename = pdf_source_name
                st.session_state.pdf_messages = []

        st.success(f"PDF loaded: **{pdf_source_name}**")

        # If the PDF has no extractable text, explain this clearly and stop here
        if not st.session_state.pdf_text or not st.session_state.pdf_text.strip():
            st.warning(
                "This PDF does not contain any readable text (it may be a scanned image "
                "or an empty document). The assistant cannot answer questions about it. "
                "Please upload a PDF that contains selectable text (for example, one "
                "exported directly from Word or another editor)."
            )
            return
        if "pdf_messages" not in st.session_state:
            st.session_state.pdf_messages = []

        st.subheader("Ask questions about this document")

        pdf_question = st.text_input(
            "Type your question about the uploaded PDF",
            key="pdf_user_input",
        )
        ask = st.button("Ask", key="pdf_ask_button")

        if ask and pdf_question:
            st.session_state.pdf_messages.append(
                {"role": "user", "content": pdf_question}
            )

            with st.spinner("Analyzing the document..."):
                try:
                    answer = call_pdf_assistant(pdf_question, st.session_state.pdf_text)
                except Exception as e:
                    answer = (
                        "An error occurred while calling the OpenAI API: "
                        f"{e}"
                    )

            st.session_state.pdf_messages.append(
                {"role": "assistant", "content": answer}
            )

        # Show full Q&A history including the latest turn
        for msg in st.session_state.pdf_messages:
            prefix = ":iphone: You: " if msg["role"] == "user" else ":key: Assistant: "
            st.markdown(f"{prefix} {msg['content']}")

    else:
        st.info("Upload a PDF file above to start asking questions about it.")


def main():
    st.set_page_config(
        page_title="Jyro Real Estate – Chatbot Demos",
        page_icon="🏠",
        layout="centered",
    )

    st.sidebar.title("Demo navigation")
    page = st.sidebar.radio(
        "Choose a demo",
        ["Website chatbot", "PDF document chatbot"],
    )

    if page == "Website chatbot":
        render_website_chat_page()
    else:
        render_pdf_chat_page()


if __name__ == "__main__":
    main()










