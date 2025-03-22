import streamlit as st
import concurrent.futures

from typing import Literal
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

from constants import AVAILABLE_MODELS
from api_call import get_base_model_response, get_finetuned_model_response

st.set_page_config(layout="wide")


# Initialize session state for chatbot conversations
def setup():
    st.session_state.finetuned_messages = []
    st.session_state.base_messages = []


def add_finetuned_message(role: Literal["user", "assistant"], content: str):
    st.session_state.finetuned_messages.append({"role": role, "content": content})


def add_base_message(role: Literal["user", "assistant"], content: str):
    st.session_state.base_messages.append({"role": role, "content": content})


if (
    "finetuned_messages" not in st.session_state
    and "base_messages" not in st.session_state
):
    setup()

with st.sidebar:
    if st.button("Reset chat"):
        setup()

    selected_model = st.selectbox("Finetuned models", AVAILABLE_MODELS.values())

# Layout for separate chatbot response threads
col1, col2 = st.columns(2)

with col1:

    st.header(f"Fine-tuned {selected_model} 🤖")
    for msg in st.session_state.finetuned_messages:
        with st.chat_message(name=msg["role"]):
            st.markdown(msg["content"])
    finetuned_user_message_markdown = st.empty()
    finetuned_assistant_generating_sparql_markdown = st.empty()
    finetuned_assistant_generated_sparql_markdown = st.empty()
    finetuned_assistant_final_response_markdown = st.empty()

with col2:
    st.header("Deepseek V3 🐬")
    for msg in st.session_state.base_messages:
        with st.chat_message(name=msg["role"]):
            st.markdown(msg["content"])
    base_user_message_markdown = st.empty()
    base_assistant_final_response_markdown = st.empty()


# User input (one input for both chatbots)
prompt = st.chat_input("Enter your messages")
if prompt:
    add_finetuned_message(role="user", content=prompt)
    add_base_message(role="user", content=prompt)
    with finetuned_user_message_markdown.chat_message("user"):
        st.markdown(prompt)
    with base_user_message_markdown.chat_message("user"):
        st.markdown(prompt)

    def on_generating_query_start(ctx):
        add_script_run_ctx(ctx=ctx)  # Bind the Streamlit session context
        text = "**Generating SparQL query**"
        with finetuned_assistant_generating_sparql_markdown.chat_message("assistant"):
            st.markdown(text)
        add_finetuned_message(role="assistant", content=text)

    def on_query_generated(ctx, generated_sparql_query: str):
        add_script_run_ctx(ctx=ctx)
        text = f"**Generated SparQL query**:\n\n{generated_sparql_query}"
        with finetuned_assistant_generated_sparql_markdown.chat_message("assistant"):
            st.markdown(text)
        add_finetuned_message(role="assistant", content=text)

    def on_finetuned_response_end(ctx, final_response: str):
        add_script_run_ctx(ctx=ctx)
        text = f"**Final response**:\n{final_response}"
        with finetuned_assistant_final_response_markdown.chat_message("assistant"):
            st.markdown(text)
        add_finetuned_message(role="assistant", content=text)

    def on_base_response_end(ctx, final_response: str):
        add_script_run_ctx(ctx=ctx)
        text = f"**Final response**:\n{final_response}"
        with base_assistant_final_response_markdown.chat_message("assistant"):
            st.markdown(text)
        add_base_message(role="assistant", content=text)

    ctx = get_script_run_ctx()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_finetuned = executor.submit(
            get_finetuned_model_response,
            question=prompt,
            model_name=selected_model,
            on_generating_query_start=lambda: on_generating_query_start(ctx),
            on_query_generated=lambda generated_sparql_query: on_query_generated(
                ctx, generated_sparql_query
            ),
            on_end=lambda final_response: on_finetuned_response_end(
                ctx, final_response
            ),
        )
        future_base = executor.submit(
            get_base_model_response,
            question=prompt,
            on_end=lambda final_response: on_base_response_end(ctx, final_response),
        )

        # Get results
        finetuned_response = future_finetuned.result()
        base_response = future_base.result()

    st.rerun()
