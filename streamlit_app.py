import os
import time
import json
import logging
from typing import List, Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field, PositiveInt, validator

# =========================
# Configuração básica
# =========================
st.set_page_config(
    page_title="Meu ChatGPT • Streamlit",
    page_icon="🤖",
    layout="centered"
)

# Logger simples
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatgpt-app")

# =========================
# Segurança de Secrets
# =========================
def get_openai_api_key() -> Optional[str]:
    # Streamlit Cloud: defina em Settings → Secrets como:
    # OPENAI_API_KEY = "sk-..."
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

api_key = get_openai_api_key()
if not api_key:
    st.error("Defina OPENAI_API_KEY em Secrets antes de usar.")
    st.stop()

# SDK moderno da OpenAI
os.environ["OPENAI_API_KEY"] = api_key
from openai import OpenAI
client = OpenAI()

# =========================
# Modelos e validação
# =========================
AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1"
]

class AppSettings(BaseModel):
    model: Literal["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"] = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: PositiveInt = Field(default=1024)
    system_prompt: str = Field(
        default=(
            "Você é um assistente de IA útil, objetivo e seguro. "
            "Responda em português do Brasil, de forma clara e prática."
        )
    )

    @validator("max_tokens")
    def clamp_max_tokens(cls, v):
        return min(v, 4096)

# =========================
# Estado da sessão
# =========================
if "history" not in st.session_state:
    st.session_state.history = []   # lista de {role, content}
if "settings" not in st.session_state:
    st.session_state.settings = AppSettings().dict()

# =========================
# Sidebar de Configuração
# =========================
with st.sidebar:
    st.markdown("## Configurações")
    model = st.selectbox("Modelo", AVAILABLE_MODELS, index=0, key="model_select")
    temperature = st.slider("Temperatura", 0.0, 2.0, float(st.session_state.settings["temperature"]), 0.1)
    max_tokens = st.number_input("Máx tokens de saída", min_value=64, max_value=4096, value=int(st.session_state.settings["max_tokens"]), step=64)
    system_prompt = st.text_area("System Prompt", value=st.session_state.settings["system_prompt"], height=140)

    # Atualiza objeto de configuração validado
    try:
        cfg = AppSettings(
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens),
            system_prompt=system_prompt
        )
        st.session_state.settings = cfg.dict()
    except Exception as e:
        st.error(f"Configuração inválida: {e}")

    st.markdown("---")
    if st.button("Limpar histórico"):
        st.session_state.history = []
        st.success("Histórico limpo.")

# =========================
# Cabeçalho
# =========================
st.title("🤖 Meu ChatGPT com OpenAI API")
st.caption("Base pronta para seus projetos no Streamlit Cloud. Histórico por sessão e streaming ativado.")

# =========================
# Funções auxiliares
# =========================
def render_history(messages: List[dict]):
    for m in messages:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])

def stream_chat_completion(messages: List[dict], model: str, temperature: float, max_tokens: int, system_prompt: str):
    """
    Faz streaming de uma resposta usando a API Chat Completions.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Chamada com streaming
    stream = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    partial = ""
    placeholder = st.empty()
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            partial += delta.content
            placeholder.markdown(partial)
    return partial

# =========================
# Área de chat
# =========================
render_history(st.session_state.history)

user_input = st.chat_input("Digite sua mensagem...")
if user_input:
    # Adiciona prompt do usuário
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            s = st.session_state.settings
            reply = stream_chat_completion(
                messages=st.session_state.history,
                model=s["model"],
                temperature=s["temperature"],
                max_tokens=s["max_tokens"],
                system_prompt=s["system_prompt"]
            )
            # Salva resposta no histórico
            st.session_state.history.append({"role": "assistant", "content": reply})
        except Exception as e:
            logger.exception("Erro ao gerar resposta")
            st.error(f"Ocorreu um erro ao chamar o modelo: {e}")

# =========================
# Rodapé com diagnósticos
# =========================
with st.expander("Diagnóstico técnico"):
    s = st.session_state.settings
    st.json(
        {
            "model": s["model"],
            "temperature": s["temperature"],
            "max_tokens": s["max_tokens"],
            "history_len": len(st.session_state.history),
            "sdk_openai_version_hint": ">=1.47.0",
            "streamlit_cloud_ready": True
        }
    )

st.caption(
    "Dica: defina a variável OPENAI_API_KEY em Secrets. "
    "O app não armazena chaves no código."
)
