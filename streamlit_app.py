import os
import re
import json
import logging
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, timezone

import streamlit as st
from pydantic import BaseModel, Field, PositiveInt, validator

# =========================
# Configuração básica
# =========================
st.set_page_config(
    page_title="Meu ChatGPT + Web (Tavily) + Raspagem HTML",
    page_icon="🧠",
    layout="centered"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatgpt-web-scraper-app")

# =========================
# Secrets seguros (sem chaves hardcoded)
# =========================
def get_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name, None)
    except Exception:
        return None

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    st.error("Defina OPENAI_API_KEY em Secrets antes de usar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
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
    # Tavily
    use_tavily: bool = Field(default=True)
    tavily_depth: Literal["basic", "advanced"] = Field(default="basic")
    tavily_max_results: PositiveInt = Field(default=5)
    # Scraper
    inject_scrape: bool = Field(default=True)
    scrape_char_limit: PositiveInt = Field(default=8000)

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
if "scrape_context" not in st.session_state:
    st.session_state.scrape_context = ""   # último conteúdo raspado
if "scrape_url" not in st.session_state:
    st.session_state.scrape_url = ""       # última URL raspada

# =========================
# Sidebar de Configuração
# =========================
with st.sidebar:
    st.markdown("## Configurações")
    model = st.selectbox("Modelo", AVAILABLE_MODELS, index=0, key="model_select")
    temperature = st.slider("Temperatura", 0.0, 2.0, float(st.session_state.settings["temperature"]), 0.1)
    max_tokens = st.number_input(
        "Máx tokens de saída", min_value=64, max_value=4096,
        value=int(st.session_state.settings["max_tokens"]), step=64
    )
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.settings["system_prompt"],
        height=140
    )

    st.markdown("### Busca online (Tavily)")
    use_tavily = st.checkbox(
        "Ativar busca online",
        value=bool(st.session_state.settings.get("use_tavily", True))
    )
    tavily_depth = st.selectbox(
        "Profundidade", ["basic", "advanced"],
        index=0 if st.session_state.settings.get("tavily_depth") == "basic" else 1
    )
    tavily_max_results = st.slider(
        "Máx resultados", 1, 10,
        int(st.session_state.settings.get("tavily_max_results", 5))
    )

    st.markdown("### Raspagem de HTML (URL)")
    inject_scrape = st.checkbox(
        "Injetar conteúdo raspado na conversa",
        value=bool(st.session_state.settings.get("inject_scrape", True))
    )
    scrape_char_limit = st.slider(
        "Limite de caracteres do texto raspado",
        min_value=1000, max_value=20000, value=int(st.session_state.settings.get("scrape_char_limit", 8000)), step=500
    )

    try:
        cfg = AppSettings(
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens),
            system_prompt=system_prompt,
            use_tavily=use_tavily,
            tavily_depth=tavily_depth,  # type: ignore
            tavily_max_results=int(tavily_max_results),
            inject_scrape=inject_scrape,
            scrape_char_limit=int(scrape_char_limit)
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
st.title("🧠 Meu ChatGPT com Tavily + Raspagem HTML")
st.caption("Cole um link para raspar HTML. Se o site bloquear (403), usamos Tavily Extract como fallback.")

# =========================
# Utilidades: OpenAI + Tavily + Scraper
# =========================
def openai_stream_chat(messages: List[dict], model: str, temperature: float, max_tokens: int, system_prompt: str):
    full_messages = [{"role": "system", "content": system_prompt}] + messages
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

def tavily_search(query: str, depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY não definida em Secrets."}
    import requests
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": depth,
        "include_images": False,
        "include_answer": True,
        "max_results": int(max_results)
    }
    try:
        resp = requests.post(url, json=payload, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.exception("Erro Tavily (search)")
        return {"error": f"Erro ao consultar Tavily: {e}"}

def tavily_extract(url_to_fetch: str) -> Dict[str, Any]:
    """
    Fallback para extrair texto de uma URL usando Tavily Extract API.
    Retorna um dicionário com keys 'ok', 'title', 'content', 'error'.
    """
    if not TAVILY_API_KEY:
        return {"ok": False, "title": "", "content": "", "error": "TAVILY_API_KEY não definida."}
    import requests
    api = "https://api.tavily.com/extract"
    payload = {"api_key": TAVILY_API_KEY, "url": url_to_fetch}
    try:
        resp = requests.post(api, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json() or {}
        title = data.get("title") or ""
        content = data.get("content") or data.get("text") or ""
        if not content:
            return {"ok": False, "title": title, "content": "", "error": "Sem conteúdo retornado pela Tavily."}
        return {"ok": True, "title": title, "content": content, "error": None}
    except Exception as e:
        logger.exception("Erro Tavily (extract)")
        return {"ok": False, "title": "", "content": "", "error": f"Falha no Tavily Extract: {e}"}

def format_tavily_context(data: Dict[str, Any]) -> str:
    if "error" in data:
        return f"(Falha ao obter contexto da web: {data['error']})"
    answer = data.get("answer") or ""
    results = data.get("results", []) or []
    top = results[:5]
    lines = []
    for i, r in enumerate(top, start=1):
        title = r.get("title") or "Fonte"
        url = r.get("url") or ""
        snippet = (r.get("content") or "").strip()
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        lines.append(f"{i}. **{title}** — {snippet}\n   {url}")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"### Contexto de pesquisa web (Tavily) — {ts}\n"
    if answer:
        header += f"\n**Síntese:** {answer}\n"
    if lines:
        header += "\n**Fontes:**\n" + "\n".join(lines)
    else:
        header += "\n(Nenhuma fonte relevante retornada.)"
    return header

# -------- Scraper HTML --------
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def scrape_direct(url: str, user_agent: str = "Mozilla/5.0 (compatible; StreamlitScraper/1.0)") -> Dict[str, Any]:
    """
    Tenta baixar HTML diretamente. Retorna dict com ok/title/text/error/status_code.
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=45)
        content_type = resp.headers.get("Content-Type", "")
        status = resp.status_code
        if status >= 400:
            return {"ok": False, "title": "", "text": "", "error": f"HTTP {status}", "status_code": status}
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return {"ok": False, "title": "", "text": "", "error": f"Conteúdo não-HTML: {content_type}", "status_code": status}
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        title = (soup.title.string if soup.title else "") or ""
        text = normalize_whitespace(soup.get_text(separator="\n"))
        return {"ok": True, "title": title.strip(), "text": text, "error": None, "status_code": status}
    except Exception as e:
        logger.exception("Erro no download direto")
        return {"ok": False, "title": "", "text": "", "error": f"Falha ao baixar HTML: {e}", "status_code": None}

def scrape_url_to_text(url: str, char_limit: int = 8000) -> Dict[str, Any]:
    """
    Estratégia em camadas:
      1) Tenta raspagem direta (headers realistas).
      2) Se 401/403/406/429 ou erro → fallback para Tavily Extract (se chave existir).
    """
    if not url or not isinstance(url, str):
        return {"ok": False, "url": url, "title": "", "text": "", "error": "URL inválida."}

    # 1) Tentativa direta
    direct = scrape_direct(url)
    if direct.get("ok"):
        text = direct["text"]
        if len(text) > char_limit:
            text = text[:char_limit] + "\n\n[...conteúdo truncado…]"
        return {"ok": True, "url": url, "title": direct["title"], "text": text, "error": None}

    status = direct.get("status_code")
    hard_block = status in {401, 403, 406, 429}
    if not hard_block and TAVILY_API_KEY is None:
        # erro que não é bloqueio + sem Tavily → reporta direto
        return {"ok": False, "url": url, "title": "", "text": "", "error": direct.get("error") or "Falha desconhecida"}

    # 2) Fallback via Tavily Extract
    te = tavily_extract(url)
    if not te.get("ok"):
        return {"ok": False, "url": url, "title": te.get("title") or "", "text": "", "error": te.get("error") or direct.get("error")}

    text = normalize_whitespace(te["content"])
    if len(text) > char_limit:
        text = text[:char_limit] + "\n\n[...conteúdo truncado…]"
    return {"ok": True, "url": url, "title": te.get("title") or url, "text": text, "error": None}

def render_history(messages: List[dict]):
    for m in messages:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(m["content"])

def inject_web_context_if_enabled(user_msg: str) -> Optional[str]:
    s = st.session_state.settings
    if not s.get("use_tavily", True):
        return None
    data = tavily_search(
        query=user_msg,
        depth=s.get("tavily_depth", "basic"),
        max_results=int(s.get("tavily_max_results", 5))
    )
    return format_tavily_context(data)

# =========================
# Bloco: Raspagem de HTML
# =========================
st.subheader("Raspagem de HTML (cole uma URL)")
col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input(
        "URL a raspar",
        value=st.session_state.scrape_url,
        placeholder="https://exemplo.com/pagina"
    )
with col2:
    scrape_btn = st.button("Raspar", use_container_width=True)

if scrape_btn and url_input:
    st.session_state.scrape_url = url_input
    with st.spinner("Baixando e processando HTML..."):
        res = scrape_url_to_text(url_input, char_limit=st.session_state.settings["scrape_char_limit"])
    if not res.get("ok"):
        st.error(f"Falha na raspagem: {res.get('error')}")
    else:
        st.session_state.scrape_context = res["text"]
        st.success(f"Conteúdo raspado de: {res.get('title') or res.get('url')}")
        with st.expander("Prévia do texto raspado"):
            st.write(res["text"])

# =========================
# Área de chat
# =========================
st.markdown("---")
render_history(st.session_state.history)

user_input = st.chat_input("Digite sua mensagem...")
if user_input:
    # 1) Mensagem do usuário
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Opcional: injeta contexto do scraper
    s = st.session_state.settings
    if s.get("inject_scrape", True) and st.session_state.scrape_context:
        scraped_md = "### Contexto de página (raspado via URL)\n\n" + st.session_state.scrape_context
        with st.chat_message("assistant"):
            st.markdown(scraped_md)
        st.session_state.history.append({"role": "assistant", "content": scraped_md})

    # 3) Opcional: contexto via Tavily
    web_ctx_md = inject_web_context_if_enabled(user_input)
    if web_ctx_md:
        with st.chat_message("assistant"):
            st.markdown(web_ctx_md)
        st.session_state.history.append({"role": "assistant", "content": f"(Contexto externo adicionado da web)\n\n{web_ctx_md}"})

    # 4) Geração da resposta
    with st.chat_message("assistant"):
        try:
            reply = openai_stream_chat(
                messages=st.session_state.history,
                model=s["model"],
                temperature=s["temperature"],
                max_tokens=s["max_tokens"],
                system_prompt=s["system_prompt"]
            )
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
            "use_tavily": s["use_tavily"],
            "tavily_depth": s["tavily_depth"],
            "tavily_max_results": s["tavily_max_results"],
            "inject_scrape": s["inject_scrape"],
            "scrape_char_limit": s["scrape_char_limit"],
            "tem_scrape_context": bool(st.session_state.scrape_context),
            "streamlit_cloud_ready": True
        }
    )

st.caption(
    "Secrets necessários: OPENAI_API_KEY (obrigatório) e TAVILY_API_KEY (recomendado para fallback de raspagem e busca). "
    "Nenhuma chave é armazenada no código."
)
