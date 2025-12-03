import streamlit as st
import os
import tempfile
import nest_asyncio
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- LOGGING CLÍNICO ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MediSync_Log")

# 1. Configuração
nest_asyncio.apply()
st.set_page_config(page_title="MediSync AI - Saúde Integrada", page_icon="🏥", layout="wide")

# --- VISUAL "CLINICAL CLEAN" (CORRIGIDO) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Roboto:wght@400;500;700&display=swap');

    /* 1. FORÇAR TEMA CLARO (Variáveis Globais) */
    :root {
        --primary-color: #00b894;
        --background-color: #f4f7f6;
        --secondary-background-color: #ffffff;
        --text-color: #2d3436;
        --font: 'Lato', sans-serif;
    }

    /* 2. APLICAÇÃO GERAL */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Lato', sans-serif;
    }

    /* Força a cor do texto para TODOS os elementos para evitar "texto invisível" */
    .stApp, .stApp p, .stApp label, .stApp div, .stApp span, .stMarkdown, .stMarkdown p {
        color: #2d3436 !important;
    }

    /* 3. BARRA LATERAL */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dfe6e9;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00b894 !important; /* Verde no título da sidebar */
    }

    /* 4. TÍTULOS PRINCIPAIS */
    h1, h2, h3, h4 {
        font-family: 'Roboto', sans-serif !important;
        color: #2d3436 !important;
        font-weight: 700 !important;
    }

    /* 5. BOTÕES (Verde Saúde) */
    .stButton > button {
        background-color: #00b894;
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 184, 148, 0.2);
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #00a884;
        box-shadow: 0 6px 8px rgba(0, 184, 148, 0.3);
        transform: translateY(-2px);
        color: white !important; /* Garante texto branco no hover */
    }
    .stButton > button p {
        color: white !important; /* Força texto branco dentro do botão */
    }

    /* 6. CHAT E INPUTS */
    .stChatInput textarea {
        background-color: #ffffff !important;
        border: 1px solid #b2bec3 !important;
        color: #2d3436 !important;
    }
    
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Balão do Usuário (Cinza Claro) */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #dfe6e9;
    }
    
    /* Avatar */
    [data-testid="stChatMessage"] [data-testid="stImage"] {
        background-color: #e3fdfd;
        border: 2px solid #00b894;
    }

    /* 7. CORREÇÃO DO UPLOAD (Remove o fundo preto) */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px dashed #00b894;
    }
    [data-testid="stFileUploader"] section {
        background-color: #f8f9fa !important; /* Fundo cinza claro dentro do uploader */
    }
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
        color: #636e72 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Autenticação Segura
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    # Fallback para evitar crash se não tiver secrets configurado localmente
    # Mas o ideal é usar o st.secrets no deploy
    st.warning("⚠️ Chave API não encontrada. Usando modo demonstração (pode falhar).")
os.environ["GROQ_API_KEY"] = api_key if api_key else ""

# 4. Protocolos (Prompts)
PROMPT_PROFISSIONAL = (
    "ATUE COMO: Especialista Clínico Sênior.\n"
    "CONTEXTO: Análise de documentação médica.\n"
    "DIRETRIZES:\n"
    "1. Terminologia técnica (CID-10, Farmacologia).\n"
    "2. Foco em conduta clínica e evidências.\n"
    "3. Seja direto e técnico.\n"
    "---------------------\n"
    "DADOS: {context_str}\n"
    "PERGUNTA: {query_str}\n"
    "PARECER TÉCNICO:"
)

PROMPT_PACIENTE = (
    "ATUE COMO: Profissional de Saúde Humanizado.\n"
    "MISSÃO: Explicar para o paciente de forma simples.\n"
    "DIRETRIZES:\n"
    "1. Linguagem acessível e empática.\n"
    "2. Foco no entendimento e tranquilidade.\n"
    "3. Sem jargões técnicos sem explicação.\n"
    "---------------------\n"
    "DADOS: {context_str}\n"
    "DÚVIDA: {query_str}\n"
    "RESPOSTA:"
)

# 5. Carregar IA
@st.cache_resource
def carregar_sistema():
    try:
        Settings.llm = Groq(model="llama-3.3-70b-versatile")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return True
    except Exception as e:
        st.error(f"Erro ao conectar com a IA: {e}")
        return False

if carregar_sistema():
    # 6. Sidebar
    with st.sidebar:
        st.title("🏥 MediSync AI")
        st.caption("Inteligência Clínica Avançada")
        st.markdown("---")
        
        perfil = st.radio(
            "MODO DE OPERAÇÃO:",
            ["PROFISSIONAL DE SAÚDE", "PACIENTE / FAMILIAR"],
            index=0
        )
        
        st.info("Formatos: Prontuários, Exames (PDF/TXT), Artigos.")
        uploaded_files = st.file_uploader("Arquivo Médico", accept_multiple_files=True)
        
        if uploaded_files and st.button("ANALISAR DADOS"):
            with st.spinner("Processando..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uploaded_file in uploaded_files:
                            path = os.path.join(temp_dir, uploaded_file.name)
                            with open(path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        documents = SimpleDirectoryReader(temp_dir).load_data()
                        st.session_state.index = VectorStoreIndex.from_documents(documents)
                        st.session_state.loaded = True
                        logger.info(f"UPLOAD: {len(uploaded_files)} arquivos.")
                    st.success("✅ Prontuário Indexado!")
                except Exception as e:
                    st.error(f"Erro: {e}")

    # 7. Chat Principal
    if "messages" not in st.session_state: st.session_state.messages = []

    st.title("Prontuário Inteligente")
    
    # Área de Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Digite a dúvida clínica..."):
        if not st.session_state.get("loaded"):
            st.warning("⚠️ Anexe o prontuário/exame na barra lateral primeiro.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        logger.info(f"PERGUNTA [{perfil}]: {prompt}")

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    template = PromptTemplate(PROMPT_PROFISSIONAL if perfil == "PROFISSIONAL DE SAÚDE" else PROMPT_PACIENTE)
                    engine = st.session_state.index.as_query_engine(text_qa_template=template, similarity_top_k=5)
                    
                    response = engine.query(prompt)
                    st.markdown(str(response))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error("Erro na análise.")
                    logger.error(f"ERRO: {e}")
