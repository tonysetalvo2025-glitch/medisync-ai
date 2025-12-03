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

# --- VISUAL "CLINICAL CLEAN" (CSS Hospitalar) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Roboto:wght@400;500&display=swap');
    
    /* Ambiente Estéril/Clean */
    .stApp { background-color: #f8f9fa; font-family: 'Lato', sans-serif; }
    
    /* Cabeçalhos */
    h1, h2, h3 { font-family: 'Roboto', sans-serif !important; color: #2d3436 !important; font-weight: 700 !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1e4e8; }
    
    /* Botões (Verde Saúde) */
    .stButton > button {
        background-color: #00b894; color: white !important; border: none;
        border-radius: 25px; padding: 0.6rem 1.2rem; font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 184, 148, 0.2); width: 100%;
    }
    .stButton > button:hover { background-color: #00a884; transform: translateY(-1px); }

    /* Chat (Balões) */
    [data-testid="stChatMessage"] { background-color: #ffffff; border: 1px solid #dfe6e9; border-radius: 15px; }
    
    /* Avatar da IA */
    [data-testid="stChatMessage"] [data-testid="stImage"] { background-color: #e3fdfd; border: 2px solid #00b894; }
</style>
""", unsafe_allow_html=True)

# 3. Autenticação
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ SISTEMA OFF-LINE: Configure a chave API nos Secrets.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# 4. Protocolos de IA (Prompts)
PROMPT_PROFISSIONAL = (
    "ATUE COMO: Especialista Clínico Multidisciplinar (Enfermagem/Medicina/Psicologia).\n"
    "CONTEXTO: Análise de prontuários, artigos científicos e exames.\n"
    "DIRETRIZES:\n"
    "1. Use terminologia técnica padrão (CID-10, DSM-5, NANDA, Terminologia Cirúrgica).\n"
    "2. Seja direto, focado em diagnóstico diferencial, farmacologia e conduta clínica.\n"
    "3. Cite valores de referência e evidências científicas encontradas no texto.\n"
    "---------------------\n"
    "DADOS CLÍNICOS: {context_str}\n"
    "QUERY PROFISSIONAL: {query_str}\n"
    "PARECER TÉCNICO:"
)

PROMPT_PACIENTE = (
    "ATUE COMO: Um Profissional de Saúde Humanizado e Empático.\n"
    "OBJETIVO: Explicar saúde de forma simples, sem causar pânico.\n"
    "DIRETRIZES:\n"
    "1. Traduza termos técnicos para linguagem do dia a dia.\n"
    "2. Foque no cuidado, prevenção e bem-estar.\n"
    "3. Seja acolhedor. Se algo for grave, oriente buscar ajuda presencial com calma.\n"
    "4. Use listas ou tópicos para facilitar a leitura.\n"
    "---------------------\n"
    "INFORMAÇÕES: {context_str}\n"
    "PERGUNTA DO PACIENTE: {query_str}\n"
    "RESPOSTA ACOLHEDORA:"
)

# 5. Carregar Motor
@st.cache_resource
def carregar_sistema():
    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return True

carregar_sistema()

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
    
    st.info("Formatos aceitos: Prontuários, Exames (PDF/TXT), Bulas, Artigos.")
    uploaded_files = st.file_uploader("Arquivo Médico", accept_multiple_files=True)
    
    if uploaded_files and st.button("ANALISAR DADOS"):
        with st.spinner("Processando dados vitais..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        path = os.path.join(temp_dir, uploaded_file.name)
                        with open(path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    documents = SimpleDirectoryReader(temp_dir).load_data()
                    st.session_state.index = VectorStoreIndex.from_documents(documents)
                    st.session_state.loaded = True
                    logger.info(f"UPLOAD: {len(uploaded_files)} docs médicos.")
                st.success("✅ Prontuário Indexado.")
            except Exception as e:
                st.error(f"Erro: {e}")

# 7. Chat
if "messages" not in st.session_state: st.session_state.messages = []

st.title("Prontuário Inteligente")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Digite a dúvida clínica ou queixa..."):
    if not st.session_state.get("loaded"):
        st.warning("⚠️ Por favor, anexe o caso clínico na barra lateral.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analisando evidências..."):
            try:
                template = PromptTemplate(PROMPT_PROFISSIONAL if perfil == "PROFISSIONAL DE SAÚDE" else PROMPT_PACIENTE)
                engine = st.session_state.index.as_query_engine(text_qa_template=template, similarity_top_k=5)
                
                response = engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                logger.info(f"CONSULTA [{perfil}]: Respondida.")
            except Exception as e:
                st.error("Erro na análise clínica.")