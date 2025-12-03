import streamlit as st
import os
import tempfile
import nest_asyncio
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- LOGGING (Monitoramento de Pacientes/Usuários) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MediSync_Tracker")

# 1. Configuração do Sistema
nest_asyncio.apply()

st.set_page_config(
    page_title="MediSync AI - Inteligência Clínica",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VISUAL "CLINICAL CLEAN" (CSS) ---
st.markdown("""
<style>
    /* Fonte Limpa e Moderna (Roboto/Lato) */
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Roboto:wght@400;500&display=swap');

    /* Fundo Geral Clean */
    .stApp {
        background-color: #f4f7f6;
        font-family: 'Lato', sans-serif;
    }

    /* Barra Lateral (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dfe6e9;
    }

    /* Títulos */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif !important;
        color: #2d3436 !important;
        font-weight: 700 !important;
    }
    
    /* Texto Comum */
    p, label, li, .stMarkdown {
        color: #636e72 !important;
        font-size: 16px;
    }

    /* Botões (Verde Médico / Confiança) */
    .stButton > button {
        background-color: #00b894;
        color: white !important;
        border: none;
        border-radius: 30px; /* Botões redondos */
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 184, 148, 0.2);
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #00a884;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 184, 148, 0.3);
    }

    /* Inputs de Chat */
    .stChatInput textarea {
        background-color: #ffffff !important;
        border: 1px solid #b2bec3 !important;
        border-radius: 20px;
        color: #2d3436 !important;
    }
    .stChatInput textarea:focus {
        border-color: #00b894 !important;
        box-shadow: 0 0 5px rgba(0, 184, 148, 0.5) !important;
    }

    /* Mensagens do Chat */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #dfe6e9;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Destaque para a IA (Avatar) */
    [data-testid="stChatMessage"] [data-testid="stImage"] {
        background-color: #e3fdfd;
        border: 2px solid #00b894;
    }

    /* Expander (Protocolos) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #00b894;
        border-radius: 8px;
        color: #00b894 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 3. Credenciais (API Key)
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    # Fallback para teste local se necessário, mas ideal é usar st.secrets
    # Coloque sua chave aqui se rodar localmente e não tiver configurado secrets
    api_key =  

os.environ["GROQ_API_KEY"] = api_key

# 4. Engenharia de Prompt (Coração do Sistema)
PROMPT_PROFISSIONAL = (
    "ATUE COMO: Especialista Clínico Multidisciplinar Sênior (Médico/Enfermeiro/Psicólogo).\n"
    "CONTEXTO: Análise de prontuários, exames e literatura médica.\n"
    "DIRETRIZES:\n"
    "1. Use terminologia técnica precisa (CID-10, DSM-5, Farmacologia).\n"
    "2. Cite referências exatas do texto fornecido.\n"
    "3. Seja objetivo, focado em conduta clínica, diagnóstico diferencial e protocolos.\n"
    "4. Mantenha tom acadêmico e formal.\n"
    "---------------------\n"
    "DOCUMENTOS CLÍNICOS: {context_str}\n"
    "---------------------\n"
    "SOLICITAÇÃO DO PROFISSIONAL: {query_str}\n"
    "PARECER TÉCNICO:"
)

PROMPT_PACIENTE = (
    "ATUE COMO: Um Profissional de Saúde Empático e Didático.\n"
    "MISSÃO: Traduzir 'mediquês' para linguagem simples e acolhedora.\n"
    "DIRETRIZES:\n"
    "1. Explique termos complexos com analogias simples.\n"
    "2. Foque no cuidado, bem-estar e instruções claras.\n"
    "3. Seja tranquilizador, mas realista baseando-se nos documentos.\n"
    "4. NUNCA faça diagnósticos definitivos sem ressaltar a necessidade de consulta presencial.\n"
    "---------------------\n"
    "INFORMAÇÕES DE SAÚDE: {context_str}\n"
    "---------------------\n"
    "DÚVIDA DO PACIENTE: {query_str}\n"
    "RESPOSTA ACOLHEDORA:"
)

# 5. Carregar Modelos
@st.cache_resource
def carregar_cerebro():
    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return True

with st.spinner("Esterilizando ambiente e carregando módulos de IA..."):
    carregar_cerebro()

# 6. Sidebar (Triagem)
with st.sidebar:
    st.markdown("### 🏥 TRIAGEM")
    st.info("Sistema de Apoio à Decisão Clínica")
    
    perfil = st.radio(
        "QUEM ESTÁ ACESSANDO?",
        ["PROFISSIONAL DE SAÚDE", "PACIENTE / FAMILIAR"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📁 PRONTUÁRIO / EXAMES")
    uploaded_files = st.file_uploader("Faça upload de PDFs ou TXT", accept_multiple_files=True)
    
    processar = st.button("🔍 ANALISAR DADOS CLÍNICOS")
    
    if st.button("🧹 NOVA CONSULTA"):
        st.session_state.messages = []
        st.rerun()

# 7. Processamento (RAG)
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if uploaded_files and processar:
    with st.spinner("Analisando parâmetros fisiológicos e texto..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                documents = SimpleDirectoryReader(temp_dir).load_data()
                index = VectorStoreIndex.from_documents(documents)
                
                st.session_state.index_base = index
                st.session_state.documents_loaded = True
                logger.info(f"TRIAGEM: {len(uploaded_files)} documentos médicos processados.")
                
            st.success("✅ Prontuário Digital Indexado.")
        except Exception as e:
            st.error("Erro na leitura dos exames.")
            logger.error(f"ERRO CLÍNICO: {e}")

# 8. Interface Principal
st.title("MediSync AI")
st.markdown("##### ASSISTENTE DE SAÚDE INTEGRADA")

# Área de Ajuda (Expander)
with st.expander("📋 PROTOCOLO DE USO (LEIA COM ATENÇÃO)"):
    st.markdown("""
    **Este sistema utiliza IA Avançada para leitura de documentos de saúde.**
    
    1. **Profissionais (Médicos, Enfermagem, Psicologia, Fono, etc):**
       - Receberão análises técnicas, sugestões de conduta baseadas em evidências e correlações clínicas.
    2. **Pacientes:**
       - Receberão explicações didáticas sobre laudos, bulas e orientações de cuidado.
    
    *⚠️ Importante: Esta ferramenta é um suporte. Jamais substitui o julgamento clínico ou consulta presencial.*
    """)

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite a queixa clínica ou dúvida..."):
    
    if not st.session_state.get("documents_loaded"):
        st.warning("⚠️ POR FAVOR: Anexe os documentos clínicos na barra lateral primeiro.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    logger.info(f"CONSULTA [{perfil}]: {prompt}")

    with st.chat_message("assistant"):
        with st.spinner("Gerando parecer clínico..."):
            try:
                if perfil == "PROFISSIONAL DE SAÚDE":
                    template = PromptTemplate(PROMPT_PROFISSIONAL)
                else:
                    template = PromptTemplate(PROMPT_PACIENTE)
                
                query_engine = st.session_state.index_base.as_query_engine(
                    text_qa_template=template,
                    similarity_top_k=5
                )
                
                response = query_engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                logger.info("PARECER FINALIZADO.")
                
            except Exception as e:
                st.error("Erro ao processar solicitação.")
                logger.error(f"FALHA NA RESPOSTA: {e}")
