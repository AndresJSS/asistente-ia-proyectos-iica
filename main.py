import os
import logging
import asyncio
import functools
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ======================================================================
# 1. CONFIGURACIÓN INICIAL Y LOGGING
# ======================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# ======================================================================
# 2. DEFINICIÓN DE ESTRUCTURAS DE DATOS (PYDANTIC)
# ======================================================================

# Garantizar que el frontend siempre envíe la pregunta en el formato correcto
class ConsultaUsuario(BaseModel):
    pregunta: str = Field(..., json_schema_extra={"example": "¿Qué proyectos existen sobre recursos hídricos en el Caribe?"})
    unidad_filtro: str | None = Field(None, json_schema_extra={"example": "Representación del IICA en Bahamas"})

class RespuestaAgente(BaseModel):
    respuesta_ia: str
    fuentes_utilizadas: list[dict]

# ======================================================================
# 3. INICIALIZACIÓN DEL SERVIDOR Y CLIENTES DE IA
# ======================================================================

# Variables globales para los clientes (se inician una sola vez al prender el servidor)
pc_index = None
embeddings = None
llm_chat = None

def validar_entorno_api():
    """Patrón Fail-Fast para el servidor web."""
    variables_criticas = [
        "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
    ]
    faltantes = [var for var in variables_criticas if not os.getenv(var)]
    if faltantes:
        raise EnvironmentError(f"Faltan variables de entorno para FastAPI: {', '.join(faltantes)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor del contexto que maneja el encendido y apagado del servidor.
    Conecta Pinecone y Azure OpenAI para tenerlos listos en la memoria RAM.
    """
    global pc_index, embeddings, llm_chat

    try:
        validar_entorno_api()

        logging.info("Iniciando conexión a servicios en la nube...")

        # 1. Conectar a Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

        # 2. Conectar el vectorizador (para traducir la pregunta del usuario)
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
        )

        # 3. Conectar el Orquestador/Chat
        llm_chat = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment="gpt-4o-mini",
            openai_api_version="2024-12-01-preview",
            temperature=0.0  # Temperatura 0 para respuestas precisas, cero alucinaciones
        )
        logging.info("Todos los servicios en la nube conectados exitosamente.")
    
    except Exception as e:
        logging.critical(f"Error crítico al iniciar servicios en la nube: {e}")
        raise e
    
    # El servicor se queda "escuchando" peticiones web
    yield
    # --- FASE DE APAGADO (Después del yield) ---
    logging.info("Apagando el servidor API y liberando memoria RAM de los clientes...")

# Inicializamos FastAPI
app = FastAPI(
    title="API Agente RAG - SUGI PoC v2",
    description="Motor de búsqueda semántica con datos estructurados de SQL",
    version="2.0.0",
    lifespan=lifespan
)

# Configuración CORS para el Frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# 4. EL ORQUESTADOR RAG Y ENDPOINT PRINCIPAL
# ======================================================================

@app.post("/consultar", response_model=RespuestaAgente)

async def consultar_agente(consulta: ConsultaUsuario):
    """
    Recibe la pregunta del usuario, busca en la base vectorial (Pinecone)
    y genera una respuesta fundamentada utilizando Azure OpenAI.
    """

    # Verificar que los servicios hayan iniciado correctamente
    if not all([pc_index, embeddings, llm_chat]):
        raise HTTPException(status_code=500, detail="Los servicios de IA no están inicializados")
    
    try:
        logging.info(f"Procesando pregunta: '{consulta.pregunta}'")

        # 1. Vectorizar la pregunta del usuario
        vector_pregunta = await embeddings.aembed_query(consulta.pregunta)

        # 2. Aplicar filtro estricto por Unidad si el usuario lo solicita
        filtro = {"Unidad": {"$eq": consulta.unidad_filtro}} if consulta.unidad_filtro else None
        
        # 3. Búsqueda híbrida en Pinecone (Traer los 5 fragmentos más relevantes)
        consulta_pinecone = functools.partial(
            pc_index.query,
            vector=vector_pregunta,
            top_k=20,
            include_metadata=True,
            filter=filtro
        )
        resultados = await asyncio.to_thread(consulta_pinecone)

        # Filtro de relevancia (Umbral de similitud)
        # Un saludo o pregunta no relacionada tendrá una similitud baja
        UMBRAL_SIMILITUD = 0.40

        # --- DEBUG: Ver las calificaciones de Pinecone en la terminal ---
        print("\n--- RESULTADOS PINECONE ---")
        for match in resultados.matches:
            print(f"ID: {match.id} | Score: {round(match.score, 4)}")
        print("---------------------------\n")

        matches_validos = [match for match in resultados.matches if match.score >= UMBRAL_SIMILITUD]

        textos_contexto = []
        fuentes = []

        if matches_validos:
            for match in matches_validos:
                meta = match.metadata
                titulo = meta.get('Titulo_Proyecto', 'Desconocido')
                unidad = meta.get('Unidad', 'Desconocida')
                texto_fragmento = meta.get('text', '')
                
                bloque = f"[PROYECTO: {titulo} | UNIDAD: {unidad}]\n{texto_fragmento}"
                textos_contexto.append(bloque)

                fuentes.append({
                    "id": match.id,
                    "proyecto": titulo,
                    "pais": unidad,
                    "certeza": round(match.score * 100, 2)
                })
            contexto_unido = "\n\n---\n\n".join(textos_contexto)
        else:
            # Fallback Bilingüe: Evita el sesgo de idioma cuando no hay fuentes técnicas
            contexto_unido = "[NO TECHNICAL CONTEXT FOUND / NO SE ENCONTRÓ CONTEXTO TÉCNICO]. " \
                             "TIENES ESTRICTAMENTE PROHIBIDO ofrecer consejos, resúmenes generales o " \
                             "recomendaciones basadas en tu conocimiento previo. Limítate exclusivamente " \
                             "a indicar que no tienes información y pide amablemente que reformulen la pregunta. " \
                             "YOU MUST STRICTLY FOLLOW THE CRITICAL LANGUAGE RULE AND REPLY IN THE USER'S LANGUAGE."

        # 5. El System Prompt
        prompt_sistema = f"""
        You are the Institutional Knowledge Assistant for IICA (SUGI System). Your role is to synthesize technical project information.

        [STRICT RESPONSE RULES]
        1. ZERO INFERENCE & MANDATORY TRANSLATION: Extract answers ONLY from the <contexto>. However, you MUST TRANSLATE all extracted data (Project Names, Units, Details) into the exact language of the user's prompt. TRANSLATING THE ORIGINAL DATA IS REQUIRED AND IS NOT CONSIDERED HALLUCINATION.
        2. ADAPTIVE FORMATTING (CRITICAL LOGIC): You must adapt your format based on the user's request:
           - CONDITION A (Narrative Mode): ONLY IF the user explicitly asks for a "paragraph" (párrafo), "summary", or descriptive text, write in natural narrative paragraphs without bullets.
           - CONDITION B (List Mode - DEFAULT): If the user asks for "information", a "list", or does NOT explicitly specify a paragraph, you MUST use strict Markdown syntax EXACTLY like this:
             - **[Translate label: Project]:** [Translate Project Name]
               - **[Translate label: Unit]:** [Translate Unit Name]
               - **[Translate label: Year]:** [Year]
               - **[Translate label: Lessons / Practices / Results]:**
                 - [Translate detail 1]
                 - [Translate detail 2]

        3. EXHAUSTIVENESS: Include ALL distinct projects found in the <contexto>.
        4. TONE: Maintain a diplomatic tone. Replace "sovereignty" (or "soberanía") with "governance", "management", or "autonomy".

        <contexto>
        {contexto_unido}
        </contexto>
        
        [FINAL EXECUTION STEPS - READ CAREFULLY]
        Step 1: Identify the exact language of the user's prompt (e.g., English, Portuguese).
        Step 2: Extract the data from the <contexto> and TRANSLATE IT COMPLETELY to that language.
        Step 3: Output using CONDITION B (Markdown Bullets) UNLESS the user explicitly asked for a paragraph.
        CRITICAL: DO NOT output Spanish if the user asked in English. DO NOT output flat text without Markdown asterisks (**) and dashes (-).
        """

        # 6. Ejecutar el modelo de lenguaje
        mensajes = [
            SystemMessage(content=prompt_sistema),
            HumanMessage(content=consulta.pregunta)
        ]

        respuesta_llm = await llm_chat.ainvoke(mensajes)
        logging.info("Respuesta generada con éxito")

        return RespuestaAgente(
            respuesta_ia=respuesta_llm.content,
            fuentes_utilizadas=fuentes
        )
    
    except Exception as e:
        logging.error(f"Error durante el procesamiento de la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta con la IA.")