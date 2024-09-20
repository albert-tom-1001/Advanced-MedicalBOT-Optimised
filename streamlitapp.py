import streamlit as st
import os
import time
import re
import html
from collections import deque
from markdown import markdown
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import spacy

# Function to get environment variables
def get_env_variable(var_name):
    if 'STREAMLIT_RUNTIME_ENV' in os.environ:
        # We're running on Streamlit Cloud
        return st.secrets[var_name]
    else:
        # We're running locally
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(var_name)

# Set API keys
openai_api_key = get_env_variable("OPENAI_API_KEY")
groq_api_key = get_env_variable("GROQ_API_KEY")

# Set environment variables
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Constants
MAX_HISTORY_LENGTH = 5
MAX_TOKENS = 500
EMBED_STORE_DIR = "embed_store"

# Create a folder for storing embeddings if it doesn't exist
if not os.path.exists(EMBED_STORE_DIR):
    os.makedirs(EMBED_STORE_DIR)

# Set page config
st.set_page_config(
    page_title="MediChat AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Force light theme
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f0f4f8 !important;
        }
    </style>
""", unsafe_allow_html=True)




# Custom CSS
custom_css = """
<style>
.stApp {
    background-color: #f0f4f8;
    color: #262730;
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.st-bw {
    background-color: #ffffff;
}
.st-eb {
    background-color: #5c9bbf;
    color: white;
}
.st-eb:hover {
    background-color: #4a7d9d;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #e6f3ff;
    color: #262730;
}
.chat-message.bot {
    background-color: #f0f0f0;
    color: #262730;
}
.chat-icon {
    width: 50px;
    height: 50px;
    margin-right: 1rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}
.user .chat-icon {
    background-color: #4a7d9d;
    color: white;
}
.bot .chat-icon {
    background-color: #5c9bbf;
    color: white;
}
.chat-content {
    flex-grow: 1;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    color: #262730 !important;
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    caret-color: #262730 !important;
}
body, input, textarea, button {
    color: #262730 !important;
}
::placeholder {
    color: #999999 !important;
    opacity: 1 !important;
}
.stMarkdown {
    color: #262730;
}
h1, h2, h3, h4, h5, h6 {
    color: #262730;
}
p {
    color: #262730;
}
.chat-message p {
    margin-bottom: 5px;
}
.chat-message ul, .chat-message ol {
    margin-top: 5px;
    margin-bottom: 5px;
    padding-left: 20px;
}
.chat-message li {
    margin-bottom: 5px;
}
@keyframes blink {
    0% { opacity: 0; }
    50% { opacity: 1; }
    100% { opacity: 0; }
}
.blinking-cursor {
    animation: blink 1s step-end infinite;
    display: inline-block;
}
.generating-text {
    font-size: 1rem !important;
    white-space: pre-wrap;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', (event) => {
    const style = document.createElement('style');
    style.textContent = `
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            color: #262730 !important;
            background-color: #ffffff !important;
            border: 1px solid #cccccc !important;
            caret-color: #262730 !important;
        }
    `;
    document.head.appendChild(style);

    const updateInputStyles = () => {
        const inputs = document.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.style.color = '#262730';
            input.style.backgroundColor = '#ffffff';
        });
    };

    updateInputStyles();
    setInterval(updateInputStyles, 1000);
});
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)

def initialize_session_state():
    session_vars = [
        "vectors", "embeddings", "chain", "current_response",
        "conversation_history", "conversation_context", "typing_speed"
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    if "conversation_history" not in st.session_state or st.session_state.conversation_history is None:
        st.session_state.conversation_history = []
    
    if "conversation_context" not in st.session_state or st.session_state.conversation_context is None:
        st.session_state.conversation_context = {
            "topic": None,
            "entities": [],
            "user_preferences": {},
            "previous_queries": []
        }
    
    if "typing_speed" not in st.session_state:
        st.session_state.typing_speed = 0.05  # Default typing speed

# Initialize session state
initialize_session_state()

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", max_tokens=MAX_TOKENS)

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
"""
You are an advanced AI medical assistant designed to provide comprehensive medical information. Your primary function is to offer accurate, helpful, and context-aware responses while maintaining ethical standards and user safety.

Previous Chat History:
{chat_history}

Context from documents:
{context}

Current question: {question}

OPERATIONAL PROTOCOL:

1. DIRECT ANSWERING:
   - Start your response immediately with relevant information.

2. STRUCTURED AND CLEAR RESPONSES:
   - Use proper Markdown formatting to enhance readability.
   - Use '##' for main headings and '###' for subheadings.
   - Utilize bullet points ('-') or numbered lists ('1.', '2.', etc.) for steps, symptoms, or multiple pieces of advice.
   - Separate distinct topics or ideas into different paragraphs with a blank line between them.
   - For complex topics, use headings to organize information (e.g., "## Symptoms:", "## Treatment:", "## When to Seek Medical Help:").

3. INFORMATION RETRIEVAL AND SYNTHESIS:
   - Analyze the provided context, chat history, and current question.
   - Synthesize information from multiple sources if necessary.

4. CONTEXT-AWARE ANSWERING:
   - Ensure your response is relevant to the current question and previous context if applicable.

5. COMPREHENSIVE MEDICAL GUIDANCE:
   - For questions about symptoms, clearly list possible conditions and when to seek medical attention.
   - When discussing treatments or medications, structure the information logically (e.g., home remedies, over-the-counter options, prescription treatments).

Remember to use appropriate Markdown formatting throughout your response to enhance readability and understanding. Do not present all information in a single paragraph.

Proceed with your response following this protocol:
"""
)

# Filter out non-essential sections based on keywords
def filter_documents(documents):
    excluded_sections = ["author", "acknowledgments", "preface", "introduction", "foreword", "references"]
    return [doc for doc in documents if not any(keyword in doc.page_content.lower() for keyword in excluded_sections)]

# Automatically load or create vector store
@st.cache_resource
def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings()
        
        if os.path.exists(f"{EMBED_STORE_DIR}/faiss_index"):
            vectors = FAISS.load_local(f"{EMBED_STORE_DIR}/faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader("./Data/Medical_book.pdf")
            docs = loader.load()
            docs = filter_documents(docs)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, embeddings)
            vectors.save_local(f"{EMBED_STORE_DIR}/faiss_index")
        
        return vectors, embeddings
    except Exception as e:
        st.error(f"An error occurred during vector store loading: {str(e)}")
        return None, None

# Load vector store
vectors, embeddings = load_vector_store()

# Create the conversational retrieval chain
@st.cache_resource
def get_conversation_chain(_llm, _vectors):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vectors.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )

chain = get_conversation_chain(llm, vectors)

# Enhanced topic detection using SpaCy
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

def extract_medical_entities(question):
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "SYMPTOM", "BODY_PART", "MEDICAL_CONDITION", "MEDICATION"]]
    return entities if entities else None

# Improved query preprocessing
def preprocess_query(query):
    general_queries = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what are you", "what can you do", "who made you",
        "what is your purpose", "how does this work"
    ]
    
    query_lower = query.lower()
    
    if any(gen_q in query_lower for gen_q in general_queries):
        return "general", query
    
    medical_entities = extract_medical_entities(query)
    if medical_entities:
        return "medical", query
    
    return "unknown", query

# Handle general queries
def handle_general_query(query):
    responses = {
        "hello": "Hello! I'm MediChat AI, your advanced medical assistant. How can I help you with health-related questions today?",
        "how are you": "As an AI, I'm always ready to assist. How can I help with your medical questions or concerns?",
        "what can you do": "I can provide information on medical conditions, symptoms, treatments, and general health advice. What specific topic would you like to explore?",
        "who made you": "I was developed by a team of AI specialists and medical professionals. How may I assist you with health-related inquiries?",
        "what is your purpose": "My purpose is to provide accurate medical information and guide users towards appropriate healthcare resources. What health topic shall we discuss?",
        "how does this work": "You can ask me about any medical condition, symptom, treatment, or health-related topic. I'll provide information based on my training. What health concern would you like to explore?"
    }
    
    for key, response in responses.items():
        if key in query.lower():
            return response
    
    return "I'm here to assist with medical questions and health concerns. What specific medical topic would you like to discuss?"

def format_response_realtime(text):
    # Split the text into sections based on headers
    sections = re.split(r'\n(?=#+\s)', text)
    
    formatted_sections = []
    for section in sections:
        # Format headers
        section = re.sub(r'^(#+)\s*(.*?)$', r'\1 \2\n', section, flags=re.MULTILINE)
        
        # Format lists
        section = re.sub(r'^(\s*[-*])\s*(.*?)$', r'\1 \2\n', section, flags=re.MULTILINE)
        section = re.sub(r'^(\s*\d+\.)\s*(.*?)$', r'\1 \2\n', section, flags=re.MULTILINE)
        
        # Ensure paragraphs are separated
        paragraphs = re.split(r'\n{2,}', section)
        formatted_paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs]
        section = '\n\n'.join(formatted_paragraphs)
        
        formatted_sections.append(section.strip())
    
    # Join sections with proper spacing
    formatted_text = '\n\n'.join(formatted_sections)
    
    # Convert Markdown to HTML
    html_content = markdown(formatted_text)
    
    return html_content


def process_query(query):
    query_type, processed_query = preprocess_query(query)

    if query_type == "general":
        response = handle_general_query(processed_query)
        formatted_response = format_response_realtime(response)
        return formatted_response, []

    medical_entities = extract_medical_entities(processed_query) or []
    
    # Update conversation context
    st.session_state.conversation_context.setdefault("previous_queries", [])[-MAX_HISTORY_LENGTH:].append(processed_query)
    
    if medical_entities:
        st.session_state.conversation_context.setdefault("entities", [])
        st.session_state.conversation_context["entities"] = list(set(st.session_state.conversation_context["entities"] + medical_entities))[:MAX_HISTORY_LENGTH]
        st.session_state.conversation_context["topic"] = medical_entities[0]

    # Use only recent context
    recent_queries = st.session_state.conversation_history[-MAX_HISTORY_LENGTH:]
    context_info = (
        f"Recent queries: {', '.join(recent_queries)}. "
        f"Current topic: {st.session_state.conversation_context.get('topic', 'Not set')}. "
        f"Related entities: {', '.join(st.session_state.conversation_context.get('entities', [])[:5])}"
    )
    
    # Optimize the query
    optimized_query = f"{context_info}\nQuestion: {processed_query}"
    
    try:
        response = chain({"question": optimized_query})
        
        formatted_response = format_response_realtime(response['answer'])
        
        disclaimer = "\n\n_Disclaimer: Consult a healthcare professional for personalized medical advice._"
        final_response = f"{formatted_response}{disclaimer}"
        
        # Update conversation history
        st.session_state.conversation_history.append(processed_query)
        st.session_state.conversation_history = st.session_state.conversation_history[-MAX_HISTORY_LENGTH:]
        
        return final_response, response.get('source_documents', [])
    except Exception as e:
        error_message = f"An error occurred: {str(e)}. Please try rephrasing your question or starting a new conversation."
        return error_message, []

# User input validation
def validate_user_input(user_input):
    if not user_input.strip():
        return False, "Please enter a valid question or command."
    if len(user_input) > 500:
        return False, "Your input is too long. Please limit your question to 500 characters."
    return True, ""

# Streamlit UI
st.markdown("<h1 style='color: #262730;'>🏥 MediChat AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #262730;'>Your Advanced Medical Assistant</h3>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Typing speed control
typing_speed = st.sidebar.slider(
    "Adjust typing speed",
    min_value=0.01,
    max_value=0.1,
    value=st.session_state.typing_speed,
    step=0.01,
    key="typing_speed_slider"
)
st.session_state.typing_speed = typing_speed

# React to user input
if prompt := st.chat_input("Ask your medical question here"):
    # Validate user input
    is_valid, error_message = validate_user_input(prompt)
    if not is_valid:
        st.error(error_message)
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(f"<div style='color: #262730;'>{prompt}</div>", unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with adjustable delay
            with st.spinner("MediChat AI is thinking..."):
                try:
                    response, source_documents = process_query(prompt)
                    words = response.split()
                    for i, word in enumerate(words):
                        full_response += word + " "
                        if i % 5 == 0 or i == len(words) - 1:  # Update every 5 words or at the end
                            time.sleep(st.session_state.typing_speed * 5)
                            # Use custom CSS class for generating text and apply real-time formatting
                            formatted_response = format_response_realtime(full_response)
                            message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                    
                    # Final display of the full formatted response
                    final_formatted_response = format_response_realtime(full_response)
                    message_placeholder.markdown(final_formatted_response, unsafe_allow_html=True)
                    
                    # Display source documents if available
                    if source_documents:
                        with st.expander("Source Documents"):
                            for doc in source_documents[:3]:
                                st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
                    
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}. Please try rephrasing your question or starting a new conversation."
                    message_placeholder.error(error_message)
                    final_formatted_response = f"<div class='error-message'>{error_message}</div>"

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_formatted_response})

# Sidebar with app information
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/flat-hand-drawn-hospital-reception-scene_52683-54613.jpg?w=1380&t=st=1691881051~exp=1691881651~hmac=e8ef2148b778d3937b7410ff3a72439b2fb46496da5d37463fb4d374bc7baa1c", use_column_width=True)
    st.title("About MediChat AI")
    st.info(
        "MediChat AI is an advanced medical assistant powered by artificial intelligence. "
        "It provides information on various medical topics, symptoms, and general health advice. "
        "Always consult with a healthcare professional for personalized medical advice and treatment."
    )
    st.warning(
        "⚠️ Disclaimer: MediChat AI is for informational purposes only and should not be considered as a substitute for professional medical advice, diagnosis, or treatment."
    )