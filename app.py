from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
import secrets
import json
import time
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime, timezone
import uuid
from flask_cors import CORS
from langchain_core.documents import Document
import urllib.parse
from playwright.sync_api import sync_playwright

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# MongoDB Atlas Search Index configuration
ATLAS_PUBLIC_KEY = os.getenv("ATLAS_PUBLIC_KEY")
ATLAS_PRIVATE_KEY = os.getenv("ATLAS_PRIVATE_KEY")
ATLAS_GROUP_ID = os.getenv("ATLAS_GROUP_ID")
ATLAS_CLUSTER_NAME = os.getenv("ATLAS_CLUSTER_NAME")
DATABASE_NAME = "Chatbot"
INDEX_NAME = "vector_index"

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client.Chatbot
chat_collection = db.chat_history
collection_name = "website_data"

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """
Your name is Nisaa - the smart bot of   - These are your Operating Instructions

I. Welcome Message:
When a user starts the conversation or says 'hi', greet them with:
"Hi, this is Nisaa! 😊 How can I assist you today?"

I. Purpose:
Your primary purpose is to assist website visitors and answer their questions about this website and our services.

II. Output Style Instructions:
1.  Every answer must be no more than 2–3 lines.
2. Be direct and to the point; avoid unnecessary details or filler.
3. When listing products, services, or important information, always display each item on a separate line, using a numbered or bulleted list. Never list items inline or in a single sentence.
   Example:
   1. Breeth
   2. Beamti
   3. Smellika
   4. Leadesh
4. Do not write long paragraphs or explanations unless specifically asked for details.
5. Do not use the phrase "What else can I help you with today?" except in the welcome message. For other responses, end with a short, friendly, and relevant phrase such as:
   - "Let me know if you want details on any of them."
   - "Feel free to ask about any feature."
   - "I'm here if you have more questions."
   Choose the closing that best fits the context of the user's question.
6. Strict Rule: Do not include any website links, URLs, or references to external pages in your responses. Only provide information directly from the context, without suggesting users visit another page.

III. Tone of Voice & Demeanor:
1. Professional but Conversational
Use clear, concise language that reflects your expertise—but avoid sounding overly formal or robotic. Speak like a real person having a professional conversation. Balance authority with friendliness.

2. Enthusiastic & Passionate
Show genuine excitement for what your company does and the value you provide. Make it clear that you care about helping your clients succeed and take pride in your solutions.

3. Empathetic & Understanding
Recognize that every customer or business has unique needs. Acknowledge their challenges and approach each interaction with curiosity and care. Avoid one-size-fits-all answers.

4. Helpful & Resourceful
Be genuinely helpful. Offer relevant suggestions, answer questions clearly, and guide people to useful resources, tools, or services your company provides. Aim to solve problems, not just sell.

5. Subtly Persuasive
Encourage next steps (like a demo, sign-up, or call) by focusing on the benefits and real-world outcomes. Be persuasive through value, not pressure.

6. Polite & Respectful (Never Rude or Argumentative)
Always maintain a respectful tone, even if someone is frustrated or if you don't have an immediate answer.

You are a knowledgeable assistant for a website. Answer questions based on the provided context.
If the information isn't available in the context, politely say you don't have enough information and offer to help with something else.
Keep your responses concise but informative. Be friendly and professional in your tone.

Context: {context}
Chat History: {chat_history}
Question: {input}

Answer:
"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chat history management
chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

# Atlas Search Index management functions
def create_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.atlas.2024-05-30+json'}
    data = {
        "collectionName": collection_name,
        "database": DATABASE_NAME,
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"}
            ]
        }
    }
    response = requests.post(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY), 
        data=json.dumps(data)
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create Atlas Search Index: {response.status_code}, Response: {response.text}")
    return response

def get_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.get(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

def delete_atlas_search_index():
    url = f"https://cloud.mongodb.com/api/atlas/v2/groups/{ATLAS_GROUP_ID}/clusters/{ATLAS_CLUSTER_NAME}/search/indexes/{DATABASE_NAME}/{collection_name}/{INDEX_NAME}"
    headers = {'Accept': 'application/vnd.atlas.2024-05-30+json'}
    response = requests.delete(
        url, 
        headers=headers, 
        auth=HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
    )
    return response

# Web scraping function
def scrape_website(url):
    """Scrape website content using Playwright with additional error handling for cloud environments"""
    print(f"Scraping: {url}")
    documents = []
    
    try:
        with sync_playwright() as p:
            # Install browsers if they don't exist
            try:
                print("Installing browser if needed...")
                import subprocess
                subprocess.run(["playwright", "install", "chromium"], check=True)
            except Exception as e:
                print(f"Browser installation error (non-critical): {e}")
            
            # Launch browser with more robust options
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-gpu',
                    '--disable-dev-shm-usage',
                    '--disable-setuid-sandbox',
                    '--no-sandbox',
                    '--disable-extensions',
                ]
            )
            
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            page = context.new_page()
            
            # Set longer timeouts for cloud environment
            page.set_default_timeout(120000)  # 2 minutes
            
            try:
                # Add more detailed logging
                print(f"Navigating to {url}...")
                response = page.goto(url, timeout=90000, wait_until="domcontentloaded")
                
                if not response:
                    print(f"No response received for {url}")
                    return documents
                    
                if not response.ok:
                    print(f"Error response for {url}: {response.status}")
                    return documents
                    
                # Try different strategies to wait for content
                try:
                    print("Waiting for network idle...")
                    page.wait_for_load_state("networkidle", timeout=30000)
                except Exception as e:
                    print(f"Network idle timeout: {e}, continuing anyway...")
                
                # Fallback wait strategy
                try:
                    page.wait_for_selector("body", timeout=10000)
                except Exception as e:
                    print(f"Selector timeout: {e}, continuing anyway...")
                
                # Get title and content
                title = page.title()
                print(f"Page title: {title}")
                
                # Get text content from main sections of the page
                main_content_selectors = [
                    "main", "article", ".content", "#content", 
                    ".main-content", "#main-content", "body"
                ]
                
                # Try to get content from specific sections first
                content = ""
                for selector in main_content_selectors:
                    try:
                        elements = page.query_selector_all(selector)
                        if elements:
                            for element in elements:
                                element_text = element.inner_text()
                                content += element_text + "\n\n"
                                print(f"Retrieved content from {selector}: {len(element_text)} characters")
                            break  # Stop once we've found content
                    except Exception as e:
                        print(f"Error getting content from {selector}: {e}")
                
                # If no content found from specific sections, get all body text
                if not content.strip():
                    try:
                        content = page.inner_text("body")
                        print(f"Retrieved content from body: {len(content)} characters")
                    except Exception as e:
                        print(f"Error getting body content: {e}")
                
                # Fallback to entire HTML if content extraction failed
                if not content.strip():
                    try:
                        content = page.content()
                        print(f"Fallback to HTML content: {len(content)} characters")
                    except Exception as e:
                        print(f"Error getting HTML content: {e}")
                
                # Check if we have any content
                if not content.strip():
                    print(f"Warning: No content extracted from {url}")
                    # Create minimal document with just the URL to prevent complete failure
                    minimal_doc = Document(
                        page_content=f"URL: {url}\nTitle: {title or 'Unknown'}\nNote: Content extraction failed.",
                        metadata={"source": url, "title": title or "Unknown", "extraction_failed": True}
                    )
                    documents.append(minimal_doc)
                    return documents
                
                # Get links for additional context
                links = []
                try:
                    links = page.eval_on_selector_all(
                        "a", 
                        "elements => elements.map(e => ({href: e.href, text: e.innerText}))"
                    )
                    print(f"Retrieved {len(links)} links")
                except Exception as e:
                    print(f"Error getting links: {e}")
                
                important_links = [f"{link['text']}: {link['href']}" for link in links if link.get('text', '').strip()]
                
                # Clean up the content (remove excessive whitespace)
                content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
                
                # Create main document
                main_doc = Document(
                    page_content=f"Title: {title}\n\nMain Content:\n{content}\n\nImportant Links:\n" + 
                              '\n'.join(important_links[:20]),
                    metadata={"source": url, "title": title}
                )
                documents.append(main_doc)
                print(f"Created document for {url} with {len(content)} characters")
                
                # Skip subpages for cloud deployment to avoid resource issues
                print("Skipping subpages for cloud deployment to avoid resource issues")
                
            except Exception as e:
                print(f"Error processing page {url}: {e}")
                # Create minimal document with just the URL to prevent complete failure
                minimal_doc = Document(
                    page_content=f"URL: {url}\nTitle: Unknown\nNote: Processing failed with error: {str(e)}",
                    metadata={"source": url, "title": "Unknown", "processing_failed": True}
                )
                documents.append(minimal_doc)
            
            browser.close()
            
    except Exception as e:
        print(f"Critical error scraping {url}: {e}")
        # Create minimal document with just the URL to prevent complete failure
        minimal_doc = Document(
            page_content=f"URL: {url}\nTitle: Unknown\nNote: Scraping failed with error: {str(e)}",
            metadata={"source": url, "title": "Unknown", "scraping_failed": True}
        )
        documents.append(minimal_doc)
        
    return documents

# Modified initialize_vector_store function to handle both URLs and extra text
def initialize_vector_store(urls, extra_text="", replace_existing=True):
    """Initialize vector store with scraped website content and extra text"""
    # Scrape websites
    all_documents = []
    for url in urls:
        try:
            documents = scrape_website(url)
            all_documents.extend(documents)
            print(f"Retrieved {len(documents)} documents from {url}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            # Create a placeholder document to avoid complete failure
            fallback_doc = Document(
                page_content=f"URL: {url}\nNote: Scraping failed with error: {str(e)}",
                metadata={"source": url, "title": "Scraping Failed", "error": str(e)}
            )
            all_documents.append(fallback_doc)
    
    # Add extra text as a document if provided
    if extra_text and extra_text.strip():
        extra_doc = Document(
            page_content=f"Additional Information:\n\n{extra_text.strip()}",
            metadata={"source": "user_input", "title": "Additional Information", "type": "extra_text"}
        )
        all_documents.append(extra_doc)
        print(f"Added extra text document with {len(extra_text)} characters")
    
    if not all_documents:
        # Fallback to a minimal document to prevent complete failure
        fallback_doc = Document(
            page_content="Fallback content for vector store initialization.\n\n" + 
                         "This document was created because no content could be scraped from the provided URLs.\n\n" +
                         f"URLs attempted: {', '.join(urls)}",
            metadata={"source": "fallback", "title": "Scraping Failed"}
        )
        all_documents.append(fallback_doc)
        print("Using fallback document since no content was scraped")
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_documents)
    
    print(f"Split into {len(final_documents)} final document chunks")
    
    # Only clear existing data if replace_existing is True
    if replace_existing:
        # Check and manage Atlas Search Index with better error handling
        try:
            response = get_atlas_search_index()
            if response.status_code == 200:
                print("Deleting existing Atlas Search Index...")
                delete_response = delete_atlas_search_index()
                if delete_response.status_code == 204:
                    # Wait for index deletion to complete with timeout
                    print("Waiting for index deletion to complete...")
                    start_time = time.time()
                    max_wait_time = 60  # seconds
                    while time.time() - start_time < max_wait_time:
                        check_response = get_atlas_search_index()
                        if check_response.status_code == 404:
                            break
                        time.sleep(5)
                else:
                    print(f"Warning: Failed to delete existing Atlas Search Index: {delete_response.status_code}, Response: {delete_response.text}")
            elif response.status_code != 404:
                print(f"Warning: Failed to check Atlas Search Index: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Error managing Atlas Search Index: {e}")
        
        # Clear existing collection
        try:
            db[collection_name].delete_many({})
            print(f"Cleared existing {collection_name} collection")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    # Store embeddings with better error handling
    try:
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=final_documents,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=db[collection_name],
            index_name=INDEX_NAME,
        )
        
        # Debug: Verify documents in collection
        doc_count = db[collection_name].count_documents({})
        print(f"Number of documents in {collection_name}: {doc_count}")
        if doc_count > 0:
            sample_doc = db[collection_name].find_one()
            print(f"Sample document structure (keys): {sample_doc.keys() if sample_doc else 'None'}")
        
        # Create new Atlas Search Index
        print("Creating new Atlas Search Index...")
        create_response = create_atlas_search_index()
        print(f"Atlas Search Index creation status: {create_response.status_code}")
        return vector_search
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise Exception(f"Failed to create vector store: {str(e)}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_playwright', methods=['GET'])
def check_playwright():
    """Endpoint to check if Playwright is working properly"""
    try:
        with sync_playwright() as p:
            # Try to launch browser
            browser = p.chromium.launch(
                headless=True,
                args=['--disable-gpu', '--disable-dev-shm-usage', '--no-sandbox']
            )
            
            # Create a page and navigate to a simple site
            page = browser.new_page()
            page.goto("https://example.com", timeout=30000)
            title = page.title()
            content = page.content()
            
            # Close browser
            browser.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Playwright is working correctly',
                'title': title,
                'content_length': len(content)
            }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Playwright error: {str(e)}'
        }), 500

@app.route('/generate_session', methods=['GET'])
def generate_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize chatbot - replaces existing data"""
    data = request.json
    urls = data.get('urls', [])
    extra_text = data.get('extra_text', '')
    
    if not urls and not extra_text.strip():
        return jsonify({'error': 'No URLs or extra text provided'}), 400
    
    try:
        # Initialize vector store with web content and extra text (replace existing)
        global vector_search
        vector_search = initialize_vector_store(urls, extra_text, replace_existing=True)
        return jsonify({
            'status': 'success', 
            'message': f'Chatbot initialized with {len(urls)} URLs and additional text'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_status', methods=['GET'])
def check_status():
    """Check if chatbot is already initialized with data"""
    try:
        # Check if there are documents in the collection
        doc_count = db[collection_name].count_documents({})
        
        # Check if Atlas Search Index exists
        response = get_atlas_search_index()
        index_exists = response.status_code == 200
        
        if doc_count > 0 and index_exists:
            # Get some sample metadata to show what sites are indexed
            sample_docs = list(db[collection_name].find({}, {"metadata.source": 1}).limit(10))
            sources = list(set([doc.get("metadata", {}).get("source", "Unknown") for doc in sample_docs]))
            
            # Filter out fallback and user_input sources for display
            display_sources = [s for s in sources if s not in ["fallback", "user_input"]]
            
            return jsonify({
                'initialized': True,
                'document_count': doc_count,
                'sources': display_sources[:5],  # Show first 5 sources
                'has_extra_text': "user_input" in sources,
                'message': f'Chatbot is ready with {doc_count} document chunks from {len(display_sources)} sources'
            }), 200
        else:
            return jsonify({
                'initialized': False,
                'document_count': 0,
                'sources': [],
                'has_extra_text': False,
                'message': 'No chatbot data found. Please initialize with website URLs.'
            }), 200
            
    except Exception as e:
        return jsonify({
            'initialized': False,
            'error': str(e),
            'message': 'Error checking chatbot status'
        }), 500

@app.route('/add_urls', methods=['POST'])
def add_urls():
    """Add new URLs to existing chatbot (append mode)"""
    data = request.json
    urls = data.get('urls', [])
    extra_text = data.get('extra_text', '')
    
    if not urls and not extra_text.strip():
        return jsonify({'error': 'No URLs or extra text provided'}), 400
    
    try:
        # Scrape new websites
        all_documents = []
        for url in urls:
            try:
                documents = scrape_website(url)
                all_documents.extend(documents)
                print(f"Retrieved {len(documents)} documents from {url}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                fallback_doc = Document(
                    page_content=f"URL: {url}\nNote: Scraping failed with error: {str(e)}",
                    metadata={"source": url, "title": "Scraping Failed", "error": str(e)}
                )
                all_documents.append(fallback_doc)
        
        # Add extra text as a document if provided
        if extra_text and extra_text.strip():
            extra_doc = Document(
                page_content=f"Additional Information:\n\n{extra_text.strip()}",
                metadata={"source": "user_input", "title": "Additional Information", "type": "extra_text"}
            )
            all_documents.append(extra_doc)
            print(f"Added extra text document with {len(extra_text)} characters")
        
        if not all_documents:
            return jsonify({'error': 'No content could be scraped from the provided URLs'}), 400
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(all_documents)
        
        # Add new documents to existing collection (no deletion)
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=final_documents,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=db[collection_name],
            index_name=INDEX_NAME,
        )
        
        # Update global vector_search variable
        globals()['vector_search'] = vector_search
        
        # Get updated stats
        doc_count = db[collection_name].count_documents({})
        
        return jsonify({
            'status': 'success', 
            'message': f'Added {len(final_documents)} new document chunks. Total documents: {doc_count}',
            'new_chunks': len(final_documents),
            'total_documents': doc_count
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Ensure vector store is initialized
    if 'vector_search' not in globals():
        try:
            # Try to recreate vector_search from existing MongoDB data
            doc_count = db[collection_name].count_documents({})
            if doc_count > 0:
                # Recreate vector search from existing collection
                vector_search = MongoDBAtlasVectorSearch(
                    collection=db[collection_name],
                    embedding=OpenAIEmbeddings(disallowed_special=()),
                    index_name=INDEX_NAME
                )
                globals()['vector_search'] = vector_search
            else:
                return jsonify({'error': 'Vector store not initialized. Please initialize with URLs first.'}), 400
        except Exception as e:
            return jsonify({'error': f'Error initializing vector store: {str(e)}'}), 500 
        
    # Get the vector_search from globals
    vector_search = globals().get('vector_search')

    if vector_search is None:
        return jsonify({'error': 'Vector store not available. Please initialize with URLs first.'}), 400
    
    try:
        # Create RAG pipeline
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.75})
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )

        # Get response from RAG
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response['answer']
        
        # Store message in MongoDB
        chat_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)}
            },
            upsert=True
        )
        
        return jsonify({'response': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)  