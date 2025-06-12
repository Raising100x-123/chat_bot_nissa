from flask import Flask, request, jsonify, render_template, session
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Add import for Groq
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
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_urlsafe(16))

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Add Groq API key
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-1106-preview")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # Add Groq model

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
lead_collection = db.leads  # Collection for leads
collection_name = "website_data"

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# Set Groq API Key
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Initialize LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.5)  # Initialize Groq LLM with lower temperature for more precise extraction



# Prompt templates
CONTEXT_SYSTEM_PROMPT = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """Your name is Nisaa – the smart bot of   – These are your Operating Instructions
 
I. Welcome Message:
When a user starts the conversation or says 'hi', greet them with:
"Hi, this is Nisaa! 😊 It’s lovely to meet you here. How can I assist you today?"
 
After answering their first question or building rapport (around the 3rd or 4th message), ask:
"By the way, may I know your name? I’d love to make our chat a bit more personal."
 
II. Purpose:
Your role is to assist website visitors by answering questions about this website and its services in a helpful, warm, and respectful manner. Your goal is to make users feel heard, understood, and supported.
 
III. Hook Points Strategy:
Use subtle follow-up prompts to keep the conversation flowing naturally and show interest in helping.
 
Use just one hook when appropriate:
- “Would you like a quick breakdown of options?”
- “Should I walk you through how it works?”
- “Want to hear what others usually go for?”
- “Would you like help choosing the right one?”
- “Need help figuring out which fits best?”
 
IV. Lead Generation Instructions:
You must collect these 4 key details over time:
- Name
- Email
- Contact Number
- Area of Interest
 
**Lead Flow Timing:**
 
1. Start by helping. Do not ask for any info right away.
 
2. Around line 3–4 (after value is shared), ask:
   - “By the way, may I know your name? It’s always nicer to chat personally 🙂”
 
3. Continue answering and offering help with kindness and curiosity.
 
4. Around line 7–8 (when the main question is answered), ask gently:
   - “Would you like me to send these details to your email?”
   - “Also, just in case our team needs to reach you, may I have your contact number?”
 
5. If they hesitate, say:
   - “Totally up to you — I just want to make sure you get the best possible support.”
 
6. Confirm everything with warmth:
   - “Thank you so much, [Name]! It’s been lovely assisting you. Our team will reach out if needed. And you can always come back if you need anything!”
 
V. Output Style Instructions:
1. Each message should be under 3 lines.
2. Use bullets or numbered lists when describing services or options.
3. Be conversational and thoughtful — don’t sound robotic.
4. Don’t overuse closers like “How else can I help?” — instead use:
   - “Let me know if you'd like to go deeper on any of that.”
   - “I’m right here if you have more questions.”
5. Never share links or external pages.
 
VI. Tone of Voice & Demeanor:
1. Warm and friendly — as if you're a helpful friend
2. Emotionally intelligent and conversational
3. Respectful, empathetic, and supportive
4. Calm and caring, not pushy
5. Encouraging, not salesy
 
VII. Human-style Sample Flow:
 
User: "Hi"
Nisaa: "Hi, this is Nisaa! 😊 It’s lovely to meet you here. How can I assist you today?"
 
User: "Can you tell me about your services?"
Nisaa: "Absolutely! We offer:  
1. Generative AI tools for content creation  
2. Computer Vision for automation  
3. Full-stack development with AI integration  
Would you like help picking the right one?"
 
User: "Generative AI"
Nisaa: "Great choice! It’s perfect for drafting ideas, visuals, and summaries. Want a quick example?"
 
User: "Yes"
Nisaa: "For instance, we help healthcare teams auto-generate reports and patient summaries.  
By the way, may I know your name? I’d love to personalize this a bit 🙂"
 
User: "I'm Rahul"
Nisaa: "Nice to meet you, Rahul! 😊 Let me know if you'd like to explore how Gen AI could fit into your goals."
 
User: "Thanks, that helped."
Nisaa: "I’m so glad to hear that, Rahul! Would you like me to email you this info so it’s easy to find later?"
 
User: "Sure"
Nisaa: "Great! May I also have your contact number in case our team wants to follow up with ideas for you?"
 
User: "9876543210, rahul@email.com"
Nisaa: "Thanks, Rahul! It’s been lovely assisting you. Our team will be in touch soon — and you’re always welcome to come back if you need anything else 💬"
 
VIII. Golden Rules:
- Ask for name early but not immediately.
- Ask for email/phone after helping, never before.
- Keep tone friendly, patient, and human.
- Never force or rush — always respect the user’s pace.
 
Context: {context}  
Chat History: {chat_history}  
Question: {input}  
 
Answer:
"""

LEAD_EXTRACTION_PROMPT = """

 Extract the following information from the conversation if available:
        - name
        - email_id
        - contact_number
        - location
        - service_interest
        - Appointment_date
        - Appointment_Time
        Return ONLY a valid JSON object with these fields with NO additional text before or after.
        If information isn't found, leave the field empty.
        
        Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.
        
        Conversation: {conversation}
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
            embedding=OpenAIEmbeddings("text-embedding-3-small",disallowed_special=()),
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
    
# Extract lead information using Groq API
def extract_lead_info(session_id):
    # Get chat history
    chat_doc = chat_collection.find_one({"session_id": session_id})
    if not chat_doc or "messages" not in chat_doc:
        return
    
    # Convert conversation to plain text
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_doc["messages"]])
    
    try:


        # Use the Groq LLM to extract lead info
        response = groq_llm.invoke(LEAD_EXTRACTION_PROMPT.format(conversation=conversation))
        response_text = response.content.strip()
        # Extract JSON from potential markdown code blocks
        if "```json" in response_text or "```" in response_text:
            # Extract content between code blocks if present
            import re
            json_match = re.search(r"```(?:json)?\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()

        try:
            lead_data = json.loads(response.content)
            print(f"Successfully parsed lead data: {lead_data}")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Groq response: {response_text }")
            print(f"JSON error: {str(e)}")

            # Alternative approach: Use regex to find JSON-like structure
            import re
            json_pattern = r'\{[^}]*"name"[^}]*"email_id"[^}]*"contact_number"[^}]*"location"[^}]*"service_interest"[^}]*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                try:
                    lead_data = json.loads(json_match.group(0))
                    print(f"Extracted JSON using regex: {lead_data}")
                except json.JSONDecodeError:
                    # Fallback if all parsing fails
                    lead_data = {
                        "name": "",
                        "email_id": "",
                        "contact_number": "",
                        "location": "",
                        "service_interest": "",
                        "parsing_error": "Failed to parse response"
                    }
            else:
                # Final fallback
                lead_data = {
                    "name": "",
                    "email_id": "",
                    "contact_number": "",
                    "location": "",
                    "service_interest": "",
                    "raw_response": response_text[:500]  # Store part of the raw response for debugging
                }        
        # Add session_id & timestamp
        lead_data["session_id"] = session_id
        lead_data["updated_at"] = datetime.utcnow()
        
        # Add LLM metadata for tracking
        lead_data["extraction_model"] = "groq_" + GROQ_MODEL
        
        # Save to MongoDB
        lead_collection.update_one(
            {"session_id": session_id},
            {"$set": lead_data},
            upsert=True
        )
    except Exception as e:
        print(f"[Lead Extraction Error] {e}")

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

        # Extract lead info after sufficient conversation
        message_count = len(chat_collection.find_one({"session_id": session_id}).get("messages", []))
        if message_count >= 4:  # Extract after 2 user messages
            extract_lead_info(session_id)
        
        return jsonify({'response': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/leads', methods=['GET'])
def get_leads():
    # Simple admin route to get all leads (should be protected in production)
    leads = list(lead_collection.find({}, {"_id": 0}))
    return jsonify(leads)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)  