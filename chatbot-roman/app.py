import streamlit as st
from openai import OpenAI
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
try:
    import PyPDF2
except ImportError:
    st.error("Please install PyPDF2: pip install PyPDF2")

# ====================
# CONFIGURATION
# ====================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

YOUR_NAME = "Rome"
YOUR_ROLE = "role here"
BRAND_DESCRIPTION = """Brand description here"""

# Australian English Instructions
AUSTRALIAN_ENGLISH_INSTRUCTION = """
CRITICAL: Always use Australian English spelling and expressions:
- Use 'ise' not 'ize' (organise, realise, analyse, recognise)
- Use 'our' not 'or' (colour, favour, behaviour, honour)
- Use 'tre' not 'ter' (centre, metre, litre)
- Use 're' not 'er' where appropriate (fibre, calibre)
- Use 'll' in words like 'travelled', 'modelled', 'labelled'
- Use Australian vocabulary (rubbish not trash, holiday not vacation, mobile not cell phone, lift not elevator)
- Use Australian expressions naturally when appropriate
- Date format: DD/MM/YYYY
- Avoid Americanisms
"""

# ====================
# PERSISTENT STORAGE (JSON-based)
# ====================
KNOWLEDGE_FILE = "knowledge_base.json"

def load_knowledge_base():
    """Load knowledge base from JSON file"""
    if Path(KNOWLEDGE_FILE).exists():
        with open(KNOWLEDGE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_knowledge_base(knowledge_list):
    """Save knowledge base to JSON file"""
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump(knowledge_list, f, indent=2)

# ====================
# SEMANTIC SEARCH FUNCTIONS
# ====================
def get_embedding(text, client):
    """Get OpenAI embedding for text"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]  # Limit token size
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, knowledge_list, client, top_k=10):
    """Semantic search using embeddings"""
    query_embedding = get_embedding(query, client)
    if not query_embedding:
        return []
    
    scored_items = []
    for item in knowledge_list:
        # Generate embedding if not exists
        if 'embedding' not in item:
            item['embedding'] = get_embedding(item['content'], client)
            if not item['embedding']:
                continue
        
        similarity = cosine_similarity(query_embedding, item['embedding'])
        scored_items.append((similarity, item))
    
    scored_items.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in scored_items[:top_k]]

def search_knowledge(query, knowledge_list, client, use_semantic=True, top_k=10):
    """Enhanced search with semantic and keyword options"""
    
    # Check for category-specific queries first
    category_keywords = {
        'education': ['education', 'degree', 'degrees', 'qualification', 'qualifications', 
                     'university', 'college', 'diploma', 'certificate', 'studied', 'study',
                     'bachelor', 'masters', 'phd', 'certified', 'credentials', 'academic'],
        'experience': ['experience', 'work', 'job', 'role', 'position', 'career', 'worked'],
        'expertise': ['expertise', 'skills', 'expert', 'specialization', 'specialisation', 'capabilities'],
        'bio': ['bio', 'about', 'background', 'who', 'introduction'],
        'brand-voice': ['voice', 'style', 'tone', 'write', 'writing', 'communication'],
        'values': ['values', 'mission', 'believe', 'philosophy', 'principles'],
        'article': ['article', 'articles', 'wrote', 'written', 'published', 'post', 'posts']
    }
    
    query_lower = query.lower()
    matched_categories = []
    
    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_categories.append(category)
    
    # If category match found, get all items from those categories
    if matched_categories:
        category_items = []
        for item in knowledge_list:
            item_cat = item.get('category', '').lower()
            item_type = item.get('type', '').lower()
            if any(cat in item_cat or cat in item_type for cat in matched_categories):
                category_items.append(item)
        
        if category_items:
            return category_items
    
    # Use semantic search for general queries
    if use_semantic:
        return semantic_search(query, knowledge_list, client, top_k)
    
    # Fallback to keyword search
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'who', 'where', 
                  'when', 'why', 'tell', 'me', 'about', 'can', 'you', 'please'}
    
    query_words = set(word for word in query_lower.split() if word not in stop_words and len(word) > 2)
    
    scored_items = []
    for item in knowledge_list:
        content_lower = item['content'].lower()
        category_lower = item.get('category', '').lower()
        
        score = 0
        if query_lower in content_lower:
            score += 10
        
        for word in query_words:
            if word in category_lower:
                score += 5
            if word in content_lower:
                score += 2
        
        if score > 0:
            scored_items.append((score, item))
    
    scored_items.sort(reverse=True, key=lambda x: x[0])
    return [item for score, item in scored_items[:top_k]]

def find_related_articles(topic, knowledge_list, client, threshold=0.65):
    """Find articles related to a specific topic"""
    topic_embedding = get_embedding(topic, client)
    if not topic_embedding:
        return []
    
    related = []
    for item in knowledge_list:
        if item.get('type') == 'article':
            if 'embedding' not in item:
                item['embedding'] = get_embedding(item['content'], client)
            
            if item['embedding']:
                similarity = cosine_similarity(topic_embedding, item['embedding'])
                if similarity > threshold:
                    related.append({
                        'title': item.get('title', 'Untitled'),
                        'similarity': similarity,
                        'category': item.get('category', 'Uncategorised'),
                        'excerpt': item['content'][:200] + "..."
                    })
    
    return sorted(related, key=lambda x: x['similarity'], reverse=True)

# ====================
# PDF PROCESSING
# ====================
def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
        
        return full_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def auto_categorise_article(article_text, client):
    """AI-powered categorisation based on content"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Categorise this article into ONE of these categories:
                - Leadership & Strategy
                - Web3 & Blockchain
                - AI & Technology
                - Women in Tech
                - Startup Insights
                - Social Impact
                - Personal Brand
                - Industry Trends
                - Career Development
                - Innovation
                
                Return ONLY the category name, nothing else."""
            }, {
                "role": "user",
                "content": article_text[:2000]
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Auto-categorisation error: {str(e)}")
        return "Uncategorised"

# ====================
# BRAND VOICE FUNCTIONS
# ====================
def get_brand_voice_content(knowledge_list):
    """Get brand voice guide if it exists"""
    for item in knowledge_list:
        if item.get('type') == 'brand-voice' or item.get('category') == 'brand-voice':
            return item['content']
    return "No brand voice guide generated yet. Upload at least 3 articles to create one."

def get_article_summaries(knowledge_list):
    """Get list of all articles with brief summaries"""
    articles = [item for item in knowledge_list if item.get('type') == 'article']
    if not articles:
        return "No articles in library yet."
    
    summaries = []
    for article in articles[:20]:  # Limit to prevent token overflow
        title = article.get('title', 'Untitled')
        category = article.get('category', 'Uncategorised')
        excerpt = article['content'][:150] + "..."
        summaries.append(f"- '{title}' ({category}): {excerpt}")
    
    return "\n".join(summaries)

# ====================
# SETUP
# ====================
st.set_page_config(
    page_title=f"Chat with {YOUR_NAME}",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .article-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Load knowledge base
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base()

if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False

if 'use_semantic_search' not in st.session_state:
    st.session_state.use_semantic_search = True

# ====================
# SIDEBAR: KNOWLEDGE BASE MANAGEMENT
# ====================
with st.sidebar:
    st.header("📚 Publishing Assistant")
    
    # Password protection for admin features
    password_input = st.text_input("🔐 Admin Password:", type="password", key="admin_pass")
    
    # Show stats (visible to everyone)
    total_items = len(st.session_state.knowledge_base)
    articles = [item for item in st.session_state.knowledge_base if item.get('type') == 'article']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("Articles", len(articles))
    
    st.divider()
    
    # Check if password is correct
    if password_input == ADMIN_PASSWORD:
        st.success("✅ Admin access granted!")
        
        # ====================
        # PDF ARTICLE UPLOAD
        # ====================
        st.subheader("📄 Upload Article (PDF)")
        
        uploaded_pdf = st.file_uploader(
            "Upload article PDF",
            type=['pdf'],
            help="Upload your Medium or LinkedIn articles as PDFs"
        )
        
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                article_text = extract_text_from_pdf(uploaded_pdf)
            
            if article_text:
                # Show preview
                with st.expander("📖 Preview extracted text"):
                    st.text_area("", article_text[:500] + "...", height=150, disabled=True)
                
                # Article metadata
                article_title = st.text_input(
                    "Article title:",
                    value=uploaded_pdf.name.replace('.pdf', '').replace('_', ' ').title()
                )
                
                # Auto-suggest category
                if st.button("🤖 Auto-Categorise"):
                    with st.spinner("Analysing content..."):
                        suggested_cat = auto_categorise_article(article_text, client)
                        st.session_state.suggested_category = suggested_cat
                        st.success(f"Suggested: {suggested_cat}")
                
                category = st.text_input(
                    "Category:",
                    value=st.session_state.get('suggested_category', 'Article')
                )
                
                publication_date = st.date_input(
                    "Publication date (optional):",
                    value=datetime.now()
                )
                
                if st.button("💾 Save Article to Library", type="primary"):
                    with st.spinner("Creating embeddings..."):
                        embedding = get_embedding(article_text, client)
                    
                    # Add article to knowledge base
                    st.session_state.knowledge_base.append({
                        'type': 'article',
                        'title': article_title,
                        'category': category,
                        'content': article_text,
                        'date_added': datetime.now().isoformat(),
                        'publication_date': publication_date.isoformat(),
                        'embedding': embedding,
                        'word_count': len(article_text.split())
                    })
                    
                    save_knowledge_base(st.session_state.knowledge_base)
                    st.success(f"✅ '{article_title}' added to library!")
                    st.balloons()
                    
                    # Clear suggested category
                    if 'suggested_category' in st.session_state:
                        del st.session_state.suggested_category
        
        st.divider()
        
        # ====================
        # VIEW & MANAGE ARTICLES
        # ====================
        st.subheader("📚 Article Library")
        
        if articles:
            # Category filter
            all_categories = sorted(set(item.get('category', 'Uncategorised') for item in articles))
            view_category = st.selectbox(
                "Filter by category:",
                ["All Articles"] + all_categories
            )
            
            # Filter articles
            if view_category == "All Articles":
                filtered_articles = articles
            else:
                filtered_articles = [a for a in articles if a.get('category') == view_category]
            
            st.write(f"**Showing {len(filtered_articles)} articles**")
            
            # Display articles
            for idx, article in enumerate(filtered_articles[:10]):
                with st.expander(f"📝 {article.get('title', 'Untitled')} ({article.get('category', 'Uncategorised')})"):
                    st.caption(f"Added: {article.get('date_added', 'Unknown')[:10]}")
                    st.caption(f"Words: {article.get('word_count', 'Unknown')}")
                    
                    st.text_area(
                        "Content preview:",
                        value=article['content'][:300] + "...",
                        height=100,
                        disabled=True,
                        key=f"article_view_{idx}"
                    )
                    
                    # Find related articles
                    if st.button("🔗 Find Related Articles", key=f"related_{idx}"):
                        related = find_related_articles(
                            article['content'],
                            st.session_state.knowledge_base,
                            client,
                            threshold=0.6
                        )
                        
                        if related:
                            st.write("**Related articles:**")
                            for rel in related[:5]:
                                if rel['title'] != article.get('title'):
                                    st.write(f"- {rel['title']} ({rel['similarity']:.0%} similar)")
                        else:
                            st.info("No related articles found")
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"del_article_{idx}"):
                        # Find and remove from main knowledge base
                        for i, item in enumerate(st.session_state.knowledge_base):
                            if item.get('title') == article.get('title') and item.get('type') == 'article':
                                st.session_state.knowledge_base.pop(i)
                                break
                        save_knowledge_base(st.session_state.knowledge_base)
                        st.success("Article deleted")
                        st.rerun()
            
            if len(filtered_articles) > 10:
                st.info(f"Showing first 10 of {len(filtered_articles)} articles")
        else:
            st.info("📝 No articles yet. Upload PDFs above to build your library!")
        
        st.divider()
        
        # ====================
        # BRAND VOICE GENERATION
        # ====================
        st.subheader("🎨 Brand Voice")
        
        if len(articles) >= 3:
            if st.button("🎨 Generate Brand Voice Guide"):
                with st.spinner("Analysing your writing style across all articles..."):
                    all_article_text = "\n\n".join([
                        f"Article: {a.get('title', 'Untitled')}\n{a['content'][:2000]}"
                        for a in articles[:10]
                    ])
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "system",
                            "content": """Analyse this writer's style and create a detailed brand voice guide covering:
                            - Overall tone (professional/casual/authoritative/conversational)
                            - Sentence structure preferences (short/long, simple/complex)
                            - Common phrases and signature expressions
                            - Australian English usage patterns
                            - Paragraph rhythm and flow
                            - How they introduce and conclude ideas
                            - Use of examples, metaphors, and storytelling
                            - Technical vs accessible language balance
                            - Unique voice characteristics
                            
                            Format as a clear, actionable guide."""
                        }, {
                            "role": "user",
                            "content": all_article_text
                        }]
                    )
                    
                    voice_guide = response.choices[0].message.content
                    
                    # Save as special knowledge item
                    st.session_state.knowledge_base.append({
                        'type': 'brand-voice',
                        'category': 'brand-voice',
                        'content': voice_guide,
                        'date_created': datetime.now().isoformat()
                    })
                    
                    save_knowledge_base(st.session_state.knowledge_base)
                    st.success("✅ Brand voice guide created!")
                    
                    with st.expander("📖 View Brand Voice Guide"):
                        st.markdown(voice_guide)
        else:
            st.info(f"Upload {3 - len(articles)} more article(s) to generate brand voice guide")
        
        # Show existing brand voice
        current_voice = get_brand_voice_content(st.session_state.knowledge_base)
        if "No brand voice guide" not in current_voice:
            with st.expander("📖 Current Brand Voice Guide"):
                st.markdown(current_voice)
        
        st.divider()
        
        # ====================
        # TRADITIONAL KNOWLEDGE MANAGEMENT
        # ====================
        st.subheader("📝 Add Other Information")
        
        knowledge_text = st.text_area(
            "Add bio, experience, skills, etc.:",
            height=100,
            placeholder="Paste information about yourself...",
            key="add_text"
        )
        
        category = st.text_input(
            "Category:",
            placeholder="e.g., bio, experience, education, skills",
            key="add_cat"
        )
        
        if st.button("💾 Save Information"):
            if knowledge_text and category:
                chunks = [chunk.strip() for chunk in knowledge_text.split('\n\n') if chunk.strip()]
                
                for chunk in chunks:
                    embedding = get_embedding(chunk, client)
                    st.session_state.knowledge_base.append({
                        'category': category,
                        'content': chunk,
                        'embedding': embedding,
                        'date_added': datetime.now().isoformat()
                    })
                
                save_knowledge_base(st.session_state.knowledge_base)
                st.success(f"✅ Added {len(chunks)} item(s)!")
            else:
                st.error("Please fill in both fields")
        
        st.divider()
        
        # ====================
        # SETTINGS
        # ====================
        st.subheader("⚙️ Settings")
        
        st.session_state.use_semantic_search = st.checkbox(
            "Use semantic search (AI-powered)",
            value=st.session_state.use_semantic_search,
            help="Uses embeddings for more intelligent search. Disable for faster keyword search."
        )
        
        st.session_state.show_debug = st.checkbox(
            "Show debug info",
            value=st.session_state.show_debug
        )
        
        st.divider()
        
        # ====================
        # DOWNLOAD & BACKUP
        # ====================
        st.subheader("💾 Backup & Export")
        
        if st.session_state.knowledge_base:
            json_str = json.dumps(st.session_state.knowledge_base, indent=2)
            st.download_button(
                label="⬇️ Download Knowledge Base",
                data=json_str,
                file_name=f"knowledge_base_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        if st.button("🗑️ Clear All Data"):
            if st.button("⚠️ Confirm Clear All", key="confirm_clear"):
                st.session_state.knowledge_base = []
                save_knowledge_base([])
                st.success("All data cleared!")
                st.rerun()
    
    elif password_input:
        st.error("❌ Incorrect password")
    else:
        st.info("🔒 Enter admin password to manage content")

# ====================
# MAIN CHAT INTERFACE
# ====================
st.markdown(f"""
<div class="main-header">
    <h1>📝 {YOUR_NAME}'s Publishing Assistant</h1>
    <p>{YOUR_ROLE} • {BRAND_DESCRIPTION}</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if len(st.session_state.messages) == 0:
    st.info(f"""👋 Hello! I'm {YOUR_NAME}'s AI publishing assistant. 

I can help you:
- Find connections between your articles
- Maintain brand voice consistency
- Reference past work for new articles
- Answer questions about Rome's background and expertise

Ask me anything!""")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📚 Show my articles"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What articles have I written?"
            })
            st.rerun()
    with col2:
        if st.button("🎯 My expertise"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are my areas of expertise?"
            })
            st.rerun()
    with col3:
        if st.button("✍️ Writing style"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Describe my writing style"
            })
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about articles, get writing help, or anything else..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get relevant context
    context = ""
    debug_info = ""
    related_articles_text = ""
    
    if st.session_state.knowledge_base:
        # Search with semantic or keyword
        results = search_knowledge(
            prompt,
            st.session_state.knowledge_base,
            client,
            use_semantic=st.session_state.use_semantic_search,
            top_k=10
        )
        
        if results:
            context = "\n\n".join([
                f"[{item.get('type', 'info').upper()}] {item.get('title', '')}\n{item['content']}"
                for item in results
            ])
            categories_found = set(item.get('category', 'unknown') for item in results)
            debug_info = f"Retrieved {len(results)} items from: {', '.join(categories_found)}"
        
        # Check if query is about writing/articles - find related articles
        writing_keywords = ['write', 'writing', 'article', 'publish', 'topic', 'about']
        if any(kw in prompt.lower() for kw in writing_keywords):
            related = find_related_articles(prompt, st.session_state.knowledge_base, client, threshold=0.6)
            if related:
                related_articles_text = "\n\nRELATED ARTICLES YOU'VE WRITTEN:\n" + "\n".join([
                    f"- '{r['title']}' ({r['category']}) - {r['similarity']:.0%} relevant"
                    for r in related[:5]
                ])
    
    # Build system prompt
    brand_voice = get_brand_voice_content(st.session_state.knowledge_base)
    article_library = get_article_summaries(st.session_state.knowledge_base)
    
    system_prompt = f"""You are {YOUR_NAME}'s Personal Publishing Assistant for Medium and LinkedIn.

{AUSTRALIAN_ENGLISH_INSTRUCTION}

BRAND VOICE GUIDE:
{brand_voice}

ARTICLE LIBRARY (Your published work):
{article_library}

CORE RESPONSIBILITIES:
1. **Maintain Australian English**: Always use Australian spelling and expressions
2. **Brand Voice Consistency**: Write in Rome's established style  
3. **Cross-Reference Articles**: When relevant, connect to previous work
4. **Publishing Guidance**: Help with article ideation, structure, and refinement
5. **Contextual Accuracy**: Only use information from the knowledge base below

WHEN HELPING WITH ARTICLES:
- Reference related past articles when the topic connects
- Suggest fresh angles on topics already covered
- Maintain voice consistency with established style
- Structure based on Rome's typical format
- Always use Australian English

{related_articles_text}

KNOWLEDGE BASE CONTEXT:
{context if context else "No specific context retrieved for this query."}

Respond naturally, conversationally, and always in Australian English. Be insightful and helpful in supporting Rome's publishing goals."""

    # Get AI response
    with st.chat_message("assistant"):
        # Show debug info if enabled
        if password_input == ADMIN_PASSWORD and st.session_state.show_debug:
            st.caption(f"🐛 {debug_info}")
            if context:
                with st.expander("📄 Context used"):
                    st.text(context[:500] + "...")
        
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream response
        try:
            for response in client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            ):
                if response.choices[0].delta.content:
                    full_response += response.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            full_response = "I encountered an error. Please check your API configuration."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ====================
# FOOTER
# ====================
st.divider()

col1, col2 = st.columns([5, 1])
with col1:
    st.caption("💡 Visitors can chat freely. Admin access required to manage content.")
with col2:
    if st.button("🔄 New Chat"):
        st.session_state.messages = []
        st.rerun()

st.caption("🇦🇺 Powered by Australian English • Semantic search enabled")
