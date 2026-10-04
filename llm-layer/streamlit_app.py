

"""
STREAMLIT DASHBOARD - Customer Feedback Intelligence
Features:
- Executive summary dashboard
- Real-time visualizations
- Chat interface with streaming responses
- Ticket creation with approval
- Image outputs display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from PIL import Image
import os

# LangChain & Agentic AI imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from chromadb.utils import embedding_functions

from agents_app import DataQueryTools, VisualizationAgent, create_agent_graph, process_chat_query


# Initialize agents if not in session state
if 'data_tools' not in st.session_state:
    st.session_state.data_tools = None
if 'viz_agent' not in st.session_state:
    st.session_state.viz_agent = None
if 'agent_graph' not in st.session_state:
    st.session_state.agent_graph = None


# Set page config
st.set_page_config(
    page_title="InsightIO - Customer Feedback Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .critical-issue {
        background: #fee;
        border-left: 4px solid #f00;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .positive-highlight {
        background: #efe;
        border-left: 4px solid #0f0;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 600;
    }
    .ticket-card {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIGURATION
# -----------------------------
CSV_FILE = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/reviews_with_topics_and_sentiment1.csv"
VECTOR_DB_PATH = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/chroma_db"
OPENAI_API_KEY = "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"

# -----------------------------
# INITIALIZE SESSION STATE
# -----------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = None

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        streaming=True
    )

if 'pending_ticket' not in st.session_state:
    st.session_state.pending_ticket = None

if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

# -----------------------------
# DATA LOADING & CACHING
# -----------------------------
@st.cache_data
def load_data():
    """Load and prepare data"""
    df = pd.read_csv(CSV_FILE)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    return df

@st.cache_resource
def initialize_vector_db():
    """Initialize ChromaDB"""
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_collection(
            name="feedback_reviews",
            embedding_function=embedding_function
        )
        print("db success")
        return collection
    except:
        print("db not success")
        return None

# Load data
if st.session_state.df is None:
    with st.spinner("🔄 Loading data..."):
        st.session_state.df = load_data()

if st.session_state.vector_db is None:
    with st.spinner("🔄 Initializing vector database..."):
        st.session_state.vector_db = initialize_vector_db()

df = st.session_state.df



# Initialize agents
@st.cache_resource
def initialize_agents(_df, _vector_db):
    """Initialize all agents"""
    data_tools = DataQueryTools(_df, _vector_db)
    viz_agent = VisualizationAgent(_df)
    agent_graph = create_agent_graph(_df, data_tools, viz_agent,st.session_state.llm)
    return data_tools, viz_agent, agent_graph

# Initialize if not already done
if st.session_state.data_tools is None:
    st.session_state.data_tools, st.session_state.viz_agent, st.session_state.agent_graph = initialize_agents(
        st.session_state.df,
        st.session_state.vector_db
    )



# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=InsightIO", use_container_width=True)
    
    st.markdown("### 📊 Dashboard")
    page = st.radio(
        "Navigate to:",
        ["🏠 Overview", "💬 AI Chat", "🎫 Tickets", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📅 Filters")
    date_range = st.selectbox(
        "Time Period",
        ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )
    
    topics = ["All Topics"] + list(df['topic_category'].unique())
    selected_topic = st.selectbox("Topic", topics)
    
    sentiment_filter = st.selectbox(
        "Sentiment",
        ["All", "Positive", "Negative"]
    )
    
    st.markdown("---")
    
    # Stats summary
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Reviews", f"{len(df):,}")
    st.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
    negative_pct = (df['sentiment_label'] == 'NEGATIVE').sum() / len(df) * 100
    st.metric("Negative %", f"{negative_pct:.1f}%")

# Apply filters
filtered_df = df.copy()

if date_range != "All Time":
    days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days_map[date_range]))
    filtered_df = filtered_df[filtered_df['date'] >= cutoff]

if selected_topic != "All Topics":
    filtered_df = filtered_df[filtered_df['topic_category'] == selected_topic]

if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter.upper()]

# -----------------------------
# PAGE: OVERVIEW
# -----------------------------
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">🤖 InsightIO Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Customer Feedback Intelligence Platform**")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Reviews",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)}" if date_range != "All Time" else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.2f}",
            delta=f"{avg_sentiment - df['sentiment_score'].mean():.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        negative_count = (filtered_df['sentiment_label'] == 'NEGATIVE').sum()
        negative_pct = (negative_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Negative Reviews",
            value=f"{negative_count:,}",
            delta=f"{negative_pct:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        positive_count = (filtered_df['sentiment_label'] == 'POSITIVE').sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Positive Reviews",
            value=f"{positive_count:,}",
            delta=f"{(positive_count/len(filtered_df)*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Critical Issues
    st.markdown("### 🚨 Critical Issues")
    
    critical_df = filtered_df[filtered_df['sentiment_score'] < -0.3]
    if len(critical_df) > 0:
        topic_issues = critical_df.groupby('topic_category').agg({
            'sentiment_score': 'mean',
            'review_text': 'count'
        }).sort_values('sentiment_score')
        
        for idx, (topic, row) in enumerate(topic_issues.head(3).iterrows()):
            st.markdown(f"""
            <div class="critical-issue">
                <h4>#{idx+1} {topic.replace('_', ' ').title()}</h4>
                <p><strong>{int(row['review_text'])} reviews</strong> | Avg Sentiment: <strong>{row['sentiment_score']:.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical issues detected!")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'POSITIVE': '#4caf50', 'NEGATIVE': '#f44336'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Topics Distribution")
        topic_counts = filtered_df['topic_category'].value_counts().head(6)
        fig = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation='h',
            color=topic_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Count",
            yaxis_title="Topic"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment by Topic Heatmap
    st.markdown("### 🔥 Sentiment Heatmap by Topic")
    heatmap_data = filtered_df.groupby(['topic_category', 'sentiment_label']).size().unstack(fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        text=heatmap_data.values,
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend Chart
    st.markdown("### 📈 Sentiment Trend Over Time")
    trend_df = filtered_df.copy()
    trend_df['date_only'] = trend_df['date'].dt.date
    daily_sentiment = trend_df.groupby('date_only')['sentiment_score'].mean().reset_index()
    
    fig = px.line(
        daily_sentiment,
        x='date_only',
        y='sentiment_score',
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Avg Sentiment Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample Reviews
    st.markdown("### 💬 Recent Reviews")
    sample_reviews = filtered_df.nlargest(5, 'date')[['review_text', 'sentiment_label', 'topic_category', 'date']]
    
    for _, review in sample_reviews.iterrows():
        sentiment_color = "🟢" if review['sentiment_label'] == 'POSITIVE' else "🔴"
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
            {sentiment_color} <strong>{review['topic_category'].replace('_', ' ').title()}</strong> | {review['date'].strftime('%Y-%m-%d')}
            <br><em>"{review['review_text'][:150]}..."</em>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# PAGE: AI CHAT
# -----------------------------
elif page == "💬 AI Chat":
    st.markdown('<h1 class="main-header">💬 AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by LangGraph Multi-Agent System with RAG**")
    
    # Display agent status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 Active Agents", "5")
    with col2:
        st.metric("🗄️ Vector DB", "Ready" if st.session_state.vector_db else "Not Ready")
    with col3:
        st.metric("💬 Chat History", len(st.session_state.chat_history))
    
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
                    
                    # Display image if present
                    if 'image' in msg and msg['image']:
                        st.image(msg['image'], caption="Generated Visualization", use_container_width=True)
    
    # Chat input
    st.markdown("---")
    
    # Suggested queries
    st.markdown("**💡 Suggested Queries:**")
    suggestions = [
        "Show me payment issues",
        "Create a sentiment chart",
        "Find reviews about crashes",
        "Generate a summary"
    ]
    
    cols = st.columns(len(suggestions))
    for idx, suggestion in enumerate(suggestions):
        if cols[idx].button(suggestion, key=f"suggest_{idx}"):
            user_input = suggestion
            
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Process through agent system
            with st.spinner("🤖 Agents working..."):
                try:
                    # Run through LangGraph
                    result = process_chat_query(user_input, filtered_df)
                    
                    # Get response
                    response_text = result.get('data_results', 'No response generated')
                    image_buffer = result.get('image_buffer', None)
                    
                    # Add AI response
                    response_msg = {
                        'role': 'assistant',
                        'content': response_text
                    }
                    
                    if image_buffer:
                        response_msg['image'] = image_buffer
                    
                    st.session_state.chat_history.append(response_msg)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"❌ Error processing query: {str(e)}"
                    })
            
            st.rerun()
    
    # Text input
    user_input = st.chat_input("Ask me anything about customer feedback...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process through agent system
        with st.chat_message("assistant"):
            with st.spinner("🤖 Routing to appropriate agent..."):
                try:
                    # Run through LangGraph
                    result = process_chat_query(user_input, filtered_df)
                    
                    # Get response
                    response_text = result.get('data_results', 'No response generated')
                    image_buffer = result.get('image_buffer', None)
                    
                    # Stream response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate streaming
                    for word in response_text.split():
                        full_response += word + " "
                        message_placeholder.markdown(full_response + "▌")
                        import time
                        time.sleep(0.02)
                    
                    message_placeholder.markdown(full_response)
                    
                    # Display image if present
                    if image_buffer:
                        st.image(image_buffer, caption="Generated Visualization", use_container_width=True)
                    
                    # Add AI response to history
                    response_msg = {
                        'role': 'assistant',
                        'content': response_text
                    }
                    
                    if image_buffer:
                        response_msg['image'] = image_buffer
                    
                    st.session_state.chat_history.append(response_msg)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"❌ Error processing query: {str(e)}"
                    })
        
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# -----------------------------
# PAGE: TICKETS
# -----------------------------
elif page == "🎫 Tickets":
    st.markdown('<h1 class="main-header">🎫 Ticket Management</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Create New Ticket")
        
        if st.button("🤖 Auto-Generate Ticket from Critical Issue", use_container_width=True):
            # Analyze data
            critical_df = filtered_df[filtered_df['sentiment_score'] < -0.3]
            if len(critical_df) > 0:
                top_issue = critical_df['topic_category'].value_counts().index[0]
                affected_count = len(critical_df[critical_df['topic_category'] == top_issue])
                avg_sentiment = critical_df[critical_df['topic_category'] == top_issue]['sentiment_score'].mean()
                
                # Determine priority
                if affected_count > 200:
                    priority = "🔴 CRITICAL"
                elif affected_count > 100:
                    priority = "🟠 HIGH"
                else:
                    priority = "🟡 MEDIUM"
                
                st.session_state.pending_ticket = {
                    "id": f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "title": f"High Volume of Negative Feedback: {top_issue.replace('_', ' ').title()}",
                    "priority": priority,
                    "category": top_issue,
                    "affected_users": affected_count,
                    "avg_sentiment": avg_sentiment,
                    "description": f"""
**Issue Category:** {top_issue.replace('_', ' ').title()}
**Affected Users:** {affected_count}
**Average Sentiment:** {avg_sentiment:.3f}

**Top Complaints:**
{chr(10).join(['- ' + review for review in critical_df[critical_df['topic_category'] == top_issue].head(3)['review_text'].tolist()])}

**Recommendation:** Immediate investigation and resolution required.
                    """
                }
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Critical Issues", len(filtered_df[filtered_df['sentiment_score'] < -0.5]))
        st.metric("High Priority", len(filtered_df[filtered_df['sentiment_score'] < -0.3]))
        st.metric("Requires Attention", len(filtered_df[filtered_df['sentiment_score'] < 0]))
    
    # Display pending ticket for approval
    if st.session_state.pending_ticket:
        st.markdown("---")
        st.markdown("### 📋 Ticket Preview (Awaiting Approval)")
        
        ticket = st.session_state.pending_ticket
        
        st.markdown(f"""
        <div class="ticket-card">
            <h3>{ticket['title']}</h3>
            <p><strong>ID:</strong> {ticket['id']}</p>
            <p><strong>Priority:</strong> {ticket['priority']}</p>
            <p><strong>Category:</strong> {ticket['category']}</p>
            <p><strong>Affected Users:</strong> {ticket['affected_users']}</p>
            <p><strong>Avg Sentiment:</strong> {ticket['avg_sentiment']:.3f}</p>
            <hr>
            <p>{ticket['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("✅ Approve & Create", type="primary", use_container_width=True):
                st.success(f"✅ Ticket {ticket['id']} approved and created!")
                # Here you would integrate with JIRA/GitHub API
                st.session_state.pending_ticket = None
                st.rerun()
        
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                st.warning("Ticket rejected")
                st.session_state.pending_ticket = None
                st.rerun()

# -----------------------------
# PAGE: ANALYTICS
# -----------------------------
elif page == "📈 Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Analytics</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Deep Dive", "🔍 Cohort Analysis", "🎯 Predictions"])
    
    with tab1:
        st.markdown("### Topic-Wise Deep Dive")
        
        for topic in filtered_df['topic_category'].unique():
            if topic not in ['other_issues', 'uncategorized']:
                topic_df = filtered_df[filtered_df['topic_category'] == topic]
                
                with st.expander(f"📌 {topic.replace('_', ' ').title()} ({len(topic_df)} reviews)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Sentiment", f"{topic_df['sentiment_score'].mean():.2f}")
                    with col2:
                        neg_pct = (topic_df['sentiment_label'] == 'NEGATIVE').sum() / len(topic_df) * 100
                        st.metric("Negative %", f"{neg_pct:.1f}%")
                    with col3:
                        st.metric("Avg Rating", f"{topic_df['rating'].mean():.1f}")
                    
                    # Word cloud would go here
                    st.markdown("**Sample Reviews:**")
                    for review in topic_df.head(3)['review_text']:
                        st.markdown(f"- {review[:100]}...")
    
    with tab2:
        st.markdown("### Sentiment by Time Period")
        
        # Create cohorts by month
        cohort_df = filtered_df.copy()
        cohort_df['month'] = cohort_df['date'].dt.to_period('M')
        
        monthly_sentiment = cohort_df.groupby(['month', 'topic_category'])['sentiment_score'].mean().reset_index()
        monthly_sentiment['month'] = monthly_sentiment['month'].astype(str)
        
        fig = px.line(
            monthly_sentiment,
            x='month',
            y='sentiment_score',
            color='topic_category',
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Sentiment Predictions")
        st.info("🔮 ML-based sentiment prediction coming soon...")
        
        # Placeholder for predictions
        st.markdown("**Predicted Trends:**")
        st.markdown("- Payment issues likely to increase by 15% next month")
        st.markdown("- Login problems showing improvement trend")
        st.markdown("- Overall sentiment expected to stabilize around 0.2")



# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 <strong>InsightIO</strong> - AI-Powered Customer Feedback Intelligence</p>
    <p>Built with Streamlit • LangChain • GPT-4 • ChromaDB</p>
</div>
""", unsafe_allow_html=True)