"""
AGENTIC AI SYSTEM WITH RAG
Multi-agent system using LangGraph for customer feedback analysis
Features:
- RAG-powered query system
- Autonomous agents for analysis, visualization, ticket creation
- Human-in-the-loop approval
- Vector database integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import TypedDict, Annotated, List, Dict, Any
import operator

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor

# Vector Database
import chromadb
from chromadb.utils import embedding_functions

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/reviews_with_topics_and_sentiment1.csv"
VECTOR_DB_PATH = "./chroma_db"
OPENAI_API_KEY = "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"

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

# -----------------------------
# STEP 1: LOAD AND PREPARE DATA
# -----------------------------
def load_feedback_data():
    """Load processed feedback data"""
    print("📊 Loading feedback data...")
    df = pd.read_csv(CSV_FILE)
    
    # Add datetime parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    
    print(f"✅ Loaded {len(df)} reviews")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Topics: {df['topic_category'].unique()}")
    
    return df

# -----------------------------
# STEP 2: CREATE EMBEDDINGS & VECTOR DB
# -----------------------------
class VectorDBManager:
    """Manages vector database for RAG"""
    
    def __init__(self, db_path=VECTOR_DB_PATH):
        print("\n🗄️  Initializing Vector Database...")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name="feedback_reviews",
                embedding_function=self.embedding_function
            )
            print(f"✅ Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = None
            print("⚠️  No existing collection found")
    
    def create_embeddings(self, df):
        """Create embeddings for all reviews"""
        print("\n🔄 Creating embeddings for all reviews...")
        
        # Delete old collection if exists
        try:
            self.client.delete_collection("feedback_reviews")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name="feedback_reviews",
            embedding_function=self.embedding_function,
            metadata={"description": "Customer feedback reviews with topics and sentiment"}
        )
        
        # Prepare documents
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create rich document text
            doc_text = f"""
            Review: {row['review_text']}
            Category: {row['topic_category']}
            Sentiment: {row['sentiment_label']} (score: {row['sentiment_score']:.2f})
            Rating: {row['rating']}
            Date: {row['date']}
            """
            
            documents.append(doc_text.strip())
            
            # Store metadata
            metadatas.append({
                "topic_category": str(row['topic_category']),
                "sentiment_label": str(row['sentiment_label']),
                "sentiment_score": float(row['sentiment_score']),
                "rating": float(row['rating']) if pd.notna(row['rating']) else 0.0,
                "date": str(row['date']),
                "review_text": str(row['review_text'])
            })
            
            ids.append(f"review_{idx}")
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            print(f"   Added batch {i//batch_size + 1}/{(len(documents)//batch_size) + 1}")
        
        print(f"✅ Created embeddings for {len(documents)} reviews")
    
    def search(self, query: str, n_results: int = 10, filters: Dict = None):
        """Search vector database"""
        where_clause = filters if filters else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        return results

# -----------------------------
# STEP 3: DATA QUERY TOOLS
# -----------------------------
class DataQueryTools:
    """Tools for querying feedback data"""
    
    def __init__(self, df, vector_db):
        self.df = df
        self.vector_db = vector_db
    
    def query_by_topic(self, topic: str) -> str:
        """Get reviews by topic category"""
        filtered = self.df[self.df['topic_category'] == topic]
        
        if len(filtered) == 0:
            return f"No reviews found for topic: {topic}"
        
        summary = f"""
        Topic: {topic}
        Total Reviews: {len(filtered)}
        Average Sentiment: {filtered['sentiment_score'].mean():.3f}
        Negative %: {(filtered['sentiment_label'] == 'NEGATIVE').sum() / len(filtered) * 100:.1f}%
        
        Top 3 Reviews:
        {filtered.head(3)['review_text'].tolist()}
        """
        return summary
    
    def query_by_date_range(self, days_ago: int) -> str:
        """Get reviews from last N days"""
        cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=days_ago))
        filtered = self.df[self.df['date'] >= cutoff_date]
        
        if len(filtered) == 0:
            return f"No reviews found in last {days_ago} days"
        
        topic_dist = filtered['topic_category'].value_counts()
        sent_dist = filtered['sentiment_label'].value_counts()
        
        summary = f"""
        Reviews from last {days_ago} days:
        Total: {len(filtered)}
        
        By Topic:
        {topic_dist.to_dict()}
        
        By Sentiment:
        {sent_dist.to_dict()}
        
        Average Sentiment: {filtered['sentiment_score'].mean():.3f}
        """
        return summary
    
    def semantic_search(self, query: str, n_results: int = 5) -> str:
        """Semantic search using vector DB"""
        results = self.vector_db.search(query, n_results=n_results)
        
        if not results['documents'][0]:
            return "No relevant reviews found"
        
        output = f"Top {n_results} relevant reviews for: '{query}'\n\n"
        
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            output += f"{i+1}. {meta['review_text']}\n"
            output += f"   Category: {meta['topic_category']} | Sentiment: {meta['sentiment_label']}\n\n"
        
        return output
    
    def get_statistics(self) -> str:
        """Get overall statistics"""
        stats = f"""
        Overall Statistics:
        
        Total Reviews: {len(self.df)}
        Date Range: {self.df['date'].min()} to {self.df['date'].max()}
        
        Sentiment Distribution:
        - Positive: {(self.df['sentiment_label'] == 'POSITIVE').sum()} ({(self.df['sentiment_label'] == 'POSITIVE').sum()/len(self.df)*100:.1f}%)
        - Negative: {(self.df['sentiment_label'] == 'NEGATIVE').sum()} ({(self.df['sentiment_label'] == 'NEGATIVE').sum()/len(self.df)*100:.1f}%)
        
        Average Sentiment Score: {self.df['sentiment_score'].mean():.3f}
        
        Top Issues by Volume:
        {self.df['topic_category'].value_counts().head().to_dict()}
        
        Most Negative Topics:
        {self.df.groupby('topic_category')['sentiment_score'].mean().sort_values().head().to_dict()}
        """
        return stats

# -----------------------------
# STEP 4: VISUALIZATION AGENT
# -----------------------------
class VisualizationAgent:
    """Agent for creating visualizations"""
    
    def __init__(self, df):
        self.df = df
        self.output_dir = "agent_visualizations/"
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_sentiment_bar_chart(self, topic: str = None, days_ago: int = None) -> str:
        """Create bar chart of sentiment distribution"""
        df = self.df.copy()
        
        # Apply filters
        if topic:
            df = df[df['topic_category'] == topic]
        if days_ago:
            cutoff = pd.Timestamp(datetime.now() - timedelta(days=days_ago))
            df = df[df['date'] >= cutoff]
        
        if len(df) == 0:
            return "No data available for the specified filters"
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = {'POSITIVE': 'green', 'NEGATIVE': 'red'}
        sentiment_counts.plot(kind='bar', color=[colors.get(x, 'gray') for x in sentiment_counts.index])
        
        title = "Sentiment Distribution"
        if topic:
            title += f" - {topic}"
        if days_ago:
            title += f" (Last {days_ago} days)"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save
        filename = f"{self.output_dir}sentiment_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"✅ Created bar chart: {filename}\nShowing {len(df)} reviews"
    
    def create_topic_sentiment_heatmap(self) -> str:
        """Create heatmap of topics vs sentiment"""
        plt.figure(figsize=(12, 8))
        
        pivot = self.df.groupby(['topic_category', 'sentiment_label']).size().unstack(fill_value=0)
        sns.heatmap(pivot, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Count'})
        
        plt.title('Topic vs Sentiment Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Topic Category')
        plt.tight_layout()
        
        filename = f"{self.output_dir}heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"✅ Created heatmap: {filename}"
    
    def create_trend_line_chart(self, topic: str = None) -> str:
        """Create trend line chart over time"""
        df = self.df.copy()
        
        if topic:
            df = df[df['topic_category'] == topic]
        
        # Group by date
        df['date'] = pd.to_datetime(df['date'])
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean()
        
        plt.figure(figsize=(12, 6))
        daily_sentiment.plot(kind='line', marker='o')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        title = "Sentiment Trend Over Time"
        if topic:
            title += f" - {topic}"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{self.output_dir}trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"✅ Created trend chart: {filename}"

# -----------------------------
# STEP 5: TICKET CREATION AGENT
# -----------------------------
class TicketAgent:
    """Agent for creating tickets with human approval"""
    
    def __init__(self):
        self.pending_tickets = []
    
    def create_ticket(self, 
                     title: str,
                     description: str,
                     priority: str,
                     category: str,
                     affected_users: int = 0) -> Dict:
        """Create a ticket (requires human approval)"""
        
        ticket = {
            "id": f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "title": title,
            "description": description,
            "priority": priority,  # LOW, MEDIUM, HIGH, CRITICAL
            "category": category,
            "affected_users": affected_users,
            "created_at": datetime.now().isoformat(),
            "status": "PENDING_APPROVAL",
            "approved": False
        }
        
        self.pending_tickets.append(ticket)
        
        return ticket
    
    def request_human_approval(self, ticket: Dict) -> bool:
        """Request human approval for ticket"""
        print("\n" + "="*60)
        print("🎫 TICKET APPROVAL REQUIRED")
        print("="*60)
        print(f"\nTicket ID: {ticket['id']}")
        print(f"Title: {ticket['title']}")
        print(f"Priority: {ticket['priority']}")
        print(f"Category: {ticket['category']}")
        print(f"Affected Users: {ticket['affected_users']}")
        print(f"\nDescription:")
        print(ticket['description'])
        print("\n" + "="*60)
        
        # Human input
        approval = input("\nApprove this ticket? (yes/no): ").strip().lower()
        
        if approval in ['yes', 'y']:
            ticket['status'] = "APPROVED"
            ticket['approved'] = True
            ticket['approved_at'] = datetime.now().isoformat()
            print("✅ Ticket APPROVED")
            return True
        else:
            ticket['status'] = "REJECTED"
            ticket['rejected_at'] = datetime.now().isoformat()
            print("❌ Ticket REJECTED")
            return False
    
    def get_pending_tickets(self) -> List[Dict]:
        """Get all pending tickets"""
        return [t for t in self.pending_tickets if t['status'] == 'PENDING_APPROVAL']

# -----------------------------
# STEP 6: LANGGRAPH STATE
# -----------------------------
class AgentState(TypedDict):
    """State passed between agents"""
    messages: Annotated[List, operator.add]
    user_query: str
    intent: str  # query, visualize, create_ticket, summarize
    data_results: str
    visualization_path: str
    ticket_info: Dict
    next_action: str
    requires_human_approval: bool
    approved: bool

# -----------------------------
# STEP 7: AGENT NODES
# -----------------------------
def query_router_node(state: AgentState) -> AgentState:
    """Routes query to appropriate agent"""
    print("\n🔀 Query Router Agent")
    
    user_query = state['user_query'].lower()
    
    # Determine intent
    if any(word in user_query for word in ['chart', 'graph', 'plot', 'visualize', 'show me']):
        intent = "visualize"
    elif any(word in user_query for word in ['create ticket', 'raise ticket', 'open ticket', 'issue']):
        intent = "create_ticket"
    elif any(word in user_query for word in ['summarize', 'summary', 'overview', 'report']):
        intent = "summarize"
    else:
        intent = "query"
    
    print(f"   Detected intent: {intent}")
    
    state['intent'] = intent
    state['next_action'] = intent
    
    return state

def data_analysis_node(state: AgentState, data_tools: DataQueryTools) -> AgentState:
    """Analyzes data using RAG"""
    print("\n📊 Data Analysis Agent")
    
    user_query = state['user_query']
    
    # Use LLM to determine which tool to use
    prompt = f"""Given this user query: "{user_query}"
    
    Determine which analysis would be most helpful:
    1. semantic_search - for finding similar reviews
    2. query_by_topic - for topic-specific analysis
    3. query_by_date_range - for time-based analysis
    4. get_statistics - for overall statistics
    
    Return ONLY the function name, nothing else."""
    
    response = llm.invoke(prompt)
    tool_name = response.content.strip()
    
    print(f"   Using tool: {tool_name}")
    
    # Execute appropriate tool
    if "semantic_search" in tool_name:
        results = data_tools.semantic_search(user_query)
    elif "query_by_topic" in tool_name:
        # Extract topic from query
        topics = ['payment_issues', 'login_problems', 'app_crashes_bugs', 'ui_ux_issues', 'fraud_security', 'cashback_rewards']
        topic = next((t for t in topics if t.replace('_', ' ') in user_query.lower()), topics[0])
        results = data_tools.query_by_topic(topic)
    elif "date_range" in tool_name:
        # Extract days from query
        days = 30  # default
        if "week" in user_query.lower():
            days = 7
        elif "month" in user_query.lower():
            days = 30
        results = data_tools.query_by_date_range(days)
    else:
        results = data_tools.get_statistics()
    
    state['data_results'] = results
    print(f"   Results: {results[:200]}...")
    
    return state

def visualization_node(state: AgentState, viz_agent: VisualizationAgent) -> AgentState:
    """Creates visualizations"""
    print("\n📈 Visualization Agent")
    
    user_query = state['user_query'].lower()
    
    # Determine visualization type
    if "bar" in user_query or "sentiment" in user_query:
        # Extract filters
        topic = None
        days = None
        
        topics = ['payment_issues', 'login_problems', 'app_crashes_bugs']
        for t in topics:
            if t.replace('_', ' ') in user_query:
                topic = t
                break
        
        if "month" in user_query:
            days = 30
        elif "week" in user_query:
            days = 7
        
        result = viz_agent.create_sentiment_bar_chart(topic=topic, days_ago=days)
        
    elif "heatmap" in user_query:
        result = viz_agent.create_topic_sentiment_heatmap()
    
    elif "trend" in user_query:
        topic = None
        topics = ['payment_issues', 'login_problems']
        for t in topics:
            if t.replace('_', ' ') in user_query:
                topic = t
                break
        result = viz_agent.create_trend_line_chart(topic=topic)
    
    else:
        result = viz_agent.create_sentiment_bar_chart()
    
    state['visualization_path'] = result
    print(f"   {result}")
    
    return state

def ticket_creation_node(state: AgentState, ticket_agent: TicketAgent, df: pd.DataFrame) -> AgentState:
    """Creates tickets with human approval"""
    print("\n🎫 Ticket Creation Agent")
    
    user_query = state['user_query']
    
    # Analyze data to determine ticket details
    negative_reviews = df[df['sentiment_score'] < -0.3]
    top_issue = negative_reviews['topic_category'].value_counts().index[0]
    affected_count = len(negative_reviews[negative_reviews['topic_category'] == top_issue])
    
    # Determine priority
    if affected_count > 200:
        priority = "CRITICAL"
    elif affected_count > 100:
        priority = "HIGH"
    elif affected_count > 50:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    # Create ticket
    ticket = ticket_agent.create_ticket(
        title=f"High Volume of Negative Feedback: {top_issue.replace('_', ' ').title()}",
        description=f"""
        Issue Category: {top_issue}
        Affected Users: {affected_count}
        Average Sentiment: {negative_reviews[negative_reviews['topic_category'] == top_issue]['sentiment_score'].mean():.3f}
        
        Top Complaints:
        {negative_reviews[negative_reviews['topic_category'] == top_issue].head(3)['review_text'].tolist()}
        
        Recommendation: Immediate investigation and resolution required.
        """,
        priority=priority,
        category=top_issue,
        affected_users=affected_count
    )
    
    state['ticket_info'] = ticket
    state['requires_human_approval'] = True
    state['next_action'] = "human_approval"
    
    return state

def human_approval_node(state: AgentState, ticket_agent: TicketAgent) -> AgentState:
    """Human-in-the-loop approval"""
    print("\n👤 Human Approval Required")
    
    ticket = state['ticket_info']
    approved = ticket_agent.request_human_approval(ticket)
    
    state['approved'] = approved
    state['next_action'] = "end"
    
    return state

def summarization_node(state: AgentState, df: pd.DataFrame) -> AgentState:
    """Generates executive summary"""
    print("\n📝 Summarization Agent")
    
    # Prepare data summary
    stats = {
        "total_reviews": len(df),
        "positive_pct": (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100,
        "negative_pct": (df['sentiment_label'] == 'NEGATIVE').sum() / len(df) * 100,
        "avg_sentiment": df['sentiment_score'].mean(),
        "top_issues": df.groupby('topic_category')['sentiment_score'].mean().sort_values().head(3).to_dict()
    }
    
    # Use LLM to generate summary
    prompt = f"""Based on this customer feedback data, generate an executive summary:
    
    {json.dumps(stats, indent=2)}
    
    Include:
    1. Overall sentiment health
    2. Top 3 critical issues
    3. Recommendations for action
    4. Positive highlights
    
    Keep it concise (3-4 paragraphs).
    """
    
    response = llm.invoke(prompt)
    summary = response.content
    
    state['data_results'] = summary
    print(f"   Generated summary")
    
    return state

# -----------------------------
# STEP 8: BUILD LANGGRAPH
# -----------------------------
def create_agent_graph(df, data_tools, viz_agent, ticket_agent):
    """Create LangGraph workflow"""
    
    # Define graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", query_router_node)
    workflow.add_node("data_analysis", lambda state: data_analysis_node(state, data_tools))
    workflow.add_node("visualization", lambda state: visualization_node(state, viz_agent))
    workflow.add_node("ticket_creation", lambda state: ticket_creation_node(state, ticket_agent, df))
    workflow.add_node("human_approval", lambda state: human_approval_node(state, ticket_agent))
    workflow.add_node("summarization", lambda state: summarization_node(state, df))
    
    # Define edges
    workflow.set_entry_point("router")
    
    # Conditional routing based on intent
    def route_based_on_intent(state):
        intent = state.get('next_action', 'query')
        if intent == "visualize":
            return "visualization"
        elif intent == "create_ticket":
            return "ticket_creation"
        elif intent == "summarize":
            return "summarization"
        else:
            return "data_analysis"
    
    workflow.add_conditional_edges(
        "router",
        route_based_on_intent,
        {
            "data_analysis": "data_analysis",
            "visualization": "visualization",
            "ticket_creation": "ticket_creation",
            "summarization": "summarization"
        }
    )
    
    # All paths lead to END except ticket creation
    workflow.add_edge("data_analysis", END)
    workflow.add_edge("visualization", END)
    workflow.add_edge("summarization", END)
    workflow.add_edge("ticket_creation", "human_approval")
    workflow.add_edge("human_approval", END)
    
    return workflow.compile()

# -----------------------------
# STEP 9: MAIN SYSTEM
# -----------------------------
class AgenticAISystem:
    """Main agentic AI system"""
    
    def __init__(self, csv_file=CSV_FILE):
        print("\n🚀 Initializing Agentic AI System...")
        
        # Load data
        self.df = load_feedback_data()
        
        # Initialize vector DB
        self.vector_db = VectorDBManager()
        if self.vector_db.collection is None or self.vector_db.collection.count() == 0:
            self.vector_db.create_embeddings(self.df)
        
        # Initialize agents
        self.data_tools = DataQueryTools(self.df, self.vector_db)
        self.viz_agent = VisualizationAgent(self.df)
        self.ticket_agent = TicketAgent()
        
        # Create LangGraph
        self.graph = create_agent_graph(
            self.df,
            self.data_tools,
            self.viz_agent,
            self.ticket_agent
        )
        
        print("✅ System initialized and ready!")
    
    def query(self, user_query: str):
        """Process user query through agent system"""
        print("\n" + "="*60)
        print(f"USER QUERY: {user_query}")
        print("="*60)
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "intent": "",
            "data_results": "",
            "visualization_path": "",
            "ticket_info": {},
            "next_action": "",
            "requires_human_approval": False,
            "approved": False
        }
        
        # Run through graph
        final_state = self.graph.invoke(initial_state)
        
        # Return results
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        
        if final_state.get('data_results'):
            print(f"\n{final_state['data_results']}")
        
        if final_state.get('visualization_path'):
            print(f"\n{final_state['visualization_path']}")
        
        if final_state.get('ticket_info'):
            ticket = final_state['ticket_info']
            if final_state.get('approved'):
                print(f"\n✅ Ticket {ticket['id']} APPROVED and created")
            else:
                print(f"\n❌ Ticket {ticket['id']} was not approved")
        
        return final_state
    
    def interactive_mode(self):
        """Start interactive query mode"""
        print("\n" + "🤖"*30)
        print("AGENTIC AI SYSTEM - INTERACTIVE MODE")
        print("🤖"*30)
        print("\nExample queries:")
        print("  - 'Show me negative sentiment in payment issues from last month'")
        print("  - 'Give me a bar chart of sentiment distribution'")
        print("  - 'Find reviews about login problems'")
        print("  - 'Create a ticket for the most critical issue'")
        print("  - 'Generate an executive summary'")
        print("\nType 'exit' to quit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                self.query(user_input)
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

# -----------------------------
# STEP 10: RUN SYSTEM
# -----------------------------
if __name__ == "__main__":
    # Initialize system
    system = AgenticAISystem()
    
    # Example queries (uncomment to test)
    system.query("Show me a bar chart of negative sentiment in payment issues from last month")
    # system.query("Find similar reviews about crashes")
    # system.query("Create a ticket for the most critical issue")
    # system.query("Generate an executive summary of customer feedback")
    
    # Start interactive mode
    system.interactive_mode()