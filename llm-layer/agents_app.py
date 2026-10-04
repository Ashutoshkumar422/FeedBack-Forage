# -----------------------------
# AGENTIC AI SYSTEM INTEGRATION
# -----------------------------
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
import matplotlib.pyplot as plt
import io
import pandas as pd
import streamlit as st

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_query: str
    intent: str
    data_results: str
    visualization_path: str
    image_buffer: Any
    next_action: str


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
**Topic: {topic.replace('_', ' ').title()}**

- Total Reviews: {len(filtered)}
- Average Sentiment: {filtered['sentiment_score'].mean():.3f}
- Negative %: {(filtered['sentiment_label'] == 'NEGATIVE').sum() / len(filtered) * 100:.1f}%

**Sample Reviews:**
{chr(10).join(['- ' + str(review)[:100] + '...' for review in filtered.head(3)['review_text'].tolist()])}
        """
        return summary
    
    # def semantic_search(self, query: str, n_results: int = 5) -> str:
    #     """Semantic search using vector DB"""
    #     if self.vector_db is None:
    #         return "Vector database not initialized. Using keyword search instead."
        
    #     try:
    #         results = self.vector_db.query(
    #             query_texts=[query],
    #             n_results=n_results
    #         )
            
    #         if not results['documents'][0]:
    #             return "No relevant reviews found"
            
    #         output = f"**Top {n_results} relevant reviews for: '{query}'**\n\n"
            
    #         for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    #             output += f"{i+1}. *{meta['review_text'][:150]}...*\n"
    #             output += f"   - Category: {meta['topic_category']} | Sentiment: {meta['sentiment_label']}\n\n"
            
    #         return output
    #     except Exception as e:
    #         return f"Search error: {str(e)}"

    def semantic_search(self, query: str,llm, n_results: int = 5) -> str:
        """Semantic RAG search that returns ONLY the final answer to the query."""
        
        if self.vector_db is None:
            return "Vector DB not initialized."

        try:
            # Step 1: Retrieve semantically relevant reviews
            results = self.vector_db.query(
                query_texts=[query],
                n_results=n_results
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            if not docs:
                return "I could not find relevant reviews for that query."

            # Step 2: Prepare combined context from retrieved reviews
            context_blocks = []
            for meta in metas:
                block = f"Review: {meta['review_text']}\nCategory: {meta['topic_category']}\nSentiment: {meta['sentiment_label']}"
                context_blocks.append(block)

            context_text = "\n\n---\n".join(context_blocks)

            # Step 3: Ask the LLM to answer the query using RAG context
            rag_prompt = f"""
            You are an expert insights analyst. 
            Using ONLY the information from the reviews below, answer the user's query.

            USER QUERY:
            {query}

            RELEVANT REVIEWS:
            {context_text}

            INSTRUCTIONS:
            - Do NOT list individual reviews.
            - Do NOT reveal the raw text.
            - Provide a clean, executive-level summary.
            - Identify reasons, patterns, themes, or insights clearly.
            - Keep answer focused, factual, and based on the reviews only.

            FINAL ANSWER:
            """

            answer = llm.invoke(rag_prompt).content

            return answer

        except Exception as e:
            return f"Search error: {str(e)}"

    
    def get_statistics(self) -> str:
        """Get overall statistics"""
        stats = f"""
**Overall Statistics**

- Total Reviews: {len(self.df):,}
- Date Range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}

**Sentiment Distribution:**
- Positive: {(self.df['sentiment_label'] == 'POSITIVE').sum():,} ({(self.df['sentiment_label'] == 'POSITIVE').sum()/len(self.df)*100:.1f}%)
- Negative: {(self.df['sentiment_label'] == 'NEGATIVE').sum():,} ({(self.df['sentiment_label'] == 'NEGATIVE').sum()/len(self.df)*100:.1f}%)

**Average Sentiment Score:** {self.df['sentiment_score'].mean():.3f}

**Top Issues by Volume:**
{chr(10).join([f'- {k}: {v}' for k, v in self.df['topic_category'].value_counts().head().items()])}
        """
        return stats

class VisualizationAgent:
    """Agent for creating visualizations"""
    
    def __init__(self, df):
        self.df = df
    
    def create_sentiment_bar_chart(self, topic: str = None) -> io.BytesIO:
        """Create bar chart of sentiment distribution"""
        df = self.df.copy()
        
        if topic:
            df = df[df['topic_category'] == topic]
        
        if len(df) == 0:
            return None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = ['green' if x == 'POSITIVE' else 'red' for x in sentiment_counts.index]
        sentiment_counts.plot(kind='bar', ax=ax, color=colors)
        
        title = "Sentiment Distribution"
        if topic:
            title += f" - {topic.replace('_', ' ').title()}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def create_topic_chart(self) -> io.BytesIO:
        """Create topic distribution chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        topic_counts = self.df['topic_category'].value_counts().head(6)
        topic_counts.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('Topic Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        ax.set_ylabel('Topic')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

# Agent Nodes
# def query_router_node(state: AgentState) -> AgentState:
    # """Routes query to appropriate agent"""
    # user_query = state['user_query'].lower()
    
    # # Determine intent
    # if any(word in user_query for word in ['chart', 'graph', 'plot', 'visualize', 'show me']):
    #     intent = "visualize"
    # elif any(word in user_query for word in ['find', 'search', 'reviews about']):
    #     intent = "search"
    # elif any(word in user_query for word in ['statistics', 'stats', 'overview']):
    #     intent = "stats"
    # elif any(word in user_query for word in ['summary', 'summarize']):
    #     intent = "summarize"
    # else:
    #     intent = "query"
    
    # state['intent'] = intent
    # state['next_action'] = intent
    
    # return state

def query_router_node(state, llm):
    query = state["user_query"]

    system_prompt = """
    You are an intent-classification agent. Decide the NEXT ACTION.
    Allowed actions:
    - "search" → when user asks reasons, issues, root cause, trends
    - "summarize" → when user asks for summary
    - "visualize" → when user asks for chart, graph, bar, line, plot
    - "stats" → when user asks for metrics, averages, calculations

    Respond ONLY with one word: search, summarize, visualize, or stats.
    """

    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    intent = result.content.strip().lower()

    # store into graph state
    state["next_action"] = intent
    return state


def data_analysis_node(state: AgentState, data_tools: DataQueryTools) -> AgentState:
    """Analyzes data using tools"""
    user_query = state['user_query']
    
    # Extract topic if mentioned
    topics = ['payment_issues', 'login_problems', 'app_crashes_bugs', 'ui_ux_issues', 'fraud_security', 'cashback_rewards']
    topic = None
    for t in topics:
        if t.replace('_', ' ') in user_query.lower():
            topic = t
            break
    
    if topic:
        results = data_tools.query_by_topic(topic)
    else:
        results = data_tools.get_statistics()
    
    state['data_results'] = results
    return state

def search_node(state: AgentState, data_tools: DataQueryTools,llm) -> AgentState:
    """Semantic search"""
    results = data_tools.semantic_search(state['user_query'],llm)
    state['data_results'] = results
    return state

def visualization_node(state: AgentState, viz_agent: VisualizationAgent) -> AgentState:
    """Creates visualizations"""
    user_query = state['user_query'].lower()
    
    # Extract topic if mentioned
    topic = None
    topics = ['payment_issues', 'login_problems', 'app_crashes_bugs', 'ui_ux_issues']
    for t in topics:
        if t.replace('_', ' ') in user_query:
            topic = t
            break
    
    # Determine chart type
    if "topic" in user_query or "distribution" in user_query:
        buf = viz_agent.create_topic_chart()
        state['data_results'] = "✅ Created topic distribution chart"
    else:
        buf = viz_agent.create_sentiment_bar_chart(topic=topic)
        if topic:
            state['data_results'] = f"✅ Created sentiment chart for {topic.replace('_', ' ')}"
        else:
            state['data_results'] = "✅ Created sentiment distribution chart"
    
    state['image_buffer'] = buf
    return state

def summarization_node(state: AgentState, df: pd.DataFrame) -> AgentState:
    """Generates summary"""
    stats = {
        "total_reviews": len(df),
        "positive_pct": (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100,
        "negative_pct": (df['sentiment_label'] == 'NEGATIVE').sum() / len(df) * 100,
        "avg_sentiment": df['sentiment_score'].mean(),
        "top_issues": df.groupby('topic_category')['sentiment_score'].mean().sort_values().head(3).to_dict()
    }
    
    summary = f"""
**Executive Summary**

**Overall Health:** {'🟢 Positive' if stats['avg_sentiment'] > 0 else '🔴 Negative'} (Avg: {stats['avg_sentiment']:.3f})

**Sentiment Breakdown:**
- Positive: {stats['positive_pct']:.1f}%
- Negative: {stats['negative_pct']:.1f}%

**Top 3 Issues (Most Negative):**
{chr(10).join([f'- {k.replace("_", " ").title()}: {v:.3f}' for k, v in stats['top_issues'].items()])}

**Recommendation:** Focus on addressing the most negative categories to improve overall customer satisfaction.
    """
    
    state['data_results'] = summary
    return state

# def create_agent_graph(df, data_tools, viz_agent,llm):
#     """Create LangGraph workflow"""
    
#     workflow = StateGraph(AgentState)
    
#     # Add nodes
#     workflow.add_node("router", query_router_node)
#     workflow.add_node("data_analysis", lambda state: data_analysis_node(state, data_tools))
#     workflow.add_node("search", lambda state: search_node(state, data_tools))
#     workflow.add_node("visualization", lambda state: visualization_node(state, viz_agent))
#     workflow.add_node("summarization", lambda state: summarization_node(state, df))
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Conditional routing
#     def route_based_on_intent(state):
#         intent = state.get('next_action', 'query')
#         if intent == "visualize":
#             return "visualization"
#         elif intent == "search":
#             return "search"
#         elif intent == "summarize":
#             return "summarization"
#         elif intent == "stats":
#             return "data_analysis"
#         else:
#             return "data_analysis"
    
#     workflow.add_conditional_edges(
#         "router",
#         route_based_on_intent,
#         {
#             "data_analysis": "data_analysis",
#             "search": "search",
#             "visualization": "visualization",
#             "summarization": "summarization"
#         }
#     )
    
#     # All paths lead to END
#     workflow.add_edge("data_analysis", END)
#     workflow.add_edge("search", END)
#     workflow.add_edge("visualization", END)
#     workflow.add_edge("summarization", END)
    
#     return workflow.compile()

def create_agent_graph(df, data_tools, viz_agent, llm):
    """Create LangGraph workflow with LLM-based routing."""

    workflow = StateGraph(AgentState)

    # === Nodes ===
    workflow.add_node("router", lambda state: query_router_node(state, llm))
    workflow.add_node("data_analysis", lambda state: data_analysis_node(state, data_tools))
    workflow.add_node("search", lambda state: search_node(state, data_tools,llm))
    workflow.add_node("visualization", lambda state: visualization_node(state, viz_agent))
    workflow.add_node("summarization", lambda state: summarization_node(state, df))

    workflow.set_entry_point("router")

    # === LLM Routing Logic ===
    def route_based_on_intent(state):
        intent = state.get("next_action", "").lower()

        if intent in ["visualize", "chart", "plot"]:
            return "visualization"
        if intent in ["search", "retrieve", "fetch"]:
            return "search"
        if intent in ["summarize", "summary"]:
            return "summarization"
        if intent in ["stats", "statistics", "aggregate", "analysis"]:
            return "data_analysis"

        # If LLM output isn't clear → use best fallback
        return "search"

    # === Routing edges ===
    workflow.add_conditional_edges(
        "router",
        route_based_on_intent,
        {
            "data_analysis": "data_analysis",
            "search": "search",
            "visualization": "visualization",
            "summarization": "summarization",
        }
    )

    # Auto-summarize after search
    workflow.add_edge("search", END)

    # Final paths
    workflow.add_edge("data_analysis", END)
    workflow.add_edge("visualization", END)
    workflow.add_edge("summarization", END)

    return workflow.compile()



def process_chat_query(user_query: str, df: pd.DataFrame):
    """Process query through agent system"""
    
    # Initialize state
    initial_state = {
        "messages": [],
        "user_query": user_query,
        "intent": "",
        "data_results": "",
        "visualization_path": "",
        "image_buffer": None,
        "next_action": ""
    }

    agent_graph = st.session_state.get("agent_graph")

    if agent_graph is None:
        raise ValueError("Agent graph not found. Initialize it in Streamlit first.")

    final_state = agent_graph.invoke(initial_state)
    return final_state

    # ****************************
    # # Run through graph
    # final_state = st.session_state.agent_graph.invoke(initial_state)
    
    
    # return final_state

__all__ = [
       'DataQueryTools',
       'VisualizationAgent', 
       'create_agent_graph',
       'process_chat_query'
   ]

