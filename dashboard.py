
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Import backend components
from core import get_agent_framework
from agents.deals import Opportunity
from agents.search_agent import SearchAgent # NEW

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="ValuAI | Autonomous Arbitrage Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark Mode Look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #00BA51;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #363945;
        border-radius: 5px;
        padding: 15px;
    }
    .stButton>button {
        background-color: #00BA51;
        color: white;
        border-radius: 5px;
        height: 3em; 
    }
</style>
""", unsafe_allow_html=True)

#header
st.title("🤖 ValuAI Dashboard")
st.markdown("### Real-Time Market Intelligence Engine")

# ==========================================
# Input Section (The "Google" Bar)
# ==========================================
with st.container():
    c1, c2 = st.columns([4, 1])
    with c1:
        user_query = st.text_input("", placeholder="Search for products (e.g., 'Sony WH-1000XM5 headphones')...", label_visibility="collapsed")
    with c2:
        scan_btn = st.button("🚀 Find Deals", use_container_width=True)

# ==========================================
# Logic
# ==========================================
if "opportunities" not in st.session_state:
    st.session_state.opportunities = []

if scan_btn:
    st.session_state.opportunities = [] # clear previous
    
    # Progress UI
    with st.status("🔍 Active Search Agent Initialized...", expanded=True) as status:
        
        # 1. Search Phase
        status.write(f"Searching active markets for: **{user_query}**")
        search_agent = SearchAgent()
        
        if user_query:
            selection = search_agent.scan_query(user_query)
        else:
            # Fallback to RSS if empty
            from agents.scanner_agent import ScannerAgent
            scanner = ScannerAgent()
            selection = scanner.scan() # fallback
            
        if not selection or not selection.deals:
            status.update(label="❌ No deals found", state="error")
            st.error("No valid listings found. Try a different query.")
        else:
            status.write(f"✅ Found {len(selection.deals)} raw listings. Pricing...")
            
            # 2. Valuation Phase
            framework = get_agent_framework()
            opportunities = []
            
            progress_bar = st.progress(0)
            total = len(selection.deals)
            
            for i, deal in enumerate(selection.deals):
                status.write(f"Valuing Item {i+1}: {deal.product_description[:30]}...")
                
                # Direct Ensemble Call
                estimate = framework.ensemble.price(deal.product_description)
                
                discount = estimate - deal.price
                opp = Opportunity(deal=deal, estimate=estimate, discount=discount)
                opportunities.append(opp)
                
                progress_bar.progress((i+1)/total)
                
            st.session_state.opportunities = opportunities
            status.update(label="✅ Analysis Complete!", state="complete")

# ==========================================
# Results View
# ==========================================
if st.session_state.opportunities:
    st.markdown("---")
    opps = st.session_state.opportunities
    
    # 3. Visualization
    for opp in opps:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{opp.deal.product_description[:50]}...")
                st.write(f"**URL**: {opp.deal.url}")
                st.caption(opp.deal.product_description)
            with col2:
                delta_color = "normal" if opp.discount > 0 else "inverse"
                st.metric("Profit Potential", f"${opp.discount:.2f}", f"${opp.estimate:.2f} Est.", delta_color=delta_color)
                st.write(f"Ask: ${opp.deal.price:.2f}")
            st.divider()
