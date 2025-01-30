import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
import matplotlib.ticker as tkr
from matplotlib.patches import Rectangle
from math import ceil, floor
import seaborn as sns


def compute_prices(S, K, r, sigma, T, buy):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call-buy, put-buy


def price_heatmap():
    spots = np.linspace(S*(1-spot_range), S*(1+spot_range), 9)
    vols = np.linspace(sigma*(1-vol_range), sigma*(1+vol_range), 9)
    grid_spots = np.full(shape=(9, 9), fill_value=spots)
    grid_vols = np.full(shape=(9, 9), fill_value=vols).T
    return compute_prices(grid_spots, K, r, grid_vols, T, buy)


# configure page
st.set_page_config(
    page_title="Black-Scholes Option Pricing", layout="wide", page_icon="📊", initial_sidebar_state="expanded")
st.title("Black-Scholes Option Pricing")

# reduce paddings
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'prices'


def update_tab(tab_name):
    st.session_state.active_tab = tab_name


# tab buttons in main content area
cols = st.columns(3)
with cols[0]:
    st.button("📈 Option Pricing / PnL", on_click=update_tab, args=('prices',),
              use_container_width=True, type='primary' if st.session_state.active_tab == 'prices' else 'secondary')
with cols[1]:
    st.button("📊 Implied Volatility Surface", on_click=update_tab, args=('volatility',),
              use_container_width=True, type='primary' if st.session_state.active_tab == 'volatility' else 'secondary')
with cols[2]:
    st.button("ℹ️ About", on_click=update_tab, args=('about',),
              use_container_width=True, type='primary' if st.session_state.active_tab == 'about' else 'secondary')

# sidebar content based on active tab
with st.sidebar:
    st.title(" Parameters")
    col1, col2 = st.columns(2)
    if st.session_state.active_tab == 'prices':
        show_pnl = st.checkbox("show PnL")
        with col1:
            S = st.number_input("Current Price ($S$)", value=100.0)
            r = st.number_input("Interest Rate ($r$) in %",
                                value=8., step=0.5)/100
            T = st.number_input("Maturity ($T$) in years",
                                value=1.0, step=1.0/24)
        with col2:
            K = st.number_input("Strike Price ($K$)", value=105.0)
            sigma = st.number_input(
                f"Volatility ($\\sigma$) in %", value=10., step=0.5)/100
            if show_pnl:
                buy = st.number_input("Purchase Value", value=5.5)
            else:
                buy = 0
        st.header("Heatmap Settings")
        vol_range = st.slider('Range of Stock Volatility (% of $\\sigma$)',
                              min_value=1, max_value=100, value=10, step=1)/100
        spot_range = st.slider('Range of Current Stock Price (% of $S$)',
                               min_value=1, max_value=15, value=10, step=1)/100

    elif st.session_state.active_tab == 'volatility':
        with col1:
            r = st.number_input("Risk-Free Rate ($r$)", value=0.05)
            ticker = st.text_input("Stock Ticker", "AAPL").upper()
            start_date = st.date_input(
                "Start Date", datetime.today() + timedelta(days=2))
        with col2:
            q = st.number_input("Dividend Yield ($q$)", value=0.01)
            option_type = st.selectbox("Option Type", ["Calls", "Puts"])
            end_date = st.date_input(
                "End Date", datetime.today() + timedelta(days=92))

    elif st.session_state.active_tab == 'about':
        # st.header("About")
        st.write("Data:")
        st.write("- **Volatility**:")
        st.write("- **Current Price**: ")

    # Global settings
    st.markdown("---")
    st.write('`Created by:`')
    st.write('''Gabriel Vidal<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">''', unsafe_allow_html=True)
    st.write('''<a href="https://linkedin.com/in/gabovidal" style="text-decoration:none; color:inherit;" class="list-group-item"><i class="fab fa-linkedin" style='font-size:24px'></i></a> <a href="https://github.com/gabovidal" style="text-decoration:none; color:inherit;" class="list-group-item"><i class="fab fa-github-square" style='font-size:24px'></i></a>''',
             unsafe_allow_html=True)
    st.write('''''',
             unsafe_allow_html=True)


def plot_heatmap(TYPE, values, ax, vols, spots, palette, show_pnl=False, center=None, vmin=None, vmax=None):
    sns.heatmap(values, cmap=palette, ax=ax, xticklabels=np.round(spots, 2), yticklabels=np.round(
        vols*100, 2), fmt="+.2f" if show_pnl else ".2f", annot=True, center=center, vmin=vmin, vmax=vmax)
    ax.collections[0].colorbar.ax.yaxis.set_ticks([], minor=True)
    price = values[4, 4]
    if show_pnl:
        title = f'{TYPE} PnL $ = {price:+.2f}'
    else:
        title = f'{TYPE} Price $ = {price:.2f}'
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel('Current Stock Price (S)', fontsize=13)
    ax.set_ylabel('Volatility ($\\sigma$) in %', fontsize=13)
    ax.invert_yaxis()
    ax.add_patch(
        Rectangle((4, 0), 1, 9, edgecolor='black', fill=False, lw=1, alpha=0.8))
    ax.add_patch(
        Rectangle((0, 4), 9, 1, edgecolor='black',
                  fill=False, lw=1, alpha=0.8))
    ax.add_patch(
        Rectangle((4, 4), 1, 1, edgecolor='black', fill=False, lw=2))


# Main content based on active tab
if st.session_state.active_tab == 'prices':
    col1, col2 = st.columns([1, 1])
    colorblind = st.checkbox("Color blind-friendly palette")
    palette = 'viridis' if colorblind else 'RdYlGn'
    vols = np.linspace(sigma*(1-vol_range), sigma*(1+vol_range), 9)
    spots = np.linspace(S*(1-spot_range), S*(1+spot_range), 9)
    calls, puts = price_heatmap()
    if show_pnl:
        vmin = min(calls.min(), puts.min())
        vmax = max(calls.max(), puts.max())
        center = 0
    else:
        vmax = max(calls.max(), puts.max())
        vmin = 0
        center = None
    fig_call, ax_call = plt.subplots(figsize=(8, 5))
    fig_put, ax_put = plt.subplots(figsize=(8, 5))
    for col, fig, TYPE, values, ax in [(col1, fig_call, 'CALL', calls, ax_call), (col2, fig_put, 'PUT', puts, ax_put)]:
        with col:
            plot_heatmap(TYPE, values, ax, vols, spots, palette,
                         show_pnl, center, vmin, vmax)
            st.pyplot(fig)

elif st.session_state.active_tab == 'volatility':
    st.header("TODO")
    # Add your price history code here
    # if 'hist_ticker' in locals() and hist_ticker:
    #    # Existing price history implementation
    #    pass

elif st.session_state.active_tab == 'about':
    st.markdown("""
    ### 📈 Option Pricing
    - TODO

    ### 📊 Implied Volatility Surface
    - TODO
    """)

# Add custom CSS for tab styling
st.markdown("""
<style>
    /* Custom tab styling */
    div[data-testid="stButton"] > button {
        transition: all 0.3s ease;
        margin: 2px;
    }
    div[data-testid="stButton"] > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)
