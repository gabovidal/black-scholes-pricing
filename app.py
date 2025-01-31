import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
import matplotlib.ticker as tkr
from matplotlib.patches import Rectangle
from math import ceil, floor
import seaborn as sns


def compute_prices(S, K, r, sigma, T, buy=0):
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
                    padding-left: 2rem;
                    padding-right: 2rem;
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
    st.title("Parameters")
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
                buy = st.number_input("Purchase Value", value=5.5, key='buy')
            else:
                buy = 0
        st.header("Heatmap Settings")
        vol_range = st.slider('Variation of Volatility ($\\pm\\Delta\\sigma/\\sigma$)',
                              min_value=1, max_value=100, value=10, step=1)/100
        spot_range = st.slider('Variation of Current Price ($\\pm\\Delta S/S$)',
                               min_value=1, max_value=15, value=10, step=1)/100

    elif st.session_state.active_tab == 'volatility':
        def get_data(ticker, start_date, end_date):
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            spot_price = stock.info['previousClose']

            # filter by input dates
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            valid_expirations = [
                d for d in exp_dates if start_ts <= pd.Timestamp(d) <= end_ts]

            options_data = []
            for exp in valid_expirations:
                chain = stock.option_chain(exp)
                df = chain.calls if option_type == "Call" else chain.puts
                df['expiration'] = exp
                options_data.append(df)

            df = pd.concat(options_data, ignore_index=True)

            # time to expiration
            today = pd.Timestamp.today()
            df['exp_date'] = pd.to_datetime(df['expiration'])
            df['time'] = (df['exp_date'] - today).dt.days / 365
            # cleaning data
            df['option_price'] = (df['ask']+df['bid'])/2
            df = df[['option_price', 'strike', 'time']].dropna()

            strike = df['strike'].values
            time = df['time'].values
            option_price = df['option_price'].values
            iv = np.array([ImpliedVolatility(spot_price, strike[i], r, time[i], option_price[i],
                                             option_type) for i in range(len(strike))])
            df['iv'] = iv

            df = df.dropna()

            return spot_price, df

        def ImpliedVolatility(S, K, r, T, price, TYPE):
            if price < 0:
                return np.nan

            def fun(sigma):
                call, put = compute_prices(S, K, r, sigma, T)
                value = call if TYPE == 'Calls' else put
                return value - price
            try:
                return brentq(fun, -1e-3, 3)
            except (ValueError, RuntimeError):
                return np.nan

        with col1:
            ticker = st.text_input("Stock Ticker", "AAPL").upper()
            start_date = st.date_input(
                "Start Date", datetime.today() + timedelta(days=2))
            r = st.number_input("Risk-Free Rate ($r$)", value=0.05)
        with col2:
            option_type = st.selectbox("Option Type", ["Calls", "Puts"])
            end_date = st.date_input(
                "End Date", datetime.today() + timedelta(days=92))
        st.header('3D Plot Settings')
        spot_price, df = get_data(ticker, start_date, end_date)
        slide_vol_min = max(df.iv.min(), 0.)
        slide_vol_max = df.iv.max()

        min_iv, max_iv = st.slider('Range of Implied Volatility',
                                   min_value=slide_vol_min, max_value=slide_vol_max, value=(0.1, 2.0), step=0.01)
        min_iv, max_iv = min_iv, max_iv
        slide_strike_min = df.strike.mean()
        slide_strike_max = df.strike.max()
        min_strike, max_strike = st.slider('Range of Strike Price',
                                           min_value=slide_strike_min, max_value=slide_strike_max, value=(slide_strike_min, slide_strike_max), step=0.01)
        df = df[(df['strike'] < max_strike) & (df['strike'] > min_strike) & (
            df['iv'] < max_iv) & (df['iv'] > min_iv)]
        strike = df['strike'].values
        time = df['time'].values
        iv = df['iv'].values

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
        title = f'Expected {TYPE} PnL = {price:+.2f}'
    else:
        title = f'Expected {TYPE} Price = {price:.2f}'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
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
    colorblind = st.checkbox("color blind-friendly palette")
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
    grid_strike, grid_time = np.meshgrid(
        np.linspace(strike.min(), strike.max(), 100),
        np.linspace(time.min(), time.max(), 100)
    )

    grid_iv = griddata((strike, time), iv,
                       (grid_strike, grid_time), 'cubic')  # 'linear'?

    col = st.columns(1)[0]
    bumpy = st.checkbox('contrast bumpiness')
    with col:
        if bumpy:
            grad_x, grad_y = np.gradient(grid_iv)
            grad_xx, grad_xy = np.gradient(grad_x)
            grad_yx, grad_yy = np.gradient(grad_y)
            colormap = np.array([[(grad_xx[i][j]*grad_yy[i][j]+grad_xy[i][j]*grad_yx[i][j])
                                  for j in range(len(grad_x))] for i in range(len(grad_x))])
            colormap = np.log(np.abs(colormap))
        else:
            colormap = None

        fig = go.Figure(data=[
            go.Surface(
                x=grid_strike,
                y=grid_time,
                z=grid_iv,
                colorscale='viridis',
                opacity=0.8,
                contours={
                    "z": {"show": True, "start": iv.min(), "end": iv.max(), "size": 0.02}
                },
                surfacecolor=colormap,
                showscale=False
            )
        ])

        fig.update_layout(
            title=f'Implied Volatility for {ticker} {
                option_type} (no dividends)',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Maturity (years)',
                zaxis_title='Implied Volatility',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=400,
            width=700
        )
        fig.update_scenes(xaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False})

elif st.session_state.active_tab == 'about':
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📈 Option Pricing
        - TODO

        ### 📊 Implied Volatility Surface
        - TODO
        """)
    with col2:
        mu = 0.05  # risk-free interest rate
        sigma = 0.10  # volatility
        S = 100  # (current) spot price
        n = 500  # number of time series steps
        T = 1  # time to maturity
        m = 100  # number of simulations

        dt = T/n
        dlogSt = mu * dt + sigma * \
            np.random.normal(0, np.sqrt(dt), size=(m, n))
        St = S * np.exp(np.cumsum(dlogSt, axis=-1))
        t = np.full(shape=(m, n), fill_value=np.linspace(0, T, n))

        fig, ax = plt.subplots(figure=(6, 5))
        ax.plot(t.T, St.T)
        ax.set_title(f"Realizations of a Geometric Brownian Motion")
        ax.set_xlabel("time ($t$) in years")
        ax.set_ylabel("Stock Price ($S_t$) over time")
        st.pyplot(fig)

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
