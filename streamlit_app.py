import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# BlackScholes class definition
class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price

# Function to generate heatmaps
def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Initialize session state variables
if 'fetched_price' not in st.session_state:
    st.session_state.fetched_price = 150.0  # Default price
if 'fetched_vol' not in st.session_state:
    st.session_state.fetched_vol = 0.2  # Default volatility
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ""
if 'current_price' not in st.session_state:
    st.session_state.current_price = 150.0
if 'strike' not in st.session_state:
    st.session_state.strike = 100.0
if 'time_to_maturity' not in st.session_state:
    st.session_state.time_to_maturity = 1.0
if 'volatility' not in st.session_state:
    st.session_state.volatility = 0.2
if 'interest_rate' not in st.session_state:
    st.session_state.interest_rate = 0.05

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/joseph-fori-5393a7128/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Fori Joseph</a>', unsafe_allow_html=True)

    st.write("`Inspred by:`")
    linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Prudhvi Reddy</a>', unsafe_allow_html=True)

    # Input: Ticker Symbol
    st.sidebar.markdown("### 📈 Real-Time Market Data")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")

    # Only fetch if ticker changed
    if ticker != st.session_state.last_ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")

            if not hist.empty:
                # Price
                current_price = hist['Close'].iloc[-1]
                st.session_state.fetched_price = round(current_price, 2)

                # Volatility from log returns
                hist['LogReturn'] = np.log(hist['Close'] / hist['Close'].shift(1))
                daily_vol = hist['LogReturn'].std()
                annual_vol = round(daily_vol * np.sqrt(252), 4)
                st.session_state.fetched_vol = annual_vol

                st.session_state.last_ticker = ticker
                st.sidebar.success(f"Fetched: ${current_price:.2f}, Vol: {annual_vol:.2%}")
            else:
                st.sidebar.warning("No data available.")

        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")

    # Let user adjust, pre-filled with fetched values
    st.session_state.current_price = st.sidebar.number_input(
        "Current Asset Price", value=st.session_state.fetched_price
    )
    st.session_state.strike = st.number_input(
        "Strike Price", value=st.session_state.strike
    )
    st.session_state.time_to_maturity = st.number_input(
        "Time to Maturity (Years)", value=st.session_state.time_to_maturity
    )
    st.session_state.volatility = st.sidebar.number_input(
        "Volatility (σ)", value=st.session_state.fetched_vol
    )
    st.session_state.interest_rate = st.number_input(
        "Risk-Free Interest Rate", value=st.session_state.interest_rate
    )

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=st.session_state.current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=st.session_state.current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=st.session_state.volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=st.session_state.volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [st.session_state.current_price],
    "Strike Price": [st.session_state.strike],
    "Time to Maturity (Years)": [st.session_state.time_to_maturity],
    "Volatility (σ)": [st.session_state.volatility],
    "Risk-Free Interest Rate": [st.session_state.interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(
    st.session_state.time_to_maturity, 
    st.session_state.strike, 
    st.session_state.current_price, 
    st.session_state.volatility, 
    st.session_state.interest_rate
)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, st.session_state.strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, st.session_state.strike)
    st.pyplot(heatmap_fig_put)