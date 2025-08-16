import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date

st.set_page_config(layout="wide")
st.title("OptvizV2 - Schwab Options Analysis")

# --- Schwab API Configuration ---
SCHWAB_API_KEY = os.getenv("SCHWAB_API_KEY")
SCHWAB_API_SECRET = os.getenv("SCHWAB_API_SECRET")
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
SCHWAB_MARKETDATA_URL = "https://api.schwabapi.com/marketdata/v1/chains"

if not SCHWAB_API_KEY or not SCHWAB_API_SECRET:
    st.error("Schwab API Key and Secret not found. Please set SCHWAB_API_KEY and SCHWAB_API_SECRET environment variables.")
    st.stop()

# --- API Functions ---
@st.cache_data(ttl=3500) # Token typically lasts 1 hour, refresh slightly before
def get_access_token():
    """Fetches an access token from Schwab API."""
    try:
        resp = requests.post(
            SCHWAB_TOKEN_URL,
            auth=(SCHWAB_API_KEY, SCHWAB_API_SECRET),
            data={"grant_type": "client_credentials"},
        )
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return resp.json()["access_token"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get access token: {e}")
        st.stop()

@st.cache_data(ttl=300) # Cache option chain for 5 minutes
def fetch_option_chain(sym: str, token: str):
    """Fetches option chain data for a given symbol."""
    params = {
        "symbol": sym,
        "contractType": "ALL", # Ensure we get both CALL and PUT
        "strikeCount": 20, # Fetch a reasonable number of strikes around ATM
        "includeQuotes": "TRUE", # Get bid/ask/last for calculations
        "strategy": "SINGLE", # Simple options
        "range": "ALL", # All expirations
        "fromDate": datetime.now().strftime("%Y-%m-%d"), # From today
        "toDate": (datetime.now() + pd.DateOffset(years=1)).strftime("%Y-%m-%d") # Up to 1 year out
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(SCHWAB_MARKETDATA_URL, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            st.error(f"Invalid symbol or request parameters for {sym}. Error: {e.response.json().get('error', 'Unknown error')}")
        elif e.response.status_code == 401:
            st.error("Authentication failed. Please check your Schwab API credentials.")
        elif e.response.status_code == 404:
            st.error(f"Symbol {sym} not found or no option chain available.")
        else:
            st.error(f"Error fetching data for {sym}: {e}")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching option chain: {e}")
        st.stop()

def parse_chain(js: dict) -> pd.DataFrame:
    """Parses the raw JSON option chain into a pandas DataFrame."""
    recs = []
    today = pd.Timestamp.now().normalize() # Normalize to remove time component for DTE calculation

    # Schwab API returns 'callExpDateMap' and 'putExpDateMap'
    for kind, key in [("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")]:
        exp_map = js.get(key, {})
        for exp_key, strikes_data in exp_map.items():
            # exp_key format: "YYYY-MM-DD:daysToExpiration"
            expiry_date_str = exp_key.split(":")[0]
            expiry_date = pd.to_datetime(expiry_date_str)
            
            for strike_str, options_list in strikes_data.items():
                for option_data in options_list:
                    r = {
                        "expiry": expiry_date,
                        "strike": float(strike_str),
                        "type": kind,
                        "symbol": option_data.get("symbol"), # Option symbol (e.g., SPY240308C500)
                        "description": option_data.get("description"),
                        "bid": option_data.get("bid"),
                        "ask": option_data.get("ask"),
                        "last": option_data.get("last"),
                        "mark": option_data.get("mark"), # Midpoint of bid/ask
                        "openInterest": option_data.get("openInterest"),
                        "totalVolume": option_data.get("totalVolume"),
                        "volatility": option_data.get("volatility"), # Implied Volatility
                        "delta": option_data.get("delta"),
                        "gamma": option_data.get("gamma"),
                        "theta": option_data.get("theta"),
                        "vega": option_data.get("vega"),
                        "rho": option_data.get("rho"),
                        "inTheMoney": option_data.get("inTheMoney"),
                        "multiplier": option_data.get("multiplier", 100) # Default to 100 shares per contract
                    }
                    recs.append(r)
    
    df = pd.DataFrame(recs)
    if not df.empty:
        # Calculate DTE based on normalized dates
        df["dte"] = (df["expiry"] - today).dt.days
        # Filter out options with missing critical data for calculations
        df = df.dropna(subset=["openInterest", "totalVolume", "delta", "gamma", "volatility"])
        # Ensure numeric types
        numeric_cols = ["strike", "bid", "ask", "last", "mark", "openInterest", "totalVolume", 
                        "volatility", "delta", "gamma", "theta", "vega", "rho", "multiplier"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Coerce errors to NaN, then fill 0
        
        # Filter out options with zero or negative volatility, which are often bad data points
        df = df[df["volatility"] > 0]
        
    return df

def compute_metrics(df: pd.DataFrame, underlying_price: float, focus_range: float = None) -> dict:
    """
    Computes various option chain metrics like GEX, DEX, OI, Volatility Surface, and key levels.
    
    Args:
        df (pd.DataFrame): The parsed option chain DataFrame.
        underlying_price (float): The current price of the underlying asset.
        focus_range (float, optional): A percentage range around the underlying price
                                       to focus calculations (e.g., 0.20 for +/- 20%).
                                       Defaults to None (no focus).
    Returns:
        dict: A dictionary containing computed metrics and dataframes.
    """
    
    metrics_output = {
        "gamma_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
        "oi_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
        "vol_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
        "net_gex": pd.DataFrame(columns=["strike", "net_gex"]),
        "net_dex": pd.DataFrame(columns=["strike", "net_dex"]),
        "vol_surface": pd.DataFrame(),
        "vol_term_structure": pd.DataFrame(),
        "levels": {"ATM": None, "Max OI": None, "Max Gamma": None, "Max Pain": None},
    }

    if df.empty:
        return metrics_output
    
    # Filter data if focus_range is specified
    working_df = df.copy()
    if focus_range and underlying_price > 0:
        lower_bound = underlying_price * (1 - focus_range)
        upper_bound = underlying_price * (1 + focus_range)
        working_df = working_df[(working_df["strike"] >= lower_bound) & (working_df["strike"] <= upper_bound)]
        
    if working_df.empty:
        return metrics_output # Return empty metrics if no data after filtering

    # Calculate exposures (gamma_exp and delta_exp)
    # Ensure openInterest and multiplier are not NaN or zero for these calculations
    working_df["gamma_exp"] = working_df["gamma"] * working_df["openInterest"] * working_df["multiplier"]
    working_df["delta_exp"] = working_df["delta"] * working_df["openInterest"] * working_df["multiplier"]

    # Group by strike and type for basic metrics
    metrics_output["gamma_by_type"] = (
        working_df.groupby(["strike", "type"])["gamma_exp"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    metrics_output["oi_by_type"] = (
        working_df.groupby(["strike", "type"])["openInterest"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    metrics_output["vol_by_type"] = (
        working_df.groupby(["strike", "type"])["totalVolume"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )

    # Calculate NET GEX and NET DEX by strike
    net_gex_data = []
    net_dex_data = []
    
    for strike in sorted(working_df["strike"].unique()):
        strike_data = working_df[working_df["strike"] == strike]
        
        call_gex = strike_data[strike_data["type"] == "CALL"]["gamma_exp"].sum()
        put_gex = strike_data[strike_data["type"] == "PUT"]["gamma_exp"].sum()
        net_gex = call_gex - put_gex # Calls add positive gamma, Puts add negative gamma
        
        call_dex = strike_data[strike_data["type"] == "CALL"]["delta_exp"].sum()
        put_dex = strike_data[strike_data["type"] == "PUT"]["delta_exp"].sum()
        net_dex = call_dex + put_dex # Put delta is typically negative, so adding them correctly sums exposure
        
        net_gex_data.append({"strike": strike, "net_gex": net_gex})
        net_dex_data.append({"strike": strike, "net_dex": net_dex})

    metrics_output["net_gex"] = pd.DataFrame(net_gex_data)
    metrics_output["net_dex"] = pd.DataFrame(net_dex_data)

    # Volatility surface data
    # Use both calls and puts for a more complete surface, if available
    metrics_output["vol_surface"] = working_df[
        working_df["volatility"].notna() & (working_df["volatility"] > 0)
    ][["strike", "dte", "volatility", "type", "expiry"]].drop_duplicates()

    # Volatility term structure (ATM vol across expirations)
    vol_term_data = []
    for expiry in sorted(working_df["expiry"].unique()):
        exp_data = working_df[working_df["expiry"] == expiry]
        if not exp_data.empty and not exp_data["volatility"].isna().all():
            # Find ATM strike for this expiry (closest strike to underlying price)
            atm_strike = min(exp_data["strike"].unique(), key=lambda x: abs(x - underlying_price))
            atm_data = exp_data[exp_data["strike"] == atm_strike]
            
            if not atm_data.empty and not atm_data["volatility"].isna().all():
                # Average volatility for ATM calls and puts at this expiry
                avg_vol = atm_data["volatility"].mean()
                dte = atm_data["dte"].iloc[0]
                vol_term_data.append({
                    "dte": dte, 
                    "volatility": avg_vol,
                    "expiry": expiry
                })

    metrics_output["vol_term_structure"] = pd.DataFrame(vol_term_data).sort_values("dte")

    # Calculate key levels (using the full, unfiltered df for these)
    full_df_for_levels = df.copy() # Use the original df for levels
    if not full_df_for_levels.empty:
        strikes = sorted(full_df_for_levels["strike"].unique())
        
        # ATM
        metrics_output["levels"]["ATM"] = min(strikes, key=lambda x: abs(x - underlying_price)) if strikes else None
        
        # Max OI
        oi_by_strike = full_df_for_levels.groupby("strike")["openInterest"].sum()
        metrics_output["levels"]["Max OI"] = oi_by_strike.idxmax() if not oi_by_strike.empty else None
        
        # Max Gamma
        gamma_by_strike = full_df_for_levels.groupby("strike")["gamma_exp"].sum()
        metrics_output["levels"]["Max Gamma"] = gamma_by_strike.idxmax() if not gamma_by_strike.empty else None

        # Max Pain
        pain = {}
        for S in strikes:
            # Loss for calls in the money (strike < S)
            calls_itm = full_df_for_levels[(full_df_for_levels["type"] == "CALL") & (full_df_for_levels["strike"] <= S)]
            call_loss = ((S - calls_itm["strike"]) * calls_itm["openInterest"] * calls_itm["multiplier"]).sum()
            
            # Loss for puts in the money (strike > S)
            puts_itm = full_df_for_levels[(full_df_for_levels["type"] == "PUT") & (full_df_for_levels["strike"] >= S)]
            put_loss = ((puts_itm["strike"] - S) * puts_itm["openInterest"] * puts_itm["multiplier"]).sum()
            
            pain[S] = call_loss + put_loss
        
        metrics_output["levels"]["Max Pain"] = min(pain, key=pain.get) if pain else None

    return metrics_output

def get_0dte_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters for true 0DTE (same day expiration) options."""
    # A true 0DTE option expires today, meaning DTE is 0.
    # Some platforms might consider DTE=1 as "near-term" but 0DTE is strictly today.
    return df[df["dte"] == 0].copy()

# --- Main Streamlit App Logic ---
sym = st.text_input("Enter Ticker Symbol", "").upper().strip()
if not sym:
    st.info("👆 Enter a ticker symbol to get started (e.g., SPY, AAPL)")
    st.stop()

df = pd.DataFrame() # Initialize df to an empty DataFrame
underlying_price = 0.0

try:
    token = get_access_token()
    raw_chain_data = fetch_option_chain(sym, token)
    
    underlying_price = raw_chain_data.get("underlyingPrice", 0.0)
    
    if not underlying_price:
        st.warning(f"Could not retrieve underlying price for {sym}. Some calculations may be inaccurate.")
        
    # Check if we got valid option chain data (at least one call or put map)
    if not raw_chain_data.get("callExpDateMap") and not raw_chain_data.get("putExpDateMap"):
        st.error(f"No options data found for {sym}. This symbol may not have tradeable options or may not be supported by Schwab API.")
        st.stop()
    
    df = parse_chain(raw_chain_data)
    
    if df.empty:
        st.error(f"No valid options data could be parsed for {sym}. This might be due to missing critical data points (e.g., volatility, open interest) in the API response.")
        st.stop()
        
    # Compute metrics for full dataset and a focused range
    met_full = compute_metrics(df, underlying_price)
    met_focused = compute_metrics(df, underlying_price, focus_range=0.20) # +/- 20% of underlying
    
    # Get 0DTE data and compute its metrics
    df_0dte = get_0dte_data(df)
    met_0dte = compute_metrics(df_0dte, underlying_price, focus_range=0.10) # Tighter focus for 0DTE
    
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Display current price
if underlying_price > 0:
    st.metric("Underlying Price", f"${underlying_price:.2f}")
else:
    st.warning("Underlying price not available. Some metrics might be less accurate.")

# =============== TAB STRUCTURE ===============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "GEX/DEX", 
    "Volatility Analysis", 
    "0DTE Analysis",
    "Raw Data"
])

# =============== TAB 1: OVERVIEW ===============
with tab1:
    st.header("Market Overview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Important Price Levels")
        levels_df = (
            pd.DataFrame.from_dict(met_full["levels"], orient="index", columns=["Strike"])
              .reset_index()
              .rename(columns={"index":"Level"})
        )
        st.table(levels_df)
        
        st.subheader("📋 Summary Statistics")
        total_call_oi = met_full["oi_by_type"].get("CALL", pd.Series()).sum()
        total_put_oi = met_full["oi_by_type"].get("PUT", pd.Series()).sum()
        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_0dte_options_count = len(df_0dte)
        total_options_count = len(df)
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Put/Call OI Ratio", f"{pc_ratio:.2f}")
            st.metric("Total Call OI", f"{total_call_oi:,.0f}")
        with metrics_col2:
            st.metric("0DTE Options Count", f"{total_0dte_options_count}")
            st.metric("Total Put OI", f"{total_put_oi:,.0f}")
    
    with col2:
        if total_call_oi > 0 or total_put_oi > 0:
            fig_pie = px.pie(
                values=[total_call_oi, total_put_oi], 
                names=["CALL","PUT"],
                color_discrete_map={"CALL":"green","PUT":"red"},
                title=f"Total Open Interest Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No open interest data to display pie chart.")
        
        if not df.empty:
            st.subheader("Top 10 Strikes by Total Open Interest")
            # Calculate total OI per strike across all expirations and types
            matrix = df.pivot_table(index="strike", columns="type", values="openInterest").fillna(0).round(0)
            matrix_total_oi = matrix.sum(axis=1).sort_values(ascending=False)
            
            if not matrix_total_oi.empty:
                top_strikes = matrix_total_oi.head(10).index
                matrix_display = matrix.loc[top_strikes]
                st.dataframe(matrix_display)
            else:
                st.info("No open interest data to display top strikes.")

# =============== TAB 2: FLOW ANALYSIS (GEX/DEX) ===============
with tab2:
    st.header("Gamma & Delta Exposure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gamma Exposure by Type")
        if not met_focused["gamma_by_type"].empty:
            gdf = met_focused["gamma_by_type"].melt(
                id_vars="strike", 
                value_vars=[col for col in ["CALL","PUT"] if col in met_focused["gamma_by_type"].columns],
                var_name="type", 
                value_name="gamma_exposure"
            )
            fig1 = px.bar(
                gdf, x="strike", y="gamma_exposure", color="type",
                barmode="group",
                color_discrete_map={"CALL":"green","PUT":"red"},
                title="Gamma Exposure by Type (Focused Range)"
            )
            if underlying_price > 0:
                fig1.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No gamma exposure data available for the focused range.")

    with col2:
        st.subheader("Net Gamma Exposure (Call GEX - Put GEX)")
        if not met_focused["net_gex"].empty:
            fig_net_gex = px.bar(
                met_focused["net_gex"], 
                x="strike", 
                y="net_gex",
                title="Net Gamma Exposure (Focused Range)",
                color="net_gex",
                color_continuous_scale=["red", "white", "green"] # Red for negative, Green for positive
            )
            if underlying_price > 0:
                fig_net_gex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            fig_net_gex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_gex, use_container_width=True)
        else:
            st.info("No net gamma exposure data available for the focused range.")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Net Delta Exposure")
        if not met_focused["net_dex"].empty:
            fig_net_dex = px.bar(
                met_focused["net_dex"], 
                x="strike", 
                y="net_dex",
                title="Net Delta Exposure (Focused Range)",
                color="net_dex",
                color_continuous_scale=["red", "white", "blue"] # Red for negative, Blue for positive
            )
            if underlying_price > 0:
                fig_net_dex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            fig_net_dex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_dex, use_container_width=True)
        else:
            st.info("No net delta exposure data available for the focused range.")

    with col4:
        st.subheader("Open Interest by Type")
        if not met_focused["oi_by_type"].empty:
            oidf = met_focused["oi_by_type"].melt(
                id_vars="strike", 
                value_vars=[col for col in ["CALL","PUT"] if col in met_focused["oi_by_type"].columns],
                var_name="type", 
                value_name="open_interest"
            )
            fig2 = px.bar(
                oidf, x="strike", y="open_interest", color="type",
                barmode="group",
                color_discrete_map={"CALL":"green","PUT":"red"},
                title="Open Interest by Type (Focused Range)"
            )
            if underlying_price > 0:
                fig2.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No open interest data available for the focused range.")

# =============== TAB 3: VOLATILITY ANALYSIS ===============
with tab3:
    st.header("Volatility Analysis")
    
    st.subheader("Volatility Term Structure (ATM)")
    if not met_full["vol_term_structure"].empty:
        fig_term = px.line(
            met_full["vol_term_structure"], 
            x="dte", 
            y="volatility",
            markers=True,
            title="Volatility Term Structure (ATM Implied Volatility)"
        )
        fig_term.update_traces(line_color="purple", marker_color="purple")
        fig_term.update_xaxes(title="Days to Expiration (DTE)", range=[0, met_full["vol_term_structure"]["dte"].max() * 1.1])
        fig_term.update_yaxes(title="Implied Volatility")
        st.plotly_chart(fig_term, use_container_width=True)
    else:
        st.info("No volatility term structure data available.")

    st.subheader("3D Implied Volatility Surface")
    # Ensure enough data points for a meaningful 3D surface
    if not met_full["vol_surface"].empty and len(met_full["vol_surface"]["strike"].unique()) > 2 and len(met_full["vol_surface"]["dte"].unique()) > 2:
        # Pivot table for the surface, averaging volatility if multiple options at same strike/dte
        # Using both CALL and PUT data for a more comprehensive surface
        pivot_surface = met_full["vol_surface"].pivot_table(
            index="strike", columns="dte", values="volatility", aggfunc="mean"
        )
        
        if not pivot_surface.empty:
            # Calculate min and max implied volatility from the pivot table values for z-axis range
            z_min = pivot_surface.values.min() * 0.9 if pivot_surface.values.min() > 0 else 0
            z_max = pivot_surface.values.max() * 1.1
            
            fig_surface = go.Figure(data=[go.Surface(
                z=pivot_surface.values,
                x=pivot_surface.columns,
                y=pivot_surface.index,
                colorscale="Viridis",
                colorbar_title="Implied Volatility"
            )])
            
            fig_surface.update_layout(
                title="3D Implied Volatility Surface",
                scene=dict(
                    xaxis=dict(title="Days to Expiration (DTE)", range=[0, pivot_surface.columns.max() * 1.1]),
                    yaxis=dict(title="Strike Price"),
                    zaxis=dict(title="Implied Volatility", range=[z_min, z_max]),
                    aspectmode="auto" # Let Plotly determine aspect ratio
                ),
                height=600
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        else:
            st.info("Not enough data to construct a meaningful 3D volatility surface.")
    else:
        st.info("Not enough data points (strikes or DTEs) to construct a 3D volatility surface.")

    st.subheader("Volatility Skew by Expiration")
    if not met_full["vol_surface"].empty:
        # Get unique expirations, sorted by DTE, show up to 4 for clarity
        unique_dtes_for_skew = sorted(met_full["vol_surface"]["dte"].unique())
        
        if len(unique_dtes_for_skew) > 0:
            # Select a few representative DTEs, e.g., closest, next, etc.
            # For simplicity, let's pick the first 4 available DTEs
            selected_dtes = unique_dtes_for_skew[:4] 
            
            skew_cols = st.columns(len(selected_dtes)) # Create columns dynamically
            
            for i, dte in enumerate(selected_dtes):
                exp_data = met_full["vol_surface"][met_full["vol_surface"]["dte"] == dte]
                
                if not exp_data.empty:
                    fig_skew = px.line(
                        exp_data, 
                        x="strike", 
                        y="volatility", 
                        color="type",
                        markers=True,
                        color_discrete_map={"CALL":"green","PUT":"red"},
                        title=f"Vol Skew - {int(dte)} DTE"
                    )
                    if underlying_price > 0:
                        fig_skew.add_vline(x=underlying_price, line_dash="dash", line_color="blue", annotation_text="Current")
                    fig_skew.update_xaxes(title="Strike Price")
                    fig_skew.update_yaxes(title="Implied Volatility")
                    
                    with skew_cols[i]:
                        st.plotly_chart(fig_skew, use_container_width=True)
                else:
                    with skew_cols[i]:
                        st.info(f"No volatility data for {int(dte)} DTE.")
        else:
            st.info("No unique DTEs found for volatility skew analysis.")
    else:
        st.info("No volatility surface data available for skew analysis.")

# =============== TAB 4: 0DTE ANALYSIS ===============
with tab4:
    st.header("⚡ 0DTE Options Analysis")
    
    if df_0dte.empty:
        st.warning("No 0DTE options found for this symbol. This means no options expire today.")
    else:
        st.info(f"Found {len(df_0dte)} 0DTE options expiring today.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("0DTE Net Gamma Exposure")
            if not met_0dte["net_gex"].empty:
                fig_0dte_gex = px.bar(
                    met_0dte["net_gex"], 
                    x="strike", 
                    y="net_gex",
                    title="0DTE Net Gamma Exposure",
                    color="net_gex",
                    color_continuous_scale=["red", "white", "green"]
                )
                if underlying_price > 0:
                    fig_0dte_gex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
                fig_0dte_gex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_gex, use_container_width=True)
            else:
                st.info("No 0DTE gamma exposure data available.")

        with col2:
            st.subheader("0DTE Net Delta Exposure")
            if not met_0dte["net_dex"].empty:
                fig_0dte_dex = px.bar(
                    met_0dte["net_dex"], 
                    x="strike", 
                    y="net_dex",
                    title="0DTE Net Delta Exposure",
                    color="net_dex",
                    color_continuous_scale=["red", "white", "blue"]
                )
                if underlying_price > 0:
                    fig_0dte_dex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
                fig_0dte_dex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_dex, use_container_width=True)
            else:
                st.info("No 0DTE delta exposure data available.")
        
        st.subheader("0DTE: Net GEX vs Net DEX Scatter")
        if not met_0dte["net_gex"].empty and not met_0dte["net_dex"].empty:
            combined_0dte = met_0dte["net_gex"].merge(met_0dte["net_dex"], on="strike")
            
            fig_combined = px.scatter(
                combined_0dte,
                x="net_gex",
                y="net_dex", 
                size=abs(combined_0dte["net_gex"]) + abs(combined_0dte["net_dex"]), # Size by combined magnitude
                hover_data=["strike"],
                title="0DTE: Net Gamma Exposure vs Net Delta Exposure by Strike",
                labels={"net_gex": "Net Gamma Exposure", "net_dex": "Net Delta Exposure"}
            )
            fig_combined.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_combined.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.info("Not enough 0DTE data to generate the GEX vs DEX scatter plot.")
        
        st.subheader("0DTE Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_0dte_calls = len(df_0dte[df_0dte["type"] == "CALL"])
            st.metric("0DTE Calls Count", total_0dte_calls)
        with col2:
            total_0dte_puts = len(df_0dte[df_0dte["type"] == "PUT"])
            st.metric("0DTE Puts Count", total_0dte_puts)
        with col3:
            avg_0dte_vol = df_0dte["totalVolume"].mean() if not df_0dte.empty else 0
            st.metric("Avg 0DTE Volume", f"{avg_0dte_vol:,.0f}")
        with col4:
            if not df_0dte.empty and not df_0dte["openInterest"].empty:
                max_0dte_oi_strike = df_0dte.loc[df_0dte["openInterest"].idxmax(), "strike"]
                st.metric("Max 0DTE OI Strike", f"${max_0dte_oi_strike:.0f}")
            else:
                st.metric("Max 0DTE OI Strike", "N/A")

# =============== TAB 5: RAW DATA ===============
with tab5:
    st.header("Raw Option Chain Data")
    
    if df.empty:
        st.info("No raw data to display. Please enter a valid ticker.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            option_type_filter = st.selectbox("Filter by Option Type", ["All", "CALL", "PUT"])
        with col2:
            min_dte = st.number_input("Min DTE (Days to Expiration)", value=0, min_value=0)
        with col3:
            max_dte = st.number_input("Max DTE (Days to Expiration)", value=365, min_value=0)
        
        # Apply filters
        filtered_df = df.copy()
        if option_type_filter != "All":
            filtered_df = filtered_df[filtered_df["type"] == option_type_filter]
        filtered_df = filtered_df[(filtered_df["dte"] >= min_dte) & (filtered_df["dte"] <= max_dte)]
        
        st.dataframe(
            filtered_df[[
                "symbol", "strike", "type", "expiry", "dte", "bid", "ask", "last", "mark",
                "openInterest", "totalVolume", "volatility", "delta", "gamma", "theta", "vega", "rho"
            ]],
            use_container_width=True
        )
        
        # Download button
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"{sym}_options_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No data to download after applying filters.")

