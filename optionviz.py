import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Schwab Options Viz",
    initial_sidebar_state="expanded"
)
st.title("Schwab Options Chain Visualizer")

# --- Configuration & Constants ---
SCHWAB_API_BASE_URL = "https://api.schwabapi.com"
MARKETDATA_API_URL = f"{SCHWAB_API_BASE_URL}/marketdata/v1"
OAUTH_TOKEN_URL = f"{SCHWAB_API_BASE_URL}/v1/oauth/token"

# --- API Interaction Functions ---

@st.cache_data(ttl=3600) # Token valid for 1 hour, cache for 1 hour
def get_schwab_access_token():
    """Fetches an access token from Schwab API using client credentials."""
    api_key = os.getenv("SCHWAB_API_KEY")
    api_secret = os.getenv("SCHWAB_API_SECRET")

    if not api_key or not api_secret:
        st.error("Schwab API Key or Secret not found in environment variables.")
        st.info("Please set SCHWAB_API_KEY and SCHWAB_API_SECRET.")
        st.stop()

    try:
        response = requests.post(
            OAUTH_TOKEN_URL,
            auth=(api_key, api_secret),
            data={"grant_type": "client_credentials"},
            timeout=10
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to obtain Schwab access token: {e}")
        st.stop()

@st.cache_data(ttl=300) # Cache option chain data for 5 minutes
def fetch_schwab_option_chain(symbol: str, access_token: str) -> dict:
    """Fetches option chain data for a given symbol from Schwab API."""
    endpoint = f"{MARKETDATA_API_URL}/chains"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "symbol": symbol,
        "strikeCount": 20, # Limit strikes to reduce data size and focus on relevant ones
        "strategy": "SINGLE", # Only fetch single options
        "includeQuotes": "TRUE" # Ensure quotes are included for bid/ask/last
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            st.error(f"Invalid symbol '{symbol}'. Please check the ticker symbol.")
        elif e.response.status_code == 404:
            st.error(f"No options data found for '{symbol}'. It might not have tradeable options or is not supported.")
        else:
            st.error(f"Error fetching data for '{symbol}': {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching data for '{symbol}': {e}")
        return None

# --- Data Parsing and Cleaning ---

def parse_and_clean_option_data(raw_json_data: dict) -> pd.DataFrame:
    """
    Parses raw Schwab option chain JSON into a DataFrame and performs initial cleaning.
    """
    options_records = []
    current_date = pd.Timestamp.now().normalize() # For DTE calculation

    # Extract underlying price
    underlying_price = raw_json_data.get("underlyingPrice", 0)

    for option_type, date_map_key in [("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")]:
        for exp_date_str_full, strikes_data in raw_json_data.get(date_map_key, {}).items():
            # exp_date_str_full format: YYYY-MM-DD:milliseconds_timestamp
            expiry_date = pd.to_datetime(exp_date_str_full.split(":")[0])

            for strike_price_str, option_details_list in strikes_data.items():
                for detail in option_details_list:
                    record = {
                        "symbol": detail.get("symbol"),
                        "description": detail.get("description"),
                        "type": option_type,
                        "strike": float(strike_price_str),
                        "expiry": expiry_date,
                        "bid": detail.get("bid"),
                        "ask": detail.get("ask"),
                        "last": detail.get("last"),
                        "openInterest": detail.get("openInterest"),
                        "totalVolume": detail.get("totalVolume"),
                        "volatility": detail.get("volatility"), # Implied Volatility
                        "delta": detail.get("delta"),
                        "gamma": detail.get("gamma"),
                        "theta": detail.get("theta"),
                        "vega": detail.get("vega"),
                        "rho": detail.get("rho"),
                        "inTheMoney": detail.get("inTheMoney"),
                        "multiplier": detail.get("multiplier", 100) # Default to 100 shares per contract
                    }
                    options_records.append(record)

    df = pd.DataFrame(options_records)

    if df.empty:
        return df, underlying_price

    # --- Data Cleaning and Type Conversion ---
    df["dte"] = (df["expiry"] - current_date).dt.days
    df = df[df["dte"] >= 0].copy() # Remove expired options

    numeric_cols = [
        "bid", "ask", "last", "openInterest", "totalVolume",
        "volatility", "delta", "gamma", "theta", "vega", "rho", "multiplier"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN for OI and Volume with 0, as missing means no activity
    df["openInterest"] = df["openInterest"].fillna(0)
    df["totalVolume"] = df["totalVolume"].fillna(0)

    # Filter out rows with critical NaNs for calculations
    # For Greeks and Volatility, these must be present
    df.dropna(subset=["volatility", "delta", "gamma"], inplace=True)

    # Filter out unrealistic volatility values (e.g., 0 or extremely high)
    df = df[(df["volatility"] > 0.01) & (df["volatility"] < 500)].copy() # 0.01% to 500%

    return df, underlying_price

# --- Metric Calculation Functions ---

def calculate_greeks_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates gamma and delta exposure per contract."""
    df_copy = df.copy()
    df_copy["gamma_exp"] = df_copy["gamma"] * df_copy["openInterest"] * df_copy["multiplier"]
    df_copy["delta_exp"] = df_copy["delta"] * df_copy["openInterest"] * df_copy["multiplier"]
    return df_copy

def compute_net_exposures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Net Gamma Exposure (GEX) and Net Delta Exposure (DEX) by strike."""
    net_gex_data = []
    net_dex_data = []

    for strike in sorted(df["strike"].unique()):
        strike_data = df[df["strike"] == strike]
        
        call_gex = strike_data[strike_data["type"] == "CALL"]["gamma_exp"].sum()
        put_gex = strike_data[strike_data["type"] == "PUT"]["gamma_exp"].sum()
        net_gex = call_gex - put_gex # Calls add positive gamma, Puts add negative gamma (when short)
        
        call_dex = strike_data[strike_data["type"] == "CALL"]["delta_exp"].sum()
        put_dex = strike_data[strike_data["type"] == "PUT"]["delta_exp"].sum()
        net_dex = call_dex + put_dex # Put delta is negative, so adding them correctly sums exposure
        
        net_gex_data.append({"strike": strike, "net_gex": net_gex})
        net_dex_data.append({"strike": strike, "net_dex": net_dex})

    return pd.DataFrame(net_gex_data), pd.DataFrame(net_dex_data)

def compute_oi_volume_by_type(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Open Interest and Total Volume by strike and option type."""
    oi_by_type = (
        df.groupby(["strike", "type"])["openInterest"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    vol_by_type = (
        df.groupby(["strike", "type"])["totalVolume"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    return oi_by_type, vol_by_type

def compute_key_levels(df: pd.DataFrame, underlying_price: float) -> dict:
    """Calculates important price levels like ATM, Max OI, Max Gamma, Max Pain."""
    levels = {"ATM": None, "Max OI": None, "Max Gamma": None, "Max Pain": None}
    strikes = sorted(df["strike"].unique())

    if not strikes:
        return levels

    # ATM Strike
    if underlying_price > 0:
        levels["ATM"] = min(strikes, key=lambda x: abs(x - underlying_price))

    # Max OI Strike
    oi_by_strike = df.groupby("strike")["openInterest"].sum()
    if not oi_by_strike.empty:
        levels["Max OI"] = oi_by_strike.idxmax()

    # Max Gamma Strike
    gamma_by_strike = df.groupby("strike")["gamma_exp"].sum()
    if not gamma_by_strike.empty:
        levels["Max Gamma"] = gamma_by_strike.idxmax()

    # Max Pain Strike
    pain_points = {}
    for S in strikes:
        calls_itm = df[(df["type"] == "CALL") & (df["strike"] <= S)]
        call_loss = ((S - calls_itm["strike"]) * calls_itm["openInterest"] * calls_itm["multiplier"]).sum()
        
        puts_itm = df[(df["type"] == "PUT") & (df["strike"] >= S)]
        put_loss = ((puts_itm["strike"] - S) * puts_itm["openInterest"] * puts_itm["multiplier"]).sum()
        
        pain_points[S] = call_loss + put_loss
    
    if pain_points:
        levels["Max Pain"] = min(pain_points, key=pain_points.get)

    return levels

def compute_volatility_term_structure(df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
    """Calculates ATM implied volatility across different expirations."""
    vol_term_data = []
    for expiry in sorted(df["expiry"].unique()):
        exp_data = df[df["expiry"] == expiry]
        if not exp_data.empty and not exp_data["volatility"].isna().all():
            atm_strike = min(exp_data["strike"].unique(), key=lambda x: abs(x - underlying_price))
            atm_options = exp_data[exp_data["strike"] == atm_strike]
            
            if not atm_options.empty and not atm_options["volatility"].isna().all():
                avg_vol = atm_options["volatility"].mean()
                dte = atm_options["dte"].iloc[0]
                vol_term_data.append({
                    "dte": dte,
                    "volatility": avg_vol,
                    "expiry": expiry
                })
    return pd.DataFrame(vol_term_data).sort_values("dte")

def prepare_volatility_surface_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for the 3D volatility surface, including filtering and aggregation.
    """
    # Filter for options with valid volatility and some liquidity
    vol_df = df[
        (df["volatility"].notna()) &
        (df["volatility"] > 0) &
        (df["openInterest"] > 0) # Focus on liquid options
    ].copy()

    if vol_df.empty:
        return pd.DataFrame()

    # Aggregate volatility by strike and DTE (average of calls/puts)
    # This creates the grid for the surface
    vol_surface_pivot = vol_df.groupby(["strike", "dte"])["volatility"].mean().unstack(fill_value=np.nan)

    # Optional: Interpolate missing values for a smoother surface
    # Use linear interpolation along both DTE (columns) and Strike (index)
    # This can fill small gaps but might create artificial smoothness if data is very sparse.
    # Uncomment if the surface is too jagged.
    # vol_surface_pivot = vol_surface_pivot.interpolate(method='linear', axis=1) # Interpolate across DTEs
    # vol_surface_pivot = vol_surface_pivot.interpolate(method='linear', axis=0) # Interpolate across Strikes

    # Ensure columns (DTE) and index (Strikes) are sorted
    vol_surface_pivot.columns = pd.to_numeric(vol_surface_pivot.columns)
    vol_surface_pivot = vol_surface_pivot.sort_index(axis=1) # Sort DTEs
    vol_surface_pivot = vol_surface_pivot.sort_index(axis=0) # Sort Strikes

    return vol_surface_pivot

# --- Main Data Processing Orchestrator ---

def process_option_data(df_raw: pd.DataFrame, underlying_price: float, focus_range: float = None):
    """
    Orchestrates all metric calculations for a given DataFrame.
    Applies focus_range filtering if specified.
    """
    df = df_raw.copy()
    if focus_range and underlying_price > 0:
        lower_bound = underlying_price * (1 - focus_range)
        upper_bound = underlying_price * (1 + focus_range)
        df = df[(df["strike"] >= lower_bound) & (df["strike"] <= upper_bound)]
        
        if df.empty: # If filtering results in empty df, return empty metrics
            return {
                "net_gex": pd.DataFrame(), "net_dex": pd.DataFrame(),
                "oi_by_type": pd.DataFrame(), "vol_by_type": pd.DataFrame(),
                "gamma_by_type": pd.DataFrame(), "vol_surface": pd.DataFrame(),
                "vol_term_structure": pd.DataFrame(), "levels": {}
            }

    df_with_exposure = calculate_greeks_exposure(df)
    net_gex_df, net_dex_df = compute_net_exposures(df_with_exposure)
    oi_by_type_df, vol_by_type_df = compute_oi_volume_by_type(df_with_exposure)
    
    # Gamma by type needs to be computed from df_with_exposure
    gamma_by_type_df = (
        df_with_exposure.groupby(["strike", "type"])["gamma_exp"]
                        .sum()
                        .unstack(fill_value=0)
                        .reset_index()
    )

    levels = compute_key_levels(df_with_exposure, underlying_price)
    vol_term_structure_df = compute_volatility_term_structure(df_with_exposure, underlying_price)
    vol_surface_pivot_df = prepare_volatility_surface_data(df_with_exposure) # Use df_with_exposure for surface

    return {
        "net_gex": net_gex_df,
        "net_dex": net_dex_df,
        "oi_by_type": oi_by_type_df,
        "vol_by_type": vol_by_type_df,
        "gamma_by_type": gamma_by_type_df,
        "levels": levels,
        "vol_term_structure": vol_term_structure_df,
        "vol_surface": vol_surface_pivot_df, # This is now the pivot table
        "raw_filtered_df": df_with_exposure # Return the filtered df for other uses like skew
    }

# --- Plotting Functions ---

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None,
                   barmode: str = "group", title: str = "", vline_val: float = None,
                   color_map: dict = None, showlegend: bool = True):
    """Generic bar chart plotter."""
    if df.empty:
        st.info(f"No data to plot: {title}")
        return

    fig = px.bar(
        df, x=x_col, y=y_col, color=color_col,
        barmode=barmode, title=title, color_discrete_map=color_map
    )
    if vline_val is not None and vline_val > 0:
        fig.add_vline(x=vline_val, line_dash="dash", line_color="white", annotation_text="Current Price")
    fig.update_layout(showlegend=showlegend)
    st.plotly_chart(fig, use_container_width=True)

def plot_volatility_surface(vol_surface_pivot_df: pd.DataFrame):
    """Plots the 3D implied volatility surface."""
    if vol_surface_pivot_df.empty or vol_surface_pivot_df.shape[0] < 2 or vol_surface_pivot_df.shape[1] < 2:
        st.info("Not enough data points to render a meaningful 3D volatility surface. Need at least 2 strikes and 2 DTEs.")
        return

    z_values = vol_surface_pivot_df.values[~np.isnan(vol_surface_pivot_df.values)]
    z_min = np.min(z_values) * 0.9 if z_values.size > 0 else 0
    z_max = np.max(z_values) * 1.1 if z_values.size > 0 else 100

    fig = go.Figure(data=[go.Surface(
        z=vol_surface_pivot_df.values,
        x=vol_surface_pivot_df.columns,
        y=vol_surface_pivot_df.index,
        colorscale="Viridis",
        colorbar_title="Implied Volatility (%)",
        contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
    )])

    fig.update_layout(
        title="3D Implied Volatility Surface",
        scene=dict(
            xaxis_title="Days to Expiration (DTE)",
            yaxis_title="Strike Price",
            zaxis_title="Implied Volatility (%)",
            xaxis=dict(range=[0, vol_surface_pivot_df.columns.max() * 1.1 if not vol_surface_pivot_df.columns.empty else 100]),
            zaxis=dict(range=[z_min, z_max]),
            aspectmode="auto",
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
        ),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_volatility_skew(df_raw_filtered: pd.DataFrame, underlying_price: float):
    """Plots volatility skew for selected expirations."""
    if df_raw_filtered.empty:
        st.info("No volatility data available to show skew.")
        return

    st.subheader("Volatility Skew by Expiration")
    unique_dtes_for_skew = sorted(df_raw_filtered["dte"].unique())[:4] # Show first 4 expirations

    if unique_dtes_for_skew:
        skew_cols = st.columns(len(unique_dtes_for_skew))
        for i, dte in enumerate(unique_dtes_for_skew):
            exp_data = df_raw_filtered[df_raw_filtered["dte"] == dte]
            
            if not exp_data.empty:
                fig_skew = px.line(
                    exp_data,
                    x="strike",
                    y="volatility",
                    color="type",
                    markers=True,
                    color_discrete_map={"CALL":"green","PUT":"red"},
                    title=f"Vol Skew - {int(dte)} DTE",
                    labels={"volatility": "Implied Volatility (%)"}
                )
                fig_skew.add_vline(x=underlying_price, line_dash="dash", line_color="blue", annotation_text="Current")
                fig_skew.update_yaxes(tickformat=".2f")
                with skew_cols[i]:
                    st.plotly_chart(fig_skew, use_container_width=True)
    else:
        st.info("No distinct expirations with volatility data to show skew.")

# --- Main Streamlit Application Flow ---

def main():
    ticker_symbol = st.text_input("Enter Ticker Symbol", "").upper().strip()

    if not ticker_symbol:
        st.info("👆 Please enter a ticker symbol to analyze its options chain.")
        st.stop()

    # --- Data Fetching and Initial Processing ---
    df_full_chain = pd.DataFrame()
    underlying_price = 0

    with st.spinner(f"Fetching and processing data for {ticker_symbol}..."):
        access_token = get_schwab_access_token()
        if not access_token:
            st.stop()

        raw_json_data = fetch_schwab_option_chain(ticker_symbol, access_token)
        if not raw_json_data:
            st.stop()

        df_full_chain, underlying_price = parse_and_clean_option_data(raw_json_data)

        if df_full_chain.empty:
            st.error(f"No valid options data could be parsed for {ticker_symbol}. This might be due to no tradeable options or data issues after cleaning.")
            st.stop()

    st.metric("Underlying Price", f"${underlying_price:.2f}")

    # --- Compute Metrics for different views ---
    metrics_full = process_option_data(df_full_chain, underlying_price)
    metrics_focused = process_option_data(df_full_chain, underlying_price, focus_range=0.20) # +/- 20% for GEX/DEX charts

    df_0dte = df_full_chain[df_full_chain["dte"] == 0].copy() # True 0DTE
    metrics_0dte = process_option_data(df_0dte, underlying_price, focus_range=0.10) # Tighter focus for 0DTE

    # --- TAB STRUCTURE ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "GEX/DEX Analysis",
        "Volatility Insights",
        "0DTE Deep Dive",
        "Raw Data Explorer"
    ])

    # =============== TAB 1: OVERVIEW ===============
    with tab1:
        st.header("📊 Market Overview")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📍 Key Price Levels")
            levels_df = (
                pd.DataFrame.from_dict(metrics_full["levels"], orient="index", columns=["Strike"])
                  .reset_index()
                  .rename(columns={"index":"Level"})
            )
            st.dataframe(levels_df, use_container_width=True)

            st.subheader("📋 Summary Statistics")
            total_call_oi = metrics_full["oi_by_type"].get("CALL", pd.Series()).sum()
            total_put_oi = metrics_full["oi_by_type"].get("PUT", pd.Series()).sum()
            pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Call OI", f"{total_call_oi:,.0f}")
                st.metric("Total Put OI", f"{total_put_oi:,.0f}")
            with metrics_col2:
                st.metric("Put/Call OI Ratio", f"{pc_ratio:.2f}")
                st.metric("Total 0DTE Options", f"{len(df_0dte):,}")

        with col2:
            if total_call_oi > 0 or total_put_oi > 0:
                fig_pie = px.pie(
                    values=[total_call_oi, total_put_oi],
                    names=["CALL","PUT"],
                    color_discrete_map={"CALL":"green","PUT":"red"},
                    title="Total Open Interest Distribution (Calls vs Puts)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No open interest data to display pie chart.")

            st.subheader("Top 10 Strikes by Open Interest")
            if not df_full_chain.empty:
                oi_matrix = df_full_chain.groupby(["strike", "type"])["openInterest"].sum().unstack(fill_value=0)
                oi_matrix["Total OI"] = oi_matrix.sum(axis=1)
                oi_matrix_display = oi_matrix.sort_values("Total OI", ascending=False).head(10).drop(columns=["Total OI"])
                st.dataframe(oi_matrix_display.round(0), use_container_width=True)
            else:
                st.info("No options data available to display top strikes.")

    # =============== TAB 2: GEX/DEX ANALYSIS ===============
    with tab2:
        st.header("📈 Gamma & Delta Exposure Analysis")
        st.markdown("*(Charts focused on +/- 20% range around underlying price)*")

        col1, col2 = st.columns(2)

        with col1:
            plot_bar_chart(
                metrics_focused["gamma_by_type"].melt(
                    id_vars="strike",
                    value_vars=[col for col in ["CALL","PUT"] if col in metrics_focused["gamma_by_type"].columns],
                    var_name="type",
                    value_name="gamma_exposure"
                ),
                x_col="strike", y_col="gamma_exposure", color_col="type",
                title="Gamma Exposure by Type (Total Contracts * Gamma)",
                vline_val=underlying_price, color_map={"CALL":"green","PUT":"red"}
            )

        with col2:
            plot_bar_chart(
                metrics_focused["net_gex"],
                x_col="strike", y_col="net_gex", color_col="net_gex",
                title="Net Gamma Exposure (Call Gamma - Put Gamma)",
                vline_val=underlying_price, showlegend=False
            )

        col3, col4 = st.columns(2)

        with col3:
            plot_bar_chart(
                metrics_focused["net_dex"],
                x_col="strike", y_col="net_dex", color_col="net_dex",
                title="Net Delta Exposure (Total Contracts * Delta)",
                vline_val=underlying_price, showlegend=False
            )

        with col4:
            plot_bar_chart(
                metrics_focused["oi_by_type"].melt(
                    id_vars="strike",
                    value_vars=[col for col in ["CALL","PUT"] if col in metrics_focused["oi_by_type"].columns],
                    var_name="type",
                    value_name="open_interest"
                ),
                x_col="strike", y_col="open_interest", color_col="type",
                title="Open Interest by Type",
                vline_val=underlying_price, color_map={"CALL":"green","PUT":"red"}
            )

    # =============== TAB 3: VOLATILITY ANALYSIS ===============
    with tab3:
        st.header("📉 Volatility Insights")

        st.subheader("Volatility Term Structure (ATM Implied Volatility)")
        if not metrics_full["vol_term_structure"].empty:
            fig_term = px.line(
                metrics_full["vol_term_structure"],
                x="dte",
                y="volatility",
                markers=True,
                title="ATM Implied Volatility Across Expirations",
                labels={"dte": "Days to Expiration", "volatility": "Implied Volatility (%)"}
            )
            fig_term.update_traces(line_color="purple", marker_color="purple")
            fig_term.update_xaxes(range=[0, metrics_full["vol_term_structure"]["dte"].max() * 1.1])
            fig_term.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_term, use_container_width=True)
        else:
            st.info("No volatility term structure data available.")

        plot_volatility_surface(metrics_full["vol_surface"])
        plot_volatility_skew(metrics_full["raw_filtered_df"], underlying_price)


    # =============== TAB 4: 0DTE DEEP DIVE ===============
    with tab4:
        st.header("⚡ 0DTE Options Deep Dive")

        if df_0dte.empty:
            st.warning("No 0DTE (expiring today) options found for this symbol. This could mean no options expire today or data is unavailable.")
        else:
            st.info(f"Found {len(df_0dte):,} 0DTE options expiring today.")

            col1, col2 = st.columns(2)

            with col1:
                plot_bar_chart(
                    metrics_0dte["net_gex"],
                    x_col="strike", y_col="net_gex", color_col="net_gex",
                    title="0DTE Net Gamma Exposure",
                    vline_val=underlying_price, showlegend=False
                )

            with col2:
                plot_bar_chart(
                    metrics_0dte["net_dex"],
                    x_col="strike", y_col="net_dex", color_col="net_dex",
                    title="0DTE Net Delta Exposure",
                    vline_val=underlying_price, showlegend=False
                )

            st.subheader("0DTE: Net GEX vs Net DEX Scatter")
            if not metrics_0dte["net_gex"].empty and not metrics_0dte["net_dex"].empty:
                combined_0dte = metrics_0dte["net_gex"].merge(metrics_0dte["net_dex"], on="strike")
                combined_0dte["magnitude"] = np.sqrt(combined_0dte["net_gex"]**2 + combined_0dte["net_dex"]**2)

                fig_combined = px.scatter(
                    combined_0dte,
                    x="net_gex",
                    y="net_dex",
                    size="magnitude",
                    hover_data=["strike", "net_gex", "net_dex"],
                    title="0DTE: Net Gamma Exposure vs Net Delta Exposure by Strike",
                    labels={"net_gex": "Net Gamma Exposure", "net_dex": "Net Delta Exposure"}
                )
                fig_combined.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_combined.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_combined, use_container_width=True)
            else:
                st.info("Not enough 0DTE GEX/DEX data to create a combined scatter plot.")

            st.subheader("0DTE Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_0dte_calls = len(df_0dte[df_0dte["type"] == "CALL"])
                st.metric("0DTE Calls", total_0dte_calls)
            with col2:
                total_0dte_puts = len(df_0dte[df_0dte["type"] == "PUT"])
                st.metric("0DTE Puts", total_0dte_puts)
            with col3:
                avg_0dte_vol = df_0dte["totalVolume"].mean() if not df_0dte.empty else 0
                st.metric("Avg Volume (0DTE)", f"{avg_0dte_vol:,.0f}")
            with col4:
                max_0dte_oi_strike = df_0dte.loc[df_0dte["openInterest"].idxmax(), "strike"] if not df_0dte.empty and not df_0dte["openInterest"].empty else 0
                st.metric("Max OI Strike (0DTE)", f"${max_0dte_oi_strike:.0f}")

    # =============== TAB 5: RAW DATA EXPLORER ===============
    with tab5:
        st.header("🔍 Raw Option Chain Data Explorer")

        col1, col2, col3 = st.columns(3)
        with col1:
            option_type_filter = st.selectbox("Filter by Option Type", ["All", "CALL", "PUT"])
        with col2:
            min_dte_filter = st.number_input("Minimum DTE", value=0, min_value=0)
        with col3:
            max_dte_filter = st.number_input("Maximum DTE", value=365, min_value=0)

        filtered_df_display = df_full_chain.copy()
        if option_type_filter != "All":
            filtered_df_display = filtered_df_display[filtered_df_display["type"] == option_type_filter]
        filtered_df_display = filtered_df_display[
            (filtered_df_display["dte"] >= min_dte_filter) &
            (filtered_df_display["dte"] <= max_dte_filter)
        ]

        st.dataframe(
            filtered_df_display[[
                "symbol", "strike", "type", "expiry", "dte", "bid", "ask", "last",
                "openInterest", "totalVolume", "volatility", "delta", "gamma", "theta", "vega", "rho"
            ]].round(4),
            use_container_width=True
        )

        if not filtered_df_display.empty:
            csv = filtered_df_display.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"{ticker_symbol}_options_data_filtered.csv",
                mime="text/csv"
            )
        else:
            st.info("No data to download after applying filters.")

if __name__ == "__main__":
    main()
