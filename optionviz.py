import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="OptvizV2 - Schwab Options Analysis")
st.title("OptvizV2: Schwab Options Chain Analysis")

# --- API Configuration and Caching ---

@st.cache_data(ttl=3600) # Cache token for 1 hour
def get_access_token():
    """Fetches an access token from Schwab API."""
    try:
        resp = requests.post(
            "https://api.schwabapi.com/v1/oauth/token",
            auth=(os.getenv("SCHWAB_API_KEY"), os.getenv("SCHWAB_API_SECRET")),
            data={"grant_type": "client_credentials"},
            timeout=10 # Add a timeout for the request
        )
        resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return resp.json()["access_token"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get access token from Schwab API: {e}")
        st.stop()

@st.cache_data(ttl=300) # Cache option chain for 5 minutes
def fetch_option_chain(sym: str, token: str):
    """Fetches option chain data for a given symbol."""
    url = "https://api.schwabapi.com/marketdata/v1/chains"
    params = {
        "symbol": sym,
        "strikeCount": 20, # Limit strikes to reduce data size and focus on relevant ones
        "strategy": "SINGLE", # Only fetch single options
        "includeQuotes": "TRUE" # Ensure quotes are included for bid/ask/last
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch option chain for {sym}: {e}")
        return None

# --- Data Parsing and Preprocessing ---

def parse_option_chain_data(json_data: dict) -> pd.DataFrame:
    """Parses raw Schwab option chain JSON into a clean DataFrame."""
    recs = []
    current_date = pd.Timestamp.now().normalize() # Normalize to remove time component for DTE calculation

    for kind, key in [("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")]:
        for exp_key, strikes_data in json_data.get(key, {}).items():
            # exp_key format: YYYY-MM-DD:milliseconds_timestamp
            expiry_date_str = exp_key.split(":")[0]
            expiry_date = pd.to_datetime(expiry_date_str)

            for strike_str, options_list in strikes_data.items():
                for option_data in options_list:
                    r = {
                        "expiry": expiry_date,
                        "strike": float(strike_str),
                        "type": kind,
                        "symbol": option_data.get("symbol"), # Add option symbol
                        "description": option_data.get("description"),
                        "bid": option_data.get("bid"),
                        "ask": option_data.get("ask"),
                        "last": option_data.get("last"),
                        "openInterest": option_data.get("openInterest"),
                        "totalVolume": option_data.get("totalVolume"),
                        "volatility": option_data.get("volatility"), # Implied Volatility
                        "delta": option_data.get("delta"),
                        "gamma": option_data.get("gamma"),
                        "theta": option_data.get("theta"),
                        "vega": option_data.get("vega"),
                        "rho": option_data.get("rho"),
                        "inTheMoney": option_data.get("inTheMoney"),
                        "multiplier": option_data.get("multiplier", 100) # Default to 100
                    }
                    recs.append(r)

    df = pd.DataFrame(recs)

    if not df.empty:
        # Calculate Days to Expiration (DTE)
        df["dte"] = (df["expiry"] - current_date).dt.days
        # Filter out options that have already expired (DTE < 0)
        df = df[df["dte"] >= 0].copy()

        # Convert relevant columns to numeric, coercing errors to NaN
        numeric_cols = ["bid", "ask", "last", "openInterest", "totalVolume",
                        "volatility", "delta", "gamma", "theta", "vega", "rho"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill NaN for openInterest and totalVolume with 0 for calculations
        df["openInterest"] = df["openInterest"].fillna(0)
        df["totalVolume"] = df["totalVolume"].fillna(0)
        
        # Filter out rows with NaN in critical columns for calculations
        # For volatility surface, we need volatility, strike, dte.
        # For Greeks, we need delta, gamma.
        df.dropna(subset=["volatility", "strike", "dte"], inplace=True)
        # Also ensure volatility is positive and within a reasonable range (e.g., 1% to 500%)
        df = df[(df["volatility"] > 0.01) & (df["volatility"] < 500)].copy()

    return df

# --- Metric Computations ---

def compute_options_metrics(df: pd.DataFrame, underlying_price: float, focus_range: float = None):
    """
    Computes various options metrics like GEX, DEX, OI, Volatility Surface, and Key Levels.
    
    Args:
        df (pd.DataFrame): The parsed options DataFrame.
        underlying_price (float): The current price of the underlying asset.
        focus_range (float, optional): A percentage range around the underlying price
                                       to focus calculations (ee.g., 0.20 for +/- 20%).
                                       If None, uses the full range.
    Returns:
        dict: A dictionary containing computed metrics and dataframes.
    """
    if df.empty:
        return {
            "gamma_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
            "oi_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
            "vol_by_type": pd.DataFrame(columns=["strike", "CALL", "PUT"]),
            "net_gex": pd.DataFrame(columns=["strike", "net_gex"]),
            "net_dex": pd.DataFrame(columns=["strike", "net_dex"]),
            "vol_surface": pd.DataFrame(),
            "vol_term_structure": pd.DataFrame(),
            "levels": {"ATM": None, "Max OI": None, "Max Gamma": None, "Max Pain": None},
        }

    df_filtered = df.copy()
    if focus_range and underlying_price > 0:
        lower_bound = underlying_price * (1 - focus_range)
        upper_bound = underlying_price * (1 + focus_range)
        df_filtered = df_filtered[(df_filtered["strike"] >= lower_bound) & (df_filtered["strike"] <= upper_bound)]
        
        if df_filtered.empty: # If filtering results in empty df, return empty metrics
            return compute_options_metrics(pd.DataFrame(), underlying_price)

    # Calculate exposures (using multiplier, typically 100 shares per contract)
    # Ensure delta and gamma are not NaN before multiplication
    df_filtered["gamma_exp"] = df_filtered["gamma"].fillna(0) * df_filtered["openInterest"] * df_filtered["multiplier"]
    df_filtered["delta_exp"] = df_filtered["delta"].fillna(0) * df_filtered["openInterest"] * df_filtered["multiplier"]

    # Group by strike and type for OI, Gamma, Volume
    gamma_by_type = (
        df_filtered.groupby(["strike", "type"])["gamma_exp"]
                   .sum()
                   .unstack(fill_value=0)
                   .reset_index()
    )
    oi_by_type = (
        df_filtered.groupby(["strike", "type"])["openInterest"]
                   .sum()
                   .unstack(fill_value=0)
                   .reset_index()
    )
    vol_by_type = (
        df_filtered.groupby(["strike", "type"])["totalVolume"]
                   .sum()
                   .unstack(fill_value=0)
                   .reset_index()
    )

    # Calculate NET GEX and NET DEX by strike
    net_gex_data = []
    net_dex_data = []
    
    # Ensure strikes are sorted for consistent plotting
    unique_strikes = sorted(df_filtered["strike"].unique())

    for strike in unique_strikes:
        strike_data = df_filtered[df_filtered["strike"] == strike]
        
        call_gex = strike_data[strike_data["type"] == "CALL"]["gamma_exp"].sum()
        put_gex = strike_data[strike_data["type"] == "PUT"]["gamma_exp"].sum()
        net_gex = call_gex - put_gex # Calls add positive gamma, Puts add negative gamma (when short)
        
        call_dex = strike_data[strike_data["type"] == "CALL"]["delta_exp"].sum()
        put_dex = strike_data[strike_data["type"] == "PUT"]["delta_exp"].sum()
        net_dex = call_dex + put_dex # Put delta is negative, so adding them correctly sums exposure
        
        net_gex_data.append({"strike": strike, "net_gex": net_gex})
        net_dex_data.append({"strike": strike, "net_dex": net_dex})

    net_gex_df = pd.DataFrame(net_gex_data)
    net_dex_df = pd.DataFrame(net_dex_data)

    # --- Volatility Surface Data Preparation ---
    # Filter for valid volatility values and options with some liquidity
    vol_surface_df = df_filtered[
        (df_filtered["volatility"].notna()) &
        (df_filtered["volatility"] > 0) &
        (df_filtered["openInterest"] > 0) # Only consider options with open interest
        # Or (df_filtered["totalVolume"] > 0) # Could also consider volume
    ].copy()
    
    # Optional: Outlier removal for volatility
    if not vol_surface_df.empty:
        Q1 = vol_surface_df['volatility'].quantile(0.25)
        Q3 = vol_surface_df['volatility'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_vol = Q1 - 1.5 * IQR
        upper_bound_vol = Q3 + 1.5 * IQR
        vol_surface_df = vol_surface_df[
            (vol_surface_df['volatility'] >= lower_bound_vol) &
            (vol_surface_df['volatility'] <= upper_bound_vol)
        ].copy()

    # Ensure unique strike-DTE-type combinations for surface plotting
    vol_surface_df = vol_surface_df[["strike", "dte", "volatility", "type", "expiry"]].drop_duplicates()

    # Volatility term structure (ATM vol across expirations)
    vol_term_data = []
    # Sort unique expirations by DTE
    sorted_expiries = sorted(df_filtered["expiry"].unique())

    for expiry in sorted_expiries:
        exp_data = df_filtered[df_filtered["expiry"] == expiry]
        
        if not exp_data.empty and not exp_data["volatility"].isna().all():
            # Find ATM strike for this expiry: closest strike to underlying price
            atm_strike = min(exp_data["strike"].unique(), key=lambda x: abs(x - underlying_price))
            
            # Get options at ATM strike for this expiry
            atm_options = exp_data[exp_data["strike"] == atm_strike]
            
            if not atm_options.empty and not atm_options["volatility"].isna().all():
                # Average volatility of ATM calls and puts for this expiry
                avg_vol = atm_options["volatility"].mean()
                dte = atm_options["dte"].iloc[0] # DTE should be same for all options of same expiry
                
                vol_term_data.append({
                    "dte": dte, 
                    "volatility": avg_vol,
                    "expiry": expiry
                })

    vol_term_df = pd.DataFrame(vol_term_data).sort_values("dte")

    # Calculate key levels
    strikes = sorted(df_filtered["strike"].unique()) if not df_filtered.empty else []
    
    atm_strike_level = None
    if strikes and underlying_price > 0:
        atm_strike_level = min(strikes, key=lambda x: abs(x - underlying_price))

    oi_by_strike = df_filtered.groupby("strike")["openInterest"].sum()
    max_oi_strike = oi_by_strike.idxmax() if not oi_by_strike.empty else None
    
    gamma_by_strike = df_filtered.groupby("strike")["gamma_exp"].sum()
    max_gamma_strike = gamma_by_strike.idxmax() if not gamma_by_strike.empty else None

    # Max Pain calculation
    max_pain_strike = None
    if strikes:
        pain_points = {}
        for S in strikes:
            # Calls in the money (strike <= S)
            calls_itm = df_filtered[(df_filtered["type"] == "CALL") & (df_filtered["strike"] <= S)]
            call_loss = ((S - calls_itm["strike"]) * calls_itm["openInterest"] * calls_itm["multiplier"]).sum()
            
            # Puts in the money (strike >= S)
            puts_itm = df_filtered[(df_filtered["type"] == "PUT") & (df_filtered["strike"] >= S)]
            put_loss = ((puts_itm["strike"] - S) * puts_itm["openInterest"] * puts_itm["multiplier"]).sum()
            
            pain_points[S] = call_loss + put_loss
        
        if pain_points:
            max_pain_strike = min(pain_points, key=pain_points.get)

    return {
        "gamma_by_type": gamma_by_type,
        "oi_by_type": oi_by_type,
        "vol_by_type": vol_by_type,
        "net_gex": net_gex_df,
        "net_dex": net_dex_df,
        "vol_surface": vol_surface_df,
        "vol_term_structure": vol_term_df,
        "levels": {
            "ATM": atm_strike_level,
            "Max OI": max_oi_strike,
            "Max Gamma": max_gamma_strike,
            "Max Pain": max_pain_strike,
        },
    }

def get_0dte_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters for true 0DTE (expiring today) options."""
    # True 0DTE means DTE is 0 (expiring today)
    return df[df["dte"] == 0].copy()

# --- Streamlit App Logic ---

sym = st.text_input("Enter Ticker Symbol", "").upper().strip()
if not sym:
    st.info("👆 Please enter a ticker symbol to analyze its options chain.")
    st.stop()

# --- Data Fetching and Initial Processing ---
df_full_chain = pd.DataFrame()
underlying_price = 0

with st.spinner(f"Fetching data for {sym}..."):
    try:
        token = get_access_token()
        if not token:
            st.stop() # Stop if token acquisition failed

        json_data = fetch_option_chain(sym, token)
        if not json_data:
            st.stop() # Stop if option chain fetching failed

        underlying_price = json_data.get("underlyingPrice", 0)
        if underlying_price == 0:
            st.warning(f"Could not retrieve underlying price for {sym}. Some calculations may be inaccurate.")

        df_full_chain = parse_option_chain_data(json_data)

        if df_full_chain.empty:
            st.error(f"No valid options data could be parsed for {sym}. This might be due to no tradeable options or data issues.")
            st.stop()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            st.error(f"Invalid symbol: {sym}. Please check the ticker symbol and try again.")
        elif e.response.status_code == 404:
            st.error(f"Symbol {sym} not found or no options data available.")
        else:
            st.error(f"An HTTP error occurred: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data retrieval: {e}")
        st.stop()

# --- Compute Metrics for different views ---
metrics_full = compute_options_metrics(df_full_chain, underlying_price)
metrics_focused = compute_options_metrics(df_full_chain, underlying_price, focus_range=0.20) # +/- 20% for GEX/DEX charts

df_0dte = get_0dte_data(df_full_chain)
metrics_0dte = compute_options_metrics(df_0dte, underlying_price, focus_range=0.10) # Tighter focus for 0DTE

st.metric("Underlying Price", f"${underlying_price:.2f}")

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
            # Aggregate OI by strike, then pivot
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
        st.subheader("Gamma Exposure by Type")
        if not metrics_focused["gamma_by_type"].empty:
            gdf = metrics_focused["gamma_by_type"].melt(
                id_vars="strike",
                value_vars=[col for col in ["CALL","PUT"] if col in metrics_focused["gamma_by_type"].columns],
                var_name="type",
                value_name="gamma_exposure"
            )
            fig_gamma_type = px.bar(
                gdf, x="strike", y="gamma_exposure", color="type",
                barmode="group",
                color_discrete_map={"CALL":"green","PUT":"red"},
                title="Gamma Exposure by Type (Total Contracts * Gamma)"
            )
            fig_gamma_type.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            st.plotly_chart(fig_gamma_type, use_container_width=True)
        else:
            st.info("No gamma exposure data available for this range.")

    with col2:
        st.subheader("Net Gamma Exposure (Call Gamma - Put Gamma)")
        if not metrics_focused["net_gex"].empty:
            fig_net_gex = px.bar(
                metrics_focused["net_gex"],
                x="strike",
                y="net_gex",
                title="Net Gamma Exposure (Call Gamma - Put Gamma)",
                color="net_gex",
                color_continuous_scale=["red", "white", "green"]
            )
            fig_net_gex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            fig_net_gex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_gex, use_container_width=True)
        else:
            st.info("No net gamma exposure data available for this range.")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Net Delta Exposure")
        if not metrics_focused["net_dex"].empty:
            fig_net_dex = px.bar(
                metrics_focused["net_dex"],
                x="strike",
                y="net_dex",
                title="Net Delta Exposure (Total Contracts * Delta)",
                color="net_dex",
                color_continuous_scale=["red", "white", "blue"]
            )
            fig_net_dex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            fig_net_dex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_dex, use_container_width=True)
        else:
            st.info("No net delta exposure data available for this range.")

    with col4:
        st.subheader("Open Interest by Type")
        if not metrics_focused["oi_by_type"].empty:
            oidf = metrics_focused["oi_by_type"].melt(
                id_vars="strike",
                value_vars=[col for col in ["CALL","PUT"] if col in metrics_focused["oi_by_type"].columns],
                var_name="type",
                value_name="open_interest"
            )
            fig_oi_type = px.bar(
                oidf, x="strike", y="open_interest", color="type",
                barmode="group",
                color_discrete_map={"CALL":"green","PUT":"red"},
                title="Open Interest by Type"
            )
            fig_oi_type.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
            st.plotly_chart(fig_oi_type, use_container_width=True)
        else:
            st.info("No open interest data available for this range.")

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
        fig_term.update_xaxes(range=[0, metrics_full["vol_term_structure"]["dte"].max() * 1.1]) # Dynamic range
        fig_term.update_yaxes(tickformat=".2f") # Format as percentage
        st.plotly_chart(fig_term, use_container_width=True)
    else:
        st.info("No volatility term structure data available.")

    st.subheader("3D Implied Volatility Surface")
    # Check for sufficient data points for a meaningful surface
    if not metrics_full["vol_surface"].empty and len(metrics_full["vol_surface"]) > 20: # Increased threshold
        # Create pivot table for the surface, averaging volatility for duplicate strike/dte
        # Consider only CALLs or PUTs, or average both. For a cleaner surface, often one type is used.
        # Let's use the average of CALL and PUT volatility if both exist for a strike/DTE
        
        # Aggregate by strike, dte, and type, then average volatility
        # This handles cases where multiple options might exist for same strike/dte (e.g., different series)
        surface_data_agg = metrics_full["vol_surface"].groupby(["strike", "dte"])["volatility"].mean().unstack(fill_value=np.nan)

        if not surface_data_agg.empty:
            # Optional: Interpolate missing values for a smoother surface
            # This can be dangerous if data is too sparse, but helps with visual continuity
            # Choose an interpolation method (e.g., 'linear', 'cubic', 'nearest')
            # For a surface, 'linear' or 'cubic' might be appropriate if data gaps are small.
            # For now, let's try without heavy interpolation first, as it can mask data issues.
            # If the surface is still very jagged, consider:
            # surface_data_agg = surface_data_agg.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0)

            # Ensure DTE columns are numeric and sorted
            surface_data_agg.columns = pd.to_numeric(surface_data_agg.columns)
            surface_data_agg = surface_data_agg.sort_index(axis=1) # Sort DTEs
            surface_data_agg = surface_data_agg.sort_index(axis=0) # Sort Strikes

            # Calculate min/max for Z-axis to ensure good scaling
            z_values = surface_data_agg.values[~np.isnan(surface_data_agg.values)]
            z_min = np.min(z_values) * 0.9 if z_values.size > 0 else 0
            z_max = np.max(z_values) * 1.1 if z_values.size > 0 else 100 # Default max if no data

            fig_surface = go.Figure(data=[go.Surface(
                z=surface_data_agg.values,
                x=surface_data_agg.columns,
                y=surface_data_agg.index,
                colorscale="Viridis", # Good default, consider "Plasma", "Jet", "Hot"
                colorbar_title="Implied Volatility (%)",
                # Add contour lines for better readability
                contours_z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            )])

            fig_surface.update_layout(
                title="3D Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Days to Expiration (DTE)",
                    yaxis_title="Strike Price",
                    zaxis_title="Implied Volatility (%)",
                    xaxis=dict(range=[0, surface_data_agg.columns.max() * 1.1 if not surface_data_agg.columns.empty else 100]),
                    zaxis=dict(range=[z_min, z_max]),
                    aspectmode="auto", # Let Plotly determine aspect ratio
                    camera=dict(
                        eye=dict(x=1.8, y=1.8, z=0.8) # Adjust camera angle for better initial view
                    )
                ),
                height=600
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        else:
            st.info("Not enough aggregated volatility data points to render a meaningful 3D volatility surface.")
    else:
        st.info("Not enough raw volatility data points to attempt rendering a 3D volatility surface. (Need > 20 points)")

    st.subheader("Volatility Skew by Expiration")
    if not metrics_full["vol_surface"].empty:
        # Get unique expirations, sorted by DTE, show up to 4
        unique_dtes_for_skew = sorted(metrics_full["vol_surface"]["dte"].unique())[:4]

        if unique_dtes_for_skew:
            skew_cols = st.columns(len(unique_dtes_for_skew))
            for i, dte in enumerate(unique_dtes_for_skew):
                exp_data = metrics_full["vol_surface"][metrics_full["vol_surface"]["dte"] == dte]

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
    else:
        st.info("No volatility data available to show skew.")

# =============== TAB 4: 0DTE DEEP DIVE ===============
with tab4:
    st.header("⚡ 0DTE Options Deep Dive")

    if df_0dte.empty:
        st.warning("No 0DTE (expiring today) options found for this symbol. This could mean no options expire today or data is unavailable.")
    else:
        st.info(f"Found {len(df_0dte):,} 0DTE options expiring today.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("0DTE Net Gamma Exposure")
            if not metrics_0dte["net_gex"].empty:
                fig_0dte_gex = px.bar(
                    metrics_0dte["net_gex"],
                    x="strike",
                    y="net_gex",
                    title="0DTE Net Gamma Exposure",
                    color="net_gex",
                    color_continuous_scale=["red", "white", "green"]
                )
                fig_0dte_gex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
                fig_0dte_gex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_gex, use_container_width=True)
            else:
                st.info("No 0DTE gamma exposure data available.")

        with col2:
            st.subheader("0DTE Net Delta Exposure")
            if not metrics_0dte["net_dex"].empty:
                fig_0dte_dex = px.bar(
                    metrics_0dte["net_dex"],
                    x="strike",
                    y="net_dex",
                    title="0DTE Net Delta Exposure",
                    color="net_dex",
                    color_continuous_scale=["red", "white", "blue"]
                )
                fig_0dte_dex.add_vline(x=underlying_price, line_dash="dash", line_color="white", annotation_text="Current Price")
                fig_0dte_dex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_dex, use_container_width=True)
            else:
                st.info("No 0DTE delta exposure data available.")

        st.subheader("0DTE: Net GEX vs Net DEX Scatter")
        if not metrics_0dte["net_gex"].empty and not metrics_0dte["net_dex"].empty:
            combined_0dte = metrics_0dte["net_gex"].merge(metrics_0dte["net_dex"], on="strike")
            # Add a 'size' column for visual emphasis based on magnitude
            combined_0dte["magnitude"] = np.sqrt(combined_0dte["net_gex"]**2 + combined_0dte["net_dex"]**2)

            fig_combined = px.scatter(
                combined_0dte,
                x="net_gex",
                y="net_dex",
                size="magnitude", # Size points by combined magnitude
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

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type_filter = st.selectbox("Filter by Option Type", ["All", "CALL", "PUT"])
    with col2:
        min_dte_filter = st.number_input("Minimum DTE", value=0, min_value=0)
    with col3:
        max_dte_filter = st.number_input("Maximum DTE", value=365, min_value=0)

    # Apply filters
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
        ]].round(4), # Round floats for better display
        use_container_width=True
    )

    # Download button
    if not filtered_df_display.empty:
        csv = filtered_df_display.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"{sym}_options_data_filtered.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to download after applying filters.")

