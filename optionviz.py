import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("OptvizV1")

@st.cache_data
def get_access_token():
    resp = requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        auth=(os.getenv("SCHWAB_API_KEY"), os.getenv("SCHWAB_API_SECRET")),
        data={"grant_type": "client_credentials"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

@st.cache_data(ttl=300)
def fetch_option_chain(sym, token):
    url = "https://api.schwabapi.com/marketdata/v1/chains"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params={"symbol": sym})
    resp.raise_for_status()
    return resp.json()

def parse_chain(js):
    recs = []
    for kind, key in [("CALL","callExpDateMap"), ("PUT","putExpDateMap")]:
        for exp_key, strikes in js.get(key, {}).items():
            date = exp_key.split(":",1)[0]
            for strike_str, opts in strikes.items():
                for o in opts:
                    r = {"expiry": date, "strike": float(strike_str), "type": kind}
                    r.update(o)
                    recs.append(r)
    df = pd.DataFrame(recs)
    if not df.empty:
        df["expiry"] = pd.to_datetime(df["expiry"])
        # Add days to expiration
        df["dte"] = (df["expiry"] - pd.Timestamp.now()).dt.days
    return df

def compute_metrics(df, underlying, focus_range=None):
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
    
    # Filter data if focus_range is specified
    if focus_range:
        lower_bound = underlying * (1 - focus_range)
        upper_bound = underlying * (1 + focus_range)
        df = df[(df["strike"] >= lower_bound) & (df["strike"] <= upper_bound)]
    
    # Calculate exposures
    df["gamma_exp"] = df["gamma"] * df["openInterest"] * 100
    df["delta_exp"] = df["delta"] * df["openInterest"] * 100

    # Group by strike and type
    gbt = (
        df.groupby(["strike","type"])["gamma_exp"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    oibt = (
        df.groupby(["strike","type"])["openInterest"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    vibt = (
        df.groupby(["strike","type"])["totalVolume"]
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )

    # Calculate NET GEX and NET DEX by strike
    net_gex_data = []
    net_dex_data = []
    
    for strike in sorted(df["strike"].unique()):
        strike_data = df[df["strike"] == strike]
        
        call_gex = strike_data[strike_data["type"] == "CALL"]["gamma_exp"].sum()
        put_gex = strike_data[strike_data["type"] == "PUT"]["gamma_exp"].sum()
        net_gex = call_gex - put_gex
        
        call_dex = strike_data[strike_data["type"] == "CALL"]["delta_exp"].sum()
        put_dex = strike_data[strike_data["type"] == "PUT"]["delta_exp"].sum()
        net_dex = call_dex + put_dex  # Note: put delta is negative, so we add
        
        net_gex_data.append({"strike": strike, "net_gex": net_gex})
        net_dex_data.append({"strike": strike, "net_dex": net_dex})

    net_gex_df = pd.DataFrame(net_gex_data)
    net_dex_df = pd.DataFrame(net_dex_data)

    # Volatility surface data
    vol_surface = df[df["volatility"].notna() & (df["volatility"] > 0)].copy()
    vol_surface = vol_surface[["strike", "dte", "volatility", "type", "expiry"]].drop_duplicates()

    # Volatility term structure (ATM vol across expirations)
    vol_term_data = []
    for expiry in df["expiry"].unique():
        exp_data = df[df["expiry"] == expiry]
        if not exp_data.empty and not exp_data["volatility"].isna().all():
            # Find ATM strike for this expiry
            atm_strike = min(exp_data["strike"].unique(), key=lambda x: abs(x - underlying))
            atm_data = exp_data[exp_data["strike"] == atm_strike]
            
            if not atm_data.empty and not atm_data["volatility"].isna().all():
                avg_vol = atm_data["volatility"].mean()
                dte = atm_data["dte"].iloc[0]
                vol_term_data.append({
                    "dte": dte, 
                    "volatility": avg_vol,
                    "expiry": expiry
                })

    vol_term_df = pd.DataFrame(vol_term_data).sort_values("dte")

    # Calculate levels
    strikes = sorted(df["strike"].unique()) if not df.empty else []
    atm = min(strikes, key=lambda x: abs(x-underlying)) if strikes else None
    
    oi_by_strike = df.groupby("strike")["openInterest"].sum()
    max_oi = oi_by_strike.idxmax() if not oi_by_strike.empty else None
    
    gamma_by_strike = df.groupby("strike")["gamma_exp"].sum()
    max_g = gamma_by_strike.idxmax() if not gamma_by_strike.empty else None

    pain = {}
    for S in strikes:
        c = df[(df.type=="CALL")&(df.strike<=S)]
        p = df[(df.type=="PUT") &(df.strike>=S)]
        loss = ((S-c.strike)*c.openInterest*100).sum() + ((p.strike-S)*p.openInterest*100).sum()
        pain[S] = loss
    max_pain = min(pain, key=pain.get) if pain else None

    return {
        "gamma_by_type": gbt,
        "oi_by_type": oibt,
        "vol_by_type": vibt,
        "net_gex": net_gex_df,
        "net_dex": net_dex_df,
        "vol_surface": vol_surface,
        "vol_term_structure": vol_term_df,
        "levels": {
            "ATM": atm,
            "Max OI": max_oi,
            "Max Gamma": max_g,
            "Max Pain": max_pain,
        },
    }

def get_0dte_data(df):
    """Filter for 0DTE (same day expiration) options"""
    return df[df["dte"] <= 1]  # Include today and tomorrow for 0DTE

# Main App
sym = st.text_input("Ticker", "").upper().strip()
if not sym:
    st.info("👆 Enter a ticker symbol to get started")
    st.stop()

try:
    token = get_access_token()
    js = fetch_option_chain(sym, token)
    under = js.get("underlyingPrice", 0)
    
    # Check if we got valid option chain data
    if not js.get("callExpDateMap") and not js.get("putExpDateMap"):
        st.error(f"No options data found for {sym}. This symbol may not have tradeable options or may not be supported.")
        st.stop()
    
    df = parse_chain(js)
    
    if df.empty:
        st.error(f"No options data could be parsed for {sym}")
        st.stop()
        
    # Get different datasets
    met_full = compute_metrics(df, under)
    met_focused = compute_metrics(df, under, focus_range=0.20)
    
    # Get 0DTE data
    df_0dte = get_0dte_data(df)
    met_0dte = compute_metrics(df_0dte, under, focus_range=0.10)  # Tighter focus for 0DTE
    
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        st.error(f"Invalid symbol: {sym}. Please check the ticker symbol and try again.")
    elif e.response.status_code == 404:
        st.error(f"Symbol {sym} not found or not supported for options data.")
    else:
        st.error(f"Error fetching data for {sym}: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Display current price
st.metric("Underlying Price", f"${under:.2f}")

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
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Key levels
        levels_df = (
            pd.DataFrame.from_dict(met_full["levels"], orient="index", columns=["Strike"])
              .reset_index()
              .rename(columns={"index":"Level"})
        )
        st.subheader("📍 Important Price Levels")
        st.table(levels_df)
        
        # Summary stats
        st.subheader("📋 Summary Statistics")
        total_call_oi = met_full["oi_by_type"].get("CALL", pd.Series()).sum() if "CALL" in met_full["oi_by_type"].columns else 0
        total_put_oi = met_full["oi_by_type"].get("PUT", pd.Series()).sum() if "PUT" in met_full["oi_by_type"].columns else 0
        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_0dte_oi = len(df_0dte)
        total_oi = len(df)
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Put/Call Ratio", f"{pc_ratio:.2f}")
            st.metric("Total Call OI", f"{total_call_oi:,.0f}")
        with metrics_col2:
            st.metric("0DTE Options", f"{total_0dte_oi}")
            st.metric("Total Put OI", f"{total_put_oi:,.0f}")
    
    with col2:
        # Put/Call Ratio Pie Chart
        if total_call_oi > 0 and total_put_oi > 0:
            fig_pie = px.pie(
                values=[total_call_oi, total_put_oi], 
                names=["CALL","PUT"],
                color_discrete_map={"CALL":"green","PUT":"red"},
                title=f"Open Interest Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Option Matrix
        if not df.empty:
            matrix = df.pivot_table(index="strike", columns="type", values="openInterest").fillna(0).round(0)
            # Show only top strikes by OI
            matrix_total = matrix.sum(axis=1).sort_values(ascending=False)
            top_strikes = matrix_total.head(10).index
            matrix_display = matrix.loc[top_strikes]
            
            st.subheader("🏆 Top 10 Strikes by OI")
            st.dataframe(matrix_display)

# =============== TAB 2: FLOW ANALYSIS ===============
with tab2:
    st.subheader("GEX/DEX")
    
    col1, col2 = st.columns(2)
    
    # Regular Gamma Exposure
    with col1:
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
                title="Gamma Exposure by Type"
            )
            fig1.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
            st.plotly_chart(fig1, use_container_width=True)

    # Net Gamma Exposure
    with col2:
        if not met_focused["net_gex"].empty:
            fig_net_gex = px.bar(
                met_focused["net_gex"], 
                x="strike", 
                y="net_gex",
                title="Net Gamma Exposure (Call - Put)",
                color="net_gex",
                color_continuous_scale=["red", "white", "green"]
            )
            fig_net_gex.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
            fig_net_gex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_gex, use_container_width=True)

    col3, col4 = st.columns(2)
    
    # Net Delta Exposure
    with col3:
        if not met_focused["net_dex"].empty:
            fig_net_dex = px.bar(
                met_focused["net_dex"], 
                x="strike", 
                y="net_dex",
                title="Net Delta Exposure",
                color="net_dex",
                color_continuous_scale=["red", "white", "blue"]
            )
            fig_net_dex.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
            fig_net_dex.update_layout(showlegend=False)
            st.plotly_chart(fig_net_dex, use_container_width=True)

    # Open Interest
    with col4:
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
                title="Open Interest by Type"
            )
            fig2.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
            st.plotly_chart(fig2, use_container_width=True)

# =============== TAB 3: VOLATILITY ANALYSIS ===============
with tab3:
    st.subheader("Volatility Analysis")
    
    # Volatility Term Structure
    if not met_full["vol_term_structure"].empty:
        fig_term = px.line(
            met_full["vol_term_structure"], 
            x="dte", 
            y="volatility",
            markers=True,
            title="Volatility Term Structure (ATM)"
        )
        fig_term.update_traces(line_color="purple", marker_color="purple")
        st.plotly_chart(fig_term, use_container_width=True)

    # 3D Volatility Surface
    if not met_full["vol_surface"].empty and len(met_full["vol_surface"]) > 10:
        st.subheader("3D Volatility Surface")
        
        # Create pivot table for surface
        surface_calls = met_full["vol_surface"][met_full["vol_surface"]["type"] == "CALL"]
        
        if not surface_calls.empty:
            pivot_calls = surface_calls.pivot_table(
                index="strike", columns="dte", values="volatility", aggfunc="mean"
            )
            
            if not pivot_calls.empty:
                fig_surface = go.Figure()
                
                # Add CALL surface
                fig_surface.add_trace(go.Surface(
                    z=pivot_calls.values,
                    x=pivot_calls.columns,
                    y=pivot_calls.index,
                    colorscale="Viridis",
                    name="Implied Volatility"
                ))
                
                fig_surface.update_layout(
                    title="3D Implied Volatility Surface",
                    scene=dict(
                        xaxis_title="Days to Expiration",
                        yaxis_title="Strike Price", 
                        zaxis_title="Implied Volatility"
                    ),
                    height=600
                )
                st.plotly_chart(fig_surface, use_container_width=True)

    # Volatility Skew by Expiration
    if not met_full["vol_surface"].empty:
        st.subheader("Volatility Skew by Expiration")
        
        # Get unique expirations, sorted by DTE
        unique_dtes = sorted(met_full["vol_surface"]["dte"].unique())[:4]  # Show first 4
        
        skew_cols = st.columns(2)
        for i, dte in enumerate(unique_dtes):
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
                fig_skew.add_vline(x=under, line_dash="dash", line_color="blue")
                
                with skew_cols[i % 2]:
                    st.plotly_chart(fig_skew, use_container_width=True)

# =============== TAB 4: 0DTE ANALYSIS ===============
with tab4:
    st.subheader("⚡ 0DTE Options Analysis")
    
    if df_0dte.empty:
        st.warning("No 0DTE options found for this symbol.")
    else:
        st.info(f"Found {len(df_0dte)} 0DTE options")
        
        col1, col2 = st.columns(2)
        
        # 0DTE Net Gamma Exposure
        with col1:
            if not met_0dte["net_gex"].empty:
                fig_0dte_gex = px.bar(
                    met_0dte["net_gex"], 
                    x="strike", 
                    y="net_gex",
                    title="0DTE Net Gamma Exposure",
                    color="net_gex",
                    color_continuous_scale=["red", "white", "green"]
                )
                fig_0dte_gex.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
                fig_0dte_gex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_gex, use_container_width=True)
            else:
                st.warning("No 0DTE gamma exposure data available")

        # 0DTE Net Delta Exposure  
        with col2:
            if not met_0dte["net_dex"].empty:
                fig_0dte_dex = px.bar(
                    met_0dte["net_dex"], 
                    x="strike", 
                    y="net_dex",
                    title="0DTE Net Delta Exposure",
                    color="net_dex",
                    color_continuous_scale=["red", "white", "blue"]
                )
                fig_0dte_dex.add_vline(x=under, line_dash="dash", line_color="white", annotation_text="Current")
                fig_0dte_dex.update_layout(showlegend=False)
                st.plotly_chart(fig_0dte_dex, use_container_width=True)
            else:
                st.warning("No 0DTE delta exposure data available")
        
        # 0DTE Combined Analysis
        if not met_0dte["net_gex"].empty and not met_0dte["net_dex"].empty:
            # Combine the data for scatter plot
            combined_0dte = met_0dte["net_gex"].merge(met_0dte["net_dex"], on="strike")
            
            fig_combined = px.scatter(
                combined_0dte,
                x="net_gex",
                y="net_dex", 
                size=abs(combined_0dte["net_gex"]) + abs(combined_0dte["net_dex"]),
                hover_data=["strike"],
                title="0DTE: Net GEX vs Net DEX",
                labels={"net_gex": "Net Gamma Exposure", "net_dex": "Net Delta Exposure"}
            )
            fig_combined.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_combined.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_combined, use_container_width=True)
        
        # 0DTE Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_0dte_calls = len(df_0dte[df_0dte["type"] == "CALL"])
            st.metric("0DTE Calls", total_0dte_calls)
        with col2:
            total_0dte_puts = len(df_0dte[df_0dte["type"] == "PUT"])
            st.metric("0DTE Puts", total_0dte_puts)
        with col3:
            avg_0dte_vol = df_0dte["totalVolume"].mean() if not df_0dte.empty else 0
            st.metric("Avg Volume", f"{avg_0dte_vol:,.0f}")
        with col4:
            max_0dte_oi_strike = df_0dte.loc[df_0dte["openInterest"].idxmax(), "strike"] if not df_0dte.empty else 0
            st.metric("Max OI Strike", f"${max_0dte_oi_strike:.0f}")

# =============== TAB 5: RAW DATA ===============
with tab5:
    st.subheader("Raw Option Chain Data")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type_filter = st.selectbox("Option Type", ["All", "CALL", "PUT"])
    with col2:
        min_dte = st.number_input("Min DTE", value=0, min_value=0)
    with col3:
        max_dte = st.number_input("Max DTE", value=365, min_value=0)
    
    # Apply filters
    filtered_df = df.copy()
    if option_type_filter != "All":
        filtered_df = filtered_df[filtered_df["type"] == option_type_filter]
    filtered_df = filtered_df[(filtered_df["dte"] >= min_dte) & (filtered_df["dte"] <= max_dte)]
    
    st.dataframe(
        filtered_df[["symbol", "strike", "type", "expiry", "dte", "bid", "ask", "last", 
                    "openInterest", "totalVolume", "volatility", "delta", "gamma"]],
        use_container_width=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"{sym}_options_data.csv",
        mime="text/csv"
    )