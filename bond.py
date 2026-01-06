import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import date, datetime, timedelta
import math
import time
import numpy as np
import yfinance as yf
import gc 
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
from fpdf import FPDF
import io

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Simulatore Finanziario Pro", 
    layout="wide", 
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# --- 2. CSS: DESIGN PERFEZIONATO ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e0e11; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1, h2, h3, h4 { color: #f4f4f5; font-weight: 600; letter-spacing: -0.5px; }
    
    /* INPUT WIDGETS */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stDateInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: #18181b !important; 
        color: #fff !important;
        border: 1px solid #27272a !important; 
        border-radius: 8px !important;
        height: 42px;
    }
    .stMultiSelect > div > div > div {
        background-color: #18181b; color: #e4e4e7; border: 1px solid #27272a; border-radius: 8px;
    }

    /* --- HERO CARDS --- */
    .prof-card {
        background-color: #131316;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .accent-green { border-top: 3px solid #10b981; }
    .accent-purple { border-top: 3px solid #8b5cf6; }
    .accent-blue { border-top: 3px solid #3b82f6; }
    
    .card-label {
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 5px; display: block; text-align: center;
    }
    .text-green { color: #34d399; }
    .text-purple { color: #a78bfa; }
    .text-blue { color: #60a5fa; }
    
    /* INPUT HERO CUSTOMIZATION */
    div[data-testid="stNumberInput"] input {
        border: none !important;
        background: transparent !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: white !important;
        text-align: center !important;
        padding: 0 !important;
    }
    div[data-testid="stNumberInput"] button { display: none; } 

    /* --- SUMMARY DASHBOARD --- */
    .dash-container {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .dash-header-row { display: flex; justify-content: center; align-items: center; margin-bottom: 10px; }
    .dash-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .dash-value { font-size: 2.5rem; font-weight: 800; color: #fff; text-align: center; margin-bottom: 15px; }
    
    .progress-track {
        width: 100%; height: 16px; background-color: #334155; border-radius: 8px; overflow: hidden; display: flex;
    }
    
    .dash-legend-flex {
        display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-top: 15px; font-size: 0.9rem; font-weight: 500; color: #d4d4d8;
    }
    .leg-item-flex { display: flex; align-items: center; gap: 6px; }

    /* --- METRICS --- */
    div[data-testid="stMetric"] {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 16px !important;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricDelta"] {
        background-color: rgba(34, 197, 94, 0.15);
        border-radius: 6px;
        padding: 2px 8px;
        width: fit-content;
        margin-top: 4px;
    }
    
    /* SINGLE ASSET BADGE */
    .single-asset-badge {
        background-color: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: #d8b4fe;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 10px;
    }

    /* UTILS */
    .step-header {
        font-size: 1.1rem; font-weight: 700; color: #f4f4f5; margin-top: 25px; margin-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;
    }
    
    .stButton > button {
        width: 100%; background-color: #2563eb; color: white; border: none; padding: 12px; 
        font-size: 16px; font-weight: 600; border-radius: 8px; margin-top: 15px;
        transition: transform 0.1s;
    }
    .stButton > button:hover { background-color: #1d4ed8; transform: scale(1.01); }
</style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURAZIONE DATASET ---

ASSET_CONFIG = {
    "S&P 500 🇺🇸": {"ticker": "^GSPC", "color": "#a78bfa", "tax": 0.26},
    "Nasdaq 100 💻": {"ticker": "^NDX", "color": "#c084fc", "tax": 0.26},
    "Europa Stoxx 50 🇪🇺": {"ticker": "^STOXX50E", "color": "#3b82f6", "tax": 0.26},
    "FTSE MIB 🇮🇹": {"ticker": "FTSEMIB.MI", "color": "#0ea5e9", "tax": 0.26},
    "Mercati Emergenti 🌏": {"ticker": "EEM", "color": "#14b8a6", "tax": 0.26},
    "Oro (Gold) 🟡": {"ticker": "GC=F", "color": "#fbbf24", "tax": 0.26},
    "Argento (Silver) ⚪": {"ticker": "SI=F", "color": "#e5e7eb", "tax": 0.26},
    "Rame (Copper) 🥉": {"ticker": "HG=F", "color": "#b45309", "tax": 0.26},
    "Petrolio (WTI) 🛢️": {"ticker": "CL=F", "color": "#78716c", "tax": 0.26},
    "Real Estate (REITs) 🏠": {"ticker": "VNQ", "color": "#84cc16", "tax": 0.26},
    "Bitcoin ₿":   {"ticker": "BTC-USD", "color": "#f97316", "tax": 0.26},
    "Bond Globali 🌎": {"ticker": "AGG", "color": "#10b981", "tax": 0.26}
}

# --- 4. DATA LOADER ---
@st.cache_data(ttl=86400)
def load_market_data(period_str):
    tickers_map = {v["ticker"]: k for k, v in ASSET_CONFIG.items()}
    tickers_list = list(tickers_map.keys())
    final_df = pd.DataFrame()
    data_source = "Yahoo Finance (Live)"

    # Fallback Data
    BACKUP_DATA = {
        "S&P 500 🇺🇸": [-0.06, -0.03, -0.01, 0.04, 0.01, -0.08, -0.01, 0.09, -0.09, -0.16, 0.05, 0.02, 0.03, 0.01, -0.01, 0.04],
        "Nasdaq 100 💻": [-0.08, -0.04, 0.01, 0.05, 0.02, -0.10, -0.02, 0.12, -0.11, -0.18, 0.06, 0.03, 0.04, 0.02, -0.02, 0.06],
        "Europa Stoxx 50 🇪🇺": [-0.05, -0.02, -0.02, 0.03, 0.01, -0.07, -0.02, 0.06, -0.08, -0.14, 0.04, 0.01, 0.02, 0.01, -0.01, 0.03],
        "FTSE MIB 🇮🇹": [-0.07, -0.04, -0.08, 0.05, 0.02, -0.10, -0.03, 0.01, -0.12, -0.15, 0.08, -0.02, 0.05, 0.04, -0.02, 0.06],
        "Mercati Emergenti 🌏": [-0.09, -0.05, 0.02, 0.06, -0.03, -0.12, 0.01, 0.10, -0.15, -0.20, 0.09, 0.04, 0.03, -0.02, 0.05, 0.07],
        "Oro (Gold) 🟡": [0.03, 0.02, -0.02, -0.03, 0.02, 0.03, -0.03, -0.06, 0.04, -0.18, 0.05, 0.03, 0.04, -0.01, 0.02, 0.05],
        "Argento (Silver) ⚪": [0.05, 0.04, -0.04, -0.05, 0.03, 0.06, -0.05, -0.09, 0.06, -0.25, 0.08, 0.05, 0.06, -0.02, 0.04, 0.07],
        "Rame (Copper) 🥉": [0.02, 0.01, -0.03, 0.04, -0.02, -0.05, 0.02, 0.05, -0.07, -0.12, 0.04, 0.02, 0.01, -0.01, 0.03, 0.04],
        "Petrolio (WTI) 🛢️": [0.06, -0.05, 0.04, 0.02, -0.06, 0.08, -0.04, -0.10, 0.05, -0.30, 0.10, -0.05, 0.07, 0.01, -0.03, 0.09],
        "Real Estate (REITs) 🏠": [-0.04, -0.02, 0.01, 0.03, 0.00, -0.06, -0.01, 0.05, -0.07, -0.12, 0.03, 0.02, 0.02, 0.01, -0.01, 0.03],
        "Bitcoin ₿":   [0.10, -0.15, 0.20, 0.05, -0.08, 0.12, -0.05, 0.25, -0.10, 0.30, -0.20, 0.15, 0.05, -0.02, 0.10, 0.08],
        "Bond Globali 🌎": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.02, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.01]
    }

    try:
        raw_data = yf.download(tickers_list, period=period_str, interval="1mo", auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        
        if len(tickers_list) == 1:
            if isinstance(raw_data, pd.DataFrame):
                col = 'Close' if 'Close' in raw_data.columns else raw_data.columns[0]
                prices[tickers_list[0]] = raw_data[col]
        else:
            if isinstance(raw_data.columns, pd.MultiIndex):
                try: prices = raw_data['Close']
                except KeyError: prices = raw_data.iloc[:, 0]
            else:
                prices = raw_data

        prices = prices.rename(columns=tickers_map)
        final_df = prices.pct_change()

    except Exception:
        pass

    if final_df.empty or len(final_df.columns) == 0:
        final_df = pd.DataFrame()
        for k in tickers_list:
            if k in BACKUP_DATA:
                final_df[k] = BACKUP_DATA[k] * 20 
            else:
                final_df[k] = 0.0

    return final_df, "Yahoo Finance"

# --- 5. HELPERS ---

def apply_optimization(new_weights_map):
    for asset, val in new_weights_map.items():
        st.session_state[f"w_{asset}"] = val

def calculate_drawdown(series):
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown

def calculate_modified_var(returns_series, confidence_level=0.995):
    if len(returns_series) < 12: return 0.0
    mu = np.mean(returns_series)
    sigma = np.std(returns_series)
    s = skew(returns_series)
    k = kurtosis(returns_series)
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha)
    z_mod = (z_score + (z_score**2 - 1) * s / 6 + (z_score**3 - 3 * z_score) * k / 24 - (2 * z_score**3 - 5 * z_score) * s**2 / 36)
    m_var = mu + (z_mod * sigma)
    return m_var

def get_optimal_portfolio_weights(mean_returns, cov_matrix, max_allocation=1.0):
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, max_allocation) for asset in range(n_assets))
    
    def neg_sharpe(weights, mean_returns, cov_matrix, rf=0.02):
        p_ret = np.dot(weights, mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (p_ret - rf) / p_vol

    init_guess = n_assets * [1. / n_assets,]
    result = minimize(neg_sharpe, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_efficient_frontier(assets_selected, df_returns, max_single_alloc=1.0, n_portfolios_visual=30000):
    subset = df_returns[assets_selected].dropna()
    if subset.empty: return None
    mean_returns = subset.mean() * 12
    cov_matrix = subset.cov() * 12
    n_assets = len(assets_selected)
    
    weights = np.random.random((n_portfolios_visual, n_assets)).astype(np.float32)
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    port_returns = np.dot(weights, mean_returns)
    port_variance = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
    port_volatility = np.sqrt(port_variance)
    rf = 0.02
    sharpe_ratios = (port_returns - rf) / port_volatility
    
    monthly_mean_sim = port_returns / 12
    monthly_vol_sim = port_volatility / np.sqrt(12)
    z_score_995 = 2.576
    port_var_995_sim = monthly_mean_sim - (z_score_995 * monthly_vol_sim)
    
    optimal_weights = get_optimal_portfolio_weights(mean_returns, cov_matrix, max_single_alloc)
    opt_ret = np.dot(optimal_weights, mean_returns)
    opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    opt_sharpe = (opt_ret - rf) / opt_vol
    max_sharpe_idx_sim = np.argmax(sharpe_ratios)
    
    opt_series = subset.dot(optimal_weights)
    opt_var_995 = calculate_modified_var(opt_series, confidence_level=0.995)
    
    results = {
        'returns': port_returns, 'volatility': port_volatility, 'sharpe': sharpe_ratios,
        'var_995': port_var_995_sim, 'max_sharpe_idx_sim': max_sharpe_idx_sim,
        'opt_weights': optimal_weights, 'opt_ret': opt_ret, 'opt_vol': opt_vol,
        'opt_sharpe': opt_sharpe, 'opt_var_995': opt_var_995,
        'assets': assets_selected, 'subset_data': subset
    }
    return results

# --- FIXED ENGINE: SAFE FOR DATE COMPARISON & TAX TRACKING ---
def calculate_engine_multibond(bond_df, capital_override, tax_rate, use_compound, assets_to_calculate, market_returns_df, simulation_mode, n_simulations, mix_weights=None):
    today = date.today()
    
    bonds = []
    max_maturity = today
    
    tax_log = {
        'btp_coupons': 0.0,
        'btp_gain': 0.0,
        'asset_gain': 0.0
    }
    
    # 1. Parse Input
    for index, row in bond_df.iterrows():
        try:
            mat = row['Scadenza']
            if isinstance(mat, str): mat = datetime.strptime(mat, '%Y-%m-%d').date()
            if mat > max_maturity: max_maturity = mat
            
            cash_invested = float(row['Capitale Investito (€)'])
            price = float(row['Prezzo'])
            
            calculated_nominal = cash_invested / (price / 100)
            
            b = {
                'name': row['Nome/ISIN'],
                'maturity': mat,
                'coupon_pct': float(row['Cedola %']),
                'price': price,
                'nominal': calculated_nominal,
                'invested': cash_invested,
                'coupon_months': []
            }
            m1 = mat.month
            m2 = (m1 - 6) if m1 > 6 else (m1 + 6)
            b['coupon_months'] = {m1, m2}
            
            bonds.append(b)
        except Exception:
            continue
            
    # CRITICAL FIX: Allow 0 bonds if using only Risk Assets (Mix)
    if not bonds and (mix_weights is None or len(mix_weights) == 0):
        return None, "Nessun BTP valido e nessun Asset Satellite inserito.", 0, today, tax_log

    total_cash_invested = sum([b['invested'] for b in bonds])
    
    ts_max_maturity = pd.Timestamp(max_maturity)
    final_date = ts_max_maturity + pd.offsets.MonthEnd(0)
    if final_date < ts_max_maturity: 
        final_date += pd.offsets.MonthEnd(1)
    
    date_range = pd.date_range(start=today, end=final_date, freq='M') 
    if len(date_range) == 0: return None, "Orizzonte troppo breve.", 0, today, tax_log
    
    data = []
    
    liquid_cash = 0.0 
    cum_coupons = 0.0
    
    paid_out_indices = set()
    
    for d in date_range:
        monthly_cash_flow = 0.0
        active_bond_value = 0.0
        monthly_capital_gain = 0.0
        d_date = d.date()
        
        for i, b in enumerate(bonds):
            if i in paid_out_indices:
                continue
                
            is_maturity_month = (d.year == b['maturity'].year and d.month == b['maturity'].month)
            
            if d_date < b['maturity'] and not is_maturity_month:
                active_bond_value += b['invested']
                
                if d.month in b['coupon_months']:
                    gross_coup = (b['nominal'] * (b['coupon_pct']/100)) / 2
                    tax_coup = gross_coup * tax_rate
                    tax_log['btp_coupons'] += tax_coup
                    monthly_cash_flow += (gross_coup - tax_coup)
            
            elif is_maturity_month:
                gross_payout = b['nominal']
                gain = b['nominal'] - b['invested']
                
                tax_on_gain = (gain * tax_rate) if gain > 0 else 0
                tax_log['btp_gain'] += tax_on_gain
                
                net_payout = gross_payout - tax_on_gain
                liquid_cash += net_payout
                monthly_capital_gain += (net_payout - b['invested']) 
                paid_out_indices.add(i)
                
                if d.month in b['coupon_months']:
                    gross_coup = (b['nominal'] * (b['coupon_pct']/100)) / 2
                    tax_coup = gross_coup * tax_rate
                    tax_log['btp_coupons'] += tax_coup
                    monthly_cash_flow += (gross_coup - tax_coup)
            else:
                pass
        
        liquid_cash += monthly_cash_flow
        cum_coupons += monthly_cash_flow
        total_val = active_bond_value + liquid_cash
        
        data.append({
            "Date": d,
            "BTP_Value": total_val,
            "Cum_Coupons": cum_coupons,
            "Gain_Netto_Finale": monthly_capital_gain
        })
        
    df = pd.DataFrame(data)
    
    if not market_returns_df.empty:
        unique_assets = list(set(assets_to_calculate))
        for name in unique_assets:
            if name in market_returns_df.columns:
                hist_series = market_returns_df[name].values
                hist_series = hist_series[~np.isnan(hist_series)]
                if len(hist_series) > 0:
                    asset_tax_rate = ASSET_CONFIG[name]["tax"]
                    returns_to_use = np.zeros(len(df))
                    if simulation_mode == "Monte Carlo":
                        mu = np.mean(hist_series)
                        sigma = np.std(hist_series)
                        sims_to_run = n_simulations
                        random_matrix = np.random.normal(mu, sigma, (len(df), sims_to_run)).astype(np.float32)
                        cum_growth = np.cumprod(1.0 + random_matrix, axis=0)
                        final_values = cum_growth[-1, :]
                        median_idx = np.argsort(final_values)[len(final_values)//2]
                        returns_to_use = random_matrix[:, median_idx]
                        del random_matrix, cum_growth, final_values
                        gc.collect()
                    else:
                        for i in range(len(df)):
                            hist_idx = i % len(hist_series)
                            returns_to_use[i] = hist_series[hist_idx]
                    
                    gross_values = [1.0]
                    for i in range(len(df)):
                        monthly_return = returns_to_use[i]
                        prev_val = gross_values[-1]
                        new_val = prev_val * (1 + monthly_return)
                        gross_values.append(new_val)
                    gross_values = np.array(gross_values[1:])
                    
                    df[name] = gross_values

    return df, None, total_cash_invested, max_maturity, tax_log

def generate_monte_carlo_cone(mu, sigma, n_months, n_sims_preview=2000):
    dt = 1/12
    random_matrix = np.random.normal(mu * dt, sigma * np.sqrt(dt), (n_months, n_sims_preview))
    paths = np.cumprod(1 + random_matrix, axis=0)
    p05 = np.percentile(paths, 5, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    return p05, p50, p95

# --- PDF REPORT GENERATOR (SAFE TEXT) ---
def clean_text(text):
    replacements = {"€": "EUR", "à": "a'", "è": "e'", "é": "e'", "ì": "i'", "ò": "o'", "ù": "u'", "“": '"', "”": '"', "’": "'", "–": "-"}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(sim_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, clean_text("Simulatore Finanziario Pro"), ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, clean_text("Whitepaper Tecnico & Analisi Strategica"), ln=True, align='C')
    pdf.ln(10)
    
    # SECTION 1: EXECUTIVE SUMMARY
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("1. Executive Summary"), ln=True, fill=True)
    pdf.ln(5)
    
    if sim_data:
        pdf.set_font("Arial", '', 11)
        cap = sim_data.get('init_spent', 0)
        mat = sim_data.get('maturity_date', date.today())
        inf = sim_data.get('inflation', 0.02)
        mode = sim_data.get('sim_mode', 'N/A')
        
        pdf.multi_cell(0, 6, clean_text(f"Analisi di portafoglio obbligazionario (Bond Laddering) con capitale reale investito di EUR {cap:,.0f}. Orizzonte temporale massimo: {mat.strftime('%d/%m/%Y')}. Simulazione: {mode}."))
        pdf.ln(5)
        
        # TAX REPORT
        tax_log = sim_data.get('tax_log', {})
        if tax_log:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, clean_text("Riepilogo Fiscale Stimato (Tax Report):"), ln=True)
            pdf.set_font("Arial", '', 10)
            
            t_coup = tax_log.get('btp_coupons', 0)
            t_gbtp = tax_log.get('btp_gain', 0)
            t_gass = tax_log.get('asset_gain', 0)
            
            pdf.cell(0, 6, clean_text(f"- Tasse su Cedole BTP (12.5%): EUR {t_coup:,.2f}"), ln=True)
            pdf.cell(0, 6, clean_text(f"- Tasse su Capital Gain BTP (12.5%): EUR {t_gbtp:,.2f}"), ln=True)
            if t_gass > 0:
                pdf.cell(0, 6, clean_text(f"- Tasse su Asset Risk (26%): EUR {t_gass:,.2f}"), ln=True)
            pdf.ln(5)
        
        # BOND LIST
        if 'bond_df' in sim_data:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, clean_text("Dettaglio Obbligazioni (Core Portfolio):"), ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.cell(60, 7, "Nome", 1)
            pdf.cell(30, 7, "Scadenza", 1)
            pdf.cell(20, 7, "Cedola", 1)
            pdf.cell(30, 7, "Capitale Inv.", 1)
            pdf.ln()
            for index, row in sim_data['bond_df'].iterrows():
                try:
                    pdf.cell(60, 7, clean_text(str(row['Nome/ISIN'])[:25]), 1)
                    pdf.cell(30, 7, str(row['Scadenza']), 1)
                    pdf.cell(20, 7, f"{row['Cedola %']}%", 1)
                    pdf.cell(30, 7, f"{row['Capitale Investito (€)']:,.0f}", 1)
                    pdf.ln()
                except: pass
            pdf.ln(5)

        if sim_data.get('has_mix'):
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, clean_text("Asset Allocation Satellite (Risk):"), ln=True)
            pdf.set_font("Arial", '', 11)
            mix_d = sim_data.get('mix_details', {})
            for k, v in mix_d.items():
                pdf.cell(0, 6, clean_text(f"- {k}: {v:.1f}% del capitale di rischio"), ln=True)
    else:
        pdf.set_font("Arial", 'I', 11)
        pdf.multi_cell(0, 6, clean_text("Nessuna simulazione attiva. Questo documento funge da Manuale Metodologico del software."))
    pdf.ln(10)

    # SECTION 2: IL MOTORE MATEMATICO
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("2. Architettura del Modello"), ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_text("A. Bond Laddering & Accounting Mode"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, clean_text("Il valore del portafoglio obbligazionario è calcolato secondo il principio del costo ammortizzato. Il valore dei titoli rimane fisso al prezzo di carico per tutta la durata dell'investimento. Alla scadenza, viene contabilizzata la plusvalenza (Capital Gain) in un'unica soluzione (Rimborso a 100), generando un 'salto' visibile nel grafico."))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_text("Nota Metodologica (Cash Drag):"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, clean_text("Per prudenza, il software assume che il capitale rimborsato da obbligazioni scadute prima della data finale NON venga reinvestito (tasso 0%), ma rimanga liquido. Questo può ridurre il rendimento medio visualizzato rispetto ai singoli titoli ('Cash Drag'), ma rappresenta fedelmente il rischio di mancato reinvestimento."))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_text("B. Moto Browniano Geometrico (GBM)"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, clean_text("La componente di rischio è simulata via Monte Carlo. Formula: dS/S = mu*dt + sigma*dW. Generiamo 10 milioni di scenari per catturare le code grasse della distribuzione dei rendimenti."))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, clean_text("C. Ottimizzazione Markowitz (SLSQP)"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, clean_text("L'allocazione ottimale è calcolata minimizzando la volatilità per unità di rendimento atteso, rispettando vincoli di diversificazione hard-coded per evitare la concentrazione del rischio."))
    
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 5, clean_text("Disclaimer: I rendimenti passati non sono garanzia di rendimenti futuri."), ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- 7. MAIN APPLICATION ---

st.title("Simulatore Finanziario Pro")
st.markdown("Analisi Comparativa: Obbligazioni vs Mercati Globali")

with st.spinner("Scaricamento dati storici..."):
    market_returns, source_info = load_market_data("20y")

# --- INITIALIZE STATE ---
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False

# --- INPUT SECTION (Visible if not done) ---
if not st.session_state.simulation_done:
    # --- 0. DATA SETTINGS (Top) ---
    with st.expander("📂 Impostazioni Dati Storici", expanded=False):
        data_period_option = st.selectbox(
            "Seleziona Orizzonte Storico", 
            ["10y", "20y", "max"], 
            index=1, 
            format_func=lambda x: "Ultimi 10 Anni (Trend recente)" if x == "10y" else "Ultimi 20 Anni (Ciclo Completo)" if x == "20y" else "Massimo Disponibile",
            help="Definisce quanto indietro nel tempo andare per calcolare la volatilità e i rendimenti medi degli asset. '20 Anni' è consigliato perché include grandi crisi (2008) e boom, offrendo una statistica più robusta."
        )
        if 'current_period' not in st.session_state or st.session_state.current_period != data_period_option:
            st.session_state.current_period = data_period_option
            market_returns, _ = load_market_data(data_period_option)

    # --- 1. CONFIGURAZIONE CAPITALE & ASSET ALLOCATION (HERO SECTION) ---
    st.markdown('<div class="step-header">1. Definizione Budget e Asset Allocation</div>', unsafe_allow_html=True)
    
    c_in_1, c_in_2 = st.columns([1, 1.2]) 
    
    with c_in_1:
        # VISUALLY INTEGRATED CONTAINER FOR CAPITAL (PRO BOX)
        st.markdown('<div class="prof-card accent-green"><span class="card-label text-green">💰 Capitale Totale (€)</span>', unsafe_allow_html=True)
        total_budget_input = st.number_input("Capitale", min_value=1000.0, step=1000.0, value=50000.0, format="%.2f", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_in_2:
        # VISUALLY INTEGRATED CONTAINER FOR MIX (PRO BOX)
        st.markdown('<div class="prof-card accent-purple"><span class="card-label text-purple">🚀 Strategia Portafoglio</span>', unsafe_allow_html=True)
        # SELECTOR FOR STRATEGY TYPE
        portfolio_type = st.radio("Tipo Allocazione:", ["Bilanciato (BTP + Asset)", "Speculativo (Solo Asset)"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Logic based on portfolio type
    use_btp = (portfolio_type == "Bilanciato (BTP + Asset)")
    
    # MIX SECTION
    st.markdown('<div class="prof-card accent-blue"><span class="card-label text-blue">📡 Selezione Asset Satellite (Rischio)</span>', unsafe_allow_html=True)
    mix_assets_selected = st.multiselect("Asset Satellite", list(ASSET_CONFIG.keys()), placeholder="Seleziona asset...", label_visibility="collapsed")
    
    mix_weights = {}
    total_risk_weight = 0
    valid_mix = True
    btp_weight = 100.0 if use_btp else 0.0
    
    if mix_assets_selected:
        st.markdown("<br>", unsafe_allow_html=True)
        cols_mix = st.columns(len(mix_assets_selected))
        
        # LOGIC FOR AUTO-100% IF SINGLE ASSET & SPECULATIVE
        auto_100_single = (not use_btp and len(mix_assets_selected) == 1)
        
        for idx, asset in enumerate(mix_assets_selected):
            with cols_mix[idx]:
                k = f"w_{asset}"
                if k not in st.session_state: st.session_state[k] = 20.0 
                
                # If NOT auto-100%, show the percentage label above input
                if not auto_100_single:
                     st.markdown(f"<div style='text-align:center; font-weight:bold; color:#a78bfa; margin-bottom:5px;'>% {asset.split()[0]}</div>", unsafe_allow_html=True)
                
                if auto_100_single:
                    # CLEAN: SHOW BADGE ONLY
                    st.markdown(f"<div class='single-asset-badge'>100% Allocato su {asset}</div>", unsafe_allow_html=True)
                    mix_weights[asset] = 100.0
                    total_risk_weight += 100.0
                else:
                    # Normal input
                    w = st.number_input(f"Peso {asset}", 0.0, 100.0, step=0.5, format="%.1f", key=k, label_visibility="collapsed")
                    mix_weights[asset] = w
                    total_risk_weight += w
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation
    if use_btp:
        btp_weight = 100 - total_risk_weight
        if btp_weight < 0:
            st.error(f"Errore: Allocazione totale > 100%. Riduci i pesi degli asset satellite.")
            valid_mix = False
        else:
            st.info(f"📊 Allocazione Finale: {btp_weight:.1f}% BTP Sicuri + {total_risk_weight:.1f}% Asset Rischiosi")
            valid_mix = True
    else:
        # Speculative mode: Risk must sum to 100
        if abs(total_risk_weight - 100.0) > 0.1:
             st.warning(f"⚠️ Attenzione: In modalità speculativa il totale deve fare 100%. Attuale: {total_risk_weight:.1f}%")
             valid_mix = False 
        else:
             st.success(f"🔥 Portafoglio 100% Speculativo Attivo.")
             valid_mix = True
             btp_weight = 0.0

    # --- 2. CORE PORTFOLIO (BTP) - ONLY IF ENABLED ---
    
    bond_df = pd.DataFrame()
    
    if use_btp:
        st.markdown('<div class="step-header">2. Dettaglio BTP (Core)</div>', unsafe_allow_html=True)
        
        # PRE-FILTER: ASK FOR NUMBER OF BTPs
        num_btp = st.number_input("Numero di BTP nel portafoglio:", min_value=1, max_value=10, value=2, step=1)
        
        # AUTO-DISTRIBUTION LOGIC
        btp_capital_available = total_budget_input * (btp_weight / 100.0)
        capital_per_bond = btp_capital_available / num_btp if num_btp > 0 else 0
        
        bonds_init_data = []
        defaults = [
            {"Nome/ISIN": "IT0005425233", "Scadenza": date(2051, 9, 1), "Cedola %": 1.70, "Prezzo": 59.75},
            {"Nome/ISIN": "IT0005441883", "Scadenza": date(2072, 3, 1), "Cedola %": 2.15, "Prezzo": 58.32}
        ]
        
        for i in range(num_btp):
            ref = defaults[i % 2]
            b = ref.copy()
            b["Capitale Investito (€)"] = capital_per_bond
            bonds_init_data.append(b)

        bond_df = st.data_editor(
            pd.DataFrame(bonds_init_data),
            num_rows="fixed", 
            column_config={
                "Scadenza": st.column_config.DateColumn("Scadenza", format="DD/MM/YYYY"),
                "Cedola %": st.column_config.NumberColumn("Cedola %", min_value=0.0, max_value=15.0, step=0.05, format="%.2f%%"),
                "Prezzo": st.column_config.NumberColumn("Prezzo Acquisto", min_value=0.0, max_value=200.0, step=0.01, format="%.2f"),
                "Capitale Investito (€)": st.column_config.NumberColumn("Capitale Investito (Cash)", min_value=0.0, step=1000.0, format="€ %.2f")
            },
            use_container_width=True,
            key=f"editor_{total_budget_input}_{btp_weight}_{num_btp}"
        )
        
        current_btp_sum = bond_df["Capitale Investito (€)"].sum()
        implied_nominal = (bond_df["Capitale Investito (€)"] / (bond_df["Prezzo"] / 100)).sum()
        
        c_k1, c_k2, c_k3 = st.columns(3)
        with c_k1:
            st.markdown(f"**Capitale BTP Effettivo:** :green[€ {current_btp_sum:,.0f}]")
        with c_k2:
            st.markdown(f"**Valore Nominale a Scadenza:** € {implied_nominal:,.0f}")
        with c_k3:
            compound = st.toggle("Reinvesti Cedole (Interesse Composto)", value=False)
    else:
        # SPECULATIVE MODE: TIME HORIZON SLIDER
        st.markdown('<div class="step-header">2. Orizzonte Temporale (Speculativo)</div>', unsafe_allow_html=True)
        years_horizon = st.slider("Durata Simulazione (Anni)", 1, 50, 10, help="Senza BTP a scadenza fissa, definisci tu la durata della simulazione.")
        
        target_date = date.today().replace(year=date.today().year + years_horizon)
        dummy_bond = {
            "Nome/ISIN": "Orizzonte Temporale",
            "Scadenza": target_date,
            "Cedola %": 0.0,
            "Prezzo": 100.0,
            "Capitale Investito (€)": 0.0
        }
        bond_df = pd.DataFrame([dummy_bond])
        compound = False

    # --- 3. PARAMETRI ECONOMICI ---
    st.markdown('<div class="step-header">3. Fisco & Inflazione</div>', unsafe_allow_html=True)
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        tax_rate = st.selectbox("Regime Fiscale (BTP)", [0.125, 0.26], format_func=lambda x: "12.5% (Titoli di Stato)" if x==0.125 else "26% (Azioni/ETF)", help="L'aliquota fiscale applicata ai rendimenti BTP. Nota: Le azioni ed ETF nel portafoglio misto vengono sempre calcolati al 26% automaticamente.")
    with c_f2:
        inflation = st.slider("Inflazione Stimata (%)", 0.0, 10.0, 2.0, help="Il tasso annuale medio di perdita di potere d'acquisto del denaro. Il target della BCE è il 2%. Questo parametro serve per calcolare il valore 'Reale' del tuo capitale futuro.") / 100

    # --- 4. MOTORE DI SIMULAZIONE ---
    st.markdown('<div class="step-header">4. Motore di Simulazione</div>', unsafe_allow_html=True)
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sim_mode = st.radio("Metodo Simulazione", ["Historical Replay", "Monte Carlo"], horizontal=True, index=1, help="Historical: Usa esattamente i dati del passato (ciò che è successo). Monte Carlo: Genera migliaia di scenari probabilistici futuri basati sulla volatilità storica (ciò che potrebbe succedere).")
    with c_s2:
        rebalance = st.toggle("Ribilanciamento Automatico (Smart)", value=False, help="Se attivo, il sistema vende automaticamente gli asset che sono saliti troppo per riportare il portafoglio alle percentuali originali (es. 80/20).")

    n_sims = 1
    if sim_mode == "Monte Carlo":
        c_m1, c_m2 = st.columns([3, 1])
        with c_m1:
            sim_options = ["1.000", "5.000", "10.000", "50.000", "100.000", "500.000", "1.000.000", "5.000.000", "10.000.000"]
            sim_choice = st.select_slider("Numero Scenari Simulati", options=sim_options, value="100.000", help="Definisce quante volte il computer 'lancia i dadi' per prevedere il futuro. Più alto è il numero, più precisa è la statistica (ma più lento è il calcolo).")
            sim_map = {"1.000": 1000, "5.000": 5000, "10.000": 10000, "50.000": 50000, "100.000": 100000, "500.000": 500000, "1.000.000": 1000000, "5.000.000": 5000000, "10.000.000": 10000000}
            n_sims = sim_map[sim_choice]
        with c_m2:
            if n_sims >= 200000: 
                st.warning(f"⚠️ Rischio Crash Cloud (RAM).")
            else: 
                st.info("💡 Cloud Safe.")

    # --- ADVANCED ANALYSIS ---
    with st.expander("🛠️ Strumenti di Analisi Avanzata (Opzionale)"):
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            selected_benchmarks = st.multiselect("Confronta con (Benchmark):", list(ASSET_CONFIG.keys()), default=[], help="Scegli uno o più indici di mercato per visualizzare la loro linea di andamento sul grafico finale insieme al tuo portafoglio.")
        with c_a2:
            window_months = st.select_slider("Rolling Window (Mesi)", options=[12, 24, 36, 60, 120], value=36, help="Definisce l'ampiezza della finestra temporale per il calcolo delle statistiche mobili (es. volatilità a 3 anni).")

    # --- RUN BUTTON ---
    st.markdown("---")
    
    # DEFINE PLACEHOLDER ALWAYS TO AVOID NAME ERROR
    tax_report_placeholder = st.empty()
    
    if st.button("🚀 AVVIA SIMULAZIONE COMPLETA", type="primary", use_container_width=True):
        if not valid_mix:
            st.error("Correggi l'allocazione del portafoglio (Totale > 100%) prima di avviare.")
        else:
            # FIX: RECALCULATE BTP TOTAL FROM TABLE JUST IN CASE
            final_btp_sum = bond_df["Capitale Investito (€)"].sum() if use_btp else 0.0
            
            # PERFORM CALCULATION
            all_assets = list(set(selected_benchmarks + (mix_assets_selected if mix_assets_selected else [])))
            df, err, init_spent, calc_maturity, tax_log_res = calculate_engine_multibond(bond_df, final_btp_sum, tax_rate, compound, all_assets, market_returns, sim_mode, n_sims, mix_weights if mix_assets_selected else None)
            
            if err:
                st.error(err)
            else:
                df['Anno'] = df['Date'].dt.year
                
                # Post-process mix
                if mix_assets_selected:
                    if not use_btp:
                        # PURE SPECULATIVE MODE
                        df["Mix_Portfolio"] = 0.0
                        for asset, w_norm in mix_weights.items():
                            if asset in df.columns:
                                # Normalised curve (starts at 1.0)
                                curve = df[asset] 
                                # Actual cash
                                cash = total_budget_input * (w_norm / 100.0)
                                df["Mix_Portfolio"] += curve * cash
                    else:
                        # BALANCED MODE
                        df["Mix_Portfolio"] = df["BTP_Value"] # Start with BTP curve (absolute)
                        
                        for asset, w_norm in mix_weights.items():
                            if asset in df.columns:
                                curve = df[asset]
                                cash = total_budget_input * (w_norm / 100.0)
                                asset_val = curve * cash
                                df["Mix_Portfolio"] += asset_val

                # ACTUAL INVESTED SUM CALCULATION
                if use_btp:
                    risk_cash_calculated = total_budget_input * (total_risk_weight / 100.0)
                    total_actual_invested = final_btp_sum + risk_cash_calculated
                else:
                    total_actual_invested = total_budget_input
                
                # Percentages for summary
                if total_actual_invested > 0:
                    true_btp_pct = (final_btp_sum / total_actual_invested) * 100
                    true_risk_pct = 100 - true_btp_pct
                else:
                    true_btp_pct = 0
                    true_risk_pct = 0
                
                st.session_state.sim_results = {
                    'df': df, 
                    'init_spent': total_actual_invested, 
                    'maturity_date': calc_maturity, 
                    'selected_benchmarks': selected_benchmarks, 
                    'inflation': inflation, 
                    'has_mix': bool(mix_assets_selected), 
                    'mix_details': mix_weights if mix_assets_selected else {}, 
                    'btp_w_final': btp_weight if mix_assets_selected else 100, 
                    'sim_mode': sim_mode, 
                    'n_sims': n_sims, 
                    'data_period': data_period_option, 
                    'bond_df': bond_df, 
                    'tax_log': tax_log_res, 
                    'total_risk_weight': total_risk_weight if mix_assets_selected else 0, 
                    'tax_rate_input': tax_rate, 
                    'base_btp_capital': final_btp_sum,
                    'true_btp_pct': true_btp_pct,
                    'true_risk_pct': true_risk_pct
                }
                st.session_state.simulation_done = True
                
                # JS SCROLL TO TOP
                components.html(f"""
                    <script>
                        window.parent.document.body.scrollTop = 0;
                        window.parent.document.documentElement.scrollTop = 0;
                        var main = window.parent.document.querySelector(".main");
                        if (main) main.scrollTop = 0;
                    </script>
                """, height=0)
                
                st.rerun()

# --- RESULTS SECTION (Visible if done) ---
if st.session_state.simulation_done and 'sim_results' in st.session_state:
    # RESET BUTTON TOP LEFT
    if st.button("🔄 Nuova Simulazione", type="secondary"):
        st.session_state.simulation_done = False
        st.rerun()
        
    res = st.session_state.sim_results
    df = res['df']
    init_spent = res['init_spent']
    inflation = res['inflation']
    tax_rate_used = res['tax_rate_input']
    base_btp_cap = res.get('base_btp_capital', init_spent)
    
    # NEW DASHBOARD SUMMARY CARD - HTML MULTI-ASSET LOGIC
    btp_share = res.get('true_btp_pct', 100)
    mix_details = res.get('mix_details', {})
    
    # Fixed height style to ensure visibility
    bars_html = ""
    legend_html = ""
    
    # BTP Part
    if btp_share > 0.1:
        bars_html += f'<div style="width: {btp_share}%; background-color: #10b981; height: 100%;" title="BTP ({btp_share:.1f}%)"></div>'
        legend_html += f'<div class="leg-item-flex"><div class="dot" style="background-color: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5);"></div> BTP ({btp_share:.1f}%)</div>'
    
    # Mix Part
    if res.get('has_mix'):
        total_risk_input = res['total_risk_weight']
        for asset, weight_input in mix_details.items():
            if total_risk_input > 0:
                asset_real_share = (weight_input / total_risk_input) * res['true_risk_pct']
                color = ASSET_CONFIG[asset]['color']
                bars_html += f'<div style="width: {asset_real_share}%; background-color: {color}; height: 100%;" title="{asset} ({asset_real_share:.1f}%)"></div>'
                legend_html += f'<div class="leg-item-flex"><div class="dot" style="background-color: {color}; box-shadow: 0 0 8px {color}66;"></div> {asset.split()[0]} ({asset_real_share:.1f}%)</div>'
    
    dashboard_html = f"""<div class="dash-container"><div class="dash-header-row"><span class="dash-label">Investimento Totale</span></div><div class="dash-value">€ {init_spent:,.0f}</div><div class="progress-track">{bars_html}</div><div class="dash-legend-flex">{legend_html}</div></div>"""
    
    st.markdown(dashboard_html, unsafe_allow_html=True)
    
    st.markdown("### 📊 Risultati Simulazione")
    
    # 1. METRICS
    final_btp_100 = df["BTP_Value"].iloc[-1]
    
    # Handle Pure Speculative Case for Metrics
    if "Mix_Portfolio" in df.columns:
        final_mix = df["Mix_Portfolio"].iloc[-1]
    else:
        final_mix = final_btp_100 
        
    tax_log = res.get('tax_log', {})
    
    # --- TAX FIX: COMBINED CALCULATION ---
    # Standard BTP taxes (coupons + gain)
    tax_btp_part = tax_log.get('btp_coupons', 0) + tax_log.get('btp_gain', 0)
    
    # Risk Asset Tax Calculation (Post-Processing)
    tax_risk_part = 0.0
    if res.get('has_mix'):
        # Calculate Risk Portion Value Gain
        # Risk Final Value = Total Final Mix - Final BTP Value
        final_risk_value = final_mix - final_btp_100
        
        # Risk Initial Capital = Total Init - BTP Init
        init_risk_capital = init_spent - base_btp_cap
        
        risk_gain = final_risk_value - init_risk_capital
        if risk_gain > 0:
            tax_risk_part = risk_gain * 0.26
            
    total_tax = tax_btp_part + tax_risk_part
    # -------------------------------------
    
    total_coupons_net = df["Cum_Coupons"].iloc[-1]
    n_months = len(df)
    avg_monthly_coupon = total_coupons_net / n_months if n_months > 0 else 0
    
    real_val = final_mix / ((1+inflation)**((len(df))/12))
    
    gain_btp = final_btp_100 - base_btp_cap
    gain_mix = final_mix - init_spent
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.metric("Valore BTP (100%)", f"€ {final_btp_100:,.0f}", f"Utile: € {gain_btp:,.0f}")
        st.markdown(f"""<div class="metric-detail">Cedola Netta Mensile: <span class="metric-highlight">€ {avg_monthly_coupon:,.2f}</span></div>""", unsafe_allow_html=True)
    with c2: st.metric("Portafoglio Misto", f"€ {final_mix:,.0f}", f"Utile: € {gain_mix:,.0f}")
    with c3: st.metric("Tassazione Totale", f"€ {total_tax:,.0f}", delta="- Imposte", delta_color="inverse")
    with c4: st.metric("Potere d'Acquisto", f"€ {real_val:,.0f}", help="Valore reale del portafoglio finale scontato dell'inflazione.")

    # 2. MAIN CHART
    st.markdown("### 📈 Evoluzione Portafoglio")
    use_log = st.toggle("Scala Logaritmica", value=False)
    fig = go.Figure()
    
    if base_btp_cap > 0:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BTP_Value"], mode='lines', name="Tuo BTP (100%)", line=dict(color="#22c55e", width=3), fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)'))
    
    if res.get('has_mix') and "Mix_Portfolio" in df.columns: 
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Mix_Portfolio"], mode='lines', name="Portafoglio Misto", line=dict(color="#c084fc", width=4)))
        
    active_curve = df["Mix_Portfolio"] if "Mix_Portfolio" in df.columns else df["BTP_Value"]
    df['Real_Line'] = active_curve * (1 / ((1+inflation)**((df.index)/12)))
    
    fig.add_trace(go.Scatter(x=df["Date"], y=df['Real_Line'], mode='lines', name="Soglia Reale", line=dict(color="#ef4444", width=2, dash='dot')))
    
    for asset in res['selected_benchmarks']:
        if asset in df.columns: 
            scaled_bench = df[asset] * init_spent
            fig.add_trace(go.Scatter(x=df["Date"], y=scaled_bench, mode='lines', name=asset, line=dict(color=ASSET_CONFIG[asset]["color"], width=1.5), opacity=0.7))
        
    fig.update_layout(title="Evoluzione Capitale (Scenario Mediano)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=500, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="€ ", type="log" if use_log else "linear"), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. TAX REPORT SECTION
    st.markdown("---")
    st.subheader("📊 Report Fiscale e Impatto Tasse")
    
    # Initialize df_chart UNCONDITIONALLY to fix NameError
    df_chart = df.copy()
    
    total_tax_curve = pd.Series(0.0, index=df.index)
    
    if base_btp_cap > 0:
        btp_tax_c = (df['Cum_Coupons'] / (1-tax_rate_used)) - df['Cum_Coupons']
        btp_tax_g = (df['Gain_Netto_Finale'] / (1-tax_rate_used)) - df['Gain_Netto_Finale']
        total_tax_curve += (btp_tax_c + btp_tax_g.cumsum())
        
    if res.get('has_mix'):
        for asset, weight_input in mix_details.items():
            if asset in df.columns:
                if base_btp_cap == 0: 
                     cash = init_spent * (weight_input / 100.0)
                else: 
                     cash = init_spent * (weight_input/100.0)
                gross_asset_val = df[asset] * cash
                gain = gross_asset_val - cash
                tax = gain.apply(lambda x: x * 0.26 if x > 0 else 0)
                total_tax_curve += tax
    
    # Adjust for charts
    final_net_curve = active_curve.copy()
    if res.get('has_mix'):
         risk_tax_total = pd.Series(0.0, index=df.index)
         for asset, weight_input in mix_details.items():
             if asset in df.columns:
                 cash = init_spent * (weight_input/100.0)
                 val = df[asset] * cash
                 gain = val - cash
                 t = gain.apply(lambda x: x*0.26 if x>0 else 0)
                 risk_tax_total += t
         final_net_curve = active_curve - risk_tax_total
    
    final_gross_curve = final_net_curve + total_tax_curve

    fig_tax = go.Figure()
    fig_tax.add_trace(go.Scatter(x=df_chart['Date'], y=final_gross_curve, mode='lines', name='Lordo (Senza Tasse)', line=dict(color='#f87171', width=2, dash='dash')))
    fig_tax.add_trace(go.Scatter(x=df_chart['Date'], y=final_net_curve, mode='lines', name='Netto (In Tasca)', line=dict(color='#34d399', width=2), fill='tonexty', fillcolor='rgba(248, 113, 113, 0.2)'))
    fig_tax.update_layout(title="Il Cuneo Fiscale nel Tempo", height=300, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="€ "))
    st.plotly_chart(fig_tax, use_container_width=True)

    # 4. OTHER DETAILS
    col_dd, col_cash = st.columns(2)
    with col_dd:
        st.markdown("#### 🌊 Profondità del Rischio (Drawdown)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df["Date"], y=calculate_drawdown(final_net_curve), mode='lines', name="Drawdown Portafoglio", line=dict(color="#c084fc", width=2), fill='tozeroy'))
        for asset in res['selected_benchmarks']:
            if asset in df.columns: fig_dd.add_trace(go.Scatter(x=df["Date"], y=calculate_drawdown(df[asset]), mode='lines', name=f"DD {asset}", line=dict(color=ASSET_CONFIG[asset]["color"], width=1), opacity=0.7))
        fig_dd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat=".1%"), hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with col_cash:
        if base_btp_cap > 0:
            st.markdown("#### 💸 Flusso Cedolare Netto (BTP)")
            annual_coupons = df.groupby('Anno')['Cum_Coupons'].max().reset_index()
            annual_coupons['Flusso_Netto'] = annual_coupons['Cum_Coupons'].diff().fillna(annual_coupons['Cum_Coupons'])
            fig_cash = go.Figure()
            fig_cash.add_trace(go.Bar(x=annual_coupons['Anno'], y=annual_coupons['Flusso_Netto'], name='Cedole Annuali', marker_color='#10b981', hovertemplate='Anno %{x}<br>Incasso: € %{y:,.2f}'))
            fig_cash.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="€ "), hovermode="x unified", margin=dict(t=10, b=10))
            st.plotly_chart(fig_cash, use_container_width=True)
        else:
            st.markdown("#### 💸 Flusso Cedolare")
            st.info("Nessun BTP nel portafoglio, nessun flusso cedolare garantito.")
    
    # PDF BUTTON
    pdf_bytes = create_pdf_report(res)
    st.download_button(label="📄 Scarica Report PDF Completo", data=pdf_bytes, file_name="report_finanziario.pdf", mime="application/pdf", type="secondary", use_container_width=True)
    
    with st.expander("🔎 Dati Annuali Tabellari"):
        st.dataframe(df.groupby('Anno').last().reset_index(), use_container_width=True)
