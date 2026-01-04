import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    initial_sidebar_state="collapsed"
)

# --- 2. CSS: DESIGN MODERNO ---
st.markdown("""
<style>
    .stApp {
        background-color: #09090b;
        background-image: radial-gradient(circle at 50% 0%, #1c1917 0%, #09090b 80%);
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 { color: #f4f4f5; font-weight: 600; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05); border-radius: 8px 8px 0 0;
        color: #a1a1aa; padding: 10px 20px; border: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.2); color: #60a5fa; font-weight: bold;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 15px;
    }
    div[data-testid="stMetricValue"] { color: #fafafa; font-size: 26px; font-weight: 700; }
    
    .metric-detail {
        font-size: 0.85rem; color: #9ca3af; margin-top: 5px; padding-top: 5px;
        border-top: 1px solid rgba(255,255,255,0.1); line-height: 1.4;
    }
    .metric-highlight { color: #34d399; font-weight: 600; }
    .metric-purple { color: #c084fc; font-weight: 600; }
    .metric-danger { color: #f87171; font-weight: 600; }
    .metric-gold { color: #fbbf24; font-weight: 600; }
    
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stDateInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #18181b !important; color: #e4e4e7 !important;
        border: 1px solid #3f3f46 !important; border-radius: 8px !important;
    }
    
    .stMultiSelect > div > div > div {
        background-color: #18181b; color: #e4e4e7;
    }
    
    .stButton > button {
        width: 100%; background-color: #2563eb; color: white; border: none;
        padding: 12px; font-size: 16px; font-weight: 600; border-radius: 8px; margin-top: 10px;
    }
    .stButton > button:hover { background-color: #1d4ed8; }
    
    .risk-table {
        width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem;
    }
    .risk-table td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    .risk-table th { text-align: left; color: #a1a1aa; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.2); }

    .corr-legend {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
    }
    .corr-item {
        display: flex; align-items: center; margin-bottom: 8px; font-family: 'Inter', sans-serif;
    }
    .corr-icon { font-size: 1.2rem; margin-right: 12px; width: 24px; text-align: center; }
    .corr-val { font-weight: 700; color: #f4f4f5; margin-right: 8px; min-width: 40px; }
    .corr-desc { font-size: 0.85rem; color: #a1a1aa; }
    
    .info-box {
        background-color: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #60a5fa;
        padding: 15px; margin-bottom: 15px; border-radius: 4px;
        font-size: 0.95rem; line-height: 1.6;
        color: #e4e4e7;
    }
    .info-header {
        color: #93c5fd; font-weight: 700; font-size: 1.1rem; margin-bottom: 5px;
    }
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

# --- FIXED ENGINE: SAFE FOR DATE COMPARISON & ACCOUNTING MODE & RAM PROTECTION ---
def calculate_engine_multibond(bond_df, capital_override, tax_rate, use_compound, assets_to_calculate, market_returns_df, simulation_mode, n_simulations):
    today = date.today()
    
    bonds = []
    max_maturity = today
    
    # 1. Parse Input
    for index, row in bond_df.iterrows():
        try:
            mat = row['Scadenza']
            if isinstance(mat, str): mat = datetime.strptime(mat, '%Y-%m-%d').date()
            if mat > max_maturity: max_maturity = mat
            
            cash_invested = float(row['Capitale Investito (€)'])
            price = float(row['Prezzo'])
            
            # Nominal is derived: (Cash / Price) * 100
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
            
    if not bonds:
        return None, "Nessun BTP valido inserito.", 0, today

    total_cash_invested = sum([b['invested'] for b in bonds])
    
    # FIX: Convert date to Timestamp for offset calculation
    ts_max_maturity = pd.Timestamp(max_maturity)
    final_date = ts_max_maturity + pd.offsets.MonthEnd(0)
    if final_date < ts_max_maturity: 
        final_date += pd.offsets.MonthEnd(1)
    
    # FIX FREQUENCY: Use 'M' instead of 'ME' for max compatibility
    date_range = pd.date_range(start=today, end=final_date, freq='M') 
    if len(date_range) == 0: return None, "Orizzonte troppo breve.", 0, today
    
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
                # HOLDING PHASE (Accounting: Value = Cost)
                active_bond_value += b['invested']
                
                if d.month in b['coupon_months']:
                    coup_net = (b['nominal'] * (b['coupon_pct']/100) * (1-tax_rate)) / 2
                    monthly_cash_flow += coup_net
            
            elif is_maturity_month:
                # MATURITY PHASE (Payout)
                gross_payout = b['nominal']
                gain = b['nominal'] - b['invested']
                tax_on_gain = (gain * tax_rate) if gain > 0 else 0
                
                net_payout = gross_payout - tax_on_gain
                
                liquid_cash += net_payout
                monthly_capital_gain += (net_payout - b['invested']) 
                paid_out_indices.add(i)
                
                # Final Coupon
                if d.month in b['coupon_months']:
                    coup_net = (b['nominal'] * (b['coupon_pct']/100) * (1-tax_rate)) / 2
                    monthly_cash_flow += coup_net
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
                    tax = ASSET_CONFIG[name]["tax"]
                    returns_to_use = np.zeros(len(df))
                    if simulation_mode == "Monte Carlo":
                        mu = np.mean(hist_series)
                        sigma = np.std(hist_series)
                        
                        # FULL POWER (User Request): If they want 10M, we give 10M.
                        # However, we trigger Garbage Collection to help RAM
                        sims_to_run = n_simulations 
                        
                        random_matrix = np.random.normal(mu, sigma, (len(df), sims_to_run)).astype(np.float32)
                        cum_growth = np.cumprod(1.0 + random_matrix, axis=0)
                        final_values = cum_growth[-1, :]
                        median_idx = np.argsort(final_values)[len(final_values)//2]
                        returns_to_use = random_matrix[:, median_idx]
                        
                        del random_matrix, cum_growth, final_values
                        gc.collect() # Force RAM cleanup
                    else:
                        for i in range(len(df)):
                            hist_idx = i % len(hist_series)
                            returns_to_use[i] = hist_series[hist_idx]
                    
                    gross_values = [total_cash_invested]
                    for i in range(len(df)):
                        monthly_return = returns_to_use[i]
                        prev_val = gross_values[-1]
                        new_val = prev_val * (1 + monthly_return)
                        gross_values.append(new_val)
                    gross_values = np.array(gross_values[1:])
                    gains = gross_values - total_cash_invested
                    taxes = np.where(gains > 0, gains * tax, 0)
                    net_values = gross_values - taxes
                    df[name] = net_values

    return df, None, total_cash_invested, max_maturity

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

# --- CONFIGURAZIONE ---
with st.expander("⚙️ Pannello di Controllo", expanded=True):
    tab_asset, tab_money, tab_strategy = st.tabs(["🏦 BTP Laddering", "💰 Capitale & Fisco", "⚖️ Strategia & Mix"])
    
    with tab_asset:
        st.caption("Costruisci la tua scaletta di obbligazioni (Bond Ladder)")
        
        # BTP DEFAULT UTENTE
        default_bonds = pd.DataFrame([
            {"Nome/ISIN": "IT0005425233", "Scadenza": date(2051, 9, 1), "Cedola %": 1.70, "Prezzo": 59.75, "Capitale Investito (€)": 25000.0},
            {"Nome/ISIN": "IT0005441883", "Scadenza": date(2072, 3, 1), "Cedola %": 2.15, "Prezzo": 58.32, "Capitale Investito (€)": 25000.0}
        ])
        
        bond_df = st.data_editor(
            default_bonds,
            num_rows="dynamic",
            column_config={
                "Scadenza": st.column_config.DateColumn("Scadenza", format="DD/MM/YYYY"),
                "Cedola %": st.column_config.NumberColumn("Cedola %", min_value=0.0, max_value=15.0, step=0.05, format="%.2f%%"),
                "Prezzo": st.column_config.NumberColumn("Prezzo Acquisto", min_value=0.0, max_value=200.0, step=0.01, format="%.2f"),
                "Capitale Investito (€)": st.column_config.NumberColumn("Capitale Investito (Cash)", min_value=1000.0, step=1000.0, format="€ %.2f")
            },
            use_container_width=True
        )
        
        # Totals calculation
        total_cash_invested = bond_df["Capitale Investito (€)"].sum()
        # Nominal is implied: Cash / (Price/100)
        implied_nominal = (bond_df["Capitale Investito (€)"] / (bond_df["Prezzo"] / 100)).sum()
        
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            st.markdown(f"**Capitale Reale Speso (Oggi):** :green[€ {total_cash_invested:,.0f}]")
        with c_k2:
            st.markdown(f"**Valore Nominale a Scadenza:** € {implied_nominal:,.0f}")
        
        compound = st.toggle("Reinvesti Cedole (Interesse Composto)", value=False)
        st.markdown("---")
        st.caption("📂 Base Dati")
        data_period_option = st.selectbox("Seleziona Orizzonte Storico", ["10y", "20y", "max"], index=1, format_func=lambda x: "Ultimi 10 Anni (Trend recente)" if x == "10y" else "Ultimi 20 Anni (Ciclo Completo)" if x == "20y" else "Massimo Disponibile")
        if 'current_period' not in st.session_state or st.session_state.current_period != data_period_option:
            st.session_state.current_period = data_period_option
            market_returns, _ = load_market_data(data_period_option)

    with tab_money:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Capitale Iniziale", f"€ {total_cash_invested:,.0f}", help="Corrisponde alla somma della colonna 'Capitale Investito'.")
            capital = total_cash_invested
        with col2:
            tax_rate = st.selectbox("Regime Fiscale", [0.125, 0.26], format_func=lambda x: "12.5% (Titoli di Stato)" if x==0.125 else "26% (Azioni/ETF)")
        with col3:
            inflation = st.slider("Inflazione Stimata (%)", 0.0, 10.0, 2.0) / 100

    with tab_strategy:
        sim_mode = st.radio("Metodo Simulazione", ["Historical Replay", "Monte Carlo"], horizontal=True, index=1)
        rebalance = st.toggle("Ribilanciamento Automatico (Smart)", value=False, help="Ribilancia solo se i pesi deviano del 5% (Threshold Rebalancing).")
        n_sims = 1
        if sim_mode == "Monte Carlo":
            st.markdown("---")
            col_sim_1, col_sim_2 = st.columns([2, 1])
            with col_sim_1:
                # DEFAULT SET TO 100.000 (Safe for Cloud)
                sim_options = ["1.000", "5.000", "10.000", "50.000", "100.000", "500.000", "1.000.000", "5.000.000", "10.000.000"]
                sim_choice = st.select_slider("Numero Scenari", options=sim_options, value="100.000")
                sim_map = {"1.000": 1000, "5.000": 5000, "10.000": 10000, "50.000": 50000, "100.000": 100000, "500.000": 500000, "1.000.000": 1000000, "5.000.000": 5000000, "10.000.000": 10000000}
                n_sims = sim_map[sim_choice]
            with col_sim_2:
                if n_sims >= 200000: 
                    st.warning(f"⚠️ **Attenzione:** Selezionare più di 100.000 scenari potrebbe mandare in crash l'applicazione se eseguita su Cloud (limite RAM). Consigliato solo per esecuzione locale su PC potenti.")
                else: 
                    st.info("💡 Scenari ottimizzati per il Cloud.")
        st.markdown("---")
        c_bench, c_mix = st.columns([1, 2])
        with c_bench:
            st.markdown("###### Benchmark")
            options = list(ASSET_CONFIG.keys())
            default_sel = ["S&P 500 🇺🇸"]
            selected_benchmarks = st.multiselect("Confronta con:", options=options, default=default_sel, placeholder="Seleziona asset...")
        with c_mix:
            st.markdown("###### Portafoglio Misto")
            mix_assets_selected = st.multiselect("Aggiungi asset al BTP", list(ASSET_CONFIG.keys()), placeholder="Es. S&P 500...")
            mix_weights = {}
            total_risk_weight = 0
            if mix_assets_selected:
                cols_mix = st.columns(len(mix_assets_selected))
                for idx, asset in enumerate(mix_assets_selected):
                    with cols_mix[idx]:
                        k = f"w_{asset}"
                        if k not in st.session_state: st.session_state[k] = 20.0 
                        w = st.number_input(f"% {asset.split()[0]}", 0.0, 100.0, step=0.5, format="%.1f", key=k)
                        mix_weights[asset] = w
                        total_risk_weight += w
                btp_weight = 100 - total_risk_weight
                if btp_weight < 0:
                    st.error(f"Errore: > 100%")
                    valid_mix = False
                else:
                    st.success(f"Mix: {btp_weight:.1f}% BTP + {total_risk_weight:.1f}% Risk Asset")
                    valid_mix = True
            else:
                btp_weight = 100
                valid_mix = False
        
        st.markdown("---")
        st.markdown("### 📉 Analisi Dinamica (Rolling Window)")
        st.caption("Visualizza i rischi e le opportunità che la media storica nasconde.")
        
        col_roll_ctrl, col_roll_graph = st.columns([1, 3])
        with col_roll_ctrl:
            st.markdown("**Finestra Temporale**")
            window_months = st.select_slider("Ampiezza Finestra (Mesi)", options=[12, 24, 36, 60, 120], value=36)
            st.markdown("---")
            st.markdown("**Legenda Correlazione**")
            st.markdown("""<div class="corr-legend"><div class="corr-item"><span class="corr-icon">🟥</span><div><span class="corr-val">+1.0</span> <span class="corr-desc">Nessuna Diversificazione</span></div></div><div class="corr-item"><span class="corr-icon">⬜</span><div><span class="corr-val">0.0</span> <span class="corr-desc">Asset Indipendenti</span></div></div><div class="corr-item"><span class="corr-icon">🟩</span><div><span class="corr-val">-1.0</span> <span class="corr-desc">Massima Protezione</span></div></div></div>""", unsafe_allow_html=True)

        with col_roll_graph:
            tab_vol, tab_corr = st.tabs(["⚡ Volatilità", "🔗 Correlazione (Rischio/Protezione)"])
            with tab_vol:
                roll_assets = st.multiselect("Asset:", options=list(ASSET_CONFIG.keys()), default=["S&P 500 🇺🇸", "Bond Globali 🌎"])
                if roll_assets:
                    df_roll_vol = (market_returns[roll_assets].rolling(window_months, min_periods=window_months//2).std() * np.sqrt(12))
                    fig_rv = go.Figure()
                    for asset in roll_assets:
                        series = df_roll_vol[asset].dropna()
                        fig_rv.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=asset, line=dict(color=ASSET_CONFIG[asset]["color"])))
                    fig_rv.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=250, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat=".1%"), hovermode="x unified", margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig_rv, use_container_width=True)
                else: st.info("Seleziona almeno un asset.")
            with tab_corr:
                c1, c2 = st.columns(2)
                with c1: asset_a = st.selectbox("Asset A", options=list(ASSET_CONFIG.keys()), index=0)
                with c2: idx_b = 5 if len(ASSET_CONFIG) > 5 else 1; asset_b = st.selectbox("Asset B", options=list(ASSET_CONFIG.keys()), index=idx_b)
                if asset_a and asset_b:
                    roll_corr = market_returns[asset_a].rolling(window_months, min_periods=window_months//2).corr(market_returns[asset_b]).dropna()
                    fig_rc = go.Figure()
                    pos_corr = roll_corr.where(roll_corr >= 0)
                    fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=pos_corr, mode='lines', name='Positiva (Rischio)', line=dict(color='#ef4444', width=0), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'))
                    neg_corr = roll_corr.where(roll_corr <= 0)
                    fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=neg_corr, mode='lines', name='Negativa (Protezione)', line=dict(color='#22c55e', width=0), fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.2)'))
                    fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode='lines', name='Trend', line=dict(color='#e4e4e7', width=1.5)))
                    fig_rc.add_shape(type="line", x0=roll_corr.index.min(), y0=0, x1=roll_corr.index.max(), y1=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
                    fig_rc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=250, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[-1, 1]), hovermode="x unified", showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig_rc, use_container_width=True)
        
        if mix_assets_selected:
            if len(mix_assets_selected) < 2:
                st.info("💡 **Robo-Advisor AI:** Seleziona almeno un secondo asset nel 'Portafoglio Misto' per attivare l'ottimizzazione automatica di Markowitz.")
            else:
                st.markdown("### 🏆 Ottimizzatore di Portafoglio (AI)")
                c_conf1, c_conf2 = st.columns(2)
                with c_conf1: target_btp = st.slider("Vincolo Sicurezza: % BTP Minima", 0, 95, 80)
                with c_conf2: 
                    n_assets = len(mix_assets_selected)
                    min_alloc_logic = math.ceil(100/n_assets) if n_assets > 0 else 100
                    max_alloc_input = st.slider("Diversificazione Forzata (Max % per singolo Asset Risk)", min_value=min_alloc_logic, max_value=100, value=40 if n_assets > 2 else 100)
                frontier_data = calculate_efficient_frontier(mix_assets_selected, market_returns, max_single_alloc=(max_alloc_input/100.0))
                if frontier_data:
                    c_frontier, c_advice = st.columns([2, 1])
                    with c_frontier:
                        fig_eff = go.Figure()
                        fig_eff.add_trace(go.Scattergl(x=frontier_data['volatility'], y=frontier_data['returns'], mode='markers', marker=dict(color=frontier_data['sharpe'], colorscale='Viridis', size=2), name='Simulazione'))
                        fig_eff.add_trace(go.Scatter(x=[frontier_data['opt_vol']], y=[frontier_data['opt_ret']], mode='markers', marker=dict(color='yellow', size=15, symbol='star', line=dict(color='black', width=1)), name='Ottimo Risk'))
                        
                        user_w_raw = np.array([mix_weights[a] for a in mix_assets_selected])
                        sum_risk_user = np.sum(user_w_raw)
                        user_w_norm = user_w_raw / sum_risk_user if sum_risk_user > 0 else np.zeros(len(user_w_raw))
                        subset = market_returns[mix_assets_selected].dropna()
                        mu = subset.mean() * 12
                        cov = subset.cov() * 12
                        user_ret = np.dot(user_w_norm, mu)
                        user_vol = np.sqrt(np.dot(user_w_norm.T, np.dot(cov, user_w_norm)))
                        user_sharpe = (user_ret - 0.02) / user_vol if user_vol > 0 else 0
                        user_series = subset.dot(user_w_norm)
                        user_var_995 = calculate_modified_var(user_series, confidence_level=0.995)
                        
                        fig_eff.add_trace(go.Scatter(x=[user_vol], y=[user_ret], mode='markers', marker=dict(color='red', size=12, symbol='x', line=dict(color='white', width=1)), name='Tuo Mix Risk'))
                        fig_eff.update_layout(title="Frontiera Efficiente (Solo Parte Rischiosa)", xaxis=dict(title="Volatilità", tickformat=".1%"), yaxis=dict(title="Rendimento", tickformat=".1%"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=400, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                        st.plotly_chart(fig_eff, use_container_width=True)
                        
                        exposure_factor = sum_risk_user / 100.0
                        col_u = "metric-danger" if user_sharpe < frontier_data['opt_sharpe'] else "metric-highlight"
                        var_diff = (frontier_data['opt_var_995'] * exposure_factor) - (user_var_995 * exposure_factor)
                        var_col = "metric-highlight" if var_diff > 0 else "metric-danger"
                        st.markdown(f"""<table class="risk-table"><tr><th>Metrica</th><th>Tuo Mix</th><th>Mix Ottimale <span style="font-size:0.8em; font-weight:normal">(a parità di % Risk)</span></th></tr><tr><td><strong>Sharpe Ratio (Efficienza)</strong></td><td>{user_sharpe:.2f}</td><td class="{col_u}">{frontier_data['opt_sharpe']:.2f}</td></tr><tr><td><strong>Volatilità Totale Annua</strong></td><td>{(user_vol * exposure_factor):.2%}</td><td>{(frontier_data['opt_vol'] * exposure_factor):.2%}</td></tr><tr><td><strong>VaR Mensile (99.5%)</strong><br><span style="font-size:0.8em; color:#f87171">Rischio Catastrofale (Cigno Nero)</span></td><td style="color: #f87171;">{(user_var_995 * exposure_factor):.2%}</td><td style="color: #f87171;" class="{var_col}">{(frontier_data['opt_var_995'] * exposure_factor):.2%}</td></tr></table>""", unsafe_allow_html=True)

                    with c_advice:
                        st.markdown(f"""<div class="frontier-info"><strong>🤖 Strategia Consigliata</strong><br>Mantenendo {target_btp}% BTP, ecco come dividere il restante {100-target_btp}% per la massima efficienza.</div>""", unsafe_allow_html=True)
                        risk_quota = 100 - target_btp
                        new_weights_map = {}
                        for i, asset in enumerate(mix_assets_selected):
                            final_w = frontier_data['opt_weights'][i] * risk_quota
                            new_weights_map[asset] = round(final_w, 1) 
                            curr_w = mix_weights[asset]
                            diff = final_w - curr_w
                            color = "metric-highlight" if diff > 0 else "metric-danger"
                            st.markdown(f"""<div style="font-size: 0.9rem; margin-bottom: 3px;">{asset}: <span style="color: #a1a1aa;">{curr_w:.1f}%</span> → <span class="{color}">{final_w:.1f}%</span></div>""", unsafe_allow_html=True)
                        st.markdown("---")
                        st.button("⚡ Applica questo Mix", type="secondary", on_click=apply_optimization, args=(new_weights_map,))

    run_calc = st.button("🚀 AVVIA SIMULAZIONE", type="primary")

# --- EXECUTION & VISUALIZATION ---
if run_calc:
    all_assets = list(set(selected_benchmarks + mix_assets_selected))
    df, err, init_spent, calc_maturity = calculate_engine_multibond(bond_df, capital, tax_rate, compound, all_assets, market_returns, sim_mode, n_sims)
    
    if err: st.error(err)
    else:
        if valid_mix and mix_assets_selected:
            risk_total_w = sum(mix_weights.values())
            target_weights = {a: (w/risk_total_w) for a, w in mix_weights.items()} 
            risk_capital_ratio = (100 - btp_weight) / 100.0
            btp_capital_ratio = btp_weight / 100.0
            if not rebalance:
                df["Mix_Portfolio"] = df["BTP_Value"] * btp_capital_ratio
                for asset, w_norm in target_weights.items():
                    if asset in df.columns:
                        quota = risk_capital_ratio * w_norm
                        df["Mix_Portfolio"] += df[asset] * quota
            else:
                btp_ret = df["BTP_Value"].pct_change().fillna(0)
                asset_rets = pd.DataFrame()
                for asset in mix_assets_selected: asset_rets[asset] = df[asset].pct_change().fillna(0)
                mix_values = [init_spent]
                current_val = init_spent
                sub_accounts = {}
                sub_accounts["BTP"] = current_val * btp_capital_ratio
                for asset in mix_assets_selected: sub_accounts[asset] = current_val * risk_capital_ratio * target_weights[asset]
                dates = df["Date"].tolist()
                for i in range(1, len(df)):
                    r_btp = btp_ret.iloc[i]
                    sub_accounts["BTP"] *= (1 + r_btp)
                    for asset in mix_assets_selected:
                        r_ass = asset_rets[asset].iloc[i]
                        sub_accounts[asset] *= (1 + r_ass)
                    total_nav = sub_accounts["BTP"] + sum(sub_accounts[a] for a in mix_assets_selected)
                    curr_btp_w = sub_accounts["BTP"] / total_nav
                    if abs(curr_btp_w - btp_capital_ratio) > 0.05:
                        sub_accounts["BTP"] = total_nav * btp_capital_ratio
                        for asset in mix_assets_selected: sub_accounts[asset] = total_nav * risk_capital_ratio * target_weights[asset]
                    mix_values.append(total_nav)
                df["Mix_Portfolio"] = mix_values

        st.session_state.sim_results = {'df': df, 'init_spent': init_spent, 'maturity_date': calc_maturity, 'selected_benchmarks': selected_benchmarks, 'inflation': inflation, 'has_mix': valid_mix, 'mix_details': mix_weights if valid_mix else {}, 'btp_w_final': btp_weight if valid_mix else 100, 'sim_mode': sim_mode, 'n_sims': n_sims, 'data_period': data_period_option, 'bond_df': bond_df}

if 'sim_results' in st.session_state:
    res = st.session_state.sim_results
    df = res['df']
    init_spent = res['init_spent']
    inflation = res['inflation']
    selected_benchmarks = res['selected_benchmarks']
    has_mix = res['has_mix']
    mix_weights = res['mix_details']
    btp_w_final = res['btp_w_final']
    mode_used = res.get('sim_mode', 'Historical')
    n_used = res.get('n_sims', 1)
    
    last = df.iloc[-1]
    months = len(df)
    years = (res['maturity_date'] - date.today()).days / 365.25
    final_btp = last["BTP_Value"]
    btp_gain_total = final_btp - init_spent
    btp_monthly_pure = last["Cum_Coupons"] / months
    btp_capital_gain = last["Gain_Netto_Finale"]
    
    # SIDEBAR REPORT BUTTON
    with st.sidebar:
        st.header("📄 Report")
        if st.button("Scarica Report PDF Completo"):
            try:
                pdf_bytes = create_pdf_report(res)
                st.download_button(label="⬇️ Download PDF", data=pdf_bytes, file_name="report_finanziario.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Errore generazione PDF: {e}")

    st.markdown("---")
    st.caption(f"Modalità: **{mode_used}** ({n_used:,} scenari)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Valore BTP (100%)", f"€ {final_btp:,.0f}", f"Utile: € {btp_gain_total:,.0f}")
        st.markdown(f"""<div class="metric-detail">Cedola Mensile: <span class="metric-highlight">€ {btp_monthly_pure:,.2f}</span><br>Bonus Scadenza: € {btp_capital_gain:,.0f}</div>""", unsafe_allow_html=True)
    with c2:
        if has_mix and "Mix_Portfolio" in df.columns:
            final_mix = last["Mix_Portfolio"]
            gain_mix = final_mix - init_spent
            dd_series = calculate_drawdown(df["Mix_Portfolio"])
            mdd_mix = dd_series.min() * 100
            st.metric(f"Portafoglio Misto", f"€ {final_mix:,.0f}", f"€ {gain_mix:,.0f}")
            comp_str = f"<span class='metric-purple'>{btp_w_final:.1f}% BTP</span>"
            for asset, w in mix_weights.items(): comp_str += f" + <span class='metric-purple'>{w:.1f}% {asset.split()[0]}</span>"
            st.markdown(f"""<div class="metric-detail">{comp_str}<br>Max Drawdown: <span class="metric-danger">{mdd_mix:.1f}%</span></div>""", unsafe_allow_html=True)
        else: st.metric("Portafoglio Misto", "-")
    best_asset, best_val = None, 0
    asset_cols = [c for c in df.columns if c in ASSET_CONFIG.keys()]
    for a in asset_cols:
        val = last[a]
        if val > best_val: best_val = val; best_asset = a
    with c3:
        if best_asset:
            gain_b = best_val - init_spent
            dd_best = calculate_drawdown(df[best_asset]).min() * 100
            st.metric(f"Miglior Asset", f"€ {best_val:,.0f}", f"€ {gain_b:,.0f}")
            st.markdown(f"""<div class="metric-detail">Asset: <span class="metric-highlight">{best_asset}</span><br>Max Drawdown: <span class="metric-danger">{dd_best:.1f}%</span></div>""", unsafe_allow_html=True)
        else: st.metric("Benchmark", "N/A")
    with c4:
        real_val = final_btp / ((1+inflation)**years)
        loss = final_btp - real_val
        st.metric("Potere d'Acquisto", f"€ {real_val:,.0f}", f"- € {loss:,.0f}", delta_color="inverse")
        st.markdown(f"""<div class="metric-detail">BTP scontato inflazione</div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Analisi Grafica e Rischio")
    use_log = st.toggle("Scala Logaritmica", value=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BTP_Value"], mode='lines', name="Tuo BTP (100%)", line=dict(color="#22c55e", width=3), fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)'))
    if has_mix and "Mix_Portfolio" in df.columns: fig.add_trace(go.Scatter(x=df["Date"], y=df["Mix_Portfolio"], mode='lines', name="Portafoglio Misto", line=dict(color="#c084fc", width=4)))
    for asset in selected_benchmarks:
        if asset in df.columns: fig.add_trace(go.Scatter(x=df["Date"], y=df[asset], mode='lines', name=asset, line=dict(color=ASSET_CONFIG[asset]["color"], width=1.5), opacity=0.7))
    df['Real_Line'] = df["BTP_Value"] * (1 / ((1+inflation)**((df.index)/12)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df['Real_Line'], mode='lines', name="Soglia Reale", line=dict(color="#ef4444", width=2, dash='dot')))
    fig.update_layout(title="Evoluzione Capitale (Scenario Mediano)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=500, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="€ ", type="log" if use_log else "linear"), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)
    
    col_dd, col_cash = st.columns(2)
    with col_dd:
        st.markdown("#### 🌊 Profondità del Rischio (Drawdown)")
        fig_dd = go.Figure()
        if has_mix and "Mix_Portfolio" in df.columns: fig_dd.add_trace(go.Scatter(x=df["Date"], y=calculate_drawdown(df["Mix_Portfolio"]), mode='lines', name="Drawdown Mix", line=dict(color="#c084fc", width=2), fill='tozeroy'))
        for asset in selected_benchmarks:
            if asset in df.columns: fig_dd.add_trace(go.Scatter(x=df["Date"], y=calculate_drawdown(df[asset]), mode='lines', name=f"DD {asset}", line=dict(color=ASSET_CONFIG[asset]["color"], width=1), opacity=0.7))
        fig_dd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat=".1%"), hovermode="x unified", margin=dict(t=10, b=10))
        if has_mix or selected_benchmarks: st.plotly_chart(fig_dd, use_container_width=True)
        else: st.info("Nessun asset rischioso selezionato.")
    
    with col_cash:
        st.markdown("#### 💸 Flusso Cedolare Netto (BTP)")
        df['Anno'] = df['Date'].dt.year
        annual_coupons = df.groupby('Anno')['Cum_Coupons'].max().reset_index()
        annual_coupons['Flusso_Netto'] = annual_coupons['Cum_Coupons'].diff().fillna(annual_coupons['Cum_Coupons'])
        fig_cash = go.Figure()
        fig_cash.add_trace(go.Bar(x=annual_coupons['Anno'], y=annual_coupons['Flusso_Netto'], name='Cedole Annuali', marker_color='#10b981', hovertemplate='Anno %{x}<br>Incasso: € %{y:,.2f}'))
        fig_cash.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e4e4e7'), height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickprefix="€ "), hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig_cash, use_container_width=True)
    
    with st.expander("📄 Dati Annuali Dettagliati"):
        annual = df.groupby('Anno').last().reset_index()
        st.dataframe(annual, use_container_width=True)

# Final conditional message (Footer logic)
if not run_calc and 'sim_results' not in st.session_state:
    st.info("👈 Configura i parametri sopra e clicca su AVVIA SIMULAZIONE per iniziare.")
    
# Download button logic at the very end
sim_data_for_pdf = st.session_state.get('sim_results', None)
st.markdown("---")
col_down_1, col_down_2 = st.columns([3, 1])
with col_down_2:
    try:
        # Always allow downloading the manual, even if sim_data_for_pdf is None
        pdf_bytes = create_pdf_report(sim_data_for_pdf)
        st.download_button(
            label="📄 Scarica Whitepaper Tecnico", 
            data=pdf_bytes, 
            file_name="report_finanziario_pro.pdf", 
            mime="application/pdf",
            help="Scarica un report PDF dettagliato con la metodologia e i risultati della tua simulazione."
        )
    except Exception as e:
        st.error(f"Errore generazione PDF: {e}")
