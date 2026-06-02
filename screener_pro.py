"""
backtest_bourse.py — Validation Oracle ML v15 vs Buy & Hold
===========================================================
Teste si la stratégie Random Forest + Money Management surperforme
un simple investissement Buy & Hold sur 4 ans, sans biais d'anticipation.

Nouveautés v15 :
- Univers stratégique Luxe & Santé
- Money Management (1% de risque par trade basé sur l'ATR)
- Stop Suiveur dynamique (Trailing Stop)
- Filtre de Régime (Ignore les ventes surachetées en Bull Market S&P 500)

Usage :
  python backtest_bourse.py                 → tous les actifs
  python backtest_bourse.py --ticker MC.PA      → actif unique
  python backtest_bourse.py --periode 2y        → période personnalisée
  python backtest_bourse.py --vider-cache       → force le retéléchargement
"""

import sys
import os
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ==============================================================================
# CONFIGURATION ET MONEY MANAGEMENT (Alignés sur screener_pro.py v15)
# ==============================================================================

# SELECTION STRATÉGIQUE LUXE & SANTÉ
TICKERS_BACKTEST = {
    "MC.PA":    "LVMH",
    "KER.PA":   "Kering",
    "RMS.PA":   "Hermès",
    "OR.PA":    "L'Oréal",
    "EL.PA":    "EssilorLuxottica",
    "SAN.PA":   "Sanofi"
}

PERIODE         = "4y"   
MIN_TRAIN_JOURS = 252    
RETRAIN_FREQ    = 20     
CAPITAL_INIT    = 10_000.0

# Paramètres de gestion du risque
RISQUE_PAR_TRADE = 0.01  # 1% du capital maximum à perdre par transaction

# Seuils de stratégie
SEUIL_ACHAT    = 6    
SEUIL_VENTE    = 4    
RSI_SURACHAT   = 70   

# Frais & fiscalité
FRAIS_TRANSACTION   = 0.0015   
TAUX_PRELEV_SOCIAUX = 0.172  
TAUX_FLAT_TAX       = 0.30   

JOURS_BOURSE_AN  = 252        
TAUX_SANS_RISQUE = 0.03      

# ETF de référence
ETF_REFERENCE = "CW8.PA"
ETF_REFERENCE_NOM = "ETF MSCI World (CW8)"

CACHE_DIR = "cache"
CACHE_VALIDITE_HEURES = 12   
DELAI_ENTRE_REQUETES  = 2.0  
MAX_REESSAIS          = 3    

FEATURES = ["RSI", "MACD_Hist", "ATR", "Price_52W_Pct", "SMA50_vs_SMA200", "BB_Width", "Var_1M"]

# ==============================================================================
# CACHE LOCAL DES DONNÉES
# ==============================================================================
def _chemin_cache(symbol: str) -> str:
    nom = symbol.replace(".", "_").replace("^", "INDEX_")
    return os.path.join(CACHE_DIR, f"{nom}.csv")

def _cache_valide(chemin: str) -> bool:
    if not os.path.exists(chemin): return False
    age_heures = (time.time() - os.path.getmtime(chemin)) / 3600
    return age_heures < CACHE_VALIDITE_HEURES

def telecharger_avec_cache(symbol: str, periode: str = "5y", start=None, end=None) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    chemin = _chemin_cache(symbol)

    def _filtrer(df):
        if start is None or end is None or df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)
        tz = getattr(df.index, "tz", None)
        if tz is not None:
            if ts_start.tz is None: ts_start = ts_start.tz_localize(tz)
            if ts_end.tz   is None: ts_end   = ts_end.tz_localize(tz)
        else:
            if ts_start.tz is not None: ts_start = ts_start.tz_localize(None)
            if ts_end.tz   is not None: ts_end   = ts_end.tz_localize(None)
        return df.loc[(df.index >= ts_start) & (df.index <= ts_end)]

    if _cache_valide(chemin):
        try:
            df = pd.read_csv(chemin, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[df.index.notna()]
            if not df.empty:
                return _filtrer(df)
        except Exception: pass

    for tentative in range(1, MAX_REESSAIS + 1):
        try:
            time.sleep(DELAI_ENTRE_REQUETES)
            df = yf.Ticker(symbol).history(period=periode)
            if not df.empty:
                df.to_csv(chemin)
                return _filtrer(df)
        except Exception: pass
        if tentative < MAX_REESSAIS:
            time.sleep(DELAI_ENTRE_REQUETES * (tentative + 1) * 2)

    return pd.DataFrame()

# ==============================================================================
# CALCUL DES INDICATEURS ET CONVICTION
# ==============================================================================
def calculer_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = exp1 - exp2
    df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    df["MA20"]     = df["Close"].rolling(20).mean()
    df["STD20"]    = df["Close"].rolling(20).std()
    df["BB_High"]  = df["MA20"] + df["STD20"] * 2
    df["BB_Low"]   = df["MA20"] - df["STD20"] * 2
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["MA20"] * 100

    df["SMA50"]           = df["Close"].rolling(50).mean()
    df["SMA200"]          = df["Close"].rolling(200).mean()
    df["SMA50_vs_SMA200"] = (df["SMA50"] - df["SMA200"]) / df["SMA200"] * 100

    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["H52"] = df["High"].rolling(252).max()
    df["L52"] = df["Low"].rolling(252).min()
    r52       = (df["H52"] - df["L52"]).clip(lower=1e-9)
    df["Price_52W_Pct"] = (df["Close"] - df["L52"]) / r52 * 100
    df["Var_1M"]        = df["Close"].pct_change(20) * 100

    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["OBV"]     = (df["Close"].diff().apply(np.sign) * df["Volume"]).fillna(0).cumsum()
        df["OBV_SMA"] = df["OBV"].rolling(20).mean()
        df["OBV_Bull"] = (df["OBV"] > df["OBV_SMA"]).astype(int)
    else:
        df["OBV_Bull"] = 0

    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    return df

def conviction_vectorisee(rsi, macd_bull, sma_bull, prob_up, obv_bull) -> np.ndarray:
    s  = np.zeros(len(rsi))
    s += np.where(rsi <= 35, 3, np.where(rsi <= 45, 2, np.where(rsi <= 55, 1, 0)))
    s += np.where(macd_bull, 2, 0)
    s += np.where(sma_bull, 2, 0)
    s += np.where(prob_up > 55, 2, np.where(prob_up >= 45, 1, 0))
    s += np.where(obv_bull, 1, 0)
    return s

# ==============================================================================
# WALK-FORWARD ML ET CONTEXTE MACROÉCONOMIQUE
# ==============================================================================
def walk_forward_signaux(df_clean: pd.DataFrame) -> pd.DataFrame:
    n       = len(df_clean)
    probs   = np.full(n, np.nan)
    model   = None

    for i in range(MIN_TRAIN_JOURS, n):
        if (i - MIN_TRAIN_JOURS) % RETRAIN_FREQ == 0:
            fin_train = max(0, i - 5)
            X_tr = df_clean[FEATURES].iloc[:fin_train]
            y_tr = df_clean["Target"].iloc[:fin_train]
            if y_tr.nunique() >= 2 and len(y_tr) > 50:
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                model.fit(X_tr, y_tr)

        if model is None: continue
        classes = list(model.classes_)
        if 1 not in classes: continue

        X_j       = df_clean[FEATURES].iloc[i:i+1].fillna(0)
        proba     = model.predict_proba(X_j)[0]
        probs[i]  = proba[classes.index(1)] * 100

    df_out = df_clean.copy()
    df_out["prob_up"] = probs
    df_out = df_out.dropna(subset=["prob_up"])

    df_out["conviction"] = conviction_vectorisee(
        rsi       = df_out["RSI"].values,
        macd_bull = (df_out["MACD"] > df_out["MACD_Sig"]).values,
        sma_bull  = (df_out["Close"] > df_out["SMA200"]).values,
        prob_up   = df_out["prob_up"].values,
        obv_bull  = df_out.get("OBV_Bull", pd.Series(0, index=df_out.index)).values,
    )
    
    # Intégration de la macro (Régime S&P 500) pour toute la période
    df_sp500 = telecharger_avec_cache("^GSPC", periode="5y")
    if not df_sp500.empty:
        df_sp500["SP500_SMA200"] = df_sp500["Close"].rolling(200).mean()
        df_sp500["SP500_Bull"] = df_sp500["Close"] > df_sp500["SP500_SMA200"]
        df_out = df_out.join(df_sp500[["SP500_Bull"]], how="left")
        df_out["SP500_Bull"] = df_out["SP500_Bull"].ffill().fillna(False)
    else:
        df_out["SP500_Bull"] = False

    return df_out[["Close", "RSI", "ATR", "prob_up", "conviction", "SP500_Bull"]].copy()

# ==============================================================================
# SIMULATION DE PORTEFEUILLE V15 (Money Management + Trailing Stop)
# ==============================================================================
def metriques_avancees(courbe: pd.Series, trades: list = None) -> dict:
    rendements = courbe.pct_change().dropna()
    if len(rendements) < 2 or rendements.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "volatilite": 0.0, "duree_moy": 0.0}

    rdt_moy_annuel = rendements.mean() * JOURS_BOURSE_AN
    vol_annuelle   = rendements.std() * np.sqrt(JOURS_BOURSE_AN)
    sharpe = (rdt_moy_annuel - TAUX_SANS_RISQUE) / vol_annuelle if vol_annuelle > 0 else 0.0
    rdts_neg = rendements[rendements < 0]
    downside_vol = rdts_neg.std() * np.sqrt(JOURS_BOURSE_AN) if len(rdts_neg) > 1 else 0.0
    sortino = (rdt_moy_annuel - TAUX_SANS_RISQUE) / downside_vol if downside_vol > 0 else 0.0

    duree_moy = 0.0
    if trades:
        durees = [(t[2] - t[0]).days for t in trades if t[2] and t[0]]
        duree_moy = np.mean(durees) if durees else 0.0

    return {"sharpe": sharpe, "sortino": sortino, "volatilite": vol_annuelle * 100, "duree_moy": duree_moy}

def appliquer_fiscalite(capital_final: float, capital_initial: float, regime: str = "PEA") -> dict:
    plus_value = capital_final - capital_initial
    if plus_value <= 0: return {"net": capital_final, "impot": 0.0, "regime": regime}
    taux = TAUX_PRELEV_SOCIAUX if regime == "PEA" else TAUX_FLAT_TAX
    impot = plus_value * taux
    return {"net": capital_final - impot, "impot": impot, "regime": regime}

def simuler_oracle(df_sig: pd.DataFrame) -> dict:
    capital      = CAPITAL_INIT
    cash         = capital
    en_position  = False
    nb_actions   = 0
    prix_entree  = 0.0
    stop_suiveur = 0.0
    date_entree  = None
    trades       = []
    courbe       = []
    frais_totaux = 0.0

    for date, row in df_sig.iterrows():
        prix = row["Close"]
        conv = row["conviction"]
        rsi  = row["RSI"]
        atr  = row["ATR"]
        is_bull = row["SP500_Bull"]

        if not en_position:
            if conv >= SEUIL_ACHAT:
                stop_initial = prix - 1.5 * atr
                distance_stop = prix - stop_initial
                
                if distance_stop > 0:
                    risque_eur = capital * RISQUE_PAR_TRADE
                    nb_act_calc = int(risque_eur / distance_stop)
                    
                    if nb_act_calc > 0:
                        montant_brut = nb_act_calc * prix
                        frais = montant_brut * FRAIS_TRANSACTION
                        
                        # Achat partiel du portefeuille (Liquidités conservées)
                        if (montant_brut + frais) <= cash:
                            nb_actions = nb_act_calc
                            cash -= (montant_brut + frais)
                            frais_totaux += frais
                            prix_entree = prix
                            stop_suiveur = stop_initial
                            date_entree = date
                            en_position = True
        else:
            # 1. Mise à jour du Stop Suiveur dynamique
            nouveau_stop = prix - 1.5 * atr
            if nouveau_stop > stop_suiveur:
                stop_suiveur = nouveau_stop
                
            # 2. Conditions de vente
            touche_stop = prix <= stop_suiveur
            baisse_conv = conv < SEUIL_VENTE
            surachat = (rsi > RSI_SURACHAT)
            
            # Filtre de Régime : Bloquer la vente surachetée si le SP500 est haussier
            vendre_surachat = surachat and not (is_bull and conv >= 5)

            if touche_stop or baisse_conv or vendre_surachat:
                montant_brut = nb_actions * prix
                frais = montant_brut * FRAIS_TRANSACTION
                frais_totaux += frais
                cash += (montant_brut - frais)
                gain_pct = (prix - prix_entree) / prix_entree * 100
                motif = "Stop" if touche_stop else ("Surachat" if vendre_surachat else "Signal")
                trades.append((date_entree, prix_entree, date, prix, gain_pct, motif))
                nb_actions = 0
                en_position = False

        valeur_pf = cash + (nb_actions * prix if en_position else 0)
        capital = valeur_pf
        courbe.append(valeur_pf)

    if en_position:
        prix_fin = df_sig["Close"].iloc[-1]
        brut = nb_actions * prix_fin
        frais = brut * FRAIS_TRANSACTION
        frais_totaux += frais
        cash += (brut - frais)
        trades.append((date_entree, prix_entree, df_sig.index[-1], prix_fin, (prix_fin - prix_entree) / prix_entree * 100, "Clôture Test"))
        courbe[-1] = cash

    courbe_s = pd.Series(courbe, index=df_sig.index)
    peak     = courbe_s.cummax()
    dd_max   = ((courbe_s - peak) / peak * 100).min()
    nb_win   = sum(1 for t in trades if t[4] > 0)
    metr     = metriques_avancees(courbe_s, trades)

    return {
        "capital_final":    courbe_s.iloc[-1],
        "rendement_total":  (courbe_s.iloc[-1] / CAPITAL_INIT - 1) * 100,
        "drawdown_max":     dd_max,
        "win_rate":         (nb_win / len(trades) * 100) if trades else 0.0,
        "nb_trades":        len(trades),
        "frais_totaux":     frais_totaux,
        "sharpe":           metr["sharpe"],
        "sortino":          metr["sortino"],
        "volatilite":       metr["volatilite"],
        "duree_moy":        metr["duree_moy"],
        "courbe":           courbe_s,
        "trades":           trades,
    }

def simuler_buy_hold(df_sig: pd.DataFrame) -> dict:
    p0 = df_sig["Close"].iloc[0]
    pn = df_sig["Close"].iloc[-1]
    cap_apres_achat = CAPITAL_INIT * (1 - FRAIS_TRANSACTION)
    nb_actions = cap_apres_achat / p0
    cap_f = (nb_actions * pn) * (1 - FRAIS_TRANSACTION)
    courbe = nb_actions * df_sig["Close"]
    peak = courbe.cummax()
    dd_max = ((courbe - peak) / peak * 100).min()
    metr = metriques_avancees(courbe)
    return {
        "capital_final":   cap_f,
        "rendement_total": (cap_f / CAPITAL_INIT - 1) * 100,
        "drawdown_max":    dd_max,
        "sharpe":          metr["sharpe"],
        "sortino":         metr["sortino"],
        "volatilite":      metr["volatilite"],
        "courbe":          courbe,
    }

def simuler_etf_reference(date_debut, date_fin) -> dict:
    df = telecharger_avec_cache(ETF_REFERENCE, periode="5y", start=date_debut, end=date_fin)
    if df.empty or len(df) < 30: return None
    p0, pn = df["Close"].iloc[0], df["Close"].iloc[-1]
    cap_apres_achat = CAPITAL_INIT * (1 - FRAIS_TRANSACTION)
    nb = cap_apres_achat / p0
    cap_f = (nb * pn) * (1 - FRAIS_TRANSACTION)
    courbe = nb * df["Close"]
    peak = courbe.cummax()
    metr = metriques_avancees(courbe)
    return {
        "capital_final":   cap_f,
        "rendement_total": (cap_f / CAPITAL_INIT - 1) * 100,
        "drawdown_max":    ((courbe - peak) / peak * 100).min(),
        "sharpe":          metr["sharpe"],
        "sortino":         metr["sortino"],
    }

# ==============================================================================
# AFFICHAGE ET EXÉCUTION
# ==============================================================================
def afficher_tableau(nom, res_ml, res_bh, date_debut, date_fin, res_etf=None):
    annees = (date_fin - date_debut).days / 365.25
    rend_an_ml = ((1 + res_ml["rendement_total"] / 100) ** (1 / annees) - 1) * 100
    rend_an_bh = ((1 + res_bh["rendement_total"] / 100) ** (1 / annees) - 1) * 100
    fisc_ml = appliquer_fiscalite(res_ml["capital_final"], CAPITAL_INIT, "PEA")
    fisc_bh = appliquer_fiscalite(res_bh["capital_final"], CAPITAL_INIT, "PEA")

    L = 70
    print(f"\n{'═' * L}")
    print(f"  {nom}")
    print(f"  {date_debut.date()} → {date_fin.date()}  ({annees:.1f} ans)")
    print(f"{'═' * L}")
    print(f"  {'Métrique':<34} {'Oracle v15':>14} {'Buy & Hold':>14}")
    print(f"  {'─' * 64}")
    print(f"  {'Rendement total (brut)':<34} {res_ml['rendement_total']:>+13.1f}% {res_bh['rendement_total']:>+13.1f}%")
    print(f"  {'Rendement annualisé':<34} {rend_an_ml:>+13.1f}% {rend_an_bh:>+13.1f}%")
    print(f"  {'Drawdown maximum':<34} {res_ml['drawdown_max']:>+13.1f}% {res_bh['drawdown_max']:>+13.1f}%")
    print(f"  {'Ratio de Sharpe':<34} {res_ml.get('sharpe',0):>14.2f} {res_bh.get('sharpe',0):>14.2f}")
    print(f"  {'Ratio de Sortino':<34} {res_ml.get('sortino',0):>14.2f} {res_bh.get('sortino',0):>14.2f}")
    print(f"  {'─' * 64}")
    print(f"  {'Win Rate':<34} {res_ml['win_rate']:>13.1f}%   {'—':>13}")
    print(f"  {'Nombre de trades':<34} {res_ml['nb_trades']:>14}   {'—':>13}")
    print(f"  {'Durée moy. détention (jours)':<34} {res_ml.get('duree_moy',0):>14.0f}   {'—':>13}")
    print(f"  {'Frais de transaction cumulés':<34} {res_ml.get('frais_totaux',0):>12.0f} €   {'~30 €':>13}")
    print(f"  {'─' * 64}")
    print(f"  Après fiscalité PEA :")
    print(f"  {'Capital net final':<34} {fisc_ml['net']:>12.0f} €   {fisc_bh['net']:>11.0f} €")

    if res_etf:
        print(f"  {'─' * 64}")
        print(f"  Référence passive — {ETF_REFERENCE_NOM} :")
        print(f"  {'Rendement total':<34} {res_etf['rendement_total']:>+13.1f}%")

    ecart = res_ml["rendement_total"] - res_bh["rendement_total"]
    verdict = f"✅ La stratégie surperforme de {ecart:+.1f}%" if ecart > 5 else (f"⚠️ Buy & Hold surperforme de {-ecart:.1f}%" if ecart < -5 else f"⚪ Performances similaires (écart {ecart:+.1f}%)")
    print(f"\n  {verdict}")
    print(f"{'═' * L}")

    if res_ml["trades"]:
        print(f"\n  Trades de la stratégie ({res_ml['nb_trades']}) :")
        for t in res_ml["trades"]:
            signe = "✅" if t[4] > 0 else "❌"
            print(f"    {signe}  {str(t[0].date()):>12} → {str(t[2].date()):>12}  {t[1]:>8.2f} → {t[3]:>8.2f}  ({t[4]:>+6.1f}%)  [{t[5]}]")

def tracer_courbes(nom, res_ml, res_bh):
    if not MPL_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(12, 5))
    res_bh["courbe"].plot(ax=ax, label="Buy & Hold", color="#888", linestyle="--")
    res_ml["courbe"].plot(ax=ax, label="Oracle v15", color="#2196F3", linewidth=1.5)
    ax.set_title(f"Oracle v15 (Money Management) vs Buy & Hold — {nom}")
    ax.set_ylabel("Valeur du portefeuille (€)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"backtest_{nom.replace(' ', '_').replace('/', '_')}.png", dpi=120)
    plt.close()

def backtester_actif(symbol: str, nom: str, periode: str = PERIODE):
    print(f"\n⏳  {nom} ({symbol}) — téléchargement {periode}…")
    df_raw = telecharger_avec_cache(symbol, periode=periode)
    if df_raw.empty or len(df_raw) < MIN_TRAIN_JOURS + 100:
        print(f"  ❌ Données insuffisantes")
        return

    df = calculer_indicateurs(df_raw)
    df_clean = df.dropna(subset=FEATURES + ["Target"]).copy()
    
    print(f"  Calcul du Walk-Forward et Intégration S&P 500…")
    df_signaux = walk_forward_signaux(df_clean)

    if len(df_signaux) < 50:
        print(f"  ❌ Trop peu de signaux")
        return

    res_ml = simuler_oracle(df_signaux)
    res_bh = simuler_buy_hold(df_signaux)
    res_etf = simuler_etf_reference(df_signaux.index[0].to_pydatetime(), df_signaux.index[-1].to_pydatetime()) if symbol != ETF_REFERENCE else None

    afficher_tableau(f"{nom} ({symbol})", res_ml, res_bh, df_signaux.index[0].to_pydatetime(), df_signaux.index[-1].to_pydatetime(), res_etf)
    tracer_courbes(nom, res_ml, res_bh)

if __name__ == "__main__":
    periode = PERIODE
    if "--periode" in sys.argv:
        i = sys.argv.index("--periode")
        periode = sys.argv[i + 1]

    if "--vider-cache" in sys.argv:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print(f"🗑️  Cache vidé.")

    print("═" * 70)
    print("  Backtest v15 — Money Management, Trailing Stop, Filtre de Régime")
    print(f"  Risque par trade : {RISQUE_PAR_TRADE*100}% du Capital ({CAPITAL_INIT} €)")
    print("═" * 70)

    if "--ticker" in sys.argv:
        i = sys.argv.index("--ticker")
        sym = sys.argv[i + 1]
        backtester_actif(sym, TICKERS_BACKTEST.get(sym, sym), periode)
    else:
        for sym, nom in TICKERS_BACKTEST.items():
            backtester_actif(sym, nom, periode)
