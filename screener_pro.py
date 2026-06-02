"""
Systeme d'analyse multi-couches de la bourse
=========================================================
Outil d'AIDE A LA DECISION et de VEILLE. Il produit un rapport quotidien
combinant analyse technique, intelligence artificielle, fondamentale,
macroeconomique ET Money Management.

AVERTISSEMENT IMPORTANT :
Ce systeme N'EXECUTE AUCUN ORDRE de bourse. Il ne passe ni achat ni vente.
Il fournit de l'INFORMATION et des SIGNAUX D'ANALYSE, jamais des instructions
a executer automatiquement. Toute decision d'investissement releve de
l'utilisateur seul.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ==============================================================================
# CONFIGURATION ET MONEY MANAGEMENT
# ==============================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "VOTRE_TOKEN_DE_SECOURS")
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID", "VOTRE_CHAT_ID_DE_SECOURS")
HISTORY_FILE   = Path("predictions_history.json")

# Paramètres de gestion du risque (Money Management)
CAPITAL_TOTAL    = 10000.0  # Votre capital total alloué à la stratégie (en euros)
RISQUE_PAR_TRADE = 0.01     # Risque maximum accepté par transaction (1% = 100€)

# UNIVERS DE SURVEILLANCE — SÉLECTION STRATÉGIQUE LUXE & SANTÉ
TICKERS = {
    "MC.PA":    "LVMH",
    "KER.PA":   "Kering",
    "RMS.PA":   "Hermès",
    "OR.PA":    "L'Oréal",
    "EL.PA":    "EssilorLuxottica",
    "SAN.PA":   "Sanofi"
}

portefeuille = {
    "MC.PA": {"nom": "LVMH", "prix_achat": 458.45, "quantite": 1}
}

# ==============================================================================
# FORMATAGE MARKDOWNV2
# ==============================================================================
_MD2_SPECIAL = set(r"_*[]()~`>#+-=|{}.!")

def esc(text):
    return "".join(f"\\{c}" if c in _MD2_SPECIAL else c for c in str(text))

def bold(text):
    return f"*{esc(text)}*"

# ==============================================================================
# ENVOI TELEGRAM
# ==============================================================================
def envoyer_telegram(texte, chat_id, token):
    LIMIT = 4000
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    blocs = []
    while len(texte) > LIMIT:
        coupe = texte.rfind("\n", 0, LIMIT)
        if coupe == -1: coupe = LIMIT
        blocs.append(texte[:coupe])
        texte = texte[coupe:].lstrip("\n")
    blocs.append(texte)

    for i, bloc in enumerate(blocs, 1):
        payload = {"chat_id": chat_id, "text": bloc, "parse_mode": "MarkdownV2"}
        r = requests.post(url, json=payload)
        if not r.ok:
            print(f"Échec message {i}/{len(blocs)} ({r.status_code}) : {r.text}")
        else:
            print(f"✅ Message {i}/{len(blocs)} envoyé")

# ==============================================================================
# CONTEXTE MACROÉCONOMIQUE ET FILTRE DE RÉGIME
# ==============================================================================
def get_macro_context():
    """
    Récupère le VIX, le taux à 10 ans US, ET la tendance du S&P 500.
    Le S&P 500 dicte le "Régime" (Bull ou Bear market) pour bloquer les ventes hâtives.
    """
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        
        # Filtre de régime (S&P 500 vs sa moyenne mobile 200 jours)
        sp500 = yf.Ticker("^GSPC").history(period="200d")["Close"]
        is_bull_market = sp500.iloc[-1] > sp500.mean()
        
        if   vix < 15:  regime, modifier = "🟢 CALME",    0
        elif vix < 22:  regime, modifier = "🟡 VIGILANCE", 0
        elif vix < 30:  regime, modifier = "🟠 TENSION",  -1
        else:           regime, modifier = "🔴 PANIQUE",  -2
        
        return {"vix": round(vix, 1), "tnx": round(tnx, 2),
                "regime": regime, "modifier": modifier, "is_bull": is_bull_market}
    except Exception as e:
        print(f"Avertissement macro : {e}")
        return {"vix": None, "tnx": None,
                "regime": "⚪ NEUTRE", "modifier": 0, "is_bull": False}

def fundamental_adjustment(per, margin, macro_tnx):
    adj = 0
    if per is not None and isinstance(per, (int, float)) and per > 0:
        if   per < 12:                 adj += 1
        elif per > 30:                 adj -= 1
        if macro_tnx and macro_tnx > 4.2 and per > 25:
            adj -= 1
    if margin is not None and isinstance(margin, (int, float)):
        if   margin > 0.15:            adj += 1
        elif margin < 0.03:            adj -= 1
    return max(-2, min(2, adj))

def conviction_score(data, macro_modifier=0, macro_tnx=None):
    score = 0
    rsi = data["rsi"]
    if   rsi <= 35: score += 3
    elif rsi <= 45: score += 2
    elif rsi <= 55: score += 1
    if data["macd_trend"] == "🍏": score += 2
    if data["sma_trend"]  == "↗️": score += 2
    if data.get("ml_ok"):
        if   data["prob_up"] > 55: score += 2
        elif data["prob_up"] >= 45: score += 1
    else:
        score += 1
    if data["vol_trend"] == "📈": score += 1

    fonda_adj = fundamental_adjustment(data.get("per"), data.get("margin"), macro_tnx)
    score += fonda_adj
    score += macro_modifier
    return max(0, min(10, score))

def score_emoji(score):
    if   score >= 8: return "🟢🟢"
    elif score >= 6: return "🟢"
    elif score >= 4: return "⚪"
    elif score >= 2: return "🟡"
    else:            return "🔴"

# ==============================================================================
# CALCUL DE TAILLE DE POSITION (MONEY MANAGEMENT)
# ==============================================================================
def calculer_position(prix, stop):
    """Calcule le nombre d'actions à acheter pour ne pas perdre plus que RISQUE_PAR_TRADE"""
    risque_max_euros = CAPITAL_TOTAL * RISQUE_PAR_TRADE
    distance_stop = prix - stop
    
    if distance_stop <= 0:
        return 0, 0, risque_max_euros
        
    nb_actions = int(risque_max_euros / distance_stop)
    montant_total = nb_actions * prix
    return nb_actions, montant_total, risque_max_euros

# ==============================================================================
# INTERPRÉTATION & VERDICT (AVEC FILTRE DE RÉGIME ET TRAILING STOP)
# ==============================================================================
def interpreter_ia(data):
    if not data.get("ml_ok"): return None
    p = data["prob_up"]
    if   p >= 70: return f"IA favorable ({p:.0f}%)"
    elif p >= 55: return f"IA favorable ({p:.0f}%)"
    elif p <= 30: return f"IA défavorable ({p:.0f}%)"
    elif p <= 45: return f"IA défavorable ({p:.0f}%)"
    return None

def verdict_action(data, conv, is_bull_market=False):
    rsi  = data["rsi"]
    prix = data["price"]
    atr  = data["atr"]
    
    entree   = round(prix - 0.5 * atr, 2)
    objectif = round(prix + 2.5 * atr, 2)
    
    # Trailing Stop = Prix actuel - 1.5 ATR (S'adapte tous les jours à la volatilité)
    stop = round(prix - 1.5 * atr, 2)
    
    # Calcul Money Management
    nb_actions, montant_investi, risque_eur = calculer_position(prix, stop)
    
    mm_str = f"   🛡️ Taille max : {nb_actions} parts ({montant_investi:.0f} €) · Risque : {risque_eur:.0f} €"

    # Filtre de Régime : On ne vend pas un surachat si le marché global est très haussier
    if rsi >= 70 and is_bull_market and conv >= 5:
        return (f"💎 LAISSER COURIR — Surachat ignoré (Marché Haussier)\n"
                f"   Le marché pousse, on conserve.\n"
                f"   Stop suiveur (à remonter demain) : {stop:.2f} €")

    # Logique classique
    if rsi <= 35 and conv >= 7:
        return (f"🟢 ACHETER — signal fort\n"
                f"   J'entre à {prix:.2f} € ou repli vers {entree:.2f} €\n"
                f"{mm_str}\n"
                f"   Je vise {objectif:.2f} €  ·  Stop initial sous {stop:.2f} €")
    if rsi <= 35 and conv >= 5:
        return (f"🟢 ACHETER — signal modéré\n"
                f"   J'entre si le prix tient au-dessus de {stop:.2f} €\n"
                f"{mm_str}\n"
                f"   Je vise {objectif:.2f} €  ·  Stop initial sous {stop:.2f} €")
    if rsi <= 35:
        return (f"👁️ SURVEILLER — survente, signaux mixtes\n"
                f"   J'attends stabilisation. Je reviens si conviction > 5/10.")
    if rsi <= 45 and conv >= 7:
        return (f"🔵 RENFORCER sur repli\n"
                f"   J'attends un retour vers {entree:.2f} €\n"
                f"   Stop suiveur : {stop:.2f} €")
    if rsi <= 45 and conv >= 5:
        return (f"💎 CONSERVER — profil intéressant\n"
                f"   Je renforce si repli vers {entree:.2f} €\n"
                f"   Stop suiveur actuel : {stop:.2f} €")
    if rsi >= 70 and conv <= 3:
        return (f"🔴 VENDRE — convergence baissière\n"
                f"   Je vends maintenant. Je reviens si RSI < 50.")
    if rsi >= 70 and conv <= 5:
        return (f"🟠 ALLÉGER — surachat confirmé\n"
                f"   Je réduis 30 à 50 % de ma position.\n"
                f"   Stop très serré sous {stop:.2f} €")
    if rsi >= 70:
        return (f"🟡 ATTENTION — surachat, tendance forte\n"
                f"   Je ne renforce pas. Remonter le stop suiveur à {stop:.2f} €")
    if rsi >= 65:
        return (f"🟡 ATTENDRE — zone tendue\n"
                f"   Je n'achète pas. Je reviens si RSI < 60.")
    if conv >= 6:
        return (f"💎 CONSERVER — profil favorable\n"
                f"   Stop suiveur de protection : {stop:.2f} €")
    if conv >= 4:
        return f"💎 CONSERVER — Stop suiveur : {stop:.2f} €"
        
    return (f"🟡 PRUDENCE — signaux faibles\n"
            f"   Stop suiveur rapproché à {stop:.2f} €")

def generer_resume_executif(data_actifs, macro_mod=0, macro_tnx=None):
    sorted_items = sorted(data_actifs.items(),
                          key=lambda x: conviction_score(x[1], macro_mod, macro_tnx), reverse=True)
    lines = []
    achats   = [(s,d) for s,d in sorted_items if d["rsi"] <= 38]
    ventes   = [(s,d) for s,d in sorted_items if d["rsi"] >= 70]
    top_conv = [(s,d) for s,d in sorted_items
                if conviction_score(d, macro_mod, macro_tnx) >= 7 and d["rsi"] < 70]
    if achats:
        noms = " et ".join(d["name"] for _,d in achats[:2])
        lines.append(f"⭐ {noms} en zone de survente")
    if top_conv:
        noms = ", ".join(d["name"] for _,d in top_conv[:3])
        lines.append(f"📌 Conviction maximale : {noms}")
    if ventes:
        noms = ", ".join(d["name"] for _,d in ventes[:3])
        lines.append(f"⚠️ Surachat : {noms}")
    if not lines:
        lines.append("⚪ Aucun signal fort — marché en attente")
    return lines

# ==============================================================================
# CALCUL DES INDICATEURS ET PRÉDICTION ML
# ==============================================================================
def fetch_indicators_and_predict(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2y")
        if df.empty or len(df) < 200: return None

        delta = df['Close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs    = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']      = exp1 - exp2
        df['Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MA20']    = df['Close'].rolling(window=20).mean()
        df['SMA200']  = df['Close'].rolling(window=200).mean()

        high_low   = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close  = np.abs(df['Low']  - df['Close'].shift())
        df['ATR']    = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()
        df['Var_6M'] = df['Close'].pct_change(periods=126) * 100

        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['OBV']     = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
            vol_trend = "📈" if df['OBV'].iloc[-1] > df['OBV_SMA'].iloc[-1] else "📉"
        else:
            vol_trend = "—"

        prob_up = 0.0
        ml_ok = False

        if ML_AVAILABLE:
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            features = ['RSI', 'ATR', 'Var_6M'] # Simplifié pour la robustesse
            ml_df = df.dropna(subset=features + ['Target']).copy()
            if len(ml_df) > 100:
                X = ml_df[features]; y = ml_df['Target']
                if y.nunique() >= 2:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(X, y)
                    proba = model.predict_proba(df[features].iloc[-1:].fillna(0))[0]
                    prob_up = proba[1] * 100 if len(proba) > 1 else 0
                    ml_ok   = True

        last_row = df.iloc[-1]; prev_row = df.iloc[-2]
        info = ticker.info
        current_price = last_row['Close']

        return {
            "price":       current_price,
            "change":      ((current_price - prev_row['Close']) / prev_row['Close']) * 100,
            "margin":      info.get("operatingMargins"),
            "rsi":         last_row['RSI'],
            "macd_trend":  "🍏" if last_row['MACD'] > last_row['Signal'] else "🔻",
            "sma_trend":   "↗️" if current_price > last_row['SMA200'] else "↘️",
            "atr":         last_row['ATR'],
            "momentum_6m": last_row['Var_6M'],
            "per":         info.get('trailingPE') or info.get('forwardPE'),
            "vol_trend":   vol_trend,
            "prob_up":     prob_up,
            "ml_ok":       ml_ok
        }
    except Exception as e:
        return None

# ==============================================================================
# GÉNÉRATION DU RAPPORT
# ==============================================================================
def generer_rapport():
    SEP = "════════════════════════════════"
    macro = get_macro_context()
    macro_mod = macro["modifier"]
    macro_tnx = macro["tnx"]
    is_bull   = macro["is_bull"]

    # Affichage du régime de marché
    regime_str = "🐂 MARCHÉ HAUSSIER" if is_bull else "🐻 MARCHÉ BAISSIER / INCERTAIN"
    
    if macro["vix"] is not None:
        meteo = f"{macro['regime']} (VIX {macro['vix']} · US10Y {macro['tnx']}%)"
    else:
        meteo = macro["regime"]

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    data_actifs = {}
    for symbol, name in TICKERS.items():
        res = fetch_indicators_and_predict(symbol)
        if res:
            res["name"] = name
            data_actifs[symbol] = res

    sorted_items = sorted(data_actifs.items(),
                          key=lambda x: conviction_score(x[1], macro_mod, macro_tnx), reverse=True)

    m1  = "📊 " + bold(f"ANALYSE CAC 40 — {now}") + "\n"
    m1 += SEP + "\n"
    m1 += esc(f"🌡️ {meteo}") + "\n"
    m1 += esc(f"📈 Tendance S&P 500 : {regime_str}") + "\n\n"

    m1 += bold("CE QU'IL FAUT RETENIR") + "\n"
    for line in generer_resume_executif(data_actifs, macro_mod, macro_tnx):
        m1 += esc(f"  {line}") + "\n"
    m1 += "\n" + SEP + "\n\n"

    forts = [(s,d) for s,d in sorted_items if conviction_score(d, macro_mod, macro_tnx) >= 6 and d["rsi"] < 70]
    if forts:
        m1 += bold("🎯 SIGNAUX FORTS (Avec Money Management)") + "\n\n"
        for symbol, data in forts:
            conv     = conviction_score(data, macro_mod, macro_tnx)
            m1 += esc(f"• {data['name']}") + "  " + bold(f"{conv}/10") + "\n"
            m1 += esc(f"  {verdict_action(data, conv, is_bull)}") + "\n\n"

    surachat = [(s,d) for s,d in sorted_items if d["rsi"] >= 70]
    if surachat:
        m1 += bold("⚠️ ALERTES SURACHAT") + "\n\n"
        for symbol, data in surachat:
            conv = conviction_score(data, macro_mod, macro_tnx)
            m1 += esc(f"• {data['name']}  RSI {data['rsi']:.0f}  {data['change']:+.1f}% auj.") + "\n"
            m1 += esc(f"  {verdict_action(data, conv, is_bull)}") + "\n\n"

    m1 += SEP + "\n"
    m1 += "\n💼 " + bold("MON PORTEFEUILLE") + "\n"
    total_investi = total_actuel = 0
    for symbol, pos in portefeuille.items():
        if symbol in data_actifs:
            actuel_price = data_actifs[symbol]["price"]
            val_investie = pos["prix_achat"] * pos["quantite"]
            val_actuelle = actuel_price * pos["quantite"]
            pnl_euro     = val_actuelle - val_investie
            pnl_pct      = (pnl_euro / val_investie) * 100
            
            # Stop Suiveur pour le portefeuille
            atr = data_actifs[symbol]["atr"]
            stop_suiveur = actuel_price - 1.5 * atr
            
            total_investi += val_investie
            total_actuel  += val_actuelle
            perf_s = "🟢" if pnl_euro >= 0 else "🔴"
            m1 += esc(f"  {perf_s} {pos['nom']} ({pos['quantite']} part)") + "\n"
            m1 += esc(f"     P&L : {pnl_euro:+.2f} € ({pnl_pct:+.1f}%)") + "\n"
            m1 += esc(f"     Stop Suiveur théorique : {stop_suiveur:.2f} €") + "\n\n"

    m1 += esc("⚠️ Systeme d'analyse — n'execute aucun ordre. Pas un conseil financier.")

    envoyer_telegram(m1, CHAT_ID, TELEGRAM_TOKEN)

if __name__ == "__main__":
    generer_rapport()
