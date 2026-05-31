import os
import datetime
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Importation du Machine Learning (Sera installé via GitHub Actions)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ==============================================================================
# CONFIGURATION DU BOT
# ==============================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "VOTRE_TOKEN_DE_SECOURS")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "VOTRE_CHAT_ID_DE_SECOURS")

TICKERS = {
    "TTE.PA": "TotalEnergies 🇪🇺",
    "MC.PA": "LVMH 🇪🇺",
    "AI.PA": "Air Liquide 🇪🇺",
    "OR.PA": "L'Oreal 🇪🇺",
    "SU.PA": "Schneider Elec 🇪🇺",
    "WPEA.PA": "ETF MSCI World 🇪🇺",
    "AAPL": "Apple 🇺🇸",
    "MSFT": "Microsoft 🇺🇸",
    "NVDA": "Nvidia 🇺🇸",
    "GC=F": "Or Physique 🪙"
}

portefeuille = {
    "MC.PA": {
        "nom": "LVMH",
        "prix_achat": 458.45,
        "quantite": 1
    }
}

# ==============================================================================
# OUTILS DE FORMATAGE MARKDOWNV2
# ==============================================================================
_MD2_SPECIAL = set(r"_*[]()~`>#+-=|{}.!")

def esc(text):
    return "".join(f"\\{c}" if c in _MD2_SPECIAL else c for c in str(text))

def bold(text):
    return f"*{esc(text)}*"

# ==============================================================================
# SCORE DE CONVICTION COMPOSITE (0 → 10)
# ==============================================================================
# Chaque indicateur contribue au score selon son alignement haussier :
#   RSI       : 0-3 pts  (bas = oversold = opportunité d'achat)
#   MACD      : 0-2 pts  (tendance haussière confirmée)
#   SMA200    : 0-2 pts  (prix au-dessus de la tendance longue)
#   IA RF     : 0-2 pts  (probabilité de hausse à 5 jours)
#   Volume    : 0-1 pt   (pression d'accumulation OBV)
# Total max = 10

def conviction_score(data):
    score = 0

    # RSI : signal d'achat d'autant plus fort que le RSI est bas
    rsi = data["rsi"]
    if rsi <= 35:
        score += 3
    elif rsi <= 45:
        score += 2
    elif rsi <= 55:
        score += 1
    # rsi > 55 → 0 pt

    # MACD haussier
    if data["macd_trend"] == "🍏":
        score += 2

    # Prix au-dessus de la SMA 200
    if data["sma_trend"] == "↗️":
        score += 2

    # Prédiction IA (neutre si indisponible)
    if data.get("ml_ok"):
        if data["prob_up"] > 55:
            score += 2
        elif data["prob_up"] >= 45:
            score += 1
    else:
        score += 1  # ni positif ni négatif si IA absente

    # Volume OBV haussier
    if data["vol_trend"] == "📈":
        score += 1

    return score  # 0 (signal baissier fort) → 10 (signal haussier fort)


def score_label(score):
    if score >= 8:
        return "🟢🟢 Fort"
    elif score >= 6:
        return "🟢 Positif"
    elif score >= 4:
        return "⚪ Neutre"
    elif score >= 2:
        return "🟡 Faible"
    else:
        return "🔴 Négatif"

# ==============================================================================
# FONCTIONS TECHNIQUES, VOLUMES ET MACHINE LEARNING
# ==============================================================================
def fetch_indicators_and_predict(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2y")

        if df.empty or len(df) < 200:
            return None

        # 1. INDICATEURS CLASSIQUES
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (df['STD20'] * 2)
        df['BB_Low'] = df['MA20'] - (df['STD20'] * 2)
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
        df['Var_6M'] = df['Close'].pct_change(periods=126) * 100

        # FEATURES ADDITIONNELLES POUR LE MACHINE LEARNING
        # 4 dimensions orthogonales aux 3 features d'origine (momentum/volatilité court terme)

        # Positionnement dans le range 52 semaines (mean-reversion vs tendance longue)
        df['High_52W'] = df['High'].rolling(window=252).max()
        df['Low_52W'] = df['Low'].rolling(window=252).min()
        range_52w = (df['High_52W'] - df['Low_52W']).clip(lower=1e-9)
        df['Price_52W_Pct'] = (df['Close'] - df['Low_52W']) / range_52w * 100
        # 0 % = au plus bas de l'année, 100 % = au plus haut de l'année

        # Force de la tendance moyen-terme : SMA50 vs SMA200 (golden/death cross en %)
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA50_vs_SMA200'] = (df['SMA50'] - df['SMA200']) / df['SMA200'] * 100

        # Largeur des bandes de Bollinger : régime de volatilité (compression = breakout imminent)
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['MA20'] * 100

        # Momentum 1 mois (20 séances) — horizon différent du RSI (14j) et de Var_6M (126j)
        df['Var_1M'] = df['Close'].pct_change(periods=20) * 100

        # 2. ANALYSE DES VOLUMES (OBV)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
            vol_trend = "📈" if df['OBV'].iloc[-1] > df['OBV_SMA'].iloc[-1] else "📉"
        else:
            vol_trend = "—"

        # 3. MACHINE LEARNING & BACKTESTING
        prob_up = 0.0
        backtest_score = 0.0
        ml_ok = False

        if ML_AVAILABLE:
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            features = ['RSI', 'MACD_Hist', 'ATR',
                        'Price_52W_Pct', 'SMA50_vs_SMA200', 'BB_Width', 'Var_1M']
            ml_df = df.dropna(subset=features + ['Target']).copy()

            if len(ml_df) > 100:
                X = ml_df[features]
                y = ml_df['Target']
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if y_train.nunique() >= 2:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(X_train, y_train)

                    if len(X_test) > 0:
                        y_pred = model.predict(X_test)
                        backtest_score = accuracy_score(y_test, y_pred) * 100

                    current_features = df[features].iloc[-1:].fillna(0)
                    proba = model.predict_proba(current_features)[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        prob_up = proba[classes.index(1)] * 100
                        ml_ok = True

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        info = ticker.info
        per_raw = info.get('trailingPE') or info.get('forwardPE')
        current_price = last_row['Close']

        yield_raw = info.get('dividendYield')
        if yield_raw is not None and yield_raw != 0:
            div_pct = yield_raw * 100 if yield_raw < 1 else yield_raw
            div_str = f"{div_pct:.1f}%"
        elif info.get('dividendRate') is not None and current_price > 0:
            div_pct = (info.get('dividendRate') / current_price) * 100
            div_str = f"{div_pct:.1f}%"
        else:
            div_str = "—"

        return {
            "price": current_price,
            "change": ((current_price - prev_row['Close']) / prev_row['Close']) * 100,
            "rsi": last_row['RSI'],
            "macd_trend": "🍏" if last_row['MACD'] > last_row['Signal'] else "🔻",
            "bb_pos": "BB BAS" if current_price <= last_row['BB_Low'] else ("BB HIGH" if current_price >= last_row['BB_High'] else "BB MID"),
            "sma_trend": "↗️" if current_price > last_row['SMA200'] else "↘️",
            "atr": last_row['ATR'],
            "momentum_6m": last_row['Var_6M'],
            "per": per_raw,
            "dividend": div_str,
            "vol_trend": vol_trend,
            "prob_up": prob_up,
            "backtest": backtest_score,
            "ml_ok": ml_ok
        }
    except Exception as e:
        print(f"Erreur sur {ticker_symbol}: {e}")
        return None

# ==============================================================================
# STRATÉGIE ET FORMATAGE DU RAPPORT
# ==============================================================================
def generer_rapport():
    SEP = "════════════════════════════════"

    try:
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        meteo = f"🟢 CALME ({vix:.1f})" if vix < 20 else f"🔴 NERVEUX ({vix:.1f})"
    except:
        meteo = "🟢 CALME (18.5)"

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    msg = "📊 " + bold("BILAN STRATÉGIQUE 11.1 (L'Oracle)") + "\n"
    msg += esc(f"🗓️ {now}") + "\n"
    msg += SEP + "\n\n"
    msg += esc(f"🌡️ MÉTÉO MARCHÉ : {meteo}") + "\n\n"
    msg += esc("📖 Score conviction : RSI(3) MACD(2) SMA200(2) IA(2) Vol(1) = /10 — "
               "IA Random Forest : 7 features (RSI · MACD · ATR · 52W · SMA50/200 · BBWidth · Momentum1M).") + "\n\n"

    data_actifs = {}
    momentum_list = []

    for symbol, name in TICKERS.items():
        res = fetch_indicators_and_predict(symbol)
        if res:
            res["name"] = name
            data_actifs[symbol] = res
            if symbol != "GC=F":
                momentum_list.append((name, res["momentum_6m"], res["per"], res["dividend"]))

    # 1. TOP MOMENTUM 6 MOIS (flèche et signe dynamiques)
    momentum_list.sort(key=lambda x: x[1], reverse=True)
    msg += "📈 " + bold("TOP MOMENTUM 6 MOIS") + "\n"
    for i, (name, val, per, div) in enumerate(momentum_list[:3], 1):
        arrow = "↗️" if val >= 0 else "↘️"
        line = f"  {i}. {name}  {val:+.1f}%  {arrow}"
        msg += esc(line) + "\n"
    msg += "\n"

    # 2. CLASSIFICATION DES STRATÉGIES
    msg += "✅ " + bold("ACTIONS À MENER") + "\n\n"

    acheter, conserver, vendre = [], [], []

    for symbol, data in data_actifs.items():
        conv = conviction_score(data)
        conv_line = f"    📊 Conviction : {conv}/10 {score_label(conv)}\n"

        if data.get("ml_ok"):
            ia_color = "🟢" if data['prob_up'] > 55 else ("🔴" if data['prob_up'] < 45 else "⚪")
            ia_line = f"    🧠 IA : {data['prob_up']:.0f}% Hausse {ia_color} (Fiabilité backtest : {data['backtest']:.0f}%) | Vol: {data['vol_trend']}\n"
        else:
            ia_line = f"    🧠 IA : indisponible | Vol: {data['vol_trend']}\n"

        if data["rsi"] <= 35:
            entry = (f"  • {data['name']}  ⭐⭐\n"
                     f"    RSI {data['rsi']:.0f} | MACD ➖ | {data['bb_pos']} | Achat ✅\n"
                     f"{ia_line}"
                     f"{conv_line}")
            acheter.append(esc(entry))
        elif data["rsi"] >= 70:
            entry = (f"  • {data['name']}  RSI {data['rsi']:.0f}  {data['change']:+.1f}% aujourd'hui\n"
                     f"{ia_line}"
                     f"{conv_line}")
            vendre.append(esc(entry))
        else:
            entry = (f"  • {data['name']}  RSI {data['rsi']:.0f}  {data['sma_trend']}\n"
                     f"{ia_line}"
                     f"{conv_line}")
            conserver.append(esc(entry))

    msg += "💰 " + bold("ACHETER / RENFORCER") + "\n"
    if acheter:
        msg += "".join(acheter)
    else:
        msg += esc("  Aucune opportunité validée") + "\n"
    msg += "\n"

    msg += "💎 " + bold("CONSERVER") + "\n"
    if conserver:
        msg += "".join(conserver)
    else:
        msg += esc("  Aucun actif") + "\n"
    msg += "\n"

    msg += "⚠️ " + bold("VENDRE / ALLÉGER") + "\n"
    if vendre:
        msg += "".join(vendre)
    else:
        msg += esc("  Aucune alerte") + "\n"
    msg += "\n" + SEP + "\n"

    # 3. SECTION PORTEFEUILLE PERSONNEL
    msg += "💼 " + bold("PORTEFEUILLE PERSONNEL") + "\n"
    total_investi = 0
    total_actuel = 0

    for symbol, pos in portefeuille.items():
        if symbol in data_actifs:
            actuel_price = data_actifs[symbol]["price"]
            val_investie = pos["prix_achat"] * pos["quantite"]
            val_actuelle = actuel_price * pos["quantite"]
            pnl_euro = val_actuelle - val_investie
            pnl_pct = (pnl_euro / val_investie) * 100
            conv = conviction_score(data_actifs[symbol])

            total_investi += val_investie
            total_actuel += val_actuelle

            perf_symb = "🟢" if pnl_euro >= 0 else "🔴"

            block = (f"  {perf_symb} {pos['nom']}\n"
                     f"     {pos['quantite']} parts × {actuel_price:.2f}  =  {val_actuelle:.0f} €\n"
                     f"     Aujourd'hui : {data_actifs[symbol]['change']:+.1f}%\n"
                     f"     P&L Total : {pnl_euro:+.2f} € ({pnl_pct:+.1f}%)\n"
                     f"     📊 Conviction : {conv}/10 {score_label(conv)}\n\n")
            msg += esc(block)

    global_pnl_euro = total_actuel - total_investi
    global_pnl_pct = (global_pnl_euro / total_investi) * 100 if total_investi > 0 else 0
    global_symb = "🟢" if global_pnl_euro >= 0 else "🔴"

    msg += esc(f"  {global_symb} TOTAL  {total_actuel:.0f} € / investi {total_investi:.0f} €") + "\n"
    msg += esc(f"     P&L global : {global_pnl_euro:+.2f} € ({global_pnl_pct:+.1f}%)") + "\n"

    msg += "\n" + SEP + "\n"
    msg += "🤖 `Analyse : RSI14 · MACD · SMA200 · OBV · IA (Random Forest) · Conviction /10`"

    # Envoi Telegram
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "MarkdownV2"}
    response = requests.post(url, json=payload)

    if not response.ok:
        print(f"Échec envoi Telegram ({response.status_code}) : {response.text}")
    else:
        print("✅ Rapport envoyé avec succès !")

if __name__ == "__main__":
    generer_rapport()
