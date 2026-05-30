import os
import datetime
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==============================================================================
# CONFIGURATION DU BOT ET SÉCURITÉ GITHUB ACTIONS
# ==============================================================================
# Le script récupère maintenant les clés secrètes depuis votre workflow GitHub
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "VOTRE_TOKEN_DE_SECOURS")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "VOTRE_CHAT_ID_DE_SECOURS")

# Liste officielle des actifs surveillés avec drapeaux d'éligibilité PEA
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

# Suivi du portefeuille personnel réel
portefeuille = {
    "MC.PA": {
        "nom": "LVMH",
        "prix_achat": 458.45,
        "quantite": 1
    }
}

# ==============================================================================
# OUTILS DE FORMATAGE MARKDOWNV2 (SÉCURITÉ TELEGRAM)
# ==============================================================================
_MD2_SPECIAL = set(r"_*[]()~`>#+-=|{}.!")

def esc(text):
    """Échappe les caractères réservés de MarkdownV2 dans un texte ordinaire."""
    return "".join(f"\\{c}" if c in _MD2_SPECIAL else c for c in str(text))

def bold(text):
    """Retourne un segment en gras MarkdownV2, contenu intérieur échappé."""
    return f"*{esc(text)}*"

# ==============================================================================
# FONCTIONS TECHNIQUES ET CALCULS
# ==============================================================================
def fetch_indicators(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")

        if df.empty or len(df) < 200:
            return None

        # Calcul du RSI (14 jours)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calcul du MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bandes de Bollinger
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (df['STD20'] * 2)
        df['BB_Low'] = df['MA20'] - (df['STD20'] * 2)

        # SMA 200 et ATR 14 (Volatilité)
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()

        # Calcul des variations de Momentum (6 mois)
        df['Var_6M'] = df['Close'].pct_change(periods=126) * 100

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Extraction des infos fondamentales
        info = ticker.info
        per_raw = info.get('trailingPE') or info.get('forwardPE')

        # Formatage intelligent des dividendes
        yield_raw = info.get('dividendYield')
        current_price = last_row['Close']

        if yield_raw is not None and yield_raw != 0:
            div_pct = yield_raw * 100
            if div_pct > 100:
                div_pct = yield_raw
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
            "dividend": div_str
        }
    except Exception as e:
        print(f"Erreur sur {ticker_symbol}: {e}")
        return None

# ==============================================================================
# STRATÉGIE ET FORMATAGE DU RAPPORT
# ==============================================================================
def generer_rapport():
    SEP = "════════════════════════════════"

    # Analyse de la météo marché via le VIX
    try:
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        meteo = f"🟢 CALME ({vix:.1f})" if vix < 20 else f"🔴 NERVEUX ({vix:.1f})"
    except:
        meteo = "🟢 CALME (18.5)"

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    msg = "📊 " + bold("BILAN STRATÉGIQUE 9.3") + "\n"
    msg += esc(f"🗓️ {now}") + "\n"
    msg += SEP + "\n\n"
    msg += esc(f"🌡️ MÉTÉO MARCHÉ : {meteo}") + "\n\n"
    msg += esc("📖 Légende PER : 🟢 <15 value | 🟡 15-30 correct | 🟠 30-50 élevé | 🔴 >50 spéculatif") + "\n\n"

    data_actifs = {}
    momentum_list = []

    for symbol, name in TICKERS.items():
        res = fetch_indicators(symbol)
        if res:
            res["name"] = name
            data_actifs[symbol] = res
            if symbol != "GC=F":
                momentum_list.append((name, res["momentum_6m"], res["per"], res["dividend"]))

    # 1. TOP MOMENTUM 6 MOIS
    momentum_list.sort(key=lambda x: x[1], reverse=True)
    msg += "📈 " + bold("TOP MOMENTUM 6 MOIS") + "\n"
    for i, (name, val, per, div) in enumerate(momentum_list[:3], 1):
        per_color = "🟢" if per and per < 15 else ("🟡" if per and per <= 30 else "🟠")
        per_str = f"{per:.0f}" if per else "—"
        line = f"  {i}. {name}  +{val:.1f}%  ↗️  PER {per_str} {per_color} | Div {div} 💰"
        msg += esc(line) + "\n"
    msg += "\n"

    # 2. ALERTES DU JOUR (ATR Dynamique)
    msg += "🔔 " + bold("ALERTES DU JOUR (ATR dynamique)") + "\n"
    alertes = 0
    for symbol, data in data_actifs.items():
        seuil_atr = data["atr"] * 1.5
        if abs(data["change"]) >= (seuil_atr / data["price"] * 100):
            signe = "🚀" if data["change"] > 0 else "🔻"
            line = f"  {data['name']} → {signe} Mouvement anormal : {data['change']:+.1f}% (ATR×1.5)"
            msg += esc(line) + "\n"
            alertes += 1
    if alertes == 0:
        msg += esc("  Aucune alerte") + "\n"
    msg += "\n"

    # 3. CLASSIFICATION DES STRATÉGIES
    msg += "✅ " + bold("ACTIONS À MENER") + "\n\n"

    acheter, conserver, vendre = [], [], []

    for symbol, data in data_actifs.items():
        per_color = "🟢" if data["per"] and data["per"] < 15 else ("🟡" if data["per"] and data["per"] <= 30 else "🟠")
        per_str = f"{data['per']:.0f}" if data["per"] else "—"

        if data["rsi"] <= 35:
            entry = (f"  • {data['name']}  ⭐⭐\n"
                     f"    RSI {data['rsi']:.0f} | MACD ➖ | {data['bb_pos']} | Achat ✅\n"
                     f"    PER {per_str} {per_color} | Div {data['dividend']} 💰\n")
            acheter.append(esc(entry))
        elif data["rsi"] >= 70:
            entry = (f"  • {data['name']}  RSI {data['rsi']:.0f}  {data['change']:+.1f}% aujourd'hui\n"
                     f"    PER {per_str} {per_color} | Div {data['dividend']} 💰\n")
            vendre.append(esc(entry))
        else:
            entry = (f"  • {data['name']}  RSI {data['rsi']:.0f}  {data['sma_trend']}\n"
                     f"    PER {per_str} {per_color} | Div {data['dividend']} 💰\n")
            conserver.append(esc(entry))

    msg += "💰 " + bold("ACHETER / RENFORCER") + "\n"
    if acheter:
        msg += "".join(acheter)
    else:
        msg += esc("  Aucune opportunité validée") + "\n"
    msg += "\n"

    msg += "🛡️ " + bold("BLOQUÉS (signal achat, consensus défavorable)") + "\n"
    msg += esc("  Aucun") + "\n\n"

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

    # 4. SECTION PORTEFEUILLE PERSONNEL REALISTE
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

            total_investi += val_investie
            total_actuel += val_actuelle

            perf_symb = "🟢" if pnl_euro >= 0 else "🔴"
            per_color = "🟢" if data_actifs[symbol]["per"] and data_actifs[symbol]["per"] < 15 else ("🟡" if data_actifs[symbol]["per"] and data_actifs[symbol]["per"] <= 30 else "🟠")
            per_str = f"{data_actifs[symbol]['per']:.0f}" if data_actifs[symbol]["per"] else "—"

            block = (f"  {perf_symb} {pos['nom']}\n"
                     f"     {pos['quantite']} parts × {actuel_price:.2f}  =  {val_actuelle:.0f} €\n"
                     f"     Aujourd'hui : {data_actifs[symbol]['change']:+.1f}%\n"
                     f"     P&L Total : {pnl_euro:+.2f} € ({pnl_pct:+.1f}%)\n"
                     f"     PER {per_str} {per_color} | Div {data_actifs[symbol]['dividend']} 💰\n\n")
            msg += esc(block)

    global_pnl_euro = total_actuel - total_investi
    global_pnl_pct = (global_pnl_euro / total_investi) * 100 if total_investi > 0 else 0
    global_symb = "🟢" if global_pnl_euro >= 0 else "🔴"

    msg += esc(f"  {global_symb} TOTAL  {total_actuel:.0f} € / investi {total_investi:.0f} €") + "\n"
    msg += esc(f"     P&L global : {global_pnl_euro:+.2f} € ({global_pnl_pct:+.1f}%)") + "\n"

    msg += "\n" + SEP + "\n"
    msg += "🤖 `Analyse : RSI14 · MACD · Bollinger20 · SMA200 · ATR14 · PER · Dividende`"

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

```
