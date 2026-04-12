import os
import yfinance as yf
import pandas as pd
import warnings
import requests
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')

TOKEN = os.environ.get("TELEGRAM_TOKEN")
ID = os.environ.get("TELEGRAM_CHAT_ID")

# ════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════
ALERTE_VARIATION_JOUR = 3.0   # % de variation journalière déclenchant une alerte

actifs = {
    "WPEA.PA": "ETF MSCI World",
    "ESE.PA":  "ETF S&P 500",
    "MC.PA":   "LVMH",
    "OR.PA":   "L'Oreal",
    "TTE.PA":  "TotalEnergies",
    "AI.PA":   "Air Liquide",
    "SU.PA":   "Schneider Elec",
    "AAPL":    "Apple",
    "MSFT":    "Microsoft",
    "NVDA":    "Nvidia",
    "GLD":     "Or Physique",
}

# Seuils fixes désactivés (néophyte friendly 😊)
seuils_fixes = {}

# ════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════
def envoyer(texte):
    """Envoie un message Telegram (découpe si > 4096 caractères)."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for i in range(0, len(texte), 4000):
        try:
            requests.post(url, json={"chat_id": ID, "text": texte[i:i+4000]}, timeout=10)
        except Exception as e:
            print(f"⚠️ Erreur envoi Telegram : {e}")


def calcul_rsi(close, periode=14):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(periode).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(periode).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calcul_macd(close, rapide=12, lent=26, signal=9):
    ema_r = close.ewm(span=rapide, adjust=False).mean()
    ema_l = close.ewm(span=lent,   adjust=False).mean()
    macd  = ema_r - ema_l
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig


def calcul_bollinger(close, fenetre=20, nb_ecarts=2):
    sma = close.rolling(fenetre).mean()
    std = close.rolling(fenetre).std()
    return sma + nb_ecarts * std, sma - nb_ecarts * std   # haute, basse


# ════════════════════════════════════════
# ANALYSE PAR ACTIF
# ════════════════════════════════════════
def analyser(ticker_str, nom):
    """
    Retourne un dict avec toutes les métriques, ou None si données insuffisantes.
    """
    ticker = yf.Ticker(ticker_str)
    data   = ticker.history(period="1y")

    if data.empty or len(data) < 126:
        return None

    data.dropna(subset=['Close'], inplace=True)
    close = data['Close']
    open_ = data['Open']

    prix     = close.iloc[-1]
    prix_ouv = open_.iloc[-1]
    var_jour = (prix / prix_ouv - 1) * 100

    sma200   = close.rolling(200).mean().iloc[-1]
    perf_6m  = (prix / close.iloc[-126] - 1)

    rsi        = calcul_rsi(close).iloc[-1]
    macd, sig  = calcul_macd(close)
    macd_val   = macd.iloc[-1]
    sig_val    = sig.iloc[-1]
    bb_h, bb_l = calcul_bollinger(close)
    bb_haut    = bb_h.iloc[-1]
    bb_bas     = bb_l.iloc[-1]

    # ---- Consensus analystes ----
    try:
        info = ticker.info
        rec  = info.get('recommendationKey', 'none').lower()
    except Exception:
        rec  = 'none'

    est_etf_or = ("ETF" in nom or "Or" in nom)
    if rec in ['buy', 'strong_buy']:
        avis, feu_vert = "Achat ✅", True
    elif rec == 'hold':
        avis, feu_vert = "Neutre ⚖️", False
    elif rec in ['sell', 'underperform', 'strong_sell']:
        avis, feu_vert = "Vente ❌", False
    else:
        avis, feu_vert = "Non noté", est_etf_or

    # ---- Score de conviction (0 → 3) ----
    score = 0
    if rsi < 35:           score += 1   # RSI survendu
    if macd_val > sig_val: score += 1   # MACD haussier
    if prix < bb_bas:      score += 1   # sous Bollinger basse

    # ---- Signal principal ----
    if rsi < 35:
        signal = "ACHETER" if feu_vert else "BLOQUER"
    elif rsi > 70:
        signal = "VENDRE"
    else:
        signal = "CONSERVER"

    # ---- Alertes de prix (variation journalière uniquement) ----
    alertes = []
    if abs(var_jour) >= ALERTE_VARIATION_JOUR:
        emoji = "🔻" if var_jour < 0 else "🚀"
        alertes.append(f"{emoji} Variation jour : {var_jour:+.1f}%")

    return {
        "nom":      nom,
        "prix":     prix,
        "var_jour": var_jour,
        "perf_6m":  perf_6m,
        "tendance": "↗️" if prix > sma200 else "↘️",
        "rsi":      rsi,
        "macd_ok":  macd_val > sig_val,
        "bb_pos":   "BAS" if prix < bb_bas else ("HAUT" if prix > bb_haut else "MID"),
        "avis":     avis,
        "signal":   signal,
        "score":    score,
        "alertes":  alertes,
    }


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
try:
    # --- Météo VIX ---
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        vix = vix_data['Close'].dropna().iloc[-1] if not vix_data.empty else None
        if vix is None:
            meteo = "⚪ INDISPONIBLE"
        elif vix < 20:
            meteo = f"🟢 CALME ({vix:.1f})"
        elif vix < 30:
            meteo = f"🟠 NERVEUX ({vix:.1f})"
        else:
            meteo = f"🔴 PANIQUE ({vix:.1f})"
    except Exception:
        meteo = "⚪ ERREUR FLUX VIX"

    # --- Analyse de chaque actif ---
    resultats        = {}
    verdicts         = {"ACHETER": [], "BLOQUER": [], "VENDRE": [], "CONSERVER": []}
    alertes_globales = []

    for t, nom in actifs.items():
        try:
            res = analyser(t, nom)
            if res is None:
                print(f"⚠️ Données insuffisantes : {nom}")
                continue
            resultats[nom] = res
            verdicts[res["signal"]].append(res)
            for a in res["alertes"]:
                alertes_globales.append(f"  {nom} → {a}")
        except Exception as e:
            print(f"⚠️ Erreur sur {nom} : {e}")
            continue

    # --- Construction du rapport ---
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    r  = f"📊 BILAN STRATÉGIQUE 7.0\n"
    r += f"🗓️ {date_str}\n"
    r += "═" * 32 + "\n\n"
    r += f"🌡️ MÉTÉO MARCHÉ : {meteo}\n\n"

    # Top momentum
    r += "📈 TOP MOMENTUM 6 MOIS\n"
    top3 = sorted(resultats.values(), key=lambda x: x['perf_6m'], reverse=True)[:3]
    if top3:
        for i, v in enumerate(top3, 1):
            r += f"  {i}. {v['nom']}  {v['perf_6m']*100:+.1f}%  {v['tendance']}\n"
    else:
        r += "  Données indisponibles\n"

    # Alertes journalières
    r += "\n🔔 ALERTES DU JOUR (±3%)\n"
    r += "\n".join(alertes_globales) if alertes_globales else "  Aucune alerte"
    r += "\n"

    # Signaux
    r += "\n✅ ACTIONS À MENER\n"

    r += "\n💰 ACHETER / RENFORCER\n"
    if verdicts["ACHETER"]:
        for v in sorted(verdicts["ACHETER"], key=lambda x: -x['score']):
            etoiles  = "⭐" * v['score'] if v['score'] > 0 else "○"
            macd_txt = "MACD ✅" if v['macd_ok'] else "MACD ➖"
            r += f"  • {v['nom']}  {etoiles}\n"
            r += f"    RSI {v['rsi']:.0f} | {macd_txt} | BB {v['bb_pos']} | {v['avis']}\n"
    else:
        r += "  Aucune opportunité validée\n"

    r += "\n🛡️ BLOQUÉS (signal achat, consensus défavorable)\n"
    if verdicts["BLOQUER"]:
        for v in verdicts["BLOQUER"]:
            r += f"  • {v['nom']}  (Consensus : {v['avis']})\n"
    else:
        r += "  Aucun\n"

    r += "\n💎 CONSERVER\n"
    if verdicts["CONSERVER"]:
        for v in verdicts["CONSERVER"]:
            r += f"  • {v['nom']}  RSI {v['rsi']:.0f}  {v['tendance']}\n"
    else:
        r += "  Rien à signaler\n"

    r += "\n⚠️ VENDRE / ALLÉGER\n"
    if verdicts["VENDRE"]:
        for v in verdicts["VENDRE"]:
            r += f"  • {v['nom']}  RSI {v['rsi']:.0f}  {v['var_jour']:+.1f}% aujourd'hui\n"
    else:
        r += "  Aucun signal de vente\n"

    r += "\n" + "─" * 32
    r += "\n🤖 Analyse : RSI14 · MACD · Bollinger20 · SMA200"

    envoyer(r)
    print("✅ Rapport 7.0 envoyé avec succès !")

except Exception as e:
    msg = f"❌ Erreur Critique : {e}"
    print(msg)
    try:
        envoyer(msg)
    except Exception:
        pass
