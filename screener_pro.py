import os
import yfinance as yf
import pandas as pd
import warnings
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from io import BytesIO
import numpy as np

warnings.filterwarnings('ignore')

TOKEN = os.environ.get("TELEGRAM_TOKEN")
ID    = os.environ.get("TELEGRAM_CHAT_ID")

# ════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════

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

# ── Portefeuille personnel ──────────────────────────────────────────────────
# Remplis ce dictionnaire quand tu auras tes premières positions.
# Format : "TICKER": {"nom": "...", "prix_achat": 00.00, "quantite": 0}
# Exemple : "WPEA.PA": {"nom": "ETF MSCI World", "prix_achat": 5.12, "quantite": 10}
portefeuille = {
    "MC.PA": {
        "nom": "LVMH", 
        "prix_achat": 458.45,
        "quantite": 1
    }
}

# ── Multiplicateur ATR pour les alertes ────────────────────────────────────
# Une variation > ATR_MULT × ATR déclenche une alerte (défaut : 1.5×)
ATR_MULT = 1.5

# ════════════════════════════════════════
# UTILITAIRES TELEGRAM
# ════════════════════════════════════════
def envoyer_texte(texte):
    """Envoie un message texte Telegram (découpe si > 4000 caractères)."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for i in range(0, len(texte), 4000):
        try:
            requests.post(url, json={"chat_id": ID, "text": texte[i:i+4000]}, timeout=10)
        except Exception as e:
            print(f"⚠️ Erreur envoi texte Telegram : {e}")


def envoyer_image(buf, legende=""):
    """Envoie une image (BytesIO) sur Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    buf.seek(0)
    try:
        requests.post(url, data={"chat_id": ID, "caption": legende},
                      files={"photo": ("chart.png", buf, "image/png")}, timeout=30)
    except Exception as e:
        print(f"⚠️ Erreur envoi image Telegram : {e}")


# ════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ════════════════════════════════════════
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
    return sma + nb_ecarts * std, sma - nb_ecarts * std


def calcul_atr(data, periode=14):
    """Average True Range : mesure le 'rythme cardiaque' normal d'un actif."""
    high  = data['High']
    low   = data['Low']
    close = data['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(periode).mean().iloc[-1]


# ════════════════════════════════════════
# GRAPHIQUES
# ════════════════════════════════════════
def graphique_global(resultats):
    """Graphique à barres : performance 6 mois de tous les actifs."""
    noms  = [v['nom']        for v in resultats.values()]
    perfs = [v['perf_6m']*100 for v in resultats.values()]
    couleurs = ['#2ecc71' if p >= 0 else '#e74c3c' for p in perfs]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.barh(noms, perfs, color=couleurs, edgecolor='#0f3460', linewidth=0.5)
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    for bar, val in zip(bars, perfs):
        ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right',
                color='white', fontsize=8, fontweight='bold')

    ax.set_title('📈 Performance 6 Mois', color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines[:].set_color('#0f3460')
    ax.set_xlabel('Performance (%)', color='#aaaaaa', fontsize=9)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf


def graphique_actif(ticker_str, nom, data):
    """Graphique individuel : cours 3 mois + Bollinger + MACD (90 derniers jours)."""
    data = data.tail(90).copy()
    close = data['Close']
    dates = data.index

    bb_h, bb_l = calcul_bollinger(close)
    sma20       = close.rolling(20).mean()
    macd, sig   = calcul_macd(close)
    histogramme = macd - sig

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[:].set_color('#0f3460')

    # ── Cours + Bollinger ──
    ax1.plot(dates, close,  color='#f0e68c', linewidth=1.5, label='Cours', zorder=3)
    ax1.plot(dates, sma20,  color='#87ceeb', linewidth=1,   label='SMA20',  linestyle='--', alpha=0.7)
    ax1.plot(dates, bb_h,   color='#ff7f7f', linewidth=0.8, label='BB Haut', linestyle=':')
    ax1.plot(dates, bb_l,   color='#90ee90', linewidth=0.8, label='BB Bas',  linestyle=':')
    ax1.fill_between(dates, bb_l, bb_h, alpha=0.07, color='white')
    ax1.set_title(f'{nom}  —  Cours & Bollinger (90j)', color='white',
                  fontsize=11, fontweight='bold', pad=8)
    ax1.legend(loc='upper left', fontsize=7, facecolor='#0f3460',
               labelcolor='white', framealpha=0.8)
    ax1.yaxis.label.set_color('white')

    # ── MACD ──
    colors_hist = ['#2ecc71' if h >= 0 else '#e74c3c' for h in histogramme]
    ax2.bar(dates, histogramme, color=colors_hist, alpha=0.6, label='Histogramme')
    ax2.plot(dates, macd, color='#00bfff', linewidth=1,   label='MACD')
    ax2.plot(dates, sig,  color='#ff8c00', linewidth=1,   label='Signal', linestyle='--')
    ax2.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
    ax2.set_title('MACD', color='white', fontsize=9, pad=4)
    ax2.legend(loc='upper left', fontsize=7, facecolor='#0f3460',
               labelcolor='white', framealpha=0.8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf


# ════════════════════════════════════════
# ANALYSE PAR ACTIF
# ════════════════════════════════════════
def analyser(ticker_str, nom):
    ticker = yf.Ticker(ticker_str)
    data   = ticker.history(period="1y")

    if data.empty or len(data) < 126:
        return None, None

    data.dropna(subset=['Close'], inplace=True)
    close = data['Close']
    open_ = data['Open']

    prix     = close.iloc[-1]
    prix_ouv = open_.iloc[-1]
    var_jour = (prix / prix_ouv - 1) * 100

    sma200  = close.rolling(200).mean().iloc[-1]
    perf_6m = (prix / close.iloc[-126] - 1)

    rsi        = calcul_rsi(close).iloc[-1]
    macd, sig  = calcul_macd(close)
    macd_val   = macd.iloc[-1]
    sig_val    = sig.iloc[-1]
    bb_h, bb_l = calcul_bollinger(close)
    bb_haut    = bb_h.iloc[-1]
    bb_bas     = bb_l.iloc[-1]
    atr        = calcul_atr(data)

    # ── Consensus analystes ──
    try:
        info = ticker.info
        rec  = info.get('recommendationKey', 'none').lower()
    except Exception:
        rec = 'none'

    est_etf_or = ("ETF" in nom or "Or" in nom)
    if rec in ['buy', 'strong_buy']:
        avis, feu_vert = "Achat ✅", True
    elif rec == 'hold':
        avis, feu_vert = "Neutre ⚖️", False
    elif rec in ['sell', 'underperform', 'strong_sell']:
        avis, feu_vert = "Vente ❌", False
    else:
        avis, feu_vert = "Non noté", est_etf_or

    # ── Score de conviction (0 → 3) ──
    score = 0
    if rsi < 35:           score += 1
    if macd_val > sig_val: score += 1
    if prix < bb_bas:      score += 1

    # ── Signal principal ──
    if rsi < 35:
        signal = "ACHETER" if feu_vert else "BLOQUER"
    elif rsi > 70:
        signal = "VENDRE"
    else:
        signal = "CONSERVER"

    # ── Alerte ATR dynamique ──
    alertes = []
    seuil_atr = ATR_MULT * atr
    if abs(var_jour / 100 * prix) >= seuil_atr:
        emoji = "🔻" if var_jour < 0 else "🚀"
        alertes.append(f"{emoji} Mouvement anormal : {var_jour:+.1f}% (ATR×{ATR_MULT} = {seuil_atr/prix*100:.1f}%)")

    return {
        "ticker":   ticker_str,
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
    }, data


# ════════════════════════════════════════
# SECTION PORTEFEUILLE
# ════════════════════════════════════════
def section_portefeuille(resultats):
    if not portefeuille:
        return "💼 PORTEFEUILLE\n  (Aucune position renseignée — prêt à l'emploi !)\n"

    r  = "💼 PORTEFEUILLE PERSONNEL\n"
    total_investi  = 0.0
    total_actuel   = 0.0

    for ticker_str, pos in portefeuille.items():
        nom         = pos.get("nom", ticker_str)
        prix_achat  = pos.get("prix_achat", 0)
        quantite    = pos.get("quantite", 0)
        valeur_achat = prix_achat * quantite

        res = resultats.get(nom)
        if res:
            prix_actuel  = res["prix"]
            valeur_actuelle = prix_actuel * quantite
            pv           = valeur_actuelle - valeur_achat
            pv_pct       = (pv / valeur_achat * 100) if valeur_achat else 0
            emoji        = "🟢" if pv >= 0 else "🔴"
            r += f"  {emoji} {nom}\n"
            r += f"     {quantite} parts × {prix_actuel:.2f}  =  {valeur_actuelle:.0f} €\n"
            r += f"     P&L : {pv:+.0f} € ({pv_pct:+.1f}%)\n"
            total_investi += valeur_achat
            total_actuel  += valeur_actuelle

            # Alerte +10% / -10%
            if pv_pct >= 10:
                r += f"     🎯 +10% atteint — envisage de sécuriser ?\n"
            elif pv_pct <= -10:
                r += f"     ⚠️ -10% — point de vigilance\n"
        else:
            r += f"  ⚪ {nom} — cours indisponible\n"

    if total_investi > 0:
        pv_total     = total_actuel - total_investi
        pv_total_pct = pv_total / total_investi * 100
        emoji_tot    = "🟢" if pv_total >= 0 else "🔴"
        r += f"\n  {emoji_tot} TOTAL  {total_actuel:.0f} € / investi {total_investi:.0f} €\n"
        r += f"     P&L global : {pv_total:+.0f} € ({pv_total_pct:+.1f}%)\n"
    return r


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
try:
    # ── Météo VIX ──
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        vix = vix_data['Close'].dropna().iloc[-1] if not vix_data.empty else None
        if   vix is None:  meteo = "⚪ INDISPONIBLE"
        elif vix < 20:     meteo = f"🟢 CALME ({vix:.1f})"
        elif vix < 30:     meteo = f"🟠 NERVEUX ({vix:.1f})"
        else:              meteo = f"🔴 PANIQUE ({vix:.1f})"
    except Exception:
        meteo = "⚪ ERREUR FLUX VIX"

    # ── Analyse de chaque actif ──
    resultats        = {}
    data_cache       = {}
    verdicts         = {"ACHETER": [], "BLOQUER": [], "VENDRE": [], "CONSERVER": []}
    alertes_globales = []

    for t, nom in actifs.items():
        try:
            res, data = analyser(t, nom)
            if res is None:
                print(f"⚠️ Données insuffisantes : {nom}")
                continue
            resultats[nom]  = res
            data_cache[nom] = (t, data)
            verdicts[res["signal"]].append(res)
            for a in res["alertes"]:
                alertes_globales.append(f"  {nom} → {a}")
        except Exception as e:
            print(f"⚠️ Erreur sur {nom} : {e}")
            continue

    # ══════════════════════════════════════
    # RAPPORT TEXTE
    # ══════════════════════════════════════
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    r  = f"📊 BILAN STRATÉGIQUE 8.0\n"
    r += f"🗓️ {date_str}\n"
    r += "═" * 32 + "\n\n"
    r += f"🌡️ MÉTÉO MARCHÉ : {meteo}\n\n"

    # Top momentum
    r += "📈 TOP MOMENTUM 6 MOIS\n"
    top3 = sorted(resultats.values(), key=lambda x: x['perf_6m'], reverse=True)[:3]
    for i, v in enumerate(top3, 1):
        r += f"  {i}. {v['nom']}  {v['perf_6m']*100:+.1f}%  {v['tendance']}\n"

    # Alertes ATR
    r += "\n🔔 ALERTES DU JOUR (ATR dynamique)\n"
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

    # Portefeuille
    r += "\n" + "─" * 32 + "\n"
    r += section_portefeuille(resultats)

    r += "\n" + "─" * 32
    r += "\n🤖 Analyse : RSI14 · MACD · Bollinger20 · SMA200 · ATR14"

    envoyer_texte(r)

    # ══════════════════════════════════════
    # GRAPHIQUES
    # ══════════════════════════════════════

    # 1. Graphique global performances
    try:
        buf = graphique_global(resultats)
        envoyer_image(buf, "📊 Performance comparée 6 mois")
        print("✅ Graphique global envoyé")
    except Exception as e:
        print(f"⚠️ Erreur graphique global : {e}")

    # 2. Graphiques individuels (actifs signalés ACHETER, BLOQUER ou VENDRE)
    actifs_a_grapher = verdicts["ACHETER"] + verdicts["BLOQUER"] + verdicts["VENDRE"]
    for v in actifs_a_grapher:
        nom = v["nom"]
        try:
            ticker_str, data = data_cache[nom]
            buf = graphique_actif(ticker_str, nom, data)
            legende = f"{nom}  |  RSI {v['rsi']:.0f}  |  BB {v['bb_pos']}  |  {v['avis']}"
            envoyer_image(buf, legende)
            print(f"✅ Graphique {nom} envoyé")
        except Exception as e:
            print(f"⚠️ Erreur graphique {nom} : {e}")

    print("✅ Rapport 8.0 complet envoyé !")

except Exception as e:
    msg = f"❌ Erreur Critique V8.0 : {e}"
    print(msg)
    try:
        envoyer_texte(msg)
    except Exception:
        pass
