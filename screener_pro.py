import os
import yfinance as yf
import pandas as pd
import warnings
import requests
from datetime import datetime
import numpy as np

warnings.filterwarnings('ignore')

# SECURITE : On recupere les codes depuis le coffre-fort de GitHub
TOKEN = os.environ.get("TELEGRAM_TOKEN")
ID = os.environ.get("TELEGRAM_CHAT_ID")

def envoyer(texte):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": ID, "text": texte})

# VOTRE LISTE DE SURVEILLANCE
actifs = {
    "WPEA.PA": "ETF MSCI World",
    "ESE.PA": "ETF S&P 500",
    "MC.PA": "LVMH",
    "OR.PA": "L'Oreal",
    "TTE.PA": "TotalEnergies",
    "AI.PA": "Air Liquide",
    "SU.PA": "Schneider Elec",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GLD": "Or Physique"
}

try:
    # 1. METEO (Bouclier VIX)
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        if not vix_data.empty:
            vix = vix_data['Close'].dropna().iloc[-1]
            if vix < 20: meteo = f"🟢 CALME ({vix:.2f})"
            elif vix < 30: meteo = f"🟠 NERVEUX ({vix:.2f})"
            else: meteo = f"🔴 PANIQUE ({vix:.2f})"
        else:
            meteo = "⚪ INDISPONIBLE (Marché Fermé)"
    except:
        meteo = "⚪ ERREUR FLUX VIX"

    # 2. ANALYSE (Bouclier individuel par action)
    resultats = {}
    verdicts = {"ACHETER": [], "VENDRE": [], "CONSERVER": []}
    
    for t, nom in actifs.items():
        try:
            data = yf.Ticker(t).history(period="1y")
            if data.empty or len(data) < 126: 
                continue 
            
            data.dropna(subset=['Close'], inplace=True)
            close = data['Close']
            prix = close.iloc[-1]
            
            sma200 = close.rolling(200).mean().iloc[-1]
            perf = (prix / close.iloc[-126]) - 1
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
            
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi = rsi_series.fillna(50).iloc[-1] 
            
            if rsi < 35:
                verdicts["ACHETER"].append(f"• {nom}")
            elif rsi > 70:
                verdicts["VENDRE"].append(f"• {nom}")
            else:
                verdicts["CONSERVER"].append(f"• {nom}")
                
            resultats[nom] = {"prix": prix, "perf": perf, "tendance": "↗️" if prix > sma200 else "↘️"}
            
        except Exception as e:
            print(f"⚠️ Alerte : Impossible d'analyser {nom} aujourd'hui. ({e})")
            continue 

    # 3. CONSTRUCTION DU RAPPORT
    date_str = datetime.now().strftime("%d/%m/%Y")
    r = f"📊 BILAN STRATEGIQUE ({date_str})\n"
    r += "═" * 30 + "\n\n"
    r += f"🌡️ METEO : {meteo}\n\n"
    
    r += "📈 TOP MOMENTUM (6 MOIS)\n"
    top = sorted(resultats.items(), key=lambda x: x[1]['perf'], reverse=True)[:3]
    if top:
        for i, (n, v) in enumerate(top):
            r += f"{i+1}. {n} (+{v['perf']*100:.1f}%) {v['tendance']}\n"
    else:
        r += "Marchés fermés ou flux de données interrompu.\n"
    
    r += "\n✅ ACTIONS A MENER\n"
    r += "\n💰 [ACHETER / RENFORCER]\n"
    r += "\n".join(verdicts["ACHETER"]) if verdicts["ACHETER"] else "Aucune opportunite"
    r += "\n\n💎 [CONSERVER]\n"
    r += "\n".join(verdicts["CONSERVER"]) if verdicts["CONSERVER"] else "Rien a signaler"
    r += "\n\n⚠️ [VENDRE / SURVEILLER]\n"
    r += "\n".join(verdicts["VENDRE"]) if verdicts["VENDRE"] else "Aucun signal de vente"

    envoyer(r)
    print("✅ Rapport envoyé depuis GitHub Actions !")

except Exception as e:
    print(f"❌ Erreur Critique Centrale : {e}")
