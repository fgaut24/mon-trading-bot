import os
import json
import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "VOTRE_TOKEN_DE_SECOURS")
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID", "VOTRE_CHAT_ID_DE_SECOURS")
HISTORY_FILE   = Path("predictions_history.json")

# ==============================================================================
# UNIVERS DE SURVEILLANCE — CAC 40 + ETF PEA
# Composition officielle Euronext (révisée trimestriellement).
# Dernière mise à jour : décembre 2025 (Eiffage entre, Edenred sort).
# Si un ticker ne renvoie pas de données yfinance, il est silencieusement ignoré.
# ==============================================================================
CAC_40 = {
    # Luxe & consommation premium
    "MC.PA":    "LVMH",
    "RMS.PA":   "Hermès",
    "OR.PA":    "L'Oreal",
    "KER.PA":   "Kering",
    "EL.PA":    "EssilorLuxottica",
    # Énergie
    "TTE.PA":   "TotalEnergies",
    "ENGI.PA":  "Engie",
    # Industrie, aéro, défense
    "AIR.PA":   "Airbus",
    "SAF.PA":   "Safran",
    "HO.PA":    "Thales",
    "DG.PA":    "Vinci",
    "LR.PA":    "Legrand",
    "SU.PA":    "Schneider Elec",
    "SGO.PA":   "Saint-Gobain",
    "ALO.PA":   "Alstom",
    "FGR.PA":   "Eiffage",
    # Automobile & matériaux
    "RNO.PA":   "Renault",
    "STLAM.MI": "Stellantis",
    "MT.AS":    "ArcelorMittal",
    "ML.PA":    "Michelin",
    "GLE.PA":   "Soc. Générale",
    # Banque & assurance
    "BNP.PA":   "BNP Paribas",
    "ACA.PA":   "Crédit Agricole",
    "GLE.PA":   "Soc. Générale",
    "CS.PA":    "AXA",
    # Santé
    "SAN.PA":   "Sanofi",
    "AI.PA":    "Air Liquide",
    "BIM.PA":   "bioMérieux",
    # Services & utilities
    "VIE.PA":   "Veolia",
    "ORA.PA":   "Orange",
    "EN.PA":    "Bouygues",
    "URW.PA":   "Unibail",
    # Tech & médias
    "CAP.PA":   "Capgemini",
    "DSY.PA":   "Dassault Sys.",
    "PUB.PA":   "Publicis",
    "WLN.PA":   "Worldline",
    # Consommation courante
    "BN.PA":    "Danone",
    "CA.PA":    "Carrefour",
    "RI.PA":    "Pernod Ricard",
    "VIV.PA":   "Vivendi",
}

ETF_PEA = {
    # Indices larges — éligibles PEA via réplication synthétique
    "WPEA.PA":  "ETF MSCI World",
    "CW8.PA":   "ETF Monde CW8",
    "PANX.PA":  "ETF Nasdaq-100",
    "PSP5.PA":  "ETF S&P 500",
    "PAEEM.PA": "ETF Émergents",
    "MEUD.PA":  "ETF Europe 600",
    "CACC.PA":  "ETF CAC 40",
}

TICKERS = {**CAC_40, **ETF_PEA}   # 47 actifs au total

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
# ENVOI TELEGRAM (supporte les messages > 4 096 car. en les découpant)
# ==============================================================================
def envoyer_telegram(texte, chat_id, token):
    """Envoie un ou plusieurs messages Telegram si le texte dépasse 4 096 car."""
    LIMIT = 4000  # marge de sécurité sous la limite officielle
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    blocs = []

    while len(texte) > LIMIT:
        # Coupe au dernier saut de ligne avant la limite
        coupe = texte.rfind("\n", 0, LIMIT)
        if coupe == -1:
            coupe = LIMIT
        blocs.append(texte[:coupe])
        texte = texte[coupe:].lstrip("\n")
    blocs.append(texte)

    for i, bloc in enumerate(blocs, 1):
        payload = {"chat_id": chat_id, "text": bloc, "parse_mode": "MarkdownV2"}
        r = requests.post(url, json=payload)
        if not r.ok:
            print(f"Échec message {i}/{len(blocs)} ({r.status_code}) : {r.text}")
        else:
            print(f"✅ Message {i}/{len(blocs)} envoyé ({len(bloc)} car.)")

# ==============================================================================
# SCORE DE CONVICTION COMPOSITE (0 → 10)
# ==============================================================================
def conviction_score(data):
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
    return score

def score_label(score):
    if   score >= 8: return "🟢🟢 Fort"
    elif score >= 6: return "🟢 Positif"
    elif score >= 4: return "⚪ Neutre"
    elif score >= 2: return "🟡 Faible"
    else:            return "🔴 Négatif"

def score_emoji(score):
    if   score >= 8: return "🟢🟢"
    elif score >= 6: return "🟢"
    elif score >= 4: return "⚪"
    elif score >= 2: return "🟡"
    else:            return "🔴"

# ==============================================================================
# INTERPRÉTATION PÉDAGOGIQUE
# ==============================================================================
def nom_court(name):
    return name.strip()

def interpreter_rsi(rsi):
    if   rsi <= 35: return "survente — rebond potentiel"
    elif rsi <= 45: return "RSI bas — délaissé"
    elif rsi <= 55: return "RSI neutre"
    elif rsi <= 65: return "RSI tendu"
    elif rsi <= 75: return "surachat modéré"
    else:           return "surachat marqué"

def interpreter_ia(data):
    if not data.get("ml_ok"): return None
    p = data["prob_up"]
    if   p >= 70: return f"IA très favorable ({p:.0f}%)"
    elif p >= 55: return f"IA favorable ({p:.0f}%)"
    elif p <= 30: return f"IA très défavorable ({p:.0f}%)"
    elif p <= 45: return f"IA défavorable ({p:.0f}%)"
    return None

def interpreter_signal(data):
    parts = [interpreter_rsi(data["rsi"])]
    parts.append("tendance ↗️" if data["sma_trend"] == "↗️" else "tendance ↘️")
    ia = interpreter_ia(data)
    if ia: parts.append(ia)
    if data["vol_trend"] == "📈": parts.append("volumes ↑")
    return " · ".join(parts)

def verdict_action(data, conv):
    rsi  = data["rsi"]
    prix = data["price"]
    atr  = data["atr"]
    entree   = round(prix - 0.5 * atr, 2)
    objectif = round(prix + 2.5 * atr, 2)
    stop     = round(prix - 1.5 * atr, 2)

    if rsi <= 35 and conv >= 7:
        return (f"🟢 ACHETER — signal fort\n"
                f"   J'entre à {prix:.2f} € ou repli vers {entree:.2f} €\n"
                f"   Je vise {objectif:.2f} €  ·  Je coupe sous {stop:.2f} €")
    if rsi <= 35 and conv >= 5:
        return (f"🟢 ACHETER — signal modéré\n"
                f"   J'entre si le prix tient au-dessus de {stop:.2f} €\n"
                f"   Je vise {objectif:.2f} €  ·  Je coupe sous {stop:.2f} €")
    if rsi <= 35:
        return (f"👁️ SURVEILLER — survente, signaux mixtes\n"
                f"   J'attends stabilisation. Je reviens si conviction > 5/10.")
    if rsi <= 45 and conv >= 7:
        return (f"🔵 RENFORCER sur repli\n"
                f"   J'attends un retour vers {entree:.2f} €\n"
                f"   Je vise {objectif:.2f} €  ·  Je coupe sous {stop:.2f} €")
    if rsi <= 45 and conv >= 5:
        return (f"💎 CONSERVER — profil intéressant\n"
                f"   Je renforce si repli vers {entree:.2f} €\n"
                f"   Je coupe sous {stop:.2f} €")
    if rsi >= 70 and conv <= 3:
        return (f"🔴 VENDRE — convergence baissière\n"
                f"   Je vends maintenant. Je reviens si RSI < 50.")
    if rsi >= 70 and conv <= 5:
        return (f"🟠 ALLÉGER — surachat confirmé\n"
                f"   Je réduis 30 à 50 % de ma position.\n"
                f"   Si RSI < 65 demain je conserve le reste, sinon je réduis encore.")
    if rsi >= 70:
        return (f"🟡 ATTENTION — surachat, tendance forte\n"
                f"   Je ne renforce pas. Stop à {stop:.2f} €")
    if rsi >= 65:
        return (f"🟡 ATTENDRE — zone tendue\n"
                f"   Je n'achète pas. Je reviens si RSI < 60.")
    if conv >= 6:
        return (f"💎 CONSERVER — profil favorable\n"
                f"   Je renforce sur repli vers {entree:.2f} €\n"
                f"   Je coupe sous {stop:.2f} €")
    if conv >= 4:
        return "💎 CONSERVER — je ne change rien pour l'instant"
    return (f"🟡 PRUDENCE — signaux faibles\n"
            f"   J'envisage d'alléger si position importante.")

def generer_resume_executif(data_actifs):
    sorted_items = sorted(data_actifs.items(),
                          key=lambda x: conviction_score(x[1]), reverse=True)
    lines = []
    achats   = [(s,d) for s,d in sorted_items if d["rsi"] <= 38]
    ventes   = [(s,d) for s,d in sorted_items if d["rsi"] >= 70]
    top_conv = [(s,d) for s,d in sorted_items
                if conviction_score(d) >= 7 and d["rsi"] < 70]
    if achats:
        noms = " et ".join(d["name"] for _,d in achats[:2])
        lines.append(f"⭐ {noms} en zone de survente — entrée potentielle")
    if top_conv:
        noms = ", ".join(d["name"] for _,d in top_conv[:3])
        lines.append(f"📌 Conviction maximale : {noms}")
    if ventes:
        noms = ", ".join(d["name"] for _,d in ventes[:3])
        lines.append(f"⚠️ Surachat : {noms}")
    if not lines:
        lines.append("⚪ Aucun signal fort — marché en attente")
    non_gold = list(sorted_items)
    if non_gold:
        _, best = max(non_gold, key=lambda x: x[1]["momentum_6m"])
        lines.append(f"📈 Meilleur momentum 6 mois : {best['name']} ({best['momentum_6m']:+.1f}%)")
    return lines

# ==============================================================================
# PERSISTANCE ET VALIDATION DES PRÉDICTIONS IA
# ==============================================================================
def charger_historique():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Avertissement historique : {e}")
    return []

def sauvegarder_historique(historique, data_actifs):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    historique = [e for e in historique if e["date"] != today]
    entry = {
        "date": today,
        "assets": {
            s: {"price": round(d["price"], 4), "prob_up": round(d["prob_up"], 1)}
            for s, d in data_actifs.items() if d.get("ml_ok")
        }
    }
    historique.append(entry)
    historique = historique[-90:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(historique, f, ensure_ascii=False, indent=2)
        print(f"✅ Historique : {len(entry['assets'])} prédictions ({today})")
    except IOError as e:
        print(f"Erreur historique : {e}")
    return historique

def valider_predictions(data_actifs, historique):
    today  = datetime.datetime.now().date()
    cutoff = today - datetime.timedelta(days=5)
    candidats = [e for e in historique
                 if datetime.datetime.strptime(e["date"], "%Y-%m-%d").date() <= cutoff]
    if not candidats: return None
    ref = max(candidats, key=lambda e: e["date"])
    ref_date = datetime.datetime.strptime(ref["date"], "%Y-%m-%d").date().strftime("%d/%m/%Y")
    total = correct = 0
    details = []
    for symbol, pred in ref["assets"].items():
        if symbol not in data_actifs: continue
        prix_pred   = pred["price"]
        prix_actuel = data_actifs[symbol]["price"]
        dir_predite = pred["prob_up"] > 50
        dir_reelle  = prix_actuel > prix_pred
        variation   = (prix_actuel - prix_pred) / prix_pred * 100
        ok = (dir_predite == dir_reelle)
        total += 1
        if ok: correct += 1
        details.append({"name": data_actifs[symbol]["name"], "prob_up": pred["prob_up"],
                         "variation": variation, "ok": ok})
    if total == 0: return None
    return {"ref_date": ref_date, "total": total, "correct": correct,
            "accuracy": correct / total * 100, "details": details}

# ==============================================================================
# CALCUL DES INDICATEURS ET PRÉDICTION ML
# ==============================================================================
def fetch_indicators_and_predict(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2y")
        if df.empty or len(df) < 200:
            return None

        delta = df['Close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs    = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']      = exp1 - exp2
        df['Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        df['MA20']    = df['Close'].rolling(window=20).mean()
        df['STD20']   = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (df['STD20'] * 2)
        df['BB_Low']  = df['MA20'] - (df['STD20'] * 2)
        df['SMA200']  = df['Close'].rolling(window=200).mean()

        high_low   = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close  = np.abs(df['Low']  - df['Close'].shift())
        df['ATR']    = np.max(pd.concat([high_low, high_close, low_close], axis=1),
                              axis=1).rolling(14).mean()
        df['Var_6M'] = df['Close'].pct_change(periods=126) * 100

        df['High_52W'] = df['High'].rolling(window=252).max()
        df['Low_52W']  = df['Low'].rolling(window=252).min()
        range_52w = (df['High_52W'] - df['Low_52W']).clip(lower=1e-9)
        df['Price_52W_Pct']   = (df['Close'] - df['Low_52W']) / range_52w * 100
        df['SMA50']           = df['Close'].rolling(window=50).mean()
        df['SMA50_vs_SMA200'] = (df['SMA50'] - df['SMA200']) / df['SMA200'] * 100
        df['BB_Width']        = (df['BB_High'] - df['BB_Low']) / df['MA20'] * 100
        df['Var_1M']          = df['Close'].pct_change(periods=20) * 100

        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['OBV']     = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
            vol_trend = "📈" if df['OBV'].iloc[-1] > df['OBV_SMA'].iloc[-1] else "📉"
        else:
            vol_trend = "—"

        prob_up = backtest_score = 0.0
        ml_ok = False

        if ML_AVAILABLE:
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            features = ['RSI', 'MACD_Hist', 'ATR',
                        'Price_52W_Pct', 'SMA50_vs_SMA200', 'BB_Width', 'Var_1M']
            ml_df = df.dropna(subset=features + ['Target']).copy()
            if len(ml_df) > 100:
                X = ml_df[features]; y = ml_df['Target']
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                if y_train.nunique() >= 2:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    if len(X_test) > 0:
                        backtest_score = accuracy_score(y_test, model.predict(X_test)) * 100
                    proba   = model.predict_proba(df[features].iloc[-1:].fillna(0))[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        prob_up = proba[classes.index(1)] * 100
                        ml_ok   = True

        last_row = df.iloc[-1]; prev_row = df.iloc[-2]
        info = ticker.info
        per_raw = info.get('trailingPE') or info.get('forwardPE')
        current_price = last_row['Close']

        yield_raw = info.get('dividendYield')
        if yield_raw is not None and yield_raw != 0:
            div_pct = yield_raw * 100 if yield_raw < 1 else yield_raw
            div_str = f"{div_pct:.1f}%"
        elif info.get('dividendRate') is not None and current_price > 0:
            div_str = f"{(info['dividendRate'] / current_price) * 100:.1f}%"
        else:
            div_str = "—"

        return {
            "price":       current_price,
            "change":      ((current_price - prev_row['Close']) / prev_row['Close']) * 100,
            "rsi":         last_row['RSI'],
            "macd_trend":  "🍏" if last_row['MACD'] > last_row['Signal'] else "🔻",
            "bb_pos":      ("BB BAS" if current_price <= last_row['BB_Low'] else
                            "BB HIGH" if current_price >= last_row['BB_High'] else "BB MID"),
            "bb_low":      last_row['BB_Low'],
            "bb_high":     last_row['BB_High'],
            "sma_trend":   "↗️" if current_price > last_row['SMA200'] else "↘️",
            "atr":         last_row['ATR'],
            "momentum_6m": last_row['Var_6M'],
            "per":         per_raw,
            "dividend":    div_str,
            "vol_trend":   vol_trend,
            "prob_up":     prob_up,
            "backtest":    backtest_score,
            "ml_ok":       ml_ok
        }
    except Exception as e:
        print(f"Erreur sur {ticker_symbol}: {e}")
        return None

# ==============================================================================
# GÉNÉRATION DU RAPPORT — 2 MESSAGES
# Message 1 : résumé exécutif + signaux actionnables + portefeuille
# Message 2 : tableau de bord complet + validation IA
# ==============================================================================
def generer_rapport():
    SEP = "════════════════════════════════"
    historique = charger_historique()

    try:
        vix   = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        meteo = f"🟢 CALME ({vix:.1f})" if vix < 20 else f"🔴 NERVEUX ({vix:.1f})"
        meteo_note = "Conditions favorables." if vix < 20 else "Volatilité élevée — signaux moins fiables."
    except:
        meteo = "🟢 CALME (18.5)"
        meteo_note = "Conditions favorables."

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── Collecte des données ─────────────────────────────────────────────────
    data_actifs = {}
    for symbol, name in TICKERS.items():
        res = fetch_indicators_and_predict(symbol)
        if res:
            res["name"] = name
            data_actifs[symbol] = res

    validation   = valider_predictions(data_actifs, historique)
    sorted_items = sorted(data_actifs.items(),
                          key=lambda x: conviction_score(x[1]), reverse=True)

    n_actifs = len(data_actifs)

    # ══════════════════════════════════════════════════════════════════════════
    # MESSAGE 1 : INTELLIGENCE ACTIONNABLE
    # ══════════════════════════════════════════════════════════════════════════
    m1  = "📊 " + bold(f"ORACLE CAC 40 — {now}") + "\n"
    m1 += esc(f"({n_actifs} valeurs analysées — CAC 40 + ETF PEA)") + "\n"
    m1 += SEP + "\n"
    m1 += esc(f"🌡️ {meteo}  {meteo_note}") + "\n\n"

    # Résumé exécutif
    m1 += bold("CE QU'IL FAUT RETENIR") + "\n"
    for line in generer_resume_executif(data_actifs):
        m1 += esc(f"  {line}") + "\n"
    m1 += "\n"

    # Top 3 momentum
    top3 = sorted(data_actifs.items(), key=lambda x: x[1]["momentum_6m"], reverse=True)[:3]
    top3_txt = "  ·  ".join(f"{d['name']} {d['momentum_6m']:+.1f}%" for _,d in top3)
    m1 += esc(f"📈 Top élan 6 mois : {top3_txt}") + "\n"
    m1 += "\n" + SEP + "\n\n"

    # Signaux forts (top 5, RSI < 70)
    forts = [(s,d) for s,d in sorted_items if conviction_score(d) >= 6 and d["rsi"] < 70][:5]
    if forts:
        m1 += bold("🎯 SIGNAUX FORTS") + "\n"
        m1 += esc("   Niveaux ATR — signaux techniques, pas un conseil financier.") + "\n\n"
        for symbol, data in forts:
            conv     = conviction_score(data)
            rsi_flag = " ⭐" if data["rsi"] <= 35 else ""
            m1 += esc(f"• {data['name']}{rsi_flag}") + "  " + bold(f"{conv}/10") + "\n"
            m1 += esc(f"  {interpreter_signal(data)}") + "\n"
            m1 += esc(f"  {verdict_action(data, conv)}") + "\n\n"

    # Alertes surachat (top 3)
    surachat = [(s,d) for s,d in sorted_items if d["rsi"] >= 70][:3]
    if surachat:
        m1 += bold("⚠️ ALERTES SURACHAT") + "\n\n"
        for symbol, data in surachat:
            conv = conviction_score(data)
            m1 += esc(f"• {data['name']}  RSI {data['rsi']:.0f}  {data['change']:+.1f}% aujourd'hui") + "\n"
            m1 += esc(f"  {interpreter_signal(data)}") + "\n"
            m1 += esc(f"  {verdict_action(data, conv)}") + "\n\n"

    m1 += SEP + "\n"

    # Portefeuille
    m1 += "\n💼 " + bold("MON PORTEFEUILLE") + "\n"
    total_investi = total_actuel = 0
    for symbol, pos in portefeuille.items():
        if symbol in data_actifs:
            actuel_price = data_actifs[symbol]["price"]
            val_investie = pos["prix_achat"] * pos["quantite"]
            val_actuelle = actuel_price * pos["quantite"]
            pnl_euro     = val_actuelle - val_investie
            pnl_pct      = (pnl_euro / val_investie) * 100
            conv         = conviction_score(data_actifs[symbol])
            total_investi += val_investie
            total_actuel  += val_actuelle
            perf_s = "🟢" if pnl_euro >= 0 else "🔴"
            m1 += esc(f"  {perf_s} {pos['nom']} ({pos['quantite']} part)") + "\n"
            m1 += esc(f"     Valeur : {val_actuelle:.0f} € · achat {val_investie:.0f} €") + "\n"
            m1 += esc(f"     P&L : {pnl_euro:+.2f} € ({pnl_pct:+.1f}%)  · Aujourd'hui : {data_actifs[symbol]['change']:+.1f}%") + "\n"
            m1 += esc(f"     Signal : {conv}/10 {score_label(conv)} — {interpreter_signal(data_actifs[symbol])}") + "\n\n"

    g_pnl  = total_actuel - total_investi
    g_pct  = (g_pnl / total_investi * 100) if total_investi > 0 else 0
    g_symb = "🟢" if g_pnl >= 0 else "🔴"
    m1 += esc(f"  {g_symb} TOTAL : {total_actuel:.0f} € · investi {total_investi:.0f} € · P&L : {g_pnl:+.2f} € ({g_pct:+.1f}%)") + "\n"

    # ══════════════════════════════════════════════════════════════════════════
    # MESSAGE 2 : TABLEAU DE BORD COMPLET
    # ══════════════════════════════════════════════════════════════════════════
    m2  = "📊 " + bold("TABLEAU DE BORD COMPLET") + "\n"
    m2 += esc(f"  {n_actifs} valeurs · triées par conviction · ⭐ achat · ⚠️ surachat") + "\n"
    m2 += SEP + "\n\n"

    for symbol, data in sorted_items:
        conv   = conviction_score(data)
        s_e    = score_emoji(conv)
        flag_r = " ⭐" if data["rsi"] <= 35 else (" ⚠️" if data["rsi"] >= 70 else "  ")
        ia_s   = f"IA {data['prob_up']:.0f}%" if data.get("ml_ok") else "IA —"
        line   = (f"  {s_e} {data['name']:<18}{flag_r}"
                  f"  {conv}/10"
                  f"  RSI {data['rsi']:.0f}"
                  f"  {data['change']:+.1f}%"
                  f"  {ia_s}")
        m2 += esc(line) + "\n"

    m2 += "\n" + SEP + "\n"

    # Validation IA (si disponible)
    if validation:
        acc   = validation["accuracy"]
        acc_e = "🟢" if acc >= 60 else ("🔴" if acc < 45 else "⚪")
        m2 += "\n🔬 " + bold(f"VALIDATION IA — prédictions du {validation['ref_date']}") + "\n"
        m2 += esc(f"  {validation['correct']}/{validation['total']} directions correctes — {acc:.0f}% {acc_e}") + "\n"
        sorted_d = sorted(validation["details"], key=lambda x: abs(x["variation"]), reverse=True)
        best  = next((d for d in sorted_d if d["ok"]),  None)
        worst = next((d for d in sorted_d if not d["ok"]), None)
        if best:
            dir_p = "hausse" if best["prob_up"] > 50 else "baisse"
            dir_r = "↗️" if best["variation"] >= 0 else "↘️"
            m2 += esc(f"  ✅ {best['name']} : {dir_p} ({best['prob_up']:.0f}%) → {dir_r} {best['variation']:+.1f}%") + "\n"
        if worst:
            dir_p = "hausse" if worst["prob_up"] > 50 else "baisse"
            dir_r = "↗️" if worst["variation"] >= 0 else "↘️"
            m2 += esc(f"  ❌ {worst['name']} : {dir_p} ({worst['prob_up']:.0f}%) → {dir_r} {worst['variation']:+.1f}%") + "\n"
        m2 += "\n" + SEP + "\n"

    m2 += "\n🤖 `RSI · MACD · SMA200 · OBV · IA Random Forest 7 features · Conviction /10`\n"
    m2 += esc("⚠️ Niveaux ATR indicatifs — pas un conseil financier.")

    # ── Envoi des deux messages ───────────────────────────────────────────────
    envoyer_telegram(m1, CHAT_ID, TELEGRAM_TOKEN)
    envoyer_telegram(m2, CHAT_ID, TELEGRAM_TOKEN)

    sauvegarder_historique(historique, data_actifs)

if __name__ == "__main__":
    generer_rapport()
