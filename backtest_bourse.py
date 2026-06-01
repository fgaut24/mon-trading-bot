"""
backtest_bourse.py — Validation Oracle ML vs Buy & Hold
=======================================================
Teste si la stratégie Random Forest + conviction surperforme un simple
investissement Buy & Hold sur 4 ans, sans biais d'anticipation (look-ahead).

Usage :
  python backtest_bourse.py                     → tous les actifs
  python backtest_bourse.py --ticker MC.PA      → actif unique
  python backtest_bourse.py --periode 2y        → période personnalisée

Installation des dépendances :
  pip install yfinance pandas numpy scikit-learn
  pip install vectorbt   (optionnel — simulation de portefeuille avancée)
  pip install matplotlib (optionnel — graphique de la courbe de valeur)
"""

import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Dépendances optionnelles
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ==============================================================================
# CONFIGURATION — alignée sur screener_pro.py
# ==============================================================================

TICKERS_BACKTEST = {
    "MC.PA":   "LVMH",
    "TTE.PA":  "TotalEnergies",
    "OR.PA":   "L'Oréal",
    "SAN.PA":  "Sanofi",
    "AI.PA":   "Air Liquide",
    "SU.PA":   "Schneider Elec",
    "WPEA.PA": "ETF MSCI World",
}

PERIODE        = "4y"   # Historique (modifiable via --periode)
MIN_TRAIN_JOURS = 252   # 1 an minimum pour le 1er entraînement
RETRAIN_FREQ    = 20    # Réentraîner tous les 20 jours (~1 mois boursier)
CAPITAL_INIT    = 10_000.0

# Seuils identiques à screener_pro.py
SEUIL_ACHAT    = 6    # conviction > 6/10 → achat
SEUIL_VENTE    = 4    # conviction < 4/10 → vente
RSI_SURACHAT   = 75   # RSI > 75         → vente forcée

# Features ML — exactement les mêmes 7 que screener_pro.py
FEATURES = ["RSI", "MACD_Hist", "ATR",
            "Price_52W_Pct", "SMA50_vs_SMA200", "BB_Width", "Var_1M"]

# ==============================================================================
# CALCUL DES INDICATEURS (réplication fidèle de screener_pro.py)
# ==============================================================================

def calculer_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute tous les indicateurs techniques au DataFrame OHLCV."""
    df = df.copy()

    # RSI 14
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD / Signal / Histogramme
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = exp1 - exp2
    df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    # Bollinger 20
    df["MA20"]     = df["Close"].rolling(20).mean()
    df["STD20"]    = df["Close"].rolling(20).std()
    df["BB_High"]  = df["MA20"] + df["STD20"] * 2
    df["BB_Low"]   = df["MA20"] - df["STD20"] * 2
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["MA20"] * 100

    # SMA 50 / 200
    df["SMA50"]           = df["Close"].rolling(50).mean()
    df["SMA200"]          = df["Close"].rolling(200).mean()
    df["SMA50_vs_SMA200"] = (df["SMA50"] - df["SMA200"]) / df["SMA200"] * 100

    # ATR 14
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # Momentum 52 semaines + 1 mois
    df["H52"] = df["High"].rolling(252).max()
    df["L52"] = df["Low"].rolling(252).min()
    r52       = (df["H52"] - df["L52"]).clip(lower=1e-9)
    df["Price_52W_Pct"] = (df["Close"] - df["L52"]) / r52 * 100
    df["Var_1M"]        = df["Close"].pct_change(20) * 100

    # OBV
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["OBV"]     = (df["Close"].diff().apply(np.sign) * df["Volume"]).fillna(0).cumsum()
        df["OBV_SMA"] = df["OBV"].rolling(20).mean()
        df["OBV_Bull"] = (df["OBV"] > df["OBV_SMA"]).astype(int)
    else:
        df["OBV_Bull"] = 0

    # Cible : le prix sera-t-il plus haut dans 5 jours ?
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    return df


def conviction_vectorisee(rsi, macd_bull, sma_bull, prob_up, obv_bull) -> np.ndarray:
    """
    Calcule le score de conviction (0-10) de façon vectorisée.
    Réplique conviction_score() de screener_pro.py.
    """
    s  = np.zeros(len(rsi))
    s += np.where(rsi <= 35, 3, np.where(rsi <= 45, 2, np.where(rsi <= 55, 1, 0)))
    s += np.where(macd_bull, 2, 0)
    s += np.where(sma_bull, 2, 0)
    s += np.where(prob_up > 55, 2, np.where(prob_up >= 45, 1, 0))
    s += np.where(obv_bull, 1, 0)
    return s

# ==============================================================================
# VALIDATION CROISÉE TEMPORELLE (évaluation du modèle ML)
# ==============================================================================

def evaluer_modele_tscv(df_clean: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Évalue l'accuracy du Random Forest via TimeSeriesSplit.
    Mesure la qualité intrinsèque du modèle, indépendamment de la stratégie.
    """
    X = df_clean[FEATURES]
    y = df_clean["Target"]

    # Retirer les 5 dernières lignes (Target potentiellement incomplet)
    X = X.iloc[:-5]
    y = y.iloc[:-5]

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    scores  = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        model = RandomForestClassifier(n_estimators=50, max_depth=5,
                                        random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        scores.append(acc)

    return {
        "accuracy_moyenne": np.mean(scores) if scores else 0.0,
        "accuracy_ecart":   np.std(scores)  if scores else 0.0,
        "nb_folds":         len(scores),
    }

# ==============================================================================
# WALK-FORWARD BACKTEST (simulation de trading sans look-ahead)
# ==============================================================================

def walk_forward_signaux(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les signaux journaliers sans regarder le futur.

    À chaque date t :
      1. Entraînement sur tout l'historique disponible jusqu'à t-1
         (en excluant les 5 derniers jours dont le Target est incertain)
      2. Prédiction de la probabilité de hausse à t
      3. Calcul de la conviction complète avec les indicateurs à t

    Réentraînement toutes les RETRAIN_FREQ séances pour limiter le temps de calcul.
    """
    n       = len(df_clean)
    probs   = np.full(n, np.nan)
    model   = None

    for i in range(MIN_TRAIN_JOURS, n):
        if (i - MIN_TRAIN_JOURS) % RETRAIN_FREQ == 0:
            # Données d'entraînement = passé strict (sans les 5 derniers jours)
            fin_train = max(0, i - 5)
            X_tr = df_clean[FEATURES].iloc[:fin_train]
            y_tr = df_clean["Target"].iloc[:fin_train]

            if y_tr.nunique() >= 2 and len(y_tr) > 50:
                model = RandomForestClassifier(n_estimators=50, max_depth=5,
                                               random_state=42, n_jobs=-1)
                model.fit(X_tr, y_tr)

        if model is None:
            continue

        classes = list(model.classes_)
        if 1 not in classes:
            continue

        X_j       = df_clean[FEATURES].iloc[i:i+1].fillna(0)
        proba     = model.predict_proba(X_j)[0]
        probs[i]  = proba[classes.index(1)] * 100

    df_out = df_clean.copy()
    df_out["prob_up"] = probs
    df_out = df_out.dropna(subset=["prob_up"])

    # Conviction complète
    df_out["conviction"] = conviction_vectorisee(
        rsi       = df_out["RSI"].values,
        macd_bull = (df_out["MACD"] > df_out["MACD_Sig"]).values,
        sma_bull  = (df_out["Close"] > df_out["SMA200"]).values,
        prob_up   = df_out["prob_up"].values,
        obv_bull  = df_out.get("OBV_Bull", pd.Series(0, index=df_out.index)).values,
    )

    return df_out[["Close", "RSI", "prob_up", "conviction"]].copy()

# ==============================================================================
# SIMULATION DE PORTEFEUILLE
# ==============================================================================

def simuler_oracle(df_sig: pd.DataFrame) -> dict:
    """
    Stratégie Oracle : conviction > SEUIL_ACHAT → achat,
                       conviction < SEUIL_VENTE OU RSI > RSI_SURACHAT → vente.
    Utilise VectorBT si disponible, simulation pandas sinon.
    """
    if VBT_AVAILABLE:
        entries = df_sig["conviction"] > SEUIL_ACHAT
        exits   = (df_sig["conviction"] < SEUIL_VENTE) | (df_sig["RSI"] > RSI_SURACHAT)
        pf = vbt.Portfolio.from_signals(
            df_sig["Close"], entries, exits,
            init_cash=CAPITAL_INIT, freq="1D"
        )
        stats = pf.stats()
        return {
            "capital_final":    CAPITAL_INIT * (1 + pf.total_return()),
            "rendement_total":  pf.total_return() * 100,
            "drawdown_max":     pf.max_drawdown() * 100,
            "win_rate":         float(stats.get("Win Rate [%]", 0)),
            "nb_trades":        int(pf.trades.count()),
            "courbe":           pf.value(),
            "trades":           [],
        }

    # ── Simulation pandas (fallback) ─────────────────────────────────────────
    capital      = CAPITAL_INIT
    en_position  = False
    nb_actions   = 0.0
    prix_entree  = 0.0
    date_entree  = None
    trades       = []
    courbe       = []

    for date, row in df_sig.iterrows():
        prix = row["Close"]
        conv = row["conviction"]
        rsi  = row["RSI"]

        if not en_position and conv > SEUIL_ACHAT:
            nb_actions  = capital / prix
            prix_entree = prix
            date_entree = date
            en_position = True

        elif en_position and (conv < SEUIL_VENTE or rsi > RSI_SURACHAT):
            capital = nb_actions * prix
            gain_pct = (prix - prix_entree) / prix_entree * 100
            trades.append((date_entree, prix_entree, date, prix, gain_pct))
            nb_actions  = 0.0
            en_position = False

        valeur = nb_actions * prix if en_position else capital
        courbe.append(valeur)

    # Fermer la position ouverte à la dernière date
    if en_position:
        prix_fin = df_sig["Close"].iloc[-1]
        capital  = nb_actions * prix_fin
        trades.append((date_entree, prix_entree, df_sig.index[-1], prix_fin,
                        (prix_fin - prix_entree) / prix_entree * 100))
        courbe[-1] = capital

    courbe_s = pd.Series(courbe, index=df_sig.index)
    peak     = courbe_s.cummax()
    dd_max   = ((courbe_s - peak) / peak * 100).min()

    nb_win = sum(1 for t in trades if t[4] > 0)
    return {
        "capital_final":    capital,
        "rendement_total":  (capital / CAPITAL_INIT - 1) * 100,
        "drawdown_max":     dd_max,
        "win_rate":         (nb_win / len(trades) * 100) if trades else 0.0,
        "nb_trades":        len(trades),
        "courbe":           courbe_s,
        "trades":           trades,
    }


def simuler_buy_hold(df_sig: pd.DataFrame) -> dict:
    """Buy & Hold : achat dès le 1er jour, conservation jusqu'à la fin."""
    p0    = df_sig["Close"].iloc[0]
    pn    = df_sig["Close"].iloc[-1]
    cap_f = CAPITAL_INIT * pn / p0
    courbe = CAPITAL_INIT * df_sig["Close"] / p0
    peak   = courbe.cummax()
    dd_max = ((courbe - peak) / peak * 100).min()
    return {
        "capital_final":   cap_f,
        "rendement_total": (cap_f / CAPITAL_INIT - 1) * 100,
        "drawdown_max":    dd_max,
        "courbe":          courbe,
    }

# ==============================================================================
# AFFICHAGE
# ==============================================================================

def afficher_tableau(nom, res_ml, res_bh, tscv_info,
                     date_debut, date_fin) -> None:
    annees     = (date_fin - date_debut).days / 365.25
    rend_an_ml = ((1 + res_ml["rendement_total"] / 100) ** (1 / annees) - 1) * 100
    rend_an_bh = ((1 + res_bh["rendement_total"] / 100) ** (1 / annees) - 1) * 100

    print(f"\n{'═' * 62}")
    print(f"  {nom}")
    print(f"  {date_debut.date()} → {date_fin.date()}  "
          f"({annees:.1f} ans · TSCV accuracy : "
          f"{tscv_info['accuracy_moyenne']:.1f}% ± {tscv_info['accuracy_ecart']:.1f}%)")
    print(f"{'═' * 62}")
    print(f"  {'Métrique':<32} {'Oracle ML':>12} {'Buy & Hold':>12}")
    print(f"  {'─' * 56}")
    print(f"  {'Rendement total':<32} {res_ml['rendement_total']:>+11.1f}%"
          f" {res_bh['rendement_total']:>+11.1f}%")
    print(f"  {'Rendement annualisé':<32} {rend_an_ml:>+11.1f}%"
          f" {rend_an_bh:>+11.1f}%")
    print(f"  {'Drawdown maximum':<32} {res_ml['drawdown_max']:>+11.1f}%"
          f" {res_bh['drawdown_max']:>+11.1f}%")
    print(f"  {'Win Rate':<32} {res_ml['win_rate']:>11.1f}%          —")
    print(f"  {'Nombre de trades':<32} {res_ml['nb_trades']:>12}          —")

    ecart = res_ml["rendement_total"] - res_bh["rendement_total"]
    if   ecart >  5: verdict = f"✅ Oracle surperforme Buy & Hold de {ecart:+.1f}%"
    elif ecart < -5: verdict = f"⚠️  Buy & Hold surperforme Oracle de {-ecart:.1f}%"
    else:            verdict = f"⚪  Performances similaires (écart {ecart:+.1f}%)"
    print(f"\n  {verdict}")
    print(f"{'═' * 62}")

    # Détail des trades
    if res_ml["trades"]:
        print(f"\n  Trades Oracle ({res_ml['nb_trades']}) :")
        for t in res_ml["trades"]:
            signe = "✅" if t[4] > 0 else "❌"
            print(f"    {signe}  {str(t[0].date()):>12} → {str(t[2].date()):>12}"
                  f"  {t[1]:>8.2f} → {t[3]:>8.2f}  ({t[4]:>+6.1f}%)")


def tracer_courbes(nom, res_ml, res_bh) -> None:
    """Affiche la courbe de valeur Oracle vs Buy & Hold (si matplotlib installé)."""
    if not MPL_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    res_bh["courbe"].plot(ax=ax, label="Buy & Hold", color="#888", linestyle="--")
    res_ml["courbe"].plot(ax=ax, label="Oracle ML", color="#2196F3", linewidth=1.5)
    ax.set_title(f"Oracle ML vs Buy & Hold — {nom}")
    ax.set_ylabel("Valeur du portefeuille (€)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"backtest_{nom.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(fname, dpi=120)
    print(f"  📊 Graphique sauvegardé : {fname}")
    plt.close()

# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

def backtester_actif(symbol: str, nom: str, periode: str = PERIODE) -> None:
    print(f"\n⏳  {nom} ({symbol}) — téléchargement {periode}…")

    try:
        df_raw = yf.Ticker(symbol).history(period=periode)
    except Exception as e:
        print(f"  ❌ Impossible de récupérer les données : {e}")
        return

    if df_raw.empty or len(df_raw) < MIN_TRAIN_JOURS + 100:
        print(f"  ❌ Données insuffisantes ({len(df_raw)} lignes)")
        return

    print(f"  {len(df_raw)} séances récupérées → calcul des indicateurs…")
    df = calculer_indicateurs(df_raw)
    df_clean = df.dropna(subset=FEATURES + ["Target"]).copy()

    # 1. Évaluation TSCV (qualité intrinsèque du modèle)
    print(f"  Validation croisée temporelle (TimeSeriesSplit 5 folds)…")
    tscv_info = evaluer_modele_tscv(df_clean)
    print(f"  Accuracy TSCV : {tscv_info['accuracy_moyenne']:.1f}% "
          f"± {tscv_info['accuracy_ecart']:.1f}%")

    # 2. Walk-forward (signaux sans look-ahead)
    print(f"  Walk-forward backtest "
          f"({len(df_clean) - MIN_TRAIN_JOURS} jours prédits)…")
    df_signaux = walk_forward_signaux(df_clean)

    if len(df_signaux) < 50:
        print(f"  ❌ Trop peu de signaux pour {symbol}")
        return

    # 3. Simulation des deux stratégies
    res_ml = simuler_oracle(df_signaux)
    res_bh = simuler_buy_hold(df_signaux)

    # 4. Affichage + graphique
    afficher_tableau(
        nom       = f"{nom} ({symbol})",
        res_ml    = res_ml,
        res_bh    = res_bh,
        tscv_info = tscv_info,
        date_debut = df_signaux.index[0].to_pydatetime(),
        date_fin   = df_signaux.index[-1].to_pydatetime(),
    )
    tracer_courbes(nom, res_ml, res_bh)


if __name__ == "__main__":
    periode = PERIODE
    if "--periode" in sys.argv:
        i = sys.argv.index("--periode")
        periode = sys.argv[i + 1]

    print("═" * 62)
    print("  ORACLE — Backtest Machine Learning vs Buy & Hold")
    print(f"  Période : {periode}  ·  "
          f"Achat : conviction > {SEUIL_ACHAT}/10  ·  "
          f"Vente : < {SEUIL_VENTE}/10 ou RSI > {RSI_SURACHAT}")
    print(f"  Engine : {'VectorBT' if VBT_AVAILABLE else 'pandas (pip install vectorbt pour plus)'}  ·  "
          f"Graphiques : {'matplotlib' if MPL_AVAILABLE else 'désactivés (pip install matplotlib)'}")
    print("═" * 62)

    if "--ticker" in sys.argv:
        i = sys.argv.index("--ticker")
        sym = sys.argv[i + 1]
        backtester_actif(sym, TICKERS_BACKTEST.get(sym, sym), periode)
    else:
        for sym, nom in TICKERS_BACKTEST.items():
            backtester_actif(sym, nom, periode)

    print("\n⚠️  Résultats sur données historiques — les performances passées")
    print("   ne préjugent pas des résultats futurs.")
