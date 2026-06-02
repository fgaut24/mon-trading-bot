"""
backtest_bourse.py — Validation Oracle ML vs Buy & Hold
=======================================================
Teste si la stratégie Random Forest + conviction surperforme un simple
investissement Buy & Hold sur 4 ans, sans biais d'anticipation (look-ahead).

Usage :
  python backtest_bourse.py                     → tous les actifs
  python backtest_bourse.py --ticker MC.PA      → actif unique
  python backtest_bourse.py --periode 2y        → période personnalisée
  python backtest_bourse.py --vider-cache       → force le retéléchargement

Cache : les données sont mises en cache localement dans le dossier cache/.
Une fois téléchargées, les relances ne sollicitent plus Yahoo Finance
(jusqu'à expiration du cache, 12h par défaut). Idéal pour tester plusieurs
variantes de stratégie sur les mêmes données sans déclencher le rate limit.

Installation des dépendances :
  pip install yfinance pandas numpy scikit-learn
  pip install vectorbt   (optionnel — simulation de portefeuille avancée)
  pip install matplotlib (optionnel — graphique de la courbe de valeur)
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

# ── Frais & fiscalité ─────────────────────────────────────────────────────────
# Frais de courtage + spread estimés par transaction (aller OU retour).
# 0,15 % est une estimation réaliste pour un courtier en ligne européen
# (un aller-retour coûte donc ~0,30 %). À ajuster selon votre courtier.
FRAIS_TRANSACTION = 0.0015   # 0,15 % par transaction

# Fiscalité : un PEA exonère d'impôt sur les plus-values après 5 ans
# (hors prélèvements sociaux de 17,2 %). Hors PEA (compte-titres ordinaire),
# la flat tax est de 30 %. Le backtest applique le régime PEA par défaut
# car les actifs surveillés y sont éligibles.
TAUX_PRELEV_SOCIAUX = 0.172  # 17,2 % sur la plus-value nette (PEA > 5 ans)
TAUX_FLAT_TAX       = 0.30   # 30 % hors PEA (référence comparative)

JOURS_BOURSE_AN = 252        # nombre de séances par an (annualisation)
TAUX_SANS_RISQUE = 0.03      # 3 % — proxy du taux sans risque pour Sharpe

# ETF de référence pour la comparaison passive systématique.
# CW8 (Amundi MSCI World) a le plus long historique disponible.
ETF_REFERENCE = "CW8.PA"
ETF_REFERENCE_NOM = "ETF MSCI World (CW8)"

# ── Cache local des données ───────────────────────────────────────────────────
# Pour éviter le rate limit de Yahoo Finance, chaque actif n'est téléchargé
# qu'une seule fois et stocké en CSV. Les exécutions suivantes lisent le cache.
# Indispensable pour la Phase 3 (tester plusieurs variantes sur les mêmes données
# sans retoucher Yahoo). Le cache expire après CACHE_VALIDITE_HEURES.
CACHE_DIR = "cache"
CACHE_VALIDITE_HEURES = 12   # au-delà, on retélécharge (données du jour fraîches)
DELAI_ENTRE_REQUETES  = 2.0  # secondes de pause entre deux téléchargements Yahoo
MAX_REESSAIS          = 3    # nombre de tentatives en cas d'échec réseau

# Features ML — exactement les mêmes 7 que screener_pro.py
FEATURES = ["RSI", "MACD_Hist", "ATR",
            "Price_52W_Pct", "SMA50_vs_SMA200", "BB_Width", "Var_1M"]

# ==============================================================================
# CACHE LOCAL DES DONNÉES — évite le rate limit Yahoo Finance
# ==============================================================================

def _chemin_cache(symbol: str) -> str:
    """Chemin du fichier cache pour un ticker (les '.' deviennent '_')."""
    nom = symbol.replace(".", "_").replace("^", "INDEX_")
    return os.path.join(CACHE_DIR, f"{nom}.csv")


def _cache_valide(chemin: str) -> bool:
    """Vrai si le fichier cache existe et n'est pas trop ancien."""
    if not os.path.exists(chemin):
        return False
    age_heures = (time.time() - os.path.getmtime(chemin)) / 3600
    return age_heures < CACHE_VALIDITE_HEURES


def telecharger_avec_cache(symbol: str, periode: str = "4y",
                           start=None, end=None) -> pd.DataFrame:
    """
    Télécharge l'historique d'un actif via Yahoo Finance, avec cache local.

    1. Si un cache valide existe, il est lu directement (aucune requête réseau).
    2. Sinon, téléchargement avec pause + réessais en cas d'échec (rate limit).
    3. Le résultat est sauvegardé en CSV pour les exécutions suivantes.

    Si start/end sont fournis (ex. ETF de référence), on lit le cache complet
    de la période puis on filtre sur l'intervalle demandé — sans requête réseau
    supplémentaire si le cache couvre déjà la période.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    chemin = _chemin_cache(symbol)

    def _filtrer(df):
        """Filtre le DataFrame sur [start, end] si demandé, en gérant le tz."""
        if start is None or end is None or df.empty:
            return df
        # S'assurer que l'index est bien un DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        ts_start = pd.Timestamp(start)
        ts_end   = pd.Timestamp(end)
        tz = getattr(df.index, "tz", None)
        if tz is not None:
            if ts_start.tz is None: ts_start = ts_start.tz_localize(tz)
            if ts_end.tz   is None: ts_end   = ts_end.tz_localize(tz)
        else:
            if ts_start.tz is not None: ts_start = ts_start.tz_localize(None)
            if ts_end.tz   is not None: ts_end   = ts_end.tz_localize(None)
        return df.loc[(df.index >= ts_start) & (df.index <= ts_end)]

    # Lecture du cache si valide
    if _cache_valide(chemin):
        try:
            df = pd.read_csv(chemin, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[df.index.notna()]
            if not df.empty:
                print(f"  📁 Cache : {symbol} ({len(df)} séances, lu localement)")
                return _filtrer(df)
        except Exception as e:
            print(f"  Cache illisible pour {symbol} ({e}), retéléchargement…")

    # Téléchargement avec réessais (toujours la période complète, pour le cache)
    for tentative in range(1, MAX_REESSAIS + 1):
        try:
            time.sleep(DELAI_ENTRE_REQUETES)
            df = yf.Ticker(symbol).history(period=periode)
            if not df.empty:
                df.to_csv(chemin)
                print(f"  🌐 Téléchargé : {symbol} ({len(df)} séances) → cache écrit")
                return _filtrer(df)
            else:
                print(f"  Tentative {tentative}/{MAX_REESSAIS} : réponse vide "
                      f"(rate limit probable)…")
        except Exception as e:
            print(f"  Tentative {tentative}/{MAX_REESSAIS} échouée : {e}")
        if tentative < MAX_REESSAIS:
            attente = DELAI_ENTRE_REQUETES * (tentative + 1) * 2
            print(f"  Pause {attente:.0f}s avant réessai…")
            time.sleep(attente)

    print(f"  ❌ {symbol} : échec après {MAX_REESSAIS} tentatives (rate limit Yahoo).")
    return pd.DataFrame()


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

    # Régime de marché (proxy local sans look-ahead) : prix au-dessus de sa
    # propre SMA200 = tendance de fond haussière. On utilise la SMA200 de
    # l'actif lui-même, déjà calculée à chaque date sans information future.
    df_out["is_bull"] = (df_out["Close"] > df_out["SMA200"]).astype(bool)

    # ATR conservé pour le trailing stop (variante B)
    cols = ["Close", "RSI", "prob_up", "conviction", "is_bull"]
    if "ATR" in df_out.columns:
        cols.append("ATR")
    return df_out[cols].copy()

# ==============================================================================
# SIMULATION DE PORTEFEUILLE
# ==============================================================================

def metriques_avancees(courbe: pd.Series, trades: list = None) -> dict:
    """
    Calcule les métriques de risque-rendement à partir d'une courbe de valeur.

    - Sharpe  : rendement excédentaire / volatilité totale (annualisés)
    - Sortino : rendement excédentaire / volatilité des seules baisses
                (pénalise uniquement le risque baissier, plus pertinent)
    - Volatilité annualisée
    - Durée moyenne de détention (si trades fournis)

    Le ratio de Sharpe mesure le rendement obtenu par unité de risque total.
    Le ratio de Sortino raffine en ne comptant que la volatilité négative :
    deux stratégies au même rendement mais l'une régulière et l'autre en
    dents de scie auront des Sortino très différents.
    """
    rendements = courbe.pct_change().dropna()
    if len(rendements) < 2 or rendements.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "volatilite": 0.0, "duree_moy": 0.0}

    # Annualisation
    rdt_moy_annuel = rendements.mean() * JOURS_BOURSE_AN
    vol_annuelle   = rendements.std() * np.sqrt(JOURS_BOURSE_AN)

    sharpe = (rdt_moy_annuel - TAUX_SANS_RISQUE) / vol_annuelle if vol_annuelle > 0 else 0.0

    # Sortino : volatilité des seuls rendements négatifs
    rdts_neg = rendements[rendements < 0]
    downside_vol = rdts_neg.std() * np.sqrt(JOURS_BOURSE_AN) if len(rdts_neg) > 1 else 0.0
    sortino = (rdt_moy_annuel - TAUX_SANS_RISQUE) / downside_vol if downside_vol > 0 else 0.0

    # Durée moyenne de détention (en jours calendaires)
    duree_moy = 0.0
    if trades:
        durees = [(t[2] - t[0]).days for t in trades if t[2] and t[0]]
        duree_moy = np.mean(durees) if durees else 0.0

    return {
        "sharpe":     sharpe,
        "sortino":    sortino,
        "volatilite": vol_annuelle * 100,
        "duree_moy":  duree_moy,
    }


def appliquer_fiscalite(capital_final: float, capital_initial: float,
                        regime: str = "PEA") -> dict:
    """
    Applique la fiscalité sur la plus-value nette.

    - PEA (> 5 ans)  : exonération d'IR, seuls les prélèvements sociaux (17,2 %)
    - CTO (flat tax) : 30 % sur la plus-value
    Si perte, aucun impôt (et report déficitaire non modélisé ici).
    """
    plus_value = capital_final - capital_initial
    if plus_value <= 0:
        return {"net": capital_final, "impot": 0.0, "regime": regime}

    taux = TAUX_PRELEV_SOCIAUX if regime == "PEA" else TAUX_FLAT_TAX
    impot = plus_value * taux
    return {"net": capital_final - impot, "impot": impot, "regime": regime}


def simuler_oracle(df_sig: pd.DataFrame, options: dict = None) -> dict:
    """
    Simule la stratégie avec des VARIANTES activables via `options` :

      options = {
        "filtre_regime":  False,  # Hypothèse A : ne pas vendre sur RSI>75
                                  #   si is_bull (marché haussier) ET conviction>=5
        "trailing_stop":  False,  # Hypothèse B : stop suiveur ATR au lieu de
                                  #   la sortie binaire conviction/RSI
        "trailing_mult":  1.5,    # multiplicateur ATR du stop suiveur
      }

    Sans options (ou toutes à False), reproduit la stratégie de base v13.0 :
      achat si conviction > SEUIL_ACHAT,
      vente si conviction < SEUIL_VENTE OU RSI > RSI_SURACHAT.

    Note : VectorBT n'est utilisé que pour la stratégie de base (sans options),
    car les variantes nécessitent une logique conditionnelle fine jour par jour.
    """
    if options is None:
        options = {}
    filtre_regime = options.get("filtre_regime", False)
    trailing_stop = options.get("trailing_stop", False)
    trailing_mult = options.get("trailing_mult", 1.5)
    aucune_option = not (filtre_regime or trailing_stop)

    if VBT_AVAILABLE and aucune_option:
        entries = df_sig["conviction"] > SEUIL_ACHAT
        exits   = (df_sig["conviction"] < SEUIL_VENTE) | (df_sig["RSI"] > RSI_SURACHAT)
        pf = vbt.Portfolio.from_signals(
            df_sig["Close"], entries, exits,
            init_cash=CAPITAL_INIT, freq="1D"
        )
        stats = pf.stats()
        metr  = metriques_avancees(pf.value())
        return {
            "capital_final":    CAPITAL_INIT * (1 + pf.total_return()),
            "rendement_total":  pf.total_return() * 100,
            "drawdown_max":     pf.max_drawdown() * 100,
            "win_rate":         float(stats.get("Win Rate [%]", 0)),
            "nb_trades":        int(pf.trades.count()),
            "frais_totaux":     0.0,
            "sharpe":           metr["sharpe"],
            "sortino":          metr["sortino"],
            "volatilite":       metr["volatilite"],
            "duree_moy":        0.0,
            "courbe":           pf.value(),
            "trades":           [],
        }

    # ── Simulation pandas (gère les variantes) ───────────────────────────────
    capital      = CAPITAL_INIT
    en_position  = False
    nb_actions   = 0.0
    prix_entree  = 0.0
    date_entree  = None
    stop_suiveur = 0.0     # niveau du trailing stop (variante B)
    trades       = []
    courbe       = []
    frais_totaux = 0.0
    a_atr        = "ATR" in df_sig.columns

    for date, row in df_sig.iterrows():
        prix = row["Close"]
        conv = row["conviction"]
        rsi  = row["RSI"]
        bull = bool(row.get("is_bull", False))
        atr  = row["ATR"] if a_atr else prix * 0.02  # fallback ATR ~2%

        # ── ENTRÉE ───────────────────────────────────────────────────────────
        if not en_position and conv > SEUIL_ACHAT:
            frais = capital * FRAIS_TRANSACTION
            frais_totaux += frais
            nb_actions  = (capital - frais) / prix
            prix_entree = prix
            date_entree = date
            en_position = True
            stop_suiveur = prix - trailing_mult * atr   # init stop suiveur

        # ── GESTION DE POSITION ──────────────────────────────────────────────
        elif en_position:
            vendre = False

            if trailing_stop:
                # Variante B : le stop monte avec le prix, ne descend jamais
                stop_suiveur = max(stop_suiveur, prix - trailing_mult * atr)
                # Sortie si le prix casse le stop suiveur
                if prix <= stop_suiveur:
                    vendre = True
                # Sortie aussi si conviction s'effondre (sécurité)
                elif conv < SEUIL_VENTE:
                    vendre = True
            else:
                # Logique binaire de base
                signal_sortie = (conv < SEUIL_VENTE) or (rsi > RSI_SURACHAT)
                # Variante A : ignorer la sortie RSI>75 si marché haussier + conviction OK
                if filtre_regime and rsi > RSI_SURACHAT and bull and conv >= 5:
                    signal_sortie = (conv < SEUIL_VENTE)  # on ne garde que la sortie conviction
                vendre = signal_sortie

            if vendre:
                brut  = nb_actions * prix
                frais = brut * FRAIS_TRANSACTION
                frais_totaux += frais
                capital = brut - frais
                trades.append((date_entree, prix_entree, date, prix,
                               (prix - prix_entree) / prix_entree * 100))
                nb_actions  = 0.0
                en_position = False

        valeur = nb_actions * prix if en_position else capital
        courbe.append(valeur)

    # Clôture finale
    if en_position:
        prix_fin = df_sig["Close"].iloc[-1]
        brut     = nb_actions * prix_fin
        frais    = brut * FRAIS_TRANSACTION
        frais_totaux += frais
        capital  = brut - frais
        trades.append((date_entree, prix_entree, df_sig.index[-1], prix_fin,
                        (prix_fin - prix_entree) / prix_entree * 100))
        courbe[-1] = capital

    courbe_s = pd.Series(courbe, index=df_sig.index)
    peak     = courbe_s.cummax()
    dd_max   = ((courbe_s - peak) / peak * 100).min()

    nb_win = sum(1 for t in trades if t[4] > 0)
    metr   = metriques_avancees(courbe_s, trades)
    return {
        "capital_final":    capital,
        "rendement_total":  (capital / CAPITAL_INIT - 1) * 100,
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
    """Buy & Hold : achat dès le 1er jour, conservation jusqu'à la fin."""
    p0    = df_sig["Close"].iloc[0]
    pn    = df_sig["Close"].iloc[-1]
    # Un seul aller-retour de frais (achat au début, vente à la fin)
    cap_apres_achat = CAPITAL_INIT * (1 - FRAIS_TRANSACTION)
    nb_actions = cap_apres_achat / p0
    cap_brut   = nb_actions * pn
    cap_f      = cap_brut * (1 - FRAIS_TRANSACTION)
    courbe = nb_actions * df_sig["Close"]
    peak   = courbe.cummax()
    dd_max = ((courbe - peak) / peak * 100).min()
    metr   = metriques_avancees(courbe)
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
    """
    Simule un Buy & Hold sur l'ETF de référence (MSCI World) sur la même
    période, pour comparer systématiquement à un investissement passif
    diversifié plutôt qu'au seul actif analysé.
    Retourne None si l'historique de l'ETF ne couvre pas la période.
    """
    try:
        df = telecharger_avec_cache(ETF_REFERENCE, periode="4y", start=date_debut, end=date_fin)
        if df.empty or len(df) < 30:
            return None
        p0, pn = df["Close"].iloc[0], df["Close"].iloc[-1]
        cap_apres_achat = CAPITAL_INIT * (1 - FRAIS_TRANSACTION)
        nb = cap_apres_achat / p0
        cap_f = (nb * pn) * (1 - FRAIS_TRANSACTION)
        courbe = nb * df["Close"]
        peak = courbe.cummax()
        dd = ((courbe - peak) / peak * 100).min()
        metr = metriques_avancees(courbe)
        return {
            "capital_final":   cap_f,
            "rendement_total": (cap_f / CAPITAL_INIT - 1) * 100,
            "drawdown_max":    dd,
            "sharpe":          metr["sharpe"],
            "sortino":         metr["sortino"],
        }
    except Exception as e:
        print(f"  Avertissement ETF référence : {e}")
        return None

# ==============================================================================
# AFFICHAGE
# ==============================================================================

def afficher_tableau(nom, res_ml, res_bh, tscv_info,
                     date_debut, date_fin, res_etf=None) -> None:
    annees     = (date_fin - date_debut).days / 365.25
    rend_an_ml = ((1 + res_ml["rendement_total"] / 100) ** (1 / annees) - 1) * 100
    rend_an_bh = ((1 + res_bh["rendement_total"] / 100) ** (1 / annees) - 1) * 100

    # Fiscalité PEA appliquée aux capitaux finaux
    fisc_ml = appliquer_fiscalite(res_ml["capital_final"], CAPITAL_INIT, "PEA")
    fisc_bh = appliquer_fiscalite(res_bh["capital_final"], CAPITAL_INIT, "PEA")

    L = 70
    print(f"\n{'═' * L}")
    print(f"  {nom}")
    print(f"  {date_debut.date()} → {date_fin.date()}  "
          f"({annees:.1f} ans · TSCV accuracy : "
          f"{tscv_info['accuracy_moyenne']:.1f}% ± {tscv_info['accuracy_ecart']:.1f}%)")
    print(f"{'═' * L}")
    print(f"  {'Métrique':<34} {'Stratégie':>14} {'Buy & Hold':>14}")
    print(f"  {'─' * 64}")
    print(f"  {'Rendement total (brut)':<34} {res_ml['rendement_total']:>+13.1f}%"
          f" {res_bh['rendement_total']:>+13.1f}%")
    print(f"  {'Rendement annualisé':<34} {rend_an_ml:>+13.1f}%"
          f" {rend_an_bh:>+13.1f}%")
    print(f"  {'Drawdown maximum':<34} {res_ml['drawdown_max']:>+13.1f}%"
          f" {res_bh['drawdown_max']:>+13.1f}%")
    print(f"  {'Volatilité annualisée':<34} {res_ml.get('volatilite',0):>13.1f}%"
          f" {res_bh.get('volatilite',0):>13.1f}%")
    print(f"  {'Ratio de Sharpe':<34} {res_ml.get('sharpe',0):>14.2f}"
          f" {res_bh.get('sharpe',0):>14.2f}")
    print(f"  {'Ratio de Sortino':<34} {res_ml.get('sortino',0):>14.2f}"
          f" {res_bh.get('sortino',0):>14.2f}")
    print(f"  {'─' * 64}")
    print(f"  {'Win Rate':<34} {res_ml['win_rate']:>13.1f}%   {'—':>13}")
    print(f"  {'Nombre de trades':<34} {res_ml['nb_trades']:>14}   {'—':>13}")
    print(f"  {'Durée moy. détention (jours)':<34} {res_ml.get('duree_moy',0):>14.0f}   {'—':>13}")
    print(f"  {'Frais de transaction cumulés':<34} {res_ml.get('frais_totaux',0):>12.0f} €   {'~30 €':>13}")
    print(f"  {'─' * 64}")
    print(f"  Après fiscalité PEA (prélèvements sociaux 17,2 % sur plus-value) :")
    print(f"  {'Capital net final':<34} {fisc_ml['net']:>12.0f} €   {fisc_bh['net']:>11.0f} €")
    print(f"  {'Impôt (prélèvements sociaux)':<34} {fisc_ml['impot']:>12.0f} €   {fisc_bh['impot']:>11.0f} €")

    # Comparaison ETF World de référence
    if res_etf:
        print(f"  {'─' * 64}")
        print(f"  Référence passive — {ETF_REFERENCE_NOM} sur la même période :")
        print(f"  {'Rendement total':<34} {res_etf['rendement_total']:>+13.1f}%")
        print(f"  {'Drawdown maximum':<34} {res_etf['drawdown_max']:>+13.1f}%")
        print(f"  {'Ratio de Sharpe':<34} {res_etf.get('sharpe',0):>14.2f}")

    ecart = res_ml["rendement_total"] - res_bh["rendement_total"]
    if   ecart >  5: verdict = f"✅ La stratégie surperforme Buy & Hold de {ecart:+.1f}%"
    elif ecart < -5: verdict = f"⚠️  Buy & Hold surperforme la stratégie de {-ecart:.1f}%"
    else:            verdict = f"⚪  Performances similaires (écart {ecart:+.1f}%)"
    print(f"\n  {verdict}")

    # Verdict Sharpe (qualité du rendement ajusté du risque)
    s_ml, s_bh = res_ml.get("sharpe",0), res_bh.get("sharpe",0)
    if s_ml > s_bh + 0.1:
        print(f"  📊 Meilleur rendement ajusté du risque (Sharpe {s_ml:.2f} vs {s_bh:.2f})")
    elif s_bh > s_ml + 0.1:
        print(f"  📊 Buy & Hold offre un meilleur rendement ajusté du risque "
              f"(Sharpe {s_bh:.2f} vs {s_ml:.2f})")
    print(f"{'═' * L}")

    # Détail des trades
    if res_ml["trades"]:
        print(f"\n  Trades de la stratégie ({res_ml['nb_trades']}) :")
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
        df_raw = telecharger_avec_cache(symbol, periode=periode)
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

    # 3bis. ETF de référence passive sur la même période (sauf si on backteste l'ETF lui-même)
    res_etf = None
    if symbol != ETF_REFERENCE:
        res_etf = simuler_etf_reference(
            df_signaux.index[0].to_pydatetime(),
            df_signaux.index[-1].to_pydatetime()
        )

    # 4. Affichage + graphique
    afficher_tableau(
        nom       = f"{nom} ({symbol})",
        res_ml    = res_ml,
        res_bh    = res_bh,
        tscv_info = tscv_info,
        date_debut = df_signaux.index[0].to_pydatetime(),
        date_fin   = df_signaux.index[-1].to_pydatetime(),
        res_etf   = res_etf,
    )
    tracer_courbes(nom, res_ml, res_bh)


def comparer_variantes(symbol: str, nom: str, periode: str = "4y",
                       split_ratio: float = 0.5) -> dict:
    """
    PHASE 3 — Compare rigoureusement les variantes de stratégie avec
    split temporel verrouillé (anti-curve-fitting).

    Les données sont scindées en deux :
      - ENTRAÎNEMENT (première moitié) : exploration, on regarde les résultats
      - TEST (seconde moitié) : validation finale, regardée une seule fois

    Variantes comparées :
      - BASE       : stratégie v13.0 (sortie binaire conviction/RSI)
      - +RÉGIME    : ne vend pas sur RSI>75 si marché haussier (hypothèse A)
      - +TRAILING  : stop suiveur ATR au lieu de la sortie binaire (hypothèse B)
      - +RÉGIME+TR : combinaison des deux

    Référence : Buy & Hold sur la même période.
    """
    print(f"\n{'━' * 70}")
    print(f"  COMPARAISON DE VARIANTES — {nom} ({symbol})")
    print(f"{'━' * 70}")

    df_raw = telecharger_avec_cache(symbol, periode=periode)
    if df_raw.empty or len(df_raw) < MIN_TRAIN_JOURS + 150:
        print(f"  ❌ Données insuffisantes pour {symbol}")
        return {}

    df = calculer_indicateurs(df_raw)
    df_clean = df.dropna(subset=FEATURES + ["Target"]).copy()
    df_sig = walk_forward_signaux(df_clean)
    if len(df_sig) < 100:
        print(f"  ❌ Trop peu de signaux")
        return {}

    # Split temporel
    n = len(df_sig)
    i_split = int(n * split_ratio)
    df_train = df_sig.iloc[:i_split]
    df_test  = df_sig.iloc[i_split:]

    variantes = {
        "BASE":         {},
        "+REGIME":      {"filtre_regime": True},
        "+TRAILING":    {"trailing_stop": True},
        "+REGIME+TR":   {"filtre_regime": True, "trailing_stop": True},
    }

    def _bloc(df_periode, label):
        print(f"\n  ── {label} : {df_periode.index[0].date()} → {df_periode.index[-1].date()} "
              f"({len(df_periode)} séances) ──")
        bh = simuler_buy_hold(df_periode)
        print(f"  {'Variante':<14} {'Rdt':>8} {'DD max':>9} {'Sharpe':>8} "
              f"{'Sortino':>8} {'Trades':>7}")
        print(f"  {'-'*58}")
        print(f"  {'Buy & Hold':<14} {bh['rendement_total']:>+7.1f}% "
              f"{bh['drawdown_max']:>+8.1f}% {bh['sharpe']:>8.2f} {bh['sortino']:>8.2f} {'—':>7}")
        res = {}
        for vlabel, opts in variantes.items():
            r = simuler_oracle(df_periode, opts)
            res[vlabel] = r
            print(f"  {vlabel:<14} {r['rendement_total']:>+7.1f}% "
                  f"{r['drawdown_max']:>+8.1f}% {r['sharpe']:>8.2f} "
                  f"{r['sortino']:>8.2f} {r['nb_trades']:>7}")
        res["BH"] = bh
        return res

    res_train = _bloc(df_train, "ENTRAÎNEMENT (exploration)")
    res_test  = _bloc(df_test,  "TEST (validation finale — regardé une seule fois)")

    return {"train": res_train, "test": res_test, "nom": nom, "symbol": symbol}


if __name__ == "__main__":
    periode = PERIODE
    if "--periode" in sys.argv:
        i = sys.argv.index("--periode")
        periode = sys.argv[i + 1]

    # Option pour forcer le retéléchargement (vide le cache)
    if "--vider-cache" in sys.argv:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print(f"🗑️  Cache vidé ({CACHE_DIR}/).")
        else:
            print("🗑️  Aucun cache à vider.")

    # ── Mode comparaison de variantes (Phase 3) ─────────────────────────────
    if "--variantes" in sys.argv:
        print("═" * 70)
        print("  PHASE 3 — Comparaison de variantes avec split train/test")
        print(f"  Split : 50% entraînement / 50% test  ·  Frais {FRAIS_TRANSACTION*100:.2f}%")
        print("═" * 70)
        if "--ticker" in sys.argv:
            i = sys.argv.index("--ticker")
            sym = sys.argv[i + 1]
            comparer_variantes(sym, TICKERS_BACKTEST.get(sym, sym), periode)
        else:
            for sym, nom in TICKERS_BACKTEST.items():
                comparer_variantes(sym, nom, periode)
        print("\n⚠️  Le jeu de TEST ne doit être regardé qu'une seule fois.")
        print("   Ne pas re-optimiser après l'avoir vu — sinon curve-fitting.")
        sys.exit(0)

    print("═" * 70)
    print("  Backtest — Système d'analyse vs Buy & Hold vs ETF World")
    print(f"  Période : {periode}  ·  "
          f"Achat : conviction > {SEUIL_ACHAT}/10  ·  "
          f"Vente : < {SEUIL_VENTE}/10 ou RSI > {RSI_SURACHAT}")
    print(f"  Frais : {FRAIS_TRANSACTION*100:.2f}%/transaction  ·  "
          f"Fiscalité : PEA (prélèvements sociaux 17,2%)")
    print(f"  Engine : {'VectorBT' if VBT_AVAILABLE else 'pandas'}  ·  "
          f"Graphiques : {'matplotlib' if MPL_AVAILABLE else 'désactivés'}")
    print("═" * 70)

    if "--ticker" in sys.argv:
        i = sys.argv.index("--ticker")
        sym = sys.argv[i + 1]
        backtester_actif(sym, TICKERS_BACKTEST.get(sym, sym), periode)
    else:
        for sym, nom in TICKERS_BACKTEST.items():
            backtester_actif(sym, nom, periode)

    print("\n⚠️  Résultats sur données historiques — les performances passées")
    print("   ne préjugent pas des résultats futurs.")
