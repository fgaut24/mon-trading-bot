# Système d'analyse multi-couches du CAC 40

> **Ce système N'EXÉCUTE AUCUN ORDRE de bourse.** Il fournit de l'information et des signaux d'analyse, jamais des instructions à exécuter automatiquement. Toute décision d'investissement relève de l'utilisateur seul.

---

## À propos

Ce dépôt contient le code d'un système d'analyse boursière automatisé, développé comme expérimentation personnelle au printemps 2026. Il analyse quotidiennement les 47 valeurs du CAC 40 et des ETF PEA éligibles, et envoie un rapport structuré sur Telegram après la clôture des marchés.

L'expérimentation — y compris ses résultats de backtest, ses limites et ses enseignements sur l'efficience des marchés — est documentée dans un article détaillé :

📖 **[J'ai construit un bot de trading et il m'a appris à ne pas l'écouter](https://medium.com/@fgaut24/jai-construit-un-bot-de-trading-et-il-m-a-appris-à-ne-pas-l-écouter-0df66d03be9d)**

---

## Architecture du système

Le système combine quatre couches d'analyse :

| Couche | Indicateurs |
|--------|------------|
| **Technique** | RSI, MACD, Bandes de Bollinger, ATR, OBV, SMA200 |
| **Machine Learning** | Random Forest (7 features, walk-forward backtesting) |
| **Fondamentale** | PER, marge opérationnelle |
| **Macroéconomique** | VIX, taux US 10 ans (4 régimes de marché) |

Chaque actif reçoit un **score de conviction de 0 à 10**, accompagné de niveaux techniques indicatifs calculés depuis la volatilité propre à l'actif (ATR).

---

## Fichiers principaux

```
screener_pro.py        — Système d'analyse quotidien (rapport Telegram)
backtest_bourse.py     — Backtest rigoureux avec split train/test
requirements.txt       — Dépendances Python
.github/workflows/
    bourse.yml         — Exécution automatique quotidienne (18h45 Paris)
    alertes.yml        — Alertes intraday (heures de bourse)
```

---

## Installation locale

```bash
git clone https://github.com/fgaut24/mon-trading-bot.git
cd mon-trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Créez un fichier `.env` à la racine :

```
TELEGRAM_TOKEN=votre_token_bot_telegram
TELEGRAM_CHAT_ID=votre_chat_id
```

Puis lancez :

```bash
python3 screener_pro.py          # rapport complet
python3 screener_pro.py --alertes  # mode alertes intraday
python3 backtest_bourse.py       # backtest de référence
python3 backtest_bourse.py --variantes  # comparaison de variantes (Phase 3)
python3 backtest_bourse.py --vider-cache  # forcer le retéléchargement
```

---

## Résultats du backtest (2024–2026)

Le backtest rigoureux (walk-forward, frais 0,15 %/transaction, fiscalité PEA) sur six valeurs du CAC 40 révèle un comportement asymétrique :

| Actif | Contexte | Stratégie | Buy & Hold | Écart |
|-------|----------|-----------|-----------|-------|
| LVMH | Bear market | −8,9 % | −36,1 % | **+27,2 %** ✅ |
| L'Oréal | Consolidation | +1,3 % | −11,4 % | **+12,7 %** ✅ |
| Sanofi | Consolidation | +8,7 % | −7,4 % | **+16,0 %** ✅ |
| TotalEnergies | Bull market | −15,0 % | +27,0 % | **−41,3 %** ❌ |
| Air Liquide | Bull | −11,2 % | +9,4 % | **−20,5 %** ❌ |
| Schneider Elec. | Bull volatil | −0,2 % | +25,9 % | **−26,1 %** ❌ |

**Référence passive sur la même période : ETF MSCI World +32,8 % (Sharpe 0,83)**

→ La stratégie est défensive sur les marchés baissiers, mais sous-performe significativement sur les marchés haussiers. L'article détaille pourquoi, et ce que cela enseigne sur l'efficience des marchés.

---

## Usage recommandé

Ce système est un **outil de veille**, pas un générateur de décisions d'investissement. Il informe — il ne commande pas. Le backtest démontre empiriquement que la gestion passive (ETF World en DCA) surperforme statistiquement toute approche active testée.

---

## Auteur

**Frédéric Gauthier** — Professeur de Sciences économiques et sociales

⚠️ *Les performances passées ne préjugent pas des résultats futurs. Ce projet ne constitue en aucun cas un conseil en investissement.*
