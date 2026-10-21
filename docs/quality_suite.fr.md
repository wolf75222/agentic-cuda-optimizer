# Étude comparative section 16

Le protocole par défaut contient 126 générations : six familles, trois graines indépendantes et sept méthodes (`full`, `true_upstream`, `native_codex_vanilla`, `no_autotuning`, `no_research`, `no_pmpp`, `pmpp_permanent_only`). `--include-greedy` ajoute 18 expériences de sélection du parent ; cette variante ne remplace jamais le véritable agent amont.

Préparer le protocole ne soumet aucun job :

```sh
.venv/bin/python scripts/run_quality_suite.py plan \
  --config configs/quality-romeo-v2.json --directory results/quality-study-v2 --include-greedy
.venv/bin/python scripts/run_quality_suite.py run \
  --directory results/quality-study-v2 --concurrency 4
.venv/bin/python scripts/run_quality_suite.py status --directory results/quality-study-v2
.venv/bin/python scripts/run_quality_suite.py report --directory results/quality-study-v2
```

Les travailleurs GPU indépendants sont au nombre de 2 à 4. Chaque processus utilise sa propre allocation et ses propres modèles, corpus, caches de résultats et fichiers LangGraph. Le nombre de travailleurs est un réglage de concurrence. Aucun plafond global de tokens, coût, appels ou durée cumulée n’est ajouté. Les itérations, réparations, configurations locales et paramètres de mesure sont fixés par le fichier de configuration. Les consommations effectivement observées peuvent différer entre méthodes ; le rapport ne prétend pas égaliser leurs coûts.

Toutes les méthodes reçoivent le même domaine fixe : tailles de recherche complétées par 1, 17 et 63. L’original conserve son tableau exact de lancements par taille. Aucun extrapolateur n’est ajouté à son algorithme. Chaque famille et graine dispose d’une graine finale privée engagée par SHA-256 avant la génération. Seule son empreinte est publiée avant la fermeture des générateurs. Les données finales sont créées après le gel des sources de toutes les méthodes, sur ces mêmes tailles avec de nouvelles entrées, conformément à la section 4.3.

Un groupe final ouvre une nouvelle allocation et y évalue le noyau naïf et tous les artefacts figés. L’oracle CPU est indépendant. Tous les candidats passent d’abord la correction et les quatre outils Compute Sanitizer sur tous les cas finaux ; seules les sorties qualifiées sont chronométrées. Les confirmations AB/BA et leur bootstrap suivent les paramètres préengagés. Les comparaisons directes `full` contre l’original, Codex vanilla et chaque ablation sont également fixées avant génération. Les résultats intermédiaires de chaque génération sont conservés mais exclus des gains finaux section 16. Les temps bruts d’allocations différentes ne sont pas mélangés.

`suite.sqlite` conserve les intentions avant soumission, les identités de session, les graines privées, les états et les résultats. Ce fichier doit rester hors des espaces accessibles aux générateurs et conserver ses permissions privées. Une reprise utilise la même session ; elle ne soumet pas un nouveau job lorsqu’un worker est encore actif. Des verrous par tâche protègent aussi les processus survivant à l’arrêt du contrôleur. Les résultats terminés sont authentifiés par leurs empreintes et leur reçu terminal avant réutilisation. Un artefact modifié est refusé.

Les opérations de reconnexion et la reprise vérifiée avant le premier appel
modèle sont détaillées dans [Reprise après interruption réseau](recovery_transport.fr.md).

Une panne ambiguë du modèle vanilla ou une finalisation interrompue après révélation des entrées reste `recovery_required`. Le programme ne relance pas secrètement le prompt ou le test final pour obtenir un meilleur résultat. Le reçu partiel, les coûts et les diagnostics restent consultables. La commande `run` ne transforme pas automatiquement cet état en nouvelle expérience.

Pour un groupe final incomplet à cause d’un incident de transport enregistré ou d’un terminal worker manquant, une récupération explicite est disponible :

```sh
.venv/bin/python scripts/run_quality_suite.py recover-final \
  --directory results/quality-study-v2 --task IDENTIFIANT_GROUPE \
  --reason "Incident SSH documenté dans le reçu de la tentative précédente"
.venv/bin/python scripts/run_quality_suite.py run --directory results/quality-study-v2 --concurrency 4
```

Cette commande vérifie le terminal de l’ancien job et les sources figées, puis engage une nouvelle graine, une nouvelle session et un nouveau dossier de tentative. Elle n’appelle aucun modèle et ne soumet aucun job elle-même. Toutes les anciennes décisions, pertes incluses, et leurs coûts restent dans `previous_attempts` et les archives ; les chronométrages ne sont pas regroupés entre tentatives. Le nouveau groupe complet est un contexte de mesure indépendant. Un groupe déjà complet ou une erreur numérique, de contrat ou de sanitizer ne peut pas être relancé par ce mécanisme.

`status.json`, `report.json` et `rapport-fr.md` distinguent exécution complète et qualification scientifique. Les agrégats incluent les régressions mesurées et les retours au noyau initial. Les échecs et inconnus conservent leur dénominateur ; aucun gain inventé ne leur est attribué. Les coûts CUDA Events, compilation, cache, temps réservé, temps total incluant la file d’attente, tokens/appels et recherche sont séparés. Les métriques qu’une méthode n’enregistre pas sont explicitement inconnues.

La matrice des exigences ne déduit pas une exécution GPU de la présence d’un fichier source. Pour renseigner une preuve, fournir `--requirement-evidence evidence.json` avec les identifiants `USER-LOCAL`, `USER-HPC`, `USER-PROFILE`, `SEC-1` à `SEC-19` et leurs sous-sections, ainsi que `AC-01` à `AC-20` :

```json
{"AC-08":{"implemented":{"ok":true,"scope":"Gardes de correction","evidence":["optimizer_agent/evaluation.py"]},"local_tested":{"ok":true,"scope":"Scénarios nommés dans le reçu pytest","evidence":["receipts/tests.json"]},"romeo_executed":{"ok":false,"scope":"Essai GPU encore en attente","evidence":[]}}}
```

Chaque phase reste indépendante. Un protocole préparé ou un test simulé ne prouve ni un gain GPU, ni la fin des 144 campagnes avec ablation gloutonne.

Le contraste `full_vs_pmpp_permanent_only` isole l’ajout de consultation ciblée au guide original permanent. `pmpp_permanent_only_vs_no_pmpp` isole le guide permanent. Le mode permanent seul ne consulte ni cartes ni sources externes PMPP et ne prétend pas l’avoir fait.
