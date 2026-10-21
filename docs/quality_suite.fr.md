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

Le même nombre `max_trials=3` ne désigne pas trois évaluations GPU totales. L’original conserve intégralement son workflow au commit `1e9464d` : `max_iterations=3` autorise l’initial fourni à l’itération 0, puis au plus trois propositions aux itérations 1, 2 et 3. Une réparation après échec CUDA consomme une de ces itérations ; un arrêt anticipé peut réduire leur nombre. Sa référence de correction est évaluée séparément. L’optimiseur amélioré mesure lui aussi son noyau initial hors des trois trials TreeQuest, mais peut effectuer jusqu’à deux réparations locales par trial et explorer les configurations d’autotuning déclarées, selon le plan. Un trial suspendu sans candidat ne constitue pas une évaluation CUDA réussie.

Les unités de coût restent donc distinctes : sources candidates, trials, réparations, configurations, requêtes du harness, lancements GPU et appels CLI. Les répétitions et classements de recherche de l’original restent ceux de son algorithme ; ils ne sont pas remplacés par les séries et confirmations de recherche de l’optimiseur amélioré. Le compteur amont `model_calls` est propre à une conversation de proposition ou de planification de lancement ; le total des appels CLI vient des reçus d’invocation dédupliqués. L’observateur indépendant du premier gain, lorsqu’il est activé, ajoute des mesures sans retour vers les décisions de la méthode et conserve ses coûts séparés et combinés. Cette comparaison porte sur les méthodes complètes et leurs coûts observés ; elle n’impose pas une égalité artificielle des ressources consommées.

Toutes les méthodes reçoivent le même domaine fixe : tailles de recherche complétées par 1, 17 et 63. L’original conserve son tableau exact de lancements par taille. Aucun extrapolateur n’est ajouté à son algorithme. Chaque famille et graine dispose d’une graine finale privée engagée par SHA-256 avant la génération. Seule son empreinte est publiée avant la fermeture des générateurs. Les données finales sont créées après le gel des sources de toutes les méthodes, sur ces mêmes tailles avec de nouvelles entrées, conformément à la section 4.3.

L’étude `quality-20261021-v2` utilise les tailles 1, 17, 63, 64, 128, 256 et 512.
Le GEMM porte sur des matrices carrées `n × n` ; les autres opérations conservent
leur contrat à un paramètre `n`, dont la normalisation RMS. Les entrées sont
synthétiques et reproductibles. Les résultats décrivent ces domaines et les
latences kernel mesurées par CUDA Events ; ils ne qualifient pas d’autres formes
ou des charges de production plus grandes. Les contrôles fonctionnels séparés
restent hors des cohortes comparatives.

Un groupe final ouvre une nouvelle allocation et y évalue le noyau naïf et tous les artefacts figés. L’oracle CPU est indépendant. Tous les candidats passent d’abord la correction et les quatre outils Compute Sanitizer sur tous les cas finaux ; seules les sorties qualifiées sont chronométrées. Les confirmations AB/BA et leur bootstrap suivent les paramètres préengagés. Les comparaisons directes `full` contre l’original, Codex vanilla et chaque ablation sont également fixées avant génération. Les résultats intermédiaires de chaque génération sont conservés mais exclus des gains finaux section 16. Les temps bruts d’allocations différentes ne sont pas mélangés.

Les nouvelles invocations partagent une seule qualification entre les copies exactes d’un artefact dans ce même groupe, contrat, environnement, suite, entrées et contexte de compilation. Chaque alias revérifie ses sources, paramètres, lancement, entrées et identité matérielle puis référence les octets du reçu original par SHA-256, avec une empreinte JSON normalisée distincte ; il ne reçoit ni nouvel identifiant d’évaluation ni coût d’exécution inventé. Un refus ou une qualification incomplète restent des refus ou des inconnus pour tous leurs alias. Hors de ce groupe, la qualification conserve son fonctionnement frais. Une comparaison entre artefacts identiques produit `exact_artifact_identity`, sans chronométrage apparié ni intervalle : elle ne transforme pas le bruit en gain. La réutilisation exige une identité complète GPU/compilateur/harness et un source sans préprocesseur ni options de compilation ; sinon les qualifications restent fraîches et l’identité d’exécution reste inconnue (`artifact_identity_context_unproven`), sans gain dérivé de 1. Les comparaisons entre artefacts différents exigent des identifiants candidats distincts pour préserver les deux côtés de la preuve brute. Les anciens reçus et groupes déjà engagés ne sont pas réécrits.

`suite.sqlite` conserve les intentions avant soumission, les identités de session, les graines privées, les états et les résultats. Ce fichier doit rester hors des espaces accessibles aux générateurs et conserver ses permissions privées. Une reprise utilise la même session ; elle ne soumet pas un nouveau job lorsqu’un worker est encore actif. Des verrous par tâche protègent aussi les processus survivant à l’arrêt du contrôleur. Les résultats terminés sont authentifiés par leurs empreintes et leur reçu terminal avant réutilisation. Un artefact modifié est refusé.

Le prochain dispatch privilégie les préfixes de reprise liés à une allocation
possédée, puis les transferts de contexte non encore scoré, les autres reprises
et enfin les tâches neuves. Cette priorité repose sur les reçus locaux liés par
empreintes ; elle ne prétend pas que le job est toujours actif. Les gardes du
worker vérifient encore l'allocation et chaque preuve avant réutilisation.
L'identité et l'état du journal sont relus sous le verrou de tâche : une ancienne
future ne relance pas une tâche désormais terminée, en échec ou changée de session.

Les nouveaux dispatchs conservent des observations de provenance dans
`dispatch-attestations/` : contrôleur avant création du pool, puis tâche avant
toute action modèle, SSH ou Slurm. Les epochs sont publiées sans écrasement par
l'API, avec chaîne de hashes vérifiée ; elles conservent PID, identité de tâche,
hash du payload, HEAD/dirty et empreintes des sources observées sur disque.
Elles n'attestent ni le bytecode en mémoire ni les poids du fournisseur.
Une reprise avec une autre version ajoute une epoch ; les exécutions anciennes
sans reçu ne reçoivent pas de provenance inventée. Le hash du payload est publié
sans son contenu ni les graines privées. Cette chaîne locale n'est pas une ancre
externe contre un acteur capable de réécrire le répertoire.

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

Le contrôle autonome `gpu_final_suite_acceptance.py` vérifie le chemin
`Evaluator.finalize()` lorsque SEARCH et FINAL ont des tailles différentes.
Il utilise deux lancements du même GEMM naïf, un oracle indépendant, les quatre
sanitizers et des confirmations AB/BA sur les entrées finales. Ses requêtes
effectives, paramètres et fichiers d’entrée sont confrontés aux reçus finaux.
Ses mesures restent hors des cohortes et ne démontrent aucun gain d’optimiseur.

```bash
# Préparer le plan localement, sans SSH, GPU ou modèle.
.venv/bin/python scripts/gpu_final_suite_acceptance.py \
  --output results/final-suite-control --run-id final-suite-control
# Exécuter ce contrôle dans sa propre allocation ROMEO.
.venv/bin/python scripts/gpu_final_suite_acceptance.py --run \
  --output results/final-suite-control --run-id final-suite-control \
  --execution-config configs/romeo.json
```

Une intention d’allocation existante interdit une deuxième soumission depuis
la même sortie. Après interruption, conserver cette sortie et réconcilier la
session enregistrée avant toute autre action. La présence du plan ou de ses
tests CPU ne constitue pas une preuve d’exécution ROMEO ; celle-ci exige le
reçu `final-suite-acceptance.json`, ses requêtes natives et le terminal Slurm.

Chaque phase reste indépendante. Un protocole préparé ou un test simulé ne prouve ni un gain GPU, ni la fin des 144 campagnes avec ablation gloutonne.

Le contraste `full_vs_pmpp_permanent_only` isole l’ajout de consultation ciblée au guide original permanent. `pmpp_permanent_only_vs_no_pmpp` isole le guide permanent. Le mode permanent seul ne consulte ni cartes ni sources externes PMPP et ne prétend pas l’avoir fait.
