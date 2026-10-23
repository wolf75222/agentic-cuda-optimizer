# Arrêt et reprise — 23 octobre 2026

Arrêt demandé par l’utilisateur. Aucune nouvelle campagne, suppression, allocation GPU ou invocation modèle n’est autorisée par ce document. Le checkpoint est **WIP**, pas une version qualifiée pour relancer.

## État vérifié à l’arrêt

Checkout local : `work/agentic-cuda-optimizer`, dans le dossier de cette conversation. Version stable déjà publiée sur le fork : `577d20a8b8a7de333309a32dddc0c6ce79104df9`. Sa CI exacte [39757723631](https://github.com/wolf75222/agentic-cuda-optimizer/actions/runs/39757723631) a réussi : 2 282 tests, 13 sous-tests, deux exclusions. Cela ne termine pas l’étude GPU.

Au 23 octobre à 15:09 UTC, **aucun job de rmdraux dans Slurm** et aucun contrôleur de campagne actif. Les agents de développement sont interrompus. Le processus local de protection des 17 FINAL reste volontairement actif : PID observé `4375`, reçu `work/live/hold-finals-only-20261023-ready.json`, marqueur de libération `results/quality-study-v2/release-finals-only-20261023-hold` absent. Vérifier de nouveau PID et verrous avant reprise ; ne pas écraser un ancien reçu.

| Tâches de l’étude existante | État réel |
|---|---|
| 62 générations sur 144 | Terminées au sens du journal ; cela ne garantit ni succès de chaque méthode ni gain utile |
| 78 générations | Préparation de reprise authentifiée et commitée avec le code figé `577d20a`, zéro modèle/GPU/soumission |
| 4 générations | `recovery_required`, détaillées ci-dessous |
| FINAL | 1 groupe terminé sur 18 ; 17 encore en attente et protégés |

Les 78 tâches portent `running` dans SQLite mais **toutes ont `job_id=NULL`** : elles sont préparées, pas en exécution GPU. Ne pas répéter leurs commandes `recover-bootstrap`. Aucun job nouveau n’a été soumis pendant ce dernier lot.

Les quatre cas restants sont :

- `normalization.3141.greedy_parent` : échec EDQUOT pendant l’extraction TAR du worker, avant modèle et avant Slurm. Le parseur strict existant reconnaît seulement l’erreur `mkdir`, donc il a correctement refusé ce cas. Une qualification spécifique du TAR et du contenu partiellement extrait est en chantier.
- `normalization.2718.pmpp_permanent_only`, ancien job `775345` : canonique terminée, confirmation interrompue après 119 paires complètes et une demi-paire.
- `normalization.2718.no_research`, ancien job `775347` : canonique terminée, confirmation interrompue après 28 paires. L’incumbent est le candidat `7a24af29d077d66d32000b82`, version 1, et non la baseline.
- `normalization.2718.true_upstream`, ancien job `775348` : trois itérations de l’original réellement exécutées ; erreur de l’observateur externe. Inspection interne réelle, en lecture seule : checkpoint `next=[finalize_result]`, itération 3/3, historique 0..3, meilleur candidat à l’itération 2, source SHA `85cc10c9d98dcafe08a4d3d66b1f122584f9e8248fe3fa1a1447c70260c0c6cd`. Il reste une finalisation froide puis une observation/validation indépendante de l’export. L’ancien résultat `failed` n’a pas été réécrit.

## Résultats scientifiques acquis

La comparaison complète **agent original / agent amélioré / véritable Codex vanilla Luna Max** n’est pas terminée. L’original authentique est épinglé sur `1e9464da54dfc651337a97c3643bfeefec712bc1`. `greedy_parent` reste une ablation et ne le remplace pas. Vanilla utilise réellement un prompt unique avec ses outils de compilation et benchmark.

Une seule finale commune est terminée : convolution, graine publique de campagne 1729, ROMEO job `774017`. Les chiffres ci-dessous appartiennent à ce cas et à ses sources historiques, pas au checkpoint WIP courant.

| Méthode | Accélération par rapport au naïf | IC 95 % / décision |
|---|---:|---|
| Agent amélioré | 1,000 | Même artefact retenu que le naïf ; identité, pas une nouvelle mesure de gain |
| Agent original | 1,00376 | [0,98499 ; 1,01755], `not_improved` |
| Codex vanilla | 1,00649 | [0,99619 ; 1,03046], `inconclusive` |

**Aucun gain certifié et aucun vainqueur global.** Des gains SEARCH ont été observés ailleurs, mais ils ne remplacent pas les FINAL. Les deux incidents de propriété mémoire globale déjà documentés restent bloquants pour les artefacts concernés ; ne pas supprimer leur index négatif ni transformer un passage de sanitizers en preuve d’absence générale de courses.

Profiling Nsight Compute sur ROMEO : indisponible avec `ERR_NVGPUCTRPERM`. Le Mac n’a pas de GPU CUDA. Aucun faux succès de profiling ni transfert de performance locale vers ROMEO.

## Nettoyage et blocage de stockage

Dernier nettoyage terminé : **4 069 entrées retirées** après archives complètes vérifiées avant/après suppression : 31 vieux dossiers scratch (3 453 entrées) et cinq vieux dossiers home (616). Les sources et attributs HOME sont conservés dans les archives privées et manifestes sous `/gpfs/home/rmdraux/.cleanup-preserved`. Les xattrs nécessitent une restauration explicite depuis les valeurs conservées ; un `tar` ordinaire seul ne les restaure pas.

Les nettoyages antérieurs de caches, cinq builds scratch, du build home vide et du tutoriel Kokkos avaient déjà été effectués. Le tutoriel modifié a été intégralement archivé. Installations, environnements, accès SSH, autres données scientifiques et autres jobs sont préservés.

Quota courant observé : scratch **1 496 302 / 1 500 000 fichiers**, soit seulement 3 698 places ; home 549 749 fichiers. Les comptes GPFS peuvent évoluer indépendamment des nombres exacts supprimés.

Le transfert de quatre sessions GENERATION closes `775251..775254` vers `/gpfs/projet/r250127/rmdraux-agentic-cuda-optimizer-retained-20261022` a été **retenu par les gardes**, sans publication ni retrait d’original :

- GPFS refuse les deux primitives `renameat2` testées (`EINVAL`). Les outils suivants utilisent un autre mécanisme, sans inventer de succès atomique natif.
- STAGE003 a inventorié les sources puis refusé un attribut SELinux lors de la création d’un répertoire cible `build`, avant copie des fichiers payload.
- Diagnostic réel : tous les champs d’identité source sont inchangés. Le répertoire source n’énumère aucun xattr `security.selinux`, mais `getxattr` expose un label effectif différent du label stocké/hérité au projet. Absence d’attribut listé et label effectif ne sont pas interchangeables.
- La première sonde de capacité a refusé avant tout SET. Sonde002 préparée pour distinguer ces notions, **capacité SET/readback/restauration non qualifiée à l’arrêt**.
- Le répertoire projet partiel contient intention, manifeste et journal ; **pas de `copy-proof.json`**. Le conserver comme preuve. Ne pas exécuter publication/retrait et ne pas forcer une garde pour continuer.

## Corrections en chantier et niveau de preuve

- Réconciliation terminale de `775346`, déjà terminée et rejouée sans nouvel appel modèle/GPU/job, publiée dans `577d20a`.
- Nouvelle API de reprise de confirmation : copie froide, capsule d’identité des entrées, candidat/incumbent/version figés, nouvel epoch entier, canonique acquise réutilisée, promotion et récompense idempotentes. **138 tests CPU** sur le lot épinglé dans le reçu. Aucun GPU réel pour ce nouveau chemin.
- **Bloquant de validité encore ouvert** : authentifier la cause d’infrastructure avant d’autoriser cette API. Une canonique valide et un worker terminal ne suffisent pas. Refuser un échec numérique/mémoire/readonly/compilation pendant la confirmation. Les phases natives de fichier/persistance doivent être prouvées par les sources exactes du harness et les reçus owned. Pour les deux anciennes confirmations, ne pas prétendre connaître un errno EDQUOT exact à partir de `basic_ios` seul.
- Placement facultatif sur un seul nœud, un seul GPU, compatibilité explicite des anciens ACK : **63 tests CPU** passent. Le nœud ne prouve pas l’UUID ; le nouveau runtime doit encore vérifier UUID physique, driver/NVRTC/toolkit/harness et contrat avant mesure.
- Outil de finalisation froide de l’original : **20 tests CPU**, vrai graphe original et crash réel d’un processus. Aucune copie/finalisation de la vraie campagne effectuée.
- TAR legacy : code de qualification en chantier dans `bootstrap_recovery.py`. Sonde lecture seule bornée proposée ; pas de reçu de qualification réelle acquis à l’arrêt.

Ces nombres concernent des lots et des fichiers épinglés ; ne pas les additionner en prétendant qu’une suite complète a passé sur le checkpoint. Syntaxe des 11 fichiers Python de produit modifiés vérifiée à l’arrêt. La CI du checkpoint doit être lue sur son SHA exact ; une CI verte ne ferme pas les blocages scientifiques ci-dessus.

## Ordre de reprise propre

1. Vérifier checkout/SHA/dirty, les verrous FINAL, tous les processus et la queue Slurm. Préserver les sessions, anciens échecs et caches documentaires. Ne pas recréer le plan de l’étude.
2. Résoudre le stockage avec une préservation de métadonnées réellement qualifiée ou une autre stratégie explicitement documentée. Revoir séparément publication et retrait ; attester l’accès historique aux quatre jobs avant toute suppression redondante. Obtenir une marge adaptée au prochain lot, pas seulement quelques inodes libres.
3. Terminer et relire les gardes d’admission d’infrastructure de la confirmation, qualifier le TAR legacy sans modifier son ancien échec, puis vérifier les quatre reprises. Les 119/28 paires anciennes restent historiques : **nouvelle confirmation complète, aucune fusion entre allocations**. La mesure canonique n’est pas à refaire.
4. Finir l’intégration de l’original : copie froide/finalisation seule, activation explicite d’une nouvelle tentative, nouvelle observation et validation de l’export sur environnement authentifié. Ne pas modifier l’algorithme original ni relabelliser son ancien échec comme succès.
5. Exécuter les vérifications pertinentes, publier un lot qualifié, contrôler sa CI exacte, figer ses sources avant dispatch. Conserver Codex local `gpt-6-luna`/`max`, LangGraph, TreeQuest et aucun plafond global de tokens/coût/appels/durée cumulée. Le test réel du CLI actuel `0.163.0-alpha.11.2` a réussi ; usage de ce contrôle : 10 757 tokens entrée + 35 sortie, coût monétaire inconnu.
6. Reprendre les **78 préparations existantes**, sans deuxième préparation/soumission, puis les quatre cas spécifiques. Partitions/allocations futures configurées `short`, un GPU, maximum Slurm deux heures. Fermer le lease dès la fin ; pas d’allocation de 48 h pour ces tests. Vérifier les vrais numéros de jobs et leur environnement au démarrage.
7. Garder les 17 FINAL protégés tant qu’une génération future reste prévue. Après les 144 générations et la quiescence des modèles, libérer explicitement le hold, finir les 17 groupes FINAL manquants et les ablations §16.
8. Seulement ensuite : exports froids complets, reconstruction d’un artefact réellement qualifié, contrôle GPU indépendant de reconstruction, agrégation des consommations/échecs et rapport français. Les bases, seeds privés, RNG, buffers et checkpoints ne sont pas inclus dans ce checkpoint Git ; ils restent dans les stockages existants. Ce checkpoint n’est pas l’export scientifique portable final.

## Fichiers de reprise

Les copies immuables des outils opérationnels sont dans `scripts/recovery_20261023/`. Leurs reçus publics et manifestes de SHA sont joints dans ce dossier de handoff. Les outils de publication/retrait sont **non admis pour une exécution réelle** à ce stade. Les chemins `work/live` dans les anciens reçus désignent les originaux locaux conservés, pas une validation supplémentaire.

Commande de dispatch à utiliser uniquement après résolution des points précédents, depuis une source figée et validée : `PRIMARY/.venv/bin/python -B FROZEN/scripts/run_quality_suite.py run --directory PRIMARY/results/quality-study-v2 --concurrency 4`. Ne pas lancer cette commande depuis ce checkout WIP.
