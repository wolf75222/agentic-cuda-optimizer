# Reprendre après une interruption réseau

Les compteurs de tokens, coût, appels et temps cumulé sont observés ; ils ne
ferment pas la campagne. Les délais d'un processus SSH, d'une requête GPU, d'un
appel modèle et la durée obligatoire de l'allocation Slurm restent des
protections techniques distinctes. Les nouvelles allocations ROMEO utilisent
`long` avec `48:00:00`, admis par un probe `sbatch --test-only` sans soumission.
La partition accepte au plus 30 jours, mais une demande aussi longue avait une
estimation de démarrage bien plus tardive. Les anciens workers `short`/24h
conservent leur bail et leur configuration archivés. Cette durée obligatoire
par allocation n'est pas un plafond cumulé de la campagne.

## Reconnexion pendant une opération

`SlurmConfig.reconnect_transport=true` est la valeur par défaut. Une lecture,
une publication atomique de mêmes octets, un échange de même identifiant et de
même empreinte, ou une observation `squeue`/`sacct` peut attendre le retour du
réseau sans plafond cumulé. Chaque processus SSH garde son délai propre. Le
repli attend 1, 2, 5, 10 puis au plus 30 secondes entre tentatives.

`worker/transport-reconnect.jsonl` enregistre l'opération, les tentatives, les
diagnostics et l'empreinte du corps. Une erreur d'authentification, de clé ou
d'application est renvoyée immédiatement. `sbatch`, `scancel`, le staging et les
commandes SSH génériques ne sont jamais répétés par ce mécanisme. Le protocole
de soumission conserve son intention durable et réconcilie son accusé perdu.

## Examiner avant de reprendre

```bash
.venv/bin/python scripts/run_quality_suite.py status --directory results/quality-study-v2
```

Conserver le contrôleur et les modèles encore actifs. Le verrou du contrôleur
et les verrous par tâche empêchent une reprise concurrente. Une panne réseau
n'est ni un mauvais résultat CUDA ni un gain nul. Les erreurs numériques, les
sanitizers, les quarantaines et les propriétaires contradictoires refusent le
chemin de reprise infrastructure.

## Bootstrap interrompu avant tout modèle

```bash
.venv/bin/python scripts/run_quality_suite.py recover-bootstrap \
  --directory results/quality-study-v2 \
  --task quality-20261021-v2.convolution.1729.full \
  --reason "Perte du lien réseau local, attestée par les journaux et le reçu SSH"
```

Cette commande prépare seulement la reprise : zéro appel modèle, requête GPU
ou nouvelle soumission. Elle authentifie à nouveau la session et le job.

- Worker vivant : même allocation et même GPU, préfixe de requêtes figé par
  SHA. Les résultats déjà acquis sont relus, et la dernière requête incertaine
  est collectée avec son identifiant et son contenu d'origine. Après le préfixe,
  les nouvelles mesures restent de nouvelles requêtes. Un contrôle de préfixe
  complet précède tout modèle.
- Worker terminé : tentative indépendante, nouvelle session, anciens résultats
  et coûts conservés. Les mesures des deux allocations ne sont pas fusionnées.
- Aucune soumission : la session stable est conservée uniquement si les reçus
  locaux et les intentions distantes confirment cette absence.

Le natif conserve ses fichiers antérieurs dans `bootstrap-archives/000/` sous
la racine refusée au modèle. Son nouvel espace de travail est vierge ; les
entrées recréées doivent correspondre octet pour octet aux empreintes figées.
Un PID, thread, checkpoint, appel ou résultat modèle déjà présent interdit ce
chemin : un prompt natif incertain n'est jamais réémis automatiquement.

L'exécution reprend ensuite explicitement :

```bash
.venv/bin/python scripts/run_quality_suite.py run \
  --directory results/quality-study-v2 --concurrency 4
```

Une interruption de la préparation elle-même laisse un état `prepared` qui
bloque le dispatch. Pour terminer sa transaction locale sans nouvelle activité
GPU ou modèle :

```bash
.venv/bin/python scripts/run_quality_suite.py resume-bootstrap-preparation \
  --directory results/quality-study-v2 --task IDENTIFIANT_EXACT
```

La propriété distante et l'intégrité des fichiers sont recontrôlées. La même
session préparée est conservée ; la tentative n'est pas incrémentée deux fois.

## Réparation interrompue avant la première récompense

`recover-unscored` couvre un cas plus restreint : le défaut de diagnostic sur
une validation `cases` en liste a interrompu une réparation, alors que son job
est terminé. Il exige une racine TreeQuest neutre, un seul essai en attente,
aucune évaluation/récompense/promotion intégrée et tous les appels modèle déjà
achevés et réconciliés. Une campagne ayant appris un score ne peut pas passer
par ce transfert.

```bash
.venv/bin/python scripts/run_quality_suite.py recover-unscored \
  --directory results/quality-study-v2 \
  --task quality-20261021-v2.matmul.3141.no_autotuning \
  --reason "Diagnostic cases-list corrigé ; réparation interrompue avant tout score"
```

La copie conserve tous les espaces de checkpoint LangGraph, le BLOB TreeQuest,
son RNG, les candidats et les reçus modèle. L'original reste intact. Le nouveau
worker mesure sa propre baseline ; seul ce nouveau contexte de performance est
lié à la copie encore sans score. L'ancienne baseline reste archivée et son
coût est conservé avec une clé distincte. Une nouvelle recherche web et un
nouveau dossier applicable sont obligatoires : l'ancien dossier n'est pas
requalifié par simple changement d'identifiant. L'erreur de compilation est
transmise avec son contrat et son environnement historiques, hors classement.

Une préparation interrompue se réconcilie avec
`resume-unscored-preparation --directory ... --task ...`. Les copies partielles
sont conservées ; cette commande ne soumet ni job ni prompt et ne crée pas une
deuxième session. Le dispatch reste bloqué tant que l'intention est `prepared`.

Le temps écoulé d'un essai peut traverser une interruption sans allocation. Il
ne constitue donc pas une mesure de réservation GPU. Le bilan distingue les
durées Slurm effectivement observées, le temps des requêtes natives et les
durées CUDA Events.

## Portée des preuves

L'incident du 21 octobre a révélé une perte de lien Wi-Fi et la suppression de
l'IP locale avant la première vague d'erreurs. Cela n'explique pas à lui seul
tous les timeouts suivants. Les journaux de reconnexion simulée et les tests
de crash CPU qualifient le protocole local ; une reprise effectivement exécutée
sur ROMEO exige ses propres reçus de worker, de requêtes et de fin Slurm.

Les résultats de recherche, la génération terminée et la certification finale
commune sont trois états distincts. Une promotion pendant la recherche ne
remplace jamais une comparaison finale indépendante sur ROMEO.
