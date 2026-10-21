# Reprendre après une interruption réseau

Les compteurs de tokens, coût, appels et temps cumulé sont observés ; ils ne
ferment pas la campagne. Les délais d'un processus SSH, d'une requête GPU, d'un
appel modèle et la durée obligatoire de l'allocation Slurm restent des
protections techniques distinctes. La configuration ROMEO utilise `short` avec
`24:00:00` pour les allocations longues.

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

## Portée des preuves

L'incident du 21 octobre a révélé une perte de lien Wi-Fi et la suppression de
l'IP locale avant la première vague d'erreurs. Cela n'explique pas à lui seul
tous les timeouts suivants. Les journaux de reconnexion simulée et les tests
de crash CPU qualifient le protocole local ; une reprise effectivement exécutée
sur ROMEO exige ses propres reçus de worker, de requêtes et de fin Slurm.

Les résultats de recherche, la génération terminée et la certification finale
commune sont trois états distincts. Une promotion pendant la recherche ne
remplace jamais une comparaison finale indépendante sur ROMEO.
