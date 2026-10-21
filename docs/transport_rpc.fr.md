# Transport SSH et reprise

`SlurmConfig.transport_mode` accepte `auto` (défaut), `exchange` ou `legacy`.
Le worker annonce `exchange_v1` dans son reçu de disponibilité. Le mode auto
utilise alors une connexion SSH pour publier une requête, attendre son résultat
et rapatrier ses artefacts. Sans cette capacité, il conserve le protocole historique
et inscrit la raison du repli dans le reçu. Le mode exchange explicite refuse
un ancien worker.

Les données sont adressées par SHA256 et vérifiées avant chaque publication.
Le worker publie l'enveloppe en dernier. Une perte d'accusé reprend le même
identifiant et la même empreinte ; elle ne crée ni nouveau benchmark ni nouvelle
soumission Slurm. Une requête différente exige un nouvel identifiant. Le helper
du login ne compile ni n'exécute CUDA : seul le worker dans l'allocation le fait.

Les réponses incluent le nombre de connexions, les transferts, le cache de blobs
et la durée du contrôleur. Ces durées sont séparées des CUDA Events. L'ordre
statistique, les échantillons, les oracles et les règles de promotion sont inchangés.

Un worker compromis ou terminal refuse tout nouveau travail. Ses résultats
déjà terminés restent récupérables. Une reprise avec des sources de worker
différentes est refusée avant toute modification ou soumission : utiliser le
snapshot initial pour terminer la session, ou une nouvelle session pour le
nouveau code. Les anciens artefacts restent conservés.

Le contrôle fonctionnel ci-dessous écrit seulement un plan sans `--run` :

```sh
.venv/bin/python scripts/gpu_rpc_acceptance.py \
  --output results/rpc-proof-IDENTIFIANT --execution-config configs/romeo.json --run
```

Il ouvre une allocation propre et compare les deux transports sur une séquence
fixée avant soumission : mêmes sources, entrées, lancements, sorties complètes,
compteurs bruts et environnement. Il mesure le coût SSH sans revendiquer un gain
du kernel. Son reçu terminal est nécessaire pour déclarer le contrôle acquis.
