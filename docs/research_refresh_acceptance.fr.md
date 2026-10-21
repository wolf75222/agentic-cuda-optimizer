# Contrôle documentaire du renouvellement de recherche (AC02)

`scripts/run_research_refresh_acceptance.py` vérifie la chaîne recherche → dossier → proposition → refus d’une technique non couverte → diagnostic → recherche ciblée → nouvelle proposition couverte. Le plan est le mode par défaut. Seul `--run` lance réellement Codex local, avec `gpt-6-luna` et l’effort `max`, ainsi que la recherche web et la lecture des sources.

Ce contrôle utilise `CodexBackend`, `ResearchService`, `ResearchDossier`, `CandidateGenerator` et les méthodes `research`, `generate` et `diagnose` héritées de `CampaignRuntime`. L’adaptateur documentaire reconstruit uniquement un contexte historique authentifié. Il interdit `evaluate`, `integrate`, `finalize` et toute requête native/GPU ; il ne remplace aucune garde de recherche, d’applicabilité, de contrat ou d’ABI.

## Préparer et lancer

Depuis la racine du dépôt, utiliser une campagne dont la génération et le worker sont terminés proprement. Cet exemple désigne l’archive connue du job ROMEO **771748** ; le SHA est celui de son `result.json`, et non celui du contrôleur courant.

```sh
ac02_bootstrap=results/quality-study-v2/campaigns/quality-20261021-v2.convolution.1729.no_autotuning
ac02_output=results/research-refresh-acceptance/ac02-documentaire-001
ac02_result_sha=01bf49fd7fc57ad62dc9183aee2e33a7afdaba85e99768dbd0b4e2cb74bd424a

.venv/bin/python scripts/run_research_refresh_acceptance.py --plan \
  --run-id ac02-documentaire-001 --family tiling \
  --bootstrap-campaign "$ac02_bootstrap" \
  --bootstrap-result-sha256 "$ac02_result_sha" --output "$ac02_output"
```

Inspecter `plan.json`, puis lancer explicitement le contrôle réel, après vérification de la source du contrôleur et de la configuration locale de Codex :

```sh
.venv/bin/python scripts/run_research_refresh_acceptance.py --run \
  --run-id ac02-documentaire-001 --family tiling \
  --bootstrap-campaign "$ac02_bootstrap" \
  --bootstrap-result-sha256 "$ac02_result_sha" --output "$ac02_output"
```

Le même dossier peut recevoir le plan puis son premier lancement. Pour un autre contrôle, choisir un nouveau dossier et un nouvel identifiant. `--codex-executable CHEMIN` permet de désigner explicitement le CLI ; sinon le résolveur commun utilise `CODEX_CLI_PATH`, le CLI intégré disponible, puis `codex`.

## Portée et lecture du résultat

Le bootstrap vérifie la configuration d’origine, la propriété job/session et sa terminaison `COMPLETED/0:0`, l’identité matérielle et logicielle archivée, la source et le lancement du kernel naïf, les entrées publiques déterministes et le hash complet du contrat `Evaluator`. Les latences historiques sont recalculées à partir des séries brutes avec les mêmes estimateurs que l’évaluateur et comparées aux agrégats archivés. L’archive reste inchangée. Son déplacement peut invalider le contrat, qui inclut les chemins originaux des entrées.

Cette authentification locale ne constitue **ni une observation actuelle de ROMEO, ni une nouvelle validation GPU**. Les anciennes mesures servent uniquement au contexte initial ; leur coût GPU n’est pas transféré au nouveau contrôle. Aucun benchmark, score comparatif, résultat final ou conclusion de qualité de la cohorte §16 n’en découle.

Les instructions de contrôle sont ajoutées et archivées **avant** les appels CLI aux propositions. Deux propositions compatibles doivent réutiliser exactement le même dossier, sans recherche intermédiaire. Une demande explicite de technique absente du dossier doit ensuite être refusée avant création d’un nouveau candidat. Les réponses du modèle et les dossiers ne sont jamais modifiés pour obtenir ce résultat. Les créneaux nommés `repair` servent ici aux variations documentaires ; aucune erreur de compilation fictive n’est annoncée.

L’autorité finale est `research-refresh-acceptance.json`. Elle référence le manifeste `proof-sha256.json`, les appels et leur consommation observée, ainsi que les pièces du bootstrap historique. Les prompts exacts figurent dans `campaign/acceptance-controls/`, et les traces CLI dans `campaign/model/`.

Un modèle qui n’obéit pas au contrôle, une couverture documentaire insuffisante ou une garde refusée laisse le statut **`incomplete`**. Ne pas contourner le refus. `run-intent.json` est créé exclusivement et synchronisé avant les appels : sa présence interdit une relance automatique, même après perte de réponse ou interruption. Inspecter et réconcilier les traces existantes avant de décider d’un nouveau contrôle indépendant ; ne pas supprimer l’intention pour rejouer un appel incertain.
