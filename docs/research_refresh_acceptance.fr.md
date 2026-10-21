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

## Reprendre le refus documentaire observé

La reprise explicite supporte uniquement le point où les trois premières propositions et leur diagnostic sont achevés, puis où la recherche ciblée est refusée. Elle authentifie les sept appels CLI terminés, leurs payloads, les deux candidats admis, le refus sémantique, le diagnostic et les événements SQLite. L’ancien helper n’ayant pas sérialisé son dernier état en mémoire, `resume-state.json` décrit sa reconstruction à partir de ces pièces ; aucune garde de recherche réussie n’est déduite de cette reconstruction.

L’ancienne autorité reste `incomplete` et son dossier reste inchangé. L’exemple suivant épingle le refus réel du contrôle `research-refresh-ac02-20261021` :

```sh
ac02_resume_source=results/research-refresh-ac02-20261021
ac02_resume_output=results/research-refresh-ac02-resumed-20261021
ac02_failed_sha=b260de2a19daf7a9c23064019c113b4940a0fdeb7dfb69cd3f20a757c30cf37d
ac02_proof_sha=28c06ce27cbc643447af4d3cddc3766bfcd7556a20a8937901ec023155ca69ee

.venv/bin/python scripts/run_research_refresh_acceptance.py --plan \
  --run-id research-refresh-ac02-resumed-20261021 \
  --resume-from "$ac02_resume_source" --resume-receipt-sha256 "$ac02_failed_sha" \
  --resume-proof-manifest-sha256 "$ac02_proof_sha" --output "$ac02_resume_output"

.venv/bin/python scripts/run_research_refresh_acceptance.py --run \
  --run-id research-refresh-ac02-resumed-20261021 \
  --resume-from "$ac02_resume_source" --resume-receipt-sha256 "$ac02_failed_sha" \
  --resume-proof-manifest-sha256 "$ac02_proof_sha" --output "$ac02_resume_output"
```

La nouvelle intention précède les appels. Une copie isolée conserve les anciens événements, dossiers et traces ; seuls une nouvelle recherche ciblée puis une proposition couverte sont autorisés. Ni la sélection, ni les propositions antérieures, ni le diagnostic ne sont réémis. Un nouveau refus laisse la reprise `incomplete`, sans nouvel essai automatique.

Une annexe authentifiée de l’allocation historique joint la ligne `nvidia-smi` au contexte de recherche : paquet pilote `590.48.01`, valeur API CUDA brute `13010` et toolkit `12.6` restent distincts. Ses références SHA et liens job/session/UUID sont vérifiés par `archived_hardware_research_context`. Cette annexe ne modifie aucun environnement, fingerprint ou contrat et ne satisfait aucune condition de compatibilité ; la recherche doit lire les documents officiels et passer les gardes normales.

Le nouveau reçu possède le type `documentary_research_refresh_resumption_acceptance` et lie l’ancienne autorité et son manifeste. `inherited_model_calls` conserve les sept appels antérieurs ; `model_calls` et `new_usage_delta` décrivent la consommation nouvelle, tandis que le ledger copié conserve l’historique. Les anciennes instructions restent dans `campaign/acceptance-controls/`, les nouvelles dans `campaign/acceptance-controls-resume/`. Aucun score ou coût GPU historique ne devient une nouvelle mesure.
