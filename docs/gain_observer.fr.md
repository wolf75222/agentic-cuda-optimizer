# Observation indépendante du premier gain utile

La suite §16 active `run_upstream_campaign(..., observe_first_gain=True)` et
`VanillaGPUService(..., observe_first_gain=True,
observer_started_at_unix=debut_campagne)`. Ces options sont désactivées par défaut.
L'observateur est un callback de mesure extérieur à la politique : il n'ajoute
aucun message, outil, hypothèse, récompense ou choix au graphe original et ne
renvoie aucun résultat dans les réponses HTTP destinées à Codex vanilla.

Après une proposition originale complètement validée et benchmarkée, ou après
un appel natif `benchmark` réussi, il utilise une nouvelle confirmation appariée
du candidat contre le kernel naïf avec le contrat commun et les seules données
de recherche. Les seuils de gain, l'intervalle de confiance et les régressions
critiques sont ceux de `Evaluator.confirm`. Seul `promoted` établit un gain utile.
Cette observation ne remplace pas la qualification finale sur données nouvelles.

Une intention synchronisée sur disque précède toute mesure. L'empreinte de
l'artefact identifie une décision unique : une observation inconclusive ou
défavorable n'est pas répétée jusqu'à obtenir une victoire. Une interruption ou
un incident laisse la métrique inconnue et bloque toute nouvelle activité GPU
de cette méthode, y compris après reprise. Les protections du worker restent
applicables ; l'observateur ne contourne aucune quarantaine.

Les reçus se trouvent dans `gain-observer/summary.json`, accompagnés des demandes
et rapports bruts dans `gain-observer/request-receipts/`. Le parent obtient les
métriques natives par `service.observer_metrics()`. Après un crash, la fonction
`read_observer_metrics(repertoire / "gain-observer")` effectue une lecture locale
pure, sans Evaluator, modèle, connexion ni mesure. En l'absence d'un snapshot
final, les coûts cumulés s'arrêtent au dernier callback durable et ce périmètre
est explicite.

Le temps jusqu'au premier gain inclut les coûts d'observation. Les durées de la
méthode et de l'observateur sont enregistrées séparément ainsi que leur somme
lorsqu'elles sont connues. Cela ne constitue pas une comparaison à budget égal.
L'intercalage de mesures peut modifier l'état thermique ou les caches du GPU.
Les tokens du graphe original proviennent des appels achevés ; pour le tour CLI
natif, les tokens au moment du gain restent inconnus, même si la consommation
totale est connue à la fin. Aucune interpolation de tokens n'est effectuée.
