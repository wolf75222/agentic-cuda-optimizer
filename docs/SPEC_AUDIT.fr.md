# Audit de la spécification et des preuves

La spécification a été relue intégralement : **644 lignes**, SHA-256 `cc031ea26282fba6f1fa59fcc4629feebd69c05180df5f6bb8cd0e84d608e8fd`. Sa copie `docs/spec.md` est identique. Le registre [requirement-evidence.json](validation/requirement-evidence.json) contient **78 entrées** : 19 sections et leurs 26 sous-sections, 20 critères AC, 10 décisions du §1 et 3 demandes utilisateur.

Checkout examiné : branche `codex/luna-romeo`, HEAD `45407a2696202044bfea6863893d05564e0d5bc3`. **Aucune conformité globale ni amélioration générale de l’optimiseur n’est acquise.** Une implémentation, un test local, une soumission Slurm et une qualification scientifique sont des preuves différentes.

L’instruction humaine « Ne met pas de budget laisse les agents trql » remplace les plafonds globaux proposés. Aucun plafond global de tokens, coût, appels, consommation GPU ou durée cumulée n’est imposé. Les coûts restent observés ; les nombres d’itérations, réparations, configurations et les délais techniques restent explicites. Aucune égalité de budgets imposés n’est revendiquée.

Le JUnit existant `work/live/policy-tests-release.xml` déclare **590 entrées de résultat**, comprenant **585 éléments testcase et 5 sous-tests**, sans échec, erreur ni skip. Durée : 38.578 s ; horodatage : 2026-10-21T16:52:00+02. SHA-256 : `db605ce4a4e0e8eebf14a37bd4685bb208f35ce9b9081326039ec0e66432a297`. Le XML ne contient pas de SHA source : il ne prouve pas, à lui seul, une validation du HEAD exact. La [CI 39621303226](https://github.com/wolf75222/agentic-cuda-optimizer/actions/runs/39621303226) est désormais vérifiée par le JSON archivé : succès terminé au commit exact 45407a2, SHA du reçu `2bb7f7a208717644a1c4dd77dca8ed94616f9e43fdcfb96ca8f9ab0e8d6d6227`. Le checkout courant comporte des changements postérieurs : cette CI ne les valide pas. Les hashes au début et à la fin de l’audit sont distingués dans le registre.

Les preuves prioritaires manquantes restent :

La nouvelle suite locale passe **892 tests et 5 sous-tests**, sans échec, erreur ni skip. Le JUnit déclare 897 résultats en 42.731 s ; la sortie pytest rapportée par le parent indique 43.21 s et 13 avertissements amont. Le reçu [unscored-local-tests.json](validation/unscored-local-tests.json) identifie le HEAD `06bd4be0ab2e725deb52f38412d6f42c123d51e6`, le checkout modifié et les SHA-256 de ses 140 fichiers sources, tests, fixtures et configurations concernés. Les empreintes ont été capturées après le test ; le JUnit ne les embarque pas. Cette preuve précise la reprise sans nouvelle récompense et les bornes de consommation observées, avec déduplication des appels et leases Slurm. Le reçu antérieur [recovery-local-tests.json](validation/recovery-local-tests.json),785 tests +5 sous-tests, est conservé. **Aucune nouvelle exécution ROMEO ni CI des changements non commités n’est déduite de ces tests.**

- **§5.2, lecteur v2** : la chaîne réelle de démarrage est désormais acquise au 45407a2 pour GEMM 3141 full et greedy : Luna → dossier v2 applicable → proposition → validation/mesure → promotion sur cas de recherche. Les passages et références ont été revérifiés avec les gardes du code figé. Les dossiers ont diagnostic_ids=[] : cela ne prouve pas un nouveau passage diagnostic → recherche en v2. Les tours 2/3 sont bloqués par une entrée CLI dépassant sa limite technique, sans plafond de ressources. La boucle diagnostic 770679 reste acquise au 76da, avec certaines lectures historiques partielles.
- **§16, étude v2** : 144 générations prévues (6 familles × 3 graines × 8 modes), puis18 finales communes. Après les erreursSSH/DNS initiales, les jobs 772656/772658 ont terminé leur génération GEMM 3141, sans finale commune. Le véritable original et le native v2 ne sont toujours pas qualifiés par une comparaison finale commune.
- **Profilage** : refus réel `ERR_NVGPUCTRPERM` sur ROMEO. Aucun profil de compteurs réussi ni GPU CUDA local disponible pour un repli effectif. L’indisponibilité reste distincte d’un succès.

**acquis*** signifie uniquement « scénario cité acquis sur les versions et dans la portée du reçu ». Cette marque ne certifie ni tout le critère sur tous les workloads, ni le HEAD actuel. « Local » désigne les régressions JUnit citées, souvent simulées. Les détails par phase, fonctions et lignes, hashes, versions, reçus et limites sont dans le registre JSON.

| Section | Exigence | Implémentation | Local | ROMEO | Reste ou limite |
| --- | --- | --- | --- | --- | --- |
| §1 | Objectif et décisions retenues | acquis* | acquis* | partiel | Étude §16 et chaîne modèle avec passages v2 non acquises; aucune supériorité globale. |
| §1.1 | Périmètre de la première version complète | acquis* | acquis* | acquis* | Multi-kernels, allocations supplémentaires et autres ABI hors contrat. |
| §2 | Point de départ : préserver et renforcer l’existant | acquis* | acquis* | partiel | Véritable original sans campagne réelle qualifiée dans la finale commune. |
| §3 | Architecture et responsabilités | acquis* | acquis* | partiel | Workflow actuel complet après panne réseau non acquis. |
| §3.1 | Graphe de campagne | acquis* | acquis* | partiel | Démarrage complet v2 acquis sur deux GEMM; retours/reprise et toutes familles non qualifiés. |
| §3.2 | Sous-graphe d’une expérience | acquis* | acquis* | partiel | Réparations réelles multibranches et DSL récent à compléter. |
| §4 | Contrat immuable et oracle indépendant | acquis* | acquis* | acquis* | Six opérations non encore qualifiées dans une étude commune. |
| §4.1 | `WorkloadContract` | acquis* | acquis* | acquis* | AutresABI/domaines exigent un nouveau contrat; admission statique conservative. |
| §4.2 | Trois références distinctes | acquis* | acquis* | acquis* | Aucun gain contre cuBLAS revendiqué; référence bibliothèque facultative. |
| §4.3 | Jeux de tests | acquis* | acquis* | acquis* | Génération native v2 avec barrière nouvelle non exercée; probeCPU distinct acquis. |
| §5 | Recherche obligatoire : code, documentation, web et PMPP | acquis* | acquis* | partiel | Démarrage v2 acquis sur deux campagnes GEMM. Nouvelle famille, retour technique non couverte et diagnostic ciblé v2 restent à exercer; tours suivants bloqués par taille d’entrée CLI. |
| §5.1 | Corpus principal | acquis* | acquis* | acquis* | Source complète et passages ciblés distingués. Douze passages partiels par dossier portent truncated=true; code/lanceur utiles réellement fournis, aucune lecture intégrale prétendue des autres fichiers. |
| §5.2 | Déclencheurs et garde LangGraph | acquis* | acquis* | partiel | Startup v2 acquis. Diagnostic → recherche réelle acquis seulement au76da historique; diagnostic_ids des dossiers actuels vide. Tours2/3 bloqués par entrée CLI de2.11–2.16Mcaractères, limite technique1 048 576; correctif résumé+hash des mesures non encore exercé dans ces campagnes figées. |
| §5.3 | Dossier de recherche | acquis* | acquis* | acquis* | Réutilisation compatible et introduction d’une technique non couverte:scénario réel complet AC02 encore manquant. |
| §5.4 | Maintien et enrichissement de PMPP | acquis* | acquis* | acquis* | Ablations permanent seul/consultations ciblées/sans PMPP non exécutées; aucun livre intégral prétendu. |
| §6 | TreeQuest et politique d’exploration | acquis* | acquis* | partiel | Scénarios multibranches complets et comparaison original non acquis. |
| §6.1 | Interface et parent sélectionné | acquis* | acquis* | partiel | Parent choisi plus lent que meilleur: preuve locale, scénario GPU spécifique manquant. |
| §6.2 | Fermeture réelle et réparations bornées | acquis* | acquis* | en attente | Toutes portées/épuisement réparations avec autres branches réelles continuant non exercés. |
| §6.3 | Score stationnaire | acquis* | acquis* | partiel | Deux branches mêmes latences et petits gains: tests locaux; pas de scénario GPU complet distinct. |
| §6.4 | Un essai TreeQuest et ses variantes locales | acquis* | acquis* | acquis* | DSL récent de contraintes pas encore exercé par campagne modèle réelle. |
| §7 | Validation, mesure et promotion | acquis* | acquis* | acquis* | Pilotes acquis pour artefacts cités; aucune certification des cohortes §16. |
| §7.1 | Parcours obligatoire | acquis* | acquis* | acquis* | Sanitizers limités aux cas / artefacts cités; sans garantie générale d’absence de bogue. |
| §7.2 | Protocole de mesure | acquis* | acquis* | acquis* | Temps locaux, instrumentés ou interGPU jamais substitués aux mesuresROMEO. |
| §7.3 | Confirmation d’une amélioration | acquis* | acquis* | acquis* | Conflit concurrent de promotion validé localement; pas garantie statistique universelle. |
| §7.4 | Certification finale | acquis* | acquis* | acquis* | Étudev2 sans finale commune; statuts acquis seulement pour artefacts exacts cités. |
| §8 | Autotuning des paramètres | acquis* | acquis* | acquis* | DSL récent:63 tests locaux mais campagneGPU non acquise; plafonds globaux retirés par demande humaine. |
| §9 | Banc d’essai, cache et parallélisme | acquis* | acquis* | partiel | Aucun profil de compteurs réussi ni repliCUDA local effectif. |
| §9.1 | Compilation et workers | acquis* | acquis* | acquis* | Pas d’upgrade réel pilote/toolkit revendiqué; dépendances non transportées refusées. |
| §9.2 | Ordonnancement | acquis* | acquis* | acquis* | Exclusivité prouvée pour workers/UUID cités, pas toute charge externe imaginable. |
| §9.3 | Profilage utile et proportionné | acquis* | acquis* | partiel | ERR_NVGPUCTRPERM sur ROMEO; aucun profil réussi ni GPU CUDA local de repli. |
| §10 | Modèle de données et interfaces | acquis* | acquis* | acquis* | Exports complets des futures cohortes requis; sources/version différentes jamais assimilées. |
| §10.1 | Identités et objets | acquis* | acquis* | acquis* | Intermédiaires, observation arbre et certification restent distincts. |
| §10.2 | Interfaces applicatives à développer | acquis* | acquis* | partiel | Interfaces et première chaîne modèle v2 réelle acquises; réutilisation/retourtechnique complète et correction de taille du contexte restent à exercer. |
| §11 | Persistance, reprise et budgets | acquis* | acquis* | partiel | Réconcilier transport/reprise v2 sans double récompense ni mélange de mesures. |
| §11.1 | État durable | acquis* | acquis* | partiel | Crash/replay local réel de preuveCUDA archivée; pas de nouvelle exécution ROMEO. |
| §11.2 | Idempotence | acquis* | acquis* | partiel | Reprise complète144 générations/18 groupes non acquise; preuves locales ne sont pas nouveau GPU. |
| §11.3 | Budgets | acquis* | acquis* | partiel | Budgets globaux explicitement remplacés par compteurs; délais techniques et duréeSlurm restent. |
| §12 | Configuration proposée | acquis* | acquis* | acquis* | YAML proposé adapté à override humain sans plafonds globaux. |
| §13 | Sorties et observabilité | acquis* | acquis* | partiel | Bundle/rapport finaux §16 non acquis; exports incomplets explicitement non certifiés. |
| §14 | Plan de modification du code | acquis* | acquis* | acquis* | Responsabilités intégrées via adaptateurs; noms de modules proposés non obligatoires. |
| §15 | Critères d’acceptation fonctionnels | acquis* | acquis* | partiel | Voir20AC individuels; présence de code/tests ne qualifie pas leur ensemble. |
| §16 | Évaluer l’amélioration de l’optimiseur | acquis* | acquis* | partiel | Étude partielle:premiers GEMM full/greedy3141 generation_complete avec promotions de recherche; aucune finale commune acquise. Achever144générations/18groupes et qualifier véritable original/nativev2; publier coûts, taux, dispersion et tous échecs. |
| §17 | Ordre de réalisation | acquis* | acquis* | sans objet | Conditions des lots dépendent desAC/§16 encore non acquis. |
| §18 | Traçabilité des demandes | acquis* | acquis* | sans objet | Traçabilité documentaire, aucune exécution ajoutée par la correspondance. |
| §19 | Références techniques | acquis* | acquis* | partiel | Chaque campagne conserve versions réellement lues; un lien de la spec ne prouve aucune consultation. |

| Critère | Scénario | Implémentation | Local | ROMEO | Reste ou limite |
| --- | --- | --- | --- | --- | --- |
| AC-01 | Génération demandée sans dossier applicable. | acquis* | acquis* | acquis* | Garde réelle démarrage v2 acceptée puis première proposition mesurée sur deux GEMM; autres déclencheurs AC02/§5.2 demeurent distincts. |
| AC-02 | Nouvelle famille, ajustement compatible, puis technique non couverte introduite dans une proposition. | acquis* | acquis* | partiel | Dossier initial de famille tiling réel acquis; pas de séquence réelle complète nouvelle famille → ajustement compatible → technique non couverte → retour recherche. |
| AC-03 | Recherche web indisponible ou sans exemple pertinent. | acquis* | acquis* | en attente | Panneweb et constat négatif complet réels à distinguer; aucun faux succès de recherche. |
| AC-04 | Consultation PMPP avec uniquement le guide et les supports installés. | acquis* | acquis* | acquis* | AblationsPMPP encore en attente; aucun livre intégral ni pagination inventés. |
| AC-05 | TreeQuest sélectionne un candidat moins rapide que le meilleur global. | acquis* | acquis* | partiel | Parent plus lent que meilleur testé localement; scénario réel spécifique non acquis. |
| AC-06 | Réparations épuisées et fermeture d’un parent/action ou sous-arbre. | acquis* | acquis* | en attente | Réparations/fermetures/batch avec autres branches réelles continuant non exercés intégralement. |
| AC-07 | Deux branches obtiennent les mêmes temps avec des parents différents. | acquis* | acquis* | sans objet | Propriété algorithmique démontrée localement; pas une mesureGPU. |
| AC-08 | Résultat incorrect, sortie partiellement écrite ou entrée modifiée illicitement. | acquis* | acquis* | acquis* | ProbesGPU intentionnels acquis pour kernels cités; sans garantie universelle. |
| AC-09 | Changement de précision, tolérance ou périmètre par le candidat. | acquis* | acquis* | acquis* | Rejet réel double/TF32/half avantharness; admission conservative CUDA arbitraire. |
| AC-10 | Gain inférieur au bruit ou régression d’un cas critique. | acquis* | acquis* | partiel | Intervalles/bruit et régressions visibles; scénario dédié régression critique+incumbent à compléter sur cible. |
| AC-11 | Gain reproductible satisfaisant la règle fixée. | acquis* | acquis* | acquis* | Certification attachée aux artefacts historiques exacts, aucune certification étude v2. |
| AC-12 | Autotuning avec configurations invalides et candidat gagnant. | acquis* | acquis* | acquis* | Gagnant remesuré réel acquis; DSL déclaratif récent encore sans campagneGPU. |
| AC-13 | Même compilation, puis modification d’une dépendance ou option. | acquis* | acquis* | acquis* | Invalidations options/manifeste/tags acquises; pas changement réel pilote physique. |
| AC-14 | Deux benchmarks ou un benchmark et un profil demandent le même GPU. | acquis* | acquis* | acquis* | Clients concurrents et serveur sérialisé même worker/job/UUID acquis; compteurs profil refusés. |
| AC-15 | Worker avec contexte CUDA compromis. | acquis* | acquis* | acquis* | 769644compromis puis769661neuf; allocation compromise jamais réactivée. |
| AC-16 | Interruption après résultat durable avant `tell`, puis après `tell` avant persistance de l’arbre. | acquis* | acquis* | sans objet | Crash/replay de vrais processus locaux sur preuveCUDA archivée; aucun nouveau GPU. |
| AC-17 | `trial_id` répété, deux essais pointant vers la même mesure cachée, puis remesure intentionnelle. | acquis* | acquis* | sans objet | Déduplication/rejeu local acquis; aucune nouvelle observationGPU ni secondtell correctif. |
| AC-18 | Budget global épuisé, panne d’outil ou changement de matériel à la reprise. | acquis* | acquis* | partiel | Reprise réseau v2/réconciliation à achever; budget global retiré par override humain. |
| AC-19 | Nouvelle forme admissible, puis échec sur le jeu final. | acquis* | acquis* | acquis* | Formes admissibles/échecfinal sans générateur acquis; modèle natif v2 avec isolation nouvelle non exercé. |
| AC-20 | Reconstruction d’un candidat retenu. | acquis* | acquis* | acquis* | Finaliste réel reconstruit/validé n316, 0 benchmark/0 modèle; exports §16 restent à produire. |

| Décision / demande | Exigence | ROMEO | Reste ou limite |
| --- | --- | --- | --- |
| USER-LOCAL | Codex Luna Max local et LangGraph | acquis* | Processus et orchestration locaux; poids modèle hébergés par fournisseur. Démarrage lecteurv2 réel acquis; autres déclencheurs et étude complète restent en attente. |
| USER-HPC | SSH ROMEO et jobsGPU Slurm | acquis* | Des tâches initiales ont échouéSSH/DNS; deux nouvelles générationsGEMM ont ensuite terminé. La reprise complète et la disponibilité réseau présente ne se déduisent pas de ces reçus. |
| USER-PROFILE | Profilage hybride sans mélange matériel | partiel | CompteursROMEO interdits; aucun repliCUDA local effectif. |
| ARCH-01 | Conserver LangGraph pour les étapes d’exécution, les transitions, les outils et la reprise. | partiel | Démarrage complet v2 acquis sur deux GEMM; retours/reprise et toutes familles non qualifiés. |
| TREE-01 | Intégrer TreeQuest / AB-MCTS pour sélectionner le parent et l’action d’une expérience, avec une politique explicite d’éligibilité et de fermeture des branches. | partiel | Scénarios multibranches complets et comparaison original non acquis. |
| RES-01 | Faire du code CUDA optimisé existant la principale source d’inspiration : CCCL/CUB, CUTLASS/CuTe et bibliothèques spécialisées pertinentes. | acquis* | Source complète et passages ciblés distingués. Douze passages partiels par dossier portent truncated=true; code/lanceur utiles réellement fournis, aucune lecture intégrale prétendue des autres fichiers. |
| RES-02 | Rendre la recherche de code, de documentation et sur le web obligatoire, au démarrage et avant chaque nouvelle famille d’optimisations. | partiel | Startup v2 acquis. Diagnostic → recherche réelle acquis seulement au76da historique; diagnostic_ids des dossiers actuels vide. Tours2/3 bloqués par entrée CLI de2.11–2.16Mcaractères, limite technique1 048 576; correctif résumé+hash des mesures non encore exercé dans ces campagnes figées. |
| PMPP-01 | Conserver et enrichir PMPP comme cadre de raisonnement permanent et comme source de consultation ciblée. | acquis* | Ablations permanent seul/consultations ciblées/sans PMPP non exécutées; aucun livre intégral prétendu. |
| EVAL-01 | Séparer référence de correction, baseline de performance, validation, mesure et promotion du meilleur candidat. | acquis* | Pilotes acquis pour artefacts cités; aucune certification des cohortes §16. |
| TUNE-01 | Confier au LLM les choix de stratégie et à un autotuner l’exploration bornée des paramètres numériques. | acquis* | DSL récent:63 tests locaux mais campagneGPU non acquise; plafonds globaux retirés par demande humaine. |
| EXEC-01 | Réduire le coût des essais par un cache de compilation et des workers réutilisables, avec des mesures exclusives par GPU. | partiel | Aucun profil de compteurs réussi ni repliCUDA local effectif. |
| STATE-01 | Persister l’état LangGraph, l’arbre, les expériences, les budgets et les artefacts ; rendre l’intégration des résultats idempotente. | partiel | Réconcilier transport/reprise v2 sans double récompense ni mélange de mesures. |
| QUALITY-01 | Évaluer l’optimiseur à budgets comparables sur plusieurs familles de kernels, avec mesures répétées et ablations. | partiel | Étude partielle:premiers GEMM full/greedy3141 generation_complete avec promotions de recherche; aucune finale commune acquise. Achever144générations/18groupes et qualifier véritable original/nativev2; publier coûts, taux, dispersion et tous échecs. |

| Preuve | Sources et portée | SHA-256 du reçu |
| --- | --- | --- |
| diagnostic_loop | 76da03e4b80a4b1b4aff6390b3f57d76925cac6e ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `7bbe27e150d3f3292bc986ac235be705f4088ad1f60706eb20dc9c7c5f70d571` |
| autotuning | 88880bebfc7ed259325129261905fec9e50243ee ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `38dd03c2a58ca0fe476bf5ccf80cd7852af882d38f9074d34706d8fee8e2f071` |
| contracts_cache_reconstruction | 07d791acace7213075a1c805532ff7caadeb24a1 ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `5541ccaab222a13c3d9a5f6b55c059700017351b684fb557e8592249d9638ff7` |
| process_recovery | not embedded in receipt ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `7d59e2f85f526cf5a8e9142aee0470edc2025d979244d3f50638ee307cd486ec` |
| frozen_gemm | Hashes du code et des artefacts dans le registre ; Raw frozen generated kernels, not retained-best replacement; no original-upstream kernel; excluded from section16 cohorts | `47f2294905d7f6f3c8bb75e7be98db61c908c136b3656c47487cd610fde073ca` |
| rpc_transport | 6292b7d88467ebd528c816d5bdb9daf1760d3b86 ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `06456a6a1cd0f8b1917b171769aa6f16108b656e821e95206b20fc5f8882f1c5` |
| final_isolation | 6292b7d88467ebd528c816d5bdb9daf1760d3b86 ; Uniquement les contrôles explicitement nommés dans le reçu; aucune qualification globale. | `831afbe556adce37e6037cc9231a6525e80fd46a16ef6bee16ccf488ed128001` |
| reconstructed_native_finalist | 412d857391cd1627392f9695614a6369e781400f ; One correctness-only execution of an authenticated reconstructed actual native finalist; no new benchmark, gain or model call | `83ff0e36b012f0362a053a7a07a9b317a51e26411aa21b0f8644ce61cd5a9cb3` |
| native_vanilla | codex-cli 0.163.0-alpha.10.2 ; 770605 COMPLETED 0:0: véritable CLI Luna Max, un prompt, shell+serviceGPU. Finale indépendante770762 distincte; ancien run sans read-deny natif v2. | `1f48b0b29f80e8ef75a910a0292f06a221e99237d783bd9e09e3e0e5eb0f3894` |
| passages_v2_http | Hashes du code et des artefacts dans le registre ; TéléchargementsHTTP réels CPU: CCCL, CUTLASS, sections CUDA12.6. Aucun modèle/GPU; aucune synthèse/dossier v2/proposition mesurée. | `828a9da08477a59fea818cf2fa901a9fc4567b45c0fafba9501dfa51e7a161fb` |
| native_read_isolation_cpu | codex-cli 0.163.0-alpha.10.2 ; Profil natif macOS: parent/traversée/symlink/hardlink refusés; workspace et vrai clientHTTP GPU autorisés. CLI / contrôleur fiables, workspace neuf obligatoire; aucun modèle/GPU dans ce probe. | `fb314df377948ddfdd15a5c3b5a26a7eb0028f8735b5352c296af2bc32d559dd` |

Le pilote GEMM 770762, **COMPLETED 0:0**, compare quatre artefacts bruts figés : naïf, agent amélioré brut, proposition structurée sans outils, véritable CLI natif. Il couvre trois tailles et 48 contrôles Sanitizer. **Aucun original réel ne figure dans ce pilote**, exclu des cohortes §16. Agent / naïf : 0.830731 [0.823756, 0.835579] ; native / naïf : 1.120191 [1.101939, 1.142764] ; comparaison directe agent / native : 0.742984. La régression de l’agent reste visible, sans remplacement par le meilleur retenu.

L’isolation native macOS porte sur les accès fichiers des **outils du modèle**. Le CLI et le contrôleur fiables peuvent lire schéma et authentification. Un workspace neuf est obligatoire ; un hardlink préexistant fourni par le contrôleur reste hors garantie. Le probe CPU précède la soumission, mais ne prouve pas une génération native v2.

La première chaîne v2 réelle vérifiée conserve ces identités :

| Mode GEMM 3141 | Job terminal | Dossier v2 | Candidat / évaluation tree_reward fraîche | Gain de recherche, IC95% |
| --- | --- | --- | --- | --- |
| full | 772656 COMPLETED 0:0 | `751beb4adf294422ac1e42d61713920e` | `c39a56406a0df91a7d113c8e` / `8554ad0eb98b4a44bd308a70432a6897` | 1.098862 [1.081022, 1.106020] |
| greedy_parent | 772658 COMPLETED 0:0 | `7803c5205f6143b89e6978f403184654` | `8ffae8dda88e482c84ef0747` / `d33a76204c7043db94963bfb7f719217` | 1.132916 [1.125592, 1.140106] |

Ces promotions sont des résultats de recherche, **sans qualification finale commune**. Full : 7 recherches et26 requêtes observées, 22 déclarées ; 8 sources, 6 conditions applicables, 95 652 caractères fournis. Greedy :7 recherches et27 requêtes observées, 8 déclarées ; 12 sources, 5 conditions, 124 846 caractères. Douze passages ciblés par dossier sont explicitement partiels ; les fondations complètes et les lanceurs fournis restent distingués des en-têtes sans lanceur. Les reçus archivés, leur contenu exact et les hashes sont dans le registre. Le backend ne conserve pas séparément tous les octets du stdin ni son hash : la portée de preuve relie le manifeste fourni, les appels achevés et le code figé qui incorpore ce manifeste, sans prétendre une preuve de transport plus forte.

Cet audit ne relance aucun test, modèle, job ou connexion et ne lit aucun seed final privé. Les preuves et journaux existants restent inchangés.
