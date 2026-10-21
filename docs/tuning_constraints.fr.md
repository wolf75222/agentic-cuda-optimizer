# Contraintes de l'autotuning

Une proposition peut déclarer `constraints`, une liste de clauses structurées.
Le langage ne contient aucune expression Python et n'utilise pas `eval`.
Les noms doivent appartenir au domaine de paramètres de la proposition.
Les clauses sont combinées par ET et vérifiées avant compilation CUDA.

```json
[
  {"op":"bounds", "parameters":["BLOCK_X"], "values":[8,256]},
  {"op":"divisible", "parameters":["TILE","BLOCK_X"], "values":[]},
  {"op":"aligned", "parameters":["STRIDE_BYTES"], "values":[16,0]},
  {"op":"forbid", "parameters":["BLOCK_X","STAGES"], "values":[256,8]},
  {"op":"product_le", "parameters":["BLOCK_X","BLOCK_Y"], "values":[1024]}
]
```

`divisible(a,b)` vérifie `a % b == 0`. `divides(a,[N])` vérifie
`N % a == 0` pour une constante déclarée. `aligned(a,[m,o])` vérifie
`(a-o) % m == 0`. `forbid` rejette la combinaison exacte indiquée.
`bounds` inclut ses deux bornes. Les comparaisons `le`, `lt`, `eq`, `ne`,
`ge` et `gt` acceptent deux paramètres, ou un paramètre et une constante.
Les valeurs sont des entiers ; les booléens, expressions, références inconnues
et arités incorrectes sont refusés. Une division dynamique par zéro filtre
la configuration.

Les limites du GPU restent vérifiées séparément à partir de ses propriétés
réelles. Ces clauses complètent les bindings du source et du lancement ; elles
ne modifient ni les tolérances, ni l'oracle, ni le contrat mémoire. Une
configuration filtrée est comptée et journalisée sans appel GPU. Les
configurations admissibles passent ensuite la validation indépendante complète
avant mesure. Le gagnant est remesuré sur de nouvelles séries.

Le mode expérimental sans autotuning vérifie aussi les clauses sur son candidat
concret. Les anciens reçus sans ce champ sont lus avec une liste vide. Les
propositions nouvelles le fournissent explicitement. `constraints.json`, les
événements de campagne et les reçus du tuner conservent les déclarations et les
rejets. Les nombres de variantes par essai définissent le protocole ; aucun
plafond global de tokens, coût, appels ou durée cumulée n'est ajouté.
