# 2024-10-07	CSI Aurian Quelennec

Q: Vincent Lostanlen
- Linear probe fine-tuning ?
- Titre: perspnnalized / customized

Q: Pietro Gori
- alpha=0.9 -> est-ce que tu as cross-validé (est-ce que c'est statistiquement différent de alpha=1) ?
						-> est-ce que les deux loss vont dans la même direction ?
						-> ça ressemble un peu à Dino-V2
- intérêt de la deuxième loss	?
- est-ce que c'est important de les faire simultanément et pas l'une après l'autre ?
- linear probing marche bien sur les méthodes de masking

Le SSL est un "exercice de gratification retardé".


# 2024-10-07	CSI Antonin Gagneré

Comparison ZeroNS


Q: Vincent Lostanlen
- Why a Transformer, positional encoding ?
- Pourquoi l'embedding (appris par le Transformer) est de si grande dimension (512) ? que veut dire la distance en dimension 512 ?
	visualiser les clusters de beat 1, 2, 3, 4 avec tSNE

Q: Pietro Gori
- Pourquoi utiliser un seul positif ? (triplet loss) pourquoi pas plusieurs ?
- Pourquoi moitié négatif , moitié positif dans un batch ? il y a beaucoup de papiers la-dessus. Tu peux aussi pondéré (relatif à la négativité)



# 2024-10-07	CSI Xuanyu Zhuang

Q: Michel Crucianu
- what is the initial goal of the PhD ? why Image ?
- program for the second year ?
	1 hot music style transfer
- ProtoNet:
- Team ? collaboration with other PhD students ? yes with SSL and generative modeling
- Visit in ther Labs ? yes when I go to conferences

Q: Vincent Gripon
- you are inspired by old-fashioned FSL algorithm (ProtoNet, MAML), why not using foundation models which is the new hype in Computer Vision
- how you see the plan for the second year, which conference ?
	using SSL to hence episodic fine-tuning,
