# Cover Detection

- author: Geoffroy
- code: based on ???
- datasets: datatacos

## Goal of the task ?

Cover(Version) Detection(Identification) is the task aiming at detecting if a given music track is a cover/version of another track.
For example detecting that this [Aretha Franklin](https://www.youtube.com/watch?v=LsU_HDS4cGE) is a cover/version of this [Beathes](https://www.youtube.com/watch?v=QDYfEBY9NM4). They are covers/versions of the same composition (work-id or ISWC).

Considering the very large number of possible work-id it is not possible to solve this as a classification (multi-class) problem (too many classes).
To solve this, the approach commonly used is to have a reference dataset $R$, containing tracks $\{r_i\}$ with known work-id, and to compare the query track $q$ to each track $r_i$ the reference dataset. If one track $q$ is similar to one track of the dataset (the distance $d(q,r_i)$ is small) , we decide that $q$ is a cover of $r_i$ and they share the same work-id.
This involves setting a threshold $\tau$ on $d(q,r_i)$. If $d(q,r_i)<\tau$ we decide they are cover of each other.

In practice to evaluate the task, another problem is considered. The distances between $q$ and the $r_i$ are compuated and ranked. If we denote by $w(.)$ the function that give the work-id of a track, we then check at which position in the ranked list $w(r_i)==w(q)$.
We can then use the ranking/recommendation performance metrics.

## How is the task evaluated ?

The algorithm use allow to compute a distance/similarity between two tracks $A$ and $B$.
It is applied to get if a query track $A$ is a cover of a track already existing in a datasets $B \in {b_1, b_2, b_3, \ldots\}$
For this we compute the distance between $A$ and each track of the dataset $d(A,b_1)$, $d(A,b_2)$, $d(A,b_3)$, ...
We then rank the of distances (from the smallest to the largest).
From the ranked list we computer
- mean rank
- mean reciprocal rank
- precision @ k
- mean average precision

## Some popular datasets

A (close to) exhaustive list of MIR datasets is available in the [ismir.net web site](https://ismir.net/resources/datasets/).

The first dataset proposed for this task was the ([cover80](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/) containing 80 different work-id (or clique) with each 2 versions.

Since then, much larger datasets have been created mostly relying on the data provided by the collaborative website [SecondHandSongs](https://secondhandsongs.com/).
For our implementations, we will consider the two following ones (notes that those do not provide access to the audio but t already extracted audio features):
- [Cover-1000](https://www.covers1000.net/dataset.html)
- [DA-TACOS](https://github.com/MTG/da-tacos)


## How we can solve it using deep learning

The usual way to solve the cover version problem is to develop an algorithm that allows to compute a distance between two tracks $q$ and $r_i$ which is related to their coverness.

Before the rise of deep learning, the usual way to compute this distance was to compute the cost of a DTW alignement between the sequence of chroma of $q$ and the one of $r_i$. This was however very costly in terms of computation time and prevented the algorithm to scale.

Today, the common deep learning technique used is based on metric learning, i.e. we train a neural network $f_{\theta}$ such that the resulting projections of $q$, $f_{\theta}(q)$ can be directly compared (using Euclidean distance) to the projections of $r_i$, $f_{\theta}(r_i)$. In this case, only the projections (named embedding) of the elements of the reference-set $R$ are stored and the comparison simply reduce to the computation of Eucliden distances.
Various approaches can be used for metric learning, but the most common is the [triplet loss](lab_triplet).

For the proposal code we will used the [MOVE model](https://arxiv.org/pdf/1910.12551) {cite}`DBLP:conf/icassp/YesilerSG20` and follow its [implementation](https://github.com/furkanyesiler/move).

![task_cover_move](/images/task_cover_move.png)
