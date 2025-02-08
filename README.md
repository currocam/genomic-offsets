# genomic-offsets
### Attention: this repository is a work in progress ⚠️

Genomic offsets (GO) statistics measure current individuals/populations’
maladaptation in the event of a drastic climate change. These metrics
are obtained from Genotype environment associations (GEA). This package
efficiently implements the four most popular genomic offset metrics:
[RONA (Rellstab et al. 2016)](RONA.html), [RDA GO (Capblancq and
Forester 2021)](rda.html), [Geometric GO (Gain et
al. 2023)](geometric.html), and [Gradient Forest GO (Fitzpatrick and
Keller 2014)](gf.html), as well as simulation templates that we suggest
should be used to explicitly consider under which assumptions empirical
genomic offsets can have predictive value (built on top of
[SLiM](https://messerlab.org/slim/),
[msprime](https://tskit.dev/software/msprime.html) and [tree-sequence
recording](https://tskit.dev/learn/)).

You can find more information in the [documentation web
page](https://currocam.github.io/genomic-offsets/).
