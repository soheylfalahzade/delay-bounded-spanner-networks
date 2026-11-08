# Benchmark-bearing literature (venues checked against publisher / proceedings / institutional records)

**Corrections to the previous version of this table:** (i) arXiv:2107.02018 *Spanner Approximations in Practice* is by Chimani and Stutzenstein, not "Angrick et al."; (ii) "JCSS 119:60-77, 2021" belongs to a different paper (Enright et al., epidemic size in temporal networks); the journal version of *Temporal Cliques Admit Sparse Spanners* is doi:10.1016/j.jcss.2021.04.004 (volume/pages still to be copied from the publisher page).

| # | Reference | Venue (confirmed) | Still to confirm | Role in this repository |
|---|---|---|---|---|
| 1 | M. Farshi, J. Gudmundsson. *Experimental Study of Geometric t-Spanners.* | ESA 2005, LNCS 3669, pp. 556-567 | - | Protocol and metrics for `static` |
| 2 | M. Farshi, J. Gudmundsson. *Experimental Study of Geometric t-Spanners: A Running Time Comparison.* | WEA 2007, LNCS 4525, pp. 270-284 | - | Running-time protocol |
| 3 | F. Anderson, A. Ghosh, M. Graham, L. Mougeot, D. Wisnosky. *Bounded-Degree Plane Geometric Spanners in Practice.* | ACM J. Exp. Algorithmics 28(1), Art. 1.1, April 2023, doi:10.1145/3582497 | - | Stretch-factor estimation, lightness/degree metrics |
| 4 | G.V. Batz, R. Geisberger, P. Sanders, C. Vetter. *Minimum Time-Dependent Travel Times with Contraction Hierarchies.* | ACM J. Exp. Algorithmics 18, Art. 1.4, 2013 | - | Time-dependent road-network protocol |
| 5 | D. Delling, A.V. Goldberg, T. Pajor, R.F. Werneck. *Customizable Route Planning in Road Networks.* | Transportation Science 51(2):566-591, 2017 | - | Related work (metric-independent preprocessing) |
| 6 | Y. Li, R. Yu, C. Shahabi, Y. Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* | ICLR 2018 | - | METR-LA / PEMS-BAY (`metrla`) |
| 7 | A. Casteigts, J.G. Peters, J. Schoeters. *Temporal Cliques Admit Sparse Spanners.* | ICALP 2019 (LIPIcs 134, doi:10.4230/LIPIcs.ICALP.2019.134); JCSS, doi:10.1016/j.jcss.2021.04.004 | JCSS volume and pages | Temporal-graph spanners (journey model; contrast) |
| 8 | S. Angrick, B. Bals, T. Friedrich, H. Gawendowicz, N. Hastrich, N. Klodt, P. Lenzner, J. Schmidt, G. Skretas, A. Wells. *How to Reduce Temporal Cliques to Find Sparse Spanners.* | ESA 2024, LIPIcs 308, pp. 11:1-11:15, doi:10.4230/LIPIcs.ESA.2024.11 | - | Temporal-graph spanners (contrast) |
| 9 | M. Chimani, F. Stutzenstein. *Spanner Approximations in Practice.* | arXiv:2107.02018 (July 2021); ESA 2022, LIPIcs 244, Art. 37, doi:10.4230/LIPIcs.ESA.2022.37 | - | General-graph spanner experiments (`general`) |
| 10 | F. Bökler, M. Chimani, H. Jasper. *General Multiplicative Spanners in Practice.* | SEA 2026, LIPIcs 371, Art. 8, pp. 8:1-8:21, doi:10.4230/LIPIcs.SEA.2026.8 | - | Decoupled weight/length spanners; Greedy as practical standard |
| 11 | S. Baswana, S. Sen. *A Simple and Linear Time Randomized Algorithm for Computing Sparse Spanners in Weighted Graphs.* | Random Structures & Algorithms 30(4):532-563, 2007 (ICALP 2003, LNCS 2719, pp. 384-396 for the conference version) | - | (2k-1)-spanner baseline (`general`) |
| 12 | I. Althofer, G. Das, D. Dobkin, D. Joseph, J. Soares. *On Sparse Spanners of Weighted Graphs.* | Discrete & Computational Geometry 9:81-100, 1993 | - (already in README) | Path-greedy spanner (`static`) |
| 13 | C. Demetrescu, A.V. Goldberg, D.S. Johnson (eds.). *The Shortest Path Problem: Ninth DIMACS Implementation Challenge.* | DIMACS Series in Discrete Mathematics and Theoretical Computer Science 74, AMS, 2009, doi:10.1090/dimacs/074 | - | Standard static road-network instances (`dimacs`) |

Implementation cross-check available in NetworkX: `networkx.algorithms.sparsifiers.spanner` implements Baswana-Sen (documented reference: RSA 30(4):532-563, 2007).

## Additional references (added with README 5l-5m; metadata not yet checked against publisher records)

| # | Reference | Confirmed from | Still to confirm | Role in this repository |
|---|---|---|---|---|
| 15 | A. Filtser, S. Solomon. *The Greedy Spanner is Existentially Optimal.* | arXiv:1605.06852 (preliminary version PODC 2016; SICOMP 2020 according to the arXiv record) | SICOMP volume, issue and pages; exact statement and number of the lemma that the only t-spanner of a greedy t-spanner is itself | Related work; forced edges vs greedy |
| 16 | H. Le, S. Solomon. *Truly Optimal Euclidean Spanners.* | FOCS 2019 proceedings listing | Pages, DOI | Related work |
| 17 | H. Le, S. Solomon, C. Than, C. D. Tóth, T. Zhang. *Towards Instance-Optimal Euclidean Spanners.* | arXiv listing (authors, title) | Venue, identifier | Instance optimality of greedy fails for Euclidean point sets |
| 18 | Casteigts, Peters, Schoeters (see row 7) | see row 7 | JCSS volume and pages | Temporal spanners (contrast) |
| - | M. A. Abam, M. de Berg, M. Farshi, J. Gudmundsson. *Region-Fault Tolerant Geometric Spanners.* | DCG 41(4):556-582, 2009 (README reference 1) | Check the formal definition quoted in README 5l against the paper's text (reference graph = complete graph minus F) | Definition used in 5l |
| 19 | U. Brandes, D. Handke. *NP-completeness results for minimum planar spanners.* DMTCS 3(1):1-10, 1998 | doi:10.46298/dmtcs.257 (abstract) | none | Hardness (docs/RELATED_WORK.md, section 3) |
| 20 | D. Bilo, G. D'Angelo, L. Guala, S. Leucci, M. Rossi. *Sparse temporal spanners with low stretch.* ESA 2022, LIPIcs 244, 19:1-19:16 | doi:10.4230/LIPIcs.ESA.2022.19 | none | Temporal spanners (contrast) |
| 21 | Same authors. *Blackout-tolerant temporal spanners.* JCSS 141 (2024) 103495 | doi:10.1016/j.jcss.2023.103495 | none | Fault model in time (contrast) |
| 22 | S. Chechik, M. Langberg, D. Peleg, L. Roditty. *Fault tolerant spanners for general graphs.* STOC 2009, 435-444; SICOMP 39(7):3403-3423, 2010 | publisher and library records | none | Fault tolerance, G-F convention |
| 23 | M. Dinitz, R. Krauthgamer. *Fault-tolerant spanners: better and simpler.* PODC 2011 | arXiv:1101.5753 | PODC pages | Fault tolerance |
| 24 | L. Roditty, M. Thorup, U. Zwick. *Roundtrip spanners and roundtrip routing in directed graphs.* ACM TALG 4(3), 2008 | ACM record | none | Directed spanners (section 5) |
| 25 | M. Farshi, M. J. Hekmat Nasab. Experimental study of greedy spanners. *Scientia Iranica* 21 (2014) 2142-2152 | page header of the PDF | exact title | Greedy spanners in practice |
