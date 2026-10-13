# MitoVerse probe split v1

- train: 31 volumes; han24, muller21, kunduri22, turner22
- validation: 8 volumes; guay21, mekuc20
- test1: 8 volumes; wei20, jiang25, casser20
- test2: 3 volumes; xiao18, haberl18
- test3: 5 volumes; conrad23
- excluded: 157 volumes; openorganelle

Whole source grouping is conservative, not a claim of verified individual-animal identities across publications. OpenOrganelle is excluded because P2 selection used CMCC/CMCT. Test1 includes documented SSL-source exposure (EM MitoEM/Lucchi and P2 Jiang vol1), with per-volume uncertainty retained. Test2 is anisotropic/cross-acquisition challenge. Test3 is an unseen-candidate source, NOT proven SSL-unseen. Strict unseen summary must be unavailable if no rows qualify. No test labels or predictions used for protocol selection. Training source sampling is equal, then volumes equal.


Historical evaluation exposure: conrad23 supplied five validation patches in the round1 smoke (see mitotopo_20260908_first_round/notes.md). No round4 test data is used for tuning, but conrad23 is not a historically pristine sealed test. Interpret it as exploratory held-out-source evaluation.
