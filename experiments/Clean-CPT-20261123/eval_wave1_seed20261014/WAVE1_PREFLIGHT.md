# Clean-CPT Wave 1 preflight

2026-11-24 20:08 HKT. Both hosts had zero eligible idle GPUs. Physical GPUs
1, 2 and 3 were occupied by user-owned Python PID 351263 on .98, or PID
885278 on .97, with roughly 16.4-16.6 GiB per card. Wave 1 was not launched.

Final checkpoint SHA-256 values, computed from files:
- P0: 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035
- Clean-C0 100k: a5e4d0fcc804ed21db23b1e70292b9b89fe4722cac16c369d6e4711f8f4f1260
- Clean-C1 100k: 22c3104f954b15f1259c52ae969898034e8cae61469cf183bea3615b0b34d250
- Clean-C1 0-30k teacher: f1cb92880a86b14939e23100d7ed5838dd0f9366dddf79ffc8b51fe59abf0a09

Clean-C1 0-30k also has the 5,607,208,037-byte full checkpoint. Final rank-0
loss (total, weighted barrier): C0 100k (13.3125, disabled), C1 100k
(13.1261063, 0.00110657), C1 0-30k (14.5625181, 0.0000183173).

Local MitoVerse MitoEM 2.0 overlay has one validation and ten test volumes
with paired image/instance arrays, 5,202 raw axial planes total. Exact volume
IDs are absent from the Clean-CPT MitoVerse training manifest. This evidence
does not establish per-image provenance across all CEM archives or P0. Use
Han24 validity masks per protocol. The overlay does not enumerate ME2-Pyra
test coordinates. CellMap official test inventory contains 16 crops from
six acquisitions; no corresponding local official test GT was found.
Existing local CellMap training labels and an alternate kidney crop are
not official test GT.
