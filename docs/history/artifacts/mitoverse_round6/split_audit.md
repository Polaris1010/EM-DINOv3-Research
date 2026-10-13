# Round6 data gate

Local HF revision: d0af9a6fd11faaf16ce0822233b7bbf72553ce7d

Existing split retained: 31 train / 8 validation / 16 legacy_dev_test / 157 excluded. No difficult volume moved. The six test sources are previously consulted development tests.

**New SSL blocked:** all current target training volumes have incomplete original specimen/scan provenance. Grouping by publication is not proof of independent specimens. No fabricated identifiers or SSL allowlisting. JRC named scans overlap excluded MitoVerse OpenOrganelle assets; ROI origins are absent, so overlap extent is unresolved. MitoEM2 is an overlay, not independent data. CMCC/CMCT conversion assets are indexed separately and must not be assumed to define prior evaluation splits.

**Mask issue:** han24 carries region-valued masks. Sampled positive labels lie within nonzero mask, but the importer only says “ignore mask” without polarity. Formal han24 supervision remains blocked pending authoritative semantics; fixed-patch diagnostics use unmasked training sources.

Historical P1 exposure is retained as volume/domain/uncertain, not reset after exclusions. Old C10k cannot be reused: requested source→volume→slice sampling differs from R5 source→patch. T0 reuse cannot be certified while annotation-mask semantics remain unresolved.
