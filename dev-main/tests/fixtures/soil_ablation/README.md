# Photographs for the Ollama ablation

These are the identical prepared JPEG inputs used in the five live dialogues.
`cases.json` records each image's attribution, licence, source page, modifications
and SHA-256. Each image remains under its listed CC BY-SA 3.0 or 4.0 licence.
Follow the licence link for the applicable terms.

The reference groups are captions from the published sources, not independent
expert labels. Russian, WRB and Canadian classifications are not automatically
equivalent. Reference labels, source titles and filenames are never sent to
the answering model. Only the visitor's question and image are sent.

The first question and later visitor-supplied facts are combined into one turn
to keep the same facts across modes without different assistant histories.
Default cases: 01 (Podzol) and 06 (Histosol); `--all` includes all five photos.
