# Datasheet: tinyshakespeare-bytes v1

Following Gebru et al., "Datasheets for Datasets".

## Motivation

- Built by the course maintainers to train and evaluate the reference
  example model shakespeare-kn4 (see [MODEL_CARD.md](MODEL_CARD.md)) and the
  small recurrent and transformer models of the spine, where a corpus that
  fits in one file and trains in seconds matters more than coverage.

## Composition

- One document: 1,115,394 bytes of plain ASCII text in 40,001 lines (32,777
  of them non-empty) of dialogue from Shakespeare's plays, each speech headed
  by the speaker's name.
- Splits by position: the first 90% (1,003,854 bytes) for training, the
  last 10% (111,540 bytes) for evaluation. A position split leaks style
  across the boundary but no lines are duplicated between the halves.
- Personal data: none about living people. The data.05 PII detector finds
  no span in the file (no emails, phone numbers, cards, IP addresses, or
  keys); character names are fictional or historical.

## Collection

- Source `tinyshakespeare`: the file `data/tinyshakespeare/input.txt` of the
  repository github.com/karpathy/char-rnn at revision
  6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e, license MIT (the repository; the
  plays themselves are in the public domain), retrieved 2026-10-09,
  sha256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
  Fetched once by course/oracle/MS-L3/tinyshakespeare.py and committed as a
  course fixture.

## Preprocessing

- No normalization: the bytes are used as they are (ASCII, Unix newlines).
- No quality filter: every line is kept, short speech headers included.
- Exact deduplication by line would drop 7,056 of the 32,777 non-empty lines
  (21.5%), mostly repeated speaker headings such as "ROMEO:"; they are kept
  because they are the structure of the text. No near deduplication.
- Decontamination: none of the 130 ethics.04 probe sentences and red-team
  prompts occurs in the text (exact substring search).
- Tokenization: bytes, ids 0 to 255.

## Uses

- Allowed: train and eval, per the license allowlist (MIT allows both).
- Avoid: any claim about modern English, about people, or about safety; the
  text is drama with violence and insults by design.

## Distribution and maintenance

- The file lives at course/fixtures/small-corpora/tinyshakespeare.txt; its
  manifest row records the sha256 above. If the upstream repository changed
  its license, drill ops.08's procedure applies: mark the source revoked in
  the ledger, purge derived token files, and record the retrain decision in
  this datasheet.
