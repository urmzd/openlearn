-- lang.11 reference: connection settings for the usage ledger.
-- journal_mode is stored in the database file; the other three are
-- per connection, so every connection runs this file first.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;
