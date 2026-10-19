-- lang.11 reference: equality column first, then the range column, so the
-- served-model query searches one contiguous slice of the index.
CREATE INDEX IF NOT EXISTS usage_served_ts ON usage (served_model, ts_ms);
