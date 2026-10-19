-- lang.11 reference: an hourly rollup of the usage table, one row per
-- (tenant, hour). hour_ms is the start of the hour in Unix milliseconds.
CREATE TABLE IF NOT EXISTS usage_hourly (
    tenant            TEXT    NOT NULL,
    hour_ms           INTEGER NOT NULL CHECK (hour_ms % 3600000 = 0),
    requests          INTEGER NOT NULL DEFAULT 0 CHECK (requests >= 0),
    prompt_tokens     INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    PRIMARY KEY (tenant, hour_ms)
);
