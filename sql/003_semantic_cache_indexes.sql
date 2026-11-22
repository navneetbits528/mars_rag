-- Supporting indexes and cleanup-friendly access for verified semantic cache entries.
CREATE INDEX IF NOT EXISTS semantic_cache_verified_expiry_idx
    ON semantic_cache(verified, expires_at);

CREATE INDEX IF NOT EXISTS semantic_cache_key_expiry_idx
    ON semantic_cache(cache_key, expires_at);

-- The cache lookup always filters by cache_key and verified=true before
-- applying vector similarity. Expired rows are removed by the application.
