ALTER TABLE facts
ADD COLUMN IF NOT EXISTS domain TEXT NOT NULL DEFAULT 'General';

ALTER TABLE facts
ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'Other';

CREATE INDEX IF NOT EXISTS facts_domain_idx
ON facts(domain);

CREATE INDEX IF NOT EXISTS facts_category_idx
ON facts(category);