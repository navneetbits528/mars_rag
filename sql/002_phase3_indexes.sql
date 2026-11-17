CREATE INDEX IF NOT EXISTS chunks_parent_id_idx ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS parents_report_section_idx ON parents(report_no, section_path);
CREATE INDEX IF NOT EXISTS glossary_report_acronym_idx ON glossary(report_no, acronym);
CREATE INDEX IF NOT EXISTS facts_report_parameter_idx ON facts(report_no, parameter);
