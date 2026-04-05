-- This is init file
-- LiftLens initial schema
-- Applied automatically by postgres container on first boot

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── Jobs ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id    TEXT NOT NULL UNIQUE,
    object_name TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | error
    progress    INTEGER NOT NULL DEFAULT 0,        -- 0-100
    stage       TEXT NOT NULL DEFAULT 'Queued',
    error       TEXT,
    wandb_url   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status   ON jobs(status);

-- ── Results (full artifact stored as JSONB) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS results (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id    TEXT NOT NULL UNIQUE REFERENCES jobs(video_id) ON DELETE CASCADE,
    artifact    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_video_id ON results(video_id);
CREATE INDEX IF NOT EXISTS idx_results_artifact_gin ON results USING gin(artifact);

-- ── Backups log ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backup_log (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename    TEXT NOT NULL,
    size_bytes  BIGINT,
    destination TEXT NOT NULL,   -- 'minio' | 'local'
    status      TEXT NOT NULL DEFAULT 'ok',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Auto-update updated_at trigger ──────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_updated_at ON jobs;
CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
