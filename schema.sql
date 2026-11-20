CREATE TABLE IF NOT EXISTS grade_datasets (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename text NOT NULL,
    sha256 text NOT NULL,
    row_count integer NOT NULL CHECK (row_count >= 0),
    columns jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (sha256, filename)
);

CREATE TABLE IF NOT EXISTS grade_training_runs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id bigint NOT NULL REFERENCES grade_datasets(id),
    configuration jsonb NOT NULL,
    selected_model text NOT NULL,
    metrics jsonb NOT NULL,
    comparison jsonb NOT NULL,
    artifact bytea NOT NULL,
    versions jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS grade_predictions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES grade_training_runs(id),
    source text NOT NULL CHECK (source IN ('desktop', 'eval')),
    case_name text,
    evidence jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS grade_llm_evaluations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id bigint NOT NULL UNIQUE REFERENCES grade_predictions(id),
    explainer_model text NOT NULL,
    judge_model text NOT NULL,
    explanation text NOT NULL,
    judgment jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS grade_runs_by_dataset ON grade_training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS grade_predictions_by_run ON grade_predictions(run_id, id);
