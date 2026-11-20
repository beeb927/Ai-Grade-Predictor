# AI Grade Predictor

Train and compare machine learning models on a spreadsheet, then use the trained
model to predict a new record. Choose the target and input columns in the Tkinter
app. The project uses pandas and scikit-learn, with optional PostgreSQL history and
an optional local Ollama explanation and evaluation workflow.


The AI Grade Predictor is a Python desktop app that turns spreadsheet data into a reusable prediction model. Users can import an Excel or CSV file, choose the target and input columns, and use the trained model to make predictions for new records. The interface uses Tkinter, while pandas and scikit-learn handle data preparation, training, and evaluation.

The app can compare three models using cross-validation and report how the selected model performs on held-out data. The app also supports optional PostgreSQL storage for training runs, predictions, fitted models, and evaluation results. With Ollama enabled, a local language model explains individual predictions, and an automated judge scores those explanations on four criteria. A synthetic sample dataset is included so anyone can try training and prediction right away.

## Quick start

Use Python 3.12 with Tk support. On Linux, install your distribution's Tkinter
package if `python -m tkinter` cannot open a test window. Use a desktop session for
the GUI. The command-line evaluator can run headlessly once Tkinter is importable.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python grade_predictor.py
```

On macOS or Linux, use `python3.12 -m venv .venv` and `. .venv/bin/activate` in
place of the first two lines.

1. Open `examples/demo_grades.csv`. It contains synthetic data for trying the app.
2. Select **Mark on Test** as the target and **Hours of Sleep** and **Hours studied**
   as inputs. Select **Regression** and click **Train and evaluate**.
3. Open **Predict a new record** and try 8 hours of sleep and 4 hours of study.

The app also accepts your own CSV, XLSX or XLS file. Use synthetic or anonymized
data when sharing datasets or evaluation reports.

## Optional local explanations

Install [Ollama](https://ollama.com/download), start it, and run
`ollama pull qwen3:4b`. Then use **Explain and judge with local Ollama** in the
prediction form. The default endpoint is `http://127.0.0.1:11434/api/chat`.

## Set up PostgreSQL history on Windows

Install and start PostgreSQL locally. In pgAdmin, connect to your local server as
the PostgreSQL admin and open the Query Tool for the postgres database. Create
a dedicated login, replacing the sample password with your own.

```sql
CREATE ROLE grade_app LOGIN PASSWORD 'replace_with_your_local_password';
```

Run this separately in the Query Tool to create the application database.

```sql
CREATE DATABASE grade_predictor OWNER grade_app;
```

In the same Command Prompt with the virtual environment active, set the
connection variables before launching the desktop app or the evaluation runner.

```bat
set "GRADE_PREDICTOR_DSN=host=127.0.0.1 port=5432 dbname=grade_predictor user=grade_app"
set "PGPASSWORD=replace_with_your_local_password"
python history.py --init
python grade_predictor.py
```

Use your server's configured port if it differs from 5432. The environment variables
are scoped to the Command Prompt session. Keep your database password out of source
control. Running `python history.py --init` again preserves existing history.

1. Upload a dataset and train a model. The status line shows its saved run ID.
2. Open Predict a new record and click Predict to log a prediction. Explain and
   judge logs a prediction for that action and links the resulting LLM evaluation
   to it. If Ollama fails, the prediction still appears in history.
3. Open PostgreSQL history and select a training run. The window lists its
   settings, test metrics, cross-validation scores, predictions, and any judge results.
   Refresh after adding records. It displays up to 100 recent runs and 100 predictions
   per selected run.
4. Use Export model to download the fitted pipeline as a Joblib file. The run
   records the dependency versions used to train it. Load Joblib files only from
   sources you trust, and use matching training dependencies.

The desktop app and evaluation runner also run without PostgreSQL when
GRADE_PREDICTOR_DSN is unset. With it configured, successful evaluation cases
save into the same history and the JSON report includes their database IDs.
The separate judge calibration result remains in the JSON report.

Training history includes a dataset fingerprint and metadata, selected features,
split settings, model versions, metrics, and the fitted pipeline. PostgreSQL
stores input values you enter for individual predictions. Uploaded spreadsheets
and their raw rows stay outside the database.

## Model configuration

Defaults use qwen3:4b for both roles. Override in Command Prompt before launching:
set OLLAMA_EXPLAIN_MODEL=qwen3:4b
set OLLAMA_JUDGE_MODEL=qwen3:4b

Use locally downloaded tags. Larger models need more memory.
The same model in both roles can repeat its own mistakes.
Judge decisions are advisory. Expand calibration examples and review them manually.

## Evaluation

After setting up Ollama, run this from the project root:

```sh
python run_evals.py --data examples/demo_grades.csv --cases eval_cases.json
```

The case file matches the bundled synthetic CSV. Edit its target, features and cases to evaluate another dataset.

This evaluates loading, preprocessing, model selection, prediction, explanation and judging.
It includes invalid inputs, a range warning, and a deliberately bad explanation to check the judge.
Output is eval_results.json, containing configuration, evidence, explanation, scores and errors.
Review/error outcomes produce exit code 1. Desktop clicks are not automated.

For functional checks without Ollama:

```sh
python run_evals.py --data examples/demo_grades.csv --cases eval_cases.json --skip-llm
```

Skipped LLM stages are recorded as skipped, not passed.
All four judge scores must be at least 4/5 for an advisory pass.
This initial threshold has not been calibrated for reliability.
Prediction accuracy remains evaluated through MAE/RMSE or accuracy.

