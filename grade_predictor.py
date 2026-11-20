"""Configurable Random Forest explorer for Excel and CSV datasets."""
import math
import tkinter as tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

from history import History, HistoryError, load_dataset
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor


def train_model(frame, target, features, task, test_fraction, trees, seed, compare=False):
    """Fit preprocessing only on training rows and evaluate on held-out rows."""
    if not features or target in features:
        raise ValueError("Select at least one input column, excluding the target.")
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("Test percentage must be between 10 and 50.")
    if not 1 <= trees <= 2000:
        raise ValueError("Trees must be between 1 and 2000.")
    if not 0 <= seed <= 4294967295:
        raise ValueError("Seed must be between 0 and 4294967295.")
    usable = frame.dropna(subset=[target]).copy()
    if len(usable) < 10:
        raise ValueError("At least 10 rows with a target value are required.")
    x = usable[features].copy()
    y = usable[target]
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two different values.")
    numeric = list(x.select_dtypes(include="number").columns)
    categorical = [column for column in features if column not in numeric]
    for column in categorical:
        x[column] = x[column].map(lambda value: str(value) if pd.notna(value) else float("nan"))
    if numeric and x[numeric].isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Numeric inputs contain infinite values. Correct these in the source file.")
    regression = task == "Regression"
    if regression:
        y = pd.to_numeric(y, errors="raise")
        if y.isin([float("inf"), float("-inf")]).any():
            raise ValueError("Target contains infinite values.")
    else:
        y = y.astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed
    )
    empty = list(x_train.columns[x_train.isna().all()])
    if empty:
        raise ValueError("These inputs have no training values: " + ", ".join(empty))
    transforms = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    estimator = RandomForestRegressor if regression else RandomForestClassifier
    prepare = ColumnTransformer(transforms)
    candidates = {
        "Random Forest": estimator(n_estimators=trees, random_state=seed, n_jobs=-1),
        "Decision Tree": (DecisionTreeRegressor if regression else DecisionTreeClassifier)(random_state=seed),
        "Baseline": DummyRegressor(strategy="mean") if regression else DummyClassifier(strategy="most_frequent"),
    }
    selected = "Random Forest"
    comparison = []
    if compare:
        folds = min(5, len(x_train))
        if regression:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            folds = min(folds, int(y_train.value_counts().min()))
            if folds < 2 or y_train.nunique() < 2:
                raise ValueError("Comparison needs at least two training rows per class and two classes. "
                                 "Use regression for numeric marks, or disable comparison.")
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = "neg_mean_absolute_error" if regression else "accuracy"
        best_score = -float("inf")
        for name, candidate in candidates.items():
            pipeline = Pipeline([("prepare", clone(prepare)), ("model", candidate)])
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=splitter, scoring=scoring, error_score="raise"
            )
            mean_score = float(scores.mean())
            comparison.append({
                "Model": name,
                "CV metric": "MAE (lower is better)" if regression else "Accuracy (higher is better)",
                "Mean": round(-mean_score if regression else mean_score, 4),
                "Std. deviation": round(float(scores.std()), 4),
                "Folds": folds,
            })
            if mean_score > best_score:
                best_score, selected = mean_score, name
    model = Pipeline([("prepare", prepare), ("model", candidates[selected])])
    model.fit(x_train, y_train)
    # Bind the prediction form to this fitted model, independent of later UI edits.
    model.input_schema_ = [
        {"name": column, "numeric": column in numeric,
         "minimum": float(x_train[column].min()) if column in numeric else None,
         "maximum": float(x_train[column].max()) if column in numeric else None,
         "categories": sorted(x_train[column].dropna().unique().tolist()) if column in categorical else []}
        for column in features
    ]
    model.target_name_ = target
    model.task_ = task
    model.selected_name_ = selected
    model.comparison_ = pd.DataFrame(comparison)
    predictions = model.predict(x_test)
    baseline = DummyRegressor(strategy="mean") if regression else DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    metrics = {
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "missing_targets_excluded": len(frame) - len(usable),
    }
    if regression:
        metrics.update(mae=float(mean_absolute_error(y_test, predictions)),
                       rmse=float(mean_squared_error(y_test, predictions) ** 0.5),
                       baseline_mae=float(mean_absolute_error(y_test, baseline_predictions)))
        score = (f"MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | "
                 f"Baseline MAE: {metrics['baseline_mae']:.2f}")
    else:
        metrics.update(accuracy=float(accuracy_score(y_test, predictions)),
                       baseline_accuracy=float(accuracy_score(y_test, baseline_predictions)))
        score = (f"Accuracy: {metrics['accuracy']:.1%} | "
                 f"Baseline accuracy: {metrics['baseline_accuracy']:.1%}")
    model.metrics_ = metrics
    unseen = set(y_test) - set(y_train) if not regression else set()
    note = f" | {len(unseen)} test classes absent from training" if unseen else ""
    summary = (
        f"Model: {selected} | Train: {len(x_train)} | Test: {len(x_test)} | "
        f"Missing targets excluded: {len(frame) - len(usable)}\n{score}{note}"
    )
    results = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions})
    model.evaluation_summary_ = summary
    return model, results, summary


def train_and_record(history, dataset, frame, configuration):
    model, results, summary = train_model(frame, **configuration)
    run_id = history.save_run(dataset, configuration, model) if history else None
    return model, results, summary, run_id


def predict_record(model, values):
    """Validate a single record against the fitted model's original input schema."""
    record = {}
    notes = []
    for field in model.input_schema_:
        name = field["name"]
        raw = values.get(name, "").strip()
        if not raw:
            raise ValueError(f"Enter a value for {name}.")
        if field["numeric"]:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number.") from None
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if value < field["minimum"] or value > field["maximum"]:
                notes.append(f"{name} is outside the training range.")
        else:
            value = raw
            if value not in field["categories"]:
                notes.append(f"{name} contains a category absent from training.")
        record[name] = value
    prediction = model.predict(pd.DataFrame([record]))[0]
    return prediction, notes


class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.frame = None
        self.dataset = None
        self.history = History.configured()
        self.run_id = None
        self.model = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        root.title("Grade Predictor and Model Comparison")
        root.geometry("1000x800")
        self.panel = ttk.Frame(root, padding=12)
        self.panel.pack(fill="both", expand=True)
        ttk.Button(self.panel, text="Upload Excel / CSV", command=self.open_file).pack(anchor="w")
        self.filename = ttk.Label(self.panel, text="Choose a dataset to begin.")
        self.filename.pack(anchor="w", pady=6)
        self.preview = self.table(self.panel, height=7)
        controls = ttk.Frame(self.panel)
        controls.pack(fill="x", pady=12)
        ttk.Label(controls, text="Target column").grid(row=0, column=0, sticky="w")
        self.target = ttk.Combobox(controls, state="readonly", width=26)
        self.target.grid(row=1, column=0, sticky="nw", padx=(0, 14))
        self.target.bind("<<ComboboxSelected>>", self.update_features)
        ttk.Label(controls, text="Input columns (Ctrl/Shift to select multiple)").grid(row=0, column=1, sticky="w")
        self.features = tk.Listbox(controls, selectmode="extended", exportselection=False, height=6, width=40)
        self.features.grid(row=1, column=1, sticky="w")
        settings = ttk.Frame(self.panel)
        settings.pack(fill="x")
        self.task = self.setting(settings, "Task", "Regression", 0, ["Regression", "Classification"])
        self.test = self.setting(settings, "Test %", "20", 1)
        self.trees = self.setting(settings, "Trees", "100", 2)
        self.seed = self.setting(settings, "Random seed", "20", 3)
        self.compare = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.panel, text="Compare models using training-set cross-validation",
                        variable=self.compare).pack(anchor="w", pady=5)
        self.predict_button = ttk.Button(self.panel, text="Predict a new record",
                                         command=self.open_prediction, state="disabled")
        self.predict_button.pack(anchor="w")
        self.comparison_button = ttk.Button(self.panel, text="View model comparison",
                                            command=self.show_comparison, state="disabled")
        self.comparison_button.pack(anchor="w")
        self.run_button = ttk.Button(self.panel, text="Train and evaluate", command=self.start_training)
        self.run_button.pack(anchor="w", pady=8)
        if self.history:
            ttk.Button(self.panel, text="PostgreSQL history", command=self.show_history).pack(anchor="w")
        self.status = ttk.Label(self.panel, text="Select inputs deliberately. Exclude IDs and data unavailable at prediction time.", wraplength=930)
        self.status.pack(anchor="w", pady=6)
        self.results = self.table(self.panel, height=12)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def table(self, parent, height):
        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True)
        tree = ttk.Treeview(holder, show="headings", height=height)
        vertical = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        return tree

    def setting(self, parent, label, default, column, choices=None):
        ttk.Label(parent, text=label).grid(row=0, column=column, sticky="w", padx=8)
        value = tk.StringVar(value=default)
        widget = ttk.Combobox(parent, textvariable=value, values=choices, state="readonly", width=18) if choices else ttk.Entry(parent, textvariable=value, width=18)
        widget.grid(row=1, column=column, padx=8)
        return value

    def display(self, tree, frame):
        tree.delete(*tree.get_children())
        tree["columns"] = list(frame.columns)
        for column in frame.columns:
            tree.heading(column, text=column)
            tree.column(column, width=160, anchor="center")
        for row in frame.itertuples(index=False, name=None):
            tree.insert("", "end", values=tuple("" if pd.isna(v) else str(v) for v in row))

    def open_file(self):
        if self.future is not None:
            messagebox.showinfo("Training", "Wait for training to finish before loading another dataset.")
            return
        path = filedialog.askopenfilename(filetypes=[("Datasets", "*.xlsx *.xls *.csv")])
        if not path:
            return
        try:
            frame, self.dataset = load_dataset(path)
            self.frame = frame
            self.model = None
            self.predict_button.state(["disabled"])
            self.comparison_button.state(["disabled"])
            self.filename.config(text=path)
            self.target["values"] = list(frame.columns)
            self.target.set("")
            self.features.delete(0, "end")
            self.display(self.preview, frame.head(100))
            self.display(self.results, pd.DataFrame())
            self.status.config(text=f"{len(frame)} rows loaded. Preview shows up to 100. Select a target and input columns.")
        except Exception as error:
            messagebox.showerror("Cannot load dataset", str(error))

    def update_features(self, event=None):
        self.features.delete(0, "end")
        for column in self.frame.columns:
            if column != self.target.get():
                self.features.insert("end", column)

    def start_training(self):
        if self.future is not None:
            return
        try:
            if self.frame is None or not self.target.get():
                raise ValueError("Upload a dataset and select its target column.")
            features = [self.features.get(i) for i in self.features.curselection()]
            configuration = {
                "target": self.target.get(), "features": features, "task": self.task.get(),
                "test_fraction": float(self.test.get()) / 100, "trees": int(self.trees.get()),
                "seed": int(self.seed.get()), "compare": self.compare.get(),
            }
            self.model = None
            self.predict_button.state(["disabled"])
            self.comparison_button.state(["disabled"])
            self.display(self.results, pd.DataFrame())
            self.future = self.executor.submit(
                train_and_record, self.history, self.dataset, self.frame.copy(), configuration
            )
            self.run_button.state(["disabled"])
            self.status.config(text="Training on the selected configuration…")
            self.root.after(100, self.poll_training)
        except Exception as error:
            messagebox.showerror("Check settings", str(error))

    def poll_training(self):
        if not self.future.done():
            self.root.after(100, self.poll_training)
            return
        try:
            self.model, results, summary, self.run_id = self.future.result()
            self.display(self.results, results)
            self.predict_button.state(["!disabled"])
            if not self.model.comparison_.empty:
                self.comparison_button.state(["!disabled"])
            self.status.config(text=summary + (f" | Saved run #{self.run_id}" if self.run_id is not None else ""))
        except Exception as error:
            self.status.config(text="Training failed. Review the dataset and settings.")
            messagebox.showerror("Training failed", str(error))
        finally:
            self.future = None
            self.run_button.state(["!disabled"])

    def background(self, window, work, on_success, on_error):
        future = self.executor.submit(work)

        def poll():
            if not window.winfo_exists():
                return
            if not future.done():
                self.root.after(150, poll)
                return
            try:
                result = future.result()
            except Exception as error:
                on_error(error)
            else:
                on_success(result)

        self.root.after(150, poll)

    def show_history(self):
        import json

        window = tk.Toplevel(self.root)
        window.title("PostgreSQL history")
        window.geometry("960x660")
        status = ttk.Label(window, text="Loading the latest training runs…")
        status.pack(anchor="w", padx=10, pady=6)
        runs = self.table(window, 6)
        details = ScrolledText(window, wrap="word", height=16, state="disabled")
        details.pack(fill="both", expand=True, padx=10, pady=6)
        controls = ttk.Frame(window)
        controls.pack(fill="x", padx=10, pady=6)
        records = {}
        generation = 0

        def show_details(value):
            details.configure(state="normal")
            details.delete("1.0", "end")
            details.insert("end", json.dumps(value, indent=2, default=str, ensure_ascii=False))
            details.configure(state="disabled")

        def show_error(error):
            status.config(text=str(error))

        def selection(_event=None):
            nonlocal generation
            generation += 1
            request = generation
            selected = runs.selection()
            if not selected:
                export.state(["disabled"])
                return
            record = records[runs.index(selected[0])]
            export.state(["!disabled"])
            show_details({"run": record})
            status.config(text=f"Loading predictions for run #{record['id']}…")

            def display_predictions(predictions):
                if request == generation:
                    show_details({"run": record, "predictions": predictions})
                    status.config(text=f"Run #{record['id']} · {len(predictions)} latest predictions")

            self.background(window, lambda: self.history.recent_predictions(record["id"]),
                            display_predictions, lambda error: show_error(error) if request == generation else None)

        def load():
            nonlocal generation
            generation += 1
            request = generation
            status.config(text="Loading the latest training runs…")

            def display_runs(rows):
                if request != generation:
                    return
                records.clear()
                records.update(enumerate(rows))
                self.display(runs, pd.DataFrame([
                    {"Run": row["id"], "Dataset": row["filename"], "Model": row["selected_model"],
                     "Created": row["created_at"].isoformat(timespec="seconds")}
                    for row in rows
                ], columns=["Run", "Dataset", "Model", "Created"]))
                export.state(["disabled"])
                show_details({})
                status.config(text=f"{len(rows)} latest runs. Select a row to see its details.")

            self.background(window, self.history.recent_runs, display_runs,
                            lambda error: show_error(error) if request == generation else None)

        def export_model():
            selected = runs.selection()
            if not selected:
                return
            run_id = records[runs.index(selected[0])]["id"]
            path = filedialog.asksaveasfilename(parent=window, initialfile=f"grade-run-{run_id}.joblib",
                                                defaultextension=".joblib", filetypes=[("Joblib model", "*.joblib")])
            if path:
                self.background(window, lambda: Path(path).write_bytes(self.history.model_artifact(run_id)),
                                lambda _: status.config(text=f"Model for run #{run_id} exported."), show_error)

        ttk.Button(controls, text="Refresh", command=load).pack(side="left")
        export = ttk.Button(controls, text="Export model", command=export_model, state="disabled")
        export.pack(side="left", padx=8)
        runs.bind("<<TreeviewSelect>>", selection)
        load()

    def show_comparison(self):
        if self.model is None:
            return
        window = tk.Toplevel(self.root)
        window.title("Training-set model comparison")
        window.geometry("850x300")
        ttk.Label(window, text=f"Selected: {self.model.selected_name_}. "
                  "Cross-validation uses training rows only; test metrics appear in the main window.",
                  wraplength=810).pack(padx=12, pady=12)
        self.display(self.table(window, 4), self.model.comparison_)

    def open_prediction(self):
        if self.model is None:
            return
        model = self.model
        run_id = self.run_id
        history = self.history
        window = tk.Toplevel(self.root)
        window.title(f"Predict {model.target_name_}")
        window.geometry("620x520")
        ttk.Label(window, text=f"Target: {model.target_name_} | Model: {model.selected_name_}",
                  wraplength=580).pack(padx=12, pady=12)
        ttk.Label(window, text="This form uses the inputs from the last successful training run.",
                  wraplength=580).pack(padx=12)
        canvas = tk.Canvas(window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True, padx=12, pady=8)
        form = ttk.Frame(canvas, padding=8)
        canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        entries = {}
        for row, field in enumerate(model.input_schema_):
            name = field["name"]
            ttk.Label(form, text=name).grid(row=row * 2, column=0, sticky="w", pady=5)
            value = tk.StringVar()
            entries[name] = value
            if field["numeric"]:
                widget = ttk.Entry(form, textvariable=value, width=30)
                hint = f"Training range: {field['minimum']:g} to {field['maximum']:g}"
            else:
                widget = ttk.Combobox(form, textvariable=value, values=field["categories"], width=30)
                hint = "Select a training category or enter a new value."
            widget.grid(row=row * 2, column=1, sticky="w", padx=10)
            ttk.Label(form, text=hint, wraplength=500).grid(
                row=row * 2 + 1, column=0, columnspan=2, sticky="w")
        output = ttk.Label(window, text="", wraplength=580)
        output.pack(padx=12, pady=10)

        def predict():
            from llm_eval import evidence_for
            try:
                evidence = evidence_for(model, {name: value.get() for name, value in entries.items()})
            except ValueError as error:
                output.config(text="")
                messagebox.showerror("Check input values", str(error), parent=window)
                return
            prediction = evidence["prediction"]
            shown = f"{prediction:.2f}" if model.task_ == "Regression" else prediction
            text = f"Predicted {model.target_name_}: {shown}"
            if evidence["warnings"]:
                text += "\n" + " ".join(evidence["warnings"])
            output.config(text=text)
            if history:
                predict_button.state(["disabled"])

                def finish(message):
                    output.config(text=text + "\n" + message)
                    predict_button.state(["!disabled"])

                self.background(window, lambda: history.save_prediction(run_id, evidence),
                                lambda record_id: finish(f"Saved prediction #{record_id}"),
                                lambda error: finish(f"History save failed. {error}"))

        predict_button = ttk.Button(window, text="Predict", command=predict)
        predict_button.pack(pady=(0, 12))

        def explain_local():
            from llm_eval import evidence_for, evaluate_explanation
            values = {name: value.get() for name, value in entries.items()}
            try:
                evidence = evidence_for(model, values)
            except ValueError as error:
                messagebox.showerror("Check inputs", str(error), parent=window)
                return
            report_window = tk.Toplevel(window)
            report_window.title("Local explanation and judge")
            report_window.geometry("700x500")
            report = ScrolledText(report_window, wrap="word", padx=12, pady=12)
            report.pack(fill="both", expand=True)
            report.insert("end", "Waiting for local Ollama…")
            report.configure(state="disabled")
            def evaluate():
                prediction_id = history.save_prediction(run_id, evidence) if history else None
                result = evaluate_explanation(evidence)
                if history:
                    try:
                        history.save_evaluation(prediction_id, result)
                    except HistoryError as error:
                        result["history_error"] = str(error)
                return result

            future = self.executor.submit(evaluate)
            llm_button.state(["disabled"])

            def poll():
                if not window.winfo_exists():
                    return
                if not future.done():
                    self.root.after(150, poll)
                    return
                try:
                    result = future.result()
                    judgment = result["judgment"]
                    text = (result["explanation"] + "\n\nJudge scores (1–5):\n"
                            + "\n".join(f"{key}: {value}" for key, value in judgment["scores"].items())
                            + f"\n\nAdvisory result: {'PASS' if judgment['passed'] else 'REVIEW'}"
                            + "\n" + judgment["reason"]
                            + "\n\nLLM judgments need human review.")
                    if result.get("history_error"):
                        text += "\nHistory save failed. " + result["history_error"]
                    elif history:
                        text += "\nSaved to PostgreSQL history."
                except Exception as error:
                    text = str(error)
                if report_window.winfo_exists():
                    report.configure(state="normal")
                    report.delete("1.0", "end")
                    report.insert("end", text)
                    report.configure(state="disabled")
                llm_button.state(["!disabled"])
            self.root.after(150, poll)

        llm_button = ttk.Button(window, text="Explain and judge with local Ollama", command=explain_local)
        llm_button.pack(pady=(0, 8))


    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PredictorApp(root)
    root.mainloop()
