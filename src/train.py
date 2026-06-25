import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu
    # Tinh phan phoi nhan (ty le mau cua tung lop 0, 1, 2) trong tap huan luyen
    target_counts = df_train["target"].value_counts(normalize=True)
    distribution = {f"class_{c}_pct": float(target_counts.get(c, 0.0)) for c in [0, 1, 2]}
    for c in [0, 1, 2]:
        pct = distribution[f"class_{c}_pct"]
        if pct < 0.10:
            print(f"WARNING: Lop {c} chiem {pct:.2%} (< 10%) tong so mau trong tap huan luyen!")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen mo hinh phu hop (Bonus 2)
        model_type = params.get("model_type", "random_forest")
        model_params = {k: v for k, v in params.items() if k != "model_type"}

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            # Set default max_iter if not present
            if "max_iter" not in model_params:
                model_params["max_iter"] = 1000
            model = LogisticRegression(**model_params, random_state=42)
        else:
            raise ValueError(f"Thuong hieu model_type khong hop le: {model_type}")

        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Model Type: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # Them ty le phan phoi nhan vao metrics.json (Bonus 5)
        os.makedirs("outputs", exist_ok=True)
        metrics_data = {
            "accuracy": acc,
            "f1_score": f1,
            **distribution
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=4)

        # Bonus 3: Báo Cáo Hiệu Suất Tự Động
        # Tinh toan confusion matrix va precision / recall cho tung lop
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(y_eval, preds)
        report_text = classification_report(y_eval, preds, zero_division=0)
        with open("outputs/report.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report_text + "\n")
        print("Da luu bao cao hieu suat vao outputs/report.txt.")

        # TODO 9: Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    # Kiem tra neu bien moi truong MLflow tracking duoc set tu truoc
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
