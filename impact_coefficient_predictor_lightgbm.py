import sys
import os
import traceback
from typing import List

import joblib
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "Impact Coefficient Predictor"
APP_SUBTITLE = "LightGBM-based desktop tool for single-case and batch prediction"
FEATURES: List[str] = ["m", "theta", "phi", "Vx", "Vz", "xi", "IRI", "Hp", "L"]
DEFAULT_MODEL_PATH = r"C:\MODEL\BO-LGB.joblib"

HELP_TEXT = """Single-case mode
• Enter the 9 input variables.
• Click Predict to obtain the impact coefficient and grade.

Batch mode
• Export the CSV template.
• Fill in the required 9 columns.
• Load the CSV file and run batch prediction.
• Save the output CSV with I_pred and Grade.

Impact-grade thresholds
• I < 0.3: Low-impact
• 0.3 ≤ I ≤ 0.6: Medium-impact
• I > 0.6: High-impact
"""


def grade_from_I(value: float) -> str:
    if value < 0.3:
        return "Low-impact"
    if value <= 0.6:
        return "Medium-impact"
    return "High-impact"


class PredictorBackend:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found:\n{self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_single(self, values: List[float]) -> float:
        self._ensure_model_loaded()
        x = pd.DataFrame([values], columns=FEATURES)
        return float(self.model.predict(x)[0])

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_model_loaded()
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        work = df.copy()
        for col in FEATURES:
            work[col] = pd.to_numeric(work[col], errors="raise")

        pred = self.model.predict(work[FEATURES])
        work["I_pred"] = pred
        work["Grade"] = [grade_from_I(float(v)) for v in pred]
        return work

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            self.load_model()


class MetricCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardLabel")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("cardValue")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


class SingleCaseTab(QWidget):
    def __init__(self, backend: PredictorBackend):
        super().__init__()
        self.backend = backend
        self.inputs = {}
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(14)

        intro = QLabel(
            "Enter one case to obtain the predicted impact coefficient and its corresponding impact grade."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionIntro")
        main_layout.addWidget(intro)

        content_row = QHBoxLayout()
        content_row.setSpacing(14)

        input_box = QGroupBox("Single-case input")
        input_layout = QFormLayout(input_box)
        input_layout.setContentsMargins(18, 20, 18, 18)
        input_layout.setHorizontalSpacing(16)
        input_layout.setVerticalSpacing(12)

        for feature in FEATURES:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Enter {feature}")
            line_edit.setMinimumHeight(36)
            self.inputs[feature] = line_edit
            input_layout.addRow(f"{feature}", line_edit)

        content_row.addWidget(input_box, 3)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)

        output_box = QGroupBox("Prediction output")
        output_layout = QVBoxLayout(output_box)
        output_layout.setContentsMargins(18, 20, 18, 18)
        output_layout.setSpacing(12)

        self.pred_card = MetricCard("Predicted impact coefficient I")
        self.grade_card = MetricCard("Predicted grade")
        output_layout.addWidget(self.pred_card)
        output_layout.addWidget(self.grade_card)
        output_layout.addStretch(1)

        right_panel.addWidget(output_box)

        button_box = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_box)
        button_layout.setContentsMargins(18, 20, 18, 18)
        button_layout.setSpacing(10)

        self.predict_btn = QPushButton("Predict")
        self.clear_btn = QPushButton("Clear")
        self.predict_btn.setMinimumHeight(40)
        self.clear_btn.setMinimumHeight(40)

        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch(1)

        right_panel.addWidget(button_box)
        content_row.addLayout(right_panel, 2)

        main_layout.addLayout(content_row)
        main_layout.addStretch(1)

        self.predict_btn.clicked.connect(self.on_predict)
        self.clear_btn.clicked.connect(self.on_clear)

    def on_predict(self) -> None:
        try:
            values = []
            for feature in FEATURES:
                text = self.inputs[feature].text().strip()
                if text == "":
                    raise ValueError(f"Please enter a value for {feature}.")
                values.append(float(text))

            pred = self.backend.predict_single(values)
            self.pred_card.set_value(f"{pred:.6f}")
            self.grade_card.set_value(grade_from_I(pred))
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def on_clear(self) -> None:
        for line_edit in self.inputs.values():
            line_edit.clear()
        self.pred_card.set_value("—")
        self.grade_card.set_value("—")


class BatchTab(QWidget):
    def __init__(self, backend: PredictorBackend):
        super().__init__()
        self.backend = backend
        self.current_df = None
        self.pred_df = None
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(14)

        intro = QLabel(
            "Load a CSV file with the required 9 columns to perform multi-case prediction and export the results."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionIntro")
        main_layout.addWidget(intro)

        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        workflow_box = QGroupBox("Batch workflow")
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_layout.setContentsMargins(18, 20, 18, 18)
        workflow_layout.setSpacing(10)

        self.export_template_btn = QPushButton("Export CSV Template")
        self.load_csv_btn = QPushButton("Load CSV File")
        self.run_batch_btn = QPushButton("Run Batch Prediction")
        self.save_result_btn = QPushButton("Save Output CSV")

        for btn in [self.export_template_btn, self.load_csv_btn, self.run_batch_btn, self.save_result_btn]:
            btn.setMinimumHeight(40)
            workflow_layout.addWidget(btn)
        workflow_layout.addStretch(1)

        top_row.addWidget(workflow_box, 2)

        status_box = QGroupBox("File status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(18, 20, 18, 18)
        status_layout.setSpacing(10)

        self.file_label = QLabel("No CSV file loaded.")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("fileStatus")
        status_layout.addWidget(self.file_label)

        self.rows_card = MetricCard("Rows in current file")
        self.rows_card.set_value("0")
        status_layout.addWidget(self.rows_card)
        status_layout.addStretch(1)

        top_row.addWidget(status_box, 3)
        main_layout.addLayout(top_row)

        preview_box = QGroupBox("Preview and log")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(18, 20, 18, 18)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Batch status, file preview, and output preview will appear here.")
        preview_layout.addWidget(self.log_box)
        main_layout.addWidget(preview_box)

        self.export_template_btn.clicked.connect(self.export_template)
        self.load_csv_btn.clicked.connect(self.load_csv)
        self.run_batch_btn.clicked.connect(self.run_batch_prediction)
        self.save_result_btn.clicked.connect(self.save_result)

    def export_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV Template", "template.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            pd.DataFrame(columns=FEATURES).to_csv(path, index=False, encoding="utf-8-sig")
            self._log(f"Template saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Template Error", str(exc))

    def load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                raise ValueError("Missing required columns: " + ", ".join(missing))
            self.current_df = df
            self.pred_df = None
            self.file_label.setText(f"Loaded file:\n{path}")
            self.rows_card.set_value(str(len(df)))
            self._log(f"Loaded CSV with {len(df)} rows.")
            self._log("Columns: " + ", ".join(df.columns.astype(str)))
            self._log("Preview:\n" + df.head().to_string(index=False))
        except Exception as exc:
            QMessageBox.critical(self, "Load CSV Error", str(exc))

    def run_batch_prediction(self) -> None:
        if self.current_df is None:
            QMessageBox.warning(self, "No Input File", "Please load a CSV file first.")
            return
        try:
            self.pred_df = self.backend.predict_batch(self.current_df)
            self._log(f"Batch prediction completed for {len(self.pred_df)} rows.")
            self._log("Output preview:\n" + self.pred_df.head().to_string(index=False))
        except Exception as exc:
            QMessageBox.critical(self, "Batch Prediction Error", str(exc))

    def save_result(self) -> None:
        if self.pred_df is None:
            QMessageBox.warning(self, "No Prediction Result", "Please run batch prediction first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Output CSV", "batch_prediction_output.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.pred_df.to_csv(path, index=False, encoding="utf-8-sig")
            self._log(f"Output saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Result Error", str(exc))

    def _log(self, text: str) -> None:
        self.log_box.append(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.backend = PredictorBackend(DEFAULT_MODEL_PATH)
        self.setWindowTitle(APP_TITLE)
        self.resize(1120, 820)
        self._build_ui()
        self._build_menu()
        self._load_model_on_startup()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        header = QFrame()
        header.setObjectName("headerPanel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        header_layout.setSpacing(6)

        title = QLabel(APP_TITLE)
        title.setObjectName("mainTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("subTitle")
        subtitle.setWordWrap(True)

        self.model_label = QLabel(f"Current model: {self.backend.model_path}")
        self.model_label.setObjectName("modelPath")
        self.model_label.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(self.model_label)
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        self.single_tab = SingleCaseTab(self.backend)
        self.batch_tab = BatchTab(self.backend)
        self.tabs.addTab(self.single_tab, "Single-case prediction")
        self.tabs.addTab(self.batch_tab, "Batch prediction")
        main_layout.addWidget(self.tabs, 1)

        help_box = QGroupBox("Usage notes")
        help_layout = QVBoxLayout(help_box)
        help_layout.setContentsMargins(18, 20, 18, 18)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setText(HELP_TEXT)
        help_layout.addWidget(help_text)
        main_layout.addWidget(help_box)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        choose_model_action = QAction("Change model file", self)
        choose_model_action.triggered.connect(self.change_model_file)
        file_menu.addAction(choose_model_action)

        help_menu = menu.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _load_model_on_startup(self) -> None:
        try:
            self.backend.load_model()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Model Load Warning",
                "The default model file could not be loaded.\n\n"
                + str(exc)
                + "\n\nYou can manually choose the model file from File > Change model file.",
            )

    def change_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select LightGBM Model", "", "Joblib Files (*.joblib)")
        if not path:
            return
        try:
            self.backend.model_path = path
            self.backend.load_model()
            self.model_label.setText(f"Current model: {path}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Model Load Error", str(exc))

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            APP_TITLE + "\n\n"
            "LightGBM-based desktop tool for rapid single-case and batch prediction of runway-bridge impact coefficients.",
        )


def set_global_style(app: QApplication) -> None:
    app.setFont(QFont("Times New Roman", 10))
    app.setStyleSheet(
        """
        QWidget {
            font-family: 'Times New Roman';
            font-size: 10.5pt;
            color: #1f2937;
        }
        QMainWindow {
            background: #eef3f8;
        }
        #headerPanel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #1f4e79, stop:1 #2f6fed);
            border-radius: 12px;
        }
        #mainTitle {
            color: white;
            font-size: 18pt;
            font-weight: bold;
        }
        #subTitle {
            color: #e8f1ff;
            font-size: 10.5pt;
        }
        #modelPath {
            color: #d6e6ff;
            font-size: 9.8pt;
        }
        QGroupBox {
            border: 1px solid #d5dde8;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 10px;
            background: white;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }
        QPushButton {
            background: #2f6fed;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: #2459be;
        }
        QPushButton:pressed {
            background: #1d4aa3;
        }
        QLineEdit, QTextEdit {
            border: 1px solid #c7d2e0;
            border-radius: 8px;
            background: white;
            padding: 6px 8px;
        }
        QLineEdit:focus, QTextEdit:focus {
            border: 1px solid #2f6fed;
        }
        QTabWidget::pane {
            border: 1px solid #d5dde8;
            border-radius: 10px;
            background: #f9fbfd;
            top: -1px;
        }
        QTabBar::tab {
            background: #dde6f3;
            padding: 10px 16px;
            margin-right: 3px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        QTabBar::tab:selected {
            background: white;
            font-weight: bold;
        }
        #metricCard {
            background: #f6f9fd;
            border: 1px solid #d9e2ee;
            border-radius: 10px;
        }
        #cardLabel {
            color: #5b6777;
            font-size: 9.5pt;
            font-weight: normal;
        }
        #cardValue {
            color: #123b70;
            font-size: 16pt;
            font-weight: bold;
        }
        #sectionIntro {
            color: #415164;
        }
        #fileStatus {
            color: #123b70;
            font-weight: bold;
        }
        """
    )


def main() -> None:
    app = QApplication(sys.argv)
    set_global_style(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err)
        app = QApplication.instance() or QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Fatal Error")
        msg.setText("The application terminated unexpectedly.")
        msg.setDetailedText(err)
        msg.exec()
