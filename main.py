import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, 
                           QProgressBar, QMessageBox, QLineEdit, QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
from document_parser import DocumentParser
from nlp_pipeline import NLPPipeline, NLPConfig
from clustering_pipeline import TextClusteringPipeline, ClusteringConfig
import logging
from automl_pipeline import AutoMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_paths, data_type, model_type, target_column, save_path):
        super().__init__()
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.data_type = data_type
        self.model_type = model_type.lower()
        self.target_column = target_column
        self.save_path = save_path
        # Initialize pipelines
        self.parser = DocumentParser()
        self.nlp_pipeline = NLPPipeline(NLPConfig(parallel_processing=False))
        # Initialize AutoML pipeline for regression/classification
        self.automl_pipeline = AutoMLPipeline()

    def run(self):
        try:
            self.progress.emit(10)
            
            if self.model_type == 'nlp':
                # Existing NLP pipeline code remains unchanged
                results = self.nlp_pipeline.process_files(self.file_paths)
                self.progress.emit(50)
                
                if results.get('processed_texts'):
                    save_file = os.path.join(self.save_path, 'nlp_results.joblib')
                    self.nlp_pipeline.save(save_file)
                    results['model_path'] = save_file
                
                self.progress.emit(100)
                self.finished.emit(results)
                
            elif self.model_type in ['regression', 'classification']:
                try:
                    # Set the problem type in AutoML pipeline
                    self.automl_pipeline.set_problem_type('regression')
                    self.progress.emit(20)

                    # Since we handle only one file for reg/class
                    file_path = self.file_paths[0]
                    
                    # Run the AutoML pipeline
                    results = {}
                    try:
                        # Run pipeline and get best model
                        best_model = self.automl_pipeline.run_pipeline(
                            file_path=file_path,
                            target_column=self.target_column,
                            output_path=self.save_path
                        )
                        
                        # Get model info for results
                        results = {
                            'best_model_name': type(best_model).__name__,
                            'best_score': self.automl_pipeline.best_score,
                            'metrics': self.automl_pipeline.best_metrics,
                            'selected_features': list(self.automl_pipeline.selected_features),
                            'model_path': os.path.join(self.save_path, 'model_info.json')
                        }
                        
                        # Add feature importances if available
                        if hasattr(best_model, 'feature_importances_'):
                            results['feature_importance'] = best_model.feature_importances_.tolist()
                        
                        self.progress.emit(100)
                        self.finished.emit(results)
                        
                    except Exception as e:
                        logger.error(f"Pipeline execution error: {str(e)}")
                        self.error.emit(f"Error during model training: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"AutoML setup error: {str(e)}")
                    self.error.emit(f"Error setting up AutoML: {str(e)}")
            
            elif self.model_type == 'clustering':
                # Existing clustering code remains unchanged
                def update_progress(progress):
                    self.progress.emit(progress)

                config = ClusteringConfig(
                    max_features=1000,
                    n_components=50,
                    n_clusters_range=(2, 10),
                    progress_callback=update_progress
                )
                
                pipeline = TextClusteringPipeline(config)
                results = pipeline.fit_files(self.file_paths)
                
                if results['success']:
                    save_path = os.path.join(self.save_path, 'clustering_model.joblib')
                    save_result = pipeline.save(save_path)
                    
                    if save_result['success']:
                        results['model_path'] = save_result['path']
                    else:
                        results['save_error'] = save_result['error']
                    
                    self.finished.emit(results)
                else:
                    self.error.emit(results['error'])
                    
        except Exception as e:
            logger.error(f"Error in ML Worker: {str(e)}")
            self.error.emit(str(e))

class AutoMLInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize variables first
        self.dataset_paths = []
        self.save_path = None
        self.ml_worker = None
        self.target_widget = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Easy ML: No Expertise Required')
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                font-size: 14px;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                font-size: 14px;
            }
        """)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Dataset selection
        dataset_layout = QHBoxLayout()
        self.dataset_path_label = QLabel('No files selected')
        self.dataset_button = QPushButton('Select Files')
        self.dataset_button.clicked.connect(self.select_dataset)
        dataset_layout.addWidget(self.dataset_path_label)
        dataset_layout.addWidget(self.dataset_button)
        layout.addLayout(dataset_layout)

        # Data type selection
        data_type_layout = QHBoxLayout()
        data_type_label = QLabel('Data Type:')
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItem('-- Select Data Type --')
        self.data_type_combo.addItems(['Structured', 'Unstructured'])
        self.data_type_combo.currentTextChanged.connect(self.on_data_type_changed)
        data_type_layout.addWidget(data_type_label)
        data_type_layout.addWidget(self.data_type_combo)
        layout.addLayout(data_type_layout)

        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel('Model Type:')
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem('-- Select Model Type --')
        self.model_type_combo.setEnabled(False)  # Disabled until data type is selected
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.model_type_combo)
        layout.addLayout(model_type_layout)

        # Create target input widget (hidden by default)
        self.target_widget = QWidget()
        target_layout = QHBoxLayout(self.target_widget)
        target_label = QLabel('Target Column:')
        self.target_input = QLineEdit()
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_input)
        self.target_widget.setVisible(False)
        layout.addWidget(self.target_widget)

        # Save location
        save_layout = QHBoxLayout()
        self.save_path_label = QLabel('No save location selected')
        self.save_button = QPushButton('Select Save Location')
        self.save_button.clicked.connect(self.select_save_location)
        save_layout.addWidget(self.save_path_label)
        save_layout.addWidget(self.save_button)
        layout.addLayout(save_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Start button
        self.start_button = QPushButton('Start Training')
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)

    def on_data_type_changed(self, data_type):
        self.model_type_combo.clear()
        self.model_type_combo.addItem('-- Select Model Type --')
        
        if data_type == 'Structured':
            self.model_type_combo.addItems(['Regression', 'Classification'])
            self.model_type_combo.setEnabled(True)
        elif data_type == 'Unstructured':
            self.model_type_combo.addItems(['Clustering', 'NLP'])
            self.model_type_combo.setEnabled(True)
        else:
            self.model_type_combo.setEnabled(False)
        
        # Hide target input when data type changes
        self.target_widget.setVisible(False)
        self.check_start_conditions()

    def on_model_type_changed(self, model_type):
        # Show target input only for Regression and Classification
        self.target_widget.setVisible(model_type in ['Regression', 'Classification'])
        self.check_start_conditions()

    def select_dataset(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)  # Allow multiple file selection
        self.dataset_paths, _ = file_dialog.getOpenFileNames(
            self, 'Select Files', '', 'All Files (*.*)')
        
        if self.dataset_paths:
            if len(self.dataset_paths) > 1:
                self.dataset_path_label.setText(f"{len(self.dataset_paths)} files selected")
            else:
                self.dataset_path_label.setText(os.path.basename(self.dataset_paths[0]))
            self.check_start_conditions()

    def select_save_location(self):
        file_dialog = QFileDialog()
        self.save_path = file_dialog.getExistingDirectory(
            self, 'Select Save Location')
        if self.save_path:
            self.save_path_label.setText(os.path.basename(self.save_path))
            self.check_start_conditions()

    def check_start_conditions(self):
        data_type_selected = self.data_type_combo.currentText() not in ['', '-- Select Data Type --']
        model_type_selected = self.model_type_combo.currentText() not in ['', '-- Select Model Type --']
        
        # Check if target column is needed and provided
        needs_target = self.model_type_combo.currentText() in ['Regression', 'Classification']
        target_valid = bool(self.target_input.text().strip()) if needs_target else True

        self.start_button.setEnabled(
            bool(self.dataset_paths) and 
            bool(self.save_path) and 
            data_type_selected and
            model_type_selected and
            target_valid
        )

    def start_training(self):
        self.progress_bar.setVisible(True)
        self.start_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        self.ml_worker = MLWorker(
            self.dataset_paths,
            self.data_type_combo.currentText(),
            self.model_type_combo.currentText(),
            self.target_input.text() if self.target_widget.isVisible() else None,
            self.save_path
        )
        
        self.ml_worker.progress.connect(self.update_progress)
        self.ml_worker.finished.connect(self.handle_results)
        self.ml_worker.error.connect(self.show_error)
        self.ml_worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def training_finished(self):
        QMessageBox.information(self, 'Success', 'Model training completed successfully!')
        self.reset_interface()

    def show_error(self, error_message):
        QMessageBox.critical(self, 'Error', f'An error occurred: {error_message}')
        self.reset_interface()

    def reset_interface(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def handle_results(self, results):
        """Handle the results from the ML worker"""
        try:
            if results.get('error'):
                self.show_error(results['error'])
                return

            # Handle failed files
            if results.get('failed_files'):
                failed_msg = "\n\nFailed files:"
                for fail in results['failed_files']:
                    failed_msg += f"\n- {os.path.basename(fail['file_path'])}: {fail['error']}"
            else:
                failed_msg = ""

            # Show success message with stats
            success_msg = f"Processing completed successfully!"
            if results.get('processed_texts'):
                success_msg += f"\nProcessed {len(results['processed_texts'])} documents."
            if results.get('model_path'):
                success_msg += f"\nModel saved to: {results['model_path']}"
            
            success_msg += failed_msg
            
            QMessageBox.information(self, 'Success', success_msg)
            
        except Exception as e:
            logger.error(f"Error handling results: {str(e)}")
            self.show_error(f"Error processing results: {str(e)}")
        finally:
            self.reset_interface()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AutoMLInterface()
    window.show()
    sys.exit(app.exec())






def main():
    app = QApplication(sys.argv)
    window = AutoMLInterface()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()