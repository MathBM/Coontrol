import os
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QTableWidgetItem, QHeaderView, QMessageBox, QInputDialog

from src.ScanManager import ScanManager
from src.DataManager import DataManager
from src.Constants import Constants
from src.interface.MainWindow_ui import Ui_MainWindow
from src.SyntheticScanCreator import SyntheticScanCreator


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.data_manager = DataManager()
        self.scan_manager = ScanManager()
        self.synthetic_creator = SyntheticScanCreator()

        self.scanList = list()

        # connects
        self.ui.btp_refreshTable.clicked.connect(self.refresh_table)
        self.ui.btp_processData.clicked.connect(self.process_data)
        self.ui.btp_startScan.clicked.connect(self.start_scan)
        self.ui.btp_stopScan.clicked.connect(self.stop_scan)
        self.ui.btp_createSyntheticScan.clicked.connect(self.create_synthetic_scan)

        # setup
        self.ui.tbw_scans.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.refresh_table()

    def start_scan(self):
        self.ui.btp_startScan.setEnabled(False)

        date = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
        output_folder = f"{Constants.SCANS_DIRECTORY}{date}/"

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        self.scan_manager.start_scan(output_folder)

        self.ui.btp_stopScan.setEnabled(True)

    def stop_scan(self):
        self.ui.btp_stopScan.setEnabled(False)

        self.scan_manager.stop_scan()

        self.ui.btp_startScan.setEnabled(True)
        self.refresh_table()

    def process_data(self):
        row_selected = self.ui.tbw_scans.selectedIndexes()

        if not row_selected:
            return

        row_index = row_selected[0].row()
        scan_folder = self.scanList[row_index]
        scan_path = f"{Constants.SCANS_DIRECTORY}{scan_folder}/"

        if self.ui.cmb_method.currentIndex() == 0:
            # Fluxo novo: alinhamento adaptativo + mapa de alturas 2D
            volume = self.data_manager.process_data(scan_path)
        else:
            # Fluxo legado: RANSAC+ICP + merge + Poisson + teorema da divergência
            volume = self.data_manager.process_data_legacy(scan_path)

        item = self.ui.tbw_scans.item(row_index, 1)
        item.setText(str(volume))

    def refresh_table(self):
        self.scanList = [scan for scan in os.listdir(
            Constants.SCANS_DIRECTORY) if not os.path.isfile(f"{Constants.SCANS_DIRECTORY}{scan}")]
        self.scanList.reverse()

        self.ui.tbw_scans.setRowCount(0)

        for scan in self.scanList:
            row = self.ui.tbw_scans.rowCount()
            self.ui.tbw_scans.insertRow(row)

            item_id = QTableWidgetItem(scan.replace("-", "/").replace("_", " "))
            item_volume = QTableWidgetItem("-")

            item_id.setTextAlignment(Qt.AlignCenter)
            item_volume.setTextAlignment(Qt.AlignCenter)
            item_id.setFlags(item_id.flags() ^ Qt.ItemIsEditable)
            item_volume.setFlags(item_volume.flags() ^ Qt.ItemIsEditable)

            self.ui.tbw_scans.setItem(row, 0, item_id)
            self.ui.tbw_scans.setItem(row, 1, item_volume)
    
    def create_synthetic_scan(self):
        """Cria um scan sintético e adiciona à lista"""
        # Diálogo para escolher o tipo de rampa
        items = ["Linear (rampa reta)", 
                "Stepped (escada)", 
                "Concave (côncava)", 
                "Convex (convexa)"]
        
        item, ok = QInputDialog.getItem(
            self, 
            "Create Synthetic Scan",
            "Escolha o tipo de rampa:", 
            items, 
            0, 
            False
        )
        
        if ok and item:
            try:
                # Mapear escolha para tipo
                type_map = {
                    "Linear (rampa reta)": "linear",
                    "Stepped (escada)": "stepped",
                    "Concave (côncava)": "concave",
                    "Convex (convexa)": "convex"
                }
                
                ramp_type = type_map[item]
                
                # Criar scan sintético
                self.ui.btp_createSyntheticScan.setEnabled(False)
                self.ui.btp_createSyntheticScan.setText("Creating...")
                
                scan_path = self.synthetic_creator.create_synthetic_scan(
                    ramp_type=ramp_type,
                    width=2000,
                    length=3000,
                    height=800,
                    point_density=8,
                    noise_level=3.0
                )
                
                self.ui.btp_createSyntheticScan.setEnabled(True)
                self.ui.btp_createSyntheticScan.setText("Create Synthetic Scan")
                
                # Atualizar tabela
                self.refresh_table()
                
                # Mensagem de sucesso
                QMessageBox.information(
                    self,
                    "Success",
                    f"Synthetic scan created successfully!\n\nType: {item}\nPath: {scan_path}\n\nYou can now select it and click 'Process Data'."
                )
                
            except Exception as e:
                self.ui.btp_createSyntheticScan.setEnabled(True)
                self.ui.btp_createSyntheticScan.setText("Create Synthetic Scan")
                
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to create synthetic scan:\n{str(e)}"
                )
