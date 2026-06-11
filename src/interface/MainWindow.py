import os
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QTableWidgetItem, QHeaderView, QMessageBox, QInputDialog, QPushButton

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

        # Connects
        self.ui.btp_refreshTable.clicked.connect(self.refresh_table)
        self.ui.btp_processData.clicked.connect(self.process_data)
        self.ui.btp_startScan.clicked.connect(self.start_scan)
        self.ui.btp_stopScan.clicked.connect(self.stop_scan)
        self.ui.btp_createSyntheticScan.clicked.connect(self.create_synthetic_scan)
        self.ui.lne_search.textChanged.connect(self.filter_table)

        self.ui.btp_setEmptyBucket = QPushButton("Definir como Caixa Vazia")
        self.ui.verticalLayout_3.addWidget(self.ui.btp_setEmptyBucket)
        self.ui.btp_setEmptyBucket.clicked.connect(self.set_empty_bucket)

        self.ui.tbw_scans.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.refresh_table()

    def start_scan(self):
        placa, ok = QInputDialog.getText(self, "Identificação", "Digite a placa do caminhão:")
        if not ok or not placa: return

        tipo, ok = QInputDialog.getItem(self, "Estado", "Tipo de inspeção:", ["caixa_cheia", "caixa_vazia"], 0, False)
        if not ok: return
        
        data = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
        folder_name = f"{tipo}_{placa}_{data}"
        output_folder = os.path.join(Constants.SCANS_DIRECTORY, folder_name)

        if not os.path.exists(output_folder): os.mkdir(output_folder)
        self.scan_manager.start_scan(output_folder)
        self.ui.btp_startScan.setEnabled(False)
        self.ui.btp_stopScan.setEnabled(True)

    def stop_scan(self):
        self.scan_manager.stop_scan()
        self.ui.btp_startScan.setEnabled(True)
        self.ui.btp_stopScan.setEnabled(False)
        self.refresh_table()

        ultimo_scan = self.scanList[0] 
        partes = ultimo_scan.split("_")
        
        # Filtro: só processa pastas que começam com "caixa_"
        if len(partes) < 4 or partes[0] != "caixa":
            return 

        estado_atual = partes[1]
        placa = partes[2]
        data_hoje = partes[3]
        
        # Busca o par
        estado_parceiro = "vazia" if estado_atual == "cheia" else "cheia"
        prefixo_parceiro = f"caixa_{estado_parceiro}_{placa}_{data_hoje}"
        parceiro = next((s for s in self.scanList if s.startswith(prefixo_parceiro)), None)

        # Se encontrou o par e este scan é a "cheia", dispara o cálculo
        if parceiro and estado_atual == "cheia":
            self.process_data_automatico(ultimo_scan)

    def process_data_automatico(self, scan_folder):
        # Como SCANS_DIRECTORY já é "./pointcloud/", o caminho correto é apenas:
        scan_path = os.path.join(Constants.SCANS_DIRECTORY, scan_folder)
        
        print(f"DEBUG: Processando dados em: {scan_path}")
        
        # Executa o cálculo
        volume = self.data_manager.process_data(scan_path) if self.ui.cmb_method.currentIndex() == 0 else self.data_manager.process_data_legacy(scan_path)
        
        # Atualiza a interface
        for i in range(self.ui.tbw_scans.rowCount()):
            if self.ui.tbw_scans.item(i, 0).text() == scan_folder.replace("_", " "):
                self.ui.tbw_scans.item(i, 1).setText(str(volume))
                break

    def filter_table(self, text):
        for i in range(self.ui.tbw_scans.rowCount()):
            item = self.ui.tbw_scans.item(i, 0)
            self.ui.tbw_scans.setRowHidden(i, text.lower() not in item.text().lower())

    def refresh_table(self):
        self.scanList = [s for s in os.listdir(Constants.SCANS_DIRECTORY) if not os.path.isfile(os.path.join(Constants.SCANS_DIRECTORY, s))]
        self.scanList.reverse()
        self.ui.tbw_scans.setRowCount(0)
        for scan in self.scanList:
            row = self.ui.tbw_scans.rowCount()
            self.ui.tbw_scans.insertRow(row)
            self.ui.tbw_scans.setItem(row, 0, QTableWidgetItem(scan.replace("_", " ")))
            self.ui.tbw_scans.setItem(row, 1, QTableWidgetItem("-"))

    def process_data(self):
        row_selected = self.ui.tbw_scans.selectedIndexes()
        if not row_selected: return
        scan_folder = self.scanList[row_selected[0].row()]
        scan_path = os.path.join(Constants.SCANS_DIRECTORY, scan_folder)
        volume = self.data_manager.process_data(scan_path) if self.ui.cmb_method.currentIndex() == 0 else self.data_manager.process_data_legacy(scan_path)
        self.ui.tbw_scans.item(row_selected[0].row(), 1).setText(str(volume))

    def create_synthetic_scan(self):
        # Mantido inalterado conforme solicitado
        items = ["Linear (rampa reta)", "Stepped (escada)", "Concave (côncava)", "Convex (convexa)"]
        item, ok = QInputDialog.getItem(self, "Create Synthetic Scan", "Escolha o tipo de rampa:", items, 0, False)
        if ok and item:
            try:
                type_map = {"Linear (rampa reta)": "linear", "Stepped (escada)": "stepped", "Concave (côncava)": "concave", "Convex (convexa)": "convex"}
                ramp_type = type_map[item]
                self.ui.btp_createSyntheticScan.setEnabled(False)
                scan_path = self.synthetic_creator.create_synthetic_scan(ramp_type=ramp_type, width=2000, length=3000, height=800, point_density=8, noise_level=3.0)
                self.ui.btp_createSyntheticScan.setEnabled(True)
                self.refresh_table()
                QMessageBox.information(self, "Success", f"Synthetic scan created successfully!\nPath: {scan_path}")
            except Exception as e:
                self.ui.btp_createSyntheticScan.setEnabled(True)
                QMessageBox.critical(self, "Error", f"Failed to create synthetic scan:\n{str(e)}")

    def set_empty_bucket(self):
        row_selected = self.ui.tbw_scans.selectedIndexes()
        if not row_selected:
            QMessageBox.warning(self, "Aviso", "Selecione um scan na tabela para definir como caixa vazia.")
            return

        row_index = row_selected[0].row()
        scan_folder = self.scanList[row_index]
        if scan_folder == os.path.basename(Constants.BUCKET_PATH):
            QMessageBox.information(self, "Info", "Este scan já é a caixa de referência.")
            return

        src_path = os.path.join(Constants.SCANS_DIRECTORY, scan_folder)
        dst_path = Constants.BUCKET_PATH
        try:
            backup_msg = ""
            if os.path.exists(dst_path):
                base_dir = os.path.dirname(dst_path)
                base_name = os.path.basename(dst_path)
                counter = 1
                backup_path = os.path.join(base_dir, f"{base_name}_{counter}")
                while os.path.exists(backup_path):
                    counter += 1
                    backup_path = os.path.join(base_dir, f"{base_name}_{counter}")
                os.rename(dst_path, backup_path)
                backup_msg = f"\nA referência anterior foi preservada como '{os.path.basename(backup_path)}'."
            
            os.rename(src_path, dst_path)
            QMessageBox.information(self, "Sucesso", f"O scan '{scan_folder}' foi definido como a nova caixa de referência.{backup_msg}")
            self.refresh_table()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao redefinir a caixa vazia:\n{str(e)}")