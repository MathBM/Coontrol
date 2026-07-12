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

        # Variáveis de controle para a automação nativa
        self.current_scan_folder = None
        self.current_placa = None

        # connects
        self.ui.btp_refreshTable.clicked.connect(self.refresh_table)
        self.ui.btp_processData.clicked.connect(self.process_data)
        self.ui.btp_startScan.clicked.connect(self.start_scan)
        self.ui.btp_stopScan.clicked.connect(self.stop_scan)
        self.ui.btp_createSyntheticScan.clicked.connect(self.create_synthetic_scan)

        self.ui.btp_setEmptyBucket = QPushButton("Definir como Caixa Vazia")
        self.ui.verticalLayout_3.addWidget(self.ui.btp_setEmptyBucket)
        self.ui.btp_setEmptyBucket.clicked.connect(self.set_empty_bucket)

        # setup
        self.ui.tbw_scans.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.refresh_table()

    def start_scan(self):
        # 1. Solicita a Placa do Caminhão
        placa, ok_placa = QInputDialog.getText(self, "Identificação", "Digite a placa do caminhão:")
        if not ok_placa or not placa.strip():
            QMessageBox.warning(self, "Aviso", "A placa é obrigatória para iniciar o escaneamento.")
            return

        # 2. Solicita o Tipo de Escaneamento (Cria sempre uma pasta nova, seja cheia ou vazia)
        tipos = ["caixa_cheia", "caixa_vazia"]
        tipo, ok_tipo = QInputDialog.getItem(self, "Tipo de Scan", "Selecione o estado da caçamba:", tipos, 0, False)
        if not ok_tipo or not tipo:
            return

        self.ui.btp_startScan.setEnabled(False)

        # 3. Formata strings e cria o diretório único do scan
        placa_formatada = placa.strip().upper().replace("-", "").replace(" ", "")
        date = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
        
        folder_name = f"{tipo}_{placa_formatada}_{date}"
        output_folder = f"{Constants.SCANS_DIRECTORY}{folder_name}/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        self.current_scan_folder = folder_name
        self.current_placa = placa_formatada

        # 4. Inicia a gravação física dos LiDars
        self.scan_manager.start_scan(output_folder)
        self.ui.btp_stopScan.setEnabled(True)

    def stop_scan(self):
        self.ui.btp_stopScan.setEnabled(False)
        self.scan_manager.stop_scan()
        self.ui.btp_startScan.setEnabled(True)
        self.refresh_table()

        placa_verificar = self.current_placa
        self.current_scan_folder = None
        self.current_placa = None

        if placa_verificar:
            self.verificar_e_processar_par_automatico(placa_verificar)

    def verificar_e_processar_par_automatico(self, placa: str):
        """Busca o par dinamicamente e injeta os caminhos direto no DataManager"""
        if not os.path.exists(Constants.SCANS_DIRECTORY):
            return

        pastas = os.listdir(Constants.SCANS_DIRECTORY)
        pasta_cheia = None
        pasta_vazia = None

        # Localiza as pastas específicas desse caminhão
        for f in pastas:
            if placa in f:
                if f.startswith("caixa_cheia"):
                    pasta_cheia = f
                elif f.startswith("caixa_vazia"):
                    pasta_vazia = f

        # Se encontrou o par correspondente nativo no disco
        if pasta_cheia and pasta_vazia:
            msg = f"Par correspondente encontrado para o caminhão {placa}!\n\nCheio: {pasta_cheia}\nVazio: {pasta_vazia}\n\nDeseja realizar o cálculo de volume nativo?"
            
            # Criando o QMessageBox customizado para traduzir os botões
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Par Detectado")
            msg_box.setText(msg)
            msg_box.setIcon(QMessageBox.Question)
            
            # Adiciona os botões de simulação lógica e altera o texto visual
            sim_button = msg_box.addButton(QMessageBox.Yes)
            nao_button = msg_box.addButton(QMessageBox.No)
            sim_button.setText("Sim")
            nao_button.setText("Não")
            
            msg_box.exec()
            resposta = msg_box.clickedButton()
            
            if resposta == sim_button:
                scan_path_cheio = f"{Constants.SCANS_DIRECTORY}{pasta_cheia}/"
                scan_path_vazio = f"{Constants.SCANS_DIRECTORY}{pasta_vazia}/"
                
                try:
                    # PROVA REAL CONFIÁVEL: Passando o caminho real do par por parâmetro na chamada
                    if self.ui.cmb_method.currentIndex() == 0:
                        volume = self.data_manager.process_data(scan_path_cheio, bucket_path=scan_path_vazio)
                    else:
                        volume = self.data_manager.process_data_legacy(scan_path_cheio, bucket_path=scan_path_vazio)
                    
                    QMessageBox.information(self, "Volume Calculado", f"O volume calculado de forma 100% dinâmica para o veículo {placa} é: {volume} m³")
                    self.refresh_table()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Erro no Processamento", f"Falha ao processar cálculo volumétrico:\n{str(e)}")
        else:
            QMessageBox.information(self, "Scan Salvo", f"Captura do veículo {placa} salva. Aguardando a contraparte para habilitar o cálculo automático.")

    def process_data(self):
        """Processamento manual quando o operador clica em uma linha da tabela"""
        row_selected = self.ui.tbw_scans.selectedIndexes()
        if not row_selected:
            return

        row_index = row_selected[0].row()
        scan_folder = self.scanList[row_index]
        scan_path = f"{Constants.SCANS_DIRECTORY}{scan_folder}/"

        # Se o operador clicar manualmente em uma linha de caixa_cheia na tabela,
        # tentamos achar uma caixa_vazia da MESMA placa para não recalcular errado.
        bucket_path = None
        if "caixa_cheia_" in scan_folder:
            partes = scan_folder.split("_")
            if len(partes) > 2:
                placa_extraida = partes[2] # Extrai a placa do nome da pasta
                for f in os.listdir(Constants.SCANS_DIRECTORY):
                    if f.startswith("caixa_vazia_") and placa_extraida in f:
                        bucket_path = f"{Constants.SCANS_DIRECTORY}{f}/"
                        break

        # Processa usando o par descoberto ou cai no fallback padrão do Constants.BUCKET_PATH
        if self.ui.cmb_method.currentIndex() == 0:
            volume = self.data_manager.process_data(scan_path, bucket_path=bucket_path)
        else:
            volume = self.data_manager.process_data_legacy(scan_path, bucket_path=bucket_path)

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
        items = ["Linear (rampa reta)", "Stepped (escada)", "Concave (côncava)", "Convex (convexa)"]
        item, ok = QInputDialog.getItem(self, "Create Synthetic Scan", "Escolha o tipo de rampa:", items, 0, False)
        
        if ok and item:
            try:
                type_map = {
                    "Linear (rampa reta)": "linear",
                    "Stepped (escada)": "stepped",
                    "Concave (côncava)": "concave",
                    "Convex (convexa)": "convex"
                }
                ramp_type = type_map[item]
                
                self.ui.btp_createSyntheticScan.setEnabled(False)
                self.ui.btp_createSyntheticScan.setText("Creating...")
                
                scan_path = self.synthetic_creator.create_synthetic_scan(
                    ramp_type=ramp_type, width=2000, length=3000, height=800, point_density=8, noise_level=3.0
                )
                
                self.ui.btp_createSyntheticScan.setEnabled(True)
                self.ui.btp_createSyntheticScan.setText("Create Synthetic Scan")
                self.refresh_table()
                
                QMessageBox.information(self, "Success", f"Synthetic scan created successfully!\n\nType: {item}\nPath: {scan_path}")
            except Exception as e:
                self.ui.btp_createSyntheticScan.setEnabled(True)
                self.ui.btp_createSyntheticScan.setText("Create Synthetic Scan")
                QMessageBox.critical(self, "Error", f"Failed to create synthetic scan:\n{str(e)}")

    def set_empty_bucket(self):
        row_selected = self.ui.tbw_scans.selectedIndexes()
        if not row_selected:
            QMessageBox.warning(self, "Aviso", "Selecione um scan na tabela para definir como caixa vazia.")
            return

        row_index = row_selected[0].row()
        scan_folder = self.scanList[row_index]
        
        if scan_folder == os.path.basename(Constants.BUCKET_PATH):
            QMessageBox.information(self, "Info", "Este scan já é a caixa de referência de backup.")
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
                backup_msg = f"\nA referência de backup anterior foi preservada como '{os.path.basename(backup_path)}'."
            
            os.rename(src_path, dst_path)
            QMessageBox.information(self, "Sucesso", f"O scan '{scan_folder}' foi definido como o backup estático de referência.{backup_msg}")
            self.refresh_table()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Failed to redefinir a caixa vazia:\n{str(e)}")