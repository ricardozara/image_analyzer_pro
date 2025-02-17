# Importação das bibliotecas necessárias
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS
import os
from datetime import datetime
import hashlib
import threading
from transformers import pipeline
import torch

# Importações opcionais com tratamento de erro
try:
    import magic
except ImportError:
    print("python-magic não instalado. Algumas funcionalidades podem estar indisponíveis.")
    magic = None

try:
    import imagehash
except ImportError:
    print("imagehash não instalado. Hash perceptual estará indisponível.")
    imagehash = None

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV não instalado. Algumas análises avançadas estarão indisponíveis.")
    cv2 = None

class ImageAnalyzer:
    """
    Classe principal para análise de imagens.
    Fornece uma interface gráfica para análise detalhada de imagens,
    incluindo informações básicas, tags avançadas e metadados.
    """
    def __init__(self, root):
        """
        Inicializa a aplicação.
        
        Args:
            root: Janela principal do Tkinter
        """
        # Configuração da janela principal
        self.root = root
        self.root.title("Analisador de Imagens")
        self.root.geometry("1200x800")
        
        # Variáveis de controle
        self.current_image = None
        self.current_image_path = None
        self.analysis_results = {}
        self.analysis_done = False
        
        # Inicializar modelo de IA
        self.setup_image_analyzer()
        
        # Criar o layout principal
        self.create_main_layout()
        
        # Configurar estilos
        self.setup_styles()
        
        # Bind para redimensionamento
        self.root.bind('<Configure>', self.on_window_resize)

    def setup_image_analyzer(self):
        """Inicializa o modelo de análise de imagem"""
        try:
            print("Carregando modelo de IA... Por favor, aguarde...")
            self.image_captioner = pipeline(
                task="image-to-text",
                model="Salesforce/blip-image-captioning-base",
                max_new_tokens=50
            )
            print("Modelo de IA carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Erro ao carregar modelo de IA")

    def setup_styles(self):
        """Configura os estilos visuais dos widgets"""
        style = ttk.Style()
        style.configure("Header.TLabel", font=('Arial', 12, 'bold'))
        style.configure("Info.TLabel", font=('Arial', 10))

    def create_main_layout(self):
        """Cria o layout principal da aplicação"""
        # Frame principal dividido em duas partes
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame esquerdo para imagem e controles
        self.left_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.left_frame, weight=1)

        # Frame direito para notebook com análises
        self.right_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.right_frame, weight=1)

        # Configurar componentes
        self.setup_left_frame()
        self.setup_right_frame()

    def setup_left_frame(self):
        """Configura o frame esquerdo com botões e canvas"""
        # Frame para botões
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        # Botão para selecionar imagem
        self.select_button = ttk.Button(
            self.button_frame, 
            text="Selecionar Imagem",
            command=self.select_image
        )
        self.select_button.pack(side=tk.LEFT, padx=5)

        # Botão para análise
        self.analyze_button = ttk.Button(
            self.button_frame,
            text="Analisar Imagem",
            command=self.perform_analysis,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            self.button_frame,
            text="",
            style="Info.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Canvas para exibir a imagem
        self.canvas_frame = ttk.Frame(self.left_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg='gray90',
            width=400,
            height=400
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def setup_right_frame(self):
        """Configura o frame direito com as abas de análise"""
        # Criar notebook para diferentes análises
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Criar as abas
        self.initial_analysis_frame = ttk.Frame(self.notebook)
        self.basic_info_frame = ttk.Frame(self.notebook)
        self.advanced_tags_frame = ttk.Frame(self.notebook)
        self.metadata_frame = ttk.Frame(self.notebook)

        # Adicionar abas ao notebook
        self.notebook.add(self.initial_analysis_frame, text="Análise Inicial")
        self.notebook.add(self.basic_info_frame, text="Informações Básicas")
        self.notebook.add(self.advanced_tags_frame, text="Tags Avançadas")
        self.notebook.add(self.metadata_frame, text="Metadados")

        # Configurar conteúdo das abas
        self.setup_initial_analysis_tab()
        self.setup_basic_info_tab()
        self.setup_advanced_tags_tab()
        self.setup_metadata_tab()

    def setup_initial_analysis_tab(self):
        """Configura a aba de análise inicial"""
        self.initial_analysis_text = scrolledtext.ScrolledText(
            self.initial_analysis_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Consolas', 10)
        )
        self.initial_analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_basic_info_tab(self):
        """Configura a aba de informações básicas"""
        self.basic_info_text = scrolledtext.ScrolledText(
            self.basic_info_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Consolas', 10)
        )
        self.basic_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_advanced_tags_tab(self):
        """Configura a aba de tags avançadas"""
        self.advanced_tags_text = scrolledtext.ScrolledText(
            self.advanced_tags_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Consolas', 10)
        )
        self.advanced_tags_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_metadata_tab(self):
        """Configura a aba de metadados"""
        self.metadata_text = scrolledtext.ScrolledText(
            self.metadata_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Consolas', 10)
        )
        self.metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_image(self):
        """Abre diálogo para seleção de imagem e prepara para análise"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text="Carregando imagem...")
            self.load_and_display_image()
            self.analyze_button.config(state=tk.NORMAL)
            self.analysis_done = False
            self.status_label.config(text="Imagem carregada. Pronto para análise.")

    def load_and_display_image(self):
        """Carrega e exibe a imagem selecionada no canvas"""
        try:
            if self.current_image_path:
                # Carregar imagem com PIL
                image = Image.open(self.current_image_path)
                
                # Obter dimensões do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width <= 1:
                    canvas_width = 400
                if canvas_height <= 1:
                    canvas_height = 400
                
                # Calcular nova dimensão
                ratio = min(canvas_width/image.width, canvas_height/image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                
                # Redimensionar
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Converter para PhotoImage
                self.current_image = ImageTk.PhotoImage(image)
                
                # Limpar canvas e exibir nova imagem
                self.canvas.delete("all")
                x = canvas_width/2
                y = canvas_height/2
                self.canvas.create_image(
                    x, y,
                    image=self.current_image,
                    anchor='center'
                )
                
        except Exception as e:
            self.status_label.config(text=f"Erro ao carregar imagem: {str(e)}")

    def perform_analysis(self):
        """Executa todas as análises quando o botão de análise é clicado"""
        if self.current_image_path and not self.analysis_done:
            self.status_label.config(text="Realizando análise...")
            self.select_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.DISABLED)
            
            # Criar thread para análise
            analysis_thread = threading.Thread(target=self._run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()

    def _run_analysis(self):
        """Executa análises em thread separada"""
        try:
            self.analyze_initial()
            self.analyze_image()
            self.analysis_done = True
            self.root.after(0, self._analysis_complete)
        except Exception as e:
            self.root.after(0, self._analysis_error, str(e))

    def _analysis_complete(self):
        """Callback para conclusão da análise"""
        self.select_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)
        self.status_label.config(text="Análise concluída")

    def _analysis_error(self, error_msg):
        """Callback para erro na análise"""
        self.select_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Erro na análise: {error_msg}")

    def analyze_image_content(self, image_path):
        """Analisa o conteúdo da imagem usando IA"""
        try:
            image = Image.open(image_path)
            result = self.image_captioner(image)
            return result[0]['generated_text']
        except Exception as e:
            return f"Erro na análise de conteúdo: {str(e)}"

    def on_window_resize(self, event=None):
        """Manipula evento de redimensionamento da janela"""
        if hasattr(self, 'current_image_path'):
            self.load_and_display_image()

    def calculate_hashes(self, image_path):
        """Calcula diferentes tipos de hashes da imagem"""
        hashes = {'md5': None, 'perceptual_hash': None}
        
        # MD5 hash
        try:
            md5_hash = hashlib.md5()
            with open(image_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            hashes['md5'] = md5_hash.hexdigest()
        except Exception as e:
            print(f"Erro ao calcular MD5: {e}")

        # Hash perceptual
        if imagehash:
            try:
                img = Image.open(image_path)
                hashes['perceptual_hash'] = str(imagehash.average_hash(img))
            except Exception as e:
                print(f"Erro ao calcular hash perceptual: {e}")
        
        return hashes

    def get_image_description(self, image_path):
        """Gera uma descrição detalhada da imagem"""
        try:
            img = Image.open(image_path)
            mime_type = "Não disponível"
            
            if magic:
                mime = magic.Magic(mime=True)
                mime_type = mime.from_file(image_path)
            
            description = []
            description.append(f"Tipo de arquivo: {mime_type}")
            description.append(f"Formato: {img.format}")
            description.append(f"Modo de cor: {img.mode}")
            description.append(f"Dimensões: {img.size[0]}x{img.size[1]} pixels")
            
            # Calcular proporção
            ratio = img.size[0] / img.size[1]
            description.append(f"Proporção: {ratio:.2f}")
            
            # Verificar orientação
            if img.size[0] > img.size[1]:
                orientation = "Paisagem"
            elif img.size[0] < img.size[1]:
                orientation = "Retrato"
            else:
                orientation = "Quadrada"
            description.append(f"Orientação: {orientation}")
            
            return "\n".join(description)
        except Exception as e:
            return f"Erro ao gerar descrição: {str(e)}"

    def analyze_initial(self):
        """Realiza a análise inicial da imagem"""
        try:
            # Limpar texto anterior
            self.root.after(0, self.initial_analysis_text.delete, 1.0, tk.END)
            
            # Preparar informações
            info = []
            info.append("=== ANÁLISE INICIAL ===\n")
            
            # Data e hora da análise
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            info.append(f"Data/Hora da análise: {current_time}\n")
            
            # Nome do arquivo
            filename = os.path.basename(self.current_image_path)
            info.append(f"Nome do arquivo: {filename}\n")
            
            # Análise de conteúdo com IA
            info.append("CONTEÚDO DA IMAGEM:")
            content_description = self.analyze_image_content(self.current_image_path)
            info.append(f"{content_description}\n")
            
            # Descrição da imagem
            info.append("CARACTERÍSTICAS DA IMAGEM:")
            description = self.get_image_description(self.current_image_path)
            info.append(f"{description}\n")
            
            # Hashes
            info.append("HASHES:")
            hashes = self.calculate_hashes(self.current_image_path)
            info.append(f"MD5: {hashes['md5']}")
            info.append(f"Hash Perceptual: {hashes['perceptual_hash']}\n")
            
            # Informações do sistema de arquivos
            stats = os.stat(self.current_image_path)
            info.append("INFORMAÇÕES DO ARQUIVO:")
            info.append(f"Tamanho: {stats.st_size/1024:.2f} KB")
            info.append(f"Criado em: {datetime.fromtimestamp(stats.st_ctime)}")
            info.append(f"Modificado em: {datetime.fromtimestamp(stats.st_mtime)}")
            info.append(f"Último acesso: {datetime.fromtimestamp(stats.st_atime)}")
            
            # Atualizar interface
            final_text = "\n".join(info)
            self.root.after(0, self.initial_analysis_text.insert, tk.END, final_text)
            
        except Exception as e:
            self.root.after(0, self.initial_analysis_text.insert, tk.END, 
                          f"Erro na análise inicial: {str(e)}")

    def analyze_image(self):
        """Realiza todas as análises complementares da imagem"""
        if self.current_image_path:
            self.analyze_basic_info()
            self.analyze_advanced_tags()
            self.analyze_metadata()

    def analyze_basic_info(self):
        """Analisa e exibe informações básicas da imagem"""
        try:
            image = Image.open(self.current_image_path)
            
            info = []
            info.append("=== INFORMAÇÕES BÁSICAS ===\n")
            info.append(f"Nome do arquivo: {os.path.basename(self.current_image_path)}")
            info.append(f"Dimensões: {image.size}")
            info.append(f"Formato: {image.format}")
            info.append(f"Modo: {image.mode}")
            info.append(f"Tamanho do arquivo: {os.path.getsize(self.current_image_path)/1024:.2f} KB")
            
            # Informações específicas do formato
            if image.format == 'JPEG':
                info.append("\nInformações específicas JPEG:")
                info.append(f"Subsampling: {image.info.get('subsampling', 'N/A')}")
                info.append(f"Qualidade: {image.info.get('quality', 'N/A')}")
            
            elif image.format == 'PNG':
                info.append("\nInformações específicas PNG:")
                info.append(f"Compressão: {image.info.get('compression', 'N/A')}")
                info.append(f"Transparência: {'Sim' if 'transparency' in image.info else 'Não'}")
            
            # Atualizar texto
            final_text = "\n".join(info)
            self.root.after(0, self.basic_info_text.delete, 1.0, tk.END)
            self.root.after(0, self.basic_info_text.insert, tk.END, final_text)
            
        except Exception as e:
            self.root.after(0, self.basic_info_text.delete, 1.0, tk.END)
            self.root.after(0, self.basic_info_text.insert, tk.END, 
                          f"Erro na análise: {str(e)}")

    def analyze_advanced_tags(self):
        """Analisa e exibe tags avançadas (EXIF) da imagem"""
        try:
            image = Image.open(self.current_image_path)
            
            info = []
            info.append("=== TAGS AVANÇADAS ===\n")
            
            # Tentar obter dados EXIF
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    for tag_id in exif:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exif.get(tag_id)
                        # Formatar dados especiais
                        if isinstance(data, bytes):
                            data = data.hex()
                        info.append(f"{tag}: {data}")
                else:
                    info.append("Nenhuma tag EXIF encontrada.")
            else:
                info.append("Imagem não contém dados EXIF.")
            
            # Atualizar texto
            final_text = "\n".join(info)
            self.root.after(0, self.advanced_tags_text.delete, 1.0, tk.END)
            self.root.after(0, self.advanced_tags_text.insert, tk.END, final_text)
            
        except Exception as e:
            self.root.after(0, self.advanced_tags_text.delete, 1.0, tk.END)
            self.root.after(0, self.advanced_tags_text.insert, tk.END, 
                          f"Erro na análise de tags: {str(e)}")

    def analyze_metadata(self):
        """Analisa e exibe metadados detalhados da imagem"""
        try:
            image = Image.open(self.current_image_path)
            
            info = []
            info.append("=== METADADOS DETALHADOS ===\n")
            
            # Informações do sistema de arquivos
            stats = os.stat(self.current_image_path)
            info.append("INFORMAÇÕES DO SISTEMA DE ARQUIVOS:")
            info.append(f"Tamanho: {stats.st_size/1024:.2f} KB")
            info.append(f"Data de criação: {datetime.fromtimestamp(stats.st_ctime)}")
            info.append(f"Última modificação: {datetime.fromtimestamp(stats.st_mtime)}")
            info.append(f"Último acesso: {datetime.fromtimestamp(stats.st_atime)}")
            info.append("")
            
            # Informações da imagem
            info.append("INFORMAÇÕES DA IMAGEM:")
            info.append(f"Formato: {image.format_description if hasattr(image, 'format_description') else image.format}")
            info.append(f"Modo de cor: {image.mode}")
            info.append(f"Paleta: {'Sim' if image.palette else 'Não'}")
            info.append(f"Dimensões: {image.size}")
            info.append("")
            
            # Informações específicas do formato
            info.append("INFORMAÇÕES ESPECÍFICAS DO FORMATO:")
            for key, value in image.info.items():
                if isinstance(value, bytes):
                    value = value.hex()
                info.append(f"{key}: {value}")
            
            # Informações OpenCV (se disponível)
            if cv2 is not None:
                try:
                    cv_image = cv2.imread(self.current_image_path)
                    if cv_image is not None:
                        info.append("\nINFORMAÇÕES OPENCV:")
                        info.append(f"Canais: {cv_image.shape[2] if len(cv_image.shape) > 2 else 1}")
                        info.append(f"Profundidade: {cv_image.dtype}")
                        info.append(f"Dimensões (altura x largura): {cv_image.shape[:2]}")
                        
                        # Análise de cores
                        if len(cv_image.shape) > 2:
                            means = cv2.mean(cv_image)
                            info.append("\nMédia de cores (BGR):")
                            info.append(f"Azul: {means[0]:.2f}")
                            info.append(f"Verde: {means[1]:.2f}")
                            info.append(f"Vermelho: {means[2]:.2f}")
                except Exception as cv_err:
                    info.append(f"\nErro ao obter informações OpenCV: {str(cv_err)}")
            
            # Atualizar texto
            final_text = "\n".join(info)
            self.root.after(0, self.metadata_text.delete, 1.0, tk.END)
            self.root.after(0, self.metadata_text.insert, tk.END, final_text)
            
        except Exception as e:
            self.root.after(0, self.metadata_text.delete, 1.0, tk.END)
            self.root.after(0, self.metadata_text.insert, tk.END, 
                          f"Erro na análise de metadados: {str(e)}")

def main():
    """Função principal que inicia a aplicação"""
    try:
        root = tk.Tk()
        app = ImageAnalyzer(root)
        root.mainloop()
    except Exception as e:
        print(f"Erro crítico: {str(e)}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()                                                  