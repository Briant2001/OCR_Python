from PyQt6.QtWidgets import QApplication, QWidget,QLabel,QVBoxLayout,QPushButton,QHBoxLayout,QDoubleSpinBox,QCheckBox,QSpinBox,QComboBox
from PyQt6.QtGui import QIcon,QImage,QPixmap,QFont  
from PyQt6.QtCore import QThread,pyqtSignal as Signal,pyqtSlot as Slot
import sys
import cv2
import easyocr
import numpy as np
from PIL import ImageEnhance,Image



reader = easyocr.Reader(['en'],gpu=True)
vcontraste = 1
vbrillo = 1
vsharp = 1
vautcontraste = 0
vgamma = 1
VX1a = 0
VX2a = 479
VY1a = 0
VY2a = 639
Vdilatacion = 0
Vinvierte=0
Vbinarizar=0
Vocr=0
VGauss=1
VRGauss=0
cap = None

class MyThread(QThread):
    frame_signal = Signal(QImage)
    
    def run(self):
        global lectura
        global vcontraste
        global vbrillo,vsharp,vgamma,VX1a,VX2a,VY1a,VY2a,vautcontraste,Vinvierte,Vdilatacion,Vbinarizar,Vocr,VRGauss,VGauss
        global cap

        #self.cap = cv2.VideoCapture(0)

        while cap and cap.isOpened():
            exito,frame = cap.read()
            if (exito):
                print("Abrio camara")
                if (VX1a<VX2a) and (VY1a<VY2a):
                    frame = frame[VY1a:VY2a+1,VX1a:VX2a+1:]
                #print(frame.shape)
                #print(vgamma)
                
                if vautcontraste==1:
                    frame,a,b=automatic_brightness_and_contrast(frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if VRGauss==1:
                    #frame = cv2.GaussianBlur(frame, (5, 5), VGauss)
                    print(VGauss)
                    frame = cv2.medianBlur(frame, VGauss)
                
                #frame= cv2.GaussianBlur(frame, (5, 5), 3)
                frame=adjust_gamma(frame, vgamma)
                
                if Vinvierte==1:

                    frame=255-frame

                kernel = np.ones((5, 5), np.uint8)
                kernel2 = np.ones((3, 3), np.uint8)
                #frame = cv2.morphologyEx(frame , cv2.MORPH_CLOSE, kernel)
                #frame = cv2.morphologyEx(frame , cv2.MORPH_OPEN, kernel2)
                #frame = cv2.erode(frame, None, iterations=1)
                
                if Vbinarizar==1:
                    frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                kernel_cuadrado = np.ones((2, 2), np.uint8)
                frame = cv2.dilate(frame, kernel_cuadrado, iterations=Vdilatacion)
                frame = cv2.erode(frame, kernel_cuadrado, iterations=Vdilatacion)

                frame = Image.fromarray(frame)
                enhancer = ImageEnhance.Brightness(frame)
                frame = enhancer.enhance(vbrillo)
                enhancer = ImageEnhance.Contrast(frame)
                frame = enhancer.enhance(vcontraste)
                enhancer = ImageEnhance.Sharpness(frame)
                frame = enhancer.enhance(vsharp)

                
                

                frame=np.asarray(frame)
                if Vocr==1:
                    result = reader.readtext(frame,allowlist='0123456789.',detail=1)
                    rp1 = [item[1] for item in result]
                    if len(rp1)>0:
                        lectura = rp1[0]
                        print(rp1[0])
                    else:
                        lectura = '0.0'
                        
                        
                #lectura=result[0][1]
                lectura='0.0'
            else:
                cap = None
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = self.cvimage_to_label(frame)
            self.frame_signal.emit(frame)

    
    
    def cvimage_to_label(self,image):
        #image = imutils.resize(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        #print(image.shape[1],image.shape[0])
        image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        return image
    # Brillo y contraste automatico
def adjust_gamma(image, gamma=1.0):
    """
        Use this function to adjust illumination in an image.
        Credit: https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        :param image: A grayscale image (NxM int array in [0, 255]
        :param gamma: A positive float. If gamma<1 the image is darken / if gamma>1 the image is enlighten / if gamma=1 nothing happens.
        :return: the enlighten/darken version of image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.available_cameras = self.detect_cameras()
        self.current_camera_index = -1  # Inicialmente no hay cámara seleccionada
        #self.cap = None

        self.setGeometry(100,50,1000,700)
        self.setWindowTitle("OCR con Python ")
        self.setWindowIcon(QIcon('python.png'))
        #self.setFixedHeight(400)
        #self.setFixedWidth(700)
        #self.setStyleSheet('background-color:green')
        #self.setWindowOpacity(0.5)
        self.label1=QLabel("No hay cámara seleccionada.")

        self.lecturar=QLabel(self)
        self.lecturar.setText("Lectura")
        self.lecturar.setStyleSheet('color:red')
        self.lecturar.setFont(QFont("Times", 25)) 
        
        self.y1a = QSpinBox()
        self.y1a.setPrefix("Y1: ")
        self.y1a.setRange(0, 475)
        self.y1a.setValue(0)

        self.y2a = QSpinBox()
        self.y2a.setPrefix("Y2: ")
        self.y2a.setRange(5, 479)
        self.y2a.setValue(479)

        self.x1a = QSpinBox()
        self.x1a.setPrefix("X1: ")
        self.x1a.setRange(0, 635)
        self.x1a.setValue(0)

        self.x2a = QSpinBox()
        self.x2a.setPrefix("X2: ")
        self.x2a.setRange(5, 639)
        self.x2a.setValue(639)

        self.autocontraste = QCheckBox("Autocontraste")
        self.autocontraste.stateChanged.connect(self.actualiza_valores)

        self.gamma = QDoubleSpinBox()
        self.gamma.valueChanged.connect(self.actualiza_valores) 

        self.contraste = QDoubleSpinBox()
        self.contraste.valueChanged.connect(self.actualiza_valores) 

        self.brillo = QDoubleSpinBox()
        self.brillo.valueChanged.connect(self.actualiza_valores) 
        
        self.sharp= QDoubleSpinBox()
        self.sharp.valueChanged.connect(self.actualiza_valores)

        self.dilatacion=QSpinBox()
        self.dilatacion.setRange(0, 10)
        self.dilatacion.setValue(0)
        self.dilatacion.setPrefix("Cierre:")
        self.dilatacion.setSingleStep(1)
        self.dilatacion.valueChanged.connect(self.actualiza_valores)

        self.invierte=QCheckBox("Invertir")
        self.invierte.stateChanged.connect(self.actualiza_valores)
        self.invierte.setChecked(False)

        self.binarizar=QCheckBox("Binarizar")
        self.binarizar.stateChanged.connect(self.actualiza_valores)
        self.binarizar.setChecked(False)

        self.ocr=QCheckBox("OCR")
        self.ocr.setChecked(False)
        self.ocr.stateChanged.connect(self.actualiza_valores)

        self.RGauss=QCheckBox("Gauss")
        self.RGauss.setChecked(False)
        self.RGauss.stateChanged.connect(self.actualiza_valores)

        self.Gauss = QSpinBox()
        self.Gauss.valueChanged.connect(self.actualiza_valores)
        self.Gauss.setRange(1, 11)
        self.Gauss.setValue(1)
        self.Gauss.setPrefix("Gauss:")
        self.Gauss.setSingleStep(2)
        


        
        vboxcontroles = QVBoxLayout()
        vboxcontroles.addWidget(self.x1a)
        vboxcontroles.addWidget(self.x2a)
        vboxcontroles.addWidget(self.y1a)
        vboxcontroles.addWidget(self.y2a)
        vboxcontroles.addWidget(self.binarizar)
        vboxcontroles.addWidget(self.invierte)
        vboxcontroles.addWidget(self.dilatacion)
        vboxcontroles.addWidget(self.RGauss)
        vboxcontroles.addWidget(self.Gauss)
        vboxcontroles.addWidget(self.autocontraste)
        vboxcontroles.addWidget(self.ocr)
        vboxcontroles.addWidget(self.gamma)
        vboxcontroles.addWidget(self.contraste)
        vboxcontroles.addWidget(self.brillo)
        vboxcontroles.addWidget(self.sharp)
        hbox = QHBoxLayout()
        hbox.addWidget(self.label1)
        hbox.addWidget(self.lecturar)
        hbox.addLayout(vboxcontroles)   
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.contraste)

        self.available_cameras = self.detect_cameras()
        self.current_camera_index = -1  # Inicialmente no hay cámara seleccionada
        #self.cap = None

        self.camera_label = QLabel("Selecciona una cámara:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Ninguna cámara seleccionada")  # Opción inicial
        self.camera_combo.addItems([f"Cámara {i}" for i in range(len(self.available_cameras))])
        self.camera_combo.currentIndexChanged.connect(self.open_camera)

        #self.open_btn = QPushButton("Abrir cámara", clicked=self.open_camera)
        vbox.addWidget(self.camera_combo)
        self.setLayout(vbox)
        
        self.gamma.setRange(0,1.0)
        self.gamma.setSingleStep(.05)
        self.gamma.setValue(1)
        self.gamma.setPrefix("Gamma:")
        

        self.contraste.setRange(0,1.0)
        self.contraste.setSingleStep(0.05)
        self.contraste.setValue(1)
        self.contraste.setPrefix("Contraste:")
        
        self.brillo.setRange(0,1.0)
        self.brillo.setSingleStep(0.05)
        self.brillo.setValue(1)
        self.brillo.setPrefix("Brillo:")

        self.sharp.setRange(0,2.0)
        self.sharp.setSingleStep(0.05)
        self.sharp.setValue(1)
        self.sharp.setPrefix("Sharp:")

        self.camera_thread = MyThread()
        self.camera_thread.frame_signal.connect(self.setImage)

        self.x1a.valueChanged.connect(self.actualiza_valores)
        self.x2a.valueChanged.connect(self.actualiza_valores)
        
        self.y1a.valueChanged.connect(self.actualiza_valores)
        
        self.y2a.valueChanged.connect(self.actualiza_valores)
    
    def detect_cameras(self):
        global cap
        """Intenta detectar las cámaras disponibles."""
        available_cameras = []
        for i in range(10):  # Prueba los primeros 10 índices de cámara
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def init_camera(self, camera_index):
        global cap
        """Intenta inicializar la captura de video con un índice de cámara específico."""
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return cap
        else:
            return None

    
    def actualiza_valores(self):
        global vcontraste
        global vbrillo
        global vsharp
        global vautcontraste
        global vgamma,VX1a,VX2a,VY1a,VY2a,Vdilatacion,Vinvierte,Vbinarizar,Vocr,VRGauss,VGauss

        Vocr = self.ocr.isChecked()
        VRGauss = self.RGauss.isChecked()
        VGauss = self.Gauss.value()
        Vdilatacion = self.dilatacion.value()
        Vbinarizar = self.binarizar.isChecked()
        Vinvierte = self.invierte.isChecked()
        VX1a= self.x1a.value()
        VX2a = self.x2a.value()
        VY1a = self.y1a.value()
        VY2a = self.y2a.value()
        vcontraste = self.contraste.value()
        vbrillo= self.brillo.value()
        vsharp= self.sharp.value()
        vautcontraste = self.autocontraste.isChecked()
        vgamma = self.gamma.value()
        #VX1a = self.x1.value()
        #VX2a = self.x2.value()
        #VY1a = self.y1.value()
        #VY2a = self.y2.value()

        

    def open_camera(self,index):
        global cap
        if cap and cap.isOpened():
            cap.release()
            #self.timer.stop()
            self.label1.setText("Cámara detenida.")
            cap = None
            self.current_camera_index = -1

        if index > 0:  # El índice 0 es "Ninguna cámara seleccionada"
            selected_camera_index = self.available_cameras[index - 1]
            cap = self.init_camera(selected_camera_index)
            if cap:
                self.current_camera_index = selected_camera_index
                self.camera_thread.start()
                #self.timer.start(30)
            else:
                self.label1.setText(f"La cámara {selected_camera_index} no está disponible.")
        else:
            self.label1.setText("No hay cámara seleccionada.")        
        
        

    @Slot(QImage)
    def setImage(self,image):
        global lectura,cap

        if (cap!=None):
            self.label1.setPixmap(QPixmap.fromImage(image))
            self.lecturar.setText(lectura)
        else:
            self.label1.setText("Cámara detenida.")
            self.current_camera_index = -1
            self.camera_combo.setCurrentIndex(0)

        
        #
        #width = image.width()
        #height = image.height()

        #ptr = image.bits()
        #ptr.setsize(height * width * 4)
        #arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        #result = reader.readtext(arr)
        #rint(result)

app=QApplication(sys.argv)

window=Window()
window.show()

sys.exit(app.exec())
