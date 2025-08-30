import pyautogui
import time

def runShortcut(*keys):
    pyautogui.hotkey(*keys)
    time.sleep(0.1) 

#screenshots -> Win + PrtSc
def screenshotFull():
    runShortcut('win', 'printscreen')

#maximizar ventana -> Win + Flecha arriba
def maximizeWindow():
    runShortcut('win', 'up')

#minimizar ventana -> Win + Flecha abajo
def minimizeWindow():
    runShortcut('win', 'down')

#cambiar de escritorio -> Ctrl + Win + Flecha derecha o Flecha izquierda
def switchDesktopRight():
    runShortcut('ctrl', 'win', 'right')

def switchDesktopLeft():
    runShortcut('ctrl', 'win', 'left')

#ver todas las ventanas abiertas -> Win + Tab
def taskView():
    runShortcut('win', 'tab')
