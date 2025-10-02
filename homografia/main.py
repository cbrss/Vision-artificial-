import numpy as np
import cv2 as cv

points = []

class camera():

    def __init__(self, cv):
        self.mode = "df"
        self.cv = cv
        self.dst_pts = np.array([
            [0, 0],
            [200, 0],
            [200, 200],
            [0, 200]
        ], dtype=np.float32)
        self.detector = cv.QRCodeDetector()

    def modeQR(self):
        self.mode = "qr"
        
    def modeHG(self):
        self.mode = "hg"

    def qrProcess(self,frame):

        _,points = self.detector.detect(frame)
        if points is None:
            return None
        points = points.astype(np.float32)
        H, _ = self.cv.findHomography(points, self.dst_pts)
        warp = self.cv.warpPerspective(frame,H,(200,200))
        return warp
    
    def hgProcess(self,frame,points):
        src_pts = np.array(points, dtype=np.float32)
        H, _ = cv.findHomography(src_pts, self.dst_pts)
        warp = cv.warpPerspective(frame, H, (200,200))
        return warp


def on_mouse_click(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # máximo 4 puntos
            points.append((x, y))
            print(f"Punto agregado: {(x, y)}")
        else:
            print("Ya se seleccionaron 4 puntos, presiona 'r' para reiniciar.")

def loop():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    cam = camera(cv)
    cv.namedWindow("Front")
    frontFrame = np.zeros((200,200,3), dtype=np.uint8)
    cv.imshow("Front", frontFrame)
    cv.namedWindow("Camera Frame")
    cv.setMouseCallback("Camera Frame", on_mouse_click)
    global points
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        #cv.imshow('frame', gray)
        #cv.imshow('frame',RGB)

        cv.imshow("Frame",frame)

        

        if cam.mode == "qr":
            warp = cam.qrProcess(frame)
            if warp is not None:
                cv.imshow("Front", warp)
        if cam.mode == "hg":
            for pt in points:
                cv.circle(frame, pt, 5, (0,255,0), -1)
            cv.imshow("Camera Frame", frame)
            if len(points) == 4:
                warp = cam.hgProcess(frame,points)
                if warp is not None:
                    cv.imshow("Front", warp)

        key = cv.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cam.modeQR()
        elif key == ord('h'):
            cam.modeHG()
        elif key == ord('r'):
            points = []
            
        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    loop()