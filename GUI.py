import cv2
import tkinter as tk
from PIL import Image, ImageTk

hand_cascade = cv2.CascadeClassifier('harrcascade/palm.xml')
fist_cascade = cv2.CascadeClassifier('harrcascade/fist.xml')
C_cascade = cv2.CascadeClassifier('harrcascade/C4.xml')
E_cascade = cv2.CascadeClassifier('harrcascade/E5.xml')
D_cascade = cv2.CascadeClassifier('harrcascade/D3.xml')
F_cascade = cv2.CascadeClassifier('harrcascade/F2.xml')
G_cascade = cv2.CascadeClassifier('harrcascade/G5.xml')
H_cascade = cv2.CascadeClassifier('harrcascade/H.xml')
I_cascade = cv2.CascadeClassifier('harrcascade/I.xml')
K_cascade = cv2.CascadeClassifier('harrcascade/K4_24.xml')
L_cascade = cv2.CascadeClassifier('harrcascade/L.xml')
# M_cascade = cv2.CascadeClassifier('harrcascade/M1.xml')
N_cascade = cv2.CascadeClassifier('harrcascade/Nkk.xml')
O_cascade = cv2.CascadeClassifier('harrcascade/O.xml')
# P_cascade = cv2.CascadeClassifier('harrcascade/P.xml')
Q_cascade = cv2.CascadeClassifier('harrcascade/Q.xml')
R_cascade = cv2.CascadeClassifier('harrcascade/R.xml')
S_cascade = cv2.CascadeClassifier('harrcascade/S.xml')
T_cascade = cv2.CascadeClassifier('harrcascade/T1.xml')
U_cascade = cv2.CascadeClassifier('harrcascade/U1.xml')
V_cascade = cv2.CascadeClassifier('harrcascade/Vkk.xml')
W_cascade = cv2.CascadeClassifier('harrcascade/W.xml')
X_cascade = cv2.CascadeClassifier('harrcascade/Xkk.xml')
Y_cascade = cv2.CascadeClassifier('harrcascade/y_own.xml')
nice_cascade = cv2.CascadeClassifier('harrcascade/nice.xml')
bad_cascade = cv2.CascadeClassifier('harrcascade/bad.xml')

# Set up GUI
window = tk.Tk()  # Makes main window
window.wm_title("SIGN LANGUAGE RECOGNITION")
window.config(background="#FFFFFF")


# Graphics window
imageFrame = tk.Frame(window, width=700, height=400)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

# Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

def show_frame():
    _, frame = cap.read()
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)
    crop = frame[100:350, 100:350]
    crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    black = cv2.imread("images/black.jpg")

    A = fist_cascade.detectMultiScale(crop_img)
    B = hand_cascade.detectMultiScale(crop_img, 1.3, 5)
    C = C_cascade.detectMultiScale(crop_img, 1.3, 5)
    D = D_cascade.detectMultiScale(crop_img, 1.3,5)
    E = E_cascade.detectMultiScale(crop_img)
    F = F_cascade.detectMultiScale(crop_img, 1.1,5)
    G = G_cascade.detectMultiScale(crop_img, 3.5, 5)
    H = H_cascade.detectMultiScale(crop_img, 1.6, 5)
    I = I_cascade.detectMultiScale(crop_img)
    K = K_cascade.detectMultiScale(crop_img, 3, 4)
    L = L_cascade.detectMultiScale(crop_img, 1.3, 4)
    # M = M_cascade.detectMultiScale(crop_img)
    N = N_cascade.detectMultiScale(crop_img, 2.4, 5)
    O = O_cascade.detectMultiScale(crop_img, 1.2, 5)
    # P = P_cascade.detectMultiScale(crop_img, 1.3, 5)
    Q = Q_cascade.detectMultiScale(crop_img)
    R = R_cascade.detectMultiScale(crop_img, 1.3, 5)
    S = S_cascade.detectMultiScale(crop_img, 1.1, 5)
    T = T_cascade.detectMultiScale(crop_img)
    U = U_cascade.detectMultiScale(crop_img)
    V = V_cascade.detectMultiScale(crop_img)
    W = W_cascade.detectMultiScale(crop_img)
    X = X_cascade.detectMultiScale(crop_img, 1.3, 5)
    Y = Y_cascade.detectMultiScale(crop_img)

    for (hx, hy, hw, hh) in A:
        cv2.putText(black, 'A', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'A', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in B:
        cv2.putText(black, 'B', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'B', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in C:
        cv2.putText(black, 'C', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'C', (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in D:
        cv2.putText(black, 'D', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'D', (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in E:
        cv2.putText(black, 'E', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'E', (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in F:
        cv2.putText(black, 'F', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'F', (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in G:
        cv2.putText(black, 'G', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'G', (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in H:
        cv2.putText(black, 'H', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'H', (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in I:
        cv2.putText(black, 'I', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'I', (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in K:
        cv2.putText(black, 'K', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'K', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in L:
        cv2.putText(black, 'L', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'L', (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # for (hx, hy, hw, hh) in M:
    #     cv2.putText(black, 'M', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, 'M', (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in N:
        cv2.putText(black, 'N', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'N', (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in O:
        cv2.putText(black, 'O', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'O', (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # for (hx, hy, hw, hh) in P:
    #     cv2.putText(black, 'P', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, 'P', (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in Q:
        cv2.putText(black, 'Q', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Q', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in R:
        cv2.putText(black, 'R', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'R', (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in S:
        cv2.putText(black, 'S', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'S', (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in W:
        cv2.putText(black, 'W', (180, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'W', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in Y:
        cv2.putText(black, 'Y', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Y', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in X:
        cv2.putText(black, 'X', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'X', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in T:
        cv2.putText(black, 'T', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'T', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in U:
        cv2.putText(black, 'U', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'U', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    for (hx, hy, hw, hh) in V:
        cv2.putText(black, 'V', (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'V', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # for (hx, hy, hw, hh) in nice:
    #     cv2.putText(black, 'Nice', (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, 'Nice', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # for (hx, hy, hw, hh) in bad:
    #     cv2.putText(black, 'Bad', (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, 'Bad', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)

    # cv2.imshow("Output", black)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


# Slider window (slider controls stage position)
#sliderFrame = tk.Frame(window, width=600, height=100)
#sliderFrame.grid(row=600, column=0, padx=10, pady=2)

cv2.destroyAllWindows()
show_frame()  # Display 2
window.mainloop()  # Starts GUI
#w2.mainloop()