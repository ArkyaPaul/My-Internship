import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

class MouthDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry('800x600')
        self.master.title('Mouth Detector')
        self.master.configure(background='#CDCDCD')

        self.label1 = Label(self.master, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.master)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        self.model = self.load_mouth_expression_model("mouth_open_close.json", "mouth_open_close_model.h5")

        self.EMOTIONS_LIST = ["Mouth open", "Mouth Closed"]

        self.setup_gui()

    def setup_gui(self):
        upload_button = Button(self.master, text="Upload Image", command=self.upload_image, padx=10, pady=5)
        upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
        upload_button.pack(side='bottom', pady=50)

        self.sign_image.pack(side='bottom', expand=True)
        self.label1.pack(side='bottom', expand=True)

        heading = Label(self.master, text='Mouth Detector', pady=20, font=('arial', 25, 'bold'))
        heading.configure(background='#CDCDCD', foreground="#364156")
        heading.pack()

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.master.winfo_width() / 2.25), (self.master.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label1.configure(text='')
            self.show_detect_button(file_path)
        except Exception as e:
            print(f"Error uploading image: {e}")

    def show_detect_button(self, file_path):
        detect_button = Button(self.master, text="Mouth Detection", command=lambda: self.detect(file_path), padx=10, pady=5)
        detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_button.place(relx=0.79, rely=0.46)

    def detect(self, file_path):
        try:
            image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = gray_image[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                pred = self.EMOTIONS_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]

            print("Predicted Emotion is " + pred)
            self.label1.configure(foreground="#011638", text=pred)
        except Exception as e:
            self.label1.configure(foreground="#011638", text="Unable to detect")

    def load_mouth_expression_model(self, mouth_open_close, mouth_open_close_model):
        try:
            with open(mouth_open_close, "r") as file:
                loaded_model_json = file.read()
                model = model_from_json(loaded_model_json)

            model.load_weights(mouth_open_close_model)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            return model
        except Exception as e:
            print(f"Error loading mouth expression model: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MouthDetectorApp(root)
    root.mainloop()
