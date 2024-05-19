import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pickle

class TrafficSignRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition App")
        self.root.geometry("1200x700")

        # Create a frame for the video feed
        self.video_frame = tk.Frame(self.root, width=900, height=500)
        self.video_frame.grid(row=1, column=1)

        # UI Frame with right margin
        self.ui_frame = tk.Frame(self.root, width=400, height=600, bg="#E6E6E6", bd=2, relief="raised")
        self.ui_frame.grid(row=1, column=2, sticky="nsew", padx=10)  # Add padding on the right

        # UI Canvas
        self.ui_canvas = tk.Canvas(self.ui_frame, width=400, height=600, bg="#E6E6E6")
        self.ui_canvas.pack(fill="both", expand=True)

        # Add labels for group and prediction
        self.group_label = tk.Label(self.ui_canvas, bg="#E6E6E6", text="GROUP 9 \n Diệp Hoài An - 21110001 \n Võ Hoàng Tùng - 21110811 \n Trần Phi Tường - 21110108", font=("Raleway", 15, "bold"), fg="black")
        self.group_label.place(relx=0.5, rely=0, anchor="n")

        # Prediction section
        prediction_frame = tk.Frame(self.ui_canvas, bg="#E6E6E6", bd=1, relief="solid")
        prediction_frame.place(relx=0.5, rely=0.5, anchor="center")

        prediction_label = tk.Label(prediction_frame,bg="#E6E6E6", text="Predicted Class:", font=("Helvetica", 15, "bold"), fg="black")
        prediction_label.pack(side="top", padx=10, pady=5)

        self.prediction_value_label = tk.Label(prediction_frame, bg="#E6E6E6", text="", font=("Helvetica", 14, "bold"), fg="green")
        self.prediction_value_label.pack(side="top", padx=10, pady=5)

        
        self.app_name_label = tk.Label(self.root, text="TRAFFIC SIGN DETECTION", font=("Rockwell Extra Bold", 30))
        self.app_name_label.grid(row=0, column=1)
        
        # Create a button to start the program
        self.start_button = tk.Button(self.root, text="Start", font=("Baskerville Old Face", 25, "bold"),bg= '#FFFBDA', height=2, width=10, command=self.start_program)
        self.start_button.grid(row=1, column=1)

        
        # Load the model
        with open('model_trained_new.p', 'rb') as file:
            self.model = pickle.load(file)

        # Set up the camera
        self.cap = cv2.VideoCapture(0)

        # Cân bằng cột và hàng
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=3)
        self.root.grid_rowconfigure(2, weight=1)
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_columnconfigure(2, weight=1)

    def start_program(self):
        self.start_button.config(state="disabled")
        self.start_button.grid_remove()
                
        # Create a label for the video feed
        self.video_label = tk.Label(self.video_frame, text="Video Feed")
        self.video_label.pack()
        
        self.video_feed()

    def video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Process the frame
            mask_red, mask_blue, mask_yellow, mask_black_white = self.return_color_masks(frame)

            # Combine the masks using bitwise OR
            mask = cv2.bitwise_or(mask_red, mask_blue)
            mask = cv2.bitwise_or(mask, mask_yellow)
            mask = cv2.bitwise_or(mask, mask_black_white)

            thresh = self.threshold(mask)
            try:
                contours = self.find_contour(thresh)
                if len(contours) > 0:
                    big_contour = self.find_biggest_contour(contours)
                    if cv2.contourArea(big_contour) > 3000:
                        img, sign = self.boundary_box(frame, big_contour)
                        sign = cv2.resize(sign, (32, 32))
                        sign = self.preprocessing(sign)
                        predictions = self.model.predict(sign)
                        class_index = np.argmax(predictions)
                        class_name = self.get_class_name(class_index)
                        probability = predictions[0][class_index]
                        self.display_prediction(class_name, probability)
                    else:
                        img = frame
                else:
                    img = frame
                # Convert the NumPy array to a PIL Image object
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Convert the PIL Image object to a PhotoImage object
                img = ImageTk.PhotoImage(img)

                # Set the image attribute of the video_label to the PhotoImage object
                self.video_label.config(image=img)

                # Keep a reference to the PhotoImage object to prevent it from being garbage collected
                self.video_label.image = img
            except Exception as e:
                print(e)
                img = frame

            # Call the video_feed function again after 30ms
            self.root.after(30, self.video_feed)

    def return_color_masks(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red mask
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = mask_red1 + mask_red2

        # Blue mask
        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Yellow mask
        lower_yellow = np.array([20, 120, 70])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Black/White mask
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        lower_white = np.array([0, 0, 200])
        upper_white= np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black_white = cv2.bitwise_or(mask_black, mask_white)

        return mask_red, mask_blue, mask_yellow, mask_black_white

    def threshold(self, img, T=150):
        _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
        return img

    def find_contour(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_biggest_contour(self, contours):
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        return max_contour

    def boundary_box(self, img, contour):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sign = img[y:y+h, x:x+w]
        return img, sign

    def preprocessing(self, img):
        img = cv2.resize(img, (32, 32))
        img = img.astype('float32') / 255
        return np.expand_dims(img,axis=0)

    def get_class_name(self, classNo):
        if   classNo == 0: return 'Speed Limit 20 km/h'
        elif classNo == 1: return 'Speed Limit 30 km/h'
        elif classNo == 2: return 'Speed Limit 50 km/h'
        elif classNo == 3: return 'Speed Limit 60 km/h'
        elif classNo == 4: return 'Speed Limit 70 km/h'
        elif classNo == 5: return 'Speed Limit 80 km/h'
        elif classNo == 6: return 'End of Speed Limit 80 km/h'
        elif classNo == 7: return 'Speed Limit 100 km/h'
        elif classNo == 8: return 'Speed Limit 120 km/h'
        elif classNo == 9: return 'No passing'
        elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
        elif classNo == 11: return 'Right-of-way at the next intersection'
        elif classNo == 12: return 'Priority road'
        elif classNo == 13: return 'Yield'
        elif classNo == 14: return 'Stop'
        elif classNo == 15: return 'No vechiles'
        elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
        elif classNo == 17: return 'No entry'
        elif classNo == 18: return 'General caution'
        elif classNo == 19: return 'Dangerous curve to the left'
        elif classNo == 20: return 'Dangerous curve to the right'
        elif classNo == 21: return 'Double curve'
        elif classNo == 22: return 'Bumpy road'
        elif classNo == 23: return 'Slippery road'
        elif classNo == 24: return 'Road narrows on the right'
        elif classNo == 25: return 'Road work'
        elif classNo == 26: return 'Traffic signals'
        elif classNo == 27: return 'Pedestrians'
        elif classNo == 28: return 'Children crossing'
        elif classNo == 29: return 'Bicycles crossing'
        elif classNo == 30: return 'Beware of ice/snow'
        elif classNo == 31: return 'Wild animals crossing'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 34: return 'Turn left ahead'
        elif classNo == 35: return 'Ahead only'
        elif classNo == 36: return 'Go straight or right'
        elif classNo == 37: return 'Go straight or left'
        elif classNo == 38: return 'Keep right'
        elif classNo == 39: return 'Keep left'
        elif classNo == 40: return 'Roundabout mandatory'
        elif classNo == 41: return 'End of no passing'
        elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


    def display_prediction(self, class_name, probability):
        self.prediction_value_label.config(text=f"{class_name}\n Probability: {probability*100:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignRecognitionApp(root)
    root.mainloop()