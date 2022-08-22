import PIL.Image, PIL.ImageTk
import model


class App:

    def __init__(self):
        self.photo = None
        self.model = model.Model()

    def update(self):
        if self.counting_enabled:
            self.predict()

        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1

        self.counter_label.config(text=f"{self.rep_counter}")

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.W)

        self.window.after(self.delay, self.update)

    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)
        if prediction != self.last_prediction:
            if prediction == 1:
                self.extended = True
                self.last_prediction = 1
            if prediction == 2:
                self.contracted = True
                self.last_prediction = 2

