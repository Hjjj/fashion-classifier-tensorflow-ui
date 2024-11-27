import tkinter as tk
from tkinter import ttk, simpledialog
from logic import AIFashionImageRecognitionModel  # Update import

class FashionImageRecognitionApp:  # Rename class
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fashion Vision")
        self.create_menu()
        self.create_widgets()
        # this is the AI model
        self.ai_model = AIFashionImageRecognitionModel(status_label=self.status_label)  # Update class name

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Add 'Retrain Model' menu item
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Retrain Model", command=self.retrain_model_dialog)
        menubar.add_cascade(label="Model", menu=model_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.next_button = ttk.Button(self.root, text="Next Image", command=self.display_next_image)
        self.next_button.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.prediction_label = tk.Label(self.root, text="")
        self.prediction_label.pack()

        self.status_label = tk.Label(self.root, text="Status: Ready")
        self.status_label.pack()

    def retrain_model_dialog(self):
        # Open a dialog to ask for the number of epochs to train
        epochs = simpledialog.askinteger("Retrain Model", "Enter number of epochs:", initialvalue=2, minvalue=1)
        if epochs is not None:
            self.ai_model.retrain(epochs)

    def display_next_image(self):
        #this function gets the next image from a dataset, and 
        #queries the AI model to recognize what the image is
        image_tk, predicted_class = self.ai_model.interpret_next_image()  # Update method call
        
        if image_tk:
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk
            self.prediction_label.config(text=f"Predicted Class: {predicted_class}")
        else:
            self.prediction_label.config(text="No more images")

    def run(self):
        self.root.mainloop()