#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw
import os
import sys

# Import your custom modules
from model_classes import GPTConvolutionalNetwork as ModelClass
from utils import device

class DrawingApp:
    def __init__(self, root, model, classes):
        self.root = root
        self.root.title("Drawing Classifier")
        self.model = model
        self.classes = classes
        
        # Configure main window
        self.root.geometry("900x600")
        self.root.config(bg="#f0f0f0")
        
        # Create main frames
        self.left_frame = tk.Frame(root, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for drawing - MAKE IT SQUARE
        self.canvas_frame = tk.Frame(self.left_frame, bg="white", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Fixed square canvas
        self.canvas_size = 400  # Square size in pixels
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg="white", cursor="cross")  # Changed background to white
        self.canvas.pack(padx=10, pady=10)
        
        # Initialize drawing variables
        self.setup_drawing()
        
        # Buttons frame
        self.button_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Clear button
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Classify button
        self.classify_button = ttk.Button(self.button_frame, text="Classify", command=self.classify_drawing)
        self.classify_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Line width options
        self.line_width_label = tk.Label(self.button_frame, text="Line Width:", bg="#f0f0f0")
        self.line_width_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.line_width_var = tk.IntVar(value=3)  # Default thinner line
        self.line_width_slider = ttk.Scale(self.button_frame, from_=1, to=10, 
                                          variable=self.line_width_var, orient="horizontal",
                                          length=100, command=self.update_line_width)
        self.line_width_slider.pack(side=tk.LEFT, padx=5)
        
        # Results area
        self.results_label = tk.Label(self.right_frame, text="Drawing Classification Results", 
                                     font=("Arial", 16), bg="#f0f0f0")
        self.results_label.pack(pady=10)
        
        # Create figure for bar chart
        self.fig, self.ax = plt.subplots(figsize=(4, 6))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Preview of processed image
        self.preview_label = tk.Label(self.right_frame, text="Processed Input:", bg="#f0f0f0")
        self.preview_label.pack(pady=(20, 5))
        
        self.preview_canvas = tk.Canvas(self.right_frame, width=140, height=140, bg="white", bd=1, relief=tk.SUNKEN)
        self.preview_canvas.pack(pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Draw something and click 'Classify'")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def update_line_width(self, event=None):
        self.line_width = self.line_width_var.get()
        
    def setup_drawing(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 3  # Start with a thinner line
        # Create new drawing image based on canvas size
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.draw_image)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
    def start_draw(self, event):
        self.old_x = event.x
        self.old_y = event.y
        
    def draw_line(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                  width=self.line_width, fill="black",  # Changed to black
                                  capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on the PIL image too
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill=0, width=self.line_width)  # 0 is black in "L" mode
            
        self.old_x = event.x
        self.old_y = event.y
        
    def end_draw(self, event):
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.draw_image)
        self.ax.clear()
        self.canvas_plot.draw()
        self.preview_canvas.delete("all")
        self.status_var.set("Canvas cleared")
        
    def preprocess_image(self):
        """
        Process the drawing to match QuickDraw dataset format.
        Returns a tensor ready for model input.
        """
        # First resize to 28x28
        img_resized = self.draw_image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy array and normalize (invert colors to match training data)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Show preview of the processed image
        self.update_preview(img_array)
        
        # Convert to tensor (add batch and channel dimensions)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return img_tensor
    
    def update_preview(self, img_array):
        """Update the preview of the processed image"""
        # Scale up the 28x28 image to show in the preview canvas
        preview_size = 140
        scale_factor = preview_size / 28
        
        # Clear previous preview
        self.preview_canvas.delete("all")
        
        # Draw each pixel as a rectangle
        for y in range(28):
            for x in range(28):
                # Get pixel value (0 to 1, where 0 is black)
                val = img_array[y, x]
                
                # Convert to 0-255 grayscale value (inverse because 0 is black)
                color_val = int((1 - val) * 255)
                color = f'#{color_val:02x}{color_val:02x}{color_val:02x}'
                
                # Calculate rectangle coordinates
                x0, y0 = x * scale_factor, y * scale_factor
                x1, y1 = (x + 1) * scale_factor, (y + 1) * scale_factor
                
                # Only draw non-white pixels for efficiency
                if color_val > 10:  # Threshold to avoid drawing very light pixels
                    self.preview_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        
    def classify_drawing(self):
        # Update status
        self.status_var.set("Classifying...")
        self.root.update()
        
        # Check if canvas is empty
        img_array = np.array(self.draw_image)
        if np.all(img_array == 255):  # All white means empty
            self.status_var.set("Please draw something first!")
            return
        
        # Preprocess the image
        img_tensor = self.preprocess_image()
        
        try:
            # Move tensor to the same device as the model
            img_tensor = img_tensor.to(device)
            
            # Get model prediction
            with torch.no_grad():
                logits = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
            # Get top 10 predictions
            top_k = min(10, len(self.classes))
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Convert to numpy for display
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            # Get class names for top predictions
            top_classes = [self.classes[idx] for idx in top_indices]
            
            # Display results
            self.display_results(top_classes, top_probs)
            
            # Update status
            self.status_var.set(f"Top prediction: {top_classes[0]} ({top_probs[0]*100:.2f}%)")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def display_results(self, classes, probabilities):
        # Clear previous plot
        self.ax.clear()
        
        # Create horizontal bar chart
        y_pos = np.arange(len(classes))
        self.ax.barh(y_pos, probabilities * 100, align='center')
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels([c.capitalize() for c in classes])
        self.ax.invert_yaxis()  # Labels read top-to-bottom
        self.ax.set_xlabel('Probability (%)')
        self.ax.set_title('Top Predictions')
        
        # Add percentage text on bars
        for i, v in enumerate(probabilities):
            self.ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center')
            
        # Adjust layout and redraw
        self.fig.tight_layout()
        self.canvas_plot.draw()

def load_model(model_path):
    """Load the trained model."""
    try:
        info = torch.load(model_path, map_location=device)
        model = ModelClass(info["num_classes"]).to(device)
        model.load_state_dict(info["state_dict"])
        model.eval()  # Set to evaluation mode
        return model, info["num_classes"]
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def load_classes(classes_file):
    """Load class names from file."""
    try:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f]
        return classes
    except Exception as e:
        print(f"Error loading classes: {str(e)}")
        raise

def main():
    # Set model path - can be passed as command line argument or hardcoded
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find a model in the models directory
        models_dir = "models"
        if not os.path.exists(models_dir):
            print(f"Error: Models directory '{models_dir}' not found!")
            sys.exit(1)
            
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])
            print(f"Using model: {model_path}")
        else:
            model_path = "models/95_classes_recognized_0.9025_acc.pth"  # Default path
            if not os.path.exists(model_path):
                print(f"Error: No model file found!")
                sys.exit(1)
    
    # Set classes file path
    classes_file = "categories/95_classes.txt"  # Path to your classes file
    if not os.path.exists(classes_file):
        # Check if the categories directory exists
        categories_dir = "categories"
        if not os.path.exists(categories_dir):
            os.makedirs(categories_dir, exist_ok=True)
            print(f"Created directory: {categories_dir}")
            
        # Check for any text files that might contain class names
        txt_files = []
        for dir_path, _, file_names in os.walk('.'):
            txt_files.extend([os.path.join(dir_path, f) for f in file_names 
                              if f.endswith('.txt') and 'class' in f.lower()])
        
        if txt_files:
            classes_file = txt_files[0]
            print(f"Using classes file: {classes_file}")
        else:
            print(f"Error: Classes file '{classes_file}' not found!")
            sys.exit(1)
    
    try:
        # Load model and classes
        print("Loading model...")
        model, num_classes = load_model(model_path)
        
        print("Loading classes...")
        classes = load_classes(classes_file)
        
        # Make sure number of classes matches
        assert len(classes) == num_classes, f"Number of classes in model ({num_classes}) doesn't match classes file ({len(classes)})"
        
        print(f"Loaded model with {num_classes} classes")
        
        # Start the application
        root = tk.Tk()
        app = DrawingApp(root, model, classes)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()