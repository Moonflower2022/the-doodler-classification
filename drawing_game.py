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
import time
import platform

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
        self.root.geometry("1100x700")  # Increased window size
        self.root.config(bg="#f0f0f0")
        
        # Create main frames
        self.left_frame = tk.Frame(root, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for drawing - MAKE IT SQUARE
        self.canvas_frame = tk.Frame(self.left_frame, bg="white", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Increased square canvas size
        self.canvas_size = 600  # Increased from 400 to 600
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg="white", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        
        # Initialize drawing variables
        self.setup_drawing()
        
        # Buttons frame
        self.button_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Clear button
        self.clear_button = ttk.Button(self.button_frame, text="Clear (c)", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Undo button
        self.undo_button = ttk.Button(self.button_frame, text="Undo (z)", command=self.undo_last_stroke)
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Classify button
        self.classify_button = ttk.Button(self.button_frame, text="Classify (space)", command=self.classify_drawing)
        self.classify_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Auto-classify toggle
        self.auto_classify_var = tk.BooleanVar(value=True)  # Enable by default
        self.auto_classify_check = ttk.Checkbutton(
            self.button_frame, 
            text="Auto-classify", 
            variable=self.auto_classify_var,
            command=self.toggle_auto_classify
        )
        self.auto_classify_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Line width options
        self.line_width_label = tk.Label(self.button_frame, text="Line Width:", bg="#f0f0f0")
        self.line_width_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.line_width_var = tk.IntVar(value=5)  # Increased default line width
        self.line_width_slider = ttk.Scale(self.button_frame, from_=1, to=15,  # Increased max width
                                          variable=self.line_width_var, orient="horizontal",
                                          length=100, command=self.update_line_width)
        self.line_width_slider.pack(side=tk.LEFT, padx=5)
        
        # Results area
        self.results_label = tk.Label(self.right_frame, text="Drawing Classification Results", 
                                     font=("Arial", 16), bg="#f0f0f0")
        self.results_label.pack(pady=10)
        
        # Create figure for bar chart
        self.fig, self.ax = plt.subplots(figsize=(5, 7))  # Increased figure size
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Preview of processed image
        self.preview_label = tk.Label(self.right_frame, text="Processed Input:", bg="#f0f0f0")
        self.preview_label.pack(pady=(20, 5))
        
        self.preview_canvas = tk.Canvas(self.right_frame, width=140, height=140, bg="white", bd=1, relief=tk.SUNKEN)
        self.preview_canvas.pack(pady=5)
        
        # Stroke history for undo functionality
        self.stroke_history = []
        self.current_stroke = []
        
        # Auto-classification variables
        self.auto_classify_enabled = True
        self.last_classify_time = 0
        self.classify_interval = 500  # 500ms = 0.5s
        self.pending_classification = False
        
        # Set up keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
        # Start auto-classification timer
        self.schedule_auto_classification()
        
    def setup_keyboard_shortcuts(self):        
        # Bind keyboard shortcuts
        self.root.bind("<space>", lambda event: self.classify_drawing())  # space for classify
        self.root.bind(f"<z>", lambda event: self.undo_last_stroke())  # z for undo
        self.root.bind(f"<c>", lambda event: self.clear_canvas())  # c for clear
        
        # Keep focus on the main window for keyboard events
        self.root.focus_set()
        
    def update_line_width(self, event=None):
        self.line_width = self.line_width_var.get()
        
    def setup_drawing(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 5  # Increased default line width
        # Create new drawing image based on canvas size
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.draw_image)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # We're not tracking mouse movement for activity anymore
        # Only drawing actions will reset the inactivity timer
        
    def start_draw(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.current_stroke = []  # Start a new stroke
        
        # Draw a single dot if just clicked (not dragged)
        # Create a small circle on the canvas
        dot_radius = self.line_width / 2
        x, y = event.x, event.y
        
        oval_id = self.canvas.create_oval(
            x - dot_radius, y - dot_radius, 
            x + dot_radius, y + dot_radius, 
            fill="black", outline="black",
            tags="drawing"
        )
        
        # Store the dot in the current stroke
        self.current_stroke.append({
            'line_id': oval_id,
            'points': (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            'width': self.line_width,
            'type': 'dot'
        })
        
        # Draw on the PIL image too
        self.draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=0)
        
        # Flag for auto-classification
        self.pending_classification = True
        
    def draw_line(self, event):        
        if self.old_x and self.old_y:
            # Create line on the canvas
            line_id = self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y, 
                width=self.line_width, fill="black",
                capstyle=tk.ROUND, smooth=tk.TRUE,
                tags="drawing"
            )
            
            # Store the line in the current stroke
            self.current_stroke.append({
                'line_id': line_id,
                'points': (self.old_x, self.old_y, event.x, event.y),
                'width': self.line_width,
                'type': 'line'
            })
            
            # Draw on the PIL image too
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                          fill=0, width=self.line_width)  # 0 is black in "L" mode
            
            # Flag for auto-classification
            self.pending_classification = True
            
        self.old_x = event.x
        self.old_y = event.y
        
    def end_draw(self, event):
        if self.current_stroke:  # Only add if not empty
            # Add current stroke to history
            self.stroke_history.append(self.current_stroke)
            self.current_stroke = []  # Reset current stroke
            
        self.old_x = None
        self.old_y = None
        
        # Force an immediate classification when the user stops drawing
        if self.auto_classify_enabled:
            self.classify_drawing()
            
        # Make sure the main window has focus for keyboard shortcuts
        self.root.focus_set()
        
    def undo_last_stroke(self):
        if self.stroke_history:
            # Get the last stroke
            last_stroke = self.stroke_history.pop()
            
            # Remove all lines in the stroke from the canvas
            for element in last_stroke:
                self.canvas.delete(element['line_id'])
            
            # Redraw the image from scratch (not efficient but simple)
            self.redraw_from_history()
            
            # Update status
            self.status_var.set("Last stroke undone")
            
            # Trigger classification after undo
            self.pending_classification = True
    
    def redraw_from_history(self):
        # Clear the PIL image
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.draw_image)
        
        # Redraw all strokes from history
        for stroke in self.stroke_history:
            for element in stroke:
                element_type = element.get('type', 'line')  # Default to line for backward compatibility
                
                if element_type == 'line':
                    x0, y0, x1, y1 = element['points']
                    width = element['width']
                    self.draw.line([x0, y0, x1, y1], fill=0, width=width)
                elif element_type == 'dot':
                    x0, y0, x1, y1 = element['points']
                    self.draw.ellipse([x0, y0, x1, y1], fill=0)
                
    def clear_canvas(self):
        self.canvas.delete("drawing")
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.draw_image)
        self.ax.clear()
        self.canvas_plot.draw()
        self.preview_canvas.delete("all")
        self.stroke_history = []  # Clear the stroke history
        self.status_var.set("Canvas cleared")
        
        # Show the prompt text
        self.prompt_label.config(text="Feel free to draw above!")
        self.inactivity_prompt_visible = True
        
    def toggle_auto_classify(self):
        self.auto_classify_enabled = self.auto_classify_var.get()
        status = "enabled" if self.auto_classify_enabled else "disabled"
        self.status_var.set(f"Auto-classification {status}")
        
    def schedule_auto_classification(self):
        """Schedule the next auto-classification check"""
        if self.auto_classify_enabled and self.pending_classification:
            current_time = int(time.time() * 1000)  # Current time in milliseconds
            time_since_last = current_time - self.last_classify_time
            
            if time_since_last >= self.classify_interval:
                # Time to classify
                self.classify_drawing()
                self.pending_classification = False
                self.last_classify_time = current_time
                
        # Schedule the next check
        self.root.after(100, self.schedule_auto_classification)  # Check every 100ms
        
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
        # Only classify if there's something on the canvas
        if np.array(self.draw_image).mean() == 255:  # If image is all white
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
        
        # Update status with top prediction
        top_class = classes[0].capitalize()
        top_prob = probabilities[0] * 100
        self.status_var.set(f"Top prediction: {top_class} ({top_prob:.1f}%)")

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