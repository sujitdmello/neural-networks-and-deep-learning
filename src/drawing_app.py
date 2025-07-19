"""
Drawing Canvas Application

A simple Python application that allows users to draw on a canvas using the mouse.
Features include:
- Mouse drawing with customizable brush size and color
- Clear canvas functionality
- Save drawing as PNG image
- Color picker
- Brush size adjustment
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from PIL import Image, ImageDraw
import os
import numpy as np
import pickle
import threading

# Import the neural network modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import network
import mnist_loader


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing Canvas")
        self.root.geometry("800x500")  # Increased width to accommodate side panel
        
        # Drawing variables
        self.old_x = None
        self.old_y = None
        self.brush_size = 5
        self.brush_color = "black"
        self.canvas_width = 280  # Made it 280 for better 28x28 scaling
        self.canvas_height = 280
        
        # Create PIL Image for saving
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.image_draw = ImageDraw.Draw(self.image)
        
        # Neural network variables
        self.net = None
        self.training_data = None
        self.test_data = None
        self.is_training = False
        
        # Store photo references to prevent garbage collection
        self.photo_refs = []
        
        self.setup_ui()
        self.bind_events()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel - First row
        control_frame1 = ttk.Frame(main_frame)
        control_frame1.pack(fill=tk.X, pady=(0, 5))
        
        # Brush size control
        ttk.Label(control_frame1, text="Brush Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.size_var = tk.IntVar(value=self.brush_size)
        size_scale = ttk.Scale(
            control_frame1, 
            from_=1, 
            to=20, 
            orient=tk.HORIZONTAL, 
            variable=self.size_var,
            command=self.update_brush_size,
            length=100
        )
        size_scale.pack(side=tk.LEFT, padx=(0, 20))
        
        # Color button
        self.color_button = tk.Button(
            control_frame1,
            text="Choose Color",
            bg=self.brush_color,
            command=self.choose_color,
            width=12
        )
        self.color_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(
            control_frame1,
            text="Clear Canvas",
            command=self.clear_canvas
        )
        clear_button.pack(side=tk.LEFT)
        
        # Control panel - Second row
        control_frame2 = ttk.Frame(main_frame)
        control_frame2.pack(fill=tk.X, pady=(0, 10))
        
        # Neural Network buttons
        train_button = ttk.Button(
            control_frame2,
            text="Train Network",
            command=self.train_network
        )
        train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        recognize_button = ttk.Button(
            control_frame2,
            text="Recognize Digit",
            command=self.recognize_digit
        )
        recognize_button.pack(side=tk.LEFT, padx=(0, 20))
        
        # Save/Load Network buttons
        save_net_button = ttk.Button(
            control_frame2,
            text="Save Network",
            command=self.save_network_file
        )
        save_net_button.pack(side=tk.LEFT, padx=(0, 10))
        
        load_net_button = ttk.Button(
            control_frame2,
            text="Load Network",
            command=self.load_network_file
        )
        load_net_button.pack(side=tk.LEFT)
        
        # Show training data button
        show_data_button = ttk.Button(
            control_frame2,
            text="Show Training Data",
            command=self.show_random_training_image
        )
        show_data_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side frame for the drawing canvas
        left_frame = ttk.Frame(canvas_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(
            left_frame,
            bg="white",
            width=self.canvas_width,
            height=self.canvas_height,
            cursor="crosshair"
        )
        
        # Draw grid lines to guide digit placement
        self.draw_grid_lines()
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Pack scrollbars and canvas
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Right side frame for processed image display
        right_frame = ttk.Frame(canvas_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Processed image label
        processed_label = ttk.Label(right_frame, text="Processed Image (28x28):", font=("Arial", 10, "bold"))
        processed_label.pack(pady=(0, 5))
        
        # Canvas for processed image
        self.processed_canvas = tk.Canvas(
            right_frame,
            bg="white",
            width=140,  # 28x28 scaled up by 5x
            height=140,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.processed_canvas.pack()
        
        # Description label
        self.processed_desc = ttk.Label(
            right_frame, 
            text="Gently enhanced for\nbetter neural network input", 
            font=("Arial", 8),
            foreground="gray",
            justify=tk.CENTER
        )
        self.processed_desc.pack(pady=(5, 0))
        
        # Recognition result section
        result_label = ttk.Label(right_frame, text="Recognized Digit:", font=("Arial", 10, "bold"))
        result_label.pack(pady=(15, 5))
        
        # Text box for recognized digit display
        self.result_var = tk.StringVar()
        self.result_var.set("Draw and recognize")
        result_display = ttk.Label(
            right_frame,
            textvariable=self.result_var,
            font=("Arial", 14, "bold"),
            foreground="blue",
            background="white",
            relief=tk.SUNKEN,
            borderwidth=2,
            padding=10,
            justify=tk.CENTER
        )
        result_display.pack(pady=(0, 10), fill=tk.X)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Draw a digit (0-9) on the canvas, then click 'Recognize Digit'")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Instructions
        instructions = ttk.Label(
            main_frame, 
            text="Instructions: Draw clearly in the center of the canvas. Train the network first for better accuracy.",
            font=("Arial", 9),
            foreground="gray"
        )
        instructions.pack(fill=tk.X, pady=(5, 0))
    
    def bind_events(self):
        """Bind mouse events to canvas"""
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
    
    def draw_grid_lines(self):
        """Draw grid lines on the canvas to guide digit placement"""
        # Grid spacing - divide canvas into sections
        grid_spacing = 35  # 280/8 = 35, creating an 8x8 grid
        
        # Draw vertical lines
        for x in range(grid_spacing, self.canvas_width, grid_spacing):
            self.canvas.create_line(
                x, 0, x, self.canvas_height,
                fill="lightgray",
                width=1,
                tags="grid"
            )
        
        # Draw horizontal lines
        for y in range(grid_spacing, self.canvas_height, grid_spacing):
            self.canvas.create_line(
                0, y, self.canvas_width, y,
                fill="lightgray",
                width=1,
                tags="grid"
            )
        
        # Draw a center cross to help with digit centering
        center_x = self.canvas_width // 2
        center_y = self.canvas_height // 2
        
        # Vertical center line
        self.canvas.create_line(
            center_x, 0, center_x, self.canvas_height,
            fill="lightblue",
            width=2,
            tags="grid"
        )
        
        # Horizontal center line
        self.canvas.create_line(
            0, center_y, self.canvas_width, center_y,
            fill="lightblue",
            width=2,
            tags="grid"
        )
        
        # Draw a suggested digit area (center 180x180 area)
        margin = 50  # (280-180)/2 = 50, increased from 70 for larger drawing area
        self.canvas.create_rectangle(
            margin, margin, 
            self.canvas_width - margin, self.canvas_height - margin,
            outline="lightcoral",
            width=2,
            tags="grid"
        )
        
        # Add text instructions
        self.canvas.create_text(
            center_x, 25,
            text="Draw digit in the center area",
            fill="gray",
            font=("Arial", 10),
            tags="grid"
        )
        
    def start_drawing(self, event):
        """Start drawing when mouse is pressed"""
        self.old_x = event.x
        self.old_y = event.y
        
    def draw(self, event):
        """Draw while mouse is being dragged"""
        if self.old_x and self.old_y:
            # Draw on tkinter canvas
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.brush_size,
                fill=self.brush_color,
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
            
            # Draw on PIL image for saving
            self.image_draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill=self.brush_color,
                width=self.brush_size
            )
            
        self.old_x = event.x
        self.old_y = event.y
        
        # Update status
        self.status_var.set(f"Drawing at ({event.x}, {event.y})")
    
    def stop_drawing(self, event):
        """Stop drawing when mouse is released"""
        self.old_x = None
        self.old_y = None
        self.status_var.set("Ready to draw!")
    
    def update_brush_size(self, value):
        """Update brush size from scale widget"""
        self.brush_size = int(float(value))
        self.status_var.set(f"Brush size: {self.brush_size}")
    
    def choose_color(self):
        """Open color chooser dialog"""
        color = colorchooser.askcolor(color=self.brush_color)[1]
        if color:
            self.brush_color = color
            self.color_button.config(bg=color)
            self.status_var.set(f"Color changed to {color}")
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        # Reset PIL image
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.image_draw = ImageDraw.Draw(self.image)
        # Redraw grid lines
        self.draw_grid_lines()
        # Reset result display
        self.result_var.set("Draw and recognize")
        # Clear processed image canvas
        self.processed_canvas.delete("all")
        self.status_var.set("Canvas cleared!")
    
    def train_network(self):
        """Train the neural network with MNIST data"""
        if self.is_training:
            messagebox.showwarning("Training", "Network is already training!")
            return
            
        result = messagebox.askyesno(
            "Train Network", 
            "This will train the neural network with MNIST data.\n"
            "Training may take several minutes. Continue?"
        )
        
        if not result:
            return
            
        # Start training in a separate thread to avoid blocking the UI
        training_thread = threading.Thread(target=self._train_network_thread)
        training_thread.daemon = True
        training_thread.start()
    
    def _train_network_thread(self):
        """Training thread to avoid blocking the UI"""
        try:
            self.is_training = True
            self.status_var.set("Loading MNIST data...")
            
            # Load MNIST data
            try:
                training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
                self.training_data = list(training_data)
                self.test_data = list(test_data)
            except Exception as load_error:
                error_msg = f"Failed to load MNIST data:\n{str(load_error)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
                return
            
            self.status_var.set("Initializing neural network...")
            
            # Create neural network (784 inputs, 30 hidden, 10 outputs)
            self.net = network.Network([784, 100, 10])
            
            self.status_var.set("Training neural network... This may take a few minutes.")
            
            # Train the network
            self.net.SGD(
                self.training_data, 
                epochs=30, 
                mini_batch_size=10, 
                eta=3.0,
                test_data=self.test_data
            )
            
            self.status_var.set("Training completed! Network is ready for digit recognition.")
            
            # Save the trained network
            self.save_trained_network()
            
        except Exception as train_error:
            error_msg = f"Error during training:\n{str(train_error)}"
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Training Error", msg))
        finally:
            self.is_training = False
    
    def save_trained_network(self):
        """Save the trained network to a file"""
        if self.net is None:
            return
        try:
            self.net.save_network('trained_network.pkl')
            self.status_var.set("Training completed and network saved!")
        except Exception as e:
            print(f"Failed to save network: {e}")
    
    def load_trained_network(self):
        """Load a previously trained network"""
        try:
            self.net = network.Network.from_file('trained_network.pkl')
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Failed to load network: {e}")
            return False
    
    def save_network_file(self):
        """Save the trained network to a user-selected file"""
        if self.net is None:
            messagebox.showwarning("No Network", "No trained network to save!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                title="Save Neural Network"
            )
            
            if file_path:
                self.net.save_network(file_path)
                self.status_var.set(f"Network saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Network saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save network:\n{str(e)}")
    
    def load_network_file(self):
        """Load a trained network from a user-selected file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                title="Load Neural Network"
            )
            
            if file_path:
                try:
                    # Load the network using the static method
                    self.net = network.Network.from_file(file_path)
                    
                    self.status_var.set(f"Network loaded from {os.path.basename(file_path)}")
                    messagebox.showinfo("Success", 
                                      f"Network loaded successfully!\n"
                                      f"Architecture: {self.net.sizes}")
                        
                except Exception as load_error:
                    messagebox.showerror("Error", f"Failed to load network:\n{str(load_error)}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading network:\n{str(e)}")

    def show_random_training_image(self):
        """Show a random image from the training dataset"""
        # Check if training data is loaded
        if self.training_data is None:
            try:
                # Try to load MNIST data
                training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
                self.training_data = list(training_data)
                self.test_data = list(test_data)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MNIST data:\n{str(e)}")
                return
        
        if not self.training_data:
            messagebox.showwarning("No Data", "No training data available!")
            return
        
        try:
            # Get a random training sample
            import random
            random_sample = random.choice(self.training_data)
            image_data, label = random_sample
            
            # Convert the image data to display format
            # MNIST data comes as a 784x1 vector, reshape to 28x28
            img_2d = image_data.reshape(28, 28)
            
            # Convert to 0-255 range for display (MNIST data is already 0-1 normalized)
            img_display = (img_2d * 255).astype(np.uint8)
            
            # Create PIL image
            pil_img = Image.fromarray(img_display, mode='L')
            
            # Scale up for better visibility (28x28 -> 140x140, 5x scaling)
            pil_img = pil_img.resize((140, 140), Image.Resampling.NEAREST)
            
            # Clear the processed canvas
            self.processed_canvas.delete("all")
            
            # Convert PIL image to tkinter PhotoImage and display
            try:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(pil_img)
                self.processed_canvas.create_image(70, 70, image=photo)
                # Store reference to prevent garbage collection
                self.photo_refs = [photo]  # Clear previous and store new reference
            except ImportError:
                # Fallback: show text description if PIL ImageTk not available
                self.processed_canvas.create_text(70, 70, text="Training\nimage\n(PIL ImageTk\nnot available)", 
                                                justify=tk.CENTER, font=("Arial", 8))
            
            # Show the actual label in the result display
            actual_digit = np.argmax(label)
            self.result_var.set(f"Training: {actual_digit}\n(Actual label)")
            
            # Update status bar
            self.status_var.set(f"Showing random training image - actual digit: {actual_digit}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying training image:\n{str(e)}")

    def recognize_digit(self):
        """Recognize the digit drawn on the canvas"""
        if self.net is None:
            # Try to load a previously trained network
            if not self.load_trained_network():
                messagebox.showwarning(
                    "No Network", 
                    "No trained network found. Please train the network first."
                )
                return
        
        try:
            # Convert canvas drawing to 28x28 grayscale image
            digit_image = self.prepare_canvas_for_recognition()
            
            # Show the processed image in the side panel
            self.update_processed_image_display(digit_image)
            
            # Use the network to predict the digit
            output = self.net.feedforward(digit_image)
            predicted_digit = np.argmax(output)
            confidence = output[predicted_digit][0]
            
            # Show result in the text box below processed image
            self.result_var.set(f"{predicted_digit}\n({confidence:.1%})")
            
            # Update status bar with general message
            self.status_var.set("Digit recognition completed!")
            
        except Exception as e:
            messagebox.showerror("Recognition Error", f"Error during recognition:\n{str(e)}")
    
    def prepare_canvas_for_recognition(self):
        """Convert the canvas drawing to a format suitable for the neural network"""
        # Get the canvas as a PIL image
        canvas_image = self.image.copy()
        
        # Convert to grayscale first
        canvas_image = canvas_image.convert('L')
        
        # Find the bounding box of the drawn content to center it
        bbox = canvas_image.getbbox()
        
        if bbox is None:
            # If nothing is drawn, return a blank 28x28 image
            img_array = np.zeros((784, 1), dtype=np.float32)
            return img_array
        
        # Crop to the bounding box with some padding
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        
        # Add padding (20% of the larger dimension)
        padding = max(width, height) * 0.2
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(canvas_image.width, right + padding)
        bottom = min(canvas_image.height, bottom + padding)
        
        # Crop the image to the padded bounding box
        cropped_image = canvas_image.crop((int(left), int(top), int(right), int(bottom)))
        
        # Create a square image by padding with white
        crop_width, crop_height = cropped_image.size
        max_dim = max(crop_width, crop_height)
        
        # Create a new square white image
        square_image = Image.new('L', (max_dim, max_dim), 255)  # 255 = white
        
        # Paste the cropped image in the center
        paste_x = (max_dim - crop_width) // 2
        paste_y = (max_dim - crop_height) // 2
        square_image.paste(cropped_image, (paste_x, paste_y))
        
        # Resize to 28x28 with high-quality anti-aliasing
        # Using LANCZOS (high-quality) resampling for smooth anti-aliasing
        resized_image = square_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(resized_image, dtype=np.float32)
        
        # Apply gentle contrast enhancement (less aggressive than before)
        # Apply contrast stretching to use full 0-255 range
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        if max_val > min_val:  # Avoid division by zero
            img_array = (img_array - min_val) * 255.0 / (max_val - min_val)
        
        # Apply morphological dilation to thicken strokes (like MNIST)
        try:
            from scipy import ndimage
            # Create a structuring element (small cross pattern for thickening)
            structure = np.array([[0, 1, 0],
                                 [1, 1, 1], 
                                 [0, 1, 0]], dtype=bool)
            
            # Apply dilation before thresholding to thicken the strokes
            # First, create a binary mask of the drawn areas
            binary_mask = img_array < 200  # Areas that are not pure white
            dilated_mask = ndimage.binary_dilation(binary_mask, structure=structure, iterations=1)
            
            # Apply the dilation effect: make dilated areas darker
            img_array = np.where(dilated_mask, img_array * 0.6, img_array)
            
            # Apply gentler threshold after dilation
            threshold = 200  # Adjusted threshold after dilation
            img_array = np.where(img_array < threshold, 
                               img_array * 0.8,  # Less aggressive darkening after dilation
                               255.0)             # Keep light areas white
        except ImportError:
            # Fallback: use more aggressive thresholding to simulate thickening
            threshold = 160  # Lower threshold to capture more pixels as "digit"
            img_array = np.where(img_array < threshold, 
                               img_array * 0.5,  # More aggressive darkening without dilation
                               255.0)             # Keep light areas white
        
        # Apply a softer binary threshold to reduce extreme processing
        # This is gentler than the previous approach
        # binary_threshold = 160  # Increased from 128 to be less harsh
        # img_array = np.where(img_array < binary_threshold, 
        #                    img_array * 0.5,  # Darken but not to pure black
        #                    255.0)
        
        # Invert colors (MNIST has white digits on black background)
        # Our canvas has black digits on white background
        img_array = 255.0 - img_array
        
        # Normalize to 0-1 range (0 = black background, 1 = white digit)
        img_array = img_array / 255.0
        
        # Ensure we have the exact 784-dimensional vector (28*28 = 784)
        assert img_array.shape == (28, 28), f"Expected (28, 28), got {img_array.shape}"
        
        # Reshape to column vector (784, 1) as expected by the network
        img_array = img_array.reshape(784, 1)
        
        # Verify we have exactly 784 dimensions
        assert img_array.shape == (784, 1), f"Expected (784, 1), got {img_array.shape}"
        
        return img_array
    
    def update_processed_image_display(self, img_array):
        """Update the processed image display in the side panel"""
        # Convert back to 28x28 for visualization
        img_2d = img_array.reshape(28, 28)
        
        # Convert to 0-255 range for display
        img_display = (img_2d * 255).astype(np.uint8)
        
        # Create PIL image
        pil_img = Image.fromarray(img_display, mode='L')
        
        # Scale up for better visibility (28x28 -> 140x140, 5x scaling)
        pil_img = pil_img.resize((140, 140), Image.Resampling.NEAREST)
        
        # Clear the processed canvas
        self.processed_canvas.delete("all")
        
        # Convert PIL image to tkinter PhotoImage and display
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(pil_img)
            self.processed_canvas.create_image(70, 70, image=photo)
            # Store reference to prevent garbage collection
            if not hasattr(self, 'photo_refs'):
                self.photo_refs = []
            self.photo_refs = [photo]  # Clear previous and store new reference
        except ImportError:
            # Fallback: show text description if PIL ImageTk not available
            self.processed_canvas.create_text(70, 70, text="Processed\n28x28 image\n(PIL ImageTk\nnot available)", 
                                            justify=tk.CENTER, font=("Arial", 8))

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DrawingApp(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
