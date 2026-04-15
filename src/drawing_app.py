"""
MNIST Digit Recognizer

A GUI application for training a neural network on MNIST data and then
drawing digits by hand to test recognition. Features:
- Draw digits with the mouse; auto-recognized on pen-up
- Confidence bar chart for all 10 digit classes
- Progress bar during training with per-epoch accuracy
- Save / load trained networks
- Browse random MNIST training samples
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import os
import numpy as np
import random
import threading

# Import the neural network modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import network
import mnist_loader

# -- Colour palette ---------------------------------------------------------
BG = "#f0f0f0"
CANVAS_BG = "#ffffff"
ACCENT = "#2563eb"        # blue-600
ACCENT_LIGHT = "#dbeafe"  # blue-100
BAR_BG = "#e5e7eb"        # gray-200
SUCCESS = "#16a34a"       # green-600
MUTED = "#6b7280"         # gray-500


class DrawingApp:
    """Main application class."""

    # -- init ----------------------------------------------------------------
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        self.root.geometry("720x520")
        self.root.minsize(680, 480)
        self.root.configure(bg=BG)

        # Drawing state
        self.old_x = None
        self.old_y = None
        self.brush_size = 14  # thicker default for digit-sized strokes
        self.canvas_width = 280
        self.canvas_height = 280

        # PIL mirror of the canvas (always in sync)
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.image_draw = ImageDraw.Draw(self.image)

        # Neural-network state
        self.net = None
        self.training_data = None
        self.test_data = None
        self.is_training = False

        # Auto-recognize timer id (so we can cancel/reset)
        self._recognize_after_id = None

        # ImageTk references (prevent GC)
        self._photo_refs = []

        self._build_ui()
        self._bind_events()

        # Try to auto-load a saved network on startup
        self._try_autoload_network()

    # -- UI construction -----------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))

        # ── top toolbar ────────────────────────────────────────────────────
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=(8, 4))

        ttk.Label(toolbar, text="Brush:").pack(side=tk.LEFT)
        self._size_var = tk.IntVar(value=self.brush_size)
        ttk.Scale(
            toolbar, from_=6, to=28, orient=tk.HORIZONTAL,
            variable=self._size_var, command=self._on_brush_size,
            length=90,
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Button(toolbar, text="Clear  (C)", command=self.clear_canvas).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._train_btn = ttk.Button(toolbar, text="Train Network", command=self.train_network)
        self._train_btn.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(toolbar, text="Save", command=self.save_network_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load", command=self.load_network_file).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(toolbar, text="Sample MNIST", command=self.show_random_training_image).pack(side=tk.LEFT, padx=4)

        # network-status indicator (right side)
        self._net_status_var = tk.StringVar(value="No network loaded")
        ttk.Label(toolbar, textvariable=self._net_status_var, foreground=MUTED,
                  font=("Segoe UI", 8)).pack(side=tk.RIGHT)

        # ── training progress bar (always packed, but hidden via height) ──
        self._progress_frame = ttk.Frame(self.root)
        self._progress_frame.pack(fill=tk.X, padx=10)
        self._progress_frame.pack_forget()  # start hidden
        self._progress_var = tk.DoubleVar()
        self._progress_bar = ttk.Progressbar(
            self._progress_frame, variable=self._progress_var,
            maximum=100, length=400,
        )
        self._progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self._progress_label = ttk.Label(self._progress_frame, text="", width=28)
        self._progress_label.pack(side=tk.LEFT, padx=(0, 10))

        # ── main content area ──────────────────────────────────────────────
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

        # -- left: drawing canvas
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(
            left, bg=CANVAS_BG,
            width=self.canvas_width, height=self.canvas_height,
            cursor="crosshair", highlightthickness=1,
            highlightbackground="#d1d5db",
        )
        self.canvas.pack()
        self._draw_canvas_hint()

        # -- right: results panel
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

        # Predicted digit – large display
        self._digit_var = tk.StringVar(value="–")
        digit_display = tk.Label(
            right, textvariable=self._digit_var,
            font=("Segoe UI", 64, "bold"), fg=ACCENT, bg=CANVAS_BG,
            width=3, relief=tk.FLAT, bd=0,
        )
        digit_display.pack(pady=(0, 2))

        self._confidence_var = tk.StringVar(value="Draw a digit to begin")
        ttk.Label(right, textvariable=self._confidence_var,
                  foreground=MUTED, font=("Segoe UI", 9)).pack()

        # Confidence bar chart
        chart_frame = ttk.LabelFrame(right, text="Class probabilities")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._bar_canvas = tk.Canvas(chart_frame, bg=CANVAS_BG,
                                     highlightthickness=0, height=180)
        self._bar_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._bar_canvas.bind("<Configure>", lambda e: self._draw_empty_bars())

        # Processed-image thumbnail (small, bottom-right)
        thumb_frame = ttk.Frame(right)
        thumb_frame.pack(anchor=tk.E, pady=(6, 0))
        ttk.Label(thumb_frame, text="28×28 input:", foreground=MUTED,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 4))
        self._thumb_canvas = tk.Canvas(
            thumb_frame, bg=CANVAS_BG, width=56, height=56,
            highlightthickness=1, highlightbackground="#d1d5db",
        )
        self._thumb_canvas.pack(side=tk.LEFT)

        # ── status bar ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Draw a digit and it will be recognized automatically.")
        ttk.Label(self.root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W,
                  font=("Segoe UI", 8)).pack(fill=tk.X, side=tk.BOTTOM)

    # -- canvas hint text (placeholder when empty)
    def _draw_canvas_hint(self):
        cx, cy = self.canvas_width // 2, self.canvas_height // 2
        self.canvas.create_text(
            cx, cy, text="Draw here",
            fill="#c0c0c0", font=("Segoe UI", 18), tags="hint",
        )

    # -- empty bar chart placeholder
    def _draw_empty_bars(self):
        self._draw_bars(np.zeros(10))

    # -- event bindings ------------------------------------------------------
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)
        self.root.bind_all("<KeyPress-c>", lambda e: self.clear_canvas())
        self.root.bind_all("<KeyPress-C>", lambda e: self.clear_canvas())

    # -- drawing -------------------------------------------------------------
    def _start_drawing(self, event):
        # Remove the hint text on first stroke
        self.canvas.delete("hint")
        self.old_x = event.x
        self.old_y = event.y
        # Cancel any pending auto-recognize
        if self._recognize_after_id:
            self.root.after_cancel(self._recognize_after_id)
            self._recognize_after_id = None

    def _draw(self, event):
        if self.old_x is not None and self.old_y is not None:
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.brush_size, fill="black",
                capstyle=tk.ROUND, smooth=tk.TRUE,
            )
            self.image_draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill="black", width=self.brush_size,
            )
        self.old_x = event.x
        self.old_y = event.y

    def _stop_drawing(self, event):
        self.old_x = None
        self.old_y = None
        # Schedule auto-recognize after a longer pause (1500 ms) so the user
        # has time to lift the pen between strokes (e.g. digits 4, 5, 8).
        if self._recognize_after_id:
            self.root.after_cancel(self._recognize_after_id)
        self._recognize_after_id = self.root.after(1500, self._auto_recognize)

    def _auto_recognize(self):
        """Silently run recognition if a network is loaded."""
        self._recognize_after_id = None
        if self.net is not None:
            self.recognize_digit()

    # -- brush size ----------------------------------------------------------
    def _on_brush_size(self, value):
        self.brush_size = int(float(value))

    # -- clear ---------------------------------------------------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.image_draw = ImageDraw.Draw(self.image)
        self._draw_canvas_hint()
        self._digit_var.set("–")
        self._confidence_var.set("Draw a digit to begin")
        self._draw_bars(np.zeros(10))
        self._thumb_canvas.delete("all")
        self._status_var.set("Canvas cleared.")

    # -- network status helpers ----------------------------------------------
    def _set_net_status(self, text, colour=MUTED):
        self._net_status_var.set(text)

    def _try_autoload_network(self):
        """Silently load trained_network.pkl if it exists next to the script."""
        try:
            self.net = network.Network.from_file("trained_network.pkl")
            self._set_net_status(f"Loaded: {self.net.sizes}")
            self._status_var.set("Network auto-loaded from trained_network.pkl. Draw a digit!")
        except Exception:
            pass

    # -- training ------------------------------------------------------------
    def train_network(self):
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress.")
            return

        if not messagebox.askyesno(
            "Train Network",
            "Train on MNIST data (30 epochs).\nThis may take a few minutes. Continue?",
        ):
            return

        self.is_training = True
        self._train_btn.configure(state=tk.DISABLED)
        self._progress_frame.pack(fill=tk.X, padx=10, after=self.root.winfo_children()[0])
        self._progress_var.set(0)
        self._progress_label.configure(text="Loading data…")

        t = threading.Thread(target=self._train_thread, daemon=True)
        t.start()

    def _train_thread(self):
        epochs = 30
        try:
            # Load data
            self.root.after(0, lambda: self._status_var.set("Loading MNIST data…"))
            try:
                tr, _val, te = mnist_loader.load_data_wrapper()
                self.training_data = list(tr)
                self.test_data = list(te)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load MNIST data:\n{e}"))
                return

            self.root.after(0, lambda: self._progress_label.configure(text="Initialising…"))
            self.net = network.Network([784, 100, 10])

            def _epoch_cb(epoch, total, accuracy):
                pct = (epoch + 1) / total * 100
                acc_str = f"{accuracy:.1%}" if accuracy is not None else "–"
                self.root.after(0, lambda: self._progress_var.set(pct))
                self.root.after(0, lambda: self._progress_label.configure(
                    text=f"Epoch {epoch + 1}/{total}  acc {acc_str}"))
                self.root.after(0, lambda: self._status_var.set(
                    f"Training… epoch {epoch + 1}/{total}  accuracy {acc_str}"))

            self.net.SGD(
                self.training_data, epochs=epochs, mini_batch_size=10,
                eta=3.0, test_data=self.test_data, epoch_callback=_epoch_cb,
            )

            # Auto-save
            try:
                self.net.save_network("trained_network.pkl")
            except Exception:
                pass

            self.root.after(0, lambda: self._set_net_status(f"Trained: {self.net.sizes}"))
            self.root.after(0, lambda: self._status_var.set(
                "Training complete! Network saved. Draw a digit to test."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.is_training = False
            self.root.after(0, lambda: self._train_btn.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self._progress_frame.pack_forget())

    # -- save / load ---------------------------------------------------------
    def save_network_file(self):
        if self.net is None:
            messagebox.showwarning("No Network", "Train or load a network first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Save Neural Network",
        )
        if path:
            self.net.save_network(path)
            self._status_var.set(f"Network saved to {os.path.basename(path)}")

    def load_network_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Load Neural Network",
        )
        if not path:
            return
        try:
            self.net = network.Network.from_file(path)
            self._set_net_status(f"Loaded: {self.net.sizes}")
            self._status_var.set(f"Loaded network from {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load network:\n{e}")

    # -- show training sample ------------------------------------------------
    def show_random_training_image(self):
        if self.training_data is None:
            try:
                self._status_var.set("Loading MNIST data…")
                tr, _v, te = mnist_loader.load_data_wrapper()
                self.training_data = list(tr)
                self.test_data = list(te)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MNIST data:\n{e}")
                return

        sample_img, sample_label = random.choice(self.training_data)
        digit = int(np.argmax(sample_label))

        # Show in the thumbnail panel
        self._show_thumbnail(sample_img)

        self._digit_var.set(str(digit))
        self._confidence_var.set("MNIST training sample")
        self._draw_bars(np.zeros(10))
        self._status_var.set(f"Showing random MNIST sample – label: {digit}")

    # -- recognition ---------------------------------------------------------
    def recognize_digit(self):
        if self.net is None:
            if not self._try_load_default():
                messagebox.showwarning("No Network", "Train or load a network first.")
                return

        img_vec = self._prepare_image()
        self._show_thumbnail(img_vec)

        output = self.net.feedforward(img_vec)
        probs = output.flatten()
        predicted = int(np.argmax(probs))
        conf = probs[predicted]

        self._digit_var.set(str(predicted))
        self._confidence_var.set(f"Confidence: {conf:.1%}")
        self._draw_bars(probs)
        self._status_var.set(f"Predicted digit: {predicted}  ({conf:.1%})")

    def _try_load_default(self):
        try:
            self.net = network.Network.from_file("trained_network.pkl")
            self._set_net_status(f"Loaded: {self.net.sizes}")
            return True
        except Exception:
            return False

    # -- image preprocessing -------------------------------------------------
    def _prepare_image(self):
        canvas_img = self.image.copy().convert("L")
        bbox = canvas_img.getbbox()
        if bbox is None:
            return np.zeros((784, 1), dtype=np.float32)

        # Crop with 20 % padding
        l, t, r, b = bbox
        pad = max(r - l, b - t) * 0.2
        l = max(0, l - pad)
        t = max(0, t - pad)
        r = min(canvas_img.width, r + pad)
        b = min(canvas_img.height, b + pad)
        cropped = canvas_img.crop((int(l), int(t), int(r), int(b)))

        # Make square
        cw, ch = cropped.size
        mx = max(cw, ch)
        square = Image.new("L", (mx, mx), 255)
        square.paste(cropped, ((mx - cw) // 2, (mx - ch) // 2))

        # Resize to 28×28
        resized = square.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(resized, dtype=np.float32)

        # Invert (MNIST = white-on-black: 0 = background, 255 = stroke)
        arr = 255.0 - arr

        # Thicken strokes slightly to match MNIST-like appearance.
        # Use a cross-shaped kernel (4-connected, not 8) for moderate dilation.
        try:
            from scipy.ndimage import grey_dilation
            cross = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
            arr = grey_dilation(arr, footprint=cross)
        except ImportError:
            # Manual cross-shaped max-pool dilation (up/down/left/right only)
            padded = np.pad(arr, 1, mode="constant", constant_values=0)
            arr = np.maximum.reduce([
                padded[1:29, 1:29],   # centre
                padded[0:28, 1:29],   # up
                padded[2:30, 1:29],   # down
                padded[1:29, 0:28],   # left
                padded[1:29, 2:30],   # right
            ])

        # Normalise to 0-1
        hi = arr.max()
        if hi > 0:
            arr = arr / hi

        return arr.reshape(784, 1)

    # -- bar-chart drawing ---------------------------------------------------
    def _draw_bars(self, probs):
        c = self._bar_canvas
        c.delete("all")
        w = c.winfo_width() or 340
        h = c.winfo_height() or 180
        if w < 10 or h < 10:
            return

        n = len(probs)
        margin_l, margin_r, margin_t, margin_b = 24, 8, 8, 20
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        bar_w = chart_w / n * 0.7
        gap = chart_w / n * 0.3

        best = int(np.argmax(probs))
        for i, p in enumerate(probs):
            x0 = margin_l + i * (bar_w + gap)
            bar_h = max(1, p * chart_h)
            y0 = margin_t + chart_h - bar_h
            y1 = margin_t + chart_h
            fill = ACCENT if i == best and p > 0 else BAR_BG
            c.create_rectangle(x0, y0, x0 + bar_w, y1, fill=fill, outline="")
            # label
            c.create_text(x0 + bar_w / 2, y1 + 10, text=str(i),
                          font=("Segoe UI", 8), fill="#374151")
            if p > 0.01:
                c.create_text(x0 + bar_w / 2, y0 - 6, text=f"{p:.0%}",
                              font=("Segoe UI", 7), fill="#374151")

    # -- thumbnail display ---------------------------------------------------
    def _show_thumbnail(self, img_vec):
        img_2d = (img_vec.reshape(28, 28) * 255).astype(np.uint8)
        pil = Image.fromarray(img_2d, mode="L").resize((56, 56), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(pil)
        self._thumb_canvas.delete("all")
        self._thumb_canvas.create_image(28, 28, image=photo)
        self._photo_refs = [photo]

    # -- saved-network helper ------------------------------------------------
    def save_trained_network(self):
        if self.net is None:
            return
        try:
            self.net.save_network("trained_network.pkl")
        except Exception as e:
            print(f"Failed to save network: {e}")


def main():
    root = tk.Tk()
    app = DrawingApp(root)
    # Centre on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    root.mainloop()


if __name__ == "__main__":
    main()
