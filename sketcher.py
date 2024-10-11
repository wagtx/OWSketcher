# OWSketcher: An image processing tool that converts images into sketch-like drawings
# This script detects outlines and details in images and animates the drawing process.

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
from matplotlib.animation import FuncAnimation  # For creating animations
from scipy.spatial import distance  # For spatial computations (unused in current version)
import tkinter as tk  # Tkinter for GUI
from tkinter import filedialog, ttk  # Additional Tkinter modules
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding Matplotlib in Tkinter

# Animation settings
ANIMATION_DURATION = 10  # Total duration of the animation in seconds
TARGET_FRAMES = 100  # Target number of frames for the animation
OUTLINE_RATIO = 0.1  # Ratio of frames dedicated to outlines

STANDARD_WIDTH = 2048  # Standard width for processing

# Line detection parameters
MIN_LINE_LENGTH = 20  # Minimum length of line to be detected
MAX_LINE_GAP = 10  # Maximum gap between line segments
MAX_LINES = 15000  # Maximum number of lines to process

# Line length thresholds
MIN_OUTLINE_LENGTH = 50  # Minimum length for outlines
MIN_DETAIL_LENGTH = 30  # Minimum length for details

# Shading parameters
MIN_SHADING_THRESHOLD = 40  # Maximum length of line to apply shading
MIN_SHADING_DENSITY = 8  # Controls density of shading lines
SHADING_LINE_LENGTH = 8  # Length of individual shading lines

# Grid parameters
GRID_ROWS = 16  # Number of rows in the grid
GRID_COLS = 16  # Number of columns in the grid

def resize_image(image, width=STANDARD_WIDTH):
    """Resize the input image to a standard width while maintaining aspect ratio."""
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def detect_outlines(image):
    """Detect outlines in the input image using edge detection techniques."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated

def detect_details(image):
    """Detect fine details in the input image using edge detection techniques."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges

def get_lines_with_intensity(edges, gray, max_lines, min_line_length, max_line_gap):
    """Detect lines in the edge image and calculate their intensities."""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return np.array([])
    
    lines_with_intensity = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        intensity = np.mean(gray[min(y1,y2):max(y1,y2)+1, min(x1,x2):max(x1,x2)+1])
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        lines_with_intensity.append([x1, y1, x2, y2, intensity, length])
    
    lines_with_intensity = np.array(lines_with_intensity)
    if len(lines_with_intensity) > max_lines:
        indices = np.random.choice(len(lines_with_intensity), max_lines, replace=False)
        lines_with_intensity = lines_with_intensity[indices]
    
    return lines_with_intensity

def order_lines(lines, mode='length', length_threshold=50):
    """Order the detected lines based on different criteria."""
    if mode == 'length':
        long_lines = lines[lines[:, 5] >= length_threshold]
        short_lines = lines[lines[:, 5] < length_threshold]
        long_lines = long_lines[long_lines[:, 5].argsort()[::-1]]
        short_lines = short_lines[short_lines[:, 5].argsort()]
        return np.concatenate((long_lines, short_lines))
    elif mode == 'vertical':
        return lines[lines[:, 1].argsort()]  # Sort by y1 coordinate
    elif mode == 'horizontal':
        return lines[lines[:, 0].argsort()]  # Sort by x1 coordinate
    elif mode == 'grid':
        # Divide the image into a grid and sort lines within each cell
        height, width = np.max(lines[:, [1, 3]]), np.max(lines[:, [0, 2]])
        grid_h, grid_w = height // GRID_ROWS, width // GRID_COLS
        cells = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                cell_lines = lines[(lines[:, 0] >= j*grid_w) & (lines[:, 0] < (j+1)*grid_w) &
                                   (lines[:, 1] >= i*grid_h) & (lines[:, 1] < (i+1)*grid_h)]
                cells.append(cell_lines)
        return np.concatenate(cells)

def draw_line_with_shading(img, x1, y1, x2, y2, intensity, length, is_outline, shading_threshold, shading_density):
    """Draw a line on the image with optional shading effect."""
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), int(255 - intensity), 1)
    
    if not is_outline and length < shading_threshold:
        num_shade_lines = max(1, int(length / shading_density))
        for _ in range(num_shade_lines):
            t = np.random.random()
            shade_x = int(x1 + t * (x2 - x1))
            shade_y = int(y1 + t * (y2 - y1))
            angle = np.arctan2(y2 - y1, x2 - x1) + np.pi/2
            end_x = int(shade_x + SHADING_LINE_LENGTH * np.cos(angle))
            end_y = int(shade_y + SHADING_LINE_LENGTH * np.sin(angle))
            cv2.line(img, (shade_x, shade_y), (end_x, end_y), int(255 - intensity * 0.5), 1)

def main():
    """Main function to run the OWSketcher application."""
    global sketch_image, original_height, original_width

    root = tk.Tk()
    root.title("OWSketcher")
    root.geometry("800x600")

    def choose_file():
        """Open a file dialog to choose an image file and process it."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            process_image(file_path)

    choose_button = ttk.Button(root, text="Choose File", command=choose_file)
    choose_button.pack(pady=20)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(side=tk.BOTTOM, pady=20)

    fig, ax = plt.subplots(figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(expand=True, fill=tk.BOTH)

    def process_image(image_path):
        """Process the selected image and create the sketch animation."""
        global sketch_image, original_height, original_width
        image = cv2.imread(image_path)
        if image is None:
            tk.messagebox.showerror("Error", f"Unable to read image at {image_path}")
            return

        original_height, original_width = image.shape[:2]
        image = resize_image(image)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        outline_edges = detect_outlines(image)
        detail_edges = detect_details(image)

        outline_lines = get_lines_with_intensity(outline_edges, gray, MAX_LINES // 2, MIN_LINE_LENGTH, MAX_LINE_GAP)
        detail_lines = get_lines_with_intensity(detail_edges, gray, MAX_LINES // 2, MIN_LINE_LENGTH, MAX_LINE_GAP)

        outline_lines = order_lines(outline_lines, mode='length', length_threshold=MIN_OUTLINE_LENGTH)
        detail_lines = order_lines(detail_lines, mode='grid')

        outline_frames = int(TARGET_FRAMES * OUTLINE_RATIO)
        interval = int(ANIMATION_DURATION * 1000 / TARGET_FRAMES)

        sketch_image = np.full((height, width), 255, dtype=np.uint8)

        ax.clear()
        img_plot = ax.imshow(sketch_image, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')

        def animate_sketch_gui(frame):
            """Animation function for creating the sketch effect frame by frame."""
            if frame < outline_frames:
                progress = frame / outline_frames
                lines_to_draw = int(len(outline_lines) * progress)
                for i in range(lines_to_draw):
                    x1, y1, x2, y2, intensity, length = outline_lines[i]
                    draw_line_with_shading(sketch_image, x1, y1, x2, y2, intensity, length, True, MIN_SHADING_THRESHOLD, MIN_SHADING_DENSITY)
            else:
                detail_progress = (frame - outline_frames) / (TARGET_FRAMES - outline_frames)
                lines_to_draw = int(len(detail_lines) * detail_progress)
                for i in range(lines_to_draw):
                    x1, y1, x2, y2, intensity, length = detail_lines[i]
                    draw_line_with_shading(sketch_image, x1, y1, x2, y2, intensity, length, False, MIN_SHADING_THRESHOLD, MIN_SHADING_DENSITY)

            img_plot.set_array(sketch_image)
            progress = min(100, (frame + 1) / TARGET_FRAMES * 100)
            progress_bar['value'] = progress
            root.update_idletasks()
            return [img_plot]

        anim = FuncAnimation(fig, animate_sketch_gui, frames=TARGET_FRAMES,
                             interval=interval, blit=True, repeat=False)

        canvas.draw()

    root.mainloop()

if __name__ == "__main__":
    main()
