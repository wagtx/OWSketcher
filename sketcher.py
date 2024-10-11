import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance

# Configuration constants
ANIMATION_DURATION = 10  # Animation duration in seconds
TARGET_FRAMES = 100  # Target number of frames for the animation
OUTLINE_PASSES = 3  # Number of passes for outlines

# Line detection parameters
MIN_LINE_LENGTH = 12
MAX_LINE_GAP = 6
MAX_LINES = 10000

# Line length thresholds
MIN_OUTLINE_LENGTH = 35
MIN_DETAIL_LENGTH = 25

# Shading parameters
MIN_SHADING_THRESHOLD = 35
MIN_SHADING_DENSITY = 10
SHADING_LINE_LENGTH = 4

# Detail factor parameters
MIN_DETAIL_FACTOR = 0.5
MAX_DETAIL_FACTOR = 1.0
REFERENCE_RESOLUTION = 1000 * 1000  # 1 megapixel as reference

def detect_outlines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated

def detect_details(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges

def get_lines_with_intensity(edges, gray, max_lines, min_line_length, max_line_gap):
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

def draw_line_with_shading(img, x1, y1, x2, y2, intensity, length, is_outline, shading_threshold, shading_density):
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

def animate_sketch(frame, outline_lines, detail_lines, img_plot, progress_text, lines_per_frame, outline_passes, shading_threshold, shading_density):
    global sketch_image
    
    total_outline_frames = len(outline_lines) * outline_passes // lines_per_frame
    
    if frame < total_outline_frames:
        # Drawing outlines
        start_idx = (frame * lines_per_frame) % len(outline_lines)
        end_idx = min(start_idx + lines_per_frame, len(outline_lines))
        for i in range(start_idx, end_idx):
            x1, y1, x2, y2, intensity, length = outline_lines[i]
            draw_line_with_shading(sketch_image, x1, y1, x2, y2, intensity, length, True, shading_threshold, shading_density)
    else:
        # Draw details
        detail_frame = frame - total_outline_frames
        start_idx = detail_frame * lines_per_frame
        end_idx = min(start_idx + lines_per_frame, len(detail_lines))
        for i in range(start_idx, end_idx):
            x1, y1, x2, y2, intensity, length = detail_lines[i]
            draw_line_with_shading(sketch_image, x1, y1, x2, y2, intensity, length, False, shading_threshold, shading_density)
    
    img_plot.set_array(sketch_image)
    progress = min(100, (frame + 1) / (total_outline_frames + len(detail_lines) // lines_per_frame) * 100)
    progress_text.set_text(f"Progress: {progress:.1f}%")
    
    return [img_plot, progress_text]

def main():
    global height, width, sketch_image
    image_path = input("Enter the path to your image: ")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate detail factor based on image resolution, with bounds
    detail_factor = max(MIN_DETAIL_FACTOR, min(MAX_DETAIL_FACTOR, np.sqrt((height * width) / REFERENCE_RESOLUTION)))
    
    outline_edges = detect_outlines(image)
    detail_edges = detect_details(image)
    
    max_lines = int(min(MAX_LINES, height * width // 100) * detail_factor)
    min_line_length = max(MIN_LINE_LENGTH, int(MIN_LINE_LENGTH * 1.5 / detail_factor))
    max_line_gap = max(MAX_LINE_GAP, int(MAX_LINE_GAP * 1.5 / detail_factor))
    
    outline_lines = get_lines_with_intensity(outline_edges, gray, max_lines // 2, min_line_length, max_line_gap)
    detail_lines = get_lines_with_intensity(detail_edges, gray, max_lines // 2, min_line_length, max_line_gap)
    
    length_threshold_outline = max(MIN_OUTLINE_LENGTH, int(MIN_OUTLINE_LENGTH * 2 / detail_factor))
    length_threshold_detail = max(MIN_DETAIL_LENGTH, int(MIN_DETAIL_LENGTH * 2 / detail_factor))
    
    outline_lines = order_lines(outline_lines, mode='length', length_threshold=length_threshold_outline)
    detail_lines = order_lines(detail_lines, mode='length', length_threshold=length_threshold_detail)
    
    total_lines = len(outline_lines) * OUTLINE_PASSES + len(detail_lines)
    lines_per_frame = max(1, total_lines // TARGET_FRAMES)
    actual_frames = -(-total_lines // lines_per_frame)  # Ceiling division
    interval = max(1, int(ANIMATION_DURATION * 1000 / actual_frames))  # Ensure minimum 1ms interval
    
    shading_threshold = max(MIN_SHADING_THRESHOLD, int(MIN_SHADING_THRESHOLD * 2 / detail_factor))
    shading_density = max(MIN_SHADING_DENSITY, int(MIN_SHADING_DENSITY * 2 / detail_factor))
    
    sketch_image = np.full((height, width), 255, dtype=np.uint8)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    img_plot = ax.imshow(sketch_image, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='red', fontsize=12, verticalalignment='top')
    
    anim = FuncAnimation(fig, animate_sketch, frames=actual_frames,
                         fargs=(outline_lines, detail_lines, img_plot, progress_text, lines_per_frame, OUTLINE_PASSES, shading_threshold, shading_density),
                         interval=interval, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    main()
