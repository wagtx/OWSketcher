# OWSketcher

OWSketcher is a Python script that creates an animated sketch of an input image. It uses edge detection and line drawing techniques to produce a sketch-like representation of the image, with the sketching process animated to show the progression from outlines to details.

## Features

- Converts images to animated sketches
- Customizable sketching parameters
- Standardizes input images to a consistent width for processing
- Separates outline and detail drawing phases
- Adds shading effects to enhance the sketch appearance
- Graphical user interface for easy image selection and progress tracking

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy
- Tkinter (usually comes with Python)

You can install the required packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

1. Ensure all requirements are installed.
2. Run the script:
   ```
   python sketcher.py
   ```
3. Click the "Choose File" button in the GUI to select an image.
4. The script will process the image and display an animated sketch.

## How It Works

1. **Image Standardization**: The input image is resized to a standard width of 2048 pixels while maintaining its aspect ratio.
2. **Edge Detection**: The script uses Canny edge detection to identify outlines and details in the image.
3. **Line Extraction**: Hough Line Transform is applied to extract lines from the edge-detected image.
4. **Line Ordering**: Lines are categorized and ordered based on their length and position in a grid.
5. **Sketching Animation**: The script draws lines progressively, starting with outlines and then adding details.
6. **Shading**: Short perpendicular lines are added to create a shading effect.

## Customization

You can customize the sketching process by modifying the constants at the top of the `sketcher.py` file:

- `ANIMATION_DURATION`: Total duration of the animation in seconds
- `TARGET_FRAMES`: Target number of frames for the animation
- `OUTLINE_RATIO`: Ratio of frames dedicated to outlines
- `STANDARD_WIDTH`: Standard width for processing images (default: 2048 pixels)
- `MIN_LINE_LENGTH`: Minimum length of line to be detected
- `MAX_LINE_GAP`: Maximum gap between line segments to treat them as a single line
- `MAX_LINES`: Maximum number of lines to process
- `MIN_OUTLINE_LENGTH`: Minimum length for a line to be considered an outline
- `MIN_DETAIL_LENGTH`: Minimum length for a line to be considered a detail
- `MIN_SHADING_THRESHOLD`: Maximum length of line to apply shading
- `MIN_SHADING_DENSITY`: Controls density of shading lines
- `SHADING_LINE_LENGTH`: Length of individual shading lines
- `GRID_ROWS` and `GRID_COLS`: Number of rows and columns in the grid for line ordering

## Recent Updates

- Added a graphical user interface (GUI) for easier image selection and progress tracking.
- Implemented image standardization to a 2048-pixel width for consistent processing across different image sizes.
- Removed resolution-based scaling in favor of standardized image processing.
- Adjusted line detection and drawing parameters for optimal results with the standard image width.
- Increased the maximum number of lines and adjusted shading parameters for better detail.
- Updated the grid system for more detailed line ordering.

## Limitations

- The script may not perform optimally on images with very unusual aspect ratios.
- Processing time can be long for complex images or when many lines are detected.
- The quality of the sketch depends on the input image's contrast and complexity.

## Contributing

Contributions to improve OWSketcher are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is open-source and available under the MIT License.
