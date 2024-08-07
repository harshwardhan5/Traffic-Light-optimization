```markdown
# Traffic Light Optimization System

Welcome to the Traffic Light Optimization System! This project aims to dynamically calculate the duration of red lights at traffic signals using YOLOv8, optimizing traffic flow based on real-time vehicle counts.

## Overview

This system uses a machine learning approach to detect and count vehicles at traffic intersections. By adjusting the red light duration based on the number of cars present, the system aims to reduce congestion and improve traffic efficiency. The YOLOv8 model is employed for real-time vehicle detection, and the system is built with Python, leveraging libraries such as Pandas, OpenCV, and PyTorch.

## Features

- **Real-time Vehicle Detection:** Utilizes YOLOv8 to detect vehicles in real-time.
- **Dynamic Red Light Duration:** Adjusts the duration of red lights based on the number of cars detected.
- **Data Processing:** Uses Pandas for data handling and OpenCV for image processing.
- **Model Training:** Employs PyTorch for model training and optimization.
- **Performance Monitoring:** Plots learning curves to visualize training and validation losses.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/traffic-light-optimization.git
   cd traffic-light-optimization
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   - Place your training and validation images in the appropriate directories as specified in your dataset configuration.

2. **Train the model:**
   ```python
   results = model.train(
       data=yaml_file_path,
       epochs=10,
       imgsz=640,
       device=0,
       patience=50,
       batch=32,
       optimizer='auto',
       lr0=0.0001,
       lrf=0.1,
       dropout=0.1
   )
   ```

3. **Plot learning curves:**
   ```python
   # Load the CSV file containing training results
   results_csv_path = os.path.join(post_training_files_path, 'results.csv')
   df = pd.read_csv(results_csv_path)
   df.columns = df.columns.str.strip()

   # Plot the learning curves
   plot_learning_curve(df, 'train/box_loss', 'val/box_loss', 'Box Loss Learning Curve')
   plot_learning_curve(df, 'train/cls_loss', 'val/cls_loss', 'Classification Loss Learning Curve')
   plot_learning_curve(df, 'train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss Learning Curve')
   ```

4. **Run inferences on validation set:**
   ```python
   # Define the directory path for validation images
   valid_image_path = os.path.join(dataset_path, 'valid', 'images')

   # List all jpg images in the directory
   image_files = [file for file in os.listdir(valid_image_path) if file.endswith('.jpg')]

   # Select 9 images at equal intervals
   num_images = len(image_files)
   selected_images = [image_files[i] for i in range(0, num_images, num_images // 9)]

   # Initialize the subplot
   fig, axes = plt.subplots(3, 3, figsize=(20, 21))
   plt.title('Validation Set Inferences', fontsize=24)

   # Perform inferences on each selected image and display it
   for i, ax in enumerate(axes.flatten()):
       image_path = os.path.join(valid_image_path, selected_images[i])
       results = best_model.predict(source=image_path, imgsz=640, conf=0.5)
       annotated_image = results[0].plot(line_width=1)
       annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
       ax.imshow(annotated_image_rgb)
       ax.axis('off')

   plt.tight_layout()
   plt.show()
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements, bugs, or other suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The YOLOv8 model from Ultralytics
- Libraries used: Pandas, OpenCV, PyTorch, Matplotlib, Seaborn

## Contact

For any questions or inquiries, please contact [your-email@example.com](mailto:your-kumarhrshwrdhn@gmail.com).
```

Feel free to replace placeholders like `yourusername` and `your-email@example.com` with your actual GitHub username and email address.
