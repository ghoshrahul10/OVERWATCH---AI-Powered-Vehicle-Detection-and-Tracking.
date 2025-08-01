from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import queue
import io
import base64
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=2)
model = None
detection_thread = None
is_running = False
video_stats = {
    'object_count': 0,
    'object_categories': {},
    'fps': 0,
    'last_update': 0
}

# Video upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global model
    try:
        # Use a smaller, faster model
        model = YOLO('yolov5n.pt')  # Use nano model for faster processing
        print(f"Model loaded successfully. Using device: {model.device}")
        
        # Set model to evaluation mode for faster inference
        model.eval()
        
        # If GPU is available, use it
        if torch.cuda.is_available():
            model.to('cuda')
            print("Using GPU for faster processing")
        else:
            print("Using CPU for processing")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Fallback to CPU model
        model = YOLO('yolov5n.pt')
        print("Model loaded with fallback settings")

def process_frame(frame):
    if model is None:
        return frame
    
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    return frame

def detection_worker():
    global is_running, video_stats
    cap = None
    frame_count = 0
    start_time = time.time()
    
    while is_running:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Use webcam by default
        
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Run detection and update stats
        results = model(frame)
        object_count = 0
        object_categories = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                object_count += 1
                
                if class_name in object_categories:
                    object_categories[class_name] += 1
                else:
                    object_categories[class_name] = 1
        
        # Update video stats
        video_stats['object_count'] = object_count
        video_stats['object_categories'] = object_categories
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            video_stats['fps'] = frame_count / elapsed_time
        
        processed_frame = process_frame(frame)
        
        if not frame_queue.full():
            frame_queue.put(processed_frame)
    
    if cap is not None:
        cap.release()

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Vehicle tracking functionality
tracked_vehicles = {}  # Store vehicle characteristics for tracking

def extract_vehicle_features(results, image):
    """Extract vehicle features for tracking"""
    vehicles = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            # Focus on vehicle classes
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Extract vehicle region
                vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Calculate vehicle characteristics
                vehicle_features = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'size': (int(x2-x1), int(y2-y1)),
                    'area': (int(x2-x1)) * (int(y2-y1)),
                    'aspect_ratio': (int(x2-x1)) / (int(y2-y1)) if (int(y2-y1)) > 0 else 0,
                    'center': (int((x1+x2)/2), int((y1+y2)/2)),
                    'region': vehicle_region
                }
                vehicles.append(vehicle_features)
    
    return vehicles

def calculate_vehicle_similarity(vehicle1, vehicle2):
    """Calculate similarity between two vehicles"""
    similarity_score = 0
    
    # Class similarity
    if vehicle1['class'] == vehicle2['class']:
        similarity_score += 0.3
    
    # Size similarity (within 20% difference)
    size_diff = abs(vehicle1['area'] - vehicle2['area']) / max(vehicle1['area'], vehicle2['area'])
    if size_diff < 0.2:
        similarity_score += 0.3
    
    # Aspect ratio similarity
    aspect_diff = abs(vehicle1['aspect_ratio'] - vehicle2['aspect_ratio'])
    if aspect_diff < 0.1:
        similarity_score += 0.2
    
    # Confidence similarity
    conf_diff = abs(vehicle1['confidence'] - vehicle2['confidence'])
    if conf_diff < 0.1:
        similarity_score += 0.2
    
    return similarity_score

def analyze_object_details(results, image):
    """Analyze detailed information about detected objects"""
    object_details = []
    
    # Common vehicle brands for detection
    vehicle_brands = {
        'car': ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Hyundai', 'Kia', 'Nissan', 'Chevrolet', 'Dodge', 'Jeep', 'Subaru', 'Mazda'],
        'truck': ['Ford', 'Chevrolet', 'Dodge', 'Toyota', 'Nissan', 'GMC', 'Ram', 'International', 'Kenworth', 'Peterbilt'],
        'bus': ['Mercedes', 'Volvo', 'Scania', 'MAN', 'Iveco', 'Setra', 'Neoplan', 'Van Hool'],
        'motorcycle': ['Honda', 'Yamaha', 'Kawasaki', 'Suzuki', 'Harley-Davidson', 'BMW', 'Ducati', 'KTM', 'Triumph', 'Indian']
    }
    
    # Color names mapping
    color_names = {
        '#ff0000': 'Red', '#ff4500': 'Orange Red', '#ff8c00': 'Dark Orange',
        '#ffff00': 'Yellow', '#9acd32': 'Yellow Green', '#00ff00': 'Lime',
        '#00fa9a': 'Medium Spring Green', '#00ffff': 'Cyan', '#0000ff': 'Blue',
        '#8a2be2': 'Blue Violet', '#ff00ff': 'Magenta', '#ff1493': 'Deep Pink',
        '#ffffff': 'White', '#d3d3d3': 'Light Gray', '#808080': 'Gray',
        '#000000': 'Black', '#8b4513': 'Saddle Brown', '#a0522d': 'Sienna'
    }
    
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            # Calculate object characteristics
            width = int(x2 - x1)
            height = int(y2 - y1)
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Extract object region for analysis
            object_region = image[int(y1):int(y2), int(x1):int(x2)]
            
            # Enhanced color analysis
            if object_region.size > 0:
                # Calculate dominant colors
                colors = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
                colors_flat = colors.reshape(-1, 3)
                
                # Get dominant color with better analysis
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    kmeans.fit(colors_flat)
                    
                    # Get top 3 dominant colors
                    unique, counts = np.unique(kmeans.labels_, return_counts=True)
                    dominant_indices = unique[np.argsort(counts)[-3:]]
                    dominant_colors = []
                    
                    for idx in dominant_indices:
                        color = kmeans.cluster_centers_[idx].astype(int)
                        color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                        dominant_colors.append({
                            'hex': color_hex,
                            'rgb': color.tolist(),
                            'percentage': (counts[idx] / len(colors_flat)) * 100
                        })
                    
                    primary_color = dominant_colors[0]
                    primary_color_name = color_names.get(primary_color['hex'], 'Custom')
                    
                except:
                    primary_color = {'hex': '#808080', 'rgb': [128, 128, 128], 'percentage': 100}
                    primary_color_name = 'Gray'
                    dominant_colors = [primary_color]
                
                # Calculate brightness
                gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_level = "Dark" if brightness < 85 else "Medium" if brightness < 170 else "Bright"
                
                # Analyze color saturation
                hsv = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                saturation_level = "Low" if saturation < 50 else "Medium" if saturation < 150 else "High"
                
            else:
                primary_color = {'hex': '#808080', 'rgb': [128, 128, 128], 'percentage': 100}
                primary_color_name = 'Gray'
                dominant_colors = [primary_color]
                brightness_level = "Unknown"
                saturation_level = "Unknown"
            
            # Determine object size category
            if area < 1000:
                size_category = "Small"
            elif area < 5000:
                size_category = "Medium"
            else:
                size_category = "Large"
            
            # Enhanced vehicle-specific analysis
            vehicle_info = {}
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                # Estimate vehicle type based on aspect ratio and size
                if class_name == 'car':
                    if aspect_ratio > 2.0:
                        vehicle_type = "Sedan"
                        body_style = "4-door sedan"
                    elif aspect_ratio > 1.5:
                        vehicle_type = "SUV"
                        body_style = "Sport Utility Vehicle"
                    else:
                        vehicle_type = "Compact"
                        body_style = "Compact car"
                elif class_name == 'truck':
                    vehicle_type = "Truck"
                    body_style = "Pickup truck"
                elif class_name == 'bus':
                    vehicle_type = "Bus"
                    body_style = "Transit bus"
                elif class_name == 'motorcycle':
                    vehicle_type = "Motorcycle"
                    body_style = "Two-wheeled vehicle"
                
                # Estimate brand (simplified - in real implementation, you'd use a brand detection model)
                possible_brands = vehicle_brands.get(class_name, [])
                estimated_brand = "Unknown"  # Placeholder for brand detection
                
                # Estimate age based on color and style (simplified)
                if primary_color_name in ['Black', 'White', 'Gray']:
                    age_estimate = "Modern"
                elif primary_color_name in ['Red', 'Blue', 'Green']:
                    age_estimate = "Contemporary"
                else:
                    age_estimate = "Classic"
                
                vehicle_info = {
                    'vehicle_type': vehicle_type,
                    'body_style': body_style,
                    'estimated_brand': estimated_brand,
                    'estimated_age': age_estimate,
                    'estimated_length': f"{width:.1f}px",
                    'estimated_width': f"{height:.1f}px",
                    'size_category': size_category,
                    'color_scheme': {
                        'primary_color': primary_color_name,
                        'color_hex': primary_color['hex'],
                        'saturation': saturation_level
                    }
                }
            
            # Object specifications based on class
            object_specs = {}
            if class_name == 'person':
                object_specs = {
                    'type': 'Human',
                    'estimated_height': f"{height:.1f}px",
                    'estimated_width': f"{width:.1f}px",
                    'clothing_color': primary_color_name,
                    'visibility': brightness_level
                }
            elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                object_specs = {
                    'type': vehicle_type,
                    'body_style': vehicle_info.get('body_style', 'Unknown'),
                    'estimated_brand': vehicle_info.get('estimated_brand', 'Unknown'),
                    'color': primary_color_name,
                    'size': size_category,
                    'age_estimate': vehicle_info.get('estimated_age', 'Unknown')
                }
            else:
                object_specs = {
                    'type': class_name.title(),
                    'color': primary_color_name,
                    'size': size_category,
                    'shape': 'Rectangular' if aspect_ratio > 0.8 and aspect_ratio < 1.2 else 'Elongated'
                }
            
            # Create detailed object information
            object_detail = {
                'id': i + 1,
                'class_name': class_name,
                'confidence': round(conf * 100, 2),
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': width,
                    'height': height,
                    'area': area,
                    'aspect_ratio': round(aspect_ratio, 2),
                    'center': (center_x, center_y)
                },
                'characteristics': {
                    'size_category': size_category,
                    'primary_color': primary_color_name,
                    'color_hex': primary_color['hex'],
                    'dominant_colors': dominant_colors,
                    'brightness_level': brightness_level,
                    'saturation_level': saturation_level,
                    'position': {
                        'x_percent': round((center_x / image.shape[1]) * 100, 1),
                        'y_percent': round((center_y / image.shape[0]) * 100, 1)
                    }
                },
                'specifications': object_specs,
                'vehicle_info': vehicle_info if class_name in ['car', 'truck', 'bus', 'motorcycle'] else None,
                'detection_quality': {
                    'high_confidence': conf > 0.7,
                    'medium_confidence': 0.5 <= conf <= 0.7,
                    'low_confidence': conf < 0.5
                }
            }
            
            object_details.append(object_detail)
    
    return object_details

@app.route('/search_objects', methods=['POST'])
def search_objects():
    """Search for specific objects in an image"""
    try:
        print("Search objects endpoint called")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get search term from form data
        search_term = request.form.get('search_term', '').lower()
        print(f"Searching for '{search_term}' in file: {file.filename}")
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Resize image for faster processing
        height, width = img.shape[:2]
        if width > 640 or height > 640:
            scale = min(640/width, 640/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Run detection
        start_time = time.time()
        results = model(img, conf=0.25)
        detection_time = time.time() - start_time
        
        # Analyze object details
        object_details = analyze_object_details(results, img)
        
        # Filter objects based on search term
        if search_term:
            filtered_objects = []
            for obj in object_details:
                # Search in class name, color, specifications
                searchable_text = f"{obj['class_name']} {obj['characteristics']['primary_color']} {obj['specifications']['type']}".lower()
                if search_term in searchable_text:
                    filtered_objects.append(obj)
            object_details = filtered_objects
        
        # Process frame for display
        result_img = process_frame(img)
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Calculate statistics
        total_objects = len(object_details)
        vehicle_count = len([obj for obj in object_details if obj['class_name'] in ['car', 'truck', 'bus', 'motorcycle']])
        person_count = len([obj for obj in object_details if obj['class_name'] == 'person'])
        
        # Group objects by class
        objects_by_class = {}
        for obj in object_details:
            class_name = obj['class_name']
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(obj)
        
        response_data = {
            'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8'),
            'processing_time': round(detection_time, 2),
            'search_term': search_term,
            'total_objects': total_objects,
            'vehicle_count': vehicle_count,
            'person_count': person_count,
            'objects_by_class': objects_by_class,
            'detailed_objects': object_details,
            'analysis_summary': {
                'high_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['high_confidence']]),
                'medium_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['medium_confidence']]),
                'low_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['low_confidence']]),
                'size_distribution': {
                    'small': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Small']),
                    'medium': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Medium']),
                    'large': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Large'])
                }
            }
        }
        
        print(f"Search completed: {total_objects} objects found for '{search_term}'")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in search_objects: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/track_vehicle', methods=['POST'])
def track_vehicle():
    """Track vehicles across multiple images/locations"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400
    
    print(f"Processing {len(files)} images for vehicle tracking")
    
    all_vehicles = []
    location_results = []
    
    for i, file in enumerate(files):
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        # Run detection
        results = model(img)
        vehicles = extract_vehicle_features(results, img)
        
        # Process frame for display
        result_img = process_frame(img)
        _, buffer = cv2.imencode('.jpg', result_img)
        
        location_data = {
            'location_id': i + 1,
            'filename': file.filename,
            'vehicles': vehicles,
            'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8'),
            'total_vehicles': len(vehicles)
        }
        location_results.append(location_data)
        all_vehicles.extend(vehicles)
    
    # Analyze vehicle patterns across locations
    tracking_analysis = analyze_vehicle_patterns(location_results)
    
    return jsonify({
        'locations': location_results,
        'tracking_analysis': tracking_analysis,
        'total_vehicles_detected': len(all_vehicles)
    })

def analyze_vehicle_patterns(location_results):
    """Analyze vehicle patterns across multiple locations"""
    analysis = {
        'vehicle_types': {},
        'size_distribution': {},
        'common_vehicles': [],
        'tracking_suggestions': []
    }
    
    # Count vehicle types across all locations
    for location in location_results:
        for vehicle in location['vehicles']:
            vehicle_type = vehicle['class']
            if vehicle_type in analysis['vehicle_types']:
                analysis['vehicle_types'][vehicle_type] += 1
            else:
                analysis['vehicle_types'][vehicle_type] = 1
    
    # Find similar vehicles across locations
    similar_vehicles = []
    for i, loc1 in enumerate(location_results):
        for j, loc2 in enumerate(location_results):
            if i != j:  # Compare different locations
                for v1 in loc1['vehicles']:
                    for v2 in loc2['vehicles']:
                        similarity = calculate_vehicle_similarity(v1, v2)
                        if similarity > 0.7:  # High similarity threshold
                            similar_vehicles.append({
                                'location1': loc1['location_id'],
                                'location2': loc2['location_id'],
                                'vehicle1': v1,
                                'vehicle2': v2,
                                'similarity': similarity
                            })
    
    analysis['similar_vehicles'] = similar_vehicles
    
    # Generate tracking suggestions
    if similar_vehicles:
        analysis['tracking_suggestions'].append(
            f"Found {len(similar_vehicles)} potential vehicle matches across locations"
        )
    
    return analysis

@app.route('/analyze_objects', methods=['POST'])
def analyze_objects():
    """Analyze detailed information about objects in an image"""
    try:
        print("Object analysis endpoint called")
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        print(f"Analyzing objects in file: {file.filename}")
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Resize image for faster processing
        height, width = img.shape[:2]
        if width > 640 or height > 640:
            scale = min(640/width, 640/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Run detection
        start_time = time.time()
        results = model(img, conf=0.25)
        detection_time = time.time() - start_time
        
        # Analyze object details
        object_details = analyze_object_details(results, img)
        
        # Process frame for display
        result_img = process_frame(img)
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Calculate statistics
        total_objects = len(object_details)
        vehicle_count = len([obj for obj in object_details if obj['class_name'] in ['car', 'truck', 'bus', 'motorcycle']])
        person_count = len([obj for obj in object_details if obj['class_name'] == 'person'])
        
        # Group objects by class
        objects_by_class = {}
        for obj in object_details:
            class_name = obj['class_name']
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(obj)
        
        response_data = {
            'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8'),
            'processing_time': round(detection_time, 2),
            'total_objects': total_objects,
            'vehicle_count': vehicle_count,
            'person_count': person_count,
            'objects_by_class': objects_by_class,
            'detailed_objects': object_details,
            'analysis_summary': {
                'high_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['high_confidence']]),
                'medium_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['medium_confidence']]),
                'low_confidence_detections': len([obj for obj in object_details if obj['detection_quality']['low_confidence']]),
                'size_distribution': {
                    'small': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Small']),
                    'medium': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Medium']),
                    'large': len([obj for obj in object_details if obj['characteristics']['size_category'] == 'Large'])
                }
            }
        }
        
        print(f"Analysis completed: {total_objects} objects analyzed")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_objects: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/test')
def test():
    return jsonify({'status': 'Server is running', 'message': 'OVERWATCH is operational'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global is_running, detection_thread
    if not is_running:
        is_running = True
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.start()
        return jsonify({"status": "success", "message": "Detection started"})
    return jsonify({"status": "error", "message": "Detection already running"})

@app.route('/stop_detection')
def stop_detection():
    global is_running
    is_running = False
    if detection_thread:
        detection_thread.join()
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        print("Upload endpoint called")
        if 'image' not in request.files:
            print("No image in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        print(f"Processing file: {file.filename}")
        
        # Read image as numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")
            return jsonify({'error': 'Invalid image format'}), 400
        
        print(f"Original image shape: {img.shape}")
        
        # Resize image to reduce processing time (max 640x640)
        height, width = img.shape[:2]
        if width > 640 or height > 640:
            scale = min(640/width, 640/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            print(f"Resized image shape: {img.shape}")
        
        # Check if model is loaded
        if model is None:
            print("Model not loaded")
            return jsonify({'error': 'Model not available'}), 500
        
        # Run detection and categorize objects
        print("Running YOLO detection...")
        start_time = time.time()
        results = model(img, conf=0.25)  # Lower confidence threshold for faster processing
        detection_time = time.time() - start_time
        
        object_count = 0
        object_categories = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                object_count += 1
                
                if class_name in object_categories:
                    object_categories[class_name] += 1
                else:
                    object_categories[class_name] = 1
        
        print(f"Detected {object_count} objects: {object_categories}")
        print(f"Detection time: {detection_time:.2f} seconds")
        
        # Process frame for display
        result_img = process_frame(img)
        
        # Encode result as JPEG with lower quality for faster transmission
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Return both image and categorized statistics
        response_data = {
            'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8'),
            'object_count': object_count,
            'object_categories': object_categories,
            'processing_time': round(detection_time, 2),
            'fps': round(1.0/detection_time, 2) if detection_time > 0 else 1.0
        }
        print("Sending response successfully")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    print(f"Processing video: {video_path}")
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 400
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Process video frames
    frame_count = 0
    total_objects = 0
    object_categories = {}
    processed_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection on frame
        results = model(frame)
        frame_objects = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                frame_objects += 1
                total_objects += 1
                
                if class_name in object_categories:
                    object_categories[class_name] += 1
                else:
                    object_categories[class_name] = 1
        
        # Process frame for display (every 10th frame to save memory)
        if frame_count % 10 == 0:
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frames.append({
                'frame': frame_count,
                'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8'),
                'objects': frame_objects
            })
        
        frame_count += 1
        
        # Limit processed frames to avoid memory issues
        if len(processed_frames) > 20:
            processed_frames = processed_frames[-20:]
    
    cap.release()
    
    # Clean up uploaded file
    os.remove(video_path)
    
    # Calculate average objects per frame
    avg_objects_per_frame = total_objects / frame_count if frame_count > 0 else 0
    
    return jsonify({
        'video_stats': {
            'total_frames': frame_count,
            'duration_seconds': duration,
            'fps': fps,
            'total_objects_detected': total_objects,
            'avg_objects_per_frame': avg_objects_per_frame,
            'object_categories': object_categories
        },
        'sample_frames': processed_frames
    })

@app.route('/get_video_stats')
def get_video_stats():
    return jsonify(video_stats)

if __name__ == '__main__':
    load_model()
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000) 