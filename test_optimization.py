import requests
import time
import base64
from PIL import Image
import io

def test_image_upload():
    """Test the optimized image upload endpoint"""
    print("Testing OVERWATCH image upload optimization...")
    
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test the upload endpoint
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    
    start_time = time.time()
    try:
        response = requests.post('http://127.0.0.1:5000/upload_image', files=files)
        total_time = time.time() - start_time
        
        print(f"Response Status: {response.status_code}")
        print(f"Total Request Time: {total_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Processing Time: {data.get('processing_time', 'N/A')} seconds")
            print(f"Objects Detected: {data.get('object_count', 0)}")
            print(f"FPS: {data.get('fps', 'N/A')}")
            print("✅ Optimization successful!")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")

if __name__ == "__main__":
    test_image_upload() 