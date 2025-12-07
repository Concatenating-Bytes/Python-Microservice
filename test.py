import cv2
import base64
import requests
import json
import time

# Configuration
API_URL = "http://127.0.0.1:8000"
USER_ID = "test_user_01"
 
def capture_image(prompt):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return None

    print(f"\n--- {prompt} ---")
    print("Press 's' to save the photo, or 'q' to quit.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow(f"{prompt} - Press 's'", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            captured_frame = frame
            print("Image captured!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"

def test_enrollment():
    img = capture_image("STEP 1: Take photo for ENROLLMENT")
    if img is None: return

    b64_img = encode_image_to_base64(img)
    payload = {"user_id": USER_ID, "image_b64": b64_img}
    
    print(f"Sending request to {API_URL}/enroll...")
    try:
        response = requests.post(f"{API_URL}/enroll", json=payload)
        print("Response Status:", response.status_code)
        print("Response Body:", response.json())
    except Exception as e:
        print("Error:", e)

def test_verification():
    img = capture_image("STEP 2: Take photo for VERIFICATION")
    if img is None: return

    b64_img = encode_image_to_base64(img)
    payload = {"user_id": USER_ID, "image_b64": b64_img}
    
    print(f"Sending request to {API_URL}/verify...")
    try:
        response = requests.post(f"{API_URL}/verify", json=payload)
        print("Response Status:", response.status_code)
        result = response.json()
        print("Response Body:", json.dumps(result, indent=2))
        
        if result.get("verified"):
            print("\n✅ SUCCESS: User Verified!")
        else:
            print("\n❌ FAILED: User NOT Verified (Low Similarity)")
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    try:
        r = requests.get(f"{API_URL}/health")
        print("Server Health:", r.json())
        
        test_enrollment()
        print("\nWaiting 2 seconds...\n")
        time.sleep(2)
        test_verification()
        
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL ERROR: Could not connect to server.")
        print("Make sure you ran 'uvicorn app:app --reload' in a separate terminal!")