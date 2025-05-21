import mss
import time
import numpy as np
from PIL import Image
import pytesseract
import cv2
import re
from pymongo import MongoClient
from datetime import datetime
from difflib import SequenceMatcher

# OPTIONAL: Set this if Tesseract is not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Tesseract to use CJK languages
# You need to install the language data files for these languages
# For Windows: Download from https://github.com/tesseract-ocr/tessdata
# For Linux: sudo apt-get install tesseract-ocr-jpn tesseract-ocr-kor tesseract-ocr-chi-sim
TESSERACT_LANG = 'eng+jpn+kor+chi_sim+chi_tra+rus'  # English + Japanese + Korean + Simplified Chinese + Traditional Chinese + Russian

# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['poe2_offers']
collection = db['items']

# Create unique index on rawLine field
collection.create_index("rawLine", unique=True)

def capture_screen(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        return img

def preprocess_image(pil_img):
    # Convert to OpenCV format
    img = np.array(pil_img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Contrast enhancement
    gray = cv2.equalizeHist(gray)
    # Upscale
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Denoise
    gray = cv2.medianBlur(gray, 3)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(thresh, -1, kernel)
    return Image.fromarray(sharp)

def extract_text(pil_img):
    custom_config = r'--oem 3 --psm 6 -l ' + TESSERACT_LANG
    return pytesseract.image_to_string(pil_img, config=custom_config).strip()

def select_monitor():
    with mss.mss() as sct:
        monitors = sct.monitors[1:]  # Skip the "all monitors" entry
        print("\n📺 Available monitors:")
        for i, monitor in enumerate(monitors, 1):
            print(f"{i}. Monitor {i}: {monitor['width']}x{monitor['height']} at position ({monitor['left']}, {monitor['top']})")
        
        while True:
            try:
                choice = int(input("\nSelect monitor number: "))
                if 1 <= choice <= len(monitors):
                    return monitors[choice - 1]
                else:
                    print("⚠️ Invalid monitor number. Please try again.")
            except ValueError:
                print("⚠️ Please enter a valid number.")

def select_region():
    print("🖱️ Please select the region to monitor:")
    print("1. Click and drag to select the region")
    print("2. Press 'Enter' to confirm the selection")
    print("3. Press 'r' to reset the selection")
    print("4. Press 'q' to quit")
    
    # Get monitor selection first
    monitor = select_monitor()
    
    # Capture selected monitor
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Initialize variables
    drawing = False
    start_point = None
    end_point = None
    temp_img = img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, temp_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            temp_img = img.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_img = img.copy()
                cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Select Region', temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('Select Region', temp_img)
    
    cv2.namedWindow('Select Region')
    cv2.setMouseCallback('Select Region', mouse_callback)
    
    while True:
        cv2.imshow('Select Region', temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Check if window was closed
        if cv2.getWindowProperty('Select Region', cv2.WND_PROP_VISIBLE) < 1:
            print("\n🛑 Region selection cancelled by user.")
            cv2.destroyAllWindows()
            exit()
        
        if key == 13:  # Enter key
            if start_point and end_point:
                break
            else:
                print("⚠️ Please select a region first!")
        elif key == ord('r'):  # Reset
            start_point = None
            end_point = None
            temp_img = img.copy()
        elif key == ord('q'):  # Quit
            print("\n🛑 Region selection cancelled by user.")
            cv2.destroyAllWindows()
            exit()
    
    cv2.destroyAllWindows()
    
    # Calculate region coordinates relative to the selected monitor
    x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
    x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
    
    # Add monitor offset to the coordinates
    region = {
        'left': monitor['left'] + x1,
        'top': monitor['top'] + y1,
        'width': x2 - x1,
        'height': y2 - y1
    }
    
    return region

def calculate_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def main():
    # Get region from user selection
    region = select_region()

    print("🔍 Starting real-time text reader (Ctrl+C to stop)...\n")
    try:
        while True:
            img = capture_screen(region)
            text = extract_text(img)
            if text:
                # Find all messages starting with timestamp
                messages = re.findall(r'\[\d{2}:\d{2}\].*?(?=\[\d{2}:\d{2}\]|$)', text, re.DOTALL)
                
                for message in messages:
                    # Remove line breaks and extra spaces
                    clean_message = ' '.join(message.strip().split())
                    print('\n\nComplete message:', clean_message)
                    
                    # Extract message date
                    message_date = re.search(r'\[(\d{2}:\d{2})\]', clean_message)
                    message_date = message_date.group(1) if message_date else None
                    
                    # Extract username (keeping all characters)
                    username_match = re.search(r'@From\s+([^:]+):', clean_message)
                    username = username_match.group(1).strip() if username_match else None
                    
                    # Remove timestamp and clean message
                    clean_message = re.sub(r'^\[\d{2}:\d{2}\].*?:', '', clean_message).strip()
                    
                    if 'your' in clean_message.lower() and 'listed' in clean_message.lower():
                        # Find item between "your" and "listed"
                        start = clean_message.lower().find('your') + 4
                        end = clean_message.lower().find('listed')
                        if start < end:
                            extracted = clean_message[start:end].strip()
                            # Remove quotes, extra spaces and special characters from item, and lowercase
                            extracted = re.sub(r'^[\'"]|[\'"]$', '', extracted).strip()
                            # Only allow a-z, A-Z, 0-9, spaces, and parentheses
                            extracted_clean = re.sub(r'[^a-zA-Z0-9 ()]', '', extracted).strip().lower()
                            # If the cleaned item contains any non-allowed chars, skip
                            if not re.fullmatch(r'[a-zA-Z0-9 ()]+', extracted_clean):
                                print(f"⚠️ Skipping item with non-alphanumeric characters: {extracted}")
                                continue
                            print("📝 Item:", extracted_clean)
                            print("👤 User:", username)
                            print("🕒 Message date:", message_date)
                            
                            # Check similarity with existing documents
                            similar_docs = collection.find({})
                            should_insert = True
                            
                            for doc in similar_docs:
                                # Compare username, item and message date
                                username_similarity = calculate_similarity((username or ''), (doc.get('username', '') or ''))
                                # Lowercase and clean item for comparison
                                doc_item_clean = re.sub(r'[^a-zA-Z0-9 ()]', '', doc.get('item', '') or '').strip().lower()
                                item_similarity = calculate_similarity(extracted_clean, doc_item_clean)
                                date_similarity = calculate_similarity((message_date or ''), (doc.get('messageDate', '') or ''))
                                
                                # Ignore only if username and date are both > 0.9
                                if (username_similarity > 0.5 and date_similarity > 0.9):
                                    print(f"⚠️ Message ignored - Similar user and same time")
                                    should_insert = False
                                    break
                            
                            if should_insert:
                                document = {
                                    'rawLine': clean_message,
                                    'item': extracted_clean,
                                    'username': username,
                                    'messageDate': message_date,
                                    'createdAt': datetime.utcnow()
                                }
                                collection.update_one(
                                    {'rawLine': clean_message},
                                    {'$set': document},
                                    upsert=True
                                )
                                print("✅ New line inserted into database")
                    
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        client.close()

if __name__ == "__main__":
    main()