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

# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['poe2_offers']
collection = db['items']

# Create unique index on rawLine field
collection.create_index("rawLine", unique=True)

REGION = {
    'top': 600,
    'left': 0,
    'width': 1300,
    'height': 445
}

def capture_screen(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        return img

def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def extract_text(pil_img):
    return pytesseract.image_to_string(pil_img).strip()

def preview_region(region):
    print("📸 Previewing captured region. Close the image window to continue...")
    img = capture_screen(region)
    img.show()  # Opens in default image viewer
    input("Press Enter after closing the preview window to start OCR...\n")

def calculate_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def main():
    preview_region(REGION)

    print("🔍 Starting real-time text reader (Ctrl+C to stop)...\n")
    try:
        while True:
            img = capture_screen(REGION)
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
                    
                    # Extract username
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
                            # Remove quotes, extra spaces and special characters from item
                            extracted = re.sub(r'^[\'"]|[\'"]$', '', extracted).strip()
                            # Remove special characters and icons (keep letters, numbers, spaces and parentheses)
                            extracted = re.sub(r'[^\w\s()]', '', extracted).strip()
                            print("📝 Item:", extracted)
                            print("👤 User:", username)
                            print("🕒 Message date:", message_date)
                            
                            # Check similarity with existing documents
                            similar_docs = collection.find({})
                            should_insert = True
                            high_item_similarity_doc = None
                            
                            for doc in similar_docs:
                                # Compare username, item and message date
                                username_similarity = calculate_similarity(username, doc.get('username', ''))
                                item_similarity = calculate_similarity(extracted, doc.get('item', ''))
                                date_similarity = calculate_similarity(message_date, doc.get('messageDate', ''))
                                
                                # Weighted average of similarities
                                total_similarity = (username_similarity * 0.3 + 
                                                  item_similarity * 0.5 + 
                                                  date_similarity * 0.2)
                                
                                if total_similarity > 0.9:  # 90% total similarity
                                    print(f"⚠️ Message ignored - {total_similarity:.2%} similarity with existing message")
                                    should_insert = False
                                    break
                                elif item_similarity > 0.95 and username_similarity < 0.9:  # Very similar item but different user
                                    high_item_similarity_doc = doc
                                    should_insert = False
                            
                            if high_item_similarity_doc:
                                # Increment counter for document with similar item
                                collection.update_one(
                                    {'_id': high_item_similarity_doc['_id']},
                                    {'$inc': {'count': 1}}
                                )
                                print(f"✅ Counter incremented for similar item ({item_similarity:.2%}) from different user")
                            elif should_insert:
                                document = {
                                    'rawLine': clean_message,
                                    'item': extracted,
                                    'username': username,
                                    'messageDate': message_date,
                                    'createdAt': datetime.utcnow(),
                                    'count': 1
                                }
                                collection.update_one(
                                    {'rawLine': clean_message},
                                    {'$set': document},
                                    upsert=True
                                )
                                print("✅ New line inserted into database")
                    
            time.sleep(20)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        client.close()

if __name__ == "__main__":
    main()