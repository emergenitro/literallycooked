import os
import time
import json
import platform
import subprocess
import tempfile
import threading
from typing import List, Dict, Tuple

from flask import Flask, render_template, Response, jsonify
from dotenv import load_dotenv, find_dotenv
import cv2
import requests
import base64
from anthropic import Anthropic

# Load environment variables
print(f"Current working directory: {os.getcwd()}")
env_path = find_dotenv()
print(f"Found .env file at: {env_path}")

# Try to load with explicit path
load_dotenv(env_path)

app = Flask(__name__)


class LiterallyCookedApp:
    def __init__(self):
        # Debug prints
        print("Loading environment variables...")
        print(f"ANTHROPIC exists: {'ANTHROPIC' in os.environ}")
        api_key = os.getenv("ANTHROPIC", "NOT_FOUND")
        print(f"Full API key length: {len(api_key)}")
        print(f"First 10 chars of key: {api_key[:10]}")

        # Core initialization
        self.anthropic = Anthropic(api_key=api_key)
        self.api_key = os.getenv("GOOGLE_CLOUD_API_KEY", "")
        self.vision_base_url = "https://vision.googleapis.com/v1/images:annotate"
        self.cap = cv2.VideoCapture(0)

        # State variables
        self.detected_items: List[str] = []
        self.current_recipe: Dict = {}
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.0
        self.last_center = None
        self.box_size = 300
        self.smoothing = 0.7
        self.current_detection = None
        self.countdown_start = None
        self.countdown_duration = 30
        self.recipe_generated = False
        self.saved_ingredients: List[str] = []
        self.lock = threading.Lock()

    def detect_foods(self, frame) -> List[str]:
        frame_height, frame_width = frame.shape[:2]

        # Convert the frame to jpg and then to base64
        success, encoded_image = cv2.imencode(".jpg", frame)
        image_content = encoded_image.tobytes()
        image_b64 = base64.b64encode(image_content).decode("utf-8")

        # Prepare the request
        request_json = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 10}],
                }
            ]
        }

        # Check if the image is being received correctly
        print(f"Image content length: {len(image_content)}")
        print(f"First 10 bytes of image content: {image_content[:10]}")

        # Save the image into a temporary file in ./tmp
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_image_path = os.path.join(tmp_dir, "image.jpg")
        with open(tmp_image_path, "wb") as f:
            f.write(image_content)

        # Make the API request
        response = requests.post(
            f"{self.vision_base_url}?key={self.api_key}", json=request_json
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return []

        results = response.json()

        # Process detections
        detected = []
        best_detection = None
        highest_confidence = 0

        print("\nFood detections:")
        for annotation in results.get("responses", [{}])[0].get(
            "localizedObjectAnnotations", []
        ):
            # Check if it's a food item
            if annotation["name"].lower() in [
                "food",
                "fruit",
                "vegetable",
                "beverage",
                "apple",
                "banana",
                "orange",
            ]:
                confidence = annotation["score"]
                print(f"{annotation['name']}: {confidence:.2f}")

                try:
                    if confidence > highest_confidence:
                        highest_confidence = confidence

                        # Get bounding box - with error handling
                        vertices = annotation["boundingPoly"]["normalizedVertices"]
                        if len(vertices) >= 4:  # Make sure we have all corners
                            x1 = int(vertices[0].get("x", 0) * frame_width)
                            y1 = int(vertices[0].get("y", 0) * frame_height)
                            x2 = int(vertices[2].get("x", 0) * frame_width)
                            y2 = int(vertices[2].get("y", 0) * frame_height)

                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            best_detection = (
                                annotation["name"],
                                confidence,
                                (center_x, center_y),
                            )
                except (KeyError, IndexError) as e:
                    print(
                        f"Warning: Could not process bounding box for {annotation['name']}"
                    )
                    continue

                if annotation["name"] not in detected:
                    detected.append(annotation["name"])

        if best_detection:
            self.current_detection = best_detection

        # Add newly detected foods to saved_ingredients if not already present
        for item in detected:
            if item not in self.saved_ingredients:
                self.saved_ingredients.append(item)

        return detected

    def open_recipe_terminal(self, recipe: Dict):
        # Since we're using Flask, we'll handle recipe display on the web page
        # Here, we'll just set the current_recipe
        with self.lock:
            self.current_recipe = recipe

    def get_recipe_suggestion(self, ingredients: List[str]) -> Dict:
        print(f"Generating recipe for ingredients: {ingredients}")

        prompt = f"""Given these ingredients: {", ".join(ingredients)}, and assuming basic kitchen tools and cutlery,
suggest a creative recipe that uses them. 
Then, suggest potential improvements or variations if the cook had additional ingredients.
You must respond with ONLY valid JSON in this exact format:
{{
    "name": "Recipe Name",
    "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity"],
    "tools_needed": ["tool 1", "tool 2"],
    "instructions": ["step 1", "step 2", "step 3"],
    "improvements": "Suggestions for additional ingredients and variations"
}}"""

        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        print(response)
        try:
            content = response.content[0].text
            print("Received response from Claude:")
            print(content)  # Add this debug print
            print("\nTrying to parse as JSON...")
            recipe = json.loads(content)
            print("Successfully parsed JSON")
            self.open_recipe_terminal(recipe)
            return recipe
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            print(f"Error parsing recipe: {e}")
            return {
                "name": f"Simple {ingredients[0]} Recipe",
                "ingredients": ingredients,
                "tools_needed": ["Basic kitchen tools"],
                "instructions": ["Prepare ingredients as desired"],
                "improvements": "Add seasonings to taste",
            }

    def draw_ui(self, frame, detected_items: List[str]):
        cv2.putText(
            frame,
            "Detected Foods:",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y_offset = 60
        for item in detected_items:
            cv2.putText(
                frame,
                f"- {item}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 30

        if self.current_recipe:
            recipe_y = y_offset + 20
            recipe = self.current_recipe

            cv2.putText(
                frame,
                f"Recipe: {recipe.get('name', '')}",
                (10, recipe_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            "Press 'r' for recipe | 's' to save | 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def draw_box(self, frame):
        if not self.current_detection:
            return

        frame_height, frame_width = frame.shape[:2]
        name, confidence, (center_x, center_y) = self.current_detection

        if self.last_center is None:
            self.last_center = (center_x, center_y)

        new_x = int(
            self.last_center[0] * self.smoothing + center_x * (1 - self.smoothing)
        )
        new_y = int(
            self.last_center[1] * self.smoothing + center_y * (1 - self.smoothing)
        )

        new_x = max(self.box_size // 2, min(frame_width - self.box_size // 2, new_x))
        new_y = max(self.box_size // 2, min(frame_height - self.box_size // 2, new_y))

        x = new_x - self.box_size // 2
        y = new_y - self.box_size // 2

        self.last_center = (new_x, new_y)

        cv2.rectangle(
            frame, (x, y), (x + self.box_size, y + self.box_size), (0, 255, 0), 2
        )

        label_text = f"{name}: {confidence:.2f}"
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.line(frame, (new_x - 10, new_y), (new_x + 10, new_y), (0, 255, 0), 1)
        cv2.line(frame, (new_x, new_y - 10), (new_x, new_y + 10), (0, 255, 0), 1)

    def draw_countdown(self, frame):
        if self.countdown_start is None:
            self.countdown_start = time.time()

        elapsed_time = time.time() - self.countdown_start
        remaining_time = max(0, self.countdown_duration - elapsed_time)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)

        countdown_text = f"Generating recipe in: {int(remaining_time)} seconds"
        cv2.putText(
            frame,
            countdown_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if remaining_time == 0 and not self.recipe_generated:
            self.current_recipe = self.get_recipe_suggestion(self.saved_ingredients)
            self.recipe_generated = True

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = time.time()

            if current_time - self.last_detection_time >= self.detection_cooldown:
                detected = self.detect_foods(frame)
                self.detected_items = detected
                self.last_detection_time = current_time

            self.draw_box(frame)
            self.draw_countdown(frame)
            self.draw_ui(frame, self.detected_items)

            if self.recipe_generated:
                # Optionally, you can highlight that the recipe is ready
                cv2.putText(
                    frame,
                    "Recipe Generated!",
                    (frame.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            if self.recipe_generated:
                # Stop after recipe is generated
                break

        self.cap.release()


# Initialize the application
literally_cooked_app = LiterallyCookedApp()


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


def generate():
    """Video streaming generator function."""
    return literally_cooked_app.generate_frames()


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/recipe")
def get_recipe():
    """Endpoint to get the generated recipe."""
    with literally_cooked_app.lock:
        recipe = literally_cooked_app.current_recipe
    return jsonify(recipe)


if __name__ == "__main__":
    app.run()
