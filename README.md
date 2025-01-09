# Literally Cooked 

**Literally Cooked** is a real-time food detection and recipe generation application. Using your webcam, the app identifies ingredients in your kitchen, and with the power of AI, it generates detailed recipes for your cooking adventure. 
> side note web deployment is completely bonkers right now, so if youd like to try it out go [here](https://github.com/ashfelloff/literallycooked/tree/an-attempt-at-something)

## 🚀 Features  

1. **Real-Time Food Detection**  
   - Leverages your webcam to identify ingredients in real-time using the Google Cloud Vision API.  

2. **30-Second Countdown Timer**  
   - Provides a brief timer to help you collect and organize ingredients before detection begins.  

3. **AI-Powered Recipe Generation**  
   - Uses Anthropic's Claude 3 to generate detailed recipes based on detected ingredients.  

4. **Interactive Web Interface**  
   - A user-friendly web interface featuring a live video feed for real-time interaction.  

5. **Comprehensive Recipe Output**  
   - Includes:  
     - Ingredients with quantities  
     - Required tools  
     - Step-by-step instructions  
     - Suggestions for recipe improvement  

## 🛠️ Technical Stack  

- **Backend:** Python 3.9+ with Flask  
- **Frontend:** HTML/CSS/JavaScript  
- **Camera Handling:** OpenCV  
- **AI Services:**  
  - Google Cloud Vision API for object detection  
  - Anthropic's Claude 3 for recipe generation  

## 🌟 Key Technologies
	•	Google Cloud Vision API: Detects objects (ingredients) in the webcam feed.
	•	Anthropic’s Claude 3: Generates AI-powered recipes based on detected ingredients.
	•	OpenCV: Handles live video feed from the webcam.

## KNOWN ISSUE
Web compatibility is actually cooked, and Vercel actually hates me. 

## 📝 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT)  .

## Credits

Developed by [ashfelloff](https://github.com/ashfelloff), shoutout to [Karthik](https://github.com/emergenitro) for helping me get this on the web (hopefully).
