# img_generation.py

import google.generativeai as genai
from PIL import Image as PILImage
from io import BytesIO
import base64
import os
import datetime
import logging
from dotenv import load_dotenv
import openai
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize APIs
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY in .env")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in .env")

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY

# Mapping of specialties to fallback images
specialty_fallback_images = {
    "cardiology": "https://plus.unsplash.com/premium_photo-1718349374495-b1d09644f973?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "general": "https://plus.unsplash.com/premium_photo-1673953509975-576678fa6710?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8ZG9jdG9yfGVufDB8fDB8fHww",
    "oncology": "https://images.unsplash.com/photo-1586773831566-893f3648d9f7?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y2FuY2VyJTIwZmlnaHRpbmclMjBkb2N0b3J8ZW58MHx8MHx8fDA%3D",
    "neurology": "https://plus.unsplash.com/premium_photo-1722684650552-bfaf747e3c9f?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "pediatrics": "https://plus.unsplash.com/premium_photo-1681995280561-d11f4b5ba8f3?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "psychiatrist": "https://plus.unsplash.com/premium_photo-1682148380543-a6fd43607dea?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "emergency medicine": "https://plus.unsplash.com/premium_photo-1679615911754-fafb37c81998?q=80&w=1863&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "surgery": "https://images.unsplash.com/photo-1640876777012-bdb00a6323e2?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "radiology": "https://plus.unsplash.com/premium_photo-1664302322745-7337f0902e13?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "pharmacy": "https://images.unsplash.com/photo-1603706580932-6befcf7d8521?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
}

def construct_image_prompt(survey_name, medical_specialty, tone, image_style="professional", include_text=True):
    base_prompt = f"""
Create a high-quality medical survey email banner image:

- Title: {survey_name}
- Specialty: {medical_specialty.title()}
- Tone: {tone.title()}
- Style: {image_style.title()}

VISUALS:
- Aspect Ratio: 16:9 (banner style)
- Resolution: 1024x576
- Professional, clean medical look.

Include subtle relevant visuals for {medical_specialty}.
"""
    return base_prompt.strip()

def fetch_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    pil_image = PILImage.open(BytesIO(response.content))
    pil_image = pil_image.resize((1024, 576), PILImage.LANCZOS)
    return pil_image

def generate_survey_image(survey_name, medical_specialty="general", tone="professional",
                          image_style="professional", include_text=True):
    try:
        prompt = construct_image_prompt(survey_name, medical_specialty, tone, image_style, include_text)
        logger.info(f"Generating image with prompt: {prompt[:100]}...")

        # Generate image using DALL-E 3
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"
        )

        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        pil_image = PILImage.open(BytesIO(image_response.content))
        pil_image = pil_image.resize((1024, 576), PILImage.LANCZOS)

        safe_name = "".join(c for c in survey_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/images/survey_{safe_name}_{timestamp}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pil_image.save(filename)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return pil_image, filename, image_base64, "Generated successfully"

    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")

        # Fallback to specific image based on specialty
        fallback_url = specialty_fallback_images.get(medical_specialty.lower(), specialty_fallback_images["general"])
        try:
            pil_image = fetch_image_from_url(fallback_url)

            safe_name = "".join(c for c in survey_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"static/images/fallback_{safe_name}_{timestamp}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pil_image.save(filename)

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return pil_image, filename, image_base64, "Fallback image used"
        except Exception as fallback_error:
            logger.error(f"Fallback image download failed: {str(fallback_error)}")
            return None, None, None, f"Image generation failed: {str(e)}"
