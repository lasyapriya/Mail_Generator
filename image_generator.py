import os
import datetime
import base64
from io import BytesIO
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found. Please set the GEMINI_API_KEY environment variable.")

# Try to import the working Google GenAI client (as used in img_maker.py)
try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
    print("Using new Google GenAI client")
except ImportError:
    try:
        # Fallback to older google-generativeai if available
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("Using older google-generativeai")
    except ImportError:
        GEMINI_AVAILABLE = False
        print("No Google GenAI packages available, using fallback only")


def construct_image_prompt(survey_name, medical_specialty, tone, image_style="professional", include_text=True):
    """
    Constructs a detailed prompt for generating HCP survey campaign images

    Args:
        survey_name: Name of the survey
        medical_specialty: Target medical specialty (e.g., "cardiology", "oncology", "primary care")
        tone: Overall tone ("professional", "modern", "approachable", "clinical")
        image_style: Visual style ("professional", "infographic", "medical_illustration", "clean_modern")
        include_text: Whether to include text overlay on the image
    """

    # Base professional requirements
    base_prompt = f"""
Create a high-quality, professional medical survey campaign image for healthcare professionals.

CAMPAIGN DETAILS:
- Survey Focus: {survey_name}
- Target Specialty: {medical_specialty}
- Visual Tone: {tone}
- Style: {image_style}

VISUAL REQUIREMENTS:
- Dimensions: 16:9 landscape format, suitable for email headers and web banners
- Resolution: High-resolution, crisp and clear
- Color scheme: Professional medical colors (blues, teals, whites, subtle accent colors)
- Professional aesthetic suitable for medical professionals
- Clean, uncluttered design with plenty of white space
"""

    # Specialty-specific visual elements
    specialty_elements = {
        "cardiology": "subtle heart imagery, ECG patterns, stethoscope elements, cardiovascular icons",
        "oncology": "cellular imagery, research lab elements, microscope motifs, hope and healing themes",
        "primary_care": "diverse patient care imagery, family medicine elements, community health themes",
        "neurology": "brain imagery, neural networks, neurological examination tools",
        "pediatrics": "child-friendly colors, pediatric care elements, family-centered themes",
        "psychiatry": "mental health awareness imagery, brain and mind connection themes",
        "emergency_medicine": "urgent care elements, emergency room themes, critical care imagery",
        "surgery": "surgical precision imagery, OR themes, medical precision elements",
        "radiology": "imaging equipment, scan imagery, diagnostic themes",
        "pharmacy": "pharmaceutical elements, medication management themes",
        "general": "universal medical symbols, healthcare collaboration imagery, medical professionalism"
    }

    specialty_visual = specialty_elements.get(medical_specialty.lower(), specialty_elements["general"])
    base_prompt += f"\n- Specialty Elements: Incorporate {specialty_visual}"

    # Style-specific instructions
    style_instructions = {
        "professional": """
PROFESSIONAL STYLE:
- Clean, corporate medical aesthetic
- Subtle gradients and professional typography
- Medical professionals in business attire
- Hospital or clinic environment backgrounds
- Sophisticated color palette with medical blues and whites
- Icons and symbols should be minimal and elegant
""",
        "infographic": """
INFOGRAPHIC STYLE:
- Data visualization elements (charts, graphs, statistics)
- Clear information hierarchy
- Iconography representing survey benefits
- Step-by-step visual flow
- Bold, readable typography
- Engaging data presentation elements
""",
        "medical_illustration": """
MEDICAL ILLUSTRATION STYLE:
- Detailed medical diagrams and anatomical elements
- Scientific accuracy in medical representations
- Educational poster aesthetic
- Medical textbook illustration quality
- Precise, technical visual elements
- Professional medical publication style
""",
        "clean_modern": """
CLEAN MODERN STYLE:
- Minimalist design with lots of white space
- Modern typography and clean lines
- Subtle shadows and depth
- Contemporary medical technology elements
- Fresh, approachable color palette
- Modern healthcare facility aesthetics
"""
    }

    base_prompt += style_instructions.get(image_style.lower(), style_instructions["professional"])

    # Text overlay specifications
    if include_text:
        base_prompt += f"""

TEXT OVERLAY REQUIREMENTS:
- Main headline: "{survey_name}" (prominent, readable typography)
- Subtitle: "Healthcare Professional Survey" or "Medical Research Study"
- Call-to-action element: "Share Your Expertise" or "Join the Research"
- Text should be easily readable against the background
- Use professional, medical-appropriate fonts
- Ensure text contrast meets accessibility standards
"""
    else:
        base_prompt += "\n- Create image without text overlay (background/template only)"

    # Final composition guidelines
    base_prompt += """

COMPOSITION GUIDELINES:
- Center-weighted composition with clear focal point
- Balance between imagery and negative space
- Professional lighting and color temperature
- Avoid overly complex or busy designs
- Ensure scalability from large banners to small email thumbnails
- Create a sense of trust, expertise, and medical authority
- Make it appealing to busy healthcare professionals
- Include subtle elements that suggest collaboration and knowledge sharing

TECHNICAL SPECIFICATIONS:
- High contrast for readability
- Professional color grading
- Sharp, crisp details
- Suitable for both digital and print applications
- Optimized for professional medical communications

Create this medical survey campaign image now with careful attention to professional medical aesthetics and clear visual communication.
"""

    return base_prompt.strip()

def generate_survey_image(survey_name, medical_specialty="general", tone="professional",
                          image_style="professional", include_text=True, save_filename=None):
    from local_fallback import generate_pro_fallback
    
    prompt = construct_image_prompt(survey_name, medical_specialty, tone, image_style, include_text)

    print(f"Generating image for: {survey_name}")
    print(f"Specialty: {medical_specialty} | Style: {image_style} | Tone: {tone}")
    print("-" * 60)

    # Try the working Gemini API pattern first (like img_maker.py)
    if GEMINI_AVAILABLE:
        pil_image, image_filename, image_base64, image_message = _try_new_gemini_api(
            prompt, survey_name, save_filename)
        
        if pil_image is not None:
            return pil_image, image_filename, image_base64, image_message
    
    # Fallback to local generation
    print("Using fallback image generation...")
    return _generate_fallback_image(survey_name, medical_specialty, save_filename)


def _try_new_gemini_api(prompt, survey_name, save_filename=None):
    """
    Attempts to generate image using the new Gemini API (like img_maker.py).
    """
    try:
        # Use the new client pattern from img_maker.py
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        image_message = "AI-generated image created successfully"
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                image_message = part.text
                print("AI Description:", part.text)
                print("-" * 60)
            elif part.inline_data is not None:
                # Create PIL image
                pil_image = PILImage.open(BytesIO(part.inline_data.data))
                
                # Save with custom filename if provided
                if save_filename:
                    image_filename = save_filename
                else:
                    # Generate filename based on survey details
                    safe_name = "".join(c for c in survey_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"survey_image_{safe_name.replace(' ', '_')}_{timestamp}.png"
                
                # Ensure 'static' and 'images' directories exist for Flask app
                output_dir = os.path.join(os.getcwd(), 'static', 'images')
                os.makedirs(output_dir, exist_ok=True)
                full_image_path = os.path.join(output_dir, image_filename)
                pil_image.save(full_image_path)
                print(f"Gemini image saved as: {full_image_path}")

                # Generate base64 string
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return pil_image, image_filename, image_base64, image_message
        
        return None, None, None, "No image generated by Gemini API"
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return None, None, None, f"Gemini API error: {str(e)}"


def _generate_fallback_image(survey_name, medical_specialty, save_filename=None):
    """
    Generates a professional-looking fallback image using local PIL.
    """
    try:
        from local_fallback import generate_pro_fallback
        
        # Generate fallback image
        pil_image = generate_pro_fallback(survey_name, medical_specialty)
        
        # Save with custom filename if provided
        if save_filename:
            image_filename = save_filename
        else:
            # Generate filename based on survey details
            safe_name = "".join(c for c in survey_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"fallback_{safe_name.replace(' ', '_')}_{timestamp}.png"

        # Ensure 'static' and 'images' directories exist for Flask app
        output_dir = os.path.join(os.getcwd(), 'static', 'images')
        os.makedirs(output_dir, exist_ok=True)
        full_image_path = os.path.join(output_dir, image_filename)
        pil_image.save(full_image_path)
        print(f"Fallback image saved as: {full_image_path}")

        # Generate base64 string
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        image_message = "Professional fallback image generated successfully."
        print("Description:", image_message)
        print("-" * 60)

        return pil_image, image_filename, image_base64, image_message

    except Exception as e:
        print(f"Error generating fallback image: {str(e)}")
        return None, None, None, str(e)

# Example 1: Cardiology Survey - Professional Style (This section is for standalone testing,
# and will not run when imported by app.py unless specified)
if __name__ == '__main__':
    print("GENERATING CARDIOLOGY SURVEY IMAGE")
    print("=" * 60)
    generate_survey_image(
            survey_name="Advanced Heart Failure Management Survey 2025",
            medical_specialty="cardiology",
            tone="professional",
            image_style="professional",
            include_text=True
    )

    # Function to generate image template without text (for reusability)
    def generate_image_template(medical_specialty, style="professional"):
        """Generate a reusable template without survey-specific text"""
        return generate_survey_image(
            survey_name="Medical Research Template",
            medical_specialty=medical_specialty,
            tone="professional",
            image_style=style,
            include_text=False,
            save_filename=f"template_{medical_specialty}_{style}.png"
        )