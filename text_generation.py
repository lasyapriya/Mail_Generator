import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found. Please set the GEMINI_API_KEY environment variable.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel("gemini-1.5-flash")

def construct_prompt(survey_name, compensation, loi, tone, educational_info=None):
    """
    Constructs a detailed prompt for generating HCP survey invitation emails
    """
    base_prompt = f"""
You are an expert medical communications specialist with 15+ years of experience crafting compelling survey invitations for Healthcare Professionals (HCPs). You understand their challenges, time constraints, and motivations.

SURVEY DETAILS:
- Survey Name: {survey_name}
- Compensation: ${compensation} USD
- Duration: {loi} minutes
- Target Tone: {tone}
"""

    if tone.lower() == "educational":
        if not educational_info:
            raise ValueError("Educational tone selected, but no educational info provided.")
        base_prompt += f"- Educational Focus: {educational_info}\n"

    tone_instructions = {
        "formal": """
TONE GUIDELINES - FORMAL:
- Use professional, respectful language appropriate for medical colleagues
- Maintain clinical terminology where relevant
- Structure with clear, organized paragraphs
- Use formal salutations and closings
- Emphasize professional value and scientific merit
- Reference peer collaboration and medical community contribution
""",
        "casual": """
TONE GUIDELINES - CASUAL:
- Use friendly, approachable language while maintaining professionalism
- Include conversational elements ("We'd love to hear from you")
- Keep sentences shorter and more direct
- Use inclusive language ("we", "together")
- Balance friendliness with respect for expertise
""",
        "educational": """
TONE GUIDELINES - EDUCATIONAL:
- Position survey as a learning and knowledge-sharing opportunity
- Explain broader context and importance of research topic
- Highlight contribution to medical advancement
- Include relevant background on subject matter
- Emphasize educational value for participants
- Use evidence-based language
""",
        "compelling": """
TONE GUIDELINES - COMPELLING:
- Create urgency without being pushy
- Emphasize critical importance of their perspective
- Highlight impact on patient care/medical practice
- Use action-oriented language
- Stress limited opportunity and exclusivity
- Include credibility indicators
"""
    }

    base_prompt += tone_instructions.get(tone.lower(), tone_instructions["formal"])

    base_prompt += """
EMAIL REQUIREMENTS:
1. SUBJECT LINE: Engaging, mentions survey topic and compensation
2. OPENING: Professional greeting acknowledging expertise
3. PURPOSE: Clear explanation of survey objective and relevance
4. VALUE: Why participation matters (beyond compensation)
5. LOGISTICS: Duration ({loi} mins), compensation (${compensation}), participation method
6. CREDIBILITY: Establish trust and legitimacy
7. CALL-TO-ACTION: Clear next steps to participate
8. CLOSING: Professional sign-off with contact info

CONTENT SPECIFICATIONS:
- Length: 150-250 words
- Include survey name naturally
- Address HCP concerns (time, relevance, legitimacy)
- Use appropriate medical terminology
- Mention compensation naturally
- Include exclusivity and professional recognition
- Ensure confidentiality/anonymity is mentioned

AVOID:
- Generic language
- Aggressive sales tactics
- Minimizing time constraints
- Unclear participation instructions

OUTPUT FORMAT:
Subject Line: [Subject]
Email Body:
[Body]
Signature:
[Signature]
"""
    
    return base_prompt.strip()

def generate_email(survey_name, compensation, loi, tone, educational_info=None):
    """
    Generates email content for HCP survey invitation
    """
    try:
        prompt = construct_prompt(survey_name, compensation, loi, tone, educational_info)
        response = text_model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, f"Error generating email: {str(e)}"