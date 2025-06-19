from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_pro_fallback(survey_name, specialty):
    """Generates professional-looking fallback images with better styling"""
    
    # Professional medical color schemes
    colors = {
        "cardiology": ("#2c5282", "#ffffff", "#e53e3e"),  # Blue, white, red accent
        "oncology": ("#553c9a", "#ffffff", "#9f7aea"),    # Purple, white, light purple accent
        "neurology": ("#1a365d", "#ffffff", "#4299e1"),   # Dark blue, white, light blue accent
        "primary_care": ("#2d3748", "#ffffff", "#48bb78"), # Dark gray, white, green accent
        "pediatrics": ("#2b6cb0", "#ffffff", "#f6ad55"),  # Blue, white, orange accent
        "surgery": ("#1a202c", "#ffffff", "#38b2ac"),     # Very dark, white, teal accent
        "general": ("#2d3748", "#ffffff", "#4299e1")      # Professional blue theme
    }
    
    bg_color, text_color, accent_color = colors.get(specialty.lower(), colors["general"])
    
    # Create larger, high-quality image (16:9 ratio)
    img = Image.new('RGB', (1280, 720), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load better fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 56)
        subtitle_font = ImageFont.truetype("arial.ttf", 36)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    # Create gradient-like effect with rectangles
    for i in range(0, 720, 20):
        alpha = max(0, 255 - (i // 3))
        overlay_color = tuple(min(255, c + alpha//8) for c in tuple(int(bg_color[j:j+2], 16) for j in (1, 3, 5)))
    
    # Add professional border and background elements
    draw.rectangle([40, 40, 1240, 680], outline=accent_color, width=4)
    draw.rectangle([60, 60, 1220, 660], outline=text_color, width=2)
    
    # Add accent line
    draw.rectangle([80, 150, 1200, 160], fill=accent_color)
    
    # Wrap survey name if it's too long
    wrapped_title = textwrap.fill(survey_name, width=35)
    title_lines = wrapped_title.split('\n')
    
    # Calculate vertical positioning
    center_x = 640
    title_start_y = 250 if len(title_lines) == 1 else 220
    
    # Draw title (survey name)
    for i, line in enumerate(title_lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        text_width = bbox[2] - bbox[0]
        y_pos = title_start_y + (i * 70)
        draw.text((center_x - text_width//2, y_pos), line, fill=text_color, font=title_font)
    
    # Add specialty subtitle
    specialty_text = f"{specialty.title()} Research Survey"
    bbox = draw.textbbox((0, 0), specialty_text, font=subtitle_font)
    text_width = bbox[2] - bbox[0]
    subtitle_y = title_start_y + len(title_lines) * 70 + 40
    draw.text((center_x - text_width//2, subtitle_y), specialty_text, fill=accent_color, font=subtitle_font)
    
    # Add professional tagline
    tagline = "Healthcare Professional Research Initiative"
    bbox = draw.textbbox((0, 0), tagline, font=small_font)
    text_width = bbox[2] - bbox[0]
    draw.text((center_x - text_width//2, 580), tagline, fill=text_color, font=small_font)
    
    # Add decorative elements
    # Top corners
    draw.polygon([(80, 80), (120, 80), (100, 100)], fill=accent_color)
    draw.polygon([(1160, 80), (1200, 80), (1200, 120)], fill=accent_color)
    
    # Bottom corners  
    draw.polygon([(80, 640), (80, 680), (120, 680)], fill=accent_color)
    draw.polygon([(1200, 600), (1200, 640), (1160, 640)], fill=accent_color)
    
    return img