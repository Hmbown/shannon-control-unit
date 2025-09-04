#!/usr/bin/env python3
"""
Create professional one-page technical brief PDF for Shannon Control Unit
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

# Colors
DARK_BLUE = HexColor('#1a237e')
ACCENT_BLUE = HexColor('#0052E0')
GRAY = HexColor('#424242')
LIGHT_GRAY = HexColor('#757575')
BACKGROUND = HexColor('#f5f5f5')
SUCCESS_GREEN = HexColor('#2e7d32')

def create_pdf():
    """Create the technical brief PDF"""
    
    # Create PDF
    filename = "scu_technical_brief.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.3*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=DARK_BLUE,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=ACCENT_BLUE,
        spaceBefore=0,
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    tagline_style = ParagraphStyle(
        'Tagline',
        parent=styles['Normal'],
        fontSize=10,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Helvetica-Oblique'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=DARK_BLUE,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=GRAY,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=13
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=GRAY,
        leftIndent=20,
        spaceAfter=4,
        leading=12
    )
    
    highlight_style = ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=11,
        textColor=SUCCESS_GREEN,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=8
    )
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=LIGHT_GRAY,
        alignment=TA_CENTER,
        spaceAfter=4
    )
    
    # Header Section
    elements.append(Paragraph("Shannon Control Unit (SCU)", title_style))
    elements.append(Paragraph("15% Training Efficiency Through Control Theory", subtitle_style))
    elements.append(Paragraph("Patent Pending | Bell Labs Heritage", tagline_style))
    
    # The Problem
    elements.append(Paragraph("<b>The Problem</b>", heading_style))
    elements.append(Paragraph(
        "LLM training wastes billions on hyperparameter sweeps and unstable dynamics. "
        "Manual λ tuning is guesswork that costs months and millions. "
        "Current approaches treat training as an open-loop process with no feedback control.",
        body_style
    ))
    
    # The Solution
    elements.append(Paragraph("<b>The Solution</b>", heading_style))
    elements.append(Paragraph("• Bounded PI controller maintains optimal information ratio S* automatically", bullet_style))
    elements.append(Paragraph("• Eliminates manual hyperparameter search entirely", bullet_style))
    elements.append(Paragraph("• First stable application of control theory to deep learning", bullet_style))
    
    elements.append(Spacer(1, 0.15*inch))
    
    # Proven Results Table
    elements.append(Paragraph("<b>Proven Results (Llama 3.2-1B)</b>", heading_style))
    
    data = [
        ['Metric', 'Baseline', 'SCU', 'Improvement'],
        ['Perplexity', '15.14', '12.78', '-15.6%'],
        ['BPT', '3.920', '3.676', '-6.2%'],
        ['Manual Tuning', 'Required', 'Eliminated', '100% automated'],
    ]
    
    table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), BACKGROUND),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (3, 1), (3, -1), SUCCESS_GREEN),
        ('FONTNAME', (3, 1), (3, -1), 'Helvetica-Bold'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Technical Innovation
    elements.append(Paragraph("<b>Technical Innovation</b>", heading_style))
    elements.append(Paragraph(
        "• <b>MDL-motivated information budget:</b> S = ParamBPT/(DataBPT+ParamBPT)",
        bullet_style
    ))
    elements.append(Paragraph(
        "• <b>Real-time PI control:</b> λ ← λ·exp(-(K<sub>p</sub>·error + K<sub>i</sub>·∫error))",
        bullet_style
    ))
    elements.append(Paragraph(
        "• Maintains S at 1.0% ± 0.2pp throughout training",
        bullet_style
    ))
    
    # Scale & Validation
    elements.append(Paragraph("<b>Scale & Validation</b>", heading_style))
    elements.append(Paragraph("• <b>Proven:</b> Llama 3.2-1B with 15.6% perplexity improvement", bullet_style))
    elements.append(Paragraph("• <b>Validating:</b> 3B model experiments ongoing", bullet_style))
    elements.append(Paragraph("• <b>Seeking partnership:</b> Scale validation for 7B-70B+ models", bullet_style))
    elements.append(Paragraph("• <b>Platform:</b> Works with any transformer architecture", bullet_style))
    
    # Economic Impact (highlighted box)
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("─────────── Economic Impact ───────────", highlight_style))
    elements.append(Paragraph("For $1B in annual training costs → $150M saved", highlight_style))
    elements.append(Paragraph("For $100M training run → $15M saved", highlight_style))
    elements.append(Paragraph("─────────────────────────────────────", highlight_style))
    
    # Why Now?
    elements.append(Paragraph("<b>Why Now?</b>", heading_style))
    elements.append(Paragraph("• Training costs doubling every 6-10 months", bullet_style))
    elements.append(Paragraph("• Compute is the bottleneck to AGI", bullet_style))
    elements.append(Paragraph("• 15% efficiency = decisive competitive advantage", bullet_style))
    
    # Strategic Value
    elements.append(Paragraph("<b>Strategic Value for Partners</b>", heading_style))
    elements.append(Paragraph("• \"Powered by SCU\" exclusive differentiator", bullet_style))
    elements.append(Paragraph("• Immediate ROI on existing infrastructure", bullet_style))
    elements.append(Paragraph("• No hardware changes required", bullet_style))
    
    # The Offer
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph(
        "<b>The Offer:</b> We're selecting one strategic GPU cloud partner for exclusive access. "
        "Partnership window closes Q1 2025.",
        body_style
    ))
    
    # Footer
    elements.append(Spacer(1, 0.2*inch))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("<b>Hunter Bown, Founder & CEO</b>", footer_style))
    elements.append(Paragraph("hunter@shannonlabs.dev | Dallas, TX", footer_style))
    elements.append(Paragraph("10 min from CoreWeave Plano facility", footer_style))
    elements.append(Paragraph("shannonlabs.dev | github.com/Hmbown/shannon-control-unit", footer_style))
    elements.append(Spacer(1, 0.05*inch))
    elements.append(Paragraph(
        "<i>\"My great-grandfather announced the transistor at Bell Labs in 1948.<br/>"
        "SCU brings the same magnitude of innovation to AI infrastructure.\"</i>",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ Technical brief created: {filename}")
    
    return filename

if __name__ == "__main__":
    create_pdf()