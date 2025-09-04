# Strategic Plot Placement Guide

## 🎯 Audience Analysis

### HuggingFace Visitors
- **Who**: ML engineers, researchers, technical evaluators
- **Goals**: Understand the method, evaluate performance, reproduce results
- **Decision**: "Is this technically sound and worth trying?"

### Website Visitors  
- **Who**: Investors, executives, business decision makers
- **Goals**: Understand value proposition, see proof of superiority
- **Decision**: "Should we invest/buy/partner?"

## 📊 Plot Assignments

### HuggingFace Model Card (Technical Proof)

**Primary Plots** (in README.md):
1. **s_curve_1b.png** - Technical innovation showcase
   - Shows precise control to 1% target
   - Demonstrates PI controller working
   - Proves controllability claim

2. **lambda_1b.png** - Adaptive mechanism  
   - Shows automatic tuning without human intervention
   - Bounded dynamics (no instability)
   - Technical sophistication

3. **control_curves_1b.png** - Combined dynamics
   - Full system behavior in one view
   - For researchers who want details

**Supporting Plot** (lower in README):
4. **validation_delta.png** - Performance proof
   - 15.6% perplexity improvement
   - Statistical significance (bootstrap CI)
   - Held-out validation (not training metrics)

### Website (Business Value)

**Hero Section** (above fold):
1. **validation_delta.png** (styled version)
   - Lead with 15.6% improvement
   - Clean table format
   - Immediate value proposition

**Technology Section** (mid-page):
2. **s_curve.png** (simplified/beautified)
   - Shows precision and control
   - "Dial in your performance"
   - Professional engineering

3. **sweep_target_vs_valbpt.png** 
   - Tradeoff curve
   - "Choose your operating point"
   - Business flexibility

**Technical Deep-Dive** (expandable/modal):
4. **lambda_curve.png**
5. **pulse_test.png** 
6. **data_bpt_curve.png**

## 📁 File Organization

```
shannon-control-unit/
├── figures/                    # HuggingFace repo (3-4 plots)
│   ├── s_curve_1b.png         # Control tracking
│   ├── lambda_1b.png          # Adaptation dynamics
│   ├── control_curves_1b.png  # Combined view
│   └── validation_delta.png   # Results table
│
├── assets/figures/            # Full set (all 10 plots)
│   └── [all generated plots]
│
└── web/assets/figures/        # Website (3-4 styled plots)
    ├── hero_validation.png    # Styled validation table
    ├── control_precision.png  # Clean S-curve
    └── tradeoff_curve.png     # Business flexibility

```

## 🎨 Styling Guidelines

### HuggingFace (Keep Original)
- Technical accuracy paramount
- Include axes, units, gridlines
- Full technical annotations
- Standard matplotlib style OK

### Website (Enhance)
- Simplify axes labels
- Larger fonts for readability
- Add subtle gradients/shadows
- Consider dark mode variants
- Highlight key metrics with color

## 📈 Plot Descriptions for Each Context

### HuggingFace Descriptions
Focus on:
- Mathematical precision
- Reproducibility details  
- Technical metrics (BPT, nats, etc.)
- Implementation specifics

### Website Descriptions
Focus on:
- Business impact ("15% more accurate")
- Competitive advantage ("First automatic tuning")
- Cost savings ("No hyperparameter search")
- Flexibility ("Dial in your tradeoff")

## 🚀 Implementation Priority

1. **Immediate** (for HuggingFace):
   - Use current plots as-is
   - Already in `figures/` directory
   - Update README to reference them

2. **Next Sprint** (for website):
   - Create styled versions
   - Simplify for non-technical audience
   - Add to web/assets/

3. **Future** (nice to have):
   - Interactive versions for website
   - Animated progression GIFs
   - Comparison sliders (before/after)

## 📊 Metrics to Emphasize

### Technical (HuggingFace)
- BPT: 3.920 → 3.676 (−6.2%)
- Perplexity: 15.14 → 12.78 (−15.6%)
- S control: 1.0% ± 0.2pp maintained
- Settling time: <200 steps

### Business (Website)
- "15% more accurate predictions"
- "Automatic tuning - no PhD required"
- "Dial in your cost/performance tradeoff"
- "Patent-pending control system"

## ✅ Checklist

### HuggingFace Launch
- [x] s_curve_1b.png in figures/
- [x] lambda_1b.png in figures/
- [x] control_curves_1b.png in figures/
- [x] validation_delta.png in figures/
- [x] All plots < 500KB
- [x] README references correct paths

### Website Launch
- [ ] Create hero_validation.png (styled)
- [ ] Create control_precision.png (simplified)
- [ ] Add to homepage above fold
- [ ] Add to /technology page
- [ ] Mobile responsive versions

## 🎯 Success Metrics

### HuggingFace
- Clear technical understanding
- Reproducible results
- Citations in papers

### Website  
- Quick grasp of value prop
- Contact form submissions
- Investor interest

---

*Remember: HuggingFace proves it works technically. Website proves it matters commercially.*