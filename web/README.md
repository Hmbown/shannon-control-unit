# Shannon Labs - SCU Landing Page

Single-page, results-first landing for Shannon Control Unit (SCU) - PI-controlled MDL for LLM regularization.

## Quick Start

### Local Development
```bash
# Start local server
python3 -m http.server 8000
# Visit http://localhost:8000
```

### Files
- `index.html` - Main landing page (SCU-only content)
- `styles.css` - Dark theme, responsive, system fonts
- `script.js` - Mobile menu & smooth scroll
- `assets/favicon.svg` - S⟂ monogram
- `robots.txt` & `sitemap.xml` - SEO

## Deployment

### GitHub Pages
1. Push to GitHub repo
2. Settings → Pages → Source: main branch, /web folder
3. Custom domain: shannonlabs.dev

### Cloudflare Pages
```bash
# Connect GitHub repo
# Build command: (none - static site)
# Build output: /web
# Deploy
```

### Netlify
```bash
# Drag & drop web folder
# Or connect GitHub repo
# Publish directory: web
```

### Vercel (IMPORTANT - Use Hunter's Personal Account)
```bash
# CRITICAL: Deploy to hunterbown account, NOT aurorabess
vercel switch
# Select "Hunter Bown (hunterbown)"
vercel --prod --yes
```

## Performance Targets
- Lighthouse Mobile: ≥95 Performance, ≥95 Accessibility, ≥95 SEO
- Total size: <60KB (excluding plot images)
- No external dependencies

## Content Updates

### Update validation metrics
Edit lines 89-101 in `index.html`:
```html
<td>3.920</td>  <!-- Base BPT -->
<td>3.676</td>  <!-- SCU BPT -->
<td>−6.2%</td> <!-- Delta -->
```

### Add plot images
Place in `/assets/figures/`:
- `s_curve.png` - S(t) control dynamics
- `lambda_curve.png` - λ(t) bounded behavior

## Key Features
- **Results-first**: ΔBPT −6.2% hero metric
- **Evidence-driven**: Table + plots above fold
- **Reproducible**: HF models + Colab links
- **Investor CTA**: "Request the deck" mailto
- **Minimal**: No NCD/security content

## Testing
```bash
# Check accessibility
# Use browser DevTools Lighthouse
# Mobile: ≥95/95/95 required

# Test responsive
# Chrome DevTools → Toggle device toolbar
# Test at 375px, 768px, 1440px widths
```

## Legacy Note
Inspired by Claude Shannon & Bell Labs tradition. Full NCD/anomaly detection archived in separate repo.