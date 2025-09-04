# Optimized Images for Shannon Web

## Bell Labs Building Images

### Optimization Summary
- **Original files**: 2 JPG images (9.5MB total)
- **Optimized files**: 7 WebP images (0.5MB total)
- **Savings**: 94.7% file size reduction

### File Structure
```
bell-labs/
├── bell-labs-existing.webp          # 116KB - existing fallback
├── bell-labs-holmdel-large.webp     # 118KB - 1200×900 (desktop)
├── bell-labs-holmdel-medium.webp    # 68KB  - 800×600 (tablet)
├── bell-labs-holmdel-small.webp     # 25KB  - 400×300 (mobile)
├── bell-labs-holmdel-2-large.webp   # 152KB - 900×1200 (portrait)
├── bell-labs-holmdel-2-medium.webp  # 71KB  - 600×800 (portrait)
└── bell-labs-holmdel-2-small.webp   # 20KB  - 300×400 (portrait)
```

### CSS Implementation
The responsive images are implemented in `css/bell-labs-design.css`:
- Large screens (>1024px): `bell-labs-holmdel-large.webp` (118KB)
- Medium screens (768-1024px): `bell-labs-holmdel-medium.webp` (68KB)
- Small screens (<768px): `bell-labs-holmdel-small.webp` (25KB)

### Performance Benefits
1. **94.7% smaller file sizes** - faster page loads
2. **WebP format** - better compression than JPG/PNG
3. **Responsive loading** - appropriate size for each device
4. **Mobile optimized** - background-attachment: scroll for better performance

### Browser Support
- Modern browsers: WebP images
- Fallback: Existing optimized WebP for older browsers
- Quality setting: 85% (optimal balance of size vs quality)

### Generated with
- **cwebp**: Google's WebP encoder
- **Quality**: 85% compression
- **Date**: September 1, 2025