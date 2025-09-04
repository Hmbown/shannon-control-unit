/**
 * Bell Labs Interactive Experience
 * Inspired by Bell Labs innovation culture
 * Smooth animations and technical precision
 */

// ============================================
// Navigation Enhancement
// ============================================

class BellLabsNav {
  constructor() {
    this.nav = document.querySelector('.nav');
    this.menuBtn = document.querySelector('.nav-menu-btn');
    this.navLinks = document.querySelector('.nav-links');
    this.lastScrollY = 0;
    
    this.init();
  }
  
  init() {
    // Scroll behavior
    window.addEventListener('scroll', () => this.handleScroll());
    
    // Mobile menu
    if (this.menuBtn) {
      this.menuBtn.addEventListener('click', () => this.toggleMenu());
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', (e) => this.smoothScroll(e));
    });
  }
  
  handleScroll() {
    const currentScrollY = window.scrollY;
    
    // Add scrolled class for compact nav
    if (currentScrollY > 50) {
      this.nav.classList.add('scrolled');
    } else {
      this.nav.classList.remove('scrolled');
    }
    
    this.lastScrollY = currentScrollY;
  }
  
  toggleMenu() {
    this.navLinks.classList.toggle('is-open');
    
    // Animate hamburger
    const spans = this.menuBtn.querySelectorAll('span');
    if (this.navLinks.classList.contains('is-open')) {
      spans[0].style.transform = 'rotate(45deg) translateY(6px)';
      spans[1].style.opacity = '0';
      spans[2].style.transform = 'rotate(-45deg) translateY(-6px)';
    } else {
      spans[0].style.transform = '';
      spans[1].style.opacity = '';
      spans[2].style.transform = '';
    }
  }
  
  smoothScroll(e) {
    e.preventDefault();
    const targetId = e.currentTarget.getAttribute('href');
    const targetSection = document.querySelector(targetId);
    
    if (targetSection) {
      const navHeight = this.nav.offsetHeight;
      const targetPosition = targetSection.offsetTop - navHeight;
      
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  }
}

// ============================================
// Scroll Progress Indicator
// ============================================

class ScrollProgress {
  constructor() {
    this.progressBar = document.querySelector('.scroll-progress');
    if (this.progressBar) {
      this.init();
    }
  }
  
  init() {
    window.addEventListener('scroll', () => this.updateProgress());
  }
  
  updateProgress() {
    const winHeight = window.innerHeight;
    const docHeight = document.documentElement.scrollHeight;
    const scrollTop = window.scrollY;
    const scrollPercent = scrollTop / (docHeight - winHeight);
    const finalPercent = Math.min(Math.max(scrollPercent, 0), 1);
    
    this.progressBar.style.width = `${finalPercent * 100}%`;
  }
}

// ============================================
// Intersection Observer for Animations
// ============================================

class AnimateOnScroll {
  constructor() {
    this.elements = document.querySelectorAll('.animate-on-scroll');
    this.init();
  }
  
  init() {
    if ('IntersectionObserver' in window) {
      const options = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      };
      
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      }, options);
      
      this.elements.forEach(el => observer.observe(el));
    }
  }
}

// ============================================
// Hero Section Enhancements
// ============================================

class HeroEffects {
  constructor() {
    this.hero = document.querySelector('.hero');
    this.heroGrid = document.querySelector('.hero-grid');
    this.init();
  }
  
  init() {
    if (!this.hero) return;
    
    // Parallax effect on scroll
    window.addEventListener('scroll', () => this.parallax());
    
    // Mouse move effect on grid
    if (this.heroGrid) {
      this.hero.addEventListener('mousemove', (e) => this.gridFloat(e));
    }
    
    // Add typewriter effect to subtitle
    this.typewriter();
  }
  
  parallax() {
    const scrolled = window.scrollY;
    const rate = scrolled * -0.5;
    
    if (this.hero) {
      const background = this.hero.querySelector('.hero-background');
      if (background) {
        background.style.transform = `translateY(${rate}px)`;
      }
    }
  }
  
  gridFloat(e) {
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    
    const xPos = (clientX / innerWidth - 0.5) * 20;
    const yPos = (clientY / innerHeight - 0.5) * 20;
    
    this.heroGrid.style.transform = `translate(${xPos}px, ${yPos}px)`;
  }
  
  typewriter() {
    const subtitle = this.hero.querySelector('.hero-subtitle');
    if (!subtitle || !subtitle.dataset.text) return;
    
    const text = subtitle.dataset.text;
    subtitle.textContent = '';
    subtitle.style.opacity = '1';
    
    let i = 0;
    const speed = 50;
    
    function type() {
      if (i < text.length) {
        subtitle.textContent += text.charAt(i);
        i++;
        setTimeout(type, speed);
      }
    }
    
    setTimeout(type, 1000);
  }
}

// ============================================
// Metric Counter Animation
// ============================================

class MetricCounters {
  constructor() {
    this.counters = document.querySelectorAll('.metric-value[data-value]');
    this.animated = false;
    this.init();
  }
  
  init() {
    if (!this.counters.length) return;
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !this.animated) {
          this.animateCounters();
          this.animated = true;
        }
      });
    }, { threshold: 0.5 });
    
    this.counters.forEach(counter => observer.observe(counter));
  }
  
  animateCounters() {
    this.counters.forEach(counter => {
      const target = parseFloat(counter.dataset.value);
      const duration = 2000;
      const start = performance.now();
      const isPercentage = counter.dataset.format === 'percentage';
      const isMultiplier = counter.dataset.format === 'multiplier';
      
      const animate = (currentTime) => {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = target * easeOutQuart;
        
        if (isPercentage) {
          counter.textContent = Math.round(current) + '%';
        } else if (isMultiplier) {
          counter.textContent = current.toFixed(1) + 'X';
        } else {
          counter.textContent = current.toFixed(3);
        }
        
        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Final value
          if (isPercentage) {
            counter.textContent = Math.round(target) + '%';
          } else if (isMultiplier) {
            counter.textContent = target.toFixed(1) + 'X';
          } else {
            counter.textContent = target.toFixed(3);
          }
        }
      };
      
      requestAnimationFrame(animate);
    });
  }
}

// ============================================
// Code Block Enhancement
// ============================================

class CodeBlockEnhancer {
  constructor() {
    this.codeBlocks = document.querySelectorAll('.code-block');
    this.init();
  }
  
  init() {
    this.codeBlocks.forEach(block => {
      // Add copy button
      this.addCopyButton(block);
      
      // Add line numbers
      this.addLineNumbers(block);
      
      // Add language label
      this.addLanguageLabel(block);
    });
  }
  
  addCopyButton(block) {
    const button = document.createElement('button');
    button.className = 'code-copy-btn';
    button.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    button.title = 'Copy code';
    
    button.addEventListener('click', () => {
      const code = block.querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        setTimeout(() => {
          button.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
        }, 2000);
      });
    });
    
    block.appendChild(button);
  }
  
  addLineNumbers(block) {
    const pre = block.querySelector('pre');
    if (!pre) return;
    
    const code = pre.querySelector('code');
    if (!code) return;
    
    const lines = code.textContent.split('\n');
    if (lines.length <= 1) return;
    
    const lineNumbers = document.createElement('div');
    lineNumbers.className = 'line-numbers';
    
    lines.forEach((_, index) => {
      const lineNumber = document.createElement('span');
      lineNumber.textContent = index + 1;
      lineNumbers.appendChild(lineNumber);
    });
    
    pre.insertBefore(lineNumbers, code);
    pre.classList.add('line-numbers-mode');
  }
  
  addLanguageLabel(block) {
    const lang = block.dataset.lang;
    if (!lang) return;
    
    const label = document.createElement('span');
    label.className = 'code-lang-label';
    label.textContent = lang.toUpperCase();
    block.appendChild(label);
  }
}

// ============================================
// Table Enhancement
// ============================================

class TableEnhancer {
  constructor() {
    this.tables = document.querySelectorAll('.data-table');
    this.init();
  }
  
  init() {
    this.tables.forEach(table => {
      // Make responsive
      this.makeResponsive(table);
      
      // Add sorting
      this.addSorting(table);
      
      // Add search
      this.addSearch(table);
    });
  }
  
  makeResponsive(table) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  }
  
  addSorting(table) {
    const headers = table.querySelectorAll('th[data-sortable]');
    
    headers.forEach(header => {
      header.style.cursor = 'pointer';
      header.addEventListener('click', () => this.sortTable(table, header));
      
      // Add sort indicator
      const indicator = document.createElement('span');
      indicator.className = 'sort-indicator';
      indicator.innerHTML = ' ↕';
      header.appendChild(indicator);
    });
  }
  
  sortTable(table, header) {
    const column = Array.from(header.parentNode.children).indexOf(header);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isAscending = header.classList.contains('sort-asc');
    
    rows.sort((a, b) => {
      const aValue = a.children[column].textContent;
      const bValue = b.children[column].textContent;
      
      // Try to parse as number
      const aNum = parseFloat(aValue);
      const bNum = parseFloat(bValue);
      
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return isAscending ? bNum - aNum : aNum - bNum;
      }
      
      // Sort as string
      return isAscending ? 
        bValue.localeCompare(aValue) : 
        aValue.localeCompare(bValue);
    });
    
    // Update classes
    table.querySelectorAll('th').forEach(th => {
      th.classList.remove('sort-asc', 'sort-desc');
    });
    header.classList.add(isAscending ? 'sort-desc' : 'sort-asc');
    
    // Update indicator
    const indicator = header.querySelector('.sort-indicator');
    indicator.innerHTML = isAscending ? ' ↓' : ' ↑';
    
    // Reorder rows
    rows.forEach(row => tbody.appendChild(row));
  }
  
  addSearch(table) {
    const searchBox = document.createElement('input');
    searchBox.type = 'text';
    searchBox.className = 'table-search';
    searchBox.placeholder = 'Search table...';
    
    searchBox.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const rows = table.querySelectorAll('tbody tr');
      
      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      });
    });
    
    table.parentNode.insertBefore(searchBox, table);
  }
}

// ============================================
// Bell Labs Grid Animation
// ============================================

class BellLabsGrid {
  constructor() {
    this.canvas = document.getElementById('bell-labs-grid');
    if (!this.canvas) return;
    
    this.ctx = this.canvas.getContext('2d');
    this.resize();
    this.init();
  }
  
  init() {
    window.addEventListener('resize', () => this.resize());
    this.animate();
  }
  
  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }
  
  animate() {
    const gridSize = 40;
    const time = Date.now() * 0.0001;
    
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = 'rgba(0, 82, 224, 0.03)';
    this.ctx.lineWidth = 1;
    
    // Draw vertical lines
    for (let x = 0; x < this.canvas.width; x += gridSize) {
      const offset = Math.sin(time + x * 0.01) * 2;
      this.ctx.beginPath();
      this.ctx.moveTo(x + offset, 0);
      this.ctx.lineTo(x + offset, this.canvas.height);
      this.ctx.stroke();
    }
    
    // Draw horizontal lines
    for (let y = 0; y < this.canvas.height; y += gridSize) {
      const offset = Math.cos(time + y * 0.01) * 2;
      this.ctx.beginPath();
      this.ctx.moveTo(0, y + offset);
      this.ctx.lineTo(this.canvas.width, y + offset);
      this.ctx.stroke();
    }
    
    requestAnimationFrame(() => this.animate());
  }
}

// ============================================
// Initialize Everything
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  // Core components
  new BellLabsNav();
  new ScrollProgress();
  new AnimateOnScroll();
  new HeroEffects();
  new MetricCounters();
  new CodeBlockEnhancer();
  new TableEnhancer();
  new BellLabsGrid();
  
  // Add CSS file if not already loaded
  if (!document.querySelector('link[href*="bell-labs-design.css"]')) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/css/bell-labs-design.css';
    document.head.appendChild(link);
  }
  
  // Add grid background to body
  if (!document.querySelector('.grid-background')) {
    const gridBg = document.createElement('div');
    gridBg.className = 'grid-background';
    document.body.insertBefore(gridBg, document.body.firstChild);
  }
  
  // Add IBM Plex fonts
  if (!document.querySelector('link[href*="fonts.googleapis"]')) {
    const preconnect1 = document.createElement('link');
    preconnect1.rel = 'preconnect';
    preconnect1.href = 'https://fonts.googleapis.com';
    document.head.appendChild(preconnect1);
    
    const preconnect2 = document.createElement('link');
    preconnect2.rel = 'preconnect';
    preconnect2.href = 'https://fonts.gstatic.com';
    preconnect2.crossOrigin = true;
    document.head.appendChild(preconnect2);
    
    const fontLink = document.createElement('link');
    fontLink.href = 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap';
    fontLink.rel = 'stylesheet';
    document.head.appendChild(fontLink);
  }
});

// Export for use in other scripts
window.BellLabs = {
  nav: BellLabsNav,
  scrollProgress: ScrollProgress,
  animateOnScroll: AnimateOnScroll,
  heroEffects: HeroEffects,
  metricCounters: MetricCounters,
  codeBlockEnhancer: CodeBlockEnhancer,
  tableEnhancer: TableEnhancer,
  grid: BellLabsGrid
};