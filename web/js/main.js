// Shannon Main JavaScript
(function() {
  'use strict';

  // Mobile menu toggle
  function initMobileMenu() {
    // Create mobile menu button if doesn't exist
    let menuButton = document.querySelector('.mobile-menu-toggle');
    if (!menuButton) {
      const nav = document.querySelector('.nav-content');
      if (nav) {
        menuButton = document.createElement('button');
        menuButton.className = 'mobile-menu-toggle';
        menuButton.innerHTML = '<span></span><span></span><span></span>';
        menuButton.setAttribute('aria-label', 'Toggle navigation');
        nav.appendChild(menuButton);
      }
    }

    if (menuButton) {
      menuButton.addEventListener('click', () => {
        const navLinks = document.querySelector('.nav-links');
        if (navLinks) {
          navLinks.classList.toggle('mobile-active');
          menuButton.classList.toggle('active');
        }
      });

      // Close menu when clicking outside
      document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-content')) {
          const navLinks = document.querySelector('.nav-links');
          if (navLinks) {
            navLinks.classList.remove('mobile-active');
            menuButton.classList.remove('active');
          }
        }
      });
    }
  }

  // Navigation active state
  function setActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
      link.classList.remove('is-active');
      const href = link.getAttribute('href');
      
      // Handle both exact matches and index.html
      if (href === currentPath || 
          (href === '/' && (currentPath === '/index.html' || currentPath === '/')) ||
          (href === '/results.html' && currentPath === '/results.html') ||
          (href === '/manifesto.html' && currentPath === '/manifesto.html') ||
          (href === '/demo.html' && currentPath === '/demo.html') ||
          (href === '/docs.html' && currentPath === '/docs.html')) {
        link.classList.add('is-active');
      }
    });
  }

  // Scroll progress bar
  function initScrollProgress() {
    // Create progress bar if it doesn't exist
    let progressBar = document.querySelector('.scroll-progress');
    if (!progressBar) {
      progressBar = document.createElement('div');
      progressBar.className = 'scroll-progress';
      document.body.appendChild(progressBar);
    }

    function updateProgress() {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight - windowHeight;
      const scrolled = window.scrollY;
      const progress = documentHeight > 0 ? (scrolled / documentHeight) * 100 : 0;
      progressBar.style.width = Math.min(100, Math.max(0, progress)) + '%';
    }

    window.addEventListener('scroll', updateProgress);
    window.addEventListener('resize', updateProgress);
    updateProgress();
  }

  // Animate counters
  function animateCounters() {
    const counters = [
      { id: 'ctr-f1', target: 0.837, format: 'decimal' },
      { id: 'ctr-domains', target: 3, format: 'integer' },
      { id: 'ctr-winrate', target: 92, format: 'percentage' }
    ];

    counters.forEach(counter => {
      const element = document.getElementById(counter.id);
      if (!element) return;

      // Check if already animated
      if (element.dataset.animated === 'true') return;
      element.dataset.animated = 'true';

      const duration = 1500;
      const steps = 60;
      const increment = counter.target / steps;
      let current = 0;
      let step = 0;

      const timer = setInterval(() => {
        current += increment;
        step++;

        if (step >= steps) {
          current = counter.target;
          clearInterval(timer);
        }

        if (counter.format === 'decimal') {
          element.textContent = current.toFixed(3);
        } else if (counter.format === 'percentage') {
          element.textContent = Math.round(current) + '%';
        } else {
          element.textContent = Math.round(current);
        }
      }, duration / steps);
    });
  }

  // Load evaluation results if available
  async function loadEvaluationResults() {
    try {
      const response = await fetch('/datasets/eval_results.json');
      if (!response.ok) return;
      
      const data = await response.json();
      
      // Update counters with real data if elements exist
      const f1Element = document.getElementById('ctr-f1');
      const domainsElement = document.getElementById('ctr-domains');
      
      if (f1Element && data.results) {
        // Find best F1 score
        let bestF1 = 0;
        data.results.forEach(result => {
          if (result.shannon_f1 > bestF1) {
            bestF1 = result.shannon_f1;
          }
        });
        if (bestF1 > 0) {
          f1Element.textContent = bestF1.toFixed(3);
          f1Element.dataset.animated = 'true'; // Prevent re-animation
        }
      }
      
      if (domainsElement && data.domains) {
        domainsElement.textContent = data.domains.length;
        domainsElement.dataset.animated = 'true';
      }
    } catch (error) {
      // Silently fail if file doesn't exist
      console.debug('Eval results not available');
    }
  }

  // Copy code handler
  function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-button');
    
    copyButtons.forEach(button => {
      button.addEventListener('click', async () => {
        const codeBlock = button.parentElement.querySelector('code');
        if (!codeBlock) return;
        
        try {
          await navigator.clipboard.writeText(codeBlock.textContent);
          button.textContent = 'Copied!';
          setTimeout(() => {
            button.textContent = 'Copy';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });
    });
  }

  // Initialize everything when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  function init() {
    initMobileMenu();
    setActiveNav();
    initScrollProgress();
    
    // Only animate counters on homepage
    if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
      setTimeout(() => {
        animateCounters();
        loadEvaluationResults();
      }, 500);
    }
    
    initCopyButtons();
  }

  // Export for testing
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      setActiveNav,
      initScrollProgress,
      animateCounters
    };
  }
})();