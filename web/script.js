// Shannon Labs - Mobile Navigation and Smooth Scroll

(function() {
  'use strict';
  
  // Mobile menu toggle
  const mobileToggle = document.querySelector('.mobile-toggle');
  const nav = document.querySelector('.nav');
  
  if (mobileToggle && nav) {
    mobileToggle.addEventListener('click', function() {
      nav.classList.toggle('active');
      mobileToggle.classList.toggle('active');
    });
    
    // Close menu when clicking a link
    const navLinks = nav.querySelectorAll('a');
    navLinks.forEach(link => {
      link.addEventListener('click', function() {
        nav.classList.remove('active');
        mobileToggle.classList.remove('active');
      });
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
      if (!nav.contains(e.target) && !mobileToggle.contains(e.target)) {
        nav.classList.remove('active');
        mobileToggle.classList.remove('active');
      }
    });
  }
  
  // Smooth scroll for anchor links
  const anchorLinks = document.querySelectorAll('a[href^="#"]');
  
  anchorLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        
        // Calculate offset for sticky header
        const header = document.querySelector('.header');
        const headerHeight = header ? header.offsetHeight : 0;
        const targetPosition = targetElement.offsetTop - headerHeight - 20;
        
        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });
        
        // Update URL without triggering scroll
        history.pushState(null, null, targetId);
      }
    });
  });
  
  // Add active state to nav links based on scroll position
  const sections = document.querySelectorAll('section[id]');
  const navItems = document.querySelectorAll('.nav-link');
  
  function updateActiveNav() {
    const scrollY = window.pageYOffset;
    
    sections.forEach(section => {
      const sectionHeight = section.offsetHeight;
      const sectionTop = section.offsetTop - 100;
      const sectionId = section.getAttribute('id');
      
      if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
        navItems.forEach(item => {
          item.classList.remove('active');
          if (item.getAttribute('href') === `#${sectionId}`) {
            item.classList.add('active');
          }
        });
      }
    });
  }
  
  if (sections.length > 0 && navItems.length > 0) {
    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav(); // Set initial state
  }
  
  // Add header shadow on scroll
  const header = document.querySelector('.header');
  
  if (header) {
    window.addEventListener('scroll', function() {
      if (window.scrollY > 10) {
        header.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
      } else {
        header.style.boxShadow = 'none';
      }
    });
  }
})();