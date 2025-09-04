// Scroll-triggered reveals. Uses GSAP if available, otherwise IO fallback.
(function(){
  const useGSAP = typeof window.gsap !== 'undefined';
  const els = Array.from(document.querySelectorAll('.reveal, .proof-card, .feature, .figure-container'));
  if (useGSAP && window.ScrollTrigger) {
    els.forEach(el => {
      el.style.opacity = 0;
      window.gsap.fromTo(el, {y: 16, opacity: 0}, {y: 0, opacity: 1, duration: 0.6, ease: 'power2.out', scrollTrigger: {trigger: el, start: 'top 80%'}});
    });
  } else {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('is-visible');
          io.unobserve(e.target);
        }
      }
    }, {rootMargin: '0px 0px -10% 0px'});
    els.forEach(el => {
      el.classList.add('reveal');
      io.observe(el);
    });
  }
})();
