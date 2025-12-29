document.addEventListener('DOMContentLoaded', () => {
    
    // ------------------------------------
    // 1. Smooth Scrolling for Navigation
    // ------------------------------------
    document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            document.querySelector(targetId).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // ------------------------------------
    // 2. Dark Mode Toggle
    // ------------------------------------
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const body = document.body;
    const icon = darkModeToggle.querySelector('i');
    
    // Check local storage for previous preference
    const isDarkMode = localStorage.getItem('darkMode') === 'enabled';

    if (isDarkMode) {
        body.classList.add('dark-mode');
        icon.classList.replace('fa-moon', 'fa-sun');
    }

    darkModeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');

        if (body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
            icon.classList.replace('fa-moon', 'fa-sun');
        } else {
            localStorage.setItem('darkMode', 'disabled');
            icon.classList.replace('fa-sun', 'fa-moon');
        }
    });

    // ------------------------------------
    // 3. Scroll Reveal Animations (The "Energy" Part)
    // ------------------------------------
    
    // Get all sections and cards
    const hiddenElements = document.querySelectorAll('.project-card, .cert-card, .skill-category, .hero-content, .section-padding h3');
    
    // Add the 'hidden' class to them initially via JS so they don't hide if JS is broken
    hiddenElements.forEach((el) => el.classList.add('hidden'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            console.log(entry)
            if (entry.isIntersecting) {
                entry.target.classList.add('show'); // Add 'show' class when in view
            }
        });
    }, {
        threshold: 0.1 // Trigger when 10% of the element is visible
    });

    hiddenElements.forEach((el) => observer.observe(el));
});