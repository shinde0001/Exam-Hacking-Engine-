/**
 * Exam-Hacking Engine — Client-Side Scripts
 */

document.addEventListener('DOMContentLoaded', () => {
    // Auto-dismiss flash alerts after 5 seconds
    const flashes = document.querySelectorAll('.flash');
    flashes.forEach(flash => {
        setTimeout(() => {
            flash.style.transition = 'opacity 0.5s ease';
            flash.style.opacity = '0';
            setTimeout(() => flash.remove(), 500);
        }, 5000);
    });
});
