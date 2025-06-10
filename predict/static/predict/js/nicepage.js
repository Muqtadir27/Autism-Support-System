document.addEventListener("DOMContentLoaded", function() {
    // Variables to store carousel items and indicators
    const carouselItems = document.querySelectorAll('.u-carousel-item');
    const indicators = document.querySelectorAll('.u-carousel-indicators li');
    let currentIndex = 0;
    const intervalTime = 5000; // Set interval time for auto slide

    // Function to show the slide at the given index
    function showSlide(index) {
        carouselItems.forEach((item, i) => {
            item.classList.remove('u-active');
            indicators[i].classList.remove('u-active');
            if (i === index) {
                item.classList.add('u-active');
                indicators[i].classList.add('u-active');
            }
        });
    }

    // Function to move to the next slide
    function nextSlide() {
        currentIndex = (currentIndex + 1) % carouselItems.length;
        showSlide(currentIndex);
    }

    // Function to move to the previous slide
    function previousSlide() {
        currentIndex = (currentIndex - 1 + carouselItems.length) % carouselItems.length;
        showSlide(currentIndex);
    }

    // Event listener for indicators
    indicators.forEach((indicator, index) => {
        indicator.addEventListener('click', () => {
            currentIndex = index;
            showSlide(index);
        });
    });

    // Auto slide functionality
    setInterval(nextSlide, intervalTime);
});
