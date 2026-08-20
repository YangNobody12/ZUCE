"""Small built-in calibration presets intended for smoke tests and examples."""

PRESETS: dict[str, list[str]] = {
    "coding": [
        "Write a Python function that returns the nth Fibonacci number.",
        "Implement binary search over a sorted list.",
        "Explain the time complexity of merge sort with a short code example.",
        "Fix the off-by-one error in a loop that visits every array element.",
    ],
    "math": [
        "Solve 3x + 7 = 25 and show the steps.",
        "Find the derivative of x squared times sine of x.",
        "What is the probability of two heads in three fair coin flips?",
        "Simplify the fraction 84 over 126.",
    ],
    "translation": [
        "Translate 'Good morning, how are you?' into Thai.",
        "Translate 'Machine learning helps automate decisions.' into French.",
        "Translate 'The meeting starts at noon.' into Japanese.",
        "Translate 'Please close the window.' into Spanish.",
    ],
}

