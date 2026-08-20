"""
Domain-specific prompt banks used for calibration, activation tracing, and capability probing.
"""

CODING_PROMPTS = [
    "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number efficiently using dynamic programming.",
    "Implement a Binary Search Tree (BST) class in Python with `insert`, `search`, and `delete` methods.",
    "Write a Python function to solve the Two Sum problem in O(n) time complexity using a hash map.",
    "Write a function `quicksort(arr)` in Python that sorts an array in-place.",
    "Implement a LRU (Least Recently Used) Cache with `get` and `put` operations in O(1) time.",
    "Write a Python function to check if a binary tree is a valid Binary Search Tree.",
    "Write a regular expression in Python to validate an email address.",
    "Implement Dijkstra's algorithm for finding the shortest paths between nodes in a weighted graph.",
    "Write a Python script using multiprocessing to download a list of URLs concurrently.",
    "Implement a function `merge_k_sorted_lists(lists)` to merge k sorted linked lists in Python.",
    "Write a function to detect and remove a cycle in a singly linked list in Python.",
    "Implement a Trie (Prefix Tree) with `insert`, `search`, and `starts_with` methods.",
    "Write a Python function to find the longest common subsequence (LCS) between two strings.",
    "Write an asynchronous Python function using `asyncio` to fetch data from multiple REST endpoints with retry logic.",
    "Implement a sliding window maximum algorithm in O(n) time using a double-ended queue (deque)."
]

MATH_PROMPTS = [
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per egg. How much in dollars does she make every day at the farmers' market?",
    "A train leaves station A at 60 mph traveling towards station B, which is 300 miles away. Another train leaves station B at 90 mph traveling towards station A at the same time. How long until they meet?",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show all intermediate factorization steps.",
    "If the probability of rain on Saturday is 0.4 and on Sunday is 0.5, and the events are independent, what is the probability that it rains on at least one day over the weekend?",
    "A rectangle has a perimeter of 40 cm. If the length is 4 cm longer than twice the width, find the dimensions and area of the rectangle.",
    "Calculate the integral of (3x^2 + 2x - 5) dx from x = 1 to x = 3.",
    "In a class of 30 students, 18 study Mathematics, 14 study Physics, and 6 study both. How many students study neither Mathematics nor Physics?",
    "A cylinder has a radius of 7 cm and a height of 10 cm. Find its total surface area and volume. (Use pi = 22/7)",
    "If 5 workers can build a wall in 12 days, how many days will it take 8 workers working at the same pace to build the same wall?",
    "Find the sum of the infinite geometric series: 8 + 4 + 2 + 1 + 0.5 + ...",
    "A store offers a 20% discount on an item, and then an additional 10% discount on the discounted price. What is the single equivalent overall discount percentage?",
    "If log_2(x) + log_2(x - 2) = 3, solve for x."
]

TRANSLATION_PROMPTS = [
    "Translate the following English text to Thai: 'Artificial intelligence is rapidly transforming software development by automating routine tasks and enabling developers to focus on creative problem solving.'",
    "Translate the following Thai text to English: 'การเรียนรู้ของเครื่องเป็นสาขาหนึ่งของปัญญาประดิษฐ์ที่ช่วยให้ระบบคอมพิวเตอร์สามารถเรียนรู้และพัฒนาตนเองได้จากข้อมูล'",
    "Translate the following English text to Thai: 'The company reported a 25% increase in annual revenue, exceeding market expectations despite global economic headwinds.'",
    "Translate the following Thai text to English: 'หากคุณต้องการพัฒนาทักษะการเขียนโปรแกรม คุณควรฝึกฝนการเขียนโค้ดและแก้ปัญหาอย่างสม่ำเสมอทุกวัน'",
    "Translate the following English sentence to Thai: 'Quantum computing leverages superposition and entanglement to perform complex calculations exponentially faster than classical computers.'",
    "Translate the following Thai sentence to English: 'ระบบคลาวด์ช่วยให้องค์กรประหยัดต้นทุนด้านโครงสร้างพื้นฐานไอทีและเพิ่มความยืดหยุ่นในการขยายระบบ'",
    "Translate the following English text to Thai: 'Please ensure that all confidential documents are securely stored and encrypted before transmitting over the network.'",
    "Translate the following Thai text to English: 'นโยบายความเป็นส่วนตัวนี้อธิบายถึงวิธีที่เราเก็บรวบรวม ใช้ และเปิดเผยข้อมูลส่วนบุคคลของคุณ'",
    "Translate the following English text to Thai: 'Climate change continues to pose significant threats to global biodiversity and agricultural stability.'",
    "Translate the following Thai text to English: 'แบบจำลองภาษาขนาดใหญ่สามารถเข้าใจและสร้างข้อความที่เป็นธรรมชาติได้อย่างน่าประทับใจ'"
]

GENERAL_PROMPTS = [
    "What is the capital of Australia and what are some of its primary cultural landmarks?",
    "Explain the process of photosynthesis in plants and why it is crucial for life on Earth.",
    "Summarize the key events that led to the Renaissance in Europe.",
    "How does the human immune system distinguish between self and non-self cells?",
    "Explain the difference between renewable and non-renewable energy sources with examples.",
    "What are the primary factors that determine the climate of a geographic region?",
    "Describe the economic concept of supply and demand and how market equilibrium is achieved.",
    "Explain how GPS technology works using satellite triangulation and atomic clocks."
]

def get_prompts_for_capability(capability: str):
    """Retrieve prompt list for a given capability name."""
    cap = capability.lower()
    if "code" in cap or "coding" in cap:
        return CODING_PROMPTS
    elif "math" in cap:
        return MATH_PROMPTS
    elif "translat" in cap:
        return TRANSLATION_PROMPTS
    elif "general" in cap or "all" in cap:
        return GENERAL_PROMPTS
    else:
        return CODING_PROMPTS
