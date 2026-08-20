"""
Domain Dataset: Coding Prompts
Covers Python, Algorithms, Data Structures, OOP, Recursion, and System Design.
"""

CODING_DISCOVERY_PROMPTS = [
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

CODING_VALIDATION_PROMPTS = [
    "Write a Python function `is_palindrome(s)` to check if a string is a palindrome after removing non-alphanumeric characters.",
    "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.",
    "Write a Python function `is_valid_parentheses(s)` using a stack to check if brackets '()[]{}' are valid.",
    "Write a Python function `max_subarray(nums)` using Kadane's algorithm to find the contiguous subarray with the largest sum.",
    "Write an efficient Python function `is_prime(n)` to check if a positive integer n is a prime number.",
    "Write a Python function `reverse_words(s)` that reverses the order of words in a sentence while preserving single spaces.",
    "Write a Python function `merge_sorted_arrays(list1, list2)` that merges two sorted lists into one sorted list in O(n+m).",
    "Write a Python function `element_counts(items)` that returns a dictionary mapping each element to its occurrence count.",
    "Write a Python function `topological_sort(graph)` that performs topological sort on a DAG using Kahn's algorithm.",
    "Write a Python function `coin_change(coins, amount)` that returns the minimum number of coins to make up the given amount."
]

CODING_CALIBRATION_PROMPTS = [
    """Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number efficiently using dynamic programming.
```python
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```""",
    """Write a Python function `two_sum(nums, target)` that returns the indices of two numbers that add up to target in O(n) time.
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []
```""",
    """Write a Python function `is_palindrome(s)` to check if a string is a palindrome after removing non-alphanumeric characters.
```python
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```""",
    """Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```""",
    """Write a Python function `is_valid_parentheses(s)` using a stack to check if brackets '()[]{}' are valid.
```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack
```""",
    """Write a Python function `max_subarray(nums)` using Kadane's algorithm to find the contiguous subarray with the largest sum.
```python
def max_subarray(nums):
    if not nums:
        return 0
    max_so_far = current_max = nums[0]
    for x in nums[1:]:
        current_max = max(x, current_max + x)
        max_so_far = max(max_so_far, current_max)
    return max_so_far
```""",
    """Write an efficient Python function `is_prime(n)` to check if a positive integer n is a prime number.
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```""",
    """Write a Python function `reverse_words(s)` that reverses the order of words in a sentence while preserving single spaces.
```python
def reverse_words(s):
    words = s.strip().split()
    return ' '.join(reversed(words))
```""",
    """Write a Python function `merge_sorted_arrays(list1, list2)` that merges two sorted lists into one sorted list in O(n+m).
```python
def merge_sorted_arrays(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
```""",
    """Write a Python function `element_counts(items)` that returns a dictionary mapping each element to its occurrence count.
```python
def element_counts(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
```"""
]
