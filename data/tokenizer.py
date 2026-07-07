from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed

        chars = list(corpus)
        merges = []

        for _ in range(num_merges):

            # Count adjacent pairs
            pairs = []
            for j in range(len(chars) - 1):
                pairs.append((chars[j], chars[j + 1]))

            if not pairs:
                break

            counts = Counter(pairs)

            # Highest frequency, lexicographically smallest if tied
            best_item = min(counts.keys(), key=lambda item: (-counts[item], item))

            # Merge non-overlapping occurrences
            new_chars = []
            j = 0

            while j < len(chars):
                if (
                    j < len(chars) - 1
                    and (chars[j], chars[j + 1]) == best_item
                ):
                    new_chars.append(chars[j] + chars[j + 1])
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1

            chars = new_chars
            merges.append([best_item[0], best_item[1]])

        return merges