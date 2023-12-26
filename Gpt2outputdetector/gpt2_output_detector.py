import requests
import re


class GPT2Detector:
    def __init__(self, base_url = 'http://localhost:8080'):
        self.base_url = base_url

    def text_predict(self, document):
        url = f'{self.base_url}/'
        headers = {
            'accept': '*/*',
            'Content-Type': 'text/plain'
        }
        response = requests.post(url, headers=headers, data=document)
        return response.json()

    def split_code(self, code_block, min_words=40, max_words=80):
        # Define the regex pattern for newline or 3+ space characters
        pattern = r'(\n| {3,})'

        # Split the code block using the regex pattern
        code_parts = re.split(pattern, code_block)
        
        # Initialize variables for the final list of code parts and a temporary code part
        final_code_parts = []
        temp_code_part = ""

        # Iterate through the code parts
        for part in code_parts:
            temp_code_part += part
            words = len(re.findall(r'\b\w+\b', temp_code_part))
            
            # If the temporary code part reaches the maximum word count, add it to the final list
            if words >= max_words:
                final_code_parts.append(temp_code_part.strip())
                temp_code_part = ""

        # Check the total words in the code_block
        total_words = len(re.findall(r'\b\w+\b', code_block))

        # If the total word count is less than the minimum, return the entire code block as a single element list
        if total_words < min_words:
            return [code_block.strip()]

        # Add the remaining code part if it has any words
        if len(re.findall(r'\b\w+\b', temp_code_part)) > 0:
            final_code_parts.append(temp_code_part.strip())
        # print(final_code_parts)
        # print("SPLITED CODE LENGTH: ",len(final_code_parts))
        return final_code_parts

    def split_code_block(self, code_block):
        code_block = str(code_block)
        # Define the regular expression pattern to match
        pattern = r"(?:\bdef\b|\()\s*\n|\n\s*"

        # Split the code block using the regular expression pattern
        parts = re.split(pattern, code_block)

        # Combine the parts into strings with each containing less than 250 words or 600 characters maximum
        result = []
        current_part = ""
        for part in parts:
            if len(current_part.split()) + len(part.split()) > 125 or len(current_part) + len(part) > 500:
                result.append(current_part)
                current_part = part
            else:
                current_part += part
        if current_part:
            result.append(current_part)

        return [elem for elem in result if elem]

    def text_predict_tuple(self, document: str):
        fake_prob = []
        real_prob = []
        result = None
        all_token = 0
        used_token = 0
        # print(document)
        queries = self.split_code(document)
        for i in range(len(queries)):
            # print(queries[i])
            result = self.text_predict(queries[i])
            fake_prob.append(result["fake_probability"])
            real_prob.append(result["real_probability"])
            all_token += result["all_tokens"]
            used_token += result["used_tokens"]
        fake_prob = sum(fake_prob)/len(fake_prob)
        real_prob = sum(real_prob)/len(real_prob)
        if all_token != used_token: print("| INFO | --------- NOT ALL TOKEN USED | ALL:", all_token,"USED: ", used_token)
        return 1 if real_prob > fake_prob else 0, {'all_tokens': all_token, 'used_tokens': used_token, 'real_probability': real_prob, 'fake_probability': fake_prob}


if __name__ == '__main__':
    gpt2_api = GPT2Detector()

    print("ok")
    # print(result)

    text = """class Solution:
	def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:

			graph = [[] for _ in range(n)]
			self.createGraph(graph, times)

			minDist = [float('inf') for _ in range(n)]
			minDist[k - 1] = 0

			min_heap = MinHeap([(idx, float('inf')) for idx in range(n)])
			min_heap.update(k - 1, 0)

			while not min_heap.check():

				vertex, cost = min_heap.remove()

				if cost == float('inf'):
					break

				for i in graph[vertex]:
					node, dist = i

					newDist = dist + cost
					if newDist < minDist[node]:
						minDist[node] = newDist
						min_heap.update(node, newDist)

			return -1 if float('inf') in minDist else max(minDist)

		def createGraph(self, graph, times):
			for time in times:
				source, dest, cost = time
				graph[source - 1].append((dest - 1, cost))


	class MinHeap:
		def __init__(self, arr):
			self.vertex = {idx: idx for idx in range(len(arr))}
			self.heap = self.buildHeap(arr)

		def check(self):
			return len(self.heap) == 0

		def buildHeap(self, arr):
			parentIdx = (len(arr) - 2) // 2
			for i in reversed(range(parentIdx + 1)):
				self.siftDown(i, len(arr) - 1, arr)
			return arr

		def remove(self):
			if self.check():
				return

			self.swapValues(0, len(self.heap) - 1, self.heap)
			idx, node = self.heap.pop()
			del self.vertex[idx]

			self.siftDown(0, len(self.heap) - 1, self.heap)
			return idx, node

		def siftDown(self, idx, length, arr):
			idxOne = idx * 2 + 1
			while idxOne <= length:
				idxTwo = idx * 2 + 2 if idx * 2 + 2 <= length else -1
				if idxTwo != -1 and arr[idxOne][1] > arr[idxTwo][1]:
					swap = idxTwo
				else:
					swap = idxOne

				if arr[swap][1] < arr[idx][1]:
					self.swapValues(swap, idx, arr)
					idx = swap
					idxOne = idx * 2 + 1
				else:
					return

		def swapValues(self, i, j, arr):
			self.vertex[arr[i][0]] = j
			self.vertex[arr[j][0]] = i
			arr[i], arr[j] = arr[j], arr[i]

		def siftUp(self, curr_idx):
			parentIdx = (curr_idx - 1) // 2
			while curr_idx > 0:
				if self.heap[curr_idx][1] < self.heap[parentIdx][1]:
					self.swapValues(curr_idx, parentIdx, self.heap)
					curr_idx = parentIdx
					parentIdx = (curr_idx - 1) // 2
				else:
					return

		def update(self, idx, value):
			curr_idx = self.vertex[idx]
			self.heap[curr_idx] = (idx, value)
			self.siftUp(curr_idx)"""

    t2 = """#Ref:https://bit.ly/3qW9FIX
import operator
def strand_sort(arr: list, reverse: bool = False, solution: list = None) -> list:
    _operator = operator.lt if reverse else operator.gt
    solution = solution or []
    if not arr:
        return solution
    sublist = [arr.pop(0)]
    for i, item in enumerate(arr):
        if _operator(item, sublist[-1]):
            sublist.append(item)
            arr.pop(i)
    #  merging sublist into solution list
    if not solution:
        solution.extend(sublist)
    else:
        while sublist:
            item = sublist.pop(0)
            for i, xx in enumerate(solution):
                if not _operator(item, xx):
                    solution.insert(i, item)
                    break
            else:
                solution.append(item)
    strand_sort(arr, reverse, solution)
    return solution
lst = [4, 3, 5, 1, 2]
print(""\nOriginal list:"")
print(lst)
print(""After applying  Strand sort the said list becomes:"")
print(strand_sort(lst))
lst = [5, 9, 10, 3, -4, 5, 178, 92, 46, -18, 0, 7]
print(""\nOriginal list:"")
print(lst)
print(""After applying Strand sort the said list becomes:"")
print(strand_sort(lst))
lst = [1.1, 1, 0, -1, -1.1, .1]
print(""\nOriginal list:"")
print(lst)
print(""After applying Strand sort the said list becomes:"")
print(strand_sort(lst))","def strand_sort(arr):
    if len(arr) == 1:
        return arr
    
    sublist = [arr[0]]
    rest = arr[1:]
    for i in range(len(rest)):
        if rest[i] > sublist[-1]:
            sublist.append(rest[i])
        else:
            self.head = oddList
    
    return merge(strand_sort(rest), sublist)
def merge(a, b):
    i, j = 0, 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result += a[i:]
    result += b[j:]
    return result
unsorted_list = [5, 9, 2, 4, 7, 6, 1, 3, 8]
sorted_list = strand_sort(unsorted_list)
print(sorted_list)"""
    # words = len(re.findall(r'\w+', text))
    # print("Number of words:", words)
    print(text.split())
    if "sublist[-1]:" in text.split():
        print(text.split().index('sublist[-1]:'))
    words = gptzero_api.split_code(text+"\n"+text, 100, 200)
    print(words)
    print(len(words))
    total = 0
    for i, part in enumerate(words):
        print(f"Part {i+1}: {part}\n")
        word_num = len(part.split())
        total += word_num
        print("---WORDNUM---", i, word_num)
        # print(part)
        # print("---TOTAL---", total)

    print(gptzero_api.split_code_block(text))
    print(len(gptzero_api.split_code_block(text)))
    # result = gptzero_api.text_predict_tuple(text)
    # print(result)
