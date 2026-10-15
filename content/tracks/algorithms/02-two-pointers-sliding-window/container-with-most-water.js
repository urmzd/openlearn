/**
 * @param {number[]} height
 * @return {number}
 */
var maxArea = function (height) {
  if (height.length === 0) {
    return 0;
  }

  // Goal: Maximize the amount of water
  // Task: find two elements with the greatest span,
  // If we sort, we can make this an O(n^log(n) solution by finding the indices furthest away from eachother )

  // Question, can we make this a greedy algorithm?
  // Answer: Yes

  // Min(start, stop) * len(stop_i - start_i)

  const calculateCapacity = (start, end) => {
    return Math.min(height[start], height[end]) * (end - start);
  };

  let [start, end] = [0, height.length - 1];
  let maxCapacity = calculateCapacity(start, end);

  while (start < end) {
    maxCapacity = Math.max(maxCapacity, calculateCapacity(start, end));
    if (height[start] < height[end]) {
      start++;
    } else {
      end--;
    }
  }

  return maxCapacity;
};
