/**
 * @param {string} s
 * @return {number}
 */
var numDecodings = function (s) {
  return recurse(s);
};

function recurse(s, i = 0, last1 = 1, last2 = s.length) {
  if (i >= s.length) {
    return last1;
  }

  // If 1-9 then the current is the last number of ways to decode
  let newCurrent = s[i] == 0 ? 0 : last1;
  if (i > 0) {
    // If in 10s or 20s
    if (s[i - 1] == 1 || (s[i - 1] == 2 && s[i] < 7)) {
      // number of way to get the one number + the number of ways to get the two numbers.
      newCurrent += last2;
    }
  }

  return recurse(s, i + 1, newCurrent, last1);
}
