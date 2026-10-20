#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) (a < b ? a : b)
#define MIN3(a, b, c) (MIN(a, b) < c ? MIN(a, b) : c)

typedef struct StringNode {
  char data;
  struct StringNode *prev;
} string_node;

string_node *string_node_new(char data, string_node *prev) {
  string_node *node = malloc(sizeof(string_node));
  node->data = data;
  node->prev = prev;

  return node;
}

// Prepend TARGET onto SOURCE.
void string_prepend(char *source, const char *target) {
  // Determine additional size to add.
  size_t len = strlen(target);
  // Determine the amount of memory to add.
  memmove(source + len, source, strlen(source) + 1);
  // Ensure TARGET comes before SOURCE.
  memcpy(source, target, len);
}

void string_node_printf(string_node *node, int m, int n) {
  string_node *current = node;
  char *output_string = malloc(m + n + 1);

  while (current != NULL) {
    string_prepend(output_string, &current->data);
    current = current->prev;
  }

  printf("%s", output_string);
  printf("\n");
  return;
}
/**
 * The following solution occurs in O(n^2) time.
 *
 * The reasoning is as follows:
 *   - Pointing to a character instead of copying multiple characters cost O(1)
 *
 * The solution can be extended to O((m+n)k) complexity by
 * allocating space only for cells |i-j| <= k.
 *
 * Similarly, we can reduce our space complexity by freeing all
 * memory which was not involved in the the calculation of the current optimal.
 */
void align(char *S, char *T, int m, int n) {
  int A[m + 1][n + 1];
  string_node *top[m + 1][n + 1];
  string_node *bottom[m + 1][n + 1];

  A[0][0] = 0;

  top[0][0] = string_node_new('\0', NULL);
  bottom[0][0] = string_node_new('\0', NULL);

  for (int i = 1; i <= m; i++) {
    A[i][0] = i;
    top[i][0] = string_node_new(S[i - 1], top[i - 1][0]);
    bottom[i][0] = string_node_new('-', bottom[i - 1][0]);
  }

  for (int j = 1; j <= n; j++) {
    A[0][j] = j;

    bottom[0][j] = string_node_new(T[j - 1], bottom[0][j - 1]);
    top[0][j] = string_node_new('-', top[0][j - 1]);
  }

  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      A[i][j] = MIN3(A[i - 1][j] + 1, A[i][j - 1] + 1,
                     A[i - 1][j - 1] + (S[i - 1] == T[j - 1] ? 0 : 1));

      // Insert
      if (A[i][j] == A[i - 1][j] + 1) {
        top[i][j] = string_node_new(S[i - 1], top[i - 1][j]);
        bottom[i][j] = string_node_new('-', bottom[i - 1][j]);
      } else if (A[i][j] == A[i][j - 1] + 1) {
        // Delete
        top[i][j] = string_node_new('-', top[i][j - 1]);
        bottom[i][j] = string_node_new(T[j - 1], bottom[i][j - 1]);
      } else {
        // Change
        top[i][j] = string_node_new(S[i - 1], top[i - 1][j - 1]);
        bottom[i][j] = string_node_new(T[j - 1], bottom[i - 1][j - 1]);
      }
    }
  }

  string_node_printf(top[m][n], m, n);
  string_node_printf(bottom[m][n], m, n);
  return;
}

int main() {
  int m;
  int n;
  char *S;
  char *T;

  scanf("%d", &m);

  if (!m) {
    char *S = "AGATACATCA";
    char *T = "GATTAGATACAT";
    align(S, T, 10, 12);
  } else {
    S = malloc(m + 1);
    scanf("%s", S);
    scanf("%d", &n);
    T = malloc(n + 1);
    scanf("%s", T);
  }

  align(S, T, m, n);
  return (0);
}
