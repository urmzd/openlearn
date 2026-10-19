# A -> Africa
# B -> Asia
# C -> Oceania
# D -> Europe
# E -> North America
# F -> South America

# Seperators
# E3, E5, A5, B6, B7, B9, D5, D6

# Define vertices


class Solution:
    a1 = a2 = a3 = a4 = a5 = a6 = 0
    b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = b11 = b12 = 0
    c1 = c2 = c3 = c4 = 0
    d1 = d2 = d3 = d4 = d5 = d6 = d7 = 0
    e1 = e2 = e3 = e4 = e5 = e6 = e7 = e8 = e9 = 0
    f1 = f2 = f3 = f4 = 0

    colors = [0, 1, 2, 3]

    def count_all(self):
        count = 0

        counts = []

        for self.e3 in range(4):
            for self.e5 in range(4):
                for self.a5 in range(4):
                    for self.b6 in range(4):
                        for self.b7 in range(4):
                            for self.b9 in range(4):
                                for self.d5 in range(4):
                                    for self.d6 in range(4):
                                        if self.d5 not in [
                                            self.a5,
                                            self.d6,
                                            self.b7,
                                        ] and self.d6 not in [self.d5, self.b7]:
                                            asia_count = self.count_asia()
                                            north_america_count = (
                                                self.count_north_america()
                                            )
                                            south_america_count = (
                                                self.count_south_america()
                                            )
                                            europe_count = self.count_europe()
                                            oceania_count = self.count_oceania()
                                            africa_count = self.count_africa()

                                            _count = (
                                                asia_count
                                                * north_america_count
                                                * south_america_count
                                                * europe_count
                                                * oceania_count
                                                * africa_count
                                            )

                                            count += _count
        return count

    def count_north_america(self):
        count = 0
        for self.e1 in range(4):
            if self.e1 != self.b6:
                for self.e6 in range(4):
                    if self.e6 not in [self.e1, self.e5]:
                        for self.e2 in range(4):
                            if self.e2 not in [self.e6, self.e1]:
                                for self.e7 in range(4):
                                    if self.e7 not in [self.e2, self.e6, self.e5]:
                                        for self.e8 in range(4):
                                            if self.e8 not in [self.e7, self.e5]:
                                                for self.e4 in range(4):
                                                    if self.e4 not in [
                                                        self.e7,
                                                        self.e8,
                                                        self.e3,
                                                    ]:
                                                        for e9 in range(4):
                                                            if e9 not in [
                                                                self.e4,
                                                                self.e2,
                                                                self.e7,
                                                                self.e3,
                                                            ]:
                                                                count += 1

        return count

    def count_south_america(self):
        count = 0
        for self.f4 in range(4):
            if self.f4 != self.e3:
                for self.f2 in range(4):
                    if self.f2 not in [self.a5, self.f4]:
                        for self.f3 in range(4):
                            if self.f3 not in [self.f4, self.f2]:
                                for self.f1 in range(4):
                                    if self.f1 not in [self.f3, self.f2]:
                                        count += 1
        return count

    def count_africa(self):
        count = 0
        for self.a3 in range(4):
            if self.a3 not in [self.a5, self.d5, self.b7]:
                for self.a2 in range(4):
                    if self.a2 not in [self.a3, self.b7, self.a5]:
                        for self.a4 in range(4):
                            if self.a4 != self.a2:
                                for self.a6 in range(4):
                                    if self.a6 not in [self.a4, self.a2]:
                                        for self.a1 in range(4):
                                            if self.a1 not in [
                                                self.a5,
                                                self.a2,
                                                self.a6,
                                            ]:
                                                count += 1

        return count

    def count_europe(self):
        count = 0
        for self.d2 in range(4):
            if self.d2 != self.e5:
                for self.d1 in range(4):
                    if self.d1 != self.d2:
                        for self.d7 in range(4):
                            if self.d7 not in [self.a5, self.d1, self.d5]:
                                for self.d3 in range(4):
                                    if self.d3 not in [
                                        self.d1,
                                        self.d5,
                                        self.d6,
                                        self.d7,
                                    ]:
                                        for self.d4 in range(4):
                                            if self.d4 not in [
                                                self.d1,
                                                self.d2,
                                                self.d3,
                                                self.d6,
                                            ]:
                                                count += 1

        return count

    def count_oceania(self):
        count = 0

        for self.c2 in range(4):
            if self.c2 != self.b9:
                for self.c3 in range(4):
                    if self.c3 != self.c2:
                        for self.c4 in range(4):
                            if self.c4 not in [self.c2, self.c3]:
                                for self.c1 in range(4):
                                    if self.c1 not in [self.c3, self.c4]:
                                        count += 1

        return count

    def count_asia(self):
        count = 0

        for self.b1 in range(4):
            if self.b1 not in [self.d6, self.b7]:
                for self.b3 in range(4):
                    if self.b3 not in [self.b1, self.b7, self.b9]:
                        for self.b2 in range(4):
                            if self.b2 not in [self.b9, self.b3, self.b1]:
                                for self.b8 in range(4):
                                    if self.b8 not in [self.b2, self.b6]:
                                        for self.b5 in range(4):
                                            if self.b5 not in [self.b2, self.b8]:
                                                for self.b12 in range(4):
                                                    if self.b12 != self.b6:
                                                        for self.b4 in range(4):
                                                            if self.b4 not in [
                                                                self.b6,
                                                                self.b8,
                                                                self.b12,
                                                            ]:
                                                                for self.b10 in range(
                                                                    4
                                                                ):
                                                                    if self.b10 not in [
                                                                        self.b2,
                                                                        self.b4,
                                                                        self.b8,
                                                                        self.b12,
                                                                    ]:
                                                                        for (
                                                                            self.b11
                                                                        ) in range(4):
                                                                            if (
                                                                                self.b11
                                                                                not in [
                                                                                    self.b1,
                                                                                    self.b2,
                                                                                    self.d6,
                                                                                    self.b10,
                                                                                ]
                                                                            ):
                                                                                count += 1

        return count


sol = Solution()
print(sol.count_all())
