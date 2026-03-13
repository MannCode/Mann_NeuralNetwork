#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));

    std::vector<int> nums;

    for (int i = 0; i < 5; i++) {
        nums.push_back(std::rand() % 100);
    }

    std::cout << "Random numbers:\n";

    for (int n : nums) {
        std::cout << n << " ";
    }

    std::cout << "\n";

    int sum = 0;
    for (int n : nums) {
        sum += n;
    }

    std::cout << "Sum = " << sum << std::endl;

    return 0;
}
