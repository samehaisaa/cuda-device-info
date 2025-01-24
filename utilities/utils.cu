#include <cstdlib>

void initializeVector(float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
}