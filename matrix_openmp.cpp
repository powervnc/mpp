#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <omp.h>

double getRandomDouble() {
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

bool looksLikeBinary(const std::string& fname) {
    return fname.size() > 4 && fname.substr(fname.size() - 4) == ".bin";
}

class Matrix {
    const size_t size_;
    std::vector<double> data_;

    explicit Matrix(size_t n) : size_(n), data_(n * n, 0.0) {}

    static void createBinaryMatrixFile(size_t dim, const std::string& filename) {
        std::ofstream fout(filename, std::ios::binary);
        size_t n = dim * dim;
        for (size_t i = 0; i < n; ++i) {
            double val = getRandomDouble();
            fout.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }

public:
    
    static Matrix readFromFile(size_t dim, const std::string& filename) {
        Matrix mat(dim);
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            createBinaryMatrixFile(dim, filename);
            fin.open(filename, std::ios::binary);
        }
        fin.read(reinterpret_cast<char*>(mat.data_.data()), sizeof(double) * dim * dim);
        return mat;
    }

    static Matrix readBinaryParallel(size_t dim, const std::string& filename, size_t num_threads) {
        Matrix mat(dim);
        std::ifstream test(filename, std::ios::binary);
        if (!test) createBinaryMatrixFile(dim, filename);

        size_t total = dim * dim;
        size_t chunk = total / num_threads;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            size_t start = tid * chunk;
            size_t end = (tid == (num_threads - 1)) ? total : start + chunk;

            std::ifstream fin(filename, std::ios::binary);
            fin.seekg(start * sizeof(double));
            fin.read(reinterpret_cast<char*>(&mat.data_[start]), (end - start) * sizeof(double));
        }

        return mat;
    }

    Matrix getTransposed() const {
        Matrix t(size_);
        #pragma omp parallel for
        for (size_t r = 0; r < size_; ++r)
            for (size_t c = 0; c < size_; ++c)
                t.data_[c * size_ + r] = data_[r * size_ + c];
        return t;
    }

    static Matrix multiplyParallel(const Matrix& X, const Matrix& Y, size_t num_threads) {
        if (X.size_ != Y.size_) throw std::runtime_error("Dimension mismatch");

        Matrix Yt = Y.getTransposed();
        Matrix result(X.size_);

        #pragma omp parallel for num_threads(num_threads)
        for (size_t r = 0; r < X.size_; ++r) {
            for (size_t c = 0; c < X.size_; ++c) {
                auto rowX = X.data_.cbegin() + r * X.size_;
                auto rowYt = Yt.data_.cbegin() + c * X.size_;
                result.data_[r * X.size_ + c] = std::inner_product(rowX, rowX + X.size_, rowYt, 0.0);
            }
        }

        return result;
    }

    void writeToFile(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(data_.data()), size_ * size_ * sizeof(double));
    }
};

void printTimingResults(
    size_t dimension,
    std::chrono::duration<double> read_time,
    std::chrono::duration<double> compute_time,
    std::chrono::duration<double> write_time,
    std::chrono::duration<double> total_time)
{
    std::cout << std::fixed
              << std::setw(10) << dimension << "\t"
              << read_time.count() << "\t"
              << compute_time.count() << "\t"
              << write_time.count() << "\t"
              << total_time.count()
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: dim A.bin B.bin C.bin mode(a/b) threads\n";
        return 1;
    }

    const size_t dim = std::stoul(argv[1]);
    std::string mode = argv[5]; 
    size_t num_threads = std::stoul(argv[6]);
    omp_set_num_threads(num_threads);

    using Clock = std::chrono::high_resolution_clock;
    auto total_start = Clock::now();

    auto read_start = Clock::now();
    Matrix A = (mode == "a") ? Matrix::readFromFile(dim, argv[2]) : Matrix::readBinaryParallel(dim, argv[2], num_threads);
    Matrix B = (mode == "a") ? Matrix::readFromFile(dim, argv[3]) : Matrix::readBinaryParallel(dim, argv[3], num_threads);
    auto read_end = Clock::now();

    auto comp_start = Clock::now();
    Matrix C = Matrix::multiplyParallel(A, B, num_threads);
    auto comp_end = Clock::now();

    auto write_start = Clock::now();
    C.writeToFile(argv[4]);
    auto write_end = Clock::now();

    auto total_end = Clock::now();

    printTimingResults(
        dim,
        read_end - read_start,
        comp_end - comp_start,
        write_end - write_start,
        total_end - total_start
    );

    return 0;
}