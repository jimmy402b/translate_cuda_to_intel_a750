#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const int N = 1024;
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);

    {
        sycl::buffer<float> buf_a(a.data(), N);
        sycl::buffer<float> buf_b(b.data(), N);
        sycl::buffer<float> buf_c(c.data(), N);

        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                acc_c[i] = acc_a[i] + acc_b[i];
            });
        });
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(c[i] - 3.0f) > 1e-5f) errors++;
    }
    std::cout << "Errors: " << errors << std::endl;
    return errors;
}
