#ifndef BNN_OptimizerBase_hpp
#define BNN_OptimizerBase_hpp

#include <cstddef>
#include "../LinearAlgebra/Matrix.hpp"

namespace Optimizer {
    using nnint = std::size_t;
    template<typename T>
    class OptimizerBase {
        using Matrix = LinearAlgebra::Matrix<T>;
    protected:
        virtual void init(nnint layers_count, nnint const *layer_size, Matrix const *layers,
                Matrix const *matrix, Matrix const *bias, Matrix const *z, std::random_device &rd) = 0;
    public:
        OptimizerBase(nnint layers_count, nnint const *layer_size, Matrix const *layers,
                      Matrix const *matrix, Matrix const *bias, Matrix const *z, std::random_device &rd) {}
        void forward(Matrix const &input) = 0;
        void backward(Matrix const &trueValue) = 0;
    };
}

#endif
