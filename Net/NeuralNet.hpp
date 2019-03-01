#ifndef BNN_NeuralNet_hpp
#define BNN_NeuralNet_hpp

#include <functional>
#include "../LinearAlgebra/Matrix.hpp"

namespace Net {

    template<typename T>
    class NeuralNet {
        using Matrix = LinearAlgebra::Matrix<T>;
        using FunctionType = std::function<Matrix(Matrix)>;
        using nnint = std::size_t;

        // (input, hidden and output) layers, weights, layers before passing activation function
        Matrix *layers, *weights, *z;

        // Function applied to layers except output layer
        FunctionType inner_function;
        // Function applied to output layer
        FunctionType outer_function;
        // derivatives of functions above
        FunctionType dinner_function;
        FunctionType douter_function;

        // number of layers, size of each layer
        // number of weights and z is layers_count - 1
        nnint layers_count, *layer_size;

        // learning rate
        double alpha;

    protected:
        virtual void init(nnint const *layer_size) {
            std::copy(layer_size, layer_size + layers_count, this->layer_size);
            for (nnint i = 0; i < layers_count; i++) {
                layers[i] = Matrix(layer_size[i], 1);
            }
            for (nnint i = 0; i < layers_count - 1; i++) {
                weights[i] = Matrix(layer_size[i], layer_size[i + 1]);
                z[i] = Matrix(layer_size[i + 1], 1);
            }
        }

        // forward propagation
        virtual void forward(Matrix input) {
            assert(input.get_row() == layer_size[0] && input.get_col() == 1);
            layers[0] = input;
            for (nnint i = 0; i < layers_count - 2; i++) {
                layers[i + 1] = inner_function(z[i] = weights[i].transpose() * layers[i]);
            }
            layers[layers_count - 1] = outer_function(
                    z[layers_count - 2] = weights[layers_count - 2].transpose() * layers[layers_count - 2]);
        }

        // get the result of forward propagation
        virtual Matrix const &result() const {
            return layers[layers_count - 1];
        }

        // error function
        virtual T const error(Matrix trueValue) const {
            T sum = 0;
            for (nnint i = 0; i < layer_size[layers_count - 1]; i++) {
                sum += (layers[layers_count](i, 0) - trueValue(i, 0)) * (layers[layers_count](i, 0) - trueValue(i, 0));
            }
            return sum / 2;
        }

        //derivative of error function
        virtual Matrix const derror(Matrix trueValue) const {
            return layers[layers_count] - trueValue;
        }

        // back propagation
        virtual void backward(Matrix trueValue) {
            Matrix delta = elementwise_multiplied(derror(trueValue), douter_function(z[layers_count - 1]));
            for (nnint i = layers_count - 2; i > 0; i--) {
                weights[i] -= alpha * layers[i] * delta.transposed();
                delta = elementwise_multiplied(weights[i] * delta, dinner_function(z[i - 1]));
            }
            weights[0] -= alpha * layers[0] * delta.transposed();
        }

    public:
        // constructor
        NeuralNet(nnint layers_count, nnint *layer_size, FunctionType inner_function,
                  FunctionType outer_function, FunctionType dinner_function,
                  FunctionType douter_function, T alpha = 0.01)
                : layers_count(layers_count), layer_size(new nnint[layers_count]),
                  inner_function(inner_function), outer_function(outer_function),
                  dinner_function(dinner_function), douter_function(douter_function),
                  layers(new Matrix[layers_count]), weights(new Matrix[layers_count - 1]),
                  z(new Matrix[layers_count - 1]),
                  alpha(alpha) {
            init(layer_size);
        }

        // copy/move constructors/assignments deleted
        NeuralNet(NeuralNet const &) = delete;

        NeuralNet(NeuralNet &&) = delete;

        NeuralNet &operator=(NeuralNet const &) = delete;

        NeuralNet &operator=(NeuralNet &&) = delete;

        void learn(nnint test_case, Matrix const *const input, Matrix const *const answer) {
            for (nnint i = 0; i < test_case; i++) {
                forward(input[i]);
                backward(answer[i]);
            }
        }

        void print_case(std::ostream &os, Matrix input, Matrix output) {
            os << "Input is :" << '\n' << input.transposed() << '\n';
            forward(input);
            os << "Output is :" << '\n' << layers[layers_count].transposed() << '\n';
            os << "Expected output is :" << '\n' << output.transposed() << '\n';
        }
    };

}

#endif
