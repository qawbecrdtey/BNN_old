#ifndef BNN_TEMPORARY_Scalar_hpp
#define BNN_TEMPORARY_Scalar_hpp

#include <cstddef>
#include "OperandBase.hpp"

template<typename T>
class Scalar : public virtual OperandBase {
    T const &s;

public:
    explicit Scalar(T const &t) : s(t) {}

    T const &operator[](std::size_t) const {
        return s;
    }

    std::size_t const size() const {
        return 0;
    }
};


#endif
