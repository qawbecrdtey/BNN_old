#ifndef BNN_TEMPORARY_OperatorBase_hpp
#define BNN_TEMPORARY_OperatorBase_hpp

#include <cstddef>
#include <functional>
#include "../SubClass/get_given_amount_of_parameters_for_constructor_class.hpp"

/*
 * OperatorBase class gets N elements with type T with a constructor.
 */

template<typename T, std::size_t N>
class OperatorBase : SubClass::BadNaming2<T, N> {

};


#endif
