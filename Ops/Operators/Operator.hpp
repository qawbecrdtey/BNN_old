#ifndef BNN_OperatorBase_hpp
#define BNN_OperatorBase_hpp

#include <functional>

namespace Operators {

    template<typename FuncOut, std::size_t size, typename ...Ts>
    class OperatorBase {
        using Function = std::function<FuncOut(Ts...)>;

        Function const &Operator;

    public:
        // constructor
        explicit constexpr OperatorBase(Function F)
                : Operator(F) {
            static_assert(size == sizeof...(Ts),
                          "In OperatorBase: size of parameter pack does not match the size given!");
        }

        // operator()
        template<typename ...Is>
        constexpr FuncOut operator()(Is... is) {
            return Operator(is...);
        }
    };

    template<typename FuncOut>
    using BinomialOperator = OperatorBase<FuncOut, 2>;

    template<typename FuncOut>
    using UnaryOperator = OperatorBase<FuncOut, 1>;

}

#endif