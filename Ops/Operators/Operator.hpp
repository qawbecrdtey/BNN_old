#ifndef BNN_OperatorBase_hpp
#define BNN_OperatorBase_hpp

#include <functional>

template<typename FuncOut, std::size_t size, typename ...Ts>
class OperatorBase {
    using Function = std::function<FuncOut(Ts...)>;

    Function const &Operator;
public:
    // constructor
    explicit constexpr OperatorBase(Function F)
            : Operator(F) {
        static_assert(size == sizeof...(Ts), "In OperatorBase: size of parameter pack does not match the size given!");
    }
    
    // operator()
    template<typename ...Is>
    constexpr FuncOut operator()(Is... is) {
        return Operator(is...);
    }
};


#endif
