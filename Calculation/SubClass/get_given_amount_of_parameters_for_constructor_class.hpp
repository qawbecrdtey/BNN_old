#ifndef BNN_get_given_amount_of_parameters_class_hpp
#define BNN_get_given_amount_of_parameters_class_hpp

#include <cstddef>
#include <utility>

/*
 * This template class is an abstract class that gets typical amount of inputs for constructor.
 * There might be some mistakes, so be careful with this.
 */


namespace SubClass {

    template<typename T, auto> using always_t = T;
    template<typename T, typename arity>
    struct get_given_amount_of_parameters_for_constructor_class;

    template<typename T, std::size_t ... Is>
    struct get_given_amount_of_parameters_for_constructor_class<T, std::index_sequence<Is...> > {
    protected:
        virtual void init(always_t<T, Is>...) = 0;

    public:
        explicit constexpr get_given_amount_of_parameters_for_constructor_class(always_t<T, Is>... type) {
            init(type...);
        }
    };

    template<std::size_t N>
    using BadNaming1 = get_given_amount_of_parameters_for_constructor_class<std::size_t, std::make_index_sequence<N> >;
    template<typename T, std::size_t N>
    using BadNaming2 = get_given_amount_of_parameters_for_constructor_class<T, std::make_index_sequence<N> >;
}

#endif
